use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use maki_agent::agent;
use maki_agent::mcp;
use maki_agent::mcp::McpHandle;
use maki_agent::mcp::config::{McpServerInfo, McpServerStatus};
use maki_agent::permissions::PermissionManager;
use maki_agent::skill::Skill;
use maki_agent::template;
use maki_agent::template::Vars;
use maki_agent::tools::{DescriptionContext, ToolCall, ToolFilter};
use maki_agent::{
    Agent, AgentConfig, AgentEvent, AgentInput, AgentParams, AgentRunParams, CancelToken,
    CancelTrigger, Envelope, EventSender, History, Instructions, McpCommand, McpSnapshot,
    PromptRole,
};
use maki_providers::{AgentError, Message, Model, TokenUsage};
use serde_json::Value;
use tracing::error;

use super::ModelSlot;
use super::cancel_map::CancelMap;
use super::shared_queue::{QueueItem, SharedQueue};

pub(super) struct AgentLoop {
    model_slot: Arc<ArcSwap<ModelSlot>>,
    skills: Arc<[Skill]>,
    config: AgentConfig,
    vars: Vars,
    instructions: Instructions,
    tools: Value,
    initial_disabled: Vec<String>,
    mcp_handle: Option<McpHandle>,
    mcp_snapshot: Arc<ArcSwap<McpSnapshot>>,
    history: History,
    shared_history: Arc<ArcSwap<Vec<Message>>>,
    cancel_map: Arc<std::sync::Mutex<CancelMap>>,
    cancel: CancelToken,
    permissions: Arc<PermissionManager>,
    min_run_id: u64,
    agent_tx: flume::Sender<Envelope>,
    answer_rx: Arc<async_lock::Mutex<flume::Receiver<String>>>,
    notify_rx: flume::Receiver<()>,
    queue: Arc<SharedQueue>,
    toggle_rx: flume::Receiver<(String, bool)>,
}

enum LoopEvent {
    Queue,
    Toggle(String, bool),
}

impl AgentLoop {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        model_slot: Arc<ArcSwap<ModelSlot>>,
        skills: Arc<[Skill]>,
        config: AgentConfig,
        initial_history: Vec<Message>,
        shared_history: Arc<ArcSwap<Vec<Message>>>,
        mcp_snapshot: Arc<ArcSwap<McpSnapshot>>,
        initial_disabled: Vec<String>,
        permissions: Arc<PermissionManager>,
        agent_tx: flume::Sender<Envelope>,
        answer_rx: flume::Receiver<String>,
        notify_rx: flume::Receiver<()>,
        queue: Arc<SharedQueue>,
        toggle_rx: flume::Receiver<(String, bool)>,
        cancel_map: Arc<std::sync::Mutex<CancelMap>>,
        cancel: CancelToken,
    ) -> Self {
        Self {
            model_slot,
            skills,
            config,
            vars: Vars::default(),
            instructions: Instructions::default(),
            tools: Value::Null,
            initial_disabled,
            mcp_handle: None,
            mcp_snapshot,
            history: History::new(initial_history),
            shared_history,
            cancel_map,
            cancel,
            permissions,
            min_run_id: 0,
            agent_tx,
            answer_rx: Arc::new(async_lock::Mutex::new(answer_rx)),
            notify_rx,
            queue,
            toggle_rx,
        }
    }

    pub(super) async fn run(mut self) {
        if !self.initialize().await {
            return;
        }

        loop {
            let event = {
                let notify_rx = &self.notify_rx;
                let toggle_rx = &self.toggle_rx;
                futures_lite::future::or(
                    async { notify_rx.recv_async().await.ok().map(|_| LoopEvent::Queue) },
                    async {
                        toggle_rx
                            .recv_async()
                            .await
                            .ok()
                            .map(|(s, e)| LoopEvent::Toggle(s, e))
                    },
                )
                .await
            };

            let Some(event) = event else { break };

            match event {
                LoopEvent::Toggle(name, enabled) => self.handle_mcp_toggle(name, enabled),
                LoopEvent::Queue => {
                    while let Some(entry) = self.queue.pop() {
                        if entry.run_id() < self.min_run_id {
                            continue;
                        }
                        self.process_entry(entry).await;
                    }
                }
            }
        }
    }

    async fn process_entry(&mut self, entry: QueueItem) {
        let run_id = entry.run_id();
        let event_tx = EventSender::new(self.agent_tx.clone(), run_id);

        let result = match entry {
            QueueItem::Message {
                text,
                image_count,
                input,
                ..
            } => {
                let _ = event_tx.send(AgentEvent::QueueItemConsumed { text, image_count });
                self.do_agent_run(input, event_tx, run_id).await
            }
            QueueItem::Compact { .. } => self.do_compact(&event_tx).await,
        };

        if let Err(e) = result {
            self.emit_error(run_id, e);
        }
    }

    async fn initialize(&mut self) -> bool {
        self.vars = template::env_vars();
        self.reload_instructions().await;
        if self.cancel.is_cancelled() {
            return false;
        }

        let slot = self.model_slot.load();
        self.tools = self.build_tools(&slot.model);

        let cwd = PathBuf::from(self.vars.apply("{cwd}").into_owned());
        self.init_mcp(&cwd).await;
        !self.cancel.is_cancelled()
    }

    async fn init_mcp(&mut self, cwd: &Path) {
        let mcp_config = mcp::config::load_config(cwd);
        let disabled = {
            let mut d = self.initial_disabled.clone();
            d.sort_unstable();
            d.dedup();
            d
        };

        if !mcp_config.is_empty() {
            let preliminary = mcp_config.preliminary_infos(&disabled);
            self.mcp_snapshot.store(Arc::new(McpSnapshot {
                infos: preliminary,
                prompts: vec![],
                pids: vec![],
                generation: 0,
            }));
        }

        let handle = self
            .cancel
            .race(mcp::start_with_config(
                mcp_config,
                disabled,
                self.mcp_snapshot.clone(),
            ))
            .await
            .unwrap_or(None);

        if let Some(ref h) = handle {
            h.extend_tools(&mut self.tools);
            let snap = h.snapshot.load();
            spawn_oauth_for_needs_auth(&snap.infos, h);
        }

        self.mcp_handle = handle;
    }

    async fn do_compact(&mut self, event_tx: &EventSender) -> Result<(), AgentError> {
        let slot = self.model_slot.load();
        let result =
            agent::compact(&*slot.provider, &slot.model, &mut self.history, event_tx).await;
        self.sync_shared_history();
        result
    }

    async fn do_agent_run(
        &mut self,
        mut input: AgentInput,
        event_tx: EventSender,
        run_id: u64,
    ) -> Result<(), AgentError> {
        let slot = self.model_slot.load();

        let old_cwd = self.vars.apply("{cwd}").into_owned();
        self.vars = template::env_vars();
        if *self.vars.apply("{cwd}") != old_cwd {
            self.reload_instructions().await;
        }
        self.rebuild_tools(&slot.model);

        for msg in mem::take(&mut input.preamble) {
            self.history.push(msg);
        }

        if let Some(ref prompt_ref) = input.prompt {
            let Some(ref mcp) = self.mcp_handle else {
                return Err(AgentError::Tool {
                    tool: "mcp_prompt".into(),
                    message: "MCP not available".into(),
                });
            };
            let messages = mcp
                .get_prompt(&prompt_ref.qualified_name, &prompt_ref.arguments)
                .await
                .map_err(|e| AgentError::Tool {
                    tool: "mcp_prompt".into(),
                    message: e.to_string(),
                })?;
            for pm in messages {
                let text = pm.content.text.unwrap_or_default();
                let msg = match pm.role {
                    PromptRole::Assistant => Message {
                        role: maki_providers::Role::Assistant,
                        content: vec![maki_providers::ContentBlock::Text { text }],
                        ..Default::default()
                    },
                    PromptRole::User => Message::user(text),
                };
                self.history.push(msg);
            }
        }

        self.sync_shared_history_with_pending(&input);

        let system = agent::build_system_prompt(&self.vars, &input.mode, &self.instructions.text);
        let (trigger, cancel) = CancelToken::new();
        self.set_cancel_trigger(run_id, trigger);

        while self.answer_rx.lock().await.try_recv().is_ok() {}

        let agent = Agent::new(
            AgentParams {
                provider: Arc::clone(&slot.provider),
                model: slot.model.clone(),
                skills: Arc::clone(&self.skills),
                config: self.config.clone(),
                permissions: Arc::clone(&self.permissions),
            },
            AgentRunParams {
                history: mem::replace(&mut self.history, History::new(Vec::new())),
                system,
                event_tx,
                tools: self.tools.clone(),
            },
        )
        .with_loaded_instructions(self.instructions.loaded.clone())
        .with_user_response_rx(Arc::clone(&self.answer_rx))
        .with_interrupt_source(Arc::clone(&self.queue) as Arc<dyn maki_agent::InterruptSource>)
        .with_cancel(cancel)
        .with_mcp(self.mcp_handle.clone());

        let outcome = agent.run(input).await;

        self.clear_cancel_trigger(run_id);
        self.history = outcome.history;
        self.sync_shared_history();

        if matches!(outcome.result, Err(AgentError::Cancelled)) {
            self.min_run_id = run_id + 1;
        }

        outcome.result
    }

    fn handle_mcp_toggle(&mut self, server_name: String, enabled: bool) {
        if let Some(ref mcp) = self.mcp_handle {
            mcp.send(McpCommand::Toggle {
                server: server_name,
                enabled,
            });
        }
    }

    fn rebuild_tools(&mut self, model: &Model) {
        let mut tools = self.build_tools(model);
        if let Some(ref mcp) = self.mcp_handle {
            mcp.extend_tools(&mut tools);
        }
        self.tools = tools;
    }

    fn build_tools(&self, model: &Model) -> Value {
        let examples = model.supports_tool_examples();
        let filter = ToolFilter::from_config(&self.config, &[]);
        let ctx = DescriptionContext {
            skills: &self.skills,
            filter: &filter,
        };
        ToolCall::definitions(&self.vars, &ctx, examples)
    }

    async fn reload_instructions(&mut self) {
        let cwd = self.vars.apply("{cwd}").into_owned();
        self.instructions = smol::unblock(move || agent::load_instructions(&cwd)).await;
    }

    fn sync_shared_history(&self) {
        self.shared_history
            .store(Arc::new(self.history.as_slice().to_vec()));
    }

    fn sync_shared_history_with_pending(&self, input: &AgentInput) {
        let mut snapshot = self.history.as_slice().to_vec();
        snapshot.push(Message::user(input.message.clone()));
        self.shared_history.store(Arc::new(snapshot));
    }

    fn set_cancel_trigger(&self, run_id: u64, trigger: CancelTrigger) {
        self.cancel_map
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(run_id, trigger);
    }

    fn clear_cancel_trigger(&self, run_id: u64) {
        self.cancel_map
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(run_id);
    }

    fn emit_error(&self, run_id: u64, error: AgentError) {
        let event_tx = EventSender::new(self.agent_tx.clone(), run_id);
        match error {
            AgentError::Cancelled => {
                let _ = event_tx.send(AgentEvent::Done {
                    usage: TokenUsage::default(),
                    num_turns: 0,
                    stop_reason: None,
                });
            }
            e => {
                error!(error = %e, "agent error");
                let _ = event_tx.send(AgentEvent::Error {
                    message: e.user_message(),
                });
            }
        }
    }
}

fn spawn_oauth_for_needs_auth(infos: &[McpServerInfo], handle: &McpHandle) {
    for info in infos {
        let McpServerStatus::NeedsAuth { ref url } = info.status else {
            continue;
        };
        let Some(ref server_url) = info.url else {
            continue;
        };
        let handle = handle.clone();
        let server_name = info.name.clone();
        let server_url = server_url.clone();
        let www_auth = url.clone();
        smol::spawn(async move {
            let storage = match maki_storage::DataDir::resolve() {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(server = %server_name, error = %e, "cannot resolve storage for OAuth");
                    return;
                }
            };
            let auth_data = match maki_agent::mcp::oauth::authenticate(
                &server_name,
                &server_url,
                www_auth.as_deref(),
                &storage,
            )
            .await
            {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(server = %server_name, error = %e, "background OAuth failed");
                    return;
                }
            };
            let Some(ref tokens) = auth_data.tokens else {
                return;
            };
            handle.send(McpCommand::Reconnect {
                server: server_name.clone(),
                url: server_url,
                token: tokens.access.clone(),
            });
            tracing::info!(server = %server_name, "MCP server authenticated via OAuth");
        })
        .detach();
    }
}
