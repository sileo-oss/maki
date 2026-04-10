//! MCP client: manages transports and routes tool calls to servers.
//!
//! Tool names are namespaced as `server__tool` (double underscore) to avoid collisions across servers.
//! Names are leaked into `&'static str` so they can be used in tool descriptors without lifetime friction.
//!
//! State is managed through a single command loop (`McpManager::run`). External callers interact
//! via `McpHandle`, which exposes read-only access through `ArcSwap<McpSnapshot>` and mutations
//! through a command channel. Tool calls bypass the command loop for performance.

pub mod config;
pub mod error;
pub mod http;
pub mod oauth;
pub mod protocol;
pub mod stdio;
pub mod transport;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use async_lock::RwLock;
use serde_json::{Value, json};
use tracing::{info, warn};

use self::config::{
    McpConfig, McpServerInfo, McpServerStatus, ServerConfig, Transport, load_config, parse_server,
    transport_kind,
};
use self::error::McpError;
use self::http::HttpTransport;
use self::stdio::StdioTransport;
use self::transport::McpTransport;

const SEPARATOR: &str = "__";

struct McpToolDef {
    qualified_name: &'static str,
    server_name: Arc<str>,
    raw_name: String,
    description: String,
    input_schema: Value,
}

struct McpPromptDef {
    qualified_name: String,
    server_name: Arc<str>,
    raw_name: String,
    description: String,
    arguments: Vec<protocol::PromptArgument>,
}

impl McpPromptDef {
    fn from_info(server_name: &Arc<str>, info: protocol::PromptInfo) -> Self {
        let qualified_name = format!("{server_name}{SEPARATOR}{}", info.name);
        Self {
            qualified_name,
            server_name: Arc::clone(server_name),
            raw_name: info.name,
            description: info.description.unwrap_or_default(),
            arguments: info.arguments,
        }
    }

    fn display_name(&self) -> String {
        format!("{}:{}", self.server_name, self.raw_name)
    }

    fn to_info(&self) -> McpPromptInfo {
        McpPromptInfo {
            display_name: self.display_name(),
            qualified_name: self.qualified_name.clone(),
            description: self.description.clone(),
            arguments: self
                .arguments
                .iter()
                .map(|a| McpPromptArg {
                    name: a.name.clone(),
                    description: a.description.clone().unwrap_or_default(),
                    required: a.required,
                })
                .collect(),
        }
    }
}

struct ServerEntry {
    name: String,
    config: Option<ServerConfig>,
    transport_kind: &'static str,
    origin: PathBuf,
    status: McpServerStatus,
}

struct McpManagerInner {
    transports: HashMap<Arc<str>, Box<dyn McpTransport>>,
    tools: Vec<McpToolDef>,
    tool_index: HashMap<&'static str, usize>,
    prompts: Vec<McpPromptDef>,
    entries: Vec<ServerEntry>,
    disabled: Vec<String>,
    generation: u64,
}

#[derive(Clone)]
pub struct McpPromptInfo {
    pub display_name: String,
    pub qualified_name: String,
    pub description: String,
    pub arguments: Vec<McpPromptArg>,
}

#[derive(Clone)]
pub struct McpPromptArg {
    pub name: String,
    pub description: String,
    pub required: bool,
}

#[derive(Clone)]
pub struct McpSnapshot {
    pub infos: Vec<McpServerInfo>,
    pub prompts: Vec<McpPromptInfo>,
    pub pids: Vec<u32>,
    pub generation: u64,
}

pub enum McpCommand {
    Toggle {
        server: String,
        enabled: bool,
    },
    Reconnect {
        server: String,
        url: String,
        token: String,
    },
    Shutdown,
}

pub struct McpHandle {
    cmd_tx: flume::Sender<McpCommand>,
    manager: Arc<McpManager>,
    pub snapshot: Arc<ArcSwap<McpSnapshot>>,
}

impl McpHandle {
    pub fn send(&self, cmd: McpCommand) {
        if let Err(e) = self.cmd_tx.try_send(cmd) {
            tracing::warn!(error = %e, "MCP command dropped — manager shut down");
        }
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.manager.has_tool(name)
    }

    pub fn interned_name(&self, name: &str) -> &'static str {
        self.manager.interned_name(name)
    }

    pub async fn call_tool(&self, qualified_name: &str, args: &Value) -> Result<String, McpError> {
        self.manager.call_tool(qualified_name, args).await
    }

    pub async fn get_prompt(
        &self,
        qualified_name: &str,
        arguments: &HashMap<String, String>,
    ) -> Result<Vec<protocol::PromptMessage>, McpError> {
        self.manager.get_prompt(qualified_name, arguments).await
    }

    pub fn extend_tools(&self, tools: &mut Value) {
        self.manager.extend_tools(tools)
    }
}

impl Clone for McpHandle {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            manager: Arc::clone(&self.manager),
            snapshot: Arc::clone(&self.snapshot),
        }
    }
}

pub async fn start(
    cwd: &Path,
    disabled: Vec<String>,
    snapshot: Arc<ArcSwap<McpSnapshot>>,
) -> Option<McpHandle> {
    McpManager::start(cwd, disabled, snapshot).await
}

pub async fn start_with_config(
    config: McpConfig,
    disabled: Vec<String>,
    snapshot: Arc<ArcSwap<McpSnapshot>>,
) -> Option<McpHandle> {
    McpManager::start_with_config(config, disabled, snapshot).await
}

fn transport_url(transport: &Transport) -> Option<String> {
    match transport {
        Transport::Http { url, .. } => Some(url.clone()),
        Transport::Stdio { .. } => None,
    }
}

struct McpManager {
    inner: RwLock<McpManagerInner>,
}

impl McpManager {
    pub async fn start(
        cwd: &Path,
        disabled: Vec<String>,
        snapshot: Arc<ArcSwap<McpSnapshot>>,
    ) -> Option<McpHandle> {
        let cwd = cwd.to_owned();
        let config = smol::unblock(move || load_config(&cwd)).await;
        Self::start_with_config(config, disabled, snapshot).await
    }

    pub async fn start_with_config(
        config: McpConfig,
        disabled: Vec<String>,
        snapshot: Arc<ArcSwap<McpSnapshot>>,
    ) -> Option<McpHandle> {
        if config.is_empty() {
            return None;
        }

        let origins = config.origins;
        let mut transports: HashMap<Arc<str>, Box<dyn McpTransport>> = HashMap::new();
        let mut tools = Vec::new();
        let mut tool_index = HashMap::new();
        let mut prompts = Vec::new();
        let mut entries = Vec::new();

        struct Pending {
            config: ServerConfig,
            kind: &'static str,
            origin: PathBuf,
        }

        let mut pending = Vec::new();

        for (name, raw) in config.mcp {
            let kind = transport_kind(&raw.transport);
            let origin = origins.get(&name).cloned().unwrap_or_default();
            let enabled = raw.enabled;

            match parse_server(name.clone(), raw) {
                Ok(sc) => {
                    if enabled {
                        pending.push(Pending {
                            config: sc,
                            kind,
                            origin,
                        });
                    } else {
                        entries.push(ServerEntry {
                            name: sc.name.clone(),
                            config: Some(sc),
                            transport_kind: kind,
                            origin,
                            status: McpServerStatus::Disabled,
                        });
                    }
                }
                Err(e) => {
                    warn!(server = %name, error = %e, "invalid MCP server config");
                    entries.push(ServerEntry {
                        name,
                        config: None,
                        transport_kind: kind,
                        origin,
                        status: McpServerStatus::Failed(e.to_string()),
                    });
                }
            }
        }

        let handles: Vec<_> = pending
            .into_iter()
            .map(|p| {
                smol::spawn(async move {
                    let result = Self::start_server(&p.config).await;
                    (p, result)
                })
            })
            .collect();

        for handle in handles {
            let (p, result) = handle.await;
            match result {
                Ok((t, server_tools, server_prompts)) => {
                    let server_name: Arc<str> = Arc::from(p.config.name.as_str());
                    for tool_info in server_tools {
                        let qualified = format!("{}{SEPARATOR}{}", p.config.name, tool_info.name);
                        let interned = intern(qualified);
                        let idx = tools.len();
                        tools.push(McpToolDef {
                            qualified_name: interned,
                            server_name: Arc::clone(&server_name),
                            raw_name: tool_info.name,
                            description: tool_info.description,
                            input_schema: tool_info.input_schema,
                        });
                        tool_index.insert(interned, idx);
                    }
                    for info in server_prompts {
                        prompts.push(McpPromptDef::from_info(&server_name, info));
                    }
                    transports.insert(Arc::clone(&server_name), t);
                    entries.push(ServerEntry {
                        name: p.config.name.clone(),
                        config: Some(p.config),
                        transport_kind: p.kind,
                        origin: p.origin,
                        status: McpServerStatus::Running,
                    });
                }
                Err(e) => {
                    let status = if let McpError::HttpError {
                        status: 401,
                        ref reason,
                        ..
                    } = e
                    {
                        McpServerStatus::NeedsAuth {
                            url: Some(reason.clone()),
                        }
                    } else {
                        warn!(server = %p.config.name, error = %e, "failed to start MCP server");
                        McpServerStatus::Failed(e.to_string())
                    };
                    entries.push(ServerEntry {
                        name: p.config.name.clone(),
                        config: Some(p.config),
                        transport_kind: p.kind,
                        origin: p.origin,
                        status,
                    });
                }
            }
        }

        info!(
            running = transports.len(),
            tools = tools.len(),
            prompts = prompts.len(),
            total = entries.len(),
            "MCP servers initialized"
        );

        let manager = Arc::new(Self {
            inner: RwLock::new(McpManagerInner {
                transports,
                tools,
                tool_index,
                prompts,
                entries,
                disabled,
                generation: 0,
            }),
        });

        snapshot.store(Arc::new(manager.build_snapshot()));

        let (cmd_tx, cmd_rx) = flume::bounded(8);
        let handle = McpHandle {
            cmd_tx,
            manager: Arc::clone(&manager),
            snapshot: Arc::clone(&snapshot),
        };

        smol::spawn(Self::run(manager, cmd_rx, snapshot)).detach();

        Some(handle)
    }

    async fn run(
        self: Arc<Self>,
        cmd_rx: flume::Receiver<McpCommand>,
        snapshot: Arc<ArcSwap<McpSnapshot>>,
    ) {
        while let Ok(cmd) = cmd_rx.recv_async().await {
            match cmd {
                McpCommand::Toggle { server, enabled } => {
                    self.handle_toggle(&server, enabled, &snapshot).await;
                }
                McpCommand::Reconnect { server, url, token } => {
                    self.handle_reconnect(&server, &url, &token, &snapshot)
                        .await;
                }
                McpCommand::Shutdown => {
                    self.do_shutdown().await;
                    break;
                }
            }
        }
    }

    async fn handle_toggle(
        &self,
        server_name: &str,
        enabled: bool,
        snapshot: &ArcSwap<McpSnapshot>,
    ) {
        let config_path = {
            let mut inner = self.inner.write().await;
            toggle_disabled(&mut inner.disabled, server_name, enabled);
            if let Some(entry) = inner.entries.iter_mut().find(|e| e.name == server_name)
                && !enabled
            {
                entry.status = McpServerStatus::Disabled;
            }
            inner.generation += 1;
            inner
                .entries
                .iter()
                .find(|e| e.name == server_name)
                .map(|e| e.origin.clone())
        };

        if let Some(path) = config_path {
            let name = server_name.to_owned();
            let server_for_log = server_name.to_owned();
            smol::spawn(async move {
                if let Err(e) =
                    smol::unblock(move || config::persist_enabled(&path, &name, enabled)).await
                {
                    tracing::warn!(error = %e, server = %server_for_log, "failed to persist MCP toggle");
                }
            })
            .detach();
        }

        if enabled && let Err(e) = self.refresh_server(server_name).await {
            tracing::warn!(
                server = %server_name,
                error = %e,
                "MCP server refresh failed"
            );
        }

        snapshot.store(Arc::new(self.build_snapshot()));
        info!(server = server_name, enabled, "MCP toggle complete");
    }

    async fn handle_reconnect(
        &self,
        server_name: &str,
        server_url: &str,
        access_token: &str,
        snapshot: &ArcSwap<McpSnapshot>,
    ) {
        let timeout = {
            let inner = self.inner.read().await;
            if inner.disabled.iter().any(|d| d == server_name) {
                info!(
                    server = server_name,
                    "ignoring reconnect for disabled server"
                );
                return;
            }
            inner
                .entries
                .iter()
                .find(|e| e.name == server_name)
                .and_then(|e| e.config.as_ref().map(|c| c.timeout))
                .unwrap_or_default()
        };
        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {access_token}"),
        );
        let config = ServerConfig {
            name: server_name.to_string(),
            timeout,
            transport: Transport::Http {
                url: server_url.to_string(),
                headers,
            },
        };
        if let Err(e) = self.refresh_server_with(server_name, Some(config)).await {
            tracing::warn!(server = %server_name, error = %e, "reconnect failed");
        }
        let snap = {
            let mut inner = self.inner.write().await;
            inner.generation += 1;
            Self::snapshot_from(&inner)
        };
        snapshot.store(Arc::new(snap));
        info!(server = server_name, "MCP reconnect complete");
    }

    async fn do_shutdown(&self) {
        let transports = {
            let mut inner = self.inner.write().await;
            std::mem::take(&mut inner.transports)
        };
        let handles: Vec<_> = transports
            .into_iter()
            .map(|(name, t)| {
                smol::spawn(async move {
                    info!(server = &*name, "shutting down MCP server");
                    t.shutdown().await;
                })
            })
            .collect();
        for h in handles {
            h.await;
        }
    }

    async fn start_server(
        config: &ServerConfig,
    ) -> Result<
        (
            Box<dyn McpTransport>,
            Vec<protocol::ToolInfo>,
            Vec<protocol::PromptInfo>,
        ),
        McpError,
    > {
        let t: Box<dyn McpTransport> = match &config.transport {
            Transport::Stdio {
                program,
                args,
                environment,
            } => Box::new(StdioTransport::spawn(
                &config.name,
                program,
                args,
                environment,
                config.timeout,
            )?),
            Transport::Http { url, headers } => Box::new(HttpTransport::new(
                &config.name,
                url,
                headers,
                config.timeout,
            )?),
        };
        transport::initialize(t.as_ref()).await?;
        let tools = transport::list_tools(t.as_ref()).await?;
        let prompts = transport::list_prompts(t.as_ref()).await?;
        info!(
            server = config.name,
            tool_count = tools.len(),
            prompt_count = prompts.len(),
            "MCP server initialized"
        );
        Ok((t, tools, prompts))
    }

    fn has_tool(&self, name: &str) -> bool {
        self.inner.read_blocking().tool_index.contains_key(name)
    }

    fn interned_name(&self, name: &str) -> &'static str {
        self.inner
            .read_blocking()
            .tool_index
            .get_key_value(name)
            .map(|(&k, _)| k)
            .unwrap_or("unknown_mcp")
    }

    async fn call_tool(&self, qualified_name: &str, args: &Value) -> Result<String, McpError> {
        let inner = self.inner.read().await;
        let idx = inner
            .tool_index
            .get(qualified_name)
            .ok_or_else(|| McpError::UnknownTool {
                name: qualified_name.into(),
            })?;
        let def = &inner.tools[*idx];
        let t = inner
            .transports
            .get(&def.server_name)
            .ok_or_else(|| McpError::ServerDied {
                server: (*def.server_name).into(),
            })?;
        transport::call_tool(t.as_ref(), &def.raw_name, args).await
    }

    fn extend_tools(&self, tools: &mut Value) {
        let inner = self.inner.read_blocking();
        let disabled = &inner.disabled;
        for t in inner
            .tools
            .iter()
            .filter(|t| !disabled.contains(&t.server_name.to_string()))
        {
            if let Some(arr) = tools.as_array_mut() {
                arr.push(json!({
                    "name": t.qualified_name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }));
            }
        }
    }

    fn build_snapshot(&self) -> McpSnapshot {
        Self::snapshot_from(&self.inner.read_blocking())
    }

    fn snapshot_from(inner: &McpManagerInner) -> McpSnapshot {
        let disabled = &inner.disabled;

        let mut tool_counts: HashMap<&str, (usize, usize)> = HashMap::new();
        for tool in &inner.tools {
            tool_counts.entry(&tool.server_name).or_default().0 += 1;
        }
        for prompt in &inner.prompts {
            tool_counts.entry(&prompt.server_name).or_default().1 += 1;
        }

        let infos = inner
            .entries
            .iter()
            .map(|entry| {
                let name = &entry.name;
                let status = if disabled.contains(name) {
                    McpServerStatus::Disabled
                } else {
                    entry.status.clone()
                };
                let (tool_count, prompt_count) = if matches!(status, McpServerStatus::Running) {
                    tool_counts.get(name.as_str()).copied().unwrap_or((0, 0))
                } else {
                    (0, 0)
                };
                McpServerInfo {
                    name: name.clone(),
                    transport_kind: entry.transport_kind,
                    tool_count,
                    prompt_count,
                    status,
                    config_path: entry.origin.clone(),
                    url: entry
                        .config
                        .as_ref()
                        .and_then(|c| transport_url(&c.transport)),
                }
            })
            .collect();

        let prompts = inner
            .prompts
            .iter()
            .filter(|p| !disabled.iter().any(|d| **d == *p.server_name))
            .map(|p| p.to_info())
            .collect();

        let pids = inner
            .transports
            .values()
            .flat_map(|t| t.child_pids())
            .collect();

        McpSnapshot {
            infos,
            prompts,
            pids,
            generation: inner.generation,
        }
    }

    async fn get_prompt(
        &self,
        qualified_name: &str,
        arguments: &HashMap<String, String>,
    ) -> Result<Vec<protocol::PromptMessage>, McpError> {
        let inner = self.inner.read().await;
        let def = inner
            .prompts
            .iter()
            .find(|p| p.qualified_name == qualified_name)
            .ok_or_else(|| McpError::UnknownPrompt {
                name: qualified_name.into(),
            })?;
        let t = inner
            .transports
            .get(&def.server_name)
            .ok_or_else(|| McpError::ServerDied {
                server: (*def.server_name).into(),
            })?;
        transport::get_prompt(t.as_ref(), &def.raw_name, arguments).await
    }

    async fn refresh_server(&self, server_name: &str) -> Result<(), McpError> {
        self.refresh_server_with(server_name, None).await
    }

    async fn refresh_server_with(
        &self,
        server_name: &str,
        config_override: Option<ServerConfig>,
    ) -> Result<(), McpError> {
        let (config, is_override) = match config_override {
            Some(c) => (c, true),
            None => {
                let inner = self.inner.read().await;
                let cfg = inner
                    .entries
                    .iter()
                    .find(|e| e.name == server_name)
                    .and_then(|e| e.config.clone())
                    .ok_or_else(|| McpError::Config(format!("unknown server '{server_name}'")))?;
                (cfg, false)
            }
        };

        let old_transport = {
            let mut inner = self.inner.write().await;
            if let Some(entry) = inner.entries.iter_mut().find(|e| e.name == server_name) {
                entry.status = McpServerStatus::Connecting;
            }
            inner.transports.remove(server_name)
        };

        if let Some(old) = old_transport {
            old.shutdown().await;
        }

        let result = Self::start_server(&config).await;

        let mut inner = self.inner.write().await;

        if is_override && let Some(entry) = inner.entries.iter_mut().find(|e| e.name == server_name)
        {
            entry.config = Some(config.clone());
        }

        let status = match result {
            Ok((transport, new_tools, new_prompts)) => {
                let server_key: Arc<str> = Arc::from(server_name);

                inner.tools.retain(|t| *t.server_name != *server_name);
                inner.prompts.retain(|p| *p.server_name != *server_name);
                inner.tool_index = inner
                    .tools
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (t.qualified_name, i))
                    .collect();

                for tool_info in new_tools {
                    let qualified = format!("{server_name}{SEPARATOR}{}", tool_info.name);
                    let interned = intern(qualified);
                    let idx = inner.tools.len();
                    inner.tools.push(McpToolDef {
                        qualified_name: interned,
                        server_name: Arc::clone(&server_key),
                        raw_name: tool_info.name,
                        description: tool_info.description,
                        input_schema: tool_info.input_schema,
                    });
                    inner.tool_index.insert(interned, idx);
                }

                for info in new_prompts {
                    inner
                        .prompts
                        .push(McpPromptDef::from_info(&server_key, info));
                }

                inner.transports.insert(Arc::clone(&server_key), transport);

                info!(
                    server = server_name,
                    tools = inner.tools.len(),
                    "MCP server refreshed"
                );
                Ok(McpServerStatus::Running)
            }
            Err(e) => {
                let s = if let McpError::HttpError {
                    status: 401,
                    ref reason,
                    ..
                } = e
                {
                    McpServerStatus::NeedsAuth {
                        url: Some(reason.clone()),
                    }
                } else {
                    warn!(server = server_name, error = %e, "failed to refresh MCP server");
                    McpServerStatus::Failed(e.to_string())
                };
                Err((s, e))
            }
        };

        if let Some(entry) = inner.entries.iter_mut().find(|e| e.name == server_name) {
            match &status {
                Ok(s) | Err((s, _)) => entry.status = s.clone(),
            }
        }

        status.map(|_| ()).map_err(|(_, e)| e)
    }
}

fn toggle_disabled(disabled: &mut Vec<String>, name: &str, enabled: bool) {
    if enabled {
        disabled.retain(|s| s != name);
    } else if !disabled.contains(&name.to_owned()) {
        disabled.push(name.to_owned());
    }
}

#[cfg(unix)]
pub fn kill_process_groups(pids: &[u32]) {
    for &pid in pids {
        unsafe { libc::killpg(pid as i32, libc::SIGKILL) };
    }
}

#[cfg(not(unix))]
pub fn kill_process_groups(_pids: &[u32]) {}

fn intern(name: String) -> &'static str {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut set = CACHE
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(&existing) = set.get(name.as_str()) {
        return existing;
    }
    let leaked: &'static str = Box::leak(name.into_boxed_str());
    set.insert(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;
    use config::{McpServerStatus, RawServerConfig, RawStdioFields, RawTransport};
    use std::collections::HashMap;
    use std::path::PathBuf;

    const DEFAULT_TIMEOUT_MS: u64 = 30_000;

    fn stdio_raw(cmd: &[&str]) -> RawServerConfig {
        RawServerConfig {
            enabled: true,
            timeout: DEFAULT_TIMEOUT_MS,
            transport: RawTransport::Stdio(RawStdioFields {
                command: cmd.iter().map(|s| s.to_string()).collect(),
                environment: HashMap::new(),
            }),
        }
    }

    fn make_config(entries: Vec<(&str, RawServerConfig)>) -> McpConfig {
        let mut mcp = HashMap::new();
        let mut origins = HashMap::new();
        for (name, cfg) in entries {
            origins.insert(name.to_string(), PathBuf::from("/test/config.toml"));
            mcp.insert(name.to_string(), cfg);
        }
        McpConfig { mcp, origins }
    }

    #[test]
    fn empty_config_returns_none() {
        smol::block_on(async {
            let config = McpConfig::default();
            let snapshot = Arc::new(ArcSwap::from_pointee(McpSnapshot {
                infos: vec![],
                prompts: vec![],
                pids: vec![],
                generation: 0,
            }));
            let result = McpManager::start_with_config(config, vec![], snapshot).await;
            assert!(result.is_none());
        });
    }

    #[test]
    fn mixed_config_produces_correct_entries() {
        smol::block_on(async {
            let mut disabled = stdio_raw(&["echo"]);
            disabled.enabled = false;
            let config = make_config(vec![
                ("disabled-srv", disabled),
                ("bad-srv", stdio_raw(&[])),
                ("also-bad", stdio_raw(&[])),
            ]);
            let snapshot = Arc::new(ArcSwap::from_pointee(McpSnapshot {
                infos: vec![],
                prompts: vec![],
                pids: vec![],
                generation: 0,
            }));
            let handle = McpManager::start_with_config(config, vec![], snapshot)
                .await
                .unwrap();
            let snap = handle.snapshot.load();
            let mut infos = snap.infos.clone();
            infos.sort_by(|a, b| a.name.cmp(&b.name));
            assert_eq!(infos.len(), 3);
            assert!(matches!(infos[0].status, McpServerStatus::Failed(_)));
            assert_eq!(infos[0].tool_count, 0);
            assert!(matches!(infos[1].status, McpServerStatus::Failed(_)));
            assert_eq!(infos[2].status, McpServerStatus::Disabled);
            assert_eq!(infos[2].config_path, PathBuf::from("/test/config.toml"));
            drop(handle);
        });
    }
}
