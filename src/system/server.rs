use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, RwLock};

use super::domain::{
    now_millis, ApprovalCategory, ProcessStartRequest, RuntimeEventKind, SandboxMode,
};
use super::runtime::{
    ConversationRuntime, EventFilter, ThreadReadParams, ThreadStartParams, TurnStartParams,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    pub data: Value,
}

struct ClientConnectionState {
    initialized: bool,
    declared_notifications: HashSet<String>,
}

struct ManagedProcess {
    child: Arc<Mutex<Child>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    output_buffer: Arc<Mutex<Vec<Value>>>,
}

struct ProcessTable {
    processes: RwLock<HashMap<String, Arc<ManagedProcess>>>,
}

impl ProcessTable {
    fn new() -> Self {
        Self {
            processes: RwLock::new(HashMap::new()),
        }
    }

    async fn start(
        &self,
        request: ProcessStartRequest,
        notification_method: &'static str,
        writer: Arc<Mutex<tokio::io::Stdout>>,
    ) -> Result<Value> {
        if request.argv.is_empty() {
            bail!("invalid params: argv must not be empty");
        }

        let mut command = Command::new(&request.argv[0]);
        if request.argv.len() > 1 {
            command.args(&request.argv[1..]);
        }
        command.current_dir(&request.cwd);
        command.stdin(if request.pipe_stdin {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::null()
        });
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        for (key, value) in &request.env {
            command.env(key, value);
        }
        let mut child = command.spawn()?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdin = child.stdin.take();

        let managed = Arc::new(ManagedProcess {
            child: Arc::new(Mutex::new(child)),
            stdin: Arc::new(Mutex::new(stdin)),
            output_buffer: Arc::new(Mutex::new(Vec::new())),
        });
        self.processes
            .write()
            .await
            .insert(request.process_id.clone(), managed.clone());

        if let Some(stdout) = stdout {
            spawn_pipe_reader(
                request.process_id.clone(),
                stdout,
                "stdout",
                notification_method,
                writer.clone(),
                managed.output_buffer.clone(),
            );
        }
        if let Some(stderr) = stderr {
            spawn_pipe_reader(
                request.process_id.clone(),
                stderr,
                "stderr",
                notification_method,
                writer.clone(),
                managed.output_buffer.clone(),
            );
        }

        let child = managed.child.clone();
        let process_id = request.process_id.clone();
        let output_buffer = managed.output_buffer.clone();
        tokio::spawn(async move {
            let exit_code = child
                .lock()
                .await
                .wait()
                .await
                .ok()
                .and_then(|status| status.code());
            output_buffer.lock().await.push(json!({
                "stream": "status",
                "exitCode": exit_code
            }));
            let _ = write_json_line(
                &writer,
                &JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: None,
                    result: Some(json!({
                        "method": if notification_method == "process/output" { "process/exited" } else { "command/outputDelta" },
                        "params": {
                            "processId": process_id,
                            "exitCode": exit_code
                        }
                    })),
                    error: None,
                },
            )
            .await;
        });

        Ok(json!({
            "processId": request.process_id,
            "started": true
        }))
    }

    async fn read(&self, process_id: &str) -> Result<Value> {
        let process = self
            .processes
            .read()
            .await
            .get(process_id)
            .cloned()
            .ok_or_else(|| anyhow!("process not found: {process_id}"))?;
        let mut buffer = process.output_buffer.lock().await;
        let output = std::mem::take(&mut *buffer);
        Ok(json!({ "processId": process_id, "chunks": output }))
    }

    async fn write(&self, process_id: &str, input: &str) -> Result<Value> {
        let process = self
            .processes
            .read()
            .await
            .get(process_id)
            .cloned()
            .ok_or_else(|| anyhow!("process not found: {process_id}"))?;
        if let Some(stdin) = process.stdin.lock().await.as_mut() {
            stdin.write_all(input.as_bytes()).await?;
            stdin.flush().await?;
        }
        Ok(json!({ "processId": process_id, "written": input.len() }))
    }

    async fn resize(&self, process_id: &str, cols: u16, rows: u16) -> Result<Value> {
        Ok(json!({ "processId": process_id, "cols": cols, "rows": rows, "resized": false }))
    }

    async fn terminate(&self, process_id: &str) -> Result<Value> {
        let process = self
            .processes
            .write()
            .await
            .remove(process_id)
            .ok_or_else(|| anyhow!("process not found: {process_id}"))?;
        process.child.lock().await.kill().await?;
        Ok(json!({ "processId": process_id, "terminated": true }))
    }
}

pub struct AppServer {
    runtime: ConversationRuntime,
    connections: RwLock<HashMap<String, ClientConnectionState>>,
    process_table: ProcessTable,
}

impl AppServer {
    pub fn new(runtime: ConversationRuntime) -> Self {
        Self {
            runtime,
            connections: RwLock::new(HashMap::new()),
            process_table: ProcessTable::new(),
        }
    }

    pub async fn start_stdio(&self) -> Result<()> {
        let connection_id = "stdio".to_string();
        self.connections.write().await.insert(
            connection_id.clone(),
            ClientConnectionState {
                initialized: false,
                declared_notifications: HashSet::new(),
            },
        );

        let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
        let mut subscription = self
            .runtime
            .event_bus()
            .subscribe("app_server".into(), EventFilter { thread_id: None });
        let writer = stdout.clone();
        tokio::spawn(async move {
            while let Some(event) = subscription.receiver.recv().await {
                let method = match event.kind {
                    RuntimeEventKind::ThreadStarted => "thread/started",
                    RuntimeEventKind::TurnStarted => "turn/started",
                    RuntimeEventKind::TurnCompleted => "turn/completed",
                    RuntimeEventKind::TurnFailed => "turn/failed",
                    RuntimeEventKind::ItemStarted => "item/started",
                    RuntimeEventKind::ItemUpdated => "item/updated",
                    RuntimeEventKind::ItemCompleted => "item/completed",
                    RuntimeEventKind::ApprovalRequested => "approval/requested",
                    RuntimeEventKind::ApprovalResolved => "approval/resolved",
                    RuntimeEventKind::Warning => "warning",
                    RuntimeEventKind::Disconnected => "warning",
                    RuntimeEventKind::CompactionStatus => "compaction/status",
                    RuntimeEventKind::ChildAgentSpawned => "agent/spawned",
                    RuntimeEventKind::TokenUsageUpdate => "token/usage/update",
                    RuntimeEventKind::SpeculativeCandidateStarted => "speculative/candidate/started",
                    RuntimeEventKind::SpeculativeCandidateCompleted => "speculative/candidate/completed",
                    RuntimeEventKind::SpeculativeJudgeStarted => "speculative/judge/started",
                    RuntimeEventKind::SpeculativeJudgeCompleted => "speculative/judge/completed",
                    RuntimeEventKind::CheckpointReviewStarted => "checkpoint/review/started",
                    RuntimeEventKind::CheckpointReviewCompleted => "checkpoint/review/completed",
                };
                let _ = write_json_line(
                    &writer,
                    &JsonRpcNotification {
                        jsonrpc: "2.0".into(),
                        method: method.into(),
                        params: serde_json::to_value(event).unwrap_or_else(|_| json!({})),
                    },
                )
                .await;
            }
        });

        let stdin = BufReader::new(tokio::io::stdin());
        let mut lines = stdin.lines();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(&line)?;
            if value.get("id").is_some() {
                let request: JsonRpcRequest = serde_json::from_value(value)?;
                let response = self
                    .handle_request(&connection_id, request, stdout.clone())
                    .await;
                write_json_line(&stdout, &response).await?;
            } else {
                let notification: JsonRpcNotification = serde_json::from_value(value)?;
                self.handle_notification(&connection_id, notification)
                    .await?;
            }
        }
        Ok(())
    }

    async fn handle_request(
        &self,
        connection_id: &str,
        request: JsonRpcRequest,
        stdout: Arc<Mutex<tokio::io::Stdout>>,
    ) -> JsonRpcResponse {
        match self
            .handle_request_inner(connection_id, &request, stdout)
            .await
        {
            Ok(result) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: Some(request.id),
                result: Some(result),
                error: None,
            },
            Err(error) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: Some(request.id),
                result: None,
                error: Some(map_app_error(error)),
            },
        }
    }

    async fn handle_request_inner(
        &self,
        connection_id: &str,
        request: &JsonRpcRequest,
        stdout: Arc<Mutex<tokio::io::Stdout>>,
    ) -> Result<Value> {
        if request.method != "initialize" {
            let initialized = self
                .connections
                .read()
                .await
                .get(connection_id)
                .map(|state| state.initialized)
                .unwrap_or(false);
            if !initialized {
                bail!("not_initialized");
            }
        }

        match request.method.as_str() {
            "initialize" => {
                if let Some(state) = self.connections.write().await.get_mut(connection_id) {
                    state.initialized = true;
                    state.declared_notifications = request
                        .params
                        .get("capabilities")
                        .and_then(|cap| cap.get("notifications"))
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(Value::as_str)
                                .map(ToOwned::to_owned)
                                .collect::<HashSet<_>>()
                        })
                        .unwrap_or_default();
                }
                Ok(json!({
                    "serverInfo": { "name": "agent-app-server", "version": env!("CARGO_PKG_VERSION") },
                    "protocolVersion": "1.0.0",
                    "capabilities": {
                        "concurrentTurnsPerThread": false,
                        "supportedSandboxModes": [SandboxMode::ReadOnly, SandboxMode::WorkspaceWrite, SandboxMode::DangerFullAccess],
                        "supportedApprovalCategories": [
                            ApprovalCategory::SandboxEscalation,
                            ApprovalCategory::FileChange,
                            ApprovalCategory::RequestPermissions
                        ],
                        "supportedReviewTargetKinds": ["WORKING_TREE", "GIT_RANGE"],
                        "streamingItemDeltaMode": "append",
                        "hooks": false,
                        "remoteInteractive": false,
                        "extensions": {}
                    },
                    "appHome": self.runtime.effective_config().app_home,
                    "platform": std::env::consts::OS
                }))
            }
            "thread/start" => Ok(serde_json::to_value(
                self.runtime
                    .start_thread(serde_json::from_value(request.params.clone())?)
                    .await?,
            )?),
            "thread/resume" => Ok(serde_json::to_value(
                self.runtime
                    .resume_thread(serde_json::from_value(request.params.clone())?)
                    .await?,
            )?),
            "thread/fork" => Ok(serde_json::to_value(
                self.runtime
                    .fork_thread(serde_json::from_value(request.params.clone())?)
                    .await?,
            )?),
            "thread/list" => Ok(serde_json::to_value(
                self.runtime
                    .list_threads(serde_json::from_value(request.params.clone())?)
                    .await?,
            )?),
            "thread/read" => Ok(serde_json::to_value(
                self.runtime
                    .read_thread(serde_json::from_value(request.params.clone())?)
                    .await?,
            )?),
            "thread/archive" => {
                let params: ThreadReadParams = serde_json::from_value(request.params.clone())?;
                self.runtime
                    .read_thread(ThreadReadParams {
                        thread_id: params.thread_id.clone(),
                    })
                    .await?;
                self.runtime.archive_thread(&params.thread_id).await?;
                Ok(json!({ "threadId": params.thread_id, "archived": true }))
            }
            "thread/compact" => Ok(serde_json::to_value(
                self.runtime
                    .compact_thread(serde_json::from_value(request.params.clone())?)
                    .await?,
            )?),
            "thread/rollback" => Ok(serde_json::to_value(
                self.runtime
                    .rollback_thread(serde_json::from_value(request.params.clone())?)
                    .await?,
            )?),
            "thread/memoryMode/set" => {
                self.runtime
                    .set_thread_memory_mode(serde_json::from_value(request.params.clone())?)
                    .await?;
                Ok(json!({ "ok": true }))
            }
            "turn/start" => Ok(serde_json::to_value(
                self.runtime
                    .start_turn(
                        serde_json::from_value(request.params.clone())?,
                        super::domain::SurfaceKind::AppServer,
                    )
                    .await?,
            )?),
            "turn/interrupt" => Ok(serde_json::to_value(
                self.runtime
                    .interrupt_turn(serde_json::from_value(request.params.clone())?)
                    .await?,
            )?),
            "turn/steer" => Ok(serde_json::to_value(
                self.runtime
                    .steer_turn(serde_json::from_value(request.params.clone())?)
                    .await?,
            )?),
            "review/start" => {
                let cwd = request
                    .params
                    .get("target")
                    .and_then(|t| t.get("basePath"))
                    .and_then(Value::as_str)
                    .unwrap_or(&self.runtime.effective_config().working_directory)
                    .to_string();
                let prompt = "Review the current working tree and report findings.".to_string();
                let thread = self
                    .runtime
                    .start_thread(ThreadStartParams {
                        cwd: Some(cwd),
                        model_settings: None,
                        approval_policy: None,
                        permission_profile: None,
                        ephemeral: true,
                    })
                    .await?;
                let turn = self
                    .runtime
                    .start_turn(
                        TurnStartParams {
                            thread_id: thread.metadata.thread_id,
                            input: vec![super::domain::UserInput::from_text(prompt)],
                            cwd: None,
                            model_settings: None,
                            approval_policy: None,
                            permission_profile: None,
                            output_schema: None,
                            repo_map_verbosity: None,
                            agent_depth: 0,
                        },
                        super::domain::SurfaceKind::Review,
                    )
                    .await?;
                Ok(serde_json::to_value(turn)?)
            }
            "model/list" => Ok(serde_json::to_value(self.runtime.list_models())?),
            "command/exec" => Ok(self
                .process_table
                .start(
                    serde_json::from_value(request.params.clone())?,
                    "command/outputDelta",
                    stdout,
                )
                .await?),
            "command/exec/write" => {
                let process_id = request
                    .params
                    .get("processId")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let chars = request
                    .params
                    .get("chars")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.process_table.write(process_id, chars).await
            }
            "command/exec/resize" => {
                let process_id = request
                    .params
                    .get("processId")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let cols = request
                    .params
                    .get("cols")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as u16;
                let rows = request
                    .params
                    .get("rows")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as u16;
                self.process_table.resize(process_id, cols, rows).await
            }
            "command/exec/terminate" => {
                let process_id = request
                    .params
                    .get("processId")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.process_table.terminate(process_id).await
            }
            "fs/readFile" => fs_read_file(&request.params).await,
            "fs/writeFile" => fs_write_file(&request.params).await,
            "fs/createDirectory" => fs_create_directory(&request.params).await,
            "fs/getMetadata" => fs_get_metadata(&request.params).await,
            "fs/readDirectory" => fs_read_directory(&request.params).await,
            "fs/remove" => fs_remove(&request.params).await,
            "fs/copy" => fs_copy(&request.params).await,
            "fs/watch" => Ok(json!({ "watchId": format!("watch_{}", now_millis()) })),
            "fs/unwatch" => Ok(json!({ "ok": true })),
            "skills/list" => Ok(serde_json::to_value(self.runtime.list_skills().await)?),
            "plugin/list" => Ok(serde_json::to_value(self.runtime.list_plugins().await)?),
            "plugin/install" => Ok(
                json!({ "installed": false, "reason": "plugin installation is not enabled in this build" }),
            ),
            "plugin/uninstall" => Ok(
                json!({ "uninstalled": false, "reason": "plugin installation is not enabled in this build" }),
            ),
            "app/list" => Ok(serde_json::to_value(self.runtime.list_connectors().await)?),
            "config/read" => Ok(serde_json::to_value(self.runtime.effective_config())?),
            "config/value/write" => Ok(
                json!({ "written": false, "reason": "runtime config writes are not exposed through this server build" }),
            ),
            "config/batchWrite" => Ok(
                json!({ "written": false, "reason": "runtime config writes are not exposed through this server build" }),
            ),
            "memory/reset" => {
                self.runtime.reset_memories()?;
                Ok(json!({ "reset": true }))
            }
            _ => bail!("method not found: {}", request.method),
        }
    }

    async fn handle_notification(
        &self,
        _connection_id: &str,
        notification: JsonRpcNotification,
    ) -> Result<()> {
        if notification.method == "initialized" {
            return Ok(());
        }
        if notification.method == "approval/resolve" {
            let approval_id = notification
                .params
                .get("approvalId")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let decision = notification
                .params
                .get("decision")
                .cloned()
                .unwrap_or_else(|| json!("DENIED"));
            self.runtime
                .resolve_approval(
                    approval_id,
                    serde_json::from_value(decision)
                        .unwrap_or(super::domain::ApprovalDecision::Denied),
                    None,
                )
                .await?;
        }
        Ok(())
    }
}

pub struct ExecServer {
    process_table: ProcessTable,
}

impl ExecServer {
    pub fn new() -> Self {
        Self {
            process_table: ProcessTable::new(),
        }
    }

    pub async fn start_stdio(&self) -> Result<()> {
        let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
        let stdin = BufReader::new(tokio::io::stdin());
        let mut lines = stdin.lines();
        let mut initialized = false;

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let request: JsonRpcRequest = serde_json::from_str(&line)?;
            let response = if !initialized && request.method != "initialize" {
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: Some(request.id),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32000,
                        message: "not_initialized".into(),
                        data: json!({ "retriable": true }),
                    }),
                }
            } else {
                let result = match request.method.as_str() {
                    "initialize" => {
                        initialized = true;
                        Ok(json!({
                            "serverInfo": { "name": "agent-exec-server", "version": env!("CARGO_PKG_VERSION") },
                            "protocolVersion": "1.0.0",
                            "capabilities": {
                                "supportedSandboxModes": [SandboxMode::ReadOnly, SandboxMode::WorkspaceWrite, SandboxMode::DangerFullAccess]
                            }
                        }))
                    }
                    "process/start" => {
                        self.process_table
                            .start(
                                serde_json::from_value(request.params.clone())?,
                                "process/output",
                                stdout.clone(),
                            )
                            .await
                    }
                    "process/read" => {
                        let process_id = request
                            .params
                            .get("processId")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        self.process_table.read(process_id).await
                    }
                    "process/write" => {
                        let process_id = request
                            .params
                            .get("processId")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let chars = request
                            .params
                            .get("chars")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        self.process_table.write(process_id, chars).await
                    }
                    "process/resize" => {
                        let process_id = request
                            .params
                            .get("processId")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let cols = request
                            .params
                            .get("cols")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as u16;
                        let rows = request
                            .params
                            .get("rows")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as u16;
                        self.process_table.resize(process_id, cols, rows).await
                    }
                    "process/terminate" => {
                        let process_id = request
                            .params
                            .get("processId")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        self.process_table.terminate(process_id).await
                    }
                    "fs/readFile" => fs_read_file(&request.params).await,
                    "fs/writeFile" => fs_write_file(&request.params).await,
                    "fs/createDirectory" => fs_create_directory(&request.params).await,
                    "fs/getMetadata" => fs_get_metadata(&request.params).await,
                    "fs/readDirectory" => fs_read_directory(&request.params).await,
                    "fs/remove" => fs_remove(&request.params).await,
                    "fs/copy" => fs_copy(&request.params).await,
                    _ => bail!("method not found: {}", request.method),
                };
                match result {
                    Ok(result) => JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: Some(request.id),
                        result: Some(result),
                        error: None,
                    },
                    Err(error) => JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: Some(request.id),
                        result: None,
                        error: Some(map_app_error(error)),
                    },
                }
            };
            write_json_line(&stdout, &response).await?;
        }
        Ok(())
    }
}

async fn fs_read_file(params: &Value) -> Result<Value> {
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing path"))?;
    let content = tokio::fs::read(path).await?;
    Ok(json!({ "path": path, "content": String::from_utf8_lossy(&content) }))
}

async fn fs_write_file(params: &Value) -> Result<Value> {
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing path"))?;
    let content = params
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if let Some(parent) = std::path::Path::new(path).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, content).await?;
    Ok(json!({ "path": path, "written": content.len() }))
}

async fn fs_create_directory(params: &Value) -> Result<Value> {
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing path"))?;
    let recursive = params
        .get("recursive")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if recursive {
        tokio::fs::create_dir_all(path).await?;
    } else {
        tokio::fs::create_dir(path).await?;
    }
    Ok(json!({ "path": path, "created": true }))
}

async fn fs_get_metadata(params: &Value) -> Result<Value> {
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing path"))?;
    let meta = tokio::fs::metadata(path).await?;
    Ok(json!({
        "path": path,
        "isFile": meta.is_file(),
        "isDirectory": meta.is_dir(),
        "size": meta.len()
    }))
}

async fn fs_read_directory(params: &Value) -> Result<Value> {
    let path = params.get("path").and_then(Value::as_str).unwrap_or(".");
    let mut entries = Vec::new();
    let mut rd = tokio::fs::read_dir(path).await?;
    while let Some(entry) = rd.next_entry().await? {
        let file_type = entry.file_type().await?;
        entries.push(json!({
            "name": entry.file_name().to_string_lossy(),
            "path": entry.path(),
            "isDirectory": file_type.is_dir()
        }));
    }
    Ok(json!({ "path": path, "entries": entries }))
}

async fn fs_remove(params: &Value) -> Result<Value> {
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing path"))?;
    let recursive = params
        .get("recursive")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let meta = tokio::fs::metadata(path).await?;
    if meta.is_dir() {
        if recursive {
            tokio::fs::remove_dir_all(path).await?;
        } else {
            tokio::fs::remove_dir(path).await?;
        }
    } else {
        tokio::fs::remove_file(path).await?;
    }
    Ok(json!({ "path": path, "removed": true }))
}

async fn fs_copy(params: &Value) -> Result<Value> {
    let source = params
        .get("source")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing source"))?;
    let target = params
        .get("target")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing target"))?;
    if let Some(parent) = std::path::Path::new(target).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::copy(source, target).await?;
    Ok(json!({ "source": source, "target": target, "copied": true }))
}

fn spawn_pipe_reader<R>(
    process_id: String,
    mut reader: R,
    stream_name: &'static str,
    notification_method: &'static str,
    writer: Arc<Mutex<tokio::io::Stdout>>,
    buffer: Arc<Mutex<Vec<Value>>>,
) where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut bytes = vec![0_u8; 4096];
        loop {
            match reader.read(&mut bytes).await {
                Ok(0) => break,
                Ok(n) => {
                    let text = String::from_utf8_lossy(&bytes[..n]).to_string();
                    let payload = json!({
                        "processId": process_id,
                        "stream": stream_name,
                        "delta": text
                    });
                    buffer.lock().await.push(payload.clone());
                    let _ = write_json_line(
                        &writer,
                        &JsonRpcNotification {
                            jsonrpc: "2.0".into(),
                            method: notification_method.into(),
                            params: payload,
                        },
                    )
                    .await;
                }
                Err(_) => break,
            }
        }
    });
}

async fn write_json_line<T: Serialize>(
    stdout: &Arc<Mutex<tokio::io::Stdout>>,
    value: &T,
) -> Result<()> {
    let mut stdout = stdout.lock().await;
    stdout
        .write_all(format!("{}\n", serde_json::to_string(value)?).as_bytes())
        .await?;
    stdout.flush().await?;
    Ok(())
}

fn map_app_error(error: anyhow::Error) -> JsonRpcError {
    let message = error.to_string();
    let (code, symbol) = if message.contains("not_initialized") {
        (-32000, "not_initialized")
    } else if message.contains("thread_not_found") {
        (-32010, "thread_not_found")
    } else if message.contains("turn_not_found") {
        (-32011, "turn_not_found")
    } else if message.contains("turn_already_active") {
        (-32012, "turn_already_active")
    } else if message.contains("turn_mismatch") {
        (-32013, "turn_mismatch")
    } else if message.contains("thread_archived") {
        (-32014, "thread_archived")
    } else if message.contains("network_blocked") {
        (-32032, "network_blocked")
    } else if message.contains("path_not_writable") {
        (-32031, "path_not_writable")
    } else if message.contains("tool_not_found") {
        (-32040, "tool_not_found")
    } else if message.contains("tool_invalid_arguments") {
        (-32041, "tool_invalid_arguments")
    } else if message.contains("tool_execution_failed") {
        (-32042, "tool_execution_failed")
    } else {
        (-32099, "unspecified_error")
    };

    JsonRpcError {
        code,
        message: symbol.into(),
        data: json!({
            "detail": message,
            "retriable": false
        }),
    }
}
