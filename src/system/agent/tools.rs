use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use regex::Regex;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::RwLock as AsyncRwLock;
use walkdir::WalkDir;

use crate::system::domain::{
    ApprovalCategory, ActionDescriptor, ToolCall, ToolDefinition, ToolExecutionContext,
    ToolListingContext, ToolCallId, ToolProviderKind, ToolResult,
};
use super::permission::PermissionManager;
use super::sandbox::SandboxManager;

#[async_trait]
pub trait ToolProvider: Send + Sync {
    #[allow(dead_code)]
    fn get_kind(&self) -> ToolProviderKind;
    fn list_tools(&self, context: &ToolListingContext) -> Vec<ToolDefinition>;
    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult>;
}

// ─── ShellToolProvider ────────────────────────────────────────────────────────

pub struct ShellToolProvider {
    sandbox: Arc<SandboxManager>,
    permissions: Arc<PermissionManager>,
}

impl ShellToolProvider {
    pub fn new(sandbox: Arc<SandboxManager>, permissions: Arc<PermissionManager>) -> Self {
        Self { sandbox, permissions }
    }
}

#[async_trait]
impl ToolProvider for ShellToolProvider {
    fn get_kind(&self) -> ToolProviderKind { ToolProviderKind::Builtin }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "shell_exec".into(),
            provider_kind: ToolProviderKind::Builtin,
            description: "Run a local command with explicit argv.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "argv": { "type": "array", "items": { "type": "string" } },
                    "cwd": { "type": "string" },
                    "env": { "type": "object", "additionalProperties": { "type": "string" } }
                },
                "required": ["argv"]
            }),
            requires_approval: true,
            supports_parallel_calls: false,
        }]
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        let argv = call
            .arguments.get("argv").and_then(Value::as_array)
            .ok_or_else(|| anyhow!("tool_invalid_arguments: argv must be an array"))?
            .iter().filter_map(Value::as_str).map(ToOwned::to_owned).collect::<Vec<_>>();
        let cwd = call.arguments.get("cwd").and_then(Value::as_str)
            .unwrap_or(&context.cwd).to_string();
        let env = call.arguments.get("env").and_then(Value::as_object)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect::<HashMap<_, _>>())
            .unwrap_or_default();
        let action = ActionDescriptor {
            action_type: "command".into(),
            command: Some(argv.clone()),
            paths: vec![cwd.clone()],
            domains: Vec::new(),
            category: ApprovalCategory::SandboxEscalation,
        };
        let sandbox = self.permissions.build_sandbox_request(&action, &context.permission_profile);
        let exec = self.sandbox.run_command(&argv, &cwd, &env, &sandbox).await?;
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: exec.exit_code.unwrap_or(-1) == 0,
            output: serde_json::to_value(exec)?,
            error_message: None,
        })
    }
}

// ─── FileToolProvider ─────────────────────────────────────────────────────────

pub struct FileToolProvider {
    sandbox: Arc<SandboxManager>,
    permissions: Arc<PermissionManager>,
}

impl FileToolProvider {
    pub fn new(sandbox: Arc<SandboxManager>, permissions: Arc<PermissionManager>) -> Self {
        Self { sandbox, permissions }
    }
}

#[async_trait]
impl ToolProvider for FileToolProvider {
    fn get_kind(&self) -> ToolProviderKind { ToolProviderKind::Builtin }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "read_file".into(), provider_kind: ToolProviderKind::Builtin,
                description: "Read a UTF-8 file.".into(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
                requires_approval: false, supports_parallel_calls: true,
            },
            ToolDefinition {
                name: "write_file".into(), provider_kind: ToolProviderKind::Builtin,
                description: "Write a UTF-8 file.".into(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
                requires_approval: true, supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "create_directory".into(), provider_kind: ToolProviderKind::Builtin,
                description: "Create a directory.".into(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"]}),
                requires_approval: true, supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "remove_path".into(), provider_kind: ToolProviderKind::Builtin,
                description: "Remove a file or directory.".into(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"},"force":{"type":"boolean"}},"required":["path"]}),
                requires_approval: true, supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "copy_path".into(), provider_kind: ToolProviderKind::Builtin,
                description: "Copy a file.".into(),
                input_schema: json!({"type":"object","properties":{"source":{"type":"string"},"target":{"type":"string"},"recursive":{"type":"boolean"}},"required":["source","target"]}),
                requires_approval: true, supports_parallel_calls: false,
            },
        ]
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        let path = |key: &str| {
            call.arguments.get(key).and_then(Value::as_str)
                .ok_or_else(|| anyhow!("tool_invalid_arguments: missing {key}"))
        };
        let write_action = matches!(call.tool_name.as_str(), "write_file" | "create_directory" | "remove_path" | "copy_path");
        let action = ActionDescriptor {
            action_type: call.tool_name.clone(),
            command: None,
            paths: match call.tool_name.as_str() {
                "copy_path" => vec![path("source")?.into(), path("target")?.into()],
                _ => vec![path("path")?.into()],
            },
            domains: Vec::new(),
            category: if write_action { ApprovalCategory::FileChange } else { ApprovalCategory::Other },
        };
        let sandbox = self.permissions.build_sandbox_request(&action, &context.permission_profile);
        let output = match call.tool_name.as_str() {
            "read_file" => {
                let content = self.sandbox.read_file(path("path")?, &sandbox).await?;
                json!({ "path": path("path")?, "content": String::from_utf8_lossy(&content) })
            }
            "write_file" => {
                let content = call.arguments.get("content").and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("tool_invalid_arguments: missing content"))?;
                self.sandbox.write_file(path("path")?, content.as_bytes(), &sandbox).await?;
                json!({ "ok": true, "path": path("path")? })
            }
            "create_directory" => {
                let recursive = call.arguments.get("recursive").and_then(Value::as_bool).unwrap_or(true);
                self.sandbox.create_directory(path("path")?, recursive, &sandbox).await?;
                json!({ "ok": true, "path": path("path")? })
            }
            "remove_path" => {
                let recursive = call.arguments.get("recursive").and_then(Value::as_bool).unwrap_or(true);
                let force = call.arguments.get("force").and_then(Value::as_bool).unwrap_or(false);
                self.sandbox.remove_path(path("path")?, recursive, force, &sandbox).await?;
                json!({ "ok": true, "path": path("path")? })
            }
            "copy_path" => {
                let recursive = call.arguments.get("recursive").and_then(Value::as_bool).unwrap_or(false);
                self.sandbox.copy_path(path("source")?, path("target")?, recursive, &sandbox).await?;
                json!({ "ok": true, "source": path("source")?, "target": path("target")? })
            }
            other => bail!("tool_not_found: {other}"),
        };
        Ok(ToolResult { tool_call_id: call.tool_call_id.clone(), ok: true, output, error_message: None })
    }
}

// ─── SearchToolProvider ───────────────────────────────────────────────────────

pub struct SearchToolProvider;

#[async_trait]
impl ToolProvider for SearchToolProvider {
    fn get_kind(&self) -> ToolProviderKind { ToolProviderKind::Builtin }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "grep_search".into(), provider_kind: ToolProviderKind::Builtin,
            description: "Search files for a text pattern (ripgrep if available, native fallback).".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regex or literal pattern to search for" },
                    "path": { "type": "string", "description": "Directory or file to search (defaults to cwd)" },
                    "literal": { "type": "boolean", "description": "If true, treat pattern as literal string" },
                    "max_results": { "type": "integer", "description": "Maximum number of matches to return (default 50)" }
                },
                "required": ["pattern"]
            }),
            requires_approval: false, supports_parallel_calls: true,
        }]
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        let pattern = call.arguments.get("pattern").and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool_invalid_arguments: missing pattern"))?;
        let search_path = call.arguments.get("path").and_then(Value::as_str).unwrap_or(&context.cwd);
        let literal = call.arguments.get("literal").and_then(Value::as_bool).unwrap_or(false);
        let max_results = call.arguments.get("max_results").and_then(Value::as_u64).unwrap_or(50) as usize;

        // Try ripgrep first (faster, respects .gitignore)
        let rg_result = if literal {
            Command::new("rg").args(["-n", "--hidden", "--no-heading", "-F", pattern, search_path])
                .stdout(Stdio::piped()).stderr(Stdio::piped()).output().await.ok()
        } else {
            Command::new("rg").args(["-n", "--hidden", "--no-heading", pattern, search_path])
                .stdout(Stdio::piped()).stderr(Stdio::piped()).output().await.ok()
        };

        if let Some(out) = rg_result.filter(|o| o.status.success() || o.status.code() == Some(1)) {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let matches: Vec<Value> = stdout.lines().take(max_results).map(|line| json!(line)).collect();
            return Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(), ok: true,
                output: json!({ "matches": matches, "source": "ripgrep" }),
                error_message: None,
            });
        }

        // Native fallback using walkdir + regex
        let re = if literal {
            Regex::new(&regex::escape(pattern))?
        } else {
            Regex::new(pattern)?
        };
        let mut matches: Vec<Value> = Vec::new();
        'walk: for entry in WalkDir::new(search_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                for (lineno, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        matches.push(json!(format!("{}:{}: {}", entry.path().display(), lineno + 1, line)));
                        if matches.len() >= max_results { break 'walk; }
                    }
                }
            }
        }
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(), ok: true,
            output: json!({ "matches": matches, "source": "native" }),
            error_message: None,
        })
    }
}

// ─── WebToolProvider ──────────────────────────────────────────────────────────

pub struct WebToolProvider {
    http: Client,
}

impl WebToolProvider {
    pub fn new() -> Self {
        Self {
            http: Client::builder()
                .timeout(Duration::from_secs(30))
                .user_agent("Codezilla/2.0 (https://github.com/basaid-dev/codezilla)")
                .build()
                .unwrap_or_default(),
        }
    }
}

#[async_trait]
impl ToolProvider for WebToolProvider {
    fn get_kind(&self) -> ToolProviderKind { ToolProviderKind::Builtin }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "web_fetch".into(), provider_kind: ToolProviderKind::Builtin,
            description: "Fetch a URL and return its content as plain text (HTML is converted).".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The URL to fetch" },
                    "max_chars": { "type": "integer", "description": "Maximum characters to return (default 8000)" }
                },
                "required": ["url"]
            }),
            requires_approval: false, supports_parallel_calls: true,
        }]
    }

    async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
        let url = call.arguments.get("url").and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool_invalid_arguments: missing url"))?;
        let max_chars = call.arguments.get("max_chars").and_then(Value::as_u64).unwrap_or(8000) as usize;

        let response = self.http.get(url).send().await
            .map_err(|e| anyhow!("web_fetch_error: {e}"))?;
        let status = response.status().as_u16();
        let content_type = response.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let bytes = response.bytes().await
            .map_err(|e| anyhow!("web_fetch_read_error: {e}"))?;

        let text = if content_type.contains("text/html") || url.ends_with(".html") || url.ends_with(".htm") {
            html2text::from_read(bytes.as_ref(), 120)
        } else {
            String::from_utf8_lossy(&bytes).into_owned()
        };

        let truncated = text.chars().take(max_chars).collect::<String>();
        let was_truncated = text.len() > max_chars;

        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: status < 400,
            output: json!({
                "url": url,
                "status": status,
                "content_type": content_type,
                "text": truncated,
                "truncated": was_truncated,
                "total_chars": text.len()
            }),
            error_message: if status >= 400 { Some(format!("HTTP {status}")) } else { None },
        })
    }
}

// ─── ImageToolProvider ────────────────────────────────────────────────────────

pub struct ImageToolProvider;

#[async_trait]
impl ToolProvider for ImageToolProvider {
    fn get_kind(&self) -> ToolProviderKind { ToolProviderKind::Builtin }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "image_metadata".into(), provider_kind: ToolProviderKind::Builtin,
            description: "Return local image metadata.".into(),
            input_schema: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
            requires_approval: false, supports_parallel_calls: true,
        }]
    }

    async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
        let path = call.arguments.get("path").and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool_invalid_arguments: missing path"))?;
        let meta = tokio::fs::metadata(path).await?;
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(), ok: true,
            output: json!({ "path": path, "size": meta.len() }),
            error_message: None,
        })
    }
}

// ─── Stub providers ───────────────────────────────────────────────────────────

pub struct SpawnAgentToolProvider;

#[async_trait]
impl ToolProvider for SpawnAgentToolProvider {
    fn get_kind(&self) -> ToolProviderKind { ToolProviderKind::Builtin }
    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> { Vec::new() }
    async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(), ok: false,
            output: json!({"message":"spawn_agent is not enabled in this build"}),
            error_message: Some("spawn_agent is not enabled in this build".into()),
        })
    }
}

pub struct RequestUserInputToolProvider;

#[async_trait]
impl ToolProvider for RequestUserInputToolProvider {
    fn get_kind(&self) -> ToolProviderKind { ToolProviderKind::Builtin }
    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> { Vec::new() }
    async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(), ok: false,
            output: json!({"message":"request_user_input is not available for model-driven calls"}),
            error_message: Some("request_user_input is not available for model-driven calls".into()),
        })
    }
}

// ─── ToolOrchestrator ─────────────────────────────────────────────────────────

pub struct ToolOrchestrator {
    registry: Arc<RwLock<Vec<Arc<dyn ToolProvider>>>>,
    running_tool_calls: Arc<AsyncRwLock<HashMap<ToolCallId, ToolExecutionContext>>>,
}

impl ToolOrchestrator {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RwLock::new(Vec::new())),
            running_tool_calls: Arc::new(AsyncRwLock::new(HashMap::new())),
        }
    }

    pub fn register_provider(&self, provider: Arc<dyn ToolProvider>) {
        self.registry.write().unwrap().push(provider);
    }

    pub fn list_available_tools(&self, context: &ToolListingContext) -> Vec<ToolDefinition> {
        self.registry.read().unwrap().iter()
            .flat_map(|p| p.list_tools(context)).collect()
    }

    pub async fn execute(&self, call: &ToolCall, context: ToolExecutionContext) -> Result<ToolResult> {
        self.running_tool_calls.write().await.insert(call.tool_call_id.clone(), context.clone());
        let providers = self.registry.read().unwrap().clone();
        for provider in providers {
            if provider.list_tools(&ToolListingContext {
                thread_id: context.thread_id.clone(),
                cwd: context.cwd.clone(),
                features: HashMap::new(),
            }).iter().any(|t| t.name == call.tool_name) {
                let result = provider.execute(call, &context).await;
                self.running_tool_calls.write().await.remove(&call.tool_call_id);
                return result;
            }
        }
        self.running_tool_calls.write().await.remove(&call.tool_call_id);
        bail!("tool_not_found: {}", call.tool_name)
    }
}
