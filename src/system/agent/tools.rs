use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use glob::Pattern as GlobPattern;
use regex::Regex;
use reqwest::Client;
use serde_json::{json, Value};
use similar::{ChangeTag, TextDiff};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::RwLock as AsyncRwLock;
use walkdir::WalkDir;

use super::permission::PermissionManager;
use super::sandbox::SandboxManager;
use crate::system::domain::{
    ActionDescriptor, ApprovalCategory, ToolCall, ToolCallId, ToolDefinition, ToolExecutionContext,
    ToolListingContext, ToolProviderKind, ToolResult,
};

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
        Self {
            sandbox,
            permissions,
        }
    }
}

#[async_trait]
impl ToolProvider for ShellToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "shell_exec".into(),
            provider_kind: ToolProviderKind::Builtin,
            description: concat!(
                "Spawn a local process directly (NO shell — advanced use only). ",
                "`argv` is a JSON array of separate string tokens, NOT a shell string. ",
                "Shell operators (|, >, 2>&1, &&) and globs (*) are NOT interpreted — ",
                "they will be passed as literal arguments and will cause errors. ",
                "Use `bash_exec` instead for any command that needs shell features. ",
                "Example: {\"argv\": [\"cargo\", \"build\", \"--release\"], \"cwd\": \"/path\"}"
            )
            .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Process argv as a JSON array of tokens. MUST be an array. Example: [\"git\", \"status\"]"
                    },
                    "cwd": { "type": "string", "description": "Working directory (absolute path)" },
                    "env": { "type": "object", "additionalProperties": { "type": "string" } }
                },
                "required": ["argv"]
            }),
            requires_approval: true,
            supports_parallel_calls: false,
        }]
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        // Accept both array and string argv for maximum model compatibility.
        // String argv is tokenized (no shell semantics — use bash_exec for shell features).
        let argv_value = call
            .arguments
            .get("argv")
            .ok_or_else(|| anyhow!("tool_invalid_arguments: argv is required"))?;

        let argv: Vec<String> = if let Some(arr) = argv_value.as_array() {
            arr.iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        } else if let Some(s) = argv_value.as_str() {
            tracing::warn!(
                tool_call_id = %call.tool_call_id,
                "shell_exec: argv was a string, not an array — tokenizing (no shell semantics; use bash_exec for pipes/redirects)"
            );
            simple_tokenize(s)
        } else {
            bail!("tool_invalid_arguments: argv must be a JSON array of strings")
        };

        if argv.is_empty() {
            bail!("tool_invalid_arguments: argv must not be empty");
        }

        let cwd = call
            .arguments
            .get("cwd")
            .and_then(Value::as_str)
            .unwrap_or(&context.cwd)
            .to_string();
        let env = call
            .arguments
            .get("env")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();
        let action = ActionDescriptor {
            action_type: "command".into(),
            command: Some(argv.clone()),
            paths: vec![cwd.clone()],
            domains: Vec::new(),
            category: ApprovalCategory::SandboxEscalation,
        };
        let sandbox = self
            .permissions
            .build_sandbox_request(&action, &context.permission_profile);
        let exec = self
            .sandbox
            .run_command(&argv, &cwd, &env, &sandbox)
            .await?;
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: exec.exit_code.unwrap_or(-1) == 0,
            output: serde_json::to_value(exec)?,
            error_message: None,
        })
    }
}

// ─── BashToolProvider ─────────────────────────────────────────────────────────

/// Runs commands via `bash -c "..."` — full shell semantics including pipes,
/// redirects, globs, &&, ||, subshells, etc. This is the primary tool for
/// agent shell interaction. `shell_exec` is kept for low-level argv use.
pub struct BashToolProvider {
    sandbox: Arc<SandboxManager>,
    permissions: Arc<PermissionManager>,
}

impl BashToolProvider {
    pub fn new(sandbox: Arc<SandboxManager>, permissions: Arc<PermissionManager>) -> Self {
        Self {
            sandbox,
            permissions,
        }
    }
}

#[async_trait]
impl ToolProvider for BashToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "bash_exec".into(),
            provider_kind: ToolProviderKind::Builtin,
            description: concat!(
                "Run a shell command via bash. ",
                "Supports pipes (|), redirects (>, >>, 2>&1), logical operators (&&, ||), ",
                "semicolons (;), globs (*, **), subshells ($(...)), and all other bash features. ",
                "The `command` field must be a plain shell string — the literal text you would type in a terminal. ",
                "For example, command=\"cargo build 2>&1\" or command=\"find . -name '*.rs' | wc -l\". ",
                "Do NOT put JSON inside the command field — write the shell command directly as a string."
            )
            .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "A bash shell command string. Supports |, >, 2>&1, &&, globs, etc. Write it exactly as you would type in a terminal."
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command (absolute path). Defaults to project root."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Maximum seconds to wait (default 60, max 300)."
                    },
                    "env": {
                        "type": "object",
                        "additionalProperties": { "type": "string" },
                        "description": "Extra environment variables to set for this command."
                    }
                },
                "required": ["command"]
            }),
            requires_approval: true,
            supports_parallel_calls: false,
        }]
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        // Primary: `command` is a plain shell string.
        // Fallback 1: model may have confused bash_exec with shell_exec and passed `argv` — join it.
        // Fallback 2: model double-encoded and put JSON inside the command string, e.g.
        //   {"command": "{\"command\":\"git status\"}{\"command\":\"git diff\"}"}
        //   Extract all "command" values from embedded JSON objects and join with " && ".
        let command_str;
        let raw: &str = if let Some(s) = call.arguments.get("command").and_then(Value::as_str) {
            s
        } else if let Some(arr) = call.arguments.get("argv").and_then(Value::as_array) {
            tracing::warn!(
                tool_call_id = %call.tool_call_id,
                "bash_exec: received `argv` instead of `command` — joining as shell string"
            );
            command_str = arr
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" ");
            &command_str
        } else {
            bail!(
                "tool_invalid_arguments: bash_exec requires a `command` string field. \
                 Example: command=\"git diff --stat\""
            )
        };

        // Detect double-encoded commands: model wrapped the shell string in JSON objects.
        // This happens when the model mistakes the tool-call JSON format for the command value.
        let command_str2;
        let command: &str = if raw.trim_start().starts_with('{') {
            let extracted = extract_commands_from_embedded_json(raw);
            if !extracted.is_empty() {
                tracing::warn!(
                    tool_call_id = %call.tool_call_id,
                    raw = %raw,
                    recovered = %extracted,
                    "bash_exec: command field contained JSON instead of a shell string — extracted and recovered"
                );
                command_str2 = extracted;
                &command_str2
            } else {
                raw
            }
        } else {
            raw
        };
        let cwd = call
            .arguments
            .get("cwd")
            .and_then(Value::as_str)
            .unwrap_or(&context.cwd)
            .to_string();
        let timeout_secs = call
            .arguments
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(60)
            .min(300);
        let env = call
            .arguments
            .get("env")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();

        // Wrap in bash so all shell operators work
        let argv = vec!["bash".to_string(), "-c".to_string(), command.to_string()];

        let action = ActionDescriptor {
            action_type: "command".into(),
            command: Some(argv.clone()),
            paths: vec![cwd.clone()],
            domains: Vec::new(),
            category: ApprovalCategory::SandboxEscalation,
        };
        let sandbox = self
            .permissions
            .build_sandbox_request(&action, &context.permission_profile);

        let exec_future = self.sandbox.run_command(&argv, &cwd, &env, &sandbox);
        let exec = tokio::time::timeout(Duration::from_secs(timeout_secs), exec_future)
            .await
            .map_err(|_| anyhow!("tool_execution_timeout: command exceeded {timeout_secs}s"))??;
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: exec.exit_code.unwrap_or(-1) == 0,
            output: serde_json::to_value(exec)?,
            error_message: None,
        })
    }
}

// ─── ListDirToolProvider ──────────────────────────────────────────────────────

/// Native Rust directory listing — no shell, no globs, no failures.
/// Use this instead of `find`, `ls`, or `tree` for directory exploration.
pub struct ListDirToolProvider;

#[async_trait]
impl ToolProvider for ListDirToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "list_dir".into(),
            provider_kind: ToolProviderKind::Builtin,
            description: concat!(
                "List the files and directories in a path. ",
                "Supports recursive listing with depth control and pattern filtering. ",
                "Use this instead of shell `find` or `ls` commands — it never fails on globs."
            )
            .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (absolute path or relative to cwd). Defaults to cwd."
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Max recursion depth: 1 = immediate children only, 0 = unlimited. Default: 1."
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden entries (starting with '.'). Default: false."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter by filename, e.g. '*.rs' or '*.{rs,toml}'."
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum number of entries to return. Default: 300."
                    }
                },
                "required": []
            }),
            requires_approval: false,
            supports_parallel_calls: true,
        }]
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        // Detect the _raw_arguments sentinel set by model_gateway when it could not
        // parse the model's argument string. Returning an error here gives the model
        // clear feedback so it can reformulate, instead of silently falling back to
        // cwd and causing an infinite retry loop.
        if call.arguments.get("_raw_arguments").is_some() {
            let raw = call.arguments["_raw_arguments"].to_string();
            return Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({ "error": "invalid_arguments", "raw": raw }),
                error_message: Some(format!(
                    "tool_invalid_arguments: list_dir received unparseable arguments: {raw}. \
                     Provide a JSON object with a \"path\" string field, e.g. {{\"path\": \"src/llm\"}}"
                )),
            });
        }
        let root = call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(&context.cwd)
            .to_string();
        let depth = call
            .arguments
            .get("depth")
            .and_then(Value::as_u64)
            .unwrap_or(1) as usize;
        let include_hidden = call
            .arguments
            .get("include_hidden")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let pattern: Option<GlobPattern> = call
            .arguments
            .get("pattern")
            .and_then(Value::as_str)
            .and_then(|s| GlobPattern::new(s).ok());
        let max_entries = call
            .arguments
            .get("max_entries")
            .and_then(Value::as_u64)
            .unwrap_or(300) as usize;

        let max_depth = if depth == 0 { usize::MAX } else { depth };
        let mut entries: Vec<Value> = Vec::new();

        let walker = WalkDir::new(&root)
            .max_depth(max_depth)
            .follow_links(false)
            .sort_by_file_name();

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if entries.len() >= max_entries {
                break;
            }
            let path = entry.path();

            // Compute the relative path string used for hidden-file checks
            let relative = path
                .strip_prefix(&root)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());

            if relative.is_empty() {
                continue; // skip the root itself
            }

            // Skip hidden entries unless explicitly requested
            if !include_hidden {
                let is_hidden = relative
                    .split(std::path::MAIN_SEPARATOR)
                    .any(|seg| seg.starts_with('.'));
                if is_hidden {
                    continue;
                }
            }

            // Apply glob pattern filter against the file name only
            if let Some(ref pat) = pattern {
                if let Some(name) = path.file_name() {
                    if !pat.matches(&name.to_string_lossy()) {
                        continue;
                    }
                }
            }

            let is_dir = entry.file_type().is_dir();
            let size = entry.metadata().ok().map(|m| m.len());

            entries.push(json!({
                "path": relative,
                "type": if is_dir { "dir" } else { "file" },
                "size_bytes": size,
            }));
        }

        let truncated = entries.len() >= max_entries;
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: true,
            output: json!({
                "root": root,
                "entries": entries,
                "count": entries.len(),
                "truncated": truncated,
            }),
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
        Self {
            sandbox,
            permissions,
        }
    }
}

#[async_trait]
impl ToolProvider for FileToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "read_file".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Read a UTF-8 file. Use `patch_file` to edit specific lines instead of rewriting with `write_file`.".into(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer","description":"1-based start line (default: 1)"},"limit":{"type":"integer","description":"Max lines to return (default: all)"}},"required":["path"]}),
                requires_approval: false,
                supports_parallel_calls: true,
            },
            ToolDefinition {
                name: "write_file".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Write or create a UTF-8 file (replaces entire content). For editing specific lines in an existing file, prefer `patch_file`.".into(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
                requires_approval: true,
                supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "patch_file".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: concat!(
                    "Replace a range of lines in an existing file. ",
                    "Much lighter than write_file — you only provide the replacement lines, not the whole file. ",
                    "Use after read_file to surgically edit specific lines. ",
                    "Lines are 1-based. The range [start_line, end_line] is inclusive and replaced by `content`."
                ).into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File to patch (absolute or relative to cwd)" },
                        "start_line": { "type": "integer", "description": "First line to replace (1-based, inclusive)" },
                        "end_line": { "type": "integer", "description": "Last line to replace (1-based, inclusive)" },
                        "content": { "type": "string", "description": "Replacement text for the specified line range" }
                    },
                    "required": ["path", "start_line", "end_line", "content"]
                }),
                requires_approval: true,
                supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "create_directory".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Create a directory.".into(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"]}),
                requires_approval: true,
                supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "remove_path".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Remove a file or directory.".into(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"},"force":{"type":"boolean"}},"required":["path"]}),
                requires_approval: true,
                supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "copy_path".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Copy a file.".into(),
                input_schema: json!({"type":"object","properties":{"source":{"type":"string"},"target":{"type":"string"},"recursive":{"type":"boolean"}},"required":["source","target"]}),
                requires_approval: true,
                supports_parallel_calls: false,
            },
        ]
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        let path = |key: &str| {
            call.arguments
                .get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("tool_invalid_arguments: missing {key}"))
        };
        let write_action = matches!(
            call.tool_name.as_str(),
            "write_file" | "patch_file" | "create_directory" | "remove_path" | "copy_path"
        );
        let action = ActionDescriptor {
            action_type: call.tool_name.clone(),
            command: None,
            paths: match call.tool_name.as_str() {
                "copy_path" => vec![path("source")?.into(), path("target")?.into()],
                _ => vec![path("path")?.into()],
            },
            domains: Vec::new(),
            category: if write_action {
                ApprovalCategory::FileChange
            } else {
                ApprovalCategory::Other
            },
        };
        let sandbox = self
            .permissions
            .build_sandbox_request(&action, &context.permission_profile);
        let output = match call.tool_name.as_str() {
            "read_file" => {
                let file_path = path("path")?;
                let raw = self.sandbox.read_file(file_path, &sandbox).await?;
                let full = String::from_utf8_lossy(&raw).into_owned();
                let lines: Vec<&str> = full.lines().collect();
                let total_lines = lines.len();
                let offset = call
                    .arguments
                    .get("offset")
                    .and_then(Value::as_u64)
                    .map(|v| (v as usize).max(1))
                    .unwrap_or(1);
                let limit = call
                    .arguments
                    .get("limit")
                    .and_then(Value::as_u64)
                    .map(|v| v as usize);
                let start = (offset - 1).min(total_lines);
                let end = limit
                    .map(|l| (start + l).min(total_lines))
                    .unwrap_or(total_lines);
                let selected: String = lines[start..end]
                    .iter()
                    .map(|line| (*line).to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                json!({
                    "path": file_path,
                    "total_lines": total_lines,
                    "showing": { "from": start + 1, "to": end },
                    "content": selected,
                })
            }
            "write_file" => {
                let file_path = path("path")?;
                let content = call
                    .arguments
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("tool_invalid_arguments: missing content"))?;

                // Read old content (if any) so we can produce a diff.
                let old_content = self
                    .sandbox
                    .read_file(file_path, &sandbox)
                    .await
                    .ok()
                    .map(|b| String::from_utf8_lossy(&b).into_owned());
                let is_new_file = old_content.is_none();

                self.sandbox
                    .write_file(file_path, content.as_bytes(), &sandbox)
                    .await?;

                let (diff_text, lines_added, lines_removed) =
                    make_unified_diff(file_path, old_content.as_deref(), content);

                json!({
                    "ok": true,
                    "path": file_path,
                    "is_new_file": is_new_file,
                    "lines_added": lines_added,
                    "lines_removed": lines_removed,
                    "diff": diff_text,
                })
            }
            "patch_file" => {
                let file_path = path("path")?;
                let start_line = call
                    .arguments
                    .get("start_line")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| anyhow!("tool_invalid_arguments: missing start_line"))?
                    as usize;
                let end_line = call
                    .arguments
                    .get("end_line")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| anyhow!("tool_invalid_arguments: missing end_line"))?
                    as usize;
                let replacement = call
                    .arguments
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("tool_invalid_arguments: missing content"))?;

                if start_line == 0 || end_line == 0 || start_line > end_line {
                    return Ok(ToolResult {
                        tool_call_id: call.tool_call_id.clone(),
                        ok: false,
                        output: json!({ "error": format!(
                            "invalid line range: start_line={start_line}, end_line={end_line}. \
                             Lines are 1-based and start_line must be <= end_line."
                        )}),
                        error_message: Some("invalid line range".into()),
                    });
                }

                let old_bytes = self.sandbox.read_file(file_path, &sandbox).await?;
                let old_content = String::from_utf8_lossy(&old_bytes).into_owned();
                let lines: Vec<&str> = old_content.lines().collect();

                if start_line > lines.len() {
                    return Ok(ToolResult {
                        tool_call_id: call.tool_call_id.clone(),
                        ok: false,
                        output: json!({ "error": format!(
                            "start_line {start_line} exceeds file length ({} lines)",
                            lines.len()
                        )}),
                        error_message: Some("start_line out of range".into()),
                    });
                }
                let clamped_end = end_line.min(lines.len());

                // Build the new content: prefix + replacement + suffix
                let mut new_content = String::new();
                for line in &lines[..start_line - 1] {
                    new_content.push_str(line);
                    new_content.push('\n');
                }
                new_content.push_str(replacement);
                if !replacement.ends_with('\n') {
                    new_content.push('\n');
                }
                for line in &lines[clamped_end..] {
                    new_content.push_str(line);
                    new_content.push('\n');
                }
                // Trim trailing newline if original didn't end with one
                if !old_content.ends_with('\n') && new_content.ends_with('\n') {
                    new_content.pop();
                }

                self.sandbox
                    .write_file(file_path, new_content.as_bytes(), &sandbox)
                    .await?;

                let (diff_text, lines_added, lines_removed) =
                    make_unified_diff(file_path, Some(&old_content), &new_content);

                json!({
                    "ok": true,
                    "path": file_path,
                    "patched_range": { "start_line": start_line, "end_line": clamped_end },
                    "lines_added": lines_added,
                    "lines_removed": lines_removed,
                    "diff": diff_text,
                })
            }
            "create_directory" => {
                let recursive = call
                    .arguments
                    .get("recursive")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                self.sandbox
                    .create_directory(path("path")?, recursive, &sandbox)
                    .await?;
                json!({ "ok": true, "path": path("path")? })
            }
            "remove_path" => {
                let recursive = call
                    .arguments
                    .get("recursive")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let force = call
                    .arguments
                    .get("force")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.sandbox
                    .remove_path(path("path")?, recursive, force, &sandbox)
                    .await?;
                json!({ "ok": true, "path": path("path")? })
            }
            "copy_path" => {
                let recursive = call
                    .arguments
                    .get("recursive")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.sandbox
                    .copy_path(path("source")?, path("target")?, recursive, &sandbox)
                    .await?;
                json!({ "ok": true, "source": path("source")?, "target": path("target")? })
            }
            other => bail!("tool_not_found: {other}"),
        };
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: true,
            output,
            error_message: None,
        })
    }
}

// ─── SearchToolProvider ───────────────────────────────────────────────────────

pub struct SearchToolProvider;

#[async_trait]
impl ToolProvider for SearchToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "grep_search".into(),
            provider_kind: ToolProviderKind::Builtin,
            description: "Search files for a text pattern (ripgrep if available, native fallback)."
                .into(),
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
            requires_approval: false,
            supports_parallel_calls: true,
        }]
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        let pattern = call
            .arguments
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool_invalid_arguments: missing pattern"))?;
        let search_path = call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(&context.cwd);
        let literal = call
            .arguments
            .get("literal")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let max_results = call
            .arguments
            .get("max_results")
            .and_then(Value::as_u64)
            .unwrap_or(50) as usize;

        // Try ripgrep first (faster, respects .gitignore)
        let rg_result = if literal {
            Command::new("rg")
                .args(["-n", "--hidden", "--no-heading", "-F", pattern, search_path])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
                .ok()
        } else {
            Command::new("rg")
                .args(["-n", "--hidden", "--no-heading", pattern, search_path])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
                .ok()
        };

        if let Some(out) = rg_result.filter(|o| o.status.success() || o.status.code() == Some(1)) {
            let stdout = String::from_utf8_lossy(&out.stdout);
            // rg -n --no-heading emits: "<file>:<lineno>:<content>"
            // Parse into structured objects so callers get file/line/content
            // as separate fields — avoids embedding line numbers in the match
            // string which makes TUI copy/paste awkward.
            let matches: Vec<Value> = stdout
                .lines()
                .take(max_results)
                .map(|raw| {
                    // Split on the first two ':' separators
                    let mut parts = raw.splitn(3, ':');
                    match (parts.next(), parts.next(), parts.next()) {
                        (Some(file), Some(lineno), Some(content)) => {
                            let line_num: Option<u64> = lineno.parse().ok();
                            json!({ "file": file, "line": line_num, "content": content })
                        }
                        _ => json!({ "file": "", "line": null, "content": raw }),
                    }
                })
                .collect();
            return Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: true,
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
                        matches.push(json!({
                            "file": entry.path().display().to_string(),
                            "line": lineno + 1,
                            "content": line,
                        }));
                        if matches.len() >= max_results {
                            break 'walk;
                        }
                    }
                }
            }
        }
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: true,
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
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "web_fetch".into(),
            provider_kind: ToolProviderKind::Builtin,
            description: "Fetch a URL and return its content as plain text (HTML is converted)."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The URL to fetch" },
                    "max_chars": { "type": "integer", "description": "Maximum characters to return (default 8000)" }
                },
                "required": ["url"]
            }),
            requires_approval: false,
            supports_parallel_calls: true,
        }]
    }

    async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
        let url = call
            .arguments
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool_invalid_arguments: missing url"))?;
        let max_chars = call
            .arguments
            .get("max_chars")
            .and_then(Value::as_u64)
            .unwrap_or(8000) as usize;

        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| anyhow!("web_fetch_error: {e}"))?;
        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| anyhow!("web_fetch_read_error: {e}"))?;

        let text = if content_type.contains("text/html")
            || url.ends_with(".html")
            || url.ends_with(".htm")
        {
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
            error_message: if status >= 400 {
                Some(format!("HTTP {status}"))
            } else {
                None
            },
        })
    }
}

// ─── ImageToolProvider ────────────────────────────────────────────────────────

pub struct ImageToolProvider;

#[async_trait]
impl ToolProvider for ImageToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "image_metadata".into(),
            provider_kind: ToolProviderKind::Builtin,
            description: "Return local image metadata.".into(),
            input_schema: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
            requires_approval: false,
            supports_parallel_calls: true,
        }]
    }

    async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
        let path = call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool_invalid_arguments: missing path"))?;
        let meta = tokio::fs::metadata(path).await?;
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: true,
            output: json!({ "path": path, "size": meta.len() }),
            error_message: None,
        })
    }
}

// ─── Placeholder stub (kept for public API compat) ────────────────────────────
// The *real* SpawnAgentToolProvider is `SpawnAgentToolProviderReal` defined in
// runtime.rs. It is late-registered after `ConversationRuntime` is constructed
// so it can hold a runtime clone. This stub is no longer registered anywhere.

pub struct SpawnAgentToolProvider;

#[async_trait]
impl ToolProvider for SpawnAgentToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }
    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        Vec::new() // deliberately empty — the real provider is registered in runtime.rs
    }
    async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: false,
            output: json!({"message":"spawn_agent: reached stub — real provider should be registered"}),
            error_message: Some("spawn_agent: stub reached unexpectedly".into()),
        })
    }
}

pub struct RequestUserInputToolProvider;

#[async_trait]
impl ToolProvider for RequestUserInputToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }
    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        Vec::new()
    }
    async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: false,
            output: json!({"message":"request_user_input is not available for model-driven calls"}),
            error_message: Some(
                "request_user_input is not available for model-driven calls".into(),
            ),
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
        self.registry
            .read()
            .unwrap()
            .iter()
            .flat_map(|p| p.list_tools(context))
            .collect()
    }

    /// Returns `true` if the named tool is registered and has `supports_parallel_calls = true`.
    /// Unknown tools default to `false` (safest assumption).
    pub fn is_parallel_safe(&self, tool_name: &str, context: &ToolListingContext) -> bool {
        self.registry
            .read()
            .unwrap()
            .iter()
            .flat_map(|p| p.list_tools(context))
            .find(|def| def.name == tool_name)
            .map(|def| def.supports_parallel_calls)
            .unwrap_or(false)
    }

    pub async fn execute(
        &self,
        call: &ToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult> {
        self.running_tool_calls
            .write()
            .await
            .insert(call.tool_call_id.clone(), context.clone());
        let providers = self.registry.read().unwrap().clone();
        for provider in providers {
            if provider
                .list_tools(&ToolListingContext {
                    thread_id: context.thread_id.clone(),
                    cwd: context.cwd.clone(),
                    features: HashMap::new(),
                })
                .iter()
                .any(|t| t.name == call.tool_name)
            {
                let result = provider.execute(call, &context).await;
                self.running_tool_calls
                    .write()
                    .await
                    .remove(&call.tool_call_id);
                return result;
            }
        }
        self.running_tool_calls
            .write()
            .await
            .remove(&call.tool_call_id);
        bail!("tool_not_found: {}", call.tool_name)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Recover shell commands from a double-encoded command string.
///
/// Some models write the entire tool-call JSON object as the value of the `command` field, e.g.:
///   `{"command":"git status"}{"command":"git diff --stat"}`
/// This function extracts all `"command"` string values from any embedded JSON objects and joins
/// them with ` && ` so the recovered string can be run as a single bash command.
fn extract_commands_from_embedded_json(s: &str) -> String {
    let mut result = Vec::new();
    let mut rest = s.trim();
    while !rest.is_empty() {
        // Find the next '{' and try to parse a JSON object from that position
        let start = match rest.find('{') {
            Some(i) => i,
            None => break,
        };
        rest = &rest[start..];
        // Try increasingly long slices until we get a valid JSON object
        let mut parsed = false;
        for end in (1..=rest.len()).rev() {
            if let Ok(v) = serde_json::from_str::<Value>(&rest[..end]) {
                if let Some(cmd) = v.get("command").and_then(Value::as_str) {
                    if !cmd.is_empty() {
                        result.push(cmd.to_string());
                    }
                }
                rest = &rest[end..];
                parsed = true;
                break;
            }
        }
        if !parsed {
            break;
        }
        rest = rest.trim_start();
    }
    result.join(" && ")
}

/// Minimal shell-lite tokenizer for the `shell_exec` string-argv fallback.
/// Splits on whitespace while respecting single and double quotes.
/// Does NOT perform any shell expansion — no globs, no variables, no redirects.
fn simple_tokenize(s: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    for ch in s.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_double => escaped = true,
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            ' ' | '\t' | '\n' if !in_single && !in_double => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

// ─── Diff helper ─────────────────────────────────────────────────────────────

/// Build a unified diff string between `old` and `new` content for `path`.
/// Returns `(diff_text, lines_added, lines_removed)`.
/// If `old` is `None` the entire new content is shown as pure additions.
fn make_unified_diff(path: &str, old: Option<&str>, new: &str) -> (String, usize, usize) {
    let old_str = old.unwrap_or("");
    let header_old = if old.is_none() {
        "/dev/null".to_string()
    } else {
        format!("a/{path}")
    };
    let header_new = format!("b/{path}");

    let diff = TextDiff::from_lines(old_str, new);
    let mut lines_added = 0usize;
    let mut lines_removed = 0usize;
    let mut out = format!("--- {header_old}\n+++ {header_new}\n");

    for group in diff.grouped_ops(3) {
        let first = &group[0];
        let last = &group[group.len() - 1];
        let old_start = first.old_range().start;
        let old_end = last.old_range().end;
        let new_start = first.new_range().start;
        let new_end = last.new_range().end;
        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            old_start + 1,
            old_end - old_start,
            new_start + 1,
            new_end - new_start,
        ));
        for op in &group {
            for change in diff.iter_changes(op) {
                let prefix = match change.tag() {
                    ChangeTag::Delete => {
                        lines_removed += 1;
                        "-"
                    }
                    ChangeTag::Insert => {
                        lines_added += 1;
                        "+"
                    }
                    ChangeTag::Equal => " ",
                };
                out.push_str(prefix);
                out.push_str(change.value());
                if !change.value().ends_with('\n') {
                    out.push('\n');
                }
            }
        }
    }

    (out, lines_added, lines_removed)
}
