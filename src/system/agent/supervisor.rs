//! Sub-agent supervision: spawning, awaiting, and cancelling child turns.
//!
//! This module owns the `spawn_agent` tool provider and the bookkeeping that
//! turns "the model called spawn_agent" into "a fresh ephemeral thread runs
//! to completion and returns its final answer text". It used to live at the
//! bottom of `runtime.rs` as part of the runtime god-object; pulling it out
//! lets the runtime be a thinner facade and gives the supervisor a clear
//! boundary for future hardening (cancellation policies, fan-out limits,
//! observability hooks).

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use uuid::Uuid;

use crate::system::agent::tools::ToolProvider;
use crate::system::agent::EventFilter;
use crate::system::domain::{
    ApprovalPolicy, ConversationItem, ItemKind, PermissionProfile, RuntimeEventKind, SurfaceKind,
    ThreadId, ToolCall, ToolDefinition, ToolExecutionContext, ToolListingContext, ToolProviderKind,
    ToolResult, TurnId, TurnStatus, UserInput,
};
use crate::system::runtime::{
    ConversationRuntime, ThreadStartParams, TurnInterruptParams, TurnStartParams,
};

const SPAWN_AGENT_DIRECTORY_INVENTORY_ERROR: &str = "deterministic_directory_inventory";

/// Outcome of awaiting a sub-agent turn.
pub enum TurnCompletionOutcome {
    Completed,
    Failed(String),
    Interrupted,
    TimedOut,
}

struct ChildTurnWait {
    outcome: TurnCompletionOutcome,
    streamed_text: String,
}

struct DirectoryInventoryRedirect {
    path: Option<String>,
    depth: Option<u64>,
}

#[derive(Clone)]
pub(crate) struct AgentSupervisor {
    runtime: ConversationRuntime,
    child_slots: Arc<Semaphore>,
}

pub(crate) struct ChildAgentRequest {
    pub prompt: String,
    pub cwd: String,
    pub approval_policy: ApprovalPolicy,
    pub permission_profile: PermissionProfile,
    pub timeout_secs: u64,
    pub agent_depth: u32,
    /// Parent thread the spawn_agent tool call originated from.
    pub parent_thread_id: ThreadId,
    /// Parent turn the spawn_agent tool call originated from.
    pub parent_turn_id: TurnId,
    /// The spawn_agent tool call that produced this child. Used to tie the
    /// child's lifecycle back to the parent's transcript entry.
    pub parent_tool_call_id: String,
}

pub(crate) struct ChildAgentRun {
    pub child_thread_id: ThreadId,
    pub child_turn_id: TurnId,
    pub result_text: String,
    pub outcome: TurnCompletionOutcome,
}

impl AgentSupervisor {
    pub fn new(runtime: ConversationRuntime, max_concurrent_children: usize) -> Self {
        Self {
            runtime,
            child_slots: Arc::new(Semaphore::new(max_concurrent_children.max(1))),
        }
    }

    pub async fn run_child(&self, request: ChildAgentRequest) -> Result<ChildAgentRun> {
        let _slot = self
            .child_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| anyhow!("child_agent_slots_closed: {e}"))?;

        let child = self
            .runtime
            .start_thread(ThreadStartParams {
                cwd: Some(request.cwd),
                model_settings: None,
                approval_policy: Some(request.approval_policy.clone()),
                permission_profile: Some(request.permission_profile.clone()),
                ephemeral: true,
            })
            .await?;
        let child_thread_id = child.metadata.thread_id;

        let child_prompt = child_agent_prompt(&request.prompt, request.timeout_secs);
        let child_turn = self
            .runtime
            .start_turn(
                TurnStartParams {
                    thread_id: child_thread_id.clone(),
                    input: vec![UserInput::from_text(&child_prompt)],
                    cwd: None,
                    model_settings: None,
                    approval_policy: Some(request.approval_policy),
                    permission_profile: Some(request.permission_profile),
                    output_schema: None,
                    agent_depth: request.agent_depth + 1,
                },
                SurfaceKind::Exec,
            )
            .await?;
        let child_turn_id = child_turn.turn.turn_id;

        // Phase 7: announce the parent → child relationship so consumers
        // (TUI activity tree, benchmarks) can subscribe to the child's events
        // and tie them back to the originating spawn_agent tool call.
        let label = request
            .prompt
            .lines()
            .next()
            .unwrap_or(&request.prompt)
            .chars()
            .take(80)
            .collect::<String>();
        let _ = self
            .runtime
            .publish_event(
                RuntimeEventKind::ChildAgentSpawned,
                Some(request.parent_thread_id.clone()),
                Some(request.parent_turn_id.clone()),
                serde_json::json!({
                    "parentThreadId": request.parent_thread_id,
                    "parentTurnId": request.parent_turn_id,
                    "parentToolCallId": request.parent_tool_call_id,
                    "childThreadId": child_thread_id,
                    "childTurnId": child_turn_id,
                    "label": label,
                }),
            )
            .await;

        let wait = self
            .await_child_turn_completion(&child_thread_id, &child_turn_id, request.timeout_secs)
            .await?;
        let mut outcome = wait.outcome;
        let mut streamed_text = wait.streamed_text;

        if matches!(&outcome, TurnCompletionOutcome::TimedOut) {
            if let Ok(cancel_wait) = self
                .cancel_child_turn(&child_thread_id, &child_turn_id, 5)
                .await
            {
                if !cancel_wait.streamed_text.trim().is_empty() {
                    if !streamed_text.trim().is_empty() {
                        streamed_text.push('\n');
                    }
                    streamed_text.push_str(cancel_wait.streamed_text.trim());
                }
            }
            outcome = TurnCompletionOutcome::TimedOut;
        }

        let result_text = if let Some(text) = self.read_last_agent_message(&child_thread_id)? {
            visible_child_result_text(&text).unwrap_or_else(|| text.trim().to_string())
        } else if !streamed_text.trim().is_empty() {
            visible_child_result_text(&streamed_text).unwrap_or_else(|| match &outcome {
                TurnCompletionOutcome::TimedOut => {
                    "[sub-agent timed out before producing final output]".into()
                }
                _ => streamed_text.trim().to_string(),
            })
        } else {
            match &outcome {
                TurnCompletionOutcome::TimedOut => {
                    "[sub-agent timed out before producing final output]".into()
                }
                TurnCompletionOutcome::Interrupted => "[sub-agent was interrupted]".into(),
                TurnCompletionOutcome::Failed(reason) => {
                    format!("[sub-agent failed before producing output: {reason}]")
                }
                TurnCompletionOutcome::Completed => "[sub-agent produced no output]".into(),
            }
        };

        if matches!(&outcome, TurnCompletionOutcome::Completed) {
            if let Err(e) = self.runtime.delete_thread(&child_thread_id).await {
                tracing::warn!(
                    thread_id = %child_thread_id,
                    "failed to delete successful ephemeral sub-agent thread: {e}"
                );
            }
        }

        Ok(ChildAgentRun {
            child_thread_id,
            child_turn_id,
            result_text,
            outcome,
        })
    }

    /// Subscribe to the event bus and block until a specific child turn
    /// reaches a terminal state. The persisted turn status is authoritative;
    /// events only wake this waiter so fast terminal transitions cannot be
    /// missed.
    async fn await_child_turn_completion(
        &self,
        thread_id: &str,
        turn_id: &str,
        timeout_secs: u64,
    ) -> Result<ChildTurnWait> {
        if let Some(outcome) = self.child_turn_terminal_outcome(thread_id, turn_id)? {
            return Ok(ChildTurnWait {
                outcome,
                streamed_text: String::new(),
            });
        }

        let subscriber_id = format!("spawn_agent_{}", Uuid::new_v4().simple());
        let mut sub = self.runtime.event_bus().subscribe(
            subscriber_id.clone(),
            EventFilter {
                thread_id: Some(thread_id.to_string()),
            },
        );
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
        let mut streamed_text = String::new();

        let outcome = loop {
            if let Some(outcome) = self.child_turn_terminal_outcome(thread_id, turn_id)? {
                break outcome;
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break TurnCompletionOutcome::TimedOut;
            }
            match tokio::time::timeout(remaining, sub.receiver.recv()).await {
                Ok(Some(event)) => {
                    // Bus filters by thread_id at publish time, but the sub-agent
                    // sends turn-level events tagged with the *child* turn — we
                    // still want to spin until that specific turn finishes.
                    match event.kind {
                        RuntimeEventKind::ItemUpdated => {
                            if event.turn_id.as_deref() != Some(turn_id) {
                                continue;
                            }
                            let item_id = event
                                .payload
                                .get("itemId")
                                .and_then(Value::as_str)
                                .unwrap_or_default();
                            if item_id.starts_with("reasoning_") {
                                continue;
                            }
                            if let Some(delta) = event.payload.get("delta").and_then(Value::as_str)
                            {
                                streamed_text.push_str(delta);
                            }
                        }
                        RuntimeEventKind::TurnCompleted | RuntimeEventKind::TurnFailed => {
                            if event.turn_id.as_deref() == Some(turn_id) {
                                continue;
                            }
                        }
                        _ => continue,
                    }
                }
                Ok(None) => break TurnCompletionOutcome::Failed("event_bus_closed".into()),
                Err(_) => break TurnCompletionOutcome::TimedOut,
            }
        };

        self.runtime.event_bus().unsubscribe(&subscriber_id);
        Ok(ChildTurnWait {
            outcome,
            streamed_text,
        })
    }

    fn child_turn_terminal_outcome(
        &self,
        thread_id: &str,
        turn_id: &str,
    ) -> Result<Option<TurnCompletionOutcome>> {
        let persisted = self
            .runtime
            .inner
            .persistence_manager
            .read_thread(thread_id)?;
        let Some(turn) = persisted.turns.iter().find(|turn| turn.turn_id == turn_id) else {
            return Ok(Some(TurnCompletionOutcome::Failed(format!(
                "child turn not found: {turn_id}"
            ))));
        };

        let outcome = match turn.status {
            TurnStatus::Completed => Some(TurnCompletionOutcome::Completed),
            TurnStatus::Failed => Some(TurnCompletionOutcome::Failed(
                last_error_message(&persisted.items, turn_id).unwrap_or_else(|| "failed".into()),
            )),
            TurnStatus::Interrupted => Some(TurnCompletionOutcome::Interrupted),
            TurnStatus::Created | TurnStatus::Running | TurnStatus::WaitingForApproval => None,
        };
        Ok(outcome)
    }

    async fn cancel_child_turn(
        &self,
        thread_id: &str,
        turn_id: &str,
        grace_secs: u64,
    ) -> Result<ChildTurnWait> {
        let _ = self
            .runtime
            .interrupt_turn(TurnInterruptParams {
                thread_id: thread_id.into(),
                turn_id: turn_id.into(),
            })
            .await?;

        self.await_child_turn_completion(thread_id, turn_id, grace_secs)
            .await
            .or_else(|_| {
                Ok(ChildTurnWait {
                    outcome: TurnCompletionOutcome::TimedOut,
                    streamed_text: String::new(),
                })
            })
    }

    /// Read the last `AgentMessage` item from a thread's persisted history.
    fn read_last_agent_message(&self, thread_id: &str) -> Result<Option<String>> {
        let persisted = self
            .runtime
            .inner
            .persistence_manager
            .read_thread(thread_id)?;
        let text = persisted
            .items
            .iter()
            .rev()
            .find(|i| i.kind == ItemKind::AgentMessage)
            .and_then(|i| i.payload.get("text").and_then(Value::as_str))
            .map(|s| s.to_string());
        Ok(text)
    }
}

fn last_error_message(items: &[ConversationItem], turn_id: &str) -> Option<String> {
    items
        .iter()
        .rev()
        .find(|item| item.turn_id == turn_id && item.kind == ItemKind::Error)
        .and_then(|item| {
            item.payload
                .get("message")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
}

fn child_agent_prompt(prompt: &str, timeout_secs: u64) -> String {
    format!(
        "You are running as a bounded sub-agent. Complete the task and return a concise final answer within {timeout_secs} seconds.\n\
         Prefer targeted reads and searches over exhaustive exploration. If a tool result is truncated or the task is too large to finish fully, summarize the evidence you have and state what remains incomplete instead of trying to exhaust every remaining item.\n\n\
         Task:\n{prompt}"
    )
}

fn visible_child_result_text(text: &str) -> Option<String> {
    let stripped = strip_think_sections(text).trim().to_string();
    if stripped.is_empty() {
        None
    } else {
        Some(stripped)
    }
}

fn strip_think_sections(text: &str) -> String {
    let mut remaining = text;
    let mut out = String::new();

    if !remaining.contains("<think>") {
        if let Some((_, after)) = remaining.rsplit_once("</think>") {
            return after.to_string();
        }
        return remaining.to_string();
    }

    while let Some(start) = remaining.find("<think>") {
        out.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];
        if let Some(end) = after_start.find("</think>") {
            remaining = &after_start[end + "</think>".len()..];
        } else {
            remaining = "";
            break;
        }
    }
    out.push_str(remaining);
    out
}

// ─── SpawnAgentToolProviderReal ───────────────────────────────────────────────
//
// Registered *after* ConversationRuntime is constructed so it can hold a
// runtime clone without creating a circular dependency.

pub(crate) struct SpawnAgentToolProviderReal {
    supervisor: AgentSupervisor,
}

impl SpawnAgentToolProviderReal {
    pub fn new(supervisor: AgentSupervisor) -> Self {
        Self { supervisor }
    }
}

#[async_trait]
impl ToolProvider for SpawnAgentToolProviderReal {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        let agent_cfg = &self.supervisor.runtime.inner.effective_config.agent;
        vec![ToolDefinition {
            name: "spawn_agent".into(),
            description: format!(
                "Spawn an independent sub-agent for a bounded task. \
                The sub-agent runs with full tool access and returns its final answer as text. \
                Use this for a small number of independent sub-tasks, such as analysing specific files or modules. \
                At most {} child agents run concurrently; extra calls queue behind that limit. \
                Avoid exhaustive directory inventory tasks unless the child can summarize partial/truncated evidence and finish promptly.",
                agent_cfg.max_child_agents
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task description for the sub-agent. Be specific and self-contained."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": format!(
                            "Maximum seconds to wait for the sub-agent (default {}, max {}). Values below the default are raised to the default.",
                            agent_cfg.child_timeout_secs,
                            agent_cfg.max_child_timeout_secs
                        )
                    }
                },
                "required": ["prompt"]
            }),
            requires_approval: false,
            supports_parallel_calls: true,
            provider_kind: ToolProviderKind::Builtin,
        }]
    }

    async fn execute(&self, call: &ToolCall, ctx: &ToolExecutionContext) -> Result<ToolResult> {
        let agent_cfg = &self.supervisor.runtime.inner.effective_config.agent;

        // Depth guard: prevent unbounded recursive agent spawning.
        if ctx.agent_depth >= agent_cfg.max_spawn_depth {
            return Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({ "error": "max sub-agent depth reached" }),
                error_message: Some(format!(
                    "spawn_agent cannot be nested more than {} levels deep",
                    agent_cfg.max_spawn_depth
                )),
            });
        }

        let prompt = call
            .arguments
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("spawn_agent: prompt is required"))?
            .to_string();

        if let Some(redirect) = directory_inventory_redirect(&prompt) {
            let mut suggested_arguments = serde_json::Map::new();
            if let Some(path) = redirect.path {
                suggested_arguments.insert("path".into(), Value::String(path));
            }
            if let Some(depth) = redirect.depth {
                suggested_arguments.insert("depth".into(), Value::from(depth));
            } else {
                suggested_arguments.insert("depth".into(), Value::from(3));
            }
            suggested_arguments.insert("include_hidden".into(), Value::Bool(false));
            suggested_arguments.insert("max_entries".into(), Value::from(300));

            return Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({
                    "status": "not_spawned",
                    "error": SPAWN_AGENT_DIRECTORY_INVENTORY_ERROR,
                    "reason": "Directory inventory is deterministic and should use direct file tools, not a model sub-agent.",
                    "suggested_tool": "list_dir",
                    "suggested_arguments": Value::Object(suggested_arguments),
                    "next_step": "Call list_dir directly, then read selected key files if the user asked for a summary.",
                }),
                error_message: Some(
                    "spawn_agent skipped: use list_dir directly for directory inventory".into(),
                ),
            });
        }

        let timeout_secs = call
            .arguments
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(agent_cfg.child_timeout_secs)
            .min(agent_cfg.max_child_timeout_secs)
            .max(agent_cfg.child_timeout_secs);

        let run = self
            .supervisor
            .run_child(ChildAgentRequest {
                prompt,
                cwd: ctx.cwd.clone(),
                approval_policy: ctx.approval_policy.clone(),
                permission_profile: ctx.permission_profile.clone(),
                timeout_secs,
                agent_depth: ctx.agent_depth,
                parent_thread_id: ctx.thread_id.clone(),
                parent_turn_id: ctx.turn_id.clone(),
                parent_tool_call_id: call.tool_call_id.clone(),
            })
            .await?;

        match run.outcome {
            TurnCompletionOutcome::Completed => Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: true,
                output: json!({
                    "thread_id": run.child_thread_id,
                    "turn_id": run.child_turn_id,
                    "result": run.result_text,
                }),
                error_message: None,
            }),
            TurnCompletionOutcome::Failed(reason) => Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({
                    "thread_id": run.child_thread_id,
                    "turn_id": run.child_turn_id,
                    "result": run.result_text,
                    "error": reason,
                }),
                error_message: Some(format!("sub-agent failed: {reason}")),
            }),
            TurnCompletionOutcome::Interrupted => Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({
                    "thread_id": run.child_thread_id,
                    "turn_id": run.child_turn_id,
                    "result": run.result_text,
                    "error": "interrupted",
                }),
                error_message: Some("sub-agent interrupted".into()),
            }),
            TurnCompletionOutcome::TimedOut => Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({
                    "thread_id": run.child_thread_id,
                    "turn_id": run.child_turn_id,
                    "result": run.result_text,
                    "error": "timeout",
                }),
                error_message: Some(format!("sub-agent timed out after {timeout_secs}s")),
            }),
        }
    }
}

fn directory_inventory_redirect(prompt: &str) -> Option<DirectoryInventoryRedirect> {
    let lower = prompt.to_ascii_lowercase();
    let asks_for_analysis = lower.contains("review")
        || lower.contains("code reviewer")
        || lower.contains("analyze")
        || lower.contains("analyse")
        || lower.contains("audit")
        || lower.contains("find bugs")
        || lower.contains("bug")
        || lower.contains("risk")
        || lower.contains("regression")
        || lower.contains("security");
    if asks_for_analysis {
        return None;
    }

    let asks_for_directory = lower.contains("directory") || lower.contains("directories");
    let asks_for_listing = lower.contains("list all files")
        || lower.contains("list the contents")
        || lower.contains("list files")
        || lower.contains("for each file")
        || lower.contains("note its path")
        || lower.contains("path and type");
    let broad_recursive = lower.contains("recursive")
        || lower.contains("recursively")
        || lower.contains("depth")
        || lower.contains("all files");

    if !(asks_for_directory && asks_for_listing && broad_recursive) {
        return None;
    }

    Some(DirectoryInventoryRedirect {
        path: extract_backtick_path(prompt),
        depth: extract_depth(prompt),
    })
}

fn extract_backtick_path(prompt: &str) -> Option<String> {
    prompt
        .split('`')
        .nth(1)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
}

fn extract_depth(prompt: &str) -> Option<u64> {
    let lower = prompt.to_ascii_lowercase();
    let (_, after_depth) = lower.split_once("depth")?;
    let digits: String = after_depth
        .chars()
        .skip_while(|ch| !ch.is_ascii_digit())
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_inventory_prompt_is_redirected() {
        let redirect = directory_inventory_redirect(
            "List the contents of the `bin` directory recursively (depth 3). For each file, note its path and type.",
        )
        .expect("directory inventory should redirect");

        assert_eq!(redirect.path.as_deref(), Some("bin"));
        assert_eq!(redirect.depth, Some(3));
    }

    #[test]
    fn targeted_analysis_prompt_is_allowed() {
        assert!(directory_inventory_redirect(
            "Inspect parser.rs and explain how parse errors flow."
        )
        .is_none());
    }

    #[test]
    fn recursive_code_review_prompt_is_allowed() {
        assert!(directory_inventory_redirect(
            "You are a code reviewer. Review ALL Rust source files in the directory `src/system/agent/` (recursively). For each file, note risks and missing tests.",
        )
        .is_none());
    }
}
