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

/// Outcome of awaiting a sub-agent turn.
pub enum TurnCompletionOutcome {
    Completed,
    Failed(String),
    Interrupted,
    TimedOut,
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

        let child_turn = self
            .runtime
            .start_turn(
                TurnStartParams {
                    thread_id: child_thread_id.clone(),
                    input: vec![UserInput::from_text(&request.prompt)],
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

        let mut outcome = self
            .await_child_turn_completion(&child_thread_id, &child_turn_id, request.timeout_secs)
            .await?;

        if matches!(outcome, TurnCompletionOutcome::TimedOut) {
            let _ = self
                .cancel_child_turn(&child_thread_id, &child_turn_id, 5)
                .await;
            outcome = TurnCompletionOutcome::TimedOut;
        }

        let result_text = self
            .read_last_agent_message(&child_thread_id)?
            .unwrap_or_else(|| "[sub-agent produced no output]".into());

        if matches!(outcome, TurnCompletionOutcome::Completed) {
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
    ) -> Result<TurnCompletionOutcome> {
        if let Some(outcome) = self.child_turn_terminal_outcome(thread_id, turn_id)? {
            return Ok(outcome);
        }

        let subscriber_id = format!("spawn_agent_{}", Uuid::new_v4().simple());
        let mut sub = self.runtime.event_bus().subscribe(
            subscriber_id.clone(),
            EventFilter {
                thread_id: Some(thread_id.to_string()),
            },
        );
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);

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
        Ok(outcome)
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
    ) -> Result<TurnCompletionOutcome> {
        let _ = self
            .runtime
            .interrupt_turn(TurnInterruptParams {
                thread_id: thread_id.into(),
                turn_id: turn_id.into(),
            })
            .await?;

        self.await_child_turn_completion(thread_id, turn_id, grace_secs)
            .await
            .or_else(|_| Ok(TurnCompletionOutcome::TimedOut))
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
            description: "Spawn an independent sub-agent to work on a task in parallel. \
                The sub-agent runs with full tool access and returns its final answer as text. \
                Use this to parallelise independent sub-tasks — for example, analysing multiple \
                files simultaneously. Multiple spawn_agent calls in the same response run \
                concurrently."
                .into(),
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
                            "Maximum seconds to wait for the sub-agent (default {}, max {}).",
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

        let timeout_secs = call
            .arguments
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(agent_cfg.child_timeout_secs)
            .min(agent_cfg.max_child_timeout_secs)
            .max(1);

        let run = self
            .supervisor
            .run_child(ChildAgentRequest {
                prompt,
                cwd: ctx.cwd.clone(),
                approval_policy: ctx.approval_policy.clone(),
                permission_profile: ctx.permission_profile.clone(),
                timeout_secs,
                agent_depth: ctx.agent_depth,
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
