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
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::system::agent::tools::ToolProvider;
use crate::system::agent::EventFilter;
use crate::system::domain::{
    ApprovalPolicy, ConversationItem, ItemKind, PermissionProfile, RuntimeEventKind, SandboxMode,
    SurfaceKind, ThreadId, ToolCall, ToolDefinition, ToolExecutionContext, ToolListingContext,
    ToolProviderKind, ToolResult, TurnId, TurnStatus, UserInput, STATUS_INTERRUPTED,
    STATUS_TIMEOUT,
};
use crate::system::runtime::{
    ConversationRuntime, ThreadStartParams, TurnInterruptParams, TurnStartParams,
};

const SPAWN_AGENT_DIRECTORY_INVENTORY_ERROR: &str = "deterministic_directory_inventory";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
struct TeamAssignment {
    role: String,
    objective: String,
    #[serde(default)]
    focus_paths: Vec<String>,
    #[serde(default)]
    questions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
struct TeamMemberReport {
    index: usize,
    role: String,
    objective: String,
    child_thread_id: Option<String>,
    child_turn_id: Option<String>,
    status: String,
    report: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct TeamMemberContext {
    pub team_id: String,
    pub member_index: usize,
    pub role: String,
    pub objective: String,
}

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
    /// Parent cancellation propagates into the child turn.
    pub cancel_token: CancellationToken,
    /// Whether this child may recursively use `spawn_agent`.
    pub allow_spawning: bool,
    /// Present when the child belongs to a coordinated research team.
    pub team_member: Option<TeamMemberContext>,
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
        let cancel_token = request.cancel_token.clone();
        let acquire_slot = self.child_slots.clone().acquire_owned();
        let _slot = tokio::select! {
            slot = acquire_slot => slot.map_err(|error| anyhow!("child_agent_slots_closed: {error}"))?,
            _ = cancel_token.cancelled() => return Err(anyhow!("child_agent_cancelled_before_start")),
        };

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
                    repo_map_verbosity: None,
                    agent_depth: if request.allow_spawning {
                        request.agent_depth + 1
                    } else {
                        self.runtime.inner.effective_config.agent.max_spawn_depth
                    },
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

        if let Some(member) = &request.team_member {
            let _ = self
                .runtime
                .publish_event(
                    RuntimeEventKind::AgentTeamMemberUpdated,
                    Some(request.parent_thread_id.clone()),
                    Some(request.parent_turn_id.clone()),
                    json!({
                        "teamId": member.team_id,
                        "parentToolCallId": request.parent_tool_call_id,
                        "memberIndex": member.member_index,
                        "role": member.role,
                        "objective": member.objective,
                        "status": "running",
                        "childThreadId": child_thread_id,
                        "childTurnId": child_turn_id,
                    }),
                )
                .await;
        }

        let wait = tokio::select! {
            wait = self.await_child_turn_completion(
                &child_thread_id,
                &child_turn_id,
                request.timeout_secs,
            ) => wait?,
            _ = cancel_token.cancelled() => {
                self.cancel_child_turn(&child_thread_id, &child_turn_id, 5).await?
            }
        };
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

// ─── AgentOrchestrationToolProvider ──────────────────────────────────────────
//
// Registered *after* ConversationRuntime is constructed so it can hold a
// runtime clone without creating a circular dependency.

pub(crate) struct AgentOrchestrationToolProvider {
    supervisor: AgentSupervisor,
}

impl AgentOrchestrationToolProvider {
    pub fn new(supervisor: AgentSupervisor) -> Self {
        Self { supervisor }
    }

    async fn run_agent_team(
        &self,
        call: &ToolCall,
        ctx: &ToolExecutionContext,
    ) -> Result<ToolResult> {
        let agent_cfg = &self.supervisor.runtime.inner.effective_config.agent;
        if !agent_cfg.teams_enabled {
            return Ok(tool_error(
                call,
                "agent teams are disabled by configuration",
            ));
        }
        if ctx.agent_depth > 0 {
            return Ok(tool_error(
                call,
                "only the top-level coordinator may create an agent team",
            ));
        }
        if agent_cfg.max_concurrent_child_agents() == 0 {
            return Ok(tool_error(
                call,
                "no child-agent concurrency slots are available",
            ));
        }

        let objective = call
            .arguments
            .get("objective")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("run_agent_team: objective is required"))?
            .to_string();
        let assignments_value = call
            .arguments
            .get("assignments")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("run_agent_team: assignments must be an array"))?;
        if assignments_value.is_empty() || assignments_value.len() > agent_cfg.team_max_members {
            return Ok(tool_error(
                call,
                &format!(
                    "team must contain between 1 and {} members",
                    agent_cfg.team_max_members
                ),
            ));
        }
        let assignments: Vec<TeamAssignment> = assignments_value
            .iter()
            .cloned()
            .map(serde_json::from_value)
            .collect::<std::result::Result<_, _>>()
            .map_err(|error| anyhow!("run_agent_team: invalid assignment: {error}"))?;
        if assignments.iter().any(|assignment| {
            assignment.role.trim().is_empty() || assignment.objective.trim().is_empty()
        }) {
            return Ok(tool_error(
                call,
                "every team assignment needs a non-empty role and objective",
            ));
        }

        let timeout_secs = call
            .arguments
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(agent_cfg.child_timeout_secs)
            .clamp(
                agent_cfg.child_timeout_secs,
                agent_cfg.max_child_timeout_secs,
            );
        let team_id = format!("team_{}", Uuid::new_v4().simple());
        let _ = self
            .supervisor
            .runtime
            .publish_event(
                RuntimeEventKind::AgentTeamStarted,
                Some(ctx.thread_id.clone()),
                Some(ctx.turn_id.clone()),
                json!({
                    "teamId": team_id,
                    "parentToolCallId": call.tool_call_id,
                    "objective": objective,
                    "memberCount": assignments.len(),
                    "readOnly": true,
                    "timeoutSecs": timeout_secs,
                }),
            )
            .await;

        for (member_index, assignment) in assignments.iter().enumerate() {
            let _ = self
                .supervisor
                .runtime
                .publish_event(
                    RuntimeEventKind::AgentTeamMemberUpdated,
                    Some(ctx.thread_id.clone()),
                    Some(ctx.turn_id.clone()),
                    json!({
                        "teamId": team_id,
                        "parentToolCallId": call.tool_call_id,
                        "memberIndex": member_index,
                        "role": assignment.role,
                        "objective": assignment.objective,
                        "status": "queued",
                    }),
                )
                .await;
        }

        let team_cancel = CancellationToken::new();
        let parent_cancel = ctx.cancel_token.clone();
        let linked_team_cancel = team_cancel.clone();
        let cancellation_link = tokio::spawn(async move {
            parent_cancel.cancelled().await;
            linked_team_cancel.cancel();
        });
        let team_deadline = tokio::time::sleep(Duration::from_secs(timeout_secs));
        tokio::pin!(team_deadline);

        let mut join_set = tokio::task::JoinSet::new();
        for (index, assignment) in assignments.into_iter().enumerate() {
            let supervisor = self.supervisor.clone();
            let mut read_only_profile = ctx.permission_profile.clone();
            read_only_profile.sandbox_mode = SandboxMode::ReadOnly;
            read_only_profile.writable_roots.clear();
            let request = ChildAgentRequest {
                prompt: team_member_prompt(&objective, &assignment),
                cwd: ctx.cwd.clone(),
                approval_policy: ctx.approval_policy.clone(),
                permission_profile: read_only_profile,
                timeout_secs,
                agent_depth: ctx.agent_depth,
                parent_thread_id: ctx.thread_id.clone(),
                parent_turn_id: ctx.turn_id.clone(),
                parent_tool_call_id: call.tool_call_id.clone(),
                cancel_token: team_cancel.clone(),
                allow_spawning: false,
                team_member: Some(TeamMemberContext {
                    team_id: team_id.clone(),
                    member_index: index,
                    role: assignment.role.clone(),
                    objective: assignment.objective.clone(),
                }),
            };
            join_set.spawn(async move {
                let result = supervisor.run_child(request).await;
                (index, assignment, result)
            });
        }

        let mut reports = Vec::new();
        let mut team_timed_out = false;
        while !join_set.is_empty() {
            let joined = if team_timed_out {
                join_set.join_next().await
            } else {
                tokio::select! {
                    joined = join_set.join_next() => joined,
                    _ = &mut team_deadline => {
                        team_timed_out = true;
                        team_cancel.cancel();
                        continue;
                    }
                }
            };
            let Some(joined) = joined else { break };
            match joined {
                Ok((index, assignment, Ok(run))) => {
                    let (status, report) = match run.outcome {
                        TurnCompletionOutcome::Completed => (
                            "completed",
                            parse_team_report(&run.result_text)
                                .unwrap_or_else(|| json!({"summary": run.result_text})),
                        ),
                        TurnCompletionOutcome::Failed(reason) => (
                            "failed",
                            json!({"summary": run.result_text, "error": reason}),
                        ),
                        TurnCompletionOutcome::Interrupted => {
                            ("interrupted", json!({"summary": run.result_text}))
                        }
                        TurnCompletionOutcome::TimedOut => {
                            ("timed_out", json!({"summary": run.result_text}))
                        }
                    };
                    let _ = self
                        .supervisor
                        .runtime
                        .publish_event(
                            RuntimeEventKind::AgentTeamMemberUpdated,
                            Some(ctx.thread_id.clone()),
                            Some(ctx.turn_id.clone()),
                            json!({
                                "teamId": team_id,
                                "parentToolCallId": call.tool_call_id,
                                "memberIndex": index,
                                "role": assignment.role,
                                "objective": assignment.objective,
                                "status": status,
                                "childThreadId": run.child_thread_id,
                                "childTurnId": run.child_turn_id,
                            }),
                        )
                        .await;
                    reports.push(TeamMemberReport {
                        index,
                        role: assignment.role,
                        objective: assignment.objective,
                        child_thread_id: Some(run.child_thread_id),
                        child_turn_id: Some(run.child_turn_id),
                        status: status.into(),
                        report,
                    });
                }
                Ok((index, assignment, Err(error))) => {
                    let _ = self
                        .supervisor
                        .runtime
                        .publish_event(
                            RuntimeEventKind::AgentTeamMemberUpdated,
                            Some(ctx.thread_id.clone()),
                            Some(ctx.turn_id.clone()),
                            json!({
                                "teamId": team_id,
                                "parentToolCallId": call.tool_call_id,
                                "memberIndex": index,
                                "role": assignment.role,
                                "objective": assignment.objective,
                                "status": "failed",
                            }),
                        )
                        .await;
                    reports.push(TeamMemberReport {
                        index,
                        role: assignment.role,
                        objective: assignment.objective,
                        child_thread_id: None,
                        child_turn_id: None,
                        status: "failed".into(),
                        report: json!({"error": error.to_string()}),
                    });
                }
                Err(error) => reports.push(TeamMemberReport {
                    index: usize::MAX,
                    role: "unknown".into(),
                    objective: "worker task".into(),
                    child_thread_id: None,
                    child_turn_id: None,
                    status: "failed".into(),
                    report: json!({"error": format!("team worker join failed: {error}")}),
                }),
            }
        }
        cancellation_link.abort();
        reports.sort_by_key(|report| report.index);
        let completed = reports
            .iter()
            .filter(|report| report.status == "completed")
            .count();
        let _ = self
            .supervisor
            .runtime
            .publish_event(
                RuntimeEventKind::AgentTeamCompleted,
                Some(ctx.thread_id.clone()),
                Some(ctx.turn_id.clone()),
                json!({
                    "teamId": team_id,
                    "memberCount": reports.len(),
                    "completed": completed,
                    "failed": reports.len().saturating_sub(completed),
                    "timedOut": team_timed_out,
                }),
            )
            .await;

        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: completed > 0,
            output: json!({
                "team_id": team_id,
                "objective": objective,
                "read_only": true,
                "coordinator_is_sole_writer": true,
                "timed_out": team_timed_out,
                "reports": reports,
                "next_step": "Synthesize the evidence, make any edits yourself, then run final verification.",
            }),
            error_message: (completed == 0).then(|| "all team members failed".into()),
        })
    }
}

fn tool_error(call: &ToolCall, message: &str) -> ToolResult {
    ToolResult {
        tool_call_id: call.tool_call_id.clone(),
        ok: false,
        output: json!({"error": message}),
        error_message: Some(message.to_string()),
    }
}

fn team_member_prompt(overall_objective: &str, assignment: &TeamAssignment) -> String {
    let focus = if assignment.focus_paths.is_empty() {
        "No path restriction; inspect only files relevant to your assignment.".to_string()
    } else {
        format!("Focus paths: {}", assignment.focus_paths.join(", "))
    };
    let questions = if assignment.questions.is_empty() {
        "No additional questions.".to_string()
    } else {
        format!("Questions:\n- {}", assignment.questions.join("\n- "))
    };
    format!(
        "You are the {role} member of a read-only research team.\n\n\
         Overall objective: {overall_objective}\n\n\
         Your bounded assignment: {member_objective}\n\n\
         {focus}\n{questions}\n\n\
         You are strictly read-only. Do not edit files, run commands that mutate the repository, \
         install dependencies, or create commits. Gather concrete evidence and finish promptly.\n\n\
         Return one JSON object with exactly these top-level fields:\n\
         {{\"summary\":\"...\",\"findings\":[{{\"claim\":\"...\",\"evidence\":\"file:line or command output\",\"severity\":\"info|warning|critical\"}}],\
         \"relevant_files\":[\"...\"],\"risks\":[\"...\"],\"recommendations\":[\"...\"],\"blockers\":[\"...\"]}}",
        role = assignment.role,
        member_objective = assignment.objective,
    )
}

fn parse_team_report(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    let value: Value = serde_json::from_str(trimmed).ok().or_else(|| {
        let start = trimmed.find('{')?;
        let end = trimmed.rfind('}')?;
        serde_json::from_str(&trimmed[start..=end]).ok()
    })?;
    let object = value.as_object()?;
    object.get("summary")?.as_str()?;
    for field in [
        "findings",
        "relevant_files",
        "risks",
        "recommendations",
        "blockers",
    ] {
        object.get(field)?.as_array()?;
    }
    Some(value)
}

#[async_trait]
impl ToolProvider for AgentOrchestrationToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        let agent_cfg = &self.supervisor.runtime.inner.effective_config.agent;
        let child_budget = agent_cfg.max_concurrent_child_agents();
        let mut tools = vec![ToolDefinition {
            name: "spawn_agent".into(),
            description: format!(
                "Spawn an independent sub-agent for a bounded task. \
                The sub-agent runs with full tool access and returns its final answer as text. \
                Use this for a small number of independent sub-tasks, such as analysing specific files or modules. \
                At most {} total agents run concurrently (including the parent); this allows up to {} child agents at once. \
                Extra calls queue behind that limit. \
                Avoid exhaustive directory inventory tasks unless the child can summarize partial/truncated evidence and finish promptly.",
                agent_cfg.max_concurrent_agents,
                child_budget
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task description for the sub-agent. Be specific and self-contained."
                    },
                    "write_paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional ownership declaration for paths this sub-agent may edit. Use an empty array for read-only analysis tasks. Omit only if edits are not expected."
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
        }];
        if agent_cfg.teams_enabled {
            tools.push(ToolDefinition {
            name: "run_agent_team".into(),
            description: format!(
                "Run 1–{} independent research agents concurrently and return structured reports. \
                 Team members are always read-only; the parent coordinator remains solely responsible \
                 for edits, integration, and final verification. Use this for genuinely independent \
                 investigation, review, or risk-analysis assignments—not routine file listing.",
                agent_cfg.team_max_members
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "Overall problem the team is helping the coordinator solve."
                    },
                    "assignments": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": agent_cfg.team_max_members,
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "objective": {"type": "string"},
                                "focus_paths": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "questions": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["role", "objective"]
                        }
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": format!(
                            "Overall team deadline and per-member timeout (default {}, max {}).",
                            agent_cfg.child_timeout_secs,
                            agent_cfg.max_child_timeout_secs
                        )
                    }
                },
                "required": ["objective", "assignments"]
            }),
            requires_approval: false,
            supports_parallel_calls: false,
            provider_kind: ToolProviderKind::Builtin,
            });
        }
        tools
    }

    async fn execute(&self, call: &ToolCall, ctx: &ToolExecutionContext) -> Result<ToolResult> {
        let agent_cfg = &self.supervisor.runtime.inner.effective_config.agent;
        if call.tool_name == "run_agent_team" {
            return self.run_agent_team(call, ctx).await;
        }
        if agent_cfg.max_concurrent_child_agents() == 0 {
            return Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({ "error": "child_agent_concurrency_disabled" }),
                error_message: Some(
                    "spawn_agent disabled: max_concurrent_agents leaves no child slots".into(),
                ),
            });
        }

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
                cancel_token: ctx.cancel_token.clone(),
                allow_spawning: true,
                team_member: None,
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
                    "error": STATUS_INTERRUPTED,
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
                    "error": STATUS_TIMEOUT,
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
