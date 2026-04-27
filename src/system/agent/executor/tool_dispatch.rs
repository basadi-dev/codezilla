use anyhow::{anyhow, Result};
use futures::future::join_all;
use serde_json::json;
use uuid::Uuid;

use crate::system::agent::executor::context::TurnContext;
use crate::system::agent::executor::utils::{
    action_for_tool_call, partition_into_batches, promote_to_bash_if_needed,
};
use crate::system::domain::{
    ApprovalDecision, ApprovalPolicy, ApprovalRequest, ConversationItem, FileChangeSummary,
    ItemKind, RuntimeEventKind, ToolCall, ToolExecutionContext, ToolResult, TurnStatus,
};

use super::TurnExecutor;

impl TurnExecutor {
    /// Dispatch one round of tool calls (everything the model emitted in a
    /// single turn response). Calls that advertise `supports_parallel_calls`
    /// are grouped into batches and executed concurrently with `join_all`.
    /// Non-parallel-safe calls get their own sequential batch.
    ///
    /// Returns `(had_any_success, file_changes)` — the first is used by the
    /// consecutive-failure guard in `run_turn`, the second collects file
    /// modifications for benchmark metrics.
    pub(crate) async fn execute_tool_round(
        &self,
        ctx: &TurnContext,
        tool_calls: Vec<ToolCall>,
    ) -> Result<(bool, Vec<FileChangeSummary>)> {
        // 1. Normalise: rewrite shell_exec with shell operators → bash_exec.
        let calls: Vec<ToolCall> = tool_calls
            .into_iter()
            .map(promote_to_bash_if_needed)
            .collect();

        // 2. Persist all ToolCall items in original order *before* executing
        //    anything, so the transcript always shows calls before results.
        for call in &calls {
            let call_item = ConversationItem {
                item_id: format!("item_{}", Uuid::new_v4().simple()),
                thread_id: ctx.params.thread_id.clone(),
                turn_id: ctx.turn_id.clone(),
                created_at: crate::system::domain::now_seconds(),
                kind: ItemKind::ToolCall,
                payload: serde_json::to_value(call)?,
            };
            self.persist_turn_item(call_item).await?;
        }

        // 3. Partition into batches (consecutive parallel-safe calls grouped;
        //    each non-parallel-safe call is a solo sequential batch).
        let batches = partition_into_batches(&calls, |name| {
            self.runtime
                .inner
                .tool_orchestrator
                .is_parallel_safe(name, &ctx.listing)
        });

        let call_summary: Vec<String> = calls
            .iter()
            .map(|c| {
                let args_preview = serde_json::to_string(&c.arguments)
                    .unwrap_or_default()
                    .chars()
                    .take(120)
                    .collect::<String>();
                format!("{}({})", c.tool_name, args_preview)
            })
            .collect();
        tracing::debug!(
            turn_id = %ctx.turn_id,
            total_calls = calls.len(),
            batch_count = batches.len(),
            calls = %call_summary.join(" | "),
            "tool_round: dispatching"
        );

        // 4. Dispatch each batch, collecting results in original-call order.
        let mut had_any_success = false;
        let mut file_changes: Vec<FileChangeSummary> = Vec::new();
        for batch in batches {
            if batch.len() > 1 {
                tracing::debug!("tool_round: parallel batch of {}", batch.len());
            }
            tracing::debug!(turn_id = %ctx.turn_id, "tool_round: calling dispatch_batch");
            let results = self.dispatch_batch(ctx, batch).await?;
            tracing::debug!(turn_id = %ctx.turn_id, n = results.len(), "tool_round: dispatch_batch returned");
            for result in results {
                if result.ok {
                    had_any_success = true;
                    tracing::debug!(
                        turn_id = %ctx.turn_id,
                        tool_call_id = %result.tool_call_id,
                        "tool_result: ok"
                    );
                    // Extract file change info for benchmark metrics.
                    if let Some(change) = extract_file_change(&result) {
                        file_changes.push(change);
                    }
                } else {
                    tracing::warn!(
                        turn_id = %ctx.turn_id,
                        tool_call_id = %result.tool_call_id,
                        error = result.error_message.as_deref().unwrap_or("<none>"),
                        output = %result.output,
                        "tool_result: FAILED"
                    );
                }
                tracing::debug!(turn_id = %ctx.turn_id, "tool_round: calling persist_tool_result");
                self.persist_tool_result(&ctx.params.thread_id, &ctx.turn_id, &result)
                    .await?;
                tracing::debug!(turn_id = %ctx.turn_id, "tool_round: persist_tool_result done");
            }
        }

        Ok((had_any_success, file_changes))
    }

    /// Execute a single batch of tool calls.
    ///
    /// Phase A — sequential approval: each call in the batch is approved
    ///   one at a time (preserves the existing single-prompt UX).
    /// Phase B — parallel execution: all approved calls are dispatched
    ///   concurrently with `join_all`.
    /// Phase C — stable ordering: results are sorted back to original
    ///   call-index order before being returned.
    async fn dispatch_batch(
        &self,
        ctx: &TurnContext,
        batch: Vec<(usize, ToolCall)>,
    ) -> Result<Vec<ToolResult>> {
        // ── Phase A: sequential approval ──────────────────────────────────────
        let mut ready: Vec<(usize, ToolCall, ApprovalPolicy)> = Vec::new();
        let mut results: Vec<(usize, ToolResult)> = Vec::new();

        for (idx, call) in batch {
            // Re-read policy live so a Ctrl+A toggle between batches takes
            // effect immediately.
            let approval_policy = {
                let t = ctx.thread.lock().await;
                t.approval_policy_override.clone()
            }
            .or_else(|| ctx.params.approval_policy.clone())
            .unwrap_or_else(|| self.runtime.inner.effective_config.approval_policy.clone());

            let action = action_for_tool_call(&call, &ctx.cwd);
            let requires_approval = self.runtime.inner.permission_manager.requires_approval(
                &action,
                &approval_policy,
                &ctx.cwd,
            );
            tracing::debug!(
                turn_id = %ctx.turn_id,
                tool_call_id = %call.tool_call_id,
                tool_name = %call.tool_name,
                policy = ?approval_policy.kind,
                requires_approval,
                "tool_round: approval decision"
            );
            if requires_approval {
                let approval = self
                    .runtime
                    .inner
                    .approval_manager
                    .create_approval(ApprovalRequest {
                        approval_id: format!("approval_{}", Uuid::new_v4().simple()),
                        thread_id: ctx.params.thread_id.clone(),
                        turn_id: ctx.turn_id.clone(),
                        category: action.category,
                        title: format!("Approve {}", call.tool_name),
                        justification: format!("The assistant requested {}", call.tool_name),
                        action: serde_json::to_value(&action)?,
                    })
                    .await;

                {
                    let mut t = ctx.thread.lock().await;
                    let turn = t
                        .turns
                        .iter_mut()
                        .find(|t| t.metadata.turn_id == ctx.turn_id)
                        .ok_or_else(|| anyhow!("turn_not_found: {}", ctx.turn_id))?;
                    turn.status = TurnStatus::WaitingForApproval;
                    turn.metadata.status = TurnStatus::WaitingForApproval;
                    self.runtime
                        .inner
                        .persistence_manager
                        .update_turn(&turn.metadata)?;
                }

                self.runtime
                    .publish_event(
                        RuntimeEventKind::ApprovalRequested,
                        Some(ctx.params.thread_id.clone()),
                        Some(ctx.turn_id.clone()),
                        serde_json::to_value(&approval)?,
                    )
                    .await?;

                tracing::debug!(
                    turn_id = %ctx.turn_id,
                    approval_id = %approval.request.approval_id,
                    "tool_round: waiting for approval"
                );
                let transcript = self
                    .runtime
                    .inner
                    .persistence_manager
                    .read_thread(&ctx.params.thread_id)?
                    .items;
                let resolution = self
                    .runtime
                    .inner
                    .approval_manager
                    .wait_for_approval(&approval.request.approval_id, 300, &transcript)
                    .await?;
                tracing::debug!(
                    turn_id = %ctx.turn_id,
                    approval_id = %approval.request.approval_id,
                    decision = ?resolution.decision,
                    "tool_round: approval resolved"
                );
                self.runtime
                    .publish_event(
                        RuntimeEventKind::ApprovalResolved,
                        Some(ctx.params.thread_id.clone()),
                        Some(ctx.turn_id.clone()),
                        serde_json::to_value(&resolution)?,
                    )
                    .await?;

                if resolution.decision != ApprovalDecision::Approved {
                    results.push((
                        idx,
                        ToolResult {
                            tool_call_id: call.tool_call_id.clone(),
                            ok: false,
                            output: json!({ "approved": false }),
                            error_message: Some(format!("approval {:?}", resolution.decision)),
                        },
                    ));
                    continue;
                }
            }

            ready.push((idx, call, approval_policy));
        }

        // ── Phase B: parallel execution ───────────────────────────────────────
        if !ready.is_empty() {
            let futures: Vec<_> = ready
                .into_iter()
                .map(|(idx, call, approval_policy)| {
                    let orchestrator = &self.runtime.inner.tool_orchestrator;
                    let ctx = ToolExecutionContext {
                        thread_id: ctx.params.thread_id.clone(),
                        turn_id: ctx.turn_id.clone(),
                        cwd: ctx.cwd.clone(),
                        permission_profile: ctx.permission_profile.clone(),
                        approval_policy,
                        agent_depth: ctx.params.agent_depth,
                    };
                    async move {
                        tracing::debug!(
                            tool_call_id = %call.tool_call_id,
                            tool_name = %call.tool_name,
                            "tool_round: executing tool"
                        );
                        let result =
                            orchestrator
                                .execute(&call, ctx)
                                .await
                                .unwrap_or_else(|e| ToolResult {
                                    tool_call_id: call.tool_call_id.clone(),
                                    ok: false,
                                    output: json!({ "error": e.to_string() }),
                                    error_message: Some(e.to_string()),
                                });
                        tracing::debug!(
                            tool_call_id = %call.tool_call_id,
                            tool_name = %call.tool_name,
                            ok = result.ok,
                            "tool_round: tool execution finished"
                        );
                        (idx, result)
                    }
                })
                .collect();

            results.extend(join_all(futures).await);
        }

        // ── Phase C: sort by original index before returning ──────────────────
        results.sort_by_key(|(idx, _)| *idx);
        Ok(results.into_iter().map(|(_, r)| r).collect())
    }
}

/// Extract a `FileChangeSummary` from a successful tool result for write_file
/// or patch_file operations. Returns `None` for non-file-change tools.
fn extract_file_change(result: &ToolResult) -> Option<FileChangeSummary> {
    let output = &result.output;
    let path = output.get("path")?.as_str()?.to_string();
    let diff = output
        .get("diff")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let lines_added = output
        .get("lines_added")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let lines_removed = output
        .get("lines_removed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let is_new_file = output
        .get("is_new_file")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let kind = if is_new_file {
        "create".to_string()
    } else {
        "modify".to_string()
    };

    Some(FileChangeSummary {
        path,
        kind,
        lines_added,
        lines_removed,
        diff,
    })
}
