//! Checkpoint review middleware — validates file changes between tool rounds.
//!
//! Wraps the existing `CheckpointReviewer` as a `TurnMiddleware::post_tools`
//! implementation. Manages its own cooldown, review count, and change
//! accumulator via interior mutability so the middleware chain can call it
//! through `&self`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use uuid::Uuid;

use super::{MiddlewareAction, TurnMiddleware, TurnMiddlewareContext};
use crate::system::domain::{
    now_seconds, ConversationItem, FileChangeSummary, ItemKind, ToolResult, TurnId,
};

// ─── CheckpointReviewMiddleware ───────────────────────────────────────────────

/// Post-tool middleware that fires a lightweight checkpoint review after
/// file-changing tool rounds. Injects corrective feedback into the next
/// model call if issues are found.
pub struct CheckpointReviewMiddleware {
    /// Accumulated file changes since the last review.
    changes_since_last_review: Mutex<Vec<FileChangeSummary>>,
    /// Number of reviews completed this turn.
    reviews_run: AtomicUsize,
    /// Cooldown: skip one tool round after injecting review feedback to
    /// prevent the review→fix→review death spiral.
    cooldown: AtomicBool,
}

impl CheckpointReviewMiddleware {
    pub fn new() -> Self {
        Self {
            changes_since_last_review: Mutex::new(Vec::new()),
            reviews_run: AtomicUsize::new(0),
            cooldown: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl TurnMiddleware for CheckpointReviewMiddleware {
    fn name(&self) -> &str {
        "checkpoint_review"
    }

    async fn post_tools(
        &self,
        ctx: &mut TurnMiddlewareContext,
        _results: &[ToolResult],
        round_file_changes: &[FileChangeSummary],
    ) -> Result<MiddlewareAction> {
        let cfg = &ctx.agent_config;

        // Not enabled or inside a child agent — skip entirely.
        if !cfg.checkpoint_review_enabled || ctx.agent_depth > 0 {
            return Ok(MiddlewareAction::Continue);
        }

        // Accumulate this round's file changes.
        {
            let mut changes = self.changes_since_last_review.lock().unwrap();
            changes.extend(round_file_changes.iter().cloned());
        }

        let reviews_run = self.reviews_run.load(Ordering::SeqCst);
        let in_cooldown = self.cooldown.load(Ordering::SeqCst);

        // Snapshot current changes for the condition check.
        let changes_count = {
            let changes = self.changes_since_last_review.lock().unwrap();
            changes.len()
        };

        if changes_count == 0 {
            return Ok(MiddlewareAction::Continue);
        }

        // Cooldown: clear it after the fix round, but don't review yet.
        if in_cooldown {
            self.cooldown.store(false, Ordering::SeqCst);
            tracing::debug!(
                turn_id = %ctx.turn_id,
                reviews_run,
                max = cfg.checkpoint_review_max_per_turn,
                "checkpoint_review_middleware: cooldown cleared after fix attempt"
            );
            return Ok(MiddlewareAction::Continue);
        }

        // Cap reached — clear accumulator and skip.
        if reviews_run >= cfg.checkpoint_review_max_per_turn {
            tracing::debug!(
                turn_id = %ctx.turn_id,
                reviews_run,
                max = cfg.checkpoint_review_max_per_turn,
                "checkpoint_review_middleware: review cap reached — skipping"
            );
            self.changes_since_last_review.lock().unwrap().clear();
            return Ok(MiddlewareAction::Continue);
        }

        // Not enough changes yet — wait for more.
        if changes_count < cfg.checkpoint_review_min_changes {
            return Ok(MiddlewareAction::Continue);
        }

        // ── Fire the review ───────────────────────────────────────────────────
        let runtime = match ctx.runtime.as_ref() {
            Some(rt) => rt,
            None => return Ok(MiddlewareAction::Continue),
        };

        let reviewer = crate::system::agent::review::CheckpointReviewer::new(
            runtime.clone(),
            &ctx.agent_config,
        );

        let changes_snapshot = {
            let changes = self.changes_since_last_review.lock().unwrap();
            changes.clone()
        };

        match reviewer
            .review(
                &ctx.user_task_text,
                &ctx.plan_steps,
                &ctx.completed_actions,
                &changes_snapshot,
                &ctx.thread_id,
                &ctx.turn_id,
            )
            .await
        {
            Ok(verdict) => {
                let review_index = self.reviews_run.fetch_add(1, Ordering::SeqCst) + 1;

                // Persist the review verdict for observability.
                persist_review_item(
                    runtime,
                    &ctx.thread_id,
                    &ctx.turn_id,
                    &verdict,
                    &changes_snapshot,
                    review_index,
                )
                .await?;

                // Reset the accumulator — next review starts fresh.
                self.changes_since_last_review.lock().unwrap().clear();

                if !verdict.approved {
                    let feedback =
                        crate::system::agent::review::build_review_feedback_instruction(&verdict);
                    tracing::info!(
                        turn_id = %ctx.turn_id,
                        issues = verdict.issues.len(),
                        review_index,
                        remaining = cfg.checkpoint_review_max_per_turn - review_index,
                        "checkpoint_review_middleware: issues found — injecting feedback"
                    );
                    // Enter cooldown to prevent review→fix→review death spiral.
                    self.cooldown.store(true, Ordering::SeqCst);
                    return Ok(MiddlewareAction::InjectInstruction(feedback));
                } else {
                    tracing::debug!(
                        turn_id = %ctx.turn_id,
                        "checkpoint_review_middleware: changes approved"
                    );
                }
            }
            Err(e) => {
                // Review failure is non-fatal — log and continue.
                tracing::warn!(
                    turn_id = %ctx.turn_id,
                    error = %e,
                    "checkpoint_review_middleware: review failed — continuing"
                );
                self.changes_since_last_review.lock().unwrap().clear();
            }
        }

        Ok(MiddlewareAction::Continue)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

async fn persist_review_item(
    runtime: &crate::system::runtime::ConversationRuntime,
    thread_id: &str,
    turn_id: &TurnId,
    verdict: &crate::system::domain::CheckpointReviewVerdict,
    changes: &[FileChangeSummary],
    review_index: usize,
) -> Result<()> {
    let item = ConversationItem {
        item_id: format!("review_{}", Uuid::new_v4().simple()),
        thread_id: thread_id.to_string(),
        turn_id: turn_id.clone(),
        created_at: now_seconds(),
        kind: ItemKind::ReviewMarker,
        payload: json!({
            "approved": verdict.approved,
            "issues": verdict.issues,
            "suggestions": verdict.suggestions,
            "filesReviewed": changes.iter().map(|f| &f.path).collect::<Vec<_>>(),
            "reviewIndex": review_index,
        }),
    };

    // Use the runtime's persistence manager directly.
    runtime.inner.persistence_manager.append_item(&item)?;

    // Update in-memory thread state.
    if let Ok(Some(thread)) = runtime.load_thread(thread_id).await {
        let mut guard = thread.lock().await;
        if let Some(lt) = guard
            .turns
            .iter_mut()
            .find(|t| t.metadata.turn_id == *turn_id)
        {
            lt.items.push(item.clone());
        }
    }

    // Publish lifecycle event for TUI rendering.
    let _ = runtime
        .publish_event(
            crate::system::domain::RuntimeEventKind::ItemStarted,
            Some(thread_id.to_string()),
            Some(turn_id.clone()),
            json!({
                "itemId": item.item_id,
                "kind": ItemKind::ReviewMarker,
                "payload": item.payload,
            }),
        )
        .await;

    Ok(())
}
