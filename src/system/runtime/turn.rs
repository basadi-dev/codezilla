//! Turn lifecycle methods on [`ConversationRuntime`].
//!
//! `start_turn` is the heart of the runtime: it allocates a turn, spawns the
//! `TurnExecutor` task, and surfaces events. `interrupt_turn` and `steer_turn`
//! are the two side-channels the TUI/exec surfaces use to influence an
//! in-flight turn. `resolve_approval` lives here too because it's how callers
//! unblock a turn that's parked on an `ApprovalRequested` event.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    ConversationRuntime, LoadedTurn, TurnInterruptParams, TurnInterruptResult, TurnStartParams,
    TurnStartResult, TurnSteerParams, TurnSteerResult,
};
use crate::system::agent::TurnExecutor;
use crate::system::domain::{
    now_seconds, ApprovalDecision, ApprovalResolution, PrefixRule, RuntimeEventKind, SurfaceKind,
    ThreadStatus, TokenUsage, TurnMetadata, TurnStatus,
};

impl ConversationRuntime {
    pub async fn start_turn(
        &self,
        params: TurnStartParams,
        surface: SurfaceKind,
    ) -> Result<TurnStartResult> {
        let thread = self
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;

        let turn = {
            let mut thread_guard = thread.lock().await;
            if thread_guard.metadata.archived {
                bail!("thread_archived: {}", thread_guard.metadata.thread_id);
            }
            if thread_guard.active_turn_id.is_some() {
                bail!("turn_already_active: {}", thread_guard.metadata.thread_id);
            }
            let turn = TurnMetadata {
                turn_id: format!("turn_{}", Uuid::new_v4().simple()),
                thread_id: thread_guard.metadata.thread_id.clone(),
                created_at: now_seconds(),
                updated_at: now_seconds(),
                status: TurnStatus::Created,
                started_by_surface: surface,
                token_usage: TokenUsage::default(),
            };
            self.inner.persistence_manager.create_turn(&turn)?;
            thread_guard.active_turn_id = Some(turn.turn_id.clone());
            thread_guard.metadata.status = ThreadStatus::Running;
            thread_guard.approval_policy_override = params.approval_policy.clone();
            thread_guard.turns.push(LoadedTurn {
                metadata: turn.clone(),
                items: Vec::new(),
                status: TurnStatus::Created,
                pending_tool_calls: HashMap::new(),
                stream_buffer: Vec::new(),
                cancel_token: CancellationToken::new(),
            });
            turn
        };

        self.publish_event(
            RuntimeEventKind::TurnStarted,
            Some(params.thread_id.clone()),
            Some(turn.turn_id.clone()),
            serde_json::to_value(&turn)?,
        )
        .await?;

        let runtime = self.clone();
        let params_clone = params.clone();
        let thread_id_for_task = params.thread_id.clone();
        let turn_id = turn.turn_id.clone();
        let turn_for_result = turn.clone();
        tokio::spawn(async move {
            let executor = TurnExecutor { runtime };
            if let Err(error) = executor.run_turn(params_clone, turn_id.clone()).await {
                let error_text = error.to_string();
                let err = crate::system::error::from_raw(&error_text);
                let _ = executor
                    .fail_turn(
                        &thread_id_for_task,
                        &turn_id,
                        err.kind.label(),
                        &err.message,
                    )
                    .await;
            }
        });

        Ok(TurnStartResult {
            turn: turn_for_result,
        })
    }

    pub async fn interrupt_turn(&self, params: TurnInterruptParams) -> Result<TurnInterruptResult> {
        let thread = self
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;
        let cancelled = {
            let mut thread_guard = thread.lock().await;
            let Some(active_id) = &thread_guard.active_turn_id else {
                return Ok(TurnInterruptResult {
                    turn_id: params.turn_id,
                    interrupted: false,
                });
            };
            if active_id != &params.turn_id {
                bail!("turn_not_found: {}", params.turn_id);
            }
            let turn = thread_guard
                .turns
                .iter_mut()
                .find(|t| t.metadata.turn_id == params.turn_id)
                .ok_or_else(|| anyhow!("turn_not_found: {}", params.turn_id))?;
            turn.cancel_token.cancel();
            true
        };

        let approvals = self
            .inner
            .approval_manager
            .cancel_for_turn(&params.thread_id, &params.turn_id)
            .await;
        for resolution in approvals {
            self.publish_event(
                RuntimeEventKind::ApprovalResolved,
                Some(params.thread_id.clone()),
                Some(params.turn_id.clone()),
                serde_json::to_value(&resolution)?,
            )
            .await?;
        }

        Ok(TurnInterruptResult {
            turn_id: params.turn_id,
            interrupted: cancelled,
        })
    }

    pub async fn steer_turn(&self, params: TurnSteerParams) -> Result<TurnSteerResult> {
        let thread = self
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;
        let queued = {
            let mut guard = thread.lock().await;
            if guard.active_turn_id.as_deref() != Some(params.expected_turn_id.as_str()) {
                bail!("turn_mismatch: {}", params.expected_turn_id);
            }
            let count = params.input.len();
            guard.pending_steering.extend(params.input);
            count
        };
        Ok(TurnSteerResult {
            turn_id: params.expected_turn_id,
            queued_items: queued,
        })
    }

    pub async fn resolve_approval(
        &self,
        approval_id: &str,
        decision: ApprovalDecision,
        rule: Option<PrefixRule>,
    ) -> Result<Option<ApprovalResolution>> {
        Ok(self
            .inner
            .approval_manager
            .resolve_approval(approval_id, decision, rule)
            .await)
    }
}
