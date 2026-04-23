use anyhow::{bail, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify, RwLock as AsyncRwLock};
use tokio::time::{timeout, Duration};

use crate::system::domain::{
    now_seconds, ApprovalDecision, ApprovalRequest, ApprovalResolution, ApprovalsReviewerKind,
    ConversationItem, PendingApproval, PrefixRule,
};

pub struct AutoReviewer;

impl AutoReviewer {
    pub fn review(
        &self,
        request: &ApprovalRequest,
        _transcript: &[ConversationItem],
    ) -> ApprovalDecision {
        let action = request.action.to_string();
        if action.contains("\"category\":\"SandboxEscalation\"")
            || action.contains("\"networkEnabled\":true")
        {
            ApprovalDecision::Denied
        } else {
            ApprovalDecision::Approved
        }
    }
}

struct PendingApprovalState {
    pending: PendingApproval,
    resolution: Mutex<Option<ApprovalResolution>>,
    notify: Notify,
}

pub struct ApprovalManager {
    pending: Arc<AsyncRwLock<HashMap<String, Arc<PendingApprovalState>>>>,
    reviewer_kind: ApprovalsReviewerKind,
    auto_reviewer: AutoReviewer,
}

impl ApprovalManager {
    pub fn new(reviewer_kind: ApprovalsReviewerKind) -> Self {
        Self {
            pending: Arc::new(AsyncRwLock::new(HashMap::new())),
            reviewer_kind,
            auto_reviewer: AutoReviewer,
        }
    }

    pub async fn create_approval(&self, request: ApprovalRequest) -> PendingApproval {
        let pending = PendingApproval {
            request,
            created_at: now_seconds(),
            reviewer_kind: self.reviewer_kind,
        };
        let state = Arc::new(PendingApprovalState {
            pending: pending.clone(),
            resolution: Mutex::new(None),
            notify: Notify::new(),
        });
        self.pending
            .write()
            .await
            .insert(pending.request.approval_id.clone(), state);
        pending
    }

    pub async fn resolve_approval(
        &self,
        approval_id: &str,
        decision: ApprovalDecision,
        rule: Option<PrefixRule>,
    ) -> Option<ApprovalResolution> {
        let pending = self.pending.read().await.get(approval_id).cloned();
        let state = pending?;
        let resolution = ApprovalResolution {
            approval_id: approval_id.to_string(),
            decision,
            persisted_rule: rule,
            reviewer_kind: state.pending.reviewer_kind,
        };
        *state.resolution.lock().await = Some(resolution.clone());
        state.notify.notify_waiters();
        Some(resolution)
    }

    pub async fn wait_for_approval(
        &self,
        approval_id: &str,
        timeout_seconds: i32,
        transcript: &[ConversationItem],
    ) -> Result<ApprovalResolution> {
        let pending = self.pending.read().await.get(approval_id).cloned();
        let Some(state) = pending else {
            bail!("approval not found: {approval_id}");
        };

        if state.pending.reviewer_kind == ApprovalsReviewerKind::AutoReviewer {
            let decision = self
                .auto_reviewer
                .review(&state.pending.request, transcript);
            return Ok(self
                .resolve_approval(approval_id, decision, None)
                .await
                .expect("approval exists"));
        }

        let waited = timeout(Duration::from_secs(timeout_seconds as u64), async {
            loop {
                if let Some(resolution) = state.resolution.lock().await.clone() {
                    return Ok(resolution);
                }
                state.notify.notified().await;
            }
        })
        .await;

        match waited {
            Ok(result) => result,
            Err(_) => Ok(self
                .resolve_approval(approval_id, ApprovalDecision::TimedOut, None)
                .await
                .expect("approval exists")),
        }
    }

    pub async fn cancel_for_turn(
        &self,
        thread_id: &str,
        turn_id: &str,
    ) -> Vec<ApprovalResolution> {
        let states = self
            .pending
            .read()
            .await
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let mut cancelled = Vec::new();
        for state in states {
            if state.pending.request.thread_id == thread_id
                && state.pending.request.turn_id == turn_id
            {
                if let Some(resolution) = self
                    .resolve_approval(
                        &state.pending.request.approval_id,
                        ApprovalDecision::Cancelled,
                        None,
                    )
                    .await
                {
                    cancelled.push(resolution);
                }
            }
        }
        cancelled
    }
}
