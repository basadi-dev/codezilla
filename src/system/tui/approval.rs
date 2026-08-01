//! Approval-modal reducer — the pure-state half of the TUI's approval flow.
//!
//! Holds two things:
//!   - `pending`: the in-flight approval the user is being asked about.
//!   - `policy_override`: a per-session policy that overrides the runtime's
//!     default (used to flip auto-approve on/off without restarting).
//!
//! The async side-effectful work (calling `runtime.resolve_approval`,
//! `runtime.set_thread_approval_policy`) stays on `InteractiveApp`. This
//! reducer only owns the state mutations and the small label/predicate
//! helpers the rendering and input layers consult on every frame.

use crate::system::domain::{ApprovalPolicy, ApprovalPolicyKind};

use super::types::PendingApprovalView;

#[derive(Debug, Default)]
pub struct ApprovalState {
    pending: Option<PendingApprovalView>,
    policy_override: Option<ApprovalPolicy>,
}

impl ApprovalState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pending(&self) -> Option<&PendingApprovalView> {
        self.pending.as_ref()
    }

    /// Replace the pending-approval slot.
    pub fn set_pending(&mut self, view: Option<PendingApprovalView>) {
        self.pending = view;
    }

    /// Take the pending approval out of the slot (e.g. after resolving it).
    #[allow(dead_code)] // public API surface; covered by tests, no live consumer yet
    pub fn take_pending(&mut self) -> Option<PendingApprovalView> {
        self.pending.take()
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn policy_override(&self) -> Option<&ApprovalPolicy> {
        self.policy_override.as_ref()
    }

    pub fn set_policy_override(&mut self, policy: Option<ApprovalPolicy>) {
        self.policy_override = policy;
    }

    /// Build the policy override for "auto-approve enabled" / "off". Lifted
    /// out so the same construction is used by the toggle path and any other
    /// callers (e.g. config-driven init) without duplicating the variant.
    pub fn override_for_auto(enabled: bool) -> Option<ApprovalPolicy> {
        if enabled {
            Some(ApprovalPolicy {
                kind: ApprovalPolicyKind::Never,
                granular: None,
            })
        } else {
            None
        }
    }

    /// True when the effective policy auto-approves every tool call. Computed
    /// against `default` (typically `runtime.effective_config().approval_policy`)
    /// when no override is set.
    pub fn auto_enabled(&self, default: &ApprovalPolicy) -> bool {
        matches!(
            self.effective_policy(default).kind,
            ApprovalPolicyKind::Never
        )
    }

    /// Effective policy = override if set, otherwise `default`.
    pub fn effective_policy(&self, default: &ApprovalPolicy) -> ApprovalPolicy {
        self.policy_override
            .clone()
            .unwrap_or_else(|| default.clone())
    }

    /// Status-bar label describing the active approval mode.
    pub fn mode_label(&self, default: &ApprovalPolicy) -> &'static str {
        if self.auto_enabled(default) {
            "auto"
        } else {
            "ask"
        }
    }
}
