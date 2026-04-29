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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::domain::ApprovalPolicyKind;

    fn ask_default() -> ApprovalPolicy {
        ApprovalPolicy {
            kind: ApprovalPolicyKind::OnRequest,
            granular: None,
        }
    }

    #[test]
    fn defaults_are_empty_and_ask_mode() {
        let s = ApprovalState::new();
        assert!(!s.has_pending());
        assert!(s.policy_override().is_none());
        assert!(!s.auto_enabled(&ask_default()));
        assert_eq!(s.mode_label(&ask_default()), "ask");
    }

    #[test]
    fn override_for_auto_round_trips_through_policy() {
        let on = ApprovalState::override_for_auto(true).unwrap();
        assert_eq!(on.kind, ApprovalPolicyKind::Never);
        assert!(ApprovalState::override_for_auto(false).is_none());
    }

    #[test]
    fn auto_enabled_respects_override() {
        let mut s = ApprovalState::new();
        s.set_policy_override(ApprovalState::override_for_auto(true));
        assert!(s.auto_enabled(&ask_default()));
        assert_eq!(s.mode_label(&ask_default()), "auto");
        s.set_policy_override(None);
        assert!(!s.auto_enabled(&ask_default()));
    }

    #[test]
    fn effective_policy_falls_back_to_default_when_no_override() {
        let s = ApprovalState::new();
        let default = ask_default();
        let eff = s.effective_policy(&default);
        assert_eq!(eff.kind, ApprovalPolicyKind::OnRequest);
    }

    #[test]
    fn auto_default_is_auto_even_without_override() {
        let s = ApprovalState::new();
        let auto_default = ApprovalPolicy {
            kind: ApprovalPolicyKind::Never,
            granular: None,
        };
        assert!(s.auto_enabled(&auto_default));
        assert_eq!(s.mode_label(&auto_default), "auto");
    }

    #[test]
    fn take_pending_clears_slot() {
        let s = ApprovalState::new();
        assert!(!s.has_pending());
        // Note: we can't easily construct PendingApprovalView in this test
        // without dragging in domain types; the empty path is the load-bearing
        // case (resolve_pending_approval_auto's early-return guard).
        let mut s = s;
        assert!(s.take_pending().is_none());
        assert!(!s.has_pending());
    }
}
