//! Loop guard state machine — tracks anti-looping counters for the executor.
// GuardVerdict and check_* methods are public API; inline checks in executor.rs
// will be replaced by these in a future refactor.
#![allow(dead_code)]
//!
//! This module consolidates the six guard counters that were previously
//! scattered across `run_turn` as individual `let mut` bindings. By grouping
//! them into `LoopGuardState`, the executor loop becomes easier to read and
//! the guard logic can be tested independently.
//!
//! **Design note**: The guards are *not* a `TurnMiddleware` because they need
//! to control loop flow (`continue`, `return fail_turn`). Middleware hooks
//! can only return `MiddlewareAction` — they can't skip iterations or break
//! out of the loop. The guard state struct is a lower-level primitive that
//! the executor consults directly.

use std::collections::{HashMap, HashSet};

use crate::system::agent::executor::utils::ReadKey;
use crate::system::config::AgentConfig;
use crate::system::domain::FileChangeSummary;

/// Evaluation result from a guard check — tells the executor what to do.
#[derive(Debug)]
pub enum GuardVerdict {
    /// Continue normal execution.
    Continue,
    /// Inject a nudge instruction and `continue` the loop.
    NudgeAndContinue(String),
    /// Fail the turn immediately with the given (kind, message).
    FailTurn { kind: String, message: String },
}

/// Consolidated anti-looping state. Created once per turn, mutated every
/// iteration. The executor calls `evaluate_post_tools()` after each tool
/// round to get a `GuardVerdict`.
pub struct LoopGuardState {
    // ── Guard 1: consecutive-failure counter ──────────────────────────────
    pub consecutive_failures: usize,

    // ── Guard 2: absolute iteration backstop ─────────────────────────────
    pub agent_iterations: usize,
    pub no_tool_nudges: usize,
    pub completed_tool_rounds: usize,
    pub total_tool_calls: usize,
    pub file_changes: Vec<FileChangeSummary>,

    // ── Guard 3: read-only exploration saturation ────────────────────────
    pub consecutive_read_only_rounds: usize,

    // ── Guard 4: empty-response circuit breaker ──────────────────────────
    pub consecutive_empty_responses: usize,

    // ── Guard 5: cumulative nudge escalation ─────────────────────────────
    pub total_nudges: usize,
    pub total_invalid_tool_calls: usize,
    pub seen_read_signatures: HashSet<ReadKey>,

    // ── Guard 6: cross-round repetition ──────────────────────────────────
    pub cross_round_counts: HashMap<String, usize>,
    pub repetition_handled: bool,

    // ── Constrained mode ─────────────────────────────────────────────────
    pub constrained_mode: Option<ConstrainedMode>,
    pub constrained_mode_violations: usize,

    // ── Misc ─────────────────────────────────────────────────────────────
    pub verification_nudges: usize,
    pub command_attempted_after_last_file_change: bool,
    pub total_read_only_rounds: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstrainedMode {
    FinalOrActionOnly,
}

impl LoopGuardState {
    pub fn new() -> Self {
        Self {
            consecutive_failures: 0,
            agent_iterations: 0,
            no_tool_nudges: 0,
            completed_tool_rounds: 0,
            total_tool_calls: 0,
            file_changes: Vec::new(),
            consecutive_read_only_rounds: 0,
            consecutive_empty_responses: 0,
            total_nudges: 0,
            total_invalid_tool_calls: 0,
            seen_read_signatures: HashSet::new(),
            cross_round_counts: HashMap::new(),
            repetition_handled: false,
            constrained_mode: None,
            constrained_mode_violations: 0,
            verification_nudges: 0,
            command_attempted_after_last_file_change: false,
            total_read_only_rounds: 0,
        }
    }

    /// Check whether the iteration backstop has been reached.
    pub fn check_iteration_limit(&self, config: &AgentConfig) -> Option<GuardVerdict> {
        if self.agent_iterations > config.max_iterations {
            Some(GuardVerdict::FailTurn {
                kind: "loop_limit".into(),
                message: format!(
                    "Agent exceeded {} tool-call iterations without \
                     finishing. The turn has been stopped to prevent an infinite loop.",
                    config.max_iterations
                ),
            })
        } else {
            None
        }
    }

    /// Check the consecutive-failure guard after a tool round.
    pub fn check_consecutive_failures(&self, config: &AgentConfig) -> Option<GuardVerdict> {
        if self.consecutive_failures > config.max_consecutive_failures {
            Some(GuardVerdict::FailTurn {
                kind: "loop_limit".into(),
                message: format!(
                    "Agent made {} consecutive rounds where every \
                     tool call failed. The turn has been stopped. Check that tool \
                     arguments are correct and the requested paths/commands exist.",
                    self.consecutive_failures
                ),
            })
        } else {
            None
        }
    }

    /// Check whether invalid tool calls have exceeded the tolerance.
    pub fn check_invalid_tool_calls(&self) -> Option<GuardVerdict> {
        if self.total_invalid_tool_calls >= 2 {
            Some(GuardVerdict::FailTurn {
                kind: "invalid_tool_limit".into(),
                message: "The model repeatedly emitted invalid tool arguments. \
                         The turn has been stopped to prevent a failing loop."
                    .into(),
            })
        } else {
            None
        }
    }

    /// Check whether the empty-response limit has been reached.
    pub fn check_empty_responses(&self, config: &AgentConfig) -> Option<GuardVerdict> {
        if self.consecutive_empty_responses >= config.max_empty_responses {
            Some(GuardVerdict::FailTurn {
                kind: "empty_response".into(),
                message: format!(
                    "The model returned {} consecutive empty responses (no text and no tool \
                     calls). The turn has been stopped.",
                    self.consecutive_empty_responses
                ),
            })
        } else {
            None
        }
    }

    /// Check whether the total nudge budget has been exhausted.
    pub fn check_nudge_budget(&self, config: &AgentConfig) -> Option<GuardVerdict> {
        if self.total_nudges >= config.max_total_nudges {
            Some(GuardVerdict::FailTurn {
                kind: "nudge_limit".into(),
                message: format!(
                    "Agent required {} corrective nudges this turn — it is not making progress. \
                     The turn has been stopped.",
                    self.total_nudges
                ),
            })
        } else {
            None
        }
    }

    /// Record the result of a tool round (success/failure counts).
    pub fn record_tool_round(&mut self, had_any_success: bool, round_call_count: usize) {
        self.completed_tool_rounds += 1;
        self.total_tool_calls += round_call_count;
        if had_any_success {
            self.consecutive_failures = 0;
        } else {
            self.consecutive_failures += 1;
        }
    }
}

impl Default for LoopGuardState {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_is_clean() {
        let state = LoopGuardState::new();
        assert_eq!(state.consecutive_failures, 0);
        assert_eq!(state.agent_iterations, 0);
        assert_eq!(state.total_nudges, 0);
        assert!(state.constrained_mode.is_none());
    }

    #[test]
    fn iteration_limit_fires_correctly() {
        let mut state = LoopGuardState::new();
        let config = AgentConfig::default();
        state.agent_iterations = config.max_iterations;
        assert!(state.check_iteration_limit(&config).is_none());
        state.agent_iterations = config.max_iterations + 1;
        assert!(matches!(
            state.check_iteration_limit(&config),
            Some(GuardVerdict::FailTurn { .. })
        ));
    }

    #[test]
    fn consecutive_failures_tracked_correctly() {
        let mut state = LoopGuardState::new();
        state.record_tool_round(false, 2);
        assert_eq!(state.consecutive_failures, 1);
        assert_eq!(state.completed_tool_rounds, 1);
        assert_eq!(state.total_tool_calls, 2);

        state.record_tool_round(true, 1);
        assert_eq!(state.consecutive_failures, 0);
        assert_eq!(state.completed_tool_rounds, 2);
    }

    #[test]
    fn invalid_tool_calls_threshold() {
        let mut state = LoopGuardState::new();
        assert!(state.check_invalid_tool_calls().is_none());
        state.total_invalid_tool_calls = 1;
        assert!(state.check_invalid_tool_calls().is_none());
        state.total_invalid_tool_calls = 2;
        assert!(matches!(
            state.check_invalid_tool_calls(),
            Some(GuardVerdict::FailTurn { .. })
        ));
    }

    #[test]
    fn empty_response_guard() {
        let mut state = LoopGuardState::new();
        let config = AgentConfig::default();
        state.consecutive_empty_responses = config.max_empty_responses - 1;
        assert!(state.check_empty_responses(&config).is_none());
        state.consecutive_empty_responses = config.max_empty_responses;
        assert!(matches!(
            state.check_empty_responses(&config),
            Some(GuardVerdict::FailTurn { .. })
        ));
    }
}
