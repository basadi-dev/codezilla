use crate::system::config::TurnPolicyMode;
use crate::system::domain::ReasoningEffort;

use super::state::{TurnEvidence, TurnPhase, VerificationStatus};
use super::utils::{intent_to_reasoning_effort, TurnIntent};

pub(super) struct TurnDecision {
    pub(super) reasoning_effort: ReasoningEffort,
    pub(super) allow_completion: bool,
    pub(super) completion_blocker: Option<String>,
    pub(super) enforceable: bool,
}

pub(super) fn decide_turn(
    intent: TurnIntent,
    phase: TurnPhase,
    evidence: &TurnEvidence,
    configured_effort: ReasoningEffort,
    verification_requested: bool,
    completion_requested: bool,
    mode: TurnPolicyMode,
) -> TurnDecision {
    let baseline_effort = intent_to_reasoning_effort(intent, configured_effort);
    let mut proposed_effort = baseline_effort;
    if configured_effort == ReasoningEffort::Auto
        && (evidence.failed_commands >= 2
            || evidence.failed_tool_rounds >= 2
            || evidence.files_changed.len() > 3)
    {
        proposed_effort = ReasoningEffort::High;
    }
    // Observe mode reports the evidence-driven choice without changing model
    // settings. Enforce mode is the explicit opt-in for behavioral changes.
    let reasoning_effort = if mode.enforces() {
        proposed_effort
    } else {
        baseline_effort
    };

    let (completion_blocker, enforceable) = if completion_requested
        && verification_requested
        && !evidence.files_changed.is_empty()
        && !matches!(evidence.verification, VerificationStatus::Passed)
    {
        let message = match evidence.verification {
            VerificationStatus::Failed => {
                "The latest verification command failed. Fix the failure and run the narrowest relevant check again."
            }
            _ => {
                "Files changed but successful verification evidence is missing. Run the narrowest relevant test, build, or lint command."
            }
        };
        (Some(message.to_string()), true)
    } else if completion_requested
        && matches!(intent, TurnIntent::Edit)
        && evidence.files_changed.is_empty()
        && evidence.completed_tool_rounds > 0
    {
        (
            Some(
                "An edit was requested, but the turn has produced no file-change evidence."
                    .to_string(),
            ),
            false,
        )
    } else {
        (None, false)
    };

    let allow_completion = !mode.enforces() || !enforceable || completion_blocker.is_none();
    if mode.observes() {
        tracing::debug!(
            phase = phase.label(),
            intent = ?intent,
            reasoning_effort = reasoning_effort.as_str(),
            proposed_reasoning_effort = proposed_effort.as_str(),
            files_read = evidence.files_read.len(),
            files_changed = evidence.files_changed.len(),
            successful_commands = evidence.successful_commands,
            failed_commands = evidence.failed_commands,
            completion_requested,
            allow_completion,
            blocker = completion_blocker.as_deref().unwrap_or("none"),
            enforceable,
            policy_mode = ?mode,
            "turn_policy: decision"
        );
    }

    TurnDecision {
        reasoning_effort,
        allow_completion,
        completion_blocker,
        enforceable,
    }
}
