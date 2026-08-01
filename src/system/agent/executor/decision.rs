use crate::system::config::TurnPolicyMode;
use crate::system::domain::ReasoningEffort;

use super::intent::{intent_to_reasoning_effort, TaskTraits, TurnIntent};
use super::state::{TurnEvidence, TurnPhase, VerificationStatus};

pub(super) struct TurnDecision {
    pub(super) profile: AgentThinkingProfile,
    pub(super) allow_completion: bool,
    pub(super) completion_blocker: Option<String>,
    pub(super) enforceable: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AgentThinkingProfile {
    pub(super) model_effort: ReasoningEffort,
    pub(super) read_budget: usize,
    pub(super) require_plan: bool,
    pub(super) require_verification: bool,
    pub(super) max_recovery_attempts: usize,
    pub(super) checkpoint_min_changes: usize,
    pub(super) checkpoint_reviews: usize,
}

pub(super) fn decide_turn(
    intent: TurnIntent,
    traits: TaskTraits,
    phase: TurnPhase,
    evidence: &TurnEvidence,
    configured_effort: ReasoningEffort,
    completion_requested: bool,
    mode: TurnPolicyMode,
) -> TurnDecision {
    let baseline_effort = intent_to_reasoning_effort(intent, configured_effort);
    let mut proposed_effort = baseline_effort;
    let risky_changes = evidence
        .files_changed
        .iter()
        .any(|path| is_risky_path(path));
    let unresolved_failure = !matches!(evidence.verification, VerificationStatus::Passed);
    if configured_effort == ReasoningEffort::Auto
        && (evidence.failed_commands >= 2
            || evidence.failed_tool_rounds >= 2
            || evidence.files_changed.len() > 3
            || traits.high_risk
            || risky_changes)
        && unresolved_failure
    {
        proposed_effort = ReasoningEffort::High;
    }
    // Observe mode reports the evidence-driven choice without changing model
    // settings. Enforce mode is the explicit opt-in for behavioral changes.
    let model_effort = if mode.enforces() {
        proposed_effort
    } else {
        baseline_effort
    };
    let mut profile = profile_for(model_effort, traits);
    let mut proposed_profile = profile_for(proposed_effort, traits);
    let documentation_only = !evidence.files_changed.is_empty()
        && evidence
            .files_changed
            .iter()
            .all(|path| is_documentation(path));
    if documentation_only && !traits.verification_requested {
        profile.require_verification = false;
        proposed_profile.require_verification = false;
    } else if risky_changes {
        profile.require_verification = true;
        proposed_profile.require_verification = true;
    }

    let (completion_blocker, enforceable) = if completion_requested
        && proposed_profile.require_verification
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
            reasoning_effort = model_effort.as_str(),
            proposed_reasoning_effort = proposed_effort.as_str(),
            files_read = evidence.files_read.len(),
            files_changed = evidence.files_changed.len(),
            successful_commands = evidence.successful_commands,
            failed_commands = evidence.failed_commands,
            successful_verifications = evidence.successful_verifications,
            failed_verifications = evidence.failed_verifications,
            completion_requested,
            allow_completion,
            blocker = completion_blocker.as_deref().unwrap_or("none"),
            enforceable,
            read_budget = proposed_profile.read_budget,
            require_plan = proposed_profile.require_plan,
            require_verification = proposed_profile.require_verification,
            max_recovery_attempts = proposed_profile.max_recovery_attempts,
            checkpoint_min_changes = proposed_profile.checkpoint_min_changes,
            checkpoint_reviews = proposed_profile.checkpoint_reviews,
            policy_mode = ?mode,
            "turn_policy: decision"
        );
    }

    TurnDecision {
        profile,
        allow_completion,
        completion_blocker,
        enforceable,
    }
}

fn is_documentation(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    path.ends_with(".md")
        || path.ends_with(".txt")
        || path.ends_with(".rst")
        || path.starts_with("docs/")
}

fn is_risky_path(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    [
        "cargo.toml",
        "cargo.lock",
        "package.json",
        "package-lock.json",
        "migration",
        "auth",
        "security",
        "permission",
        "persistence",
        "database",
    ]
    .iter()
    .any(|part| path.contains(part))
}

fn profile_for(effort: ReasoningEffort, traits: TaskTraits) -> AgentThinkingProfile {
    let (read_budget, plan, verify, recovery, checkpoint_min_changes, checkpoint_reviews) =
        match effort {
            ReasoningEffort::Off => (1, false, false, 0, usize::MAX, 0),
            ReasoningEffort::Low => (2, false, false, 1, usize::MAX, 0),
            ReasoningEffort::Medium | ReasoningEffort::Auto => (4, true, true, 2, 2, 1),
            ReasoningEffort::High => (8, true, true, 3, 1, 3),
        };
    AgentThinkingProfile {
        model_effort: effort,
        read_budget,
        require_plan: plan
            && (traits.changes_files || traits.diagnoses_failure || traits.reviews_existing_work),
        require_verification: traits.verification_requested || (verify && traits.changes_files),
        max_recovery_attempts: recovery,
        checkpoint_min_changes,
        checkpoint_reviews,
    }
}
