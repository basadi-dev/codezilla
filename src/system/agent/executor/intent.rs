use crate::system::domain::{ReasoningEffort, UserInput};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnIntent {
    Edit,
    Debug,
    Review,
    Answer,
    Inventory,
    Unknown,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct TaskTraits {
    pub(super) changes_files: bool,
    pub(super) diagnoses_failure: bool,
    pub(super) reviews_existing_work: bool,
    pub(super) high_risk: bool,
    pub(super) verification_requested: bool,
}

pub(super) fn derive_task_traits(inputs: &[UserInput]) -> TaskTraits {
    let text = input_text(inputs);
    let contains_any = |terms: &[&str]| terms.iter().any(|term| text.contains(term));
    TaskTraits {
        changes_files: contains_any(&[
            "fix",
            "implement",
            "change",
            "update",
            "refactor",
            "remove",
            "add ",
            "create",
            "improve",
            "clean up",
            "cleanup",
        ]),
        diagnoses_failure: contains_any(&[
            "debug",
            "failing",
            "failure",
            "broken",
            "regression",
            "what's wrong",
            "what is wrong",
        ]),
        reviews_existing_work: contains_any(&["review", "audit", "inspect", "assess"]),
        high_risk: contains_any(&[
            "security",
            "permission",
            "authentication",
            "authorization",
            "database",
            "migration",
            "persistence",
            "production",
            "payment",
        ]),
        verification_requested: contains_any(&[
            "test", "verify", "check", "lint", "clippy", "build", "compile", "format",
        ]),
    }
}

pub(super) fn classify_turn_intent(inputs: &[UserInput]) -> TurnIntent {
    let text = input_text(inputs);
    if text.is_empty() {
        return TurnIntent::Unknown;
    }
    if text.contains("review") || text.contains("audit") || text.contains("regression") {
        return TurnIntent::Review;
    }
    if text.contains("list ") || text.contains("inventory") || text.contains("contents of") {
        return TurnIntent::Inventory;
    }
    if ["fix", "implement", "change", "update", "refactor", "remove"]
        .iter()
        .any(|word| text.contains(word))
    {
        return TurnIntent::Edit;
    }
    if ["why", "what's wrong", "what is wrong", "debug", "loop"]
        .iter()
        .any(|phrase| text.contains(phrase))
    {
        return TurnIntent::Debug;
    }
    if ["explain", "what", "how"]
        .iter()
        .any(|word| text.contains(word))
    {
        return TurnIntent::Answer;
    }
    TurnIntent::Unknown
}

pub(super) fn wants_verbose_repo_map(inputs: &[UserInput]) -> bool {
    let text = input_text(inputs);
    [
        "binary",
        "bin files",
        "non-text",
        "compiled artifacts",
        ".git",
        "git objects",
        "git internals",
        "object store",
        "full file tree",
        "entire tree",
        "everything in the repo map",
        "all files including",
    ]
    .iter()
    .any(|phrase| text.contains(phrase))
}

/// Explicit user settings win; `auto` adapts to the current task.
pub(super) fn intent_to_reasoning_effort(
    intent: TurnIntent,
    user_setting: ReasoningEffort,
) -> ReasoningEffort {
    if user_setting != ReasoningEffort::Auto {
        return user_setting;
    }
    match intent {
        TurnIntent::Inventory => ReasoningEffort::Off,
        TurnIntent::Answer => ReasoningEffort::Low,
        TurnIntent::Edit => ReasoningEffort::Medium,
        TurnIntent::Debug | TurnIntent::Review => ReasoningEffort::High,
        TurnIntent::Unknown => ReasoningEffort::Auto,
    }
}

pub(super) fn thinking_instruction(reasoning_effort: Option<&str>) -> Option<String> {
    match reasoning_effort {
        None | Some("off") | Some("auto") => None,
        Some("low") => Some(
            "Think briefly before responding. A short internal reasoning pass is enough.".into(),
        ),
        Some("medium") => Some(
            "Think through this carefully, step by step, before giving your final answer.".into(),
        ),
        Some("high") => Some(
            "Think extra hard. Reason deeply and thoroughly, considering multiple angles and edge \
             cases, before providing your answer."
                .into(),
        ),
        Some(other) => Some(format!(
            "Reasoning effort: {other}. Think carefully before responding."
        )),
    }
}

fn input_text(inputs: &[UserInput]) -> String {
    inputs
        .iter()
        .filter_map(|input| input.text.as_ref().map(|text| text.text.as_str()))
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}
