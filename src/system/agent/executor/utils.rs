use serde_json::Value;

use crate::system::domain::{
    ActionDescriptor, ApprovalCategory, ConversationItem, ItemKind, ToolCall,
};

// ─── partition_into_batches ───────────────────────────────────────────────────

/// Split an ordered slice of `ToolCall`s into sequential execution batches.
///
/// Consecutive calls that are all parallel-safe are grouped into a single batch
/// so they can be executed with `join_all`. Any call that is *not* parallel-safe
/// gets its own single-element batch and acts as a serialisation barrier.
///
/// Examples:
///   [read, read, read]          → [(read, read, read)]
///   [read, write, read]         → [(read), (write), (read)]
///   [read, read, write, read]   → [(read, read), (write), (read)]
///   [bash, bash]                → [(bash), (bash)]
pub(crate) fn partition_into_batches<F>(calls: &[ToolCall], is_parallel: F) -> Vec<Vec<(usize, ToolCall)>>
where
    F: Fn(&str) -> bool,
{
    let mut batches: Vec<Vec<(usize, ToolCall)>> = Vec::new();
    let mut current: Vec<(usize, ToolCall)> = Vec::new();

    for (i, call) in calls.iter().enumerate() {
        if is_parallel(&call.tool_name) {
            current.push((i, call.clone()));
        } else {
            if !current.is_empty() {
                batches.push(std::mem::take(&mut current));
            }
            batches.push(vec![(i, call.clone())]);
        }
    }

    if !current.is_empty() {
        batches.push(current);
    }

    batches
}

pub(crate) fn thinking_instruction(reasoning_effort: Option<&str>) -> Option<String> {
    match reasoning_effort {
        None | Some("off") => None,
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

// ─── action_for_tool_call (module-private helper) ────────────────────────────

pub(crate) fn action_for_tool_call(call: &ToolCall, cwd: &str) -> ActionDescriptor {
    let category = match call.tool_name.as_str() {
        "bash_exec" | "shell_exec" => ApprovalCategory::SandboxEscalation,
        "write_file" | "create_directory" | "remove_path" | "copy_path" => {
            ApprovalCategory::FileChange
        }
        _ => ApprovalCategory::Other,
    };
    let paths = match call.tool_name.as_str() {
        "write_file" | "create_directory" | "remove_path" => call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .map(|p| vec![p.to_string()])
            .unwrap_or_else(|| vec![cwd.into()]),
        "copy_path" => vec![
            call.arguments
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or(cwd)
                .to_string(),
            call.arguments
                .get("target")
                .and_then(Value::as_str)
                .unwrap_or(cwd)
                .to_string(),
        ],
        _ => vec![cwd.into()],
    };
    let command = match call.tool_name.as_str() {
        "bash_exec" => call
            .arguments
            .get("command")
            .and_then(Value::as_str)
            .map(|command| vec!["bash".to_string(), "-c".to_string(), command.to_string()]),
        "shell_exec" => call
            .arguments
            .get("argv")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
            .or_else(|| {
                call.arguments
                    .get("argv")
                    .and_then(Value::as_str)
                    .map(|argv| vec![argv.to_string()])
            }),
        _ => None,
    };
    ActionDescriptor {
        action_type: call.tool_name.clone(),
        command,
        paths,
        domains: Vec::new(),
        category,
    }
}

// ─── derive_thread_title ──────────────────────────────────────────────────────

/// Build a short, human-readable title from the first user message.
/// Takes the first non-empty line, strips leading `/` commands, and
/// truncates to 72 chars with an ellipsis when needed.
pub(crate) fn derive_thread_title(text: &str) -> String {
    const MAX: usize = 72;
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('/'))
        .unwrap_or_else(|| text.lines().next().unwrap_or("").trim());

    if line.is_empty() {
        return "Untitled thread".into();
    }

    let chars: Vec<char> = line.chars().collect();
    if chars.len() <= MAX {
        line.to_string()
    } else {
        chars[..MAX].iter().collect::<String>() + "…"
    }
}

// ─── Shell-operator safety net ────────────────────────────────────────────────

/// Shell operators and patterns that only make sense inside a shell.
/// Any of these appearing as a standalone argv token is a dead giveaway that
/// the model intended shell semantics but called `shell_exec` by mistake.
const SHELL_OPERATOR_TOKENS: &[&str] = &[
    "|",
    "||",
    "&&",
    ";",
    "&",
    ">",
    ">>",
    "<",
    "<<",
    "2>&1",
    "2>/dev/null",
    "1>/dev/null",
    "1>&2",
    "2>>",
    "1>>",
];

/// Inspect a `shell_exec` ToolCall for shell operators in its argv.
/// If any are found, rewrite the call as a `bash_exec` command string so that
/// the operators are interpreted correctly by bash.
///
/// This is the runtime safety net — it catches model mistakes that slipped
/// past the system prompt and schema guidance.
pub(crate) fn promote_to_bash_if_needed(call: ToolCall) -> ToolCall {
    if call.tool_name != "shell_exec" {
        return call;
    }

    // Only inspect array argv; string argv goes through simple_tokenize in
    // ShellToolProvider which also won't support shell operators, so promote
    // string argv too if it looks like a shell command.
    let needs_promotion = if let Some(arr) = call.arguments.get("argv").and_then(|v| v.as_array()) {
        arr.iter().filter_map(|v| v.as_str()).any(|tok| {
            SHELL_OPERATOR_TOKENS.contains(&tok)
                || (tok.contains('*') && !tok.starts_with("--")) // glob (not a flag)
                || tok.starts_with("2>")
                || tok.starts_with("1>")
                || tok == "?"
        })
    } else if let Some(s) = call.arguments.get("argv").and_then(|v| v.as_str()) {
        // String argv — check if it looks like it has shell operators
        SHELL_OPERATOR_TOKENS.iter().any(|op| {
            // Match operator as a whole word, not a substring of a flag
            s.split_whitespace().any(|tok| tok == *op)
        }) || s.contains("2>&1")
            || s.contains("| ")
            || s.contains(" |")
    } else {
        false
    };

    if !needs_promotion {
        return call;
    }

    // Build the shell command string by joining argv tokens
    let shell_cmd = if let Some(arr) = call.arguments.get("argv").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else if let Some(s) = call.arguments.get("argv").and_then(|v| v.as_str()) {
        s.to_string()
    } else {
        return call; // nothing to do
    };

    tracing::warn!(
        tool_call_id = %call.tool_call_id,
        shell_cmd = %shell_cmd,
        "shell_exec contained shell operators — auto-promoting to bash_exec"
    );

    // Build a new arguments object: replace argv with command, keep cwd/env
    let mut new_args = call.arguments.clone();
    if let Some(obj) = new_args.as_object_mut() {
        obj.remove("argv");
        obj.insert("command".to_string(), serde_json::Value::String(shell_cmd));
    }

    ToolCall {
        tool_name: "bash_exec".into(),
        tool_call_id: call.tool_call_id,
        provider_kind: call.provider_kind,
        arguments: new_args,
    }
}

pub(crate) fn should_retry_no_tool_completion(
    assistant_text: &str,
    items: &[ConversationItem],
    completed_tool_rounds: usize,
) -> bool {
    if looks_like_unexecuted_tool_intent(assistant_text) {
        return true;
    }

    // After at least one actual tool round, a no-tool model response is normally
    // the final answer. Only the explicit intent check above should override it.
    if completed_tool_rounds > 0 {
        return false;
    }

    latest_user_text(items)
        .map(is_agentic_user_request)
        .unwrap_or(false)
}

fn latest_user_text(items: &[ConversationItem]) -> Option<&str> {
    items
        .iter()
        .rev()
        .find(|item| item.kind == ItemKind::UserMessage)
        .and_then(|item| item.payload.get("text"))
        .and_then(serde_json::Value::as_str)
}

fn is_agentic_user_request(text: &str) -> bool {
    let normalized = normalize_for_detection(text);

    if normalized.is_empty() {
        return false;
    }

    const NON_AGENTIC_PATTERNS: &[&str] = &[
        "how can i ",
        "how do i ",
        "what is ",
        "what's ",
        "why ",
        "explain ",
        "describe ",
        "is the following correct",
        "what would ",
        "should i ",
        "which approach",
    ];
    if NON_AGENTIC_PATTERNS
        .iter()
        .any(|pattern| normalized.contains(pattern))
    {
        return false;
    }

    const AGENTIC_PATTERNS: &[&str] = &[
        "go ahead",
        "implement",
        "fix",
        "change",
        "update",
        "modify",
        "edit",
        "patch",
        "refactor",
        "add ",
        "remove ",
        "delete ",
        "create ",
        "write ",
        "run ",
        "test ",
        "debug",
        "inspect",
        "check the code",
        "look at",
        "open ",
        "read ",
        "commit",
        "push",
        "make it",
    ];

    AGENTIC_PATTERNS
        .iter()
        .any(|pattern| normalized.contains(pattern))
}

fn looks_like_unexecuted_tool_intent(text: &str) -> bool {
    let normalized = normalize_for_detection(text);

    if normalized.is_empty() {
        return false;
    }

    const INTENT_PATTERNS: &[&str] = &[
        "let me ",
        "i'll ",
        "i will ",
        "i'm going to ",
        "i am going to ",
        "i need to inspect",
        "i need to read",
        "i need to check",
        "i need to run",
        "i need to update",
        "i need to modify",
        "i need to edit",
        "i need to open",
        "i need to search",
        "i'll start",
        "let's inspect",
        "let's read",
        "let's check",
        "let's run",
        "let's update",
        "let's modify",
        "let's edit",
        "let's open",
        "let's search",
    ];
    const TOOL_WORDS: &[&str] = &[
        "read", "inspect", "check", "run", "execute", "open", "search", "grep", "list", "write",
        "edit", "update", "modify", "patch", "test", "build", "file", ".rs", ".ts", ".tsx", ".js",
        ".json", ".toml", ".yaml", ".yml",
    ];

    let has_intent = INTENT_PATTERNS
        .iter()
        .any(|pattern| normalized.contains(pattern));
    let has_tool_word = TOOL_WORDS.iter().any(|word| normalized.contains(word));

    has_intent && has_tool_word
}

fn normalize_for_detection(text: &str) -> String {
    text
        .trim()
        .to_ascii_lowercase()
        .replace(['’', '‘'], "'")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ─── Degenerate-output detection ──────────────────────────────────────────────
//
// Some local/quantized models enter token-generation loops, repeating the same
// pattern indefinitely. These helpers detect that condition early.
//
// All comparisons are done on **bytes** (`as_bytes()`) to avoid UTF-8 boundary
// panics. If the same Unicode text repeats, the same bytes also repeat, so the
// detection is equally correct at the byte level.

/// Returns `true` when the *tail* of the text contains a byte run of length
/// ≥ `MIN_PATTERN_LEN` that repeats at least `MIN_REPEATS` times consecutively.
///
/// Only examines the last `WINDOW` bytes to stay O(1) per streaming delta.
pub(crate) fn is_degenerate_repetition(text: &str) -> bool {
    const MIN_PATTERN_LEN: usize = 40;
    const MIN_REPEATS: usize = 5;
    const WINDOW: usize = 4_000;

    let bytes = text.as_bytes();
    if bytes.len() < MIN_PATTERN_LEN * MIN_REPEATS {
        return false;
    }

    let tail = if bytes.len() > WINDOW {
        &bytes[bytes.len() - WINDOW..]
    } else {
        bytes
    };

    // Try candidate pattern lengths 40, 50, 60 … 200 bytes.
    let max_pat = 200.min(tail.len() / MIN_REPEATS);
    for pat_len in (MIN_PATTERN_LEN..=max_pat).step_by(10) {
        let pattern = &tail[tail.len() - pat_len..];
        let mut count = 0usize;
        let mut pos = tail.len() - pat_len;
        while pos >= pat_len {
            pos -= pat_len;
            if &tail[pos..pos + pat_len] == pattern {
                count += 1;
            } else {
                break;
            }
        }
        if count >= MIN_REPEATS {
            return true;
        }
    }
    false
}

/// Find the **byte** offset (into `text`) where the repetitive pattern starts,
/// so we can `truncate()` to the clean prefix. Returns `None` if no repetition
/// is detected. The returned offset is always on a UTF-8 char boundary because
/// we walk backward to the nearest boundary before returning.
pub(crate) fn find_repetition_start(text: &str) -> Option<usize> {
    const MIN_PATTERN_LEN: usize = 40;
    const MIN_REPEATS: usize = 5;
    const WINDOW: usize = 4_000;

    let bytes = text.as_bytes();
    if bytes.len() < MIN_PATTERN_LEN * MIN_REPEATS {
        return None;
    }

    let search_start = bytes.len().saturating_sub(WINDOW);
    let tail = &bytes[search_start..];

    let max_pat = 200.min(tail.len() / MIN_REPEATS);
    for pat_len in (MIN_PATTERN_LEN..=max_pat).step_by(10) {
        let pattern = &tail[tail.len() - pat_len..];
        let mut earliest = tail.len() - pat_len;
        let mut count = 0usize;
        let mut pos = tail.len() - pat_len;
        while pos >= pat_len {
            pos -= pat_len;
            if &tail[pos..pos + pat_len] == pattern {
                count += 1;
                earliest = pos;
            } else {
                break;
            }
        }
        if count >= MIN_REPEATS {
            // Walk the raw offset back to the nearest valid UTF-8 char boundary.
            let raw = search_start + earliest;
            let boundary = (0..=raw)
                .rev()
                .find(|&i| text.is_char_boundary(i))
                .unwrap_or(0);
            return Some(boundary);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user_item(text: &str) -> ConversationItem {
        ConversationItem {
            item_id: "item_test".into(),
            thread_id: "thread_test".into(),
            turn_id: "turn_test".into(),
            created_at: 0,
            kind: ItemKind::UserMessage,
            payload: json!({ "text": text }),
        }
    }

    #[test]
    fn retries_agentic_user_request_with_no_tool_call() {
        let items = vec![user_item("go ahead and implement the ideal solution")];

        assert!(should_retry_no_tool_completion("", &items, 0));
    }

    #[test]
    fn allows_non_agentic_question_to_finish_without_tools() {
        let items = vec![user_item("what's the ideal and most suitable solution")];

        assert!(!should_retry_no_tool_completion(
            "Use a bounded executor retry.",
            &items,
            0
        ));
    }

    #[test]
    fn retries_explicit_tool_intent_even_after_tool_round() {
        let items = vec![user_item("fix the failing build")];

        assert!(should_retry_no_tool_completion(
            "I'll run cargo check now.",
            &items,
            1
        ));
    }

    #[test]
    fn allows_final_answer_after_tool_round() {
        let items = vec![user_item("fix the failing build")];

        assert!(!should_retry_no_tool_completion(
            "Implemented the fix and cargo check passes.",
            &items,
            1
        ));
    }
}
