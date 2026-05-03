use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashSet;

use crate::system::domain::{
    ActionDescriptor, ApprovalCategory, ConversationItem, ItemKind, ToolCall, UserInput,
};

// ─── is_read_only_tool ────────────────────────────────────────────────────────

/// Returns `true` for tools that only read or explore without side-effects.
///
/// Used by the read-only exploration guard in the executor loop to detect when
/// the model is stuck reading files indefinitely without taking any action.
/// Any tool *not* listed here is considered an "action" tool that resets the
/// consecutive-read-only counter (write, bash, directory mutation, sub-agents).
pub(crate) fn is_read_only_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file"
            | "list_dir"
            | "grep_search"
            | "search_web"
            | "read_url"
            | "read_browser_page"
            | "view_file"
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnIntent {
    Edit,
    Debug,
    Review,
    Answer,
    Inventory,
    Unknown,
}

pub(crate) fn classify_turn_intent(inputs: &[UserInput]) -> TurnIntent {
    let text = inputs
        .iter()
        .filter_map(|i| i.text.as_ref().map(|t| t.text.as_str()))
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();

    if text.is_empty() {
        return TurnIntent::Unknown;
    }
    if text.contains("review") || text.contains("audit") || text.contains("regression") {
        return TurnIntent::Review;
    }
    if text.contains("list ") || text.contains("inventory") || text.contains("contents of") {
        return TurnIntent::Inventory;
    }
    if text.contains("fix")
        || text.contains("implement")
        || text.contains("change")
        || text.contains("update")
        || text.contains("refactor")
        || text.contains("remove")
    {
        return TurnIntent::Edit;
    }
    if text.contains("why")
        || text.contains("what's wrong")
        || text.contains("what is wrong")
        || text.contains("debug")
        || text.contains("loop")
    {
        return TurnIntent::Debug;
    }
    if text.contains("explain") || text.contains("what") || text.contains("how") {
        return TurnIntent::Answer;
    }
    TurnIntent::Unknown
}

/// Returns true when the user explicitly asks for low-level repo internals in
/// the map (e.g. binary files, `.git`, or full file tree).
pub(crate) fn wants_verbose_repo_map(inputs: &[UserInput]) -> bool {
    let text = inputs
        .iter()
        .filter_map(|i| i.text.as_ref().map(|t| t.text.as_str()))
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    if text.is_empty() {
        return false;
    }
    let mentions_binary = text.contains("binary")
        || text.contains("bin files")
        || text.contains("non-text")
        || text.contains("compiled artifacts");
    let mentions_git_internal = text.contains(".git")
        || text.contains("git objects")
        || text.contains("git internals")
        || text.contains("object store");
    let mentions_full_tree = text.contains("full file tree")
        || text.contains("entire tree")
        || text.contains("everything in the repo map")
        || text.contains("all files including");
    mentions_binary || mentions_git_internal || mentions_full_tree
}

pub(crate) fn validate_tool_call(call: &ToolCall) -> Option<String> {
    let missing = |field: &'static str| format!("missing required argument `{field}`");
    let non_empty_string = |name: &'static str| -> Option<Cow<'static, str>> {
        match call.arguments.get(name).and_then(Value::as_str) {
            Some(v) if !v.trim().is_empty() => None,
            _ => Some(Cow::Owned(missing(name))),
        }
    };
    match call.tool_name.as_str() {
        "grep_search" => non_empty_string("pattern").map(|m| m.into_owned()),
        "read_file" | "list_dir" | "view_file" => non_empty_string("path").map(|m| m.into_owned()),
        "bash_exec" => non_empty_string("command").map(|m| m.into_owned()),
        "shell_exec" => {
            if call.arguments.get("argv").is_none() {
                Some(missing("argv"))
            } else {
                None
            }
        }
        "patch_file" => {
            if call.arguments.get("path").and_then(Value::as_str).is_none() {
                return Some(missing("path"));
            }
            if call
                .arguments
                .get("start_line")
                .and_then(Value::as_u64)
                .is_none()
            {
                return Some(missing("start_line"));
            }
            if call
                .arguments
                .get("end_line")
                .and_then(Value::as_u64)
                .is_none()
            {
                return Some(missing("end_line"));
            }
            if call
                .arguments
                .get("content")
                .and_then(Value::as_str)
                .is_none()
            {
                return Some(missing("content"));
            }
            None
        }
        _ => None,
    }
}

/// Semantic key for a read-only call. Lets the dedup layer reason about
/// subsumption (a whole-file read covers any subsequent partial read of the
/// same path) rather than just exact-string equality.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ReadKey {
    WholeFile(String),
    PartialFile {
        path: String,
        offset: u64,
        limit: u64,
    },
    Grep {
        path: String,
        pattern: String,
    },
    ListDir {
        path: String,
        depth: u64,
    },
}

pub(crate) fn read_signature(call: &ToolCall) -> Option<ReadKey> {
    match call.tool_name.as_str() {
        "read_file" => {
            let path = normalize_path(call.arguments.get("path")?.as_str()?);
            let offset = call
                .arguments
                .get("offset")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let limit = call
                .arguments
                .get("limit")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if offset == 0 && limit == 0 {
                Some(ReadKey::WholeFile(path))
            } else {
                Some(ReadKey::PartialFile {
                    path,
                    offset,
                    limit,
                })
            }
        }
        "grep_search" => {
            let pattern = call.arguments.get("pattern")?.as_str()?.to_string();
            let path = call
                .arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(ReadKey::Grep {
                path: normalize_path(path),
                pattern,
            })
        }
        "list_dir" => {
            let path = normalize_path(call.arguments.get("path")?.as_str()?);
            let depth = call
                .arguments
                .get("depth")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            Some(ReadKey::ListDir { path, depth })
        }
        _ => None,
    }
}

/// True if the key has already been satisfied by a prior read this turn —
/// either by exact match or by a whole-file read subsuming a partial read.
pub(crate) fn is_duplicate_read(key: &ReadKey, seen: &HashSet<ReadKey>) -> bool {
    if seen.contains(key) {
        return true;
    }
    if let ReadKey::PartialFile { path, .. } = key {
        if seen.contains(&ReadKey::WholeFile(path.clone())) {
            return true;
        }
    }
    false
}

/// Stable signature for cross-round repetition detection. Covers ALL tool
/// calls (not just reads) — used to spot the model issuing the same call
/// repeatedly across rounds, which is a stronger signal than within-round
/// duplicate-read dedup.
pub(crate) fn cross_round_signature(call: &ToolCall) -> String {
    format!(
        "{}:{}",
        call.tool_name,
        serde_json::to_string(&call.arguments).unwrap_or_default()
    )
}

/// Lightweight path normalization — strips `./` prefix, trailing `/`, and
/// collapses runs of `/`. Does NOT touch the filesystem (no canonicalize),
/// so it's safe for paths that don't exist.
fn normalize_path(p: &str) -> String {
    let trimmed = p.trim();
    let stripped = trimmed
        .strip_prefix("./")
        .unwrap_or(trimmed)
        .trim_end_matches('/');

    let mut out = String::with_capacity(stripped.len());
    let mut prev_slash = false;
    for ch in stripped.chars() {
        if ch == '/' {
            if prev_slash {
                continue;
            }
            prev_slash = true;
        } else {
            prev_slash = false;
        }
        out.push(ch);
    }
    if out.is_empty() {
        ".".into()
    } else {
        out
    }
}

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
pub(crate) fn partition_into_batches<F>(
    calls: &[ToolCall],
    is_parallel: F,
) -> Vec<Vec<(usize, ToolCall)>>
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

// ─── action_for_tool_call (module-private helper) ────────────────────────────

pub(crate) fn action_for_tool_call(call: &ToolCall, cwd: &str) -> ActionDescriptor {
    let category = match call.tool_name.as_str() {
        "bash_exec" | "shell_exec" => ApprovalCategory::SandboxEscalation,
        "write_file" | "patch_file" | "create_directory" | "remove_path" | "copy_path" => {
            ApprovalCategory::FileChange
        }
        _ => ApprovalCategory::Other,
    };
    let paths = match call.tool_name.as_str() {
        "write_file" | "patch_file" | "create_directory" | "remove_path" => call
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

pub(crate) fn user_requested_verification(inputs: &[UserInput]) -> bool {
    let text = inputs
        .iter()
        .filter_map(|i| i.text.as_ref().map(|t| t.text.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    let normalized = normalize_for_detection(&text);

    const VERIFY_PATTERNS: &[&str] = &[
        "test it",
        "test this",
        "run test",
        "run the test",
        "run tests",
        "run the tests",
        "verify",
        "validate",
        "make sure it passes",
        "cargo test",
        "cargo check",
        "pytest",
        "npm test",
    ];

    contains_any(&normalized, VERIFY_PATTERNS)
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

    const DELEGATED_ACTION_PATTERNS: &[&str] = &[
        "can you check why",
        "could you check why",
        "please check why",
        "check why",
        "can you check what needs",
        "could you check what needs",
        "please check what needs",
        "check what needs",
        "check the code",
        "check this project",
        "check this repo",
        "can you inspect",
        "could you inspect",
        "please inspect",
        "can you look at",
        "could you look at",
        "please look at",
        "can you debug",
        "could you debug",
        "please debug",
        "can you fix",
        "could you fix",
        "please fix",
        "fix it",
        "can you change",
        "could you change",
        "please change",
        "change it",
        "can you update",
        "could you update",
        "please update",
        "can you implement",
        "could you implement",
        "please implement",
        "test it",
        "run the tests",
    ];
    if contains_any(&normalized, DELEGATED_ACTION_PATTERNS) {
        return true;
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
    if contains_any(&normalized, NON_AGENTIC_PATTERNS) {
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

    contains_any(&normalized, AGENTIC_PATTERNS)
}

fn contains_any(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|pattern| text.contains(pattern))
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
    text.trim()
        .to_ascii_lowercase()
        .replace(['’', '‘'], "'")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ─── Smart nudge context extraction ──────────────────────────────────────────

/// Extract the file paths that the model has recently read, by scanning
/// the most recent ToolCall items for read_file invocations.
/// Returns deduplicated paths in most-recent-first order, capped at `limit`.
pub(crate) fn recently_read_paths(items: &[ConversationItem], limit: usize) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }
    let mut paths = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for item in items.iter().rev() {
        if item.kind != ItemKind::ToolCall {
            continue;
        }
        let name = item
            .payload
            .get("tool_name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if name != "read_file" {
            continue;
        }
        if let Some(path) = item
            .payload
            .get("arguments")
            .and_then(|a| a.get("path"))
            .and_then(serde_json::Value::as_str)
        {
            if seen.insert(path.to_string()) {
                paths.push(path.to_string());
            }
        }
        if paths.len() >= limit {
            break;
        }
    }
    paths
}

/// Short directive that tells the model what kind of turn this is, so it
/// can shape its exploration depth accordingly. Computed once per turn from
/// the user's first message and re-injected each iteration.
pub(crate) fn intent_directive(intent: TurnIntent) -> Option<String> {
    match intent {
        TurnIntent::Edit => Some(
            "Turn intent: EDIT. Locate the relevant file, edit it with patch_file, verify if \
             relevant, and stop. Minimize exploration."
                .into(),
        ),
        TurnIntent::Debug => Some(
            "Turn intent: DEBUG. Reproduce or pinpoint the issue, identify the root cause, and \
             report. Only edit if explicitly asked."
                .into(),
        ),
        TurnIntent::Review => Some(
            "Turn intent: REVIEW. Read the relevant code and answer. Do not edit unless asked."
                .into(),
        ),
        TurnIntent::Answer => Some(
            "Turn intent: ANSWER. Read only what you need to answer the question, then answer \
             concisely. Do not edit."
                .into(),
        ),
        TurnIntent::Inventory => Some(
            "Turn intent: INVENTORY. List or summarize the requested items, then stop. Do not edit."
                .into(),
        ),
        TurnIntent::Unknown => None,
    }
}

/// Tells the model which files are already in its context this turn so it
/// stops re-reading them. Empty list returns None — no point injecting an
/// empty list.
pub(crate) fn already_read_directive(paths: &[String]) -> Option<String> {
    if paths.is_empty() {
        return None;
    }
    let list = paths.join(", ");
    Some(format!(
        "Files already read in this turn (refer back to prior tool results — do not re-read): \
         {list}"
    ))
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

    #[test]
    fn retries_delegated_diagnostic_question_with_no_tool_call() {
        let items = vec![user_item(
            "can you check why that is and how it can be fixed?",
        )];

        assert!(should_retry_no_tool_completion(
            "The issue is probably in the executor.",
            &items,
            0
        ));
    }

    #[test]
    fn retries_delegated_change_and_test_request_with_no_tool_call() {
        let items = vec![user_item(
            "check what needs to be changed, change it and test it",
        )];

        assert!(should_retry_no_tool_completion(
            "I would inspect the relevant files and run tests.",
            &items,
            0
        ));
    }

    #[test]
    fn still_allows_first_person_how_to_questions_without_tools() {
        let items = vec![user_item("how can I fix the failing build?")];

        assert!(!should_retry_no_tool_completion(
            "Start by running the failing test with verbose output.",
            &items,
            0
        ));
    }

    #[test]
    fn detects_explicit_verification_requests() {
        let input = UserInput::from_text("change it and test it");

        assert!(user_requested_verification(&[input]));
    }
}
