use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashSet;
use std::time::Duration;

use crate::system::domain::{
    ActionDescriptor, ApprovalCategory, ConversationItem, ItemKind, ToolCall, UserInput,
};

use super::intent::TurnIntent;

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

/// Build a deterministic coding-state block that is pinned into every model
/// request. This is intentionally based on local state and recent tool results,
/// not on a lossy conversation summary.
///
/// Async because it snapshots git state via `tokio::process` (bounded by
/// [`GIT_SNAPSHOT_TIMEOUT`]) so a slow repository never blocks a worker thread.
pub(crate) async fn pinned_coding_context(cwd: &str, items: &[ConversationItem]) -> Option<String> {
    let (git_status, git_diff_stat) = tokio::join!(
        run_git(cwd, &["status", "--short"], 4000),
        run_git(cwd, &["diff", "--stat"], 4000),
    );
    build_pinned_context(items, git_status, git_diff_stat)
}

/// Pure assembly of the pinned block from conversation items plus an optional
/// git snapshot. Split from [`pinned_coding_context`] so it is unit-testable
/// without spawning processes.
fn build_pinned_context(
    items: &[ConversationItem],
    git_status: Option<String>,
    git_diff_stat: Option<String>,
) -> Option<String> {
    let latest_request = latest_user_request(items);
    let files = recent_file_paths(items, 12);
    let commands = recent_commands(items, 6);
    let latest_test = latest_test_or_diagnostic_output(items);

    if latest_request.is_none()
        && files.is_empty()
        && commands.is_empty()
        && latest_test.is_none()
        && git_status.is_none()
        && git_diff_stat.is_none()
    {
        return None;
    }

    let mut out = Vec::new();
    out.push("## Pinned Coding Context".to_string());
    out.push(
        "Repository state is authoritative. Treat older chat or compaction summaries as stale when they conflict with current files, git diff, or test output."
            .to_string(),
    );

    if let Some(request) = latest_request {
        out.push(format!("Latest user request: {}", one_line(&request, 900)));
    }

    if !files.is_empty() {
        out.push(format!("Current/recent files: {}", files.join(", ")));
    }

    if !commands.is_empty() {
        out.push("Recent commands:".to_string());
        for command in commands {
            out.push(format!("- `{}`", one_line(&command, 220)));
        }
    }

    if let Some(status) = git_status.filter(|s| !s.trim().is_empty()) {
        out.push("Current git status (`git status --short`):".to_string());
        // Trim only the end: the leading porcelain column (" M foo") is data.
        out.push(fenced("text", status.trim_end()));
    }

    if let Some(stat) = git_diff_stat.filter(|s| !s.trim().is_empty()) {
        out.push("Current git diff stat (`git diff --stat`):".to_string());
        out.push(fenced("text", stat.trim()));
    }

    if let Some(test) = latest_test {
        out.push("Latest test/build/diagnostic output:".to_string());
        out.push(fenced("text", test.trim()));
    }

    Some(out.join("\n"))
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
        "read_file" | "list_dir" | "view_file" => {
            if let Some(reason) = non_empty_string("path") {
                return Some(reason.into_owned());
            }
            // Catch URL-as-path — a common model mistake.
            if let Some(path) = call.arguments.get("path").and_then(Value::as_str) {
                if path.starts_with("http://") || path.starts_with("https://") {
                    return Some(format!(
                        "`path` must be a local filesystem path, not a URL. \
                         Use `web_fetch` to read URLs. Got: {}",
                        &path[..path.len().min(80)]
                    ));
                }
            }
            None
        }
        "bash_exec" => {
            if let Some(reason) = non_empty_string("command") {
                return Some(reason.into_owned());
            }
            // Detect JSON-as-command — models sometimes emit the entire tool-call
            // JSON object as the bash command string instead of an actual command.
            if let Some(cmd) = call.arguments.get("command").and_then(Value::as_str) {
                let trimmed = cmd.trim();
                if (trimmed.starts_with('{') && trimmed.contains("\"command\""))
                    || (trimmed.starts_with('{') && trimmed.contains("\"name\""))
                {
                    return Some(
                        "`command` contains a JSON object instead of a shell command string. \
                         Provide the actual shell command, e.g. \"cargo build --release\""
                            .into(),
                    );
                }
            }
            None
        }
        "shell_exec" => {
            if call.arguments.get("argv").is_none() {
                Some(missing("argv"))
            } else {
                None
            }
        }
        "spawn_agent" => {
            if call
                .arguments
                .get("prompt")
                .and_then(Value::as_str)
                .map(|v| !v.trim().is_empty())
                != Some(true)
            {
                return Some(missing("prompt"));
            }
            if let Some(write_paths) = call.arguments.get("write_paths") {
                let Some(items) = write_paths.as_array() else {
                    return Some("`write_paths` must be an array of strings".into());
                };
                if items
                    .iter()
                    .any(|v| v.as_str().map(|s| s.trim().is_empty()).unwrap_or(true))
                {
                    return Some("`write_paths` entries must be non-empty strings".into());
                }
            }
            None
        }
        "run_agent_team" => {
            if let Some(reason) = non_empty_string("objective") {
                return Some(reason.into_owned());
            }
            let Some(assignments) = call.arguments.get("assignments").and_then(Value::as_array)
            else {
                return Some("`assignments` must be a non-empty array".into());
            };
            if assignments.is_empty() {
                return Some("`assignments` must contain at least one member".into());
            }
            if assignments.iter().any(|assignment| {
                assignment
                    .get("role")
                    .and_then(Value::as_str)
                    .is_none_or(|value| value.trim().is_empty())
                    || assignment
                        .get("objective")
                        .and_then(Value::as_str)
                        .is_none_or(|value| value.trim().is_empty())
            }) {
                return Some(
                    "every assignment must contain non-empty `role` and `objective` strings".into(),
                );
            }
            None
        }
        "write_file" => {
            if let Some(reason) = non_empty_string("path") {
                return Some(reason.into_owned());
            }
            // `content` can be empty (creating an empty file is valid) but must exist.
            if call.arguments.get("content").is_none() {
                return Some(missing("content"));
            }
            None
        }
        "patch_file" => {
            if call.arguments.get("path").and_then(Value::as_str).is_none() {
                return Some(missing("path"));
            }
            let start = call.arguments.get("start_line").and_then(Value::as_u64);
            let end = call.arguments.get("end_line").and_then(Value::as_u64);
            if start.is_none() {
                return Some(missing("start_line"));
            }
            if end.is_none() {
                return Some(missing("end_line"));
            }
            // Line range sanity checks.
            if let (Some(s), Some(e)) = (start, end) {
                if s == 0 {
                    return Some("`start_line` must be ≥ 1 (lines are 1-indexed)".into());
                }
                if s > e {
                    return Some(format!("`start_line` ({s}) must be ≤ `end_line` ({e})"));
                }
                // Guard against impossibly large ranges that suggest hallucinated
                // line numbers (e.g. patching line 999999 of a 200-line file).
                if e - s > 10_000 {
                    return Some(format!(
                        "line range {s}..{e} spans {} lines — this is unusually large \
                         and likely incorrect. Verify the line numbers.",
                        e - s
                    ));
                }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WriteSet {
    None,
    Paths(Vec<String>),
    Unknown,
}

pub(crate) fn write_set_for_call(call: &ToolCall) -> WriteSet {
    match call.tool_name.as_str() {
        "write_file" | "patch_file" | "create_directory" | "remove_path" => call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .map(normalize_path)
            .map(|p| WriteSet::Paths(vec![p]))
            .unwrap_or(WriteSet::Unknown),
        "copy_path" => {
            let source = call.arguments.get("source").and_then(Value::as_str);
            let target = call.arguments.get("target").and_then(Value::as_str);
            match (source, target) {
                (Some(s), Some(t)) => WriteSet::Paths(vec![normalize_path(s), normalize_path(t)]),
                _ => WriteSet::Unknown,
            }
        }
        "spawn_agent" => match call.arguments.get("write_paths").and_then(Value::as_array) {
            Some(paths) if paths.is_empty() => WriteSet::None,
            Some(paths) => {
                let normalized: Vec<String> = paths
                    .iter()
                    .filter_map(Value::as_str)
                    .map(normalize_path)
                    .collect();
                if normalized.is_empty() {
                    WriteSet::Unknown
                } else {
                    WriteSet::Paths(normalized)
                }
            }
            None => WriteSet::Unknown,
        },
        _ => WriteSet::None,
    }
}

fn write_sets_conflict(a: &WriteSet, b: &WriteSet) -> bool {
    match (a, b) {
        (WriteSet::None, WriteSet::None) => false,
        (WriteSet::Unknown, WriteSet::None) | (WriteSet::None, WriteSet::Unknown) => true,
        (WriteSet::Unknown, _) | (_, WriteSet::Unknown) => true,
        (WriteSet::Paths(a_paths), WriteSet::Paths(b_paths)) => a_paths
            .iter()
            .any(|a| b_paths.iter().any(|b| paths_overlap(a, b))),
        (WriteSet::Paths(_), WriteSet::None) | (WriteSet::None, WriteSet::Paths(_)) => false,
    }
}

fn paths_overlap(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    a.starts_with(&format!("{b}/")) || b.starts_with(&format!("{a}/"))
}

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
            let candidate_write_set = write_set_for_call(call);
            let conflicts = current.iter().any(|(_, existing)| {
                let existing_write_set = write_set_for_call(existing);
                write_sets_conflict(&candidate_write_set, &existing_write_set)
            });
            if conflicts && !current.is_empty() {
                batches.push(std::mem::take(&mut current));
            }
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

// ─── Phase-aware anti-looping utilities ───────────────────────────────────────

/// Returns the initial read-only round budget based on the turn intent.
/// Edit and answer tasks get fewer reads since the repo map already provides
/// structural context; review/inventory tasks need more exploration.
pub(crate) fn initial_read_budget(intent: TurnIntent) -> usize {
    match intent {
        TurnIntent::Edit => 2,
        TurnIntent::Debug => 3,
        TurnIntent::Review | TurnIntent::Inventory => 5,
        TurnIntent::Answer => 2,
        TurnIntent::Unknown => 3,
    }
}

/// Attempt to extract a structured plan from the model's response text.
/// Looks for a `## Plan` or `**Plan**` header followed by bullet points.
/// Returns the extracted steps, or empty vec if no plan is found.
pub(crate) fn extract_plan_from_response(text: &str) -> Vec<String> {
    let mut in_plan = false;
    let mut steps = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("## plan") || lower.starts_with("**plan") || lower == "plan:" {
            in_plan = true;
            continue;
        }
        if in_plan {
            // Stop at next heading
            if trimmed.starts_with("##") {
                break;
            }
            if let Some(step) = trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
            {
                let step = step.trim();
                if !step.is_empty() {
                    steps.push(step.to_string());
                }
            } else if trimmed.is_empty() && !steps.is_empty() {
                break;
            }
        }
    }
    steps
}

/// Input for the progress summary builder.
pub(crate) struct ProgressState<'a> {
    pub phase: &'a str,
    pub iteration: usize,
    pub reads_used: usize,
    pub reads_budget: usize,
    pub plan: &'a [String],
    pub completed_actions: &'a [String],
    pub changed_files: &'a [String],
    pub verified: bool,
}

/// Build a compact structured progress summary for injection into the system
/// prompt each iteration. Gives the model awareness of its current state so
/// it doesn't have to infer progress from scattered transcript.
pub(crate) fn progress_summary(state: &ProgressState<'_>) -> String {
    let mut parts = vec![
        format!("## Turn Progress (iteration {})", state.iteration),
        format!("Phase: {}", state.phase),
        format!("Reads: {}/{} used", state.reads_used, state.reads_budget),
    ];
    if !state.plan.is_empty() {
        parts.push(format!("Plan: {}", state.plan.join(" → ")));
    }
    if !state.completed_actions.is_empty() {
        let recent: Vec<&str> = state
            .completed_actions
            .iter()
            .rev()
            .take(5)
            .map(|s| s.as_str())
            .collect();
        parts.push(format!("Recent actions: {}", recent.join(", ")));
    }
    if !state.changed_files.is_empty() {
        parts.push(format!("Files changed: {}", state.changed_files.join(", ")));
    }
    parts.push(format!(
        "Verified: {}",
        if state.verified { "yes" } else { "no" }
    ));
    parts.join("\n")
}

fn latest_user_request(items: &[ConversationItem]) -> Option<String> {
    items.iter().rev().find_map(|item| {
        (item.kind == ItemKind::UserMessage)
            .then(|| item.payload.get("text").and_then(Value::as_str))
            .flatten()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn recent_file_paths(items: &[ConversationItem], limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();

    for item in items.iter().rev() {
        if paths.len() >= limit {
            break;
        }
        match item.kind {
            ItemKind::ToolCall => {
                if let Some(args) = item.payload.get("arguments") {
                    collect_paths_from_value(args, &mut seen, &mut paths, limit);
                }
            }
            ItemKind::FileChange => {
                collect_paths_from_value(&item.payload, &mut seen, &mut paths, limit);
            }
            _ => {}
        }
    }

    paths.reverse();
    paths
}

fn collect_paths_from_value(
    value: &Value,
    seen: &mut HashSet<String>,
    paths: &mut Vec<String>,
    limit: usize,
) {
    for key in [
        "path",
        "file_path",
        "filePath",
        "target",
        "source",
        "write_paths",
    ] {
        let Some(v) = value.get(key) else {
            continue;
        };
        match v {
            Value::String(path) => push_path(path, seen, paths, limit),
            Value::Array(values) => {
                for item in values {
                    if let Some(path) = item.as_str() {
                        push_path(path, seen, paths, limit);
                    }
                }
            }
            _ => {}
        }
    }
}

fn push_path(path: &str, seen: &mut HashSet<String>, paths: &mut Vec<String>, limit: usize) {
    if paths.len() >= limit {
        return;
    }
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed.len() > 300 {
        return;
    }
    if seen.insert(trimmed.to_string()) {
        paths.push(trimmed.to_string());
    }
}

fn recent_commands(items: &[ConversationItem], limit: usize) -> Vec<String> {
    let mut commands = Vec::new();
    for item in items.iter().rev() {
        if commands.len() >= limit {
            break;
        }
        if item.kind != ItemKind::ToolCall {
            continue;
        }
        let tool = item
            .payload
            .get("toolName")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !matches!(tool, "bash_exec" | "shell_exec") {
            continue;
        }
        let Some(args) = item.payload.get("arguments") else {
            continue;
        };
        if let Some(command) = args.get("command").and_then(Value::as_str) {
            commands.push(command.to_string());
        } else if let Some(argv) = args.get("argv") {
            commands.push(argv_to_string(argv));
        }
    }
    commands.reverse();
    commands
}

fn argv_to_string(argv: &Value) -> String {
    match argv {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" "),
        other => other.to_string(),
    }
}

fn latest_test_or_diagnostic_output(items: &[ConversationItem]) -> Option<String> {
    let mut last_tool = "";
    let mut result_for_tool: Option<(&str, &ConversationItem)> = None;

    for item in items {
        match item.kind {
            ItemKind::ToolCall => {
                last_tool = item
                    .payload
                    .get("toolName")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
            }
            ItemKind::ToolResult => {
                result_for_tool = Some((last_tool, item));
            }
            _ => {}
        }
    }

    let (tool, item) = result_for_tool?;
    if !matches!(tool, "bash_exec" | "shell_exec") {
        return None;
    }
    let output = item
        .payload
        .get("output")
        .map(value_to_text)
        .unwrap_or_default();
    let error = item
        .payload
        .get("errorMessage")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let combined = format!("{error}\n{output}");
    if looks_like_test_or_diagnostic(&combined) {
        Some(tail_chars(combined.trim(), 2400))
    } else {
        None
    }
}

fn looks_like_test_or_diagnostic(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    // Test-runner / build phrases that are unambiguous wherever they appear.
    const PHRASES: &[&str] = &[
        "test result:",
        "tests failed",
        "test failed",
        "failures:",
        "passed;",
        "panicked at",
        "assertion failed",
        "traceback (most recent call last)",
        "npm err!",
        "compilation failed",
    ];
    if PHRASES.iter().any(|p| lower.contains(p)) {
        return true;
    }
    // Compiler-style diagnostics, anchored at line starts so ordinary output
    // that merely mentions "error:" or "warning:" mid-sentence isn't pinned.
    lower.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with("error:")
            || trimmed.starts_with("error[")
            || trimmed.starts_with("warning:")
            || trimmed.starts_with("fatal:")
    })
}

fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Upper bound on each git snapshot command. A repo on a slow filesystem (or
/// one mid-operation holding the index lock) must never stall a turn.
const GIT_SNAPSHOT_TIMEOUT: Duration = Duration::from_millis(1500);

async fn run_git(cwd: &str, args: &[&str], max_chars: usize) -> Option<String> {
    let command = tokio::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output();
    let output = tokio::time::timeout(GIT_SNAPSHOT_TIMEOUT, command)
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(tail_chars(trimmed, max_chars))
    }
}

/// Keep the last `max_chars` characters (used for outputs, where the end —
/// the failure summary — matters most).
fn tail_chars(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        text.to_string()
    } else {
        chars[chars.len() - max_chars..].iter().collect()
    }
}

/// Keep the first `max_chars` characters (used for requests and commands,
/// where the beginning carries the intent).
fn head_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        text.chars().take(max_chars).collect()
    }
}

fn one_line(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    head_chars(&compact, max_chars)
}

fn fenced(lang: &str, body: &str) -> String {
    format!("```{lang}\n{body}\n```")
}
