//! Behavioural Pattern Miner — post-turn transcript analysis.
//!
//! After each completed turn this module inspects the transcript for recurring
//! patterns: shell commands appended after code changes, repeated tool
//! sequences, or instruction suffixes the user types consistently.
//!
//! Discovered patterns are persisted in a `behaviour_patterns` SQLite table
//! and surfaced as concise system-instruction snippets on future turns.
//!
//! # How it works
//!
//! 1. `PatternMiner::analyse_turn` is called by `TurnExecutor::complete_turn`
//!    (fire-and-forget, background task).
//! 2. It reads the full thread from persistence, extracts signal from
//!    `UserMessage`, `ToolCall`, and `ToolResult` items.
//! 3. Normalised signals are compared against the existing pattern table.
//!    A hit increments the frequency counter; a new signal seeds a new row.
//! 4. `PatternMiner::load_habits` is called by `system_instructions()` and
//!    returns the top-N confirmed patterns (frequency ≥ threshold) as plain
//!    English reminder sentences ready to append to the system prompt.
//!
//! # Pattern kinds
//!
//! | Kind                  | Example trigger                         | Injected instruction                          |
//! |-----------------------|-----------------------------------------|-----------------------------------------------|
//! | `PostEditCommand`     | User runs `cargo fmt` after every edit  | "After editing Rust files, run `cargo fmt`."  |
//! | `PostEditLint`        | `cargo clippy` always follows edits     | "After editing Rust files, run `cargo clippy --quiet`." |
//! | `SuffixInstruction`   | User appends "keep it concise" often    | "The user prefers concise responses."         |
//! | `ToolSequence`        | `read_file` → `patch_file` → `bash_exec`| Implicit — accelerates intent classification  |

use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;

use crate::system::domain::{ConversationItem, ItemKind};

// ─── Public types ──────────────────────────────────────────────────────────────

/// The category of a detected behavioural pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternKind {
    /// A shell/bash command consistently run after file edits.
    PostEditCommand,
    /// A repeated lint/format command after edits (subset of PostEditCommand
    /// kept separate so the injected hint is more precise).
    PostEditLint,
    /// A textual instruction the user appends to prompts with high regularity.
    SuffixInstruction,
    /// A recurring n-gram of tool names used together.
    ToolSequence,
}

impl std::fmt::Display for PatternKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            PatternKind::PostEditCommand => "post_edit_command",
            PatternKind::PostEditLint => "post_edit_lint",
            PatternKind::SuffixInstruction => "suffix_instruction",
            PatternKind::ToolSequence => "tool_sequence",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for PatternKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "post_edit_command" => Ok(PatternKind::PostEditCommand),
            "post_edit_lint" => Ok(PatternKind::PostEditLint),
            "suffix_instruction" => Ok(PatternKind::SuffixInstruction),
            "tool_sequence" => Ok(PatternKind::ToolSequence),
            other => Err(anyhow::anyhow!("unknown pattern kind: {other}")),
        }
    }
}

/// A single stored behavioural pattern.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BehaviourPattern {
    pub pattern_id: String,
    pub kind: PatternKind,
    /// Canonical normalised key (e.g. `"cargo fmt"`, `"cargo clippy --quiet"`).
    pub signal: String,
    /// How many turns have exhibited this pattern.
    pub frequency: u32,
    /// Human-readable reminder injected into the system prompt.
    pub hint: String,
    /// Unix-seconds timestamp of the last observation.
    pub last_seen: i64,
}

// ─── PatternMiner ──────────────────────────────────────────────────────────────

/// Analyses transcripts for recurring user behaviour and persists patterns.
pub struct PatternMiner {
    conn: Mutex<Connection>,
    /// Minimum frequency before a pattern is injected as a system hint.
    confirmation_threshold: u32,
    /// Maximum number of hint sentences to inject per turn.
    max_hints: usize,
}

impl PatternMiner {
    /// Open or create the pattern database at `db_path`.
    pub fn open(db_path: &Path) -> Result<Self> {
        Self::open_with_config(db_path, 2, 5)
    }

    /// Full constructor, exposed for tests.
    pub fn open_with_config(
        db_path: &Path,
        confirmation_threshold: u32,
        max_hints: usize,
    ) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        Self::apply_wal(&conn)?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            confirmation_threshold,
            max_hints,
        })
    }

    /// In-memory store (tests and production fallback when the DB path is unavailable).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            confirmation_threshold: 2,
            max_hints: 5,
        })
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS behaviour_patterns (
                pattern_id  TEXT    PRIMARY KEY,
                kind        TEXT    NOT NULL,
                signal      TEXT    NOT NULL,
                frequency   INTEGER NOT NULL DEFAULT 1,
                hint        TEXT    NOT NULL,
                last_seen   INTEGER NOT NULL,
                UNIQUE(kind, signal)
            );
            CREATE INDEX IF NOT EXISTS idx_bp_freq ON behaviour_patterns(frequency DESC);
            "#,
        )?;
        Ok(())
    }

    /// Apply WAL mode — only valid for file-backed databases.
    fn apply_wal(conn: &Connection) -> Result<()> {
        conn.execute_batch("PRAGMA journal_mode = WAL;")?;
        Ok(())
    }

    // ── Analysis ─────────────────────────────────────────────────────────────

    /// Inspect a completed turn's transcript items and update pattern counts.
    ///
    /// Designed to be called in a background task — never blocks the turn loop.
    pub fn analyse_turn(&self, items: &[ConversationItem]) -> Result<()> {
        let signals = extract_signals(items);
        let conn = self.conn.lock().expect("pattern miner mutex poisoned");

        for (kind, signal, hint) in signals {
            let kind_str = kind.to_string();
            // Upsert: increment frequency when the (kind, signal) pair exists.
            let updated = conn.execute(
                r#"
                UPDATE behaviour_patterns
                SET frequency = frequency + 1, last_seen = ?1
                WHERE kind = ?2 AND signal = ?3
                "#,
                params![now_secs(), kind_str, signal],
            )?;

            if updated == 0 {
                // First observation — seed with frequency = 1.
                let id = format!("bp_{}", Uuid::new_v4().simple());
                conn.execute(
                    r#"
                    INSERT INTO behaviour_patterns (pattern_id, kind, signal, frequency, hint, last_seen)
                    VALUES (?1, ?2, ?3, 1, ?4, ?5)
                    "#,
                    params![id, kind_str, signal, hint, now_secs()],
                )?;
                tracing::debug!(
                    pattern_id = %id,
                    kind = %kind_str,
                    signal = %signal,
                    "pattern_miner: new pattern seeded"
                );
            } else {
                tracing::debug!(
                    kind = %kind_str,
                    signal = %signal,
                    "pattern_miner: pattern frequency incremented"
                );
            }
        }

        Ok(())
    }

    // ── Retrieval ─────────────────────────────────────────────────────────────

    /// Return the top-N confirmed, **actionable** patterns as system-prompt hint sentences.
    ///
    /// Only patterns whose `frequency >= confirmation_threshold` are returned.
    /// `ToolSequence` patterns are excluded: they describe observation-level
    /// correlations, not user expectations, so injecting them would add prompt
    /// noise without changing model behaviour.
    pub fn load_habits(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().expect("pattern miner mutex poisoned");
        let mut stmt = conn.prepare(
            r#"
            SELECT hint
            FROM behaviour_patterns
            WHERE frequency >= ?1
              AND kind != 'tool_sequence'
            ORDER BY frequency DESC, last_seen DESC
            LIMIT ?2
            "#,
        )?;

        let hints: Vec<String> = stmt
            .query_map(
                params![self.confirmation_threshold, self.max_hints as i64],
                |row| row.get(0),
            )?
            .filter_map(|r| r.ok())
            .collect();

        Ok(hints)
    }

    /// Return all stored patterns (for debugging / TUI display).
    pub fn list_all_patterns(&self) -> Result<Vec<BehaviourPattern>> {
        let conn = self.conn.lock().expect("pattern miner mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT pattern_id, kind, signal, frequency, hint, last_seen \
             FROM behaviour_patterns \
             ORDER BY frequency DESC, last_seen DESC",
        )?;

        let patterns = stmt
            .query_map([], |row| {
                let kind_str: String = row.get(1)?;
                Ok((
                    row.get::<_, String>(0)?,
                    kind_str,
                    row.get::<_, String>(2)?,
                    row.get::<_, u32>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id, kind_str, signal, freq, hint, last_seen)| {
                let kind: PatternKind = kind_str.parse().ok()?;
                Some(BehaviourPattern {
                    pattern_id: id,
                    kind,
                    signal,
                    frequency: freq,
                    hint,
                    last_seen,
                })
            })
            .collect();

        Ok(patterns)
    }
    /// Delete a pattern by its ID. Returns true if a row was removed.
    pub fn delete_pattern(&self, pattern_id: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("pattern miner mutex poisoned");
        let removed = conn.execute(
            "DELETE FROM behaviour_patterns WHERE pattern_id = ?1",
            params![pattern_id],
        )?;
        Ok(removed > 0)
    }

    /// Delete all patterns. Returns the number of rows removed.
    pub fn delete_all_patterns(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("pattern miner mutex poisoned");
        let removed = conn.execute("DELETE FROM behaviour_patterns", [])?;
        Ok(removed)
    }
}

// ─── Signal extraction ─────────────────────────────────────────────────────────

/// Extract (kind, signal, hint) triples from a turn's conversation items.
///
/// Stateless, pure function — easy to unit-test without a database.
pub(crate) fn extract_signals(items: &[ConversationItem]) -> Vec<(PatternKind, String, String)> {
    let mut signals: Vec<(PatternKind, String, String)> = Vec::new();

    // ── 1. Post-edit commands ─────────────────────────────────────────────
    // Detect bash_exec / shell_exec calls that follow a file-editing tool
    // call within the same turn. These represent "always run X after editing"
    // patterns.
    let has_file_edit = items.iter().any(|i| {
        i.kind == ItemKind::ToolCall
            && i.payload
                .get("toolName")
                .and_then(|v| v.as_str())
                .map(|n| matches!(n, "write_file" | "patch_file" | "create_file"))
                .unwrap_or(false)
    });

    if has_file_edit {
        for item in items.iter().filter(|i| i.kind == ItemKind::ToolCall) {
            let tool_name = item
                .payload
                .get("toolName")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if !matches!(tool_name, "bash_exec" | "shell_exec") {
                continue;
            }

            // Extract the command string from tool arguments.
            let cmd = extract_command_arg(&item.payload);
            if cmd.is_empty() {
                continue;
            }

            let normalised = normalise_command(&cmd);
            if normalised.is_empty() {
                continue;
            }

            let kind = if is_lint_or_format(&normalised) {
                PatternKind::PostEditLint
            } else {
                PatternKind::PostEditCommand
            };

            let hint = build_post_edit_hint(&normalised, &kind);
            signals.push((kind, normalised, hint));
        }
    }

    // ── 2. Tool-sequence n-grams ──────────────────────────────────────────
    // Build a sequence of tool names in the turn and record meaningful
    // transitions using semantic categories (Read→Edit, Edit→Execute, etc.)
    // rather than a hardcoded pair list. Also capture trigrams for common
    // 3-step workflows (e.g. read_file→patch_file→bash_exec).
    let tool_names: Vec<&str> = items
        .iter()
        .filter(|i| i.kind == ItemKind::ToolCall)
        .filter_map(|i| i.payload.get("toolName").and_then(|v| v.as_str()))
        .collect();

    // Bigrams: any pair where the category transition is meaningful.
    for window in tool_names.windows(2) {
        if let (Some(cat_a), Some(cat_b)) = (tool_category(window[0]), tool_category(window[1])) {
            if is_meaningful_transition(&cat_a, &cat_b) {
                let bigram = format!("{}→{}", window[0], window[1]);
                let hint = format!(
                    "After using `{}`, the user typically follows up with `{}`.",
                    window[0], window[1]
                );
                signals.push((PatternKind::ToolSequence, bigram, hint));
            }
        }
    }

    // Trigrams: common 3-step workflows.
    for window in tool_names.windows(3) {
        if let (Some(cat_a), Some(cat_b), Some(cat_c)) = (
            tool_category(window[0]),
            tool_category(window[1]),
            tool_category(window[2]),
        ) {
            if is_meaningful_trigram(&cat_a, &cat_b, &cat_c) {
                let trigram = format!("{}→{}→{}", window[0], window[1], window[2]);
                let hint = format!(
                    "After `{}` then `{}`, the user typically runs `{}`.",
                    window[0], window[1], window[2]
                );
                signals.push((PatternKind::ToolSequence, trigram, hint));
            }
        }
    }

    // ── 3. Suffix instructions ────────────────────────────────────────────
    // Look for recurring short phrases the user appends to their messages.
    // We now check both the trailing fragment and the full message for
    // instruction markers, and also detect imperative sentences anywhere
    // in the text (not just the last sentence).
    for item in items.iter().filter(|i| i.kind == ItemKind::UserMessage) {
        if let Some(text) = item.payload.get("text").and_then(|v| v.as_str()) {
            // 3a. Trailing instruction suffix (original heuristic).
            if let Some(suffix) = extract_instruction_suffix(text) {
                let normalised = suffix.to_lowercase();
                let hint = format!("The user prefers: \"{suffix}\".");
                signals.push((PatternKind::SuffixInstruction, normalised, hint));
            }

            // 3b. Full-message imperative detection — catches cases where the
            //     entire message is a short instruction like "run tests" or
            //     "format the code" that doesn't appear as a trailing suffix.
            if let Some(imperative) = extract_imperative_message(text) {
                let normalised = imperative.to_lowercase();
                // Avoid duplicating a suffix we already captured.
                if !signals
                    .iter()
                    .any(|(k, s, _)| *k == PatternKind::SuffixInstruction && s == &normalised)
                {
                    let hint = format!("The user often gives the instruction: \"{imperative}\".");
                    signals.push((PatternKind::SuffixInstruction, normalised, hint));
                }
            }
        }
    }

    // Deduplicate within this turn so a single turn doesn't double-count.
    let mut seen: HashMap<(String, String), bool> = HashMap::new();
    signals.retain(|(kind, signal, _)| {
        seen.insert((kind.to_string(), signal.clone()), true)
            .is_none()
    });

    signals
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

fn extract_command_arg(payload: &serde_json::Value) -> String {
    // Tool args are stored as {"toolName": "bash_exec", "args": {"command": "..."}}
    // or {"toolName": "bash_exec", "args": {"cmd": "..."}}
    if let Some(args) = payload.get("args") {
        for key in &["command", "cmd", "script"] {
            if let Some(v) = args.get(key).and_then(|v| v.as_str()) {
                return v.trim().to_string();
            }
        }
    }
    String::new()
}

/// Strip flags/paths to produce a stable identifier.
/// `"cargo clippy -- -D warnings"` → `"cargo clippy"`
fn normalise_command(cmd: &str) -> String {
    // Take the first two whitespace-separated tokens (program + subcommand).
    let mut tokens = cmd.split_whitespace();
    let prog = match tokens.next() {
        Some(p) => p,
        None => return String::new(),
    };

    // Only track well-known toolchain commands to avoid noise.
    let known = [
        "cargo", "npm", "pnpm", "yarn", "make", "go", "python", "python3", "pytest", "jest",
        "deno", "bun", "gradle", "mvn",
    ];
    if !known.contains(&prog) {
        return String::new();
    }

    match tokens.next() {
        Some(sub) if !sub.starts_with('-') => format!("{prog} {sub}"),
        _ => prog.to_string(),
    }
}

fn is_lint_or_format(cmd: &str) -> bool {
    matches!(
        cmd,
        "cargo fmt"
            | "cargo clippy"
            | "cargo check"
            | "cargo test"
            | "npm run lint"
            | "npm run format"
            | "npm test"
            | "pnpm lint"
            | "pnpm format"
            | "pnpm test"
            | "yarn lint"
            | "yarn format"
            | "yarn test"
            | "go vet"
            | "go fmt"
            | "go test"
            | "pytest"
            | "jest"
    )
}

fn build_post_edit_hint(cmd: &str, kind: &PatternKind) -> String {
    match kind {
        PatternKind::PostEditLint => {
            format!("After making code edits, always run `{cmd}` to validate the changes.")
        }
        _ => format!("After making code edits, run `{cmd}`."),
    }
}

// ─── Tool category system ────────────────────────────────────────────────────

/// Semantic category for a tool — used for generalised sequence detection.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolCat {
    /// Reading / inspecting: read_file, list_dir, grep_search, image_metadata
    Read,
    /// Writing / modifying files: write_file, patch_file, create_file, remove_path, copy_path
    Edit,
    /// Running commands: bash_exec, shell_exec
    Execute,
    /// Searching: web_search, web_fetch
    Search,
}

/// Map a tool name to its semantic category.
fn tool_category(name: &str) -> Option<ToolCat> {
    match name {
        "read_file" | "list_dir" | "grep_search" | "image_metadata" => Some(ToolCat::Read),
        "write_file" | "patch_file" | "create_file" | "remove_path" | "copy_path" => {
            Some(ToolCat::Edit)
        }
        "bash_exec" | "shell_exec" => Some(ToolCat::Execute),
        "web_search" | "web_fetch" => Some(ToolCat::Search),
        _ => None,
    }
}

/// A bigram transition is meaningful when it represents a common workflow step.
fn is_meaningful_transition(a: &ToolCat, b: &ToolCat) -> bool {
    matches!(
        (a, b),
        (ToolCat::Read, ToolCat::Edit)
            | (ToolCat::Read, ToolCat::Execute)
            | (ToolCat::Edit, ToolCat::Execute)
            | (ToolCat::Edit, ToolCat::Read)
            | (ToolCat::Execute, ToolCat::Edit)
            | (ToolCat::Search, ToolCat::Edit)
            | (ToolCat::Search, ToolCat::Execute)
    )
}

/// A trigram is meaningful when it represents a complete 3-step workflow.
fn is_meaningful_trigram(a: &ToolCat, b: &ToolCat, c: &ToolCat) -> bool {
    matches!(
        (a, b, c),
        // Read → Edit → Execute (the classic edit-then-verify loop)
        (ToolCat::Read, ToolCat::Edit, ToolCat::Execute)
            | (ToolCat::Edit, ToolCat::Execute, ToolCat::Edit)
            | (ToolCat::Read, ToolCat::Edit, ToolCat::Read)
            | (ToolCat::Search, ToolCat::Edit, ToolCat::Execute)
            | (ToolCat::Search, ToolCat::Read, ToolCat::Edit)
    )
}

/// Extract a short, repeated instruction suffix from the user's message.
///
/// Heuristic: take the last sentence fragment (after the final `.` or `\n`)
/// if it is 6–80 characters long and matches common instructional patterns.
/// Also checks the full message for instruction markers (not just the suffix).
fn extract_instruction_suffix(text: &str) -> Option<&str> {
    let text = text.trim();
    // Skip very short messages.
    if text.len() < 20 {
        return None;
    }

    let suffix = text
        .rsplit(['.', '\n'])
        .find(|s| {
            let s = s.trim();
            !s.is_empty() && s.len() >= 6 && s.len() <= 80
        })?
        .trim();

    // Only consider fragments that look like instructions.
    let lower = suffix.to_lowercase();
    let instruction_markers = [
        "keep it",
        "make sure",
        "don't",
        "do not",
        "always",
        "never",
        "please",
        "be concise",
        "be brief",
        "no explanations",
        "no comments",
        "add tests",
        "write tests",
        "run tests",
        "cargo fmt",
        "cargo clippy",
        "run lint",
        "run format",
        "typecheck",
        "type check",
        "check types",
        "no boilerplate",
        "minimal",
        "short",
        "simple",
        "clean",
        "refactor",
        "optimize",
        "fix the",
        "ensure",
        "verify",
        "validate",
        "test it",
        "commit",
        "push",
        "deploy",
    ];

    if instruction_markers.iter().any(|m| lower.contains(m)) {
        Some(suffix)
    } else {
        None
    }
}

/// Detect when the entire user message is a short imperative instruction.
///
/// Catches messages like "run tests", "format the code", "check for errors"
/// that are too short for the suffix heuristic (which requires text ≥ 20 chars)
/// or where the instruction is the whole message, not a trailing fragment.
fn extract_imperative_message(text: &str) -> Option<&str> {
    let text = text.trim();
    // Only consider short-to-medium messages (3–120 chars) that look like commands.
    if text.len() < 3 || text.len() > 120 {
        return None;
    }
    // Skip messages that contain newlines — those are multi-part instructions
    // better handled by the suffix heuristic.
    if text.contains('\n') {
        return None;
    }

    let lower = text.to_lowercase();

    // Imperative verb starters — common short commands users give.
    let imperative_starters = [
        "run ",
        "fmt",
        "format",
        "lint",
        "check ",
        "test",
        "build",
        "deploy",
        "commit",
        "push",
        "clean",
        "fix ",
        "refactor",
        "optimize",
        "verify",
        "validate",
        "ensure",
        "typecheck",
        "cargo ",
        "npm ",
        "pnpm ",
        "yarn ",
        "go ",
        "make ",
        "python ",
        "pytest",
        "jest",
        "deno ",
        "bun ",
        "gradle ",
        "mvn ",
    ];

    // Must start with an imperative verb or toolchain command.
    if !imperative_starters.iter().any(|s| lower.starts_with(s)) {
        return None;
    }

    // Must not be a question or long explanatory message.
    if lower.contains('?') || text.len() > 80 {
        return None;
    }

    Some(text)
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::domain::{ConversationItem, ItemKind};
    use serde_json::json;

    fn make_tool_call(tool: &str, cmd: Option<&str>) -> ConversationItem {
        let mut args = json!({});
        if let Some(c) = cmd {
            args = json!({ "command": c });
        }
        ConversationItem {
            item_id: format!("item_{tool}"),
            thread_id: "t1".into(),
            turn_id: "turn1".into(),
            created_at: 0,
            kind: ItemKind::ToolCall,
            payload: json!({ "toolName": tool, "args": args }),
        }
    }

    fn make_user_msg(text: &str) -> ConversationItem {
        ConversationItem {
            item_id: "user1".into(),
            thread_id: "t1".into(),
            turn_id: "turn1".into(),
            created_at: 0,
            kind: ItemKind::UserMessage,
            payload: json!({ "text": text }),
        }
    }

    #[test]
    fn detects_cargo_fmt_after_edit() {
        let items = vec![
            make_tool_call("patch_file", None),
            make_tool_call("bash_exec", Some("cargo fmt")),
        ];
        let signals = extract_signals(&items);
        assert!(
            signals
                .iter()
                .any(|(k, s, _)| *k == PatternKind::PostEditLint && s == "cargo fmt"),
            "expected cargo fmt signal, got {signals:?}"
        );
    }

    #[test]
    fn detects_cargo_clippy_after_edit() {
        let items = vec![
            make_tool_call("write_file", None),
            make_tool_call("bash_exec", Some("cargo clippy --quiet -- -D warnings")),
        ];
        let signals = extract_signals(&items);
        assert!(
            signals
                .iter()
                .any(|(k, s, _)| *k == PatternKind::PostEditLint && s == "cargo clippy"),
            "expected cargo clippy signal, got {signals:?}"
        );
    }

    #[test]
    fn no_post_edit_signal_without_file_edit() {
        let items = vec![make_tool_call("bash_exec", Some("cargo fmt"))];
        let signals = extract_signals(&items);
        let post_edit: Vec<_> = signals
            .iter()
            .filter(|(k, _, _)| {
                *k == PatternKind::PostEditLint || *k == PatternKind::PostEditCommand
            })
            .collect();
        assert!(post_edit.is_empty(), "should not fire without a file edit");
    }

    #[test]
    fn detects_suffix_instruction() {
        let items = vec![make_user_msg("Refactor the parser module. Keep it concise")];
        let signals = extract_signals(&items);
        assert!(
            signals
                .iter()
                .any(|(k, _, _)| *k == PatternKind::SuffixInstruction),
            "expected suffix instruction signal, got {signals:?}"
        );
    }

    #[test]
    fn detects_tool_bigram_sequence() {
        let items = vec![
            make_tool_call("patch_file", None),
            make_tool_call("bash_exec", None),
        ];
        let signals = extract_signals(&items);
        assert!(
            signals
                .iter()
                .any(|(k, s, _)| *k == PatternKind::ToolSequence
                    && s.contains("patch_file")
                    && s.contains("bash_exec")),
            "expected tool sequence signal, got {signals:?}"
        );
    }

    #[test]
    fn frequency_reaches_confirmation_threshold() {
        let miner = PatternMiner::open_in_memory().unwrap();

        let items = vec![
            make_tool_call("write_file", None),
            make_tool_call("bash_exec", Some("cargo fmt")),
        ];

        // First turn — below threshold.
        miner.analyse_turn(&items).unwrap();
        let habits = miner.load_habits().unwrap();
        assert!(
            habits.is_empty(),
            "should not surface hint below threshold, got {habits:?}"
        );

        // Second turn — at threshold (default = 2).
        miner.analyse_turn(&items).unwrap();
        let habits = miner.load_habits().unwrap();
        assert!(
            !habits.is_empty(),
            "should surface habit at threshold, got empty"
        );
        assert!(
            habits.iter().any(|h| h.contains("cargo fmt")),
            "expected cargo fmt hint, got {habits:?}"
        );
    }

    #[test]
    fn deduplication_within_turn() {
        let items = vec![
            make_tool_call("patch_file", None),
            // Same command twice in one turn.
            make_tool_call("bash_exec", Some("cargo fmt")),
            make_tool_call("bash_exec", Some("cargo fmt")),
        ];
        let signals = extract_signals(&items);
        let fmt_count = signals.iter().filter(|(_, s, _)| s == "cargo fmt").count();
        assert_eq!(fmt_count, 1, "should deduplicate within a single turn");
    }

    #[test]
    fn normalise_strips_flags() {
        assert_eq!(
            normalise_command("cargo clippy -- -D warnings"),
            "cargo clippy"
        );
        assert_eq!(normalise_command("npm run build --production"), "npm run");
        assert_eq!(normalise_command("go vet ./..."), "go vet");
        assert_eq!(normalise_command("unknown_tool arg"), "");
    }

    // ── New tests for category-based sequence detection ────────────────────

    #[test]
    fn detects_category_bigram_read_then_edit() {
        let items = vec![
            make_tool_call("grep_search", None),
            make_tool_call("patch_file", None),
        ];
        let signals = extract_signals(&items);
        assert!(
            signals.iter().any(|(k, s, _)| {
                *k == PatternKind::ToolSequence
                    && s.contains("grep_search")
                    && s.contains("patch_file")
            }),
            "expected grep_search→patch_file bigram, got {signals:?}"
        );
    }

    #[test]
    fn detects_category_bigram_search_then_execute() {
        let items = vec![
            make_tool_call("web_search", None),
            make_tool_call("bash_exec", None),
        ];
        let signals = extract_signals(&items);
        assert!(
            signals.iter().any(|(k, s, _)| {
                *k == PatternKind::ToolSequence
                    && s.contains("web_search")
                    && s.contains("bash_exec")
            }),
            "expected web_search→bash_exec bigram, got {signals:?}"
        );
    }

    #[test]
    fn detects_trigram_read_edit_execute() {
        let items = vec![
            make_tool_call("read_file", None),
            make_tool_call("patch_file", None),
            make_tool_call("bash_exec", Some("cargo test")),
        ];
        let signals = extract_signals(&items);
        assert!(
            signals.iter().any(|(k, s, _)| {
                *k == PatternKind::ToolSequence && s.contains("→") && s.matches('→').count() == 2
            }),
            "expected a trigram signal, got {signals:?}"
        );
    }

    #[test]
    fn no_trigram_for_unrelated_sequence() {
        let items = vec![
            make_tool_call("bash_exec", Some("echo hi")),
            make_tool_call("bash_exec", Some("echo bye")),
            make_tool_call("bash_exec", Some("echo done")),
        ];
        let signals = extract_signals(&items);
        let trigrams: Vec<_> = signals
            .iter()
            .filter(|(k, s, _)| *k == PatternKind::ToolSequence && s.matches('→').count() == 2)
            .collect();
        assert!(
            trigrams.is_empty(),
            "should not detect trigram for execute→execute→execute, got {trigrams:?}"
        );
    }

    #[test]
    fn detects_imperative_short_message() {
        let text = "run tests";
        let result = extract_imperative_message(text);
        assert!(
            result.is_some(),
            "expected imperative detection for 'run tests'"
        );
    }

    #[test]
    fn detects_imperative_cargo_command() {
        let text = "cargo test";
        let result = extract_imperative_message(text);
        assert!(
            result.is_some(),
            "expected imperative detection for 'cargo test'"
        );
    }

    #[test]
    fn no_imperative_for_question() {
        let text = "run tests?";
        let result = extract_imperative_message(text);
        assert!(
            result.is_none(),
            "should not detect imperative for a question"
        );
    }

    #[test]
    fn no_imperative_for_long_message() {
        let text = "I think we should probably consider running the test suite after making these changes to ensure nothing is broken";
        let result = extract_imperative_message(text);
        assert!(
            result.is_none(),
            "should not detect imperative for a long explanatory message"
        );
    }

    #[test]
    fn detects_expanded_suffix_markers() {
        let items = vec![make_user_msg(
            "Please refactor the module. Ensure all types are checked",
        )];
        let signals = extract_signals(&items);
        assert!(
            signals
                .iter()
                .any(|(k, _, _)| *k == PatternKind::SuffixInstruction),
            "expected suffix instruction for 'ensure all types are checked', got {signals:?}"
        );
    }

    #[test]
    fn detects_imperative_in_extract_signals() {
        let items = vec![make_user_msg("run lint")];
        let signals = extract_signals(&items);
        assert!(
            signals
                .iter()
                .any(|(k, _, _)| *k == PatternKind::SuffixInstruction),
            "expected SuffixInstruction for imperative 'run lint', got {signals:?}"
        );
    }
}
