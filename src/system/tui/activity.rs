//! Activity reducer — structured tracking of in-flight agent work.
//!
//! Phase 7 of the original architecture plan called for "make parallel /
//! sub-agent behaviour visible to users". The first step is tracking tool
//! calls *as a set* with start times, instead of the previous single
//! `live_activity: Option<String>` field that lost the parallel structure.
//!
//! What this module owns:
//!   - `tools`: every in-flight tool call, in start order, with name + hint
//!     + `started_at`. Render code can pull elapsed time from here.
//!   - `turn_started_at`: when the current turn started; `None` while idle.
//!     Lets the header show "Working for 12s" without round-tripping the
//!     runtime.
//!   - `spinner_tick`: animation counter for the header glyph.
//!   - `streaming`: the assistant is mid-stream emitting deltas.
//!
//! What stays on `InteractiveApp` for now:
//!   - `status_message` / `error_message` (status bar — many writers, no
//!     structural payoff in moving them yet).
//!   - `active_turn_id` (string used to address `runtime.interrupt_turn`;
//!     orthogonal to the structured tracking here).
//!   - `token_usage` (accumulator, not really "activity").
//!
//! The reducer is feed-driven: `app.rs` event handlers call `start_tool` /
//! `finish_tool` / `start_turn` / `end_turn` in response to runtime events.
//! Tests exercise the reducer directly without a terminal.
//!
//! Several accessor methods (`is_idle`, `is_streaming`, `latest_tool`,
//! `turn_started_at`, `tools_in_flight`) are not yet read by `app.rs` /
//! `render.rs` — the activity tree views in the next Phase 7 slice will
//! consume them. They're part of the deliberate public API surface and are
//! covered by tests.
#![allow(dead_code)]

use std::collections::HashSet;
use std::time::{Duration, Instant};

/// What's currently holding up the agent. Set when the runtime emits an
/// event that puts the loop in a wait state; cleared when the wait resolves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockedReason {
    /// Waiting for the user to resolve an approval request.
    Approval,
}

impl BlockedReason {
    /// Short label used in the header.
    pub fn label(&self) -> &'static str {
        match self {
            BlockedReason::Approval => "⏸ waiting on approval",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolActivity {
    pub tool_call_id: String,
    pub tool_name: String,
    /// Short user-facing hint (path / command / pattern), already truncated.
    pub hint: Option<String>,
    pub started_at: Instant,
}

/// Lifecycle status of a sub-agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildAgentStatus {
    Running,
    Completed,
    Failed,
    Interrupted,
    TimedOut,
}

impl ChildAgentStatus {
    fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }
}

/// In-flight (or recently-finished) sub-agent spawned via `spawn_agent`.
#[derive(Debug, Clone)]
pub struct ChildAgentActivity {
    /// Parent's `tool_call_id` — links this child to a transcript entry.
    pub parent_tool_call_id: String,
    pub child_thread_id: String,
    pub child_turn_id: String,
    /// Short label (typically the first line of the spawn prompt).
    pub label: String,
    pub status: ChildAgentStatus,
    pub started_at: Instant,
    pub finished_at: Option<Instant>,
}

impl ChildAgentActivity {
    pub fn elapsed(&self, now: Instant) -> Duration {
        self.finished_at
            .unwrap_or(now)
            .saturating_duration_since(self.started_at)
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct SubAgentCounts {
    queued: usize,
    running: usize,
    completed: usize,
    failed: usize,
    interrupted: usize,
    timed_out: usize,
}

impl ToolActivity {
    pub fn elapsed(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.started_at)
    }

    /// One-tool display string, e.g. `"⚙ list_dir src"` or `"⚙ bash_exec"`.
    pub fn label(&self) -> String {
        match &self.hint {
            Some(h) => format!("⚙ {} {}", self.tool_name, h),
            None => format!("⚙ {}", self.tool_name),
        }
    }
}

#[derive(Debug, Default)]
pub struct ActivityState {
    /// In-flight tool calls in start order. `Vec` (not `HashMap`) because the
    /// expected count is small (≤ 4 by `max_child_agents`) and we need stable
    /// display order. Lookups are by linear scan — fine at this scale.
    tools: Vec<ToolActivity>,
    turn_started_at: Option<Instant>,
    spinner_tick: u64,
    streaming: bool,
    /// Set when the runtime is parked waiting on something the user controls
    /// (approval, future: input request). The header shows this in place of
    /// the tool list so the user sees *what they need to act on*.
    blocked: Option<BlockedReason>,
    /// Sub-agents spawned via `spawn_agent`, in start order. Stays after a
    /// child finishes so the user can still see its final outcome until the
    /// parent turn ends; cleared by `end_turn`.
    child_agents: Vec<ChildAgentActivity>,
}

impl ActivityState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spinner_tick(&self) -> u64 {
        self.spinner_tick
    }

    /// Advance the spinner one frame.
    pub fn tick(&mut self) {
        self.spinner_tick = self.spinner_tick.wrapping_add(1);
    }

    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    pub fn set_streaming(&mut self, value: bool) {
        self.streaming = value;
    }

    pub fn turn_started_at(&self) -> Option<Instant> {
        self.turn_started_at
    }

    pub fn turn_elapsed(&self, now: Instant) -> Option<Duration> {
        self.turn_started_at
            .map(|t| now.saturating_duration_since(t))
    }

    /// True when nothing is in flight — no tools, no turn, not streaming.
    pub fn is_idle(&self) -> bool {
        self.tools.is_empty() && self.turn_started_at.is_none() && !self.streaming
    }

    pub fn tools_in_flight(&self) -> &[ToolActivity] {
        &self.tools
    }

    /// The most recently started tool, if any. Used by the header for the
    /// classic single-line display.
    pub fn latest_tool(&self) -> Option<&ToolActivity> {
        self.tools.last()
    }

    /// Mark that a new turn has started. Clears any leftover in-flight state
    /// from a prior turn (defensive — we shouldn't normally see this).
    pub fn start_turn(&mut self, now: Instant) {
        self.tools.clear();
        self.streaming = false;
        self.turn_started_at = Some(now);
    }

    /// Clear everything that's tied to the active turn. Spinner tick keeps
    /// running so render-side animation stays smooth across turns.
    pub fn end_turn(&mut self) {
        self.tools.clear();
        self.streaming = false;
        self.turn_started_at = None;
        self.blocked = None;
        self.child_agents.clear();
    }

    pub fn child_agents(&self) -> &[ChildAgentActivity] {
        &self.child_agents
    }

    /// Look up a child agent by its `child_thread_id`. Used by the event
    /// handler to route a child's runtime events into the right tracker.
    pub fn child_agent_for_thread(&self, child_thread_id: &str) -> Option<&ChildAgentActivity> {
        self.child_agents
            .iter()
            .find(|c| c.child_thread_id == child_thread_id)
    }

    /// Returns true if `thread_id` belongs to a sub-agent we're tracking.
    /// Used by the TUI to filter the global event stream.
    pub fn is_known_child_thread(&self, thread_id: &str) -> bool {
        self.child_agents
            .iter()
            .any(|c| c.child_thread_id == thread_id)
    }

    pub fn start_child_agent(
        &mut self,
        parent_tool_call_id: impl Into<String>,
        child_thread_id: impl Into<String>,
        child_turn_id: impl Into<String>,
        label: impl Into<String>,
        now: Instant,
    ) {
        let child_thread_id = child_thread_id.into();
        // Idempotent on child_thread_id (defensive — duplicate ChildAgentSpawned
        // events would otherwise stack).
        if self
            .child_agents
            .iter()
            .any(|c| c.child_thread_id == child_thread_id)
        {
            return;
        }
        self.child_agents.push(ChildAgentActivity {
            parent_tool_call_id: parent_tool_call_id.into(),
            child_thread_id,
            child_turn_id: child_turn_id.into(),
            label: label.into(),
            status: ChildAgentStatus::Running,
            started_at: now,
            finished_at: None,
        });
    }

    pub fn set_child_agent_status(&mut self, child_thread_id: &str, status: ChildAgentStatus) {
        self.set_child_agent_status_at(child_thread_id, status, Instant::now());
    }

    fn set_child_agent_status_at(
        &mut self,
        child_thread_id: &str,
        status: ChildAgentStatus,
        now: Instant,
    ) {
        if let Some(child) = self
            .child_agents
            .iter_mut()
            .find(|c| c.child_thread_id == child_thread_id)
        {
            child.status = status;
            if status.is_terminal() {
                child.finished_at.get_or_insert(now);
            } else {
                child.finished_at = None;
            }
        }
    }

    pub fn blocked(&self) -> Option<&BlockedReason> {
        self.blocked.as_ref()
    }

    pub fn set_blocked(&mut self, reason: BlockedReason) {
        self.blocked = Some(reason);
    }

    pub fn clear_blocked(&mut self) {
        self.blocked = None;
    }

    pub fn start_tool(
        &mut self,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        hint: Option<String>,
        now: Instant,
    ) {
        let tool_call_id = tool_call_id.into();
        // Idempotent: if the same call_id is already tracked, refresh its
        // metadata in place rather than duplicating.
        if let Some(existing) = self
            .tools
            .iter_mut()
            .find(|t| t.tool_call_id == tool_call_id)
        {
            existing.tool_name = tool_name.into();
            existing.hint = hint;
            existing.started_at = now;
            return;
        }
        self.tools.push(ToolActivity {
            tool_call_id,
            tool_name: tool_name.into(),
            hint,
            started_at: now,
        });
    }

    /// Remove a tool from the in-flight set. Returns the removed entry, if
    /// any. Idempotent — calling on an unknown id is a no-op.
    pub fn finish_tool(&mut self, tool_call_id: &str) -> Option<ToolActivity> {
        self.tools
            .iter()
            .position(|t| t.tool_call_id == tool_call_id)
            .map(|idx| self.tools.remove(idx))
    }

    /// How many rows the activity panel should reserve in the layout.
    /// Zero means "don't show the panel"; otherwise one row per tool +
    /// child-agent, capped so a runaway parallel batch can't push the
    /// transcript off-screen.
    pub fn panel_height(&self) -> u16 {
        const MAX_PANEL_ROWS: u16 = 6;
        let total = self.tools.len();
        // Single-tool case with no children is already covered by the header
        // line; collapse the panel.
        if self.tools.len() <= 1 {
            return 0;
        }
        (total as u16).min(MAX_PANEL_ROWS)
    }

    /// One row per in-flight tool, then one row per known child agent.
    /// Child rows show their lifecycle status icon so the user can tell at
    /// a glance whether a sub-agent is still running.
    pub fn panel_rows(&self, now: Instant) -> Vec<String> {
        const MAX_PANEL_ROWS: usize = 6;
        let rows: Vec<String> = self
            .tools
            .iter()
            .map(|t| format!("{} ({}s)", t.label(), t.elapsed(now).as_secs()))
            .collect();
        let skip = rows.len().saturating_sub(MAX_PANEL_ROWS);
        rows.into_iter().skip(skip).collect()
    }

    /// Build the header label string for the current activity. Falls back to
    /// `streaming_fallback` if no tools are in flight but the assistant is
    /// streaming text. Returns `None` when truly idle.
    ///
    /// Priority: blocked state → in-flight tools → streaming → idle.
    ///
    /// Examples:
    ///   - blocked on approval:      `"⏸ waiting on approval"`
    ///   - one tool, no hint:        `"⚙ list_dir (3s)"`
    ///   - one tool, with hint:      `"⚙ read_file src/main.rs (1s)"`
    ///   - multiple tools:           `"⚙ list_dir, read_file, bash_exec (3 in flight, 5s)"`
    pub fn header_line(&self, now: Instant, streaming_fallback: &str) -> Option<String> {
        if let Some(reason) = &self.blocked {
            return Some(reason.label().to_string());
        }
        if let Some(elapsed) = self.turn_elapsed(now) {
            if !self.tools.is_empty() {
                let count = self.tools.len();
                if count == 1 {
                    let t = &self.tools[0];
                    let secs = t.elapsed(now).as_secs();
                    return Some(format!("{} ({}s)", t.label(), secs));
                }
                let mut seen = HashSet::new();
                let names: Vec<&str> = self
                    .tools
                    .iter()
                    .map(|t| t.tool_name.as_str())
                    .filter(|name| seen.insert(*name))
                    .collect();
                return Some(format!(
                    "⚙ {} ({} in flight, {}s)",
                    names.join(", "),
                    count,
                    elapsed.as_secs()
                ));
            }
            if self.streaming {
                return Some(streaming_fallback.to_string());
            }
        }
        None
    }

    fn spawn_tool_count(&self) -> usize {
        self.tools.iter().filter(|t| is_spawn_agent_tool(t)).count()
    }

    fn regular_tool_count(&self) -> usize {
        self.tools.len().saturating_sub(self.spawn_tool_count())
    }

    fn sub_agent_counts(&self) -> SubAgentCounts {
        let mut counts = SubAgentCounts::default();
        for child in &self.child_agents {
            match child.status {
                ChildAgentStatus::Running => counts.running += 1,
                ChildAgentStatus::Completed => counts.completed += 1,
                ChildAgentStatus::Failed => counts.failed += 1,
                ChildAgentStatus::Interrupted => counts.interrupted += 1,
                ChildAgentStatus::TimedOut => counts.timed_out += 1,
            }
        }

        let linked_children = self
            .tools
            .iter()
            .filter(|tool| {
                is_spawn_agent_tool(tool)
                    && self
                        .child_agents
                        .iter()
                        .any(|child| child.parent_tool_call_id == tool.tool_call_id)
            })
            .count();
        counts.queued = self.spawn_tool_count().saturating_sub(linked_children);
        counts
    }

    fn sub_agent_summary(&self) -> String {
        let counts = self.sub_agent_counts();
        let total = counts.queued
            + counts.running
            + counts.completed
            + counts.failed
            + counts.interrupted
            + counts.timed_out;
        let mut parts = Vec::new();
        push_count(&mut parts, counts.running, "running");
        push_count(&mut parts, counts.queued, "queued");
        push_count(&mut parts, counts.completed, "done");
        push_count(&mut parts, counts.failed, "failed");
        push_count(&mut parts, counts.timed_out, "timed out");
        push_count(&mut parts, counts.interrupted, "stopped");
        if parts.is_empty() {
            parts.push("starting".to_string());
        }
        format!(
            "{} sub-agent{}: {}",
            total.max(1),
            if total == 1 { "" } else { "s" },
            parts.join(", ")
        )
    }
}

fn is_spawn_agent_tool(tool: &ToolActivity) -> bool {
    tool.tool_name == "spawn_agent"
}

fn push_count(parts: &mut Vec<String>, count: usize, label: &str) {
    if count > 0 {
        parts.push(format!("{count} {label}"));
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_idle() {
        let s = ActivityState::new();
        assert!(s.is_idle());
        assert_eq!(s.spinner_tick(), 0);
        assert!(s.tools_in_flight().is_empty());
        assert!(s.latest_tool().is_none());
        assert!(s.turn_started_at().is_none());
    }

    #[test]
    fn tick_advances_spinner() {
        let mut s = ActivityState::new();
        s.tick();
        s.tick();
        assert_eq!(s.spinner_tick(), 2);
    }

    #[test]
    fn start_turn_sets_timestamp_and_end_turn_clears() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        assert!(!s.is_idle());
        assert_eq!(s.turn_started_at(), Some(t0));
        s.end_turn();
        assert!(s.is_idle());
    }

    #[test]
    fn turn_elapsed_returns_duration_since_start() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        let elapsed = s.turn_elapsed(t0 + Duration::from_secs(5)).unwrap();
        assert_eq!(elapsed.as_secs(), 5);
    }

    #[test]
    fn start_tool_and_finish_tool_round_trip() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_tool("c1", "list_dir", Some("src".into()), t0);
        s.start_tool("c2", "read_file", Some("Cargo.toml".into()), t0);
        assert_eq!(s.tools_in_flight().len(), 2);
        assert_eq!(s.latest_tool().unwrap().tool_call_id, "c2");

        let removed = s.finish_tool("c1").expect("c1 should exist");
        assert_eq!(removed.tool_name, "list_dir");
        assert_eq!(s.tools_in_flight().len(), 1);
        assert_eq!(s.latest_tool().unwrap().tool_call_id, "c2");

        // Finishing an unknown id is a no-op, not a panic.
        assert!(s.finish_tool("ghost").is_none());
    }

    #[test]
    fn start_tool_is_idempotent() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_tool("c1", "list_dir", None, t0);
        s.start_tool("c1", "list_dir", Some("renamed".into()), t0);
        assert_eq!(s.tools_in_flight().len(), 1);
        assert_eq!(s.tools_in_flight()[0].hint.as_deref(), Some("renamed"));
    }

    #[test]
    fn end_turn_clears_in_flight_tools() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_tool("c1", "list_dir", None, t0);
        s.start_tool("c2", "bash_exec", None, t0);
        s.end_turn();
        assert!(s.tools_in_flight().is_empty());
    }

    #[test]
    fn header_line_idle_returns_none() {
        let s = ActivityState::new();
        assert!(s.header_line(Instant::now(), "thinking").is_none());
    }

    #[test]
    fn header_line_single_tool_includes_elapsed() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_tool("c1", "list_dir", Some("src".into()), t0);
        let line = s
            .header_line(t0 + Duration::from_secs(3), "thinking")
            .unwrap();
        assert!(line.contains("list_dir"));
        assert!(line.contains("src"));
        assert!(line.contains("3s"), "expected elapsed time, got {line:?}");
    }

    #[test]
    fn header_line_multiple_tools_summarizes() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_tool("c1", "list_dir", None, t0);
        s.start_tool("c2", "read_file", None, t0);
        s.start_tool("c3", "bash_exec", None, t0);
        let line = s
            .header_line(t0 + Duration::from_secs(2), "thinking")
            .unwrap();
        assert!(line.contains("3 in flight"), "got {line:?}");
        assert!(line.contains("list_dir"));
        assert!(line.contains("read_file"));
        assert!(line.contains("bash_exec"));
    }

    #[test]
    fn header_line_streaming_fallback_when_no_tools() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.set_streaming(true);
        let line = s.header_line(t0 + Duration::from_secs(1), "◆ generating…");
        assert_eq!(line.as_deref(), Some("◆ generating…"));
    }

    #[test]
    fn header_line_returns_none_for_idle_streaming_with_no_turn() {
        // Defensive: streaming flag set but no turn started → no header.
        let mut s = ActivityState::new();
        s.set_streaming(true);
        assert!(s.header_line(Instant::now(), "thinking").is_none());
    }

    #[test]
    fn tool_label_includes_hint_when_present() {
        let t0 = Instant::now();
        let with_hint = ToolActivity {
            tool_call_id: "c1".into(),
            tool_name: "read_file".into(),
            hint: Some("src/main.rs".into()),
            started_at: t0,
        };
        assert_eq!(with_hint.label(), "⚙ read_file src/main.rs");

        let without = ToolActivity {
            tool_call_id: "c2".into(),
            tool_name: "bash_exec".into(),
            hint: None,
            started_at: t0,
        };
        assert_eq!(without.label(), "⚙ bash_exec");
    }

    #[test]
    fn panel_height_hides_for_zero_or_one_tool() {
        let mut s = ActivityState::new();
        assert_eq!(s.panel_height(), 0);
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_tool("c1", "list_dir", None, t0);
        assert_eq!(
            s.panel_height(),
            0,
            "single tool is shown in the header line, not the panel"
        );
    }

    #[test]
    fn panel_height_grows_with_tool_count_up_to_cap() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        for i in 0..10 {
            s.start_tool(format!("c{i}"), format!("tool{i}"), None, t0);
        }
        assert_eq!(s.panel_height(), 6, "panel is capped at 6 rows");
    }

    #[test]
    fn panel_rows_format_matches_header_with_elapsed() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_tool("c1", "list_dir", Some("src".into()), t0);
        s.start_tool("c2", "read_file", None, t0);
        let rows = s.panel_rows(t0 + Duration::from_secs(4));
        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains("list_dir"));
        assert!(rows[0].contains("src"));
        assert!(rows[0].contains("4s"));
        assert!(rows[1].contains("read_file"));
        assert!(rows[1].contains("4s"));
    }

    #[test]
    fn blocked_state_dominates_header_line() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        // Even with tools in flight, a blocked state should take precedence
        // — the user needs to see what they're being asked to act on.
        s.start_tool("c1", "list_dir", None, t0);
        s.set_blocked(BlockedReason::Approval);
        let line = s
            .header_line(t0 + Duration::from_secs(2), "thinking")
            .unwrap();
        assert!(line.contains("approval"), "got {line:?}");
        assert!(
            !line.contains("list_dir"),
            "tools should not show while blocked: {line:?}"
        );
    }

    #[test]
    fn clear_blocked_returns_to_normal_header() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_tool("c1", "list_dir", None, t0);
        s.set_blocked(BlockedReason::Approval);
        s.clear_blocked();
        let line = s
            .header_line(t0 + Duration::from_secs(2), "thinking")
            .unwrap();
        assert!(line.contains("list_dir"));
    }

    #[test]
    fn end_turn_clears_blocked_state() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.set_blocked(BlockedReason::Approval);
        s.end_turn();
        assert!(s.blocked().is_none());
        assert!(s.header_line(t0, "thinking").is_none());
    }

    #[test]
    fn child_agent_lifecycle_idempotent_and_status_updatable() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_child_agent("call_1", "thread_x", "turn_x", "do thing", t0);
        // Duplicate spawn for the same child_thread_id is a no-op (defensive
        // against duplicate ChildAgentSpawned events).
        s.start_child_agent("call_1", "thread_x", "turn_x", "do thing", t0);
        assert_eq!(s.child_agents().len(), 1);
        assert_eq!(s.child_agents()[0].status, ChildAgentStatus::Running);

        assert!(s.is_known_child_thread("thread_x"));
        assert!(!s.is_known_child_thread("unrelated"));

        s.set_child_agent_status_at("thread_x", ChildAgentStatus::Completed, t0);
        assert_eq!(s.child_agents()[0].status, ChildAgentStatus::Completed);
        assert_eq!(s.child_agents()[0].finished_at, Some(t0));

        // Status update for unknown thread is a no-op.
        s.set_child_agent_status("ghost", ChildAgentStatus::Failed);
        assert_eq!(s.child_agents().len(), 1);
    }

    #[test]
    fn child_agent_elapsed_freezes_after_terminal_status() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_child_agent("call_1", "thread_x", "turn_x", "do thing", t0);

        assert_eq!(
            s.child_agents()[0]
                .elapsed(t0 + Duration::from_secs(7))
                .as_secs(),
            7
        );

        s.set_child_agent_status_at(
            "thread_x",
            ChildAgentStatus::Failed,
            t0 + Duration::from_secs(3),
        );

        assert_eq!(
            s.child_agents()[0]
                .elapsed(t0 + Duration::from_secs(30))
                .as_secs(),
            3
        );
    }

    #[test]
    fn end_turn_clears_child_agents() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_child_agent("call_1", "thread_x", "turn_x", "do thing", t0);
        s.end_turn();
        assert!(s.child_agents().is_empty());
    }

    #[test]
    fn panel_renders_child_agents_below_tools() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_tool("c1", "list_dir", None, t0);
        s.start_tool("c2", "spawn_agent", None, t0);
        s.start_child_agent("c2", "thread_a", "turn_a", "summarise main.rs", t0);
        s.set_child_agent_status("thread_a", ChildAgentStatus::Running);
        let rows = s.panel_rows(t0 + Duration::from_secs(2));
        assert!(rows.iter().any(|r| r.contains("list_dir")));
        assert!(rows.iter().all(|r| !r.contains("summarise main.rs")));
    }

    #[test]
    fn panel_height_grows_with_child_agents() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        // One tool + two children → panel needs 3 rows.
        s.start_tool("c1", "spawn_agent", None, t0);
        s.start_child_agent("c1", "thread_a", "turn_a", "first", t0);
        s.start_child_agent("c1", "thread_b", "turn_b", "second", t0);
        assert_eq!(s.panel_height(), 0);
    }

    #[test]
    fn header_groups_parallel_spawn_agents() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        s.start_tool("call_1", "spawn_agent", Some("first".into()), t0);
        s.start_tool("call_2", "spawn_agent", Some("second".into()), t0);
        s.start_child_agent("call_1", "thread_a", "turn_a", "first task", t0);

        let line = s
            .header_line(t0 + Duration::from_secs(5), "thinking")
            .unwrap();
        assert!(
            line.contains("spawn_agent, spawn_agent (2 in flight, 5s)"),
            "got {line:?}"
        );
    }

    #[test]
    fn panel_rows_keeps_most_recent_when_overflowed() {
        let mut s = ActivityState::new();
        let t0 = Instant::now();
        s.start_turn(t0);
        for i in 0..10 {
            s.start_tool(format!("c{i}"), format!("tool{i}"), None, t0);
        }
        let rows = s.panel_rows(t0);
        assert_eq!(rows.len(), 6, "panel cap is 6 rows");
        // First retained row should be tool4 (10 - 6 = skip 4); last is tool9.
        assert!(rows[0].contains("tool4"), "got {:?}", rows[0]);
        assert!(rows[5].contains("tool9"), "got {:?}", rows[5]);
    }
}
