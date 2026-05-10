use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::oneshot;
use tracing::info;

use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::markdown::md_to_lines;

use super::super::agent::model_gateway::estimate_items_tokens;
use super::super::domain::{
    ActionDescriptor, ApprovalDecision, ApprovalPolicy, ApprovalResolution, ConversationItem,
    ItemKind, PendingApproval, ReasoningEffort, RuntimeEvent, RuntimeEventKind, ThreadMetadata,
    TokenUsage, ToolResult, TurnStatus, UserInput, EVENT_KIND_AGENT_MESSAGE,
    EVENT_KIND_REASONING_SUMMARY, EVENT_KIND_REASONING_TEXT, KEY_ARGUMENTS, KEY_ERROR,
    KEY_ERROR_MESSAGE, KEY_KIND, KEY_MESSAGE, KEY_OUTPUT, KEY_REASON, KEY_STATUS, KEY_TEXT,
    KEY_THREAD_ID, KEY_THREAD_ID_SNAKE, KEY_TOOL_CALL_ID, KEY_TOOL_NAME, KEY_TURN_ID,
    KEY_TURN_ID_SNAKE, LABEL_ASSISTANT, LABEL_REASONING, LABEL_RUNTIME, LABEL_THINKING,
    LABEL_TOOL_FALLBACK, LABEL_USER, STATUS_DONE, STATUS_FAILED, STATUS_INTERRUPTED,
    STATUS_NOT_SPAWNED, STATUS_TIMED_OUT, STATUS_TIMEOUT,
};
use super::super::error as cod_error;
use super::super::runtime::{
    ConversationRuntime, ThreadCompactParams, ThreadCompactResult, ThreadForkParams,
    ThreadListParams, ThreadModelSettingsParams, ThreadReadParams, ThreadResumeParams,
    ThreadStartParams, TurnInterruptParams, TurnStartParams, TurnSteerParams,
};
use super::activity::ChildAgentStatus;
use super::types::{
    basename, current_state_label, entry_elapsed_secs, entry_from_item, entry_style,
    format_timestamp, format_tool_result, is_diff_body, is_read_file_body, relative_time_ago,
    render_diff_chunk, render_read_file_body_lines, short_turn_id, spinner_frame, split_at_width,
    thread_label, transcript_lines, AutocompleteItem, ComposerAttachment, ComposerState, EntryKind,
    FocusPane, PendingApprovalView, SelectionPoint, SelectionRange, TranscriptEntry, COLOR_ACCENT,
    COLOR_MUTED, THREAD_LIMIT,
};

/// Sentinel item-id for the "thinking" placeholder injected immediately after
/// the user submits a message.  Removed when the first real agent content arrives.
const THINKING_PLACEHOLDER_ID: &str = "__codezilla_thinking__";
const USER_PENDING_PLACEHOLDER_ID: &str = "__codezilla_user_pending__";
/// Item-id prefix for the in-transcript "Working" timer entry. One per turn,
/// so historical durations persist after subsequent turns start. Not persisted
/// and not sent to the model.
const WORKING_ENTRY_ID_PREFIX: &str = "__codezilla_working__";

fn working_entry_id(turn_id: &str) -> String {
    format!("{WORKING_ENTRY_ID_PREFIX}{turn_id}")
}

/// Render an in-flight tool activity as a short user-facing phrase, e.g.
/// `"Reading src/main.rs"` or `"Running git status"`. Falls back to the raw
/// tool name when no hint is available or the verb is unknown.
fn describe_tool_activity(tool: &super::activity::ToolActivity) -> String {
    let verb = match tool.tool_name.as_str() {
        "read_file" => "Reading",
        "write_file" => "Writing",
        "patch_file" => "Editing",
        "list_dir" => "Listing",
        "grep_search" => "Searching",
        "bash_exec" | "shell_exec" => "Running",
        "create_directory" => "Creating",
        "remove_path" => "Removing",
        "copy_path" => "Copying",
        "web_fetch" => "Fetching",
        "image_metadata" => "Inspecting image",
        "spawn_agent" => "Running sub-agent:",
        _ => "Calling",
    };
    match &tool.hint {
        Some(h) if !h.is_empty() => format!("{verb} {h}"),
        _ => format!("{verb} {}", tool.tool_name),
    }
}

/// Truncate a single-line description so it fits comfortably on one
/// transcript row. Keeps the prefix and appends an ellipsis.
fn truncate_status_line(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Per-item char cap for the "ran"/"searched" lists in the done summary.
fn trim_long(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Join a small list, showing the first `keep` entries verbatim and a
/// "+N more" tail when there are extras.
fn join_with_more(items: &[String], keep: usize) -> String {
    if items.len() <= keep {
        return items.join(", ");
    }
    format!("{} +{} more", items[..keep].join(", "), items.len() - keep)
}

/// Pull the first non-blank, non-meta line from a tool-call body. Used to
/// recover the actual command from `bash_exec` / `shell_exec` entries whose
/// body is rendered as `<command>\ncwd: <dir>`.
fn first_meaningful_line(body: &str) -> Option<String> {
    body.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("cwd:"))
        .map(|s| s.to_string())
}

/// `format_tool_call` renders grep_search bodies as `/<pattern>/  in <path>`;
/// extract just the pattern (without slashes).
fn extract_grep_pattern(body: &str) -> Option<String> {
    let line = body.lines().next()?.trim();
    let inner = line.strip_prefix('/')?;
    let end = inner.rfind("/  in ").or_else(|| inner.rfind('/'))?;
    Some(inner[..end].to_string())
}

/// Best-effort URL pluck from a `web_fetch` body — falls back to the first
/// non-blank line when no http(s) URL is found.
fn extract_first_url(body: &str) -> Option<String> {
    for line in body.lines() {
        for token in line.split_whitespace() {
            let cleaned = token.trim_matches(|c: char| {
                c == '"' || c == '\'' || c == ',' || c == '}' || c == '{' || c == ':'
            });
            if cleaned.starts_with("http://") || cleaned.starts_with("https://") {
                return Some(cleaned.to_string());
            }
        }
    }
    body.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(|s| s.to_string())
}
#[derive(Debug, Clone)]
struct CachedTranscriptEntry {
    kind: EntryKind,
    title: String,
    timestamp: Option<i64>,
    completed_at: Option<i64>,
    pending: bool,
    /// True when this is the in-transcript Working timer entry; the renderer
    /// uses this to confine the pending spinner glyph to that single row.
    is_working_timer: bool,
    /// For markdown entries (Assistant/Summary/Reasoning), stores the original
    /// raw body text so the renderer can call md_to_lines each frame.
    /// For plain entries, this is empty.
    raw_body: String,
    /// Pre-chunked plain-text lines (used for non-markdown entries and for
    /// counting lines in the scroll arithmetic for markdown entries).
    body_lines: Vec<String>,
    line_count: usize,
    /// When true, only the header line is rendered (body is hidden).
    collapsed: bool,
}

#[derive(Debug, Default, Clone)]
struct TranscriptRenderCache {
    width: u16,
    dirty: bool,
    entries: Vec<CachedTranscriptEntry>,
    line_ends: Vec<usize>,
    total_lines: usize,
}

/// Full application state for the interactive TUI.
pub struct InteractiveApp {
    pub runtime: ConversationRuntime,
    pub current_thread_id: String,
    pub current_thread_meta: Option<ThreadMetadata>,
    /// Cached thread list + selection cursor reducer.
    pub threads: super::threads::ThreadListState,
    pub transcript: Vec<TranscriptEntry>,
    pub transcript_index: HashMap<String, usize>,
    /// Transcript scroll cursor + auto-follow flag reducer.
    pub transcript_view: super::transcript_view::TranscriptViewState,
    pub focus: FocusPane,
    pub composer: ComposerState,
    /// Approval modal + policy-override reducer.
    pub approval: super::approval::ApprovalState,
    pub active_turn_id: Option<String>,
    /// Cumulative token usage from completed turns (input + output tokens).
    pub token_usage: TokenUsage,
    /// Latest completed-turn prompt input tokens (used for context % display).
    pub latest_prompt_input_tokens: i64,
    /// Token usage for the currently streaming turn (replaced, not accumulated).
    pub streaming_turn_usage: TokenUsage,
    pub status_message: String,
    pub error_message: Option<String>,
    pub should_quit: bool,
    /// Structured tracking of agent work in flight (tools, streaming flag,
    /// turn-elapsed timer, spinner tick). Drives the header line and any
    /// future activity tree.
    pub activity: super::activity::ActivityState,
    /// Whether mouse capture is active (wheel scroll on, native select off).
    pub mouse_capture_enabled: bool,
    /// Whether the user has pressed Ctrl+Q and is awaiting confirmation.
    pub quit_requested: bool,
    // ── drag-to-select ────────────────────────────────────────────────────────
    /// Whether the user has pressed Ctrl+C once while composer is focused and
    /// is awaiting a second Ctrl+C to clear the composer prompt.
    pub composer_clear_requested: bool,
    pub transcript_selection: super::selection::TranscriptSelectionState,
    /// Drag-selection reducer for the composer pane.
    pub composer_selection: super::selection::ComposerSelectionState,
    /// Transcript area rect — written by the renderer each frame.
    pub transcript_area: Rect,
    /// Composer input area rect — written by the renderer each frame.
    pub composer_area: Rect,
    /// Header area rect — written by the renderer each frame.
    pub header_area: Rect,
    /// Status bar area rect — written by the renderer each frame.
    pub status_bar_area: Rect,
    transcript_render_cache: TranscriptRenderCache,
    /// Slash-command autocomplete reducer (suggestions + selection + scroll).
    pub autocomplete: super::autocomplete::AutocompleteState,
    /// Per-session model/reasoning override (None = use thread metadata + config defaults).
    pub model_settings_override: Option<super::super::domain::ModelSettings>,
    composer_history: super::composer_history::ComposerHistoryState,
    /// Background compaction task result receiver. Set when /compact is running.
    pub(super) pending_compact: Option<oneshot::Receiver<anyhow::Result<ThreadCompactResult>>>,
}

impl InteractiveApp {
    pub async fn bootstrap(
        runtime: ConversationRuntime,
        initial_thread_id: String,
    ) -> Result<Self> {
        let mut app = Self {
            runtime,
            current_thread_id: initial_thread_id.clone(),
            current_thread_meta: None,
            threads: {
                let mut s = super::threads::ThreadListState::new();
                s.set_selected(Some(initial_thread_id));
                s
            },
            transcript: Vec::new(),
            transcript_index: HashMap::new(),
            transcript_view: super::transcript_view::TranscriptViewState::new(),
            focus: FocusPane::Composer,
            composer: ComposerState::default(),
            approval: super::approval::ApprovalState::new(),
            active_turn_id: None,
            token_usage: TokenUsage::default(),
            latest_prompt_input_tokens: 0,
            streaming_turn_usage: TokenUsage::default(),
            status_message: "Ready".into(),
            error_message: None,
            should_quit: false,
            activity: super::activity::ActivityState::new(),
            quit_requested: false,
            composer_clear_requested: false,
            mouse_capture_enabled: true,
            transcript_selection: super::selection::TranscriptSelectionState::new(),
            composer_selection: super::selection::ComposerSelectionState::new(),
            transcript_area: Rect::default(),
            composer_area: Rect::default(),
            header_area: Rect::default(),
            status_bar_area: Rect::default(),
            transcript_render_cache: TranscriptRenderCache {
                dirty: true,
                ..TranscriptRenderCache::default()
            },
            autocomplete: super::autocomplete::AutocompleteState::new(),
            model_settings_override: None,
            composer_history: super::composer_history::ComposerHistoryState::new(),
            pending_compact: None,
        };
        app.refresh_threads().await?;
        let current = app.current_thread_id.clone();
        app.load_thread(&current).await?;
        Ok(app)
    }
    pub async fn refresh_threads(&mut self) -> Result<()> {
        let listed = self
            .runtime
            .list_threads(ThreadListParams {
                cwd: None,
                archived: Some(false),
                search_term: None,
                limit: Some(THREAD_LIMIT),
                cursor: None,
            })
            .await?
            .threads;
        self.threads.set_threads(listed);
        self.threads
            .reconcile_selection(Some(self.current_thread_id.clone()));
        Ok(())
    }

    pub async fn load_thread(&mut self, thread_id: &str) -> Result<()> {
        self.runtime
            .resume_thread(ThreadResumeParams {
                thread_id: thread_id.to_string(),
            })
            .await?;
        let persisted = self
            .runtime
            .read_thread(ThreadReadParams {
                thread_id: thread_id.to_string(),
            })
            .await?
            .thread;

        self.current_thread_id = thread_id.to_string();
        self.current_thread_meta = Some(persisted.metadata.clone());
        self.threads.set_selected(Some(thread_id.to_string()));
        self.transcript.clear();
        self.transcript_index.clear();
        self.invalidate_transcript_cache();
        for item in &persisted.items {
            self.upsert_loaded_item(item);
        }
        self.active_turn_id = persisted
            .turns
            .iter()
            .rev()
            .find(|t| {
                matches!(
                    t.status,
                    TurnStatus::Created | TurnStatus::Running | TurnStatus::WaitingForApproval
                )
            })
            .map(|t| t.turn_id.clone());
        // Aggregate token usage from all completed turns. Prefer provider-reported
        // actual usage when present; fall back to estimated usage otherwise.
        self.token_usage = persisted
            .turns
            .iter()
            .filter(|t| t.status == TurnStatus::Completed)
            .fold(TokenUsage::default(), |acc, t| {
                let actual = &t.token_usage;
                let chosen = if actual.input_tokens > 0
                    || actual.output_tokens > 0
                    || actual.cached_tokens > 0
                {
                    actual
                } else {
                    &t.estimated_token_usage
                };
                TokenUsage {
                    input_tokens: acc.input_tokens + chosen.input_tokens,
                    output_tokens: acc.output_tokens + chosen.output_tokens,
                    cached_tokens: acc.cached_tokens + chosen.cached_tokens,
                }
            });
        self.latest_prompt_input_tokens = persisted
            .turns
            .iter()
            .filter(|t| t.status == TurnStatus::Completed)
            .max_by_key(|t| t.updated_at)
            .map(|t| {
                if t.token_usage.input_tokens > 0
                    || t.token_usage.output_tokens > 0
                    || t.token_usage.cached_tokens > 0
                {
                    t.token_usage.input_tokens
                } else {
                    t.estimated_token_usage.input_tokens
                }
            })
            .unwrap_or(0);
        self.streaming_turn_usage = TokenUsage::default();
        self.approval.set_pending(None);
        self.transcript_selection.clear();
        self.transcript_view.jump_to_bottom();
        self.error_message = None;
        self.reset_composer_history_navigation();
        self.composer_history
            .replace_history(self.build_composer_history(&persisted).await?);
        self.status_message = format!("Opened {}", thread_label(&persisted.metadata));
        Ok(())
    }

    pub async fn create_new_thread(&mut self) -> Result<()> {
        let cwd = self
            .current_thread_meta
            .as_ref()
            .and_then(|t| t.cwd.clone())
            .or_else(|| Some(self.runtime.effective_config().working_directory.clone()));

        let created = self
            .runtime
            .start_thread(ThreadStartParams {
                cwd,
                model_settings: None,
                approval_policy: self.current_approval_policy_override(),
                permission_profile: None,
                ephemeral: false,
            })
            .await?;
        self.refresh_threads().await?;
        self.load_thread(&created.metadata.thread_id).await?;
        self.status_message = format!("Created {}", thread_label(&created.metadata));
        self.error_message = None;
        Ok(())
    }

    pub async fn fork_current_thread(&mut self) -> Result<()> {
        let forked = self
            .runtime
            .fork_thread(ThreadForkParams {
                thread_id: self.current_thread_id.clone(),
                ephemeral: false,
            })
            .await?;
        self.refresh_threads().await?;
        self.load_thread(&forked.metadata.thread_id).await?;
        self.status_message = format!("Forked into {}", thread_label(&forked.metadata));
        self.error_message = None;
        Ok(())
    }

    pub fn compact_current_thread(&mut self) {
        use super::super::domain::CompactionStrategy;

        if self.active_turn_id.is_some() {
            self.error_message =
                Some("Cannot compact while a turn is active — interrupt first".into());
            return;
        }
        if self.pending_compact.is_some() {
            self.error_message = Some("Compaction already in progress…".into());
            return;
        }

        self.status_message = "Compacting… (summarising with LLM)".into();
        self.error_message = None;
        let started_at = super::super::domain::now_millis();
        self.push_status_entry(
            format!("manual_compaction_started_{started_at}"),
            EntryKind::Status,
            "Compaction",
            "Compacting…",
            Some(started_at / 1000),
        );

        let (tx, rx) = oneshot::channel();
        let runtime = self.runtime.clone();
        let thread_id = self.current_thread_id.clone();
        tokio::spawn(async move {
            let result = runtime
                .compact_thread(ThreadCompactParams {
                    thread_id,
                    strategy: CompactionStrategy::SummarizePrefix,
                })
                .await;
            // Ignore send error — TUI may have quit.
            let _ = tx.send(result);
        });
        self.pending_compact = Some(rx);
    }

    /// Poll the background compaction task. Returns `true` if a result was
    /// received and the caller should force a redraw.
    pub async fn poll_compact_result(&mut self) -> Result<bool> {
        let rx = match self.pending_compact.as_mut() {
            None => return Ok(false),
            Some(rx) => rx,
        };

        // try_recv both checks readiness *and* extracts the value in one shot.
        // Never call .await after try_recv — that would panic with "called after complete".
        match rx.try_recv() {
            Ok(result) => {
                self.pending_compact = None;
                match result {
                    Ok(r) => {
                        self.load_thread(&self.current_thread_id.clone()).await?;
                        // After compaction, re-estimate context usage from the
                        // compacted items so ctx% reflects the reduced context.
                        self.recalculate_context_from_items().await;
                        let done_msg = format!(
                            "✓ Compacted — {} item(s) replaced with summary",
                            r.items_removed
                        );
                        self.status_message = done_msg.clone();
                        let completed_at = super::super::domain::now_millis();
                        self.push_status_entry(
                            format!("manual_compaction_completed_{completed_at}"),
                            EntryKind::Status,
                            "Compaction",
                            &done_msg,
                            Some(completed_at / 1000),
                        );
                        self.error_message = None;
                    }
                    Err(e) => {
                        self.load_thread(&self.current_thread_id.clone()).await?;
                        let msg = format!("Compact failed: {e}");
                        self.error_message = Some(msg.clone());
                        let failed_at = super::super::domain::now_millis();
                        self.push_status_entry(
                            format!("manual_compaction_failed_{failed_at}"),
                            EntryKind::Error,
                            "Compaction",
                            &msg,
                            Some(failed_at / 1000),
                        );
                    }
                }
                Ok(true)
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => Ok(false),
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                self.pending_compact = None;
                self.error_message = Some("Compact failed: background task closed".into());
                Ok(true)
            }
        }
    }

    /// Recalculate `latest_prompt_input_tokens` from the current persisted items.
    /// Called after compaction to ensure the status bar ctx% reflects the
    /// reduced context window usage.
    async fn recalculate_context_from_items(&mut self) {
        let Ok(persisted) = self
            .runtime
            .read_thread(ThreadReadParams {
                thread_id: self.current_thread_id.clone(),
            })
            .await
        else {
            return;
        };
        let estimated = estimate_items_tokens(&persisted.thread.items) as i64;
        self.latest_prompt_input_tokens = estimated;
        self.streaming_turn_usage = TokenUsage::default();
    }

    // ── Drag-to-select helpers ────────────────────────────────────────────────

    pub fn begin_transcript_drag(&mut self, col: u16, row: u16) {
        // Clear any composer selection when starting a transcript drag, plus
        // its click history so a follow-up click in the composer doesn't
        // accidentally pair with the cross-pane click.
        self.clear_composer_selection();
        self.composer_selection.forget_click();

        let now = std::time::Instant::now();
        let is_double_click = self.transcript_selection.is_double_click(now, col, row);

        let Some(point) = self.mouse_to_selection_point(col, row, false) else {
            self.clear_selection();
            return;
        };

        if is_double_click {
            // Double-click: select the word under the cursor and enter word-snap
            // mode so subsequent drag extends word-by-word.
            let width = self.transcript_area.width;
            self.ensure_transcript_render_cache(width);
            let plain_lines = self.transcript_plain_lines(width);
            let (word_start, word_end) =
                super::selection::transcript_word_range_at(point, &plain_lines);
            self.transcript_selection
                .start_word_snap(word_start, word_end);
            // Explicit selection should detach from bottom-follow.
            self.transcript_view.set_auto_scroll(false);
            // Reset so triple-click doesn't extend.
            self.transcript_selection.forget_click();
        } else {
            self.transcript_selection.start(point);
            self.transcript_selection.record_click(now, col, row);
        }
    }

    pub fn update_transcript_drag(&mut self, col: u16, row: u16) {
        if !self.transcript_selection.is_active() || self.transcript_selection.is_locked() {
            return;
        }
        if let Some(point) = self.mouse_to_selection_point(col, row, true) {
            if self.transcript_selection.word_snap() {
                let width = self.transcript_area.width;
                self.ensure_transcript_render_cache(width);
                let plain_lines = self.transcript_plain_lines(width);
                self.transcript_selection
                    .update_end_word_snap(point, &plain_lines);
            } else {
                self.transcript_selection.update_end(point);
            }
            // Detach from auto-follow only after actual movement (true drag),
            // not on a plain click-to-focus in the transcript.
            if self.transcript_selection.is_moved() {
                self.transcript_view.set_auto_scroll(false);
            }
        }
    }

    pub fn finish_transcript_drag(&mut self, col: u16, row: u16) {
        if !self.transcript_selection.is_locked() {
            self.update_transcript_drag(col, row);
        }
        if self.transcript_selection.is_moved() {
            self.copy_selection_to_clipboard();
        } else {
            self.clear_selection();
        }
    }
    fn mouse_to_selection_point(
        &mut self,
        col: u16,
        row: u16,
        clamp_to_viewport: bool,
    ) -> Option<SelectionPoint> {
        // ── Header area (row 0) ──────────────────────────────────────────────
        if self.header_area.width > 0
            && row >= self.header_area.y
            && row < self.header_area.y + self.header_area.height
        {
            let right = self
                .header_area
                .x
                .saturating_add(self.header_area.width.saturating_sub(1));
            let c = if clamp_to_viewport {
                col.clamp(self.header_area.x, right)
            } else if col < self.header_area.x || col > right {
                return None;
            } else {
                col
            };
            return Some(SelectionPoint {
                line: 0,
                col: c.saturating_sub(self.header_area.x) as usize,
            });
        }

        // ── Transcript area ──────────────────────────────────────────────────
        let area = self.transcript_area;
        if area.width == 0 || area.height == 0 {
            return None;
        }
        let right = area.x.saturating_add(area.width.saturating_sub(1));
        let bottom = area.y.saturating_add(area.height.saturating_sub(1));
        let (col, row) = if clamp_to_viewport {
            (col.clamp(area.x, right), row.clamp(area.y, bottom))
        } else {
            if col < area.x || col > right || row < area.y || row > bottom {
                return None;
            }
            (col, row)
        };
        Some(SelectionPoint {
            line: self.transcript_view.scroll() as usize + (row as usize - area.y as usize),
            col: col.saturating_sub(area.x) as usize,
        })
    }

    fn current_selection_range(&self) -> Option<super::types::SelectionRange> {
        use super::types::SelectionRange;

        let start = self.transcript_selection.drag_start()?;
        let end = self.transcript_selection.drag_end()?;
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };

        Some(SelectionRange {
            start_line: start.line,
            start_col: start.col,
            end_line: end.line,
            end_col: end.col,
        })
    }

    /// Build the plain-text representation of the header line (same content as
    /// rendered). Used by copy-to-clipboard when the user drag-selects text in
    /// the header area.
    pub fn header_plain_text(&self) -> String {
        let meta = self.current_thread_meta.as_ref();
        let thread = meta.map(thread_label).unwrap_or_else(|| "no thread".into());
        let cwd = meta
            .and_then(|t| t.cwd.as_ref())
            .map(|c| basename(c))
            .unwrap_or_else(|| "~".into());
        let (model, reasoning) = {
            let ms = self.effective_model_settings();
            let model = format!("{}/{}", ms.provider_id, ms.model_id);
            let reasoning = ms.reasoning_effort.as_str().to_string();
            (model, reasoning)
        };
        let approval_icon = if self.auto_approve_tools_enabled() {
            "🔓"
        } else {
            "🔒"
        };
        let state = current_state_label(self.active_turn_id.is_some(), self.approval.has_pending());
        let state_sigil = if self.active_turn_id.is_some() {
            spinner_frame(self.activity.spinner_tick())
        } else {
            "●"
        };
        let live_label: String = if self.active_turn_id.is_some() {
            self.activity
                .header_line(std::time::Instant::now(), "◆ generating…")
                .unwrap_or_else(|| state.to_string())
        } else {
            state.to_string()
        };

        // Left side
        let left = format!(" ◈ codezilla  {state_sigil} {live_label}");

        // Right side
        let mut right = cwd.to_string();
        right.push_str(" │ ");
        right.push_str(&model);
        if self
            .effective_model_settings()
            .reasoning_effort
            .is_explicit()
        {
            right.push_str(" │ ");
            right.push_str(&format!("reasoning:{reasoning}"));
        }
        right.push_str(" │ ");
        right.push_str(approval_icon);

        // Thread title in between
        let left_width = left.chars().count();
        let right_width = right.chars().count();
        let available = self.header_area.width as usize;
        let max_thread = available.saturating_sub(left_width + right_width + 2);
        let thread_display = if thread.chars().count() > max_thread && max_thread > 3 {
            let chars: Vec<char> = thread.chars().take(max_thread.saturating_sub(1)).collect();
            format!("{}…", chars.iter().collect::<String>())
        } else {
            thread
        };
        let thread_width = thread_display.chars().count();
        let used = left_width + 2 + thread_width + right_width;
        let padding = available.saturating_sub(used);

        format!("{left}  {thread_display}{}{right}", " ".repeat(padding))
    }

    pub fn clear_selection(&mut self) {
        self.transcript_selection.clear();
    }

    /// Clear the composer drag selection. Click history is preserved for
    /// double-click detection (matches transcript behaviour).
    pub fn clear_composer_selection(&mut self) {
        self.composer_selection.clear();
    }

    // ── Composer drag-to-select ──────────────────────────────────────────────

    /// Convert a mouse (col, row) inside the composer area to a character index
    /// in `composer.chars`. Returns `None` if the point is outside the composer.
    fn mouse_to_composer_index(&self, col: u16, row: u16) -> Option<usize> {
        let area = self.composer_area;
        if area.width == 0 || area.height == 0 {
            return None;
        }
        // The composer input area starts after a 1-row separator and 1-row
        // top margin (see render_composer). The actual text starts at
        // `comp_layout[2]` which is stored in `self.composer_area`.
        if col < area.x || row < area.y || col >= area.x + area.width || row >= area.y + area.height
        {
            return None;
        }

        let prefix: usize = 5; // "  ❯  " or "     "
        let text_width = area.width.saturating_sub(prefix as u16) as usize;
        let text_width = text_width.max(1);

        let local_row = (row - area.y) as usize;
        let local_col = (col - area.x) as usize;

        // Account for composer scroll
        let composer_scroll = self.composer_scroll_offset(text_width);

        let visual_row = local_row + composer_scroll;
        let visual_col = local_col
            .saturating_sub(prefix)
            .min(text_width.saturating_sub(1));

        // Convert visual (row, col) back to a char index in composer.chars
        Some(
            self.composer
                .index_for_visual_position(visual_row, visual_col, text_width, text_width),
        )
    }

    /// Compute the composer scroll offset (how many visual rows are scrolled out of view).
    fn composer_scroll_offset(&self, text_width: usize) -> usize {
        let (row, _col) = self.composer.visual_cursor_row_col(text_width, text_width);
        let visible_rows = self.composer_area.height as usize;
        if row >= visible_rows {
            row + 1 - visible_rows
        } else {
            0
        }
    }

    /// Begin a drag-to-select in the composer.
    /// Detects double-click (click within 400ms at the same position) and
    /// selects the entire line under the cursor.
    pub fn begin_composer_drag(&mut self, col: u16, row: u16) {
        // Clear any transcript selection when starting a composer drag, plus
        // its click history so a cross-pane click doesn't pair as a double.
        self.clear_selection();
        self.transcript_selection.forget_click();
        if let Some(idx) = self.mouse_to_composer_index(col, row) {
            let now = std::time::Instant::now();
            let is_double_click = self.composer_selection.is_double_click(now, col, row);

            if is_double_click {
                // Select the word under the cursor and enter word-snap mode so
                // subsequent drag extends word-by-word.
                let (lo, hi) = super::selection::composer_word_range_at(idx, &self.composer.chars);
                self.composer_selection.start_word_snap(lo, hi);
                self.composer.cursor = hi;
                // Reset click tracking so triple-click doesn't extend further.
                self.composer_selection.forget_click();
            } else {
                self.composer_selection.start(idx);
                self.composer_selection.record_click(now, col, row);
            }
            self.focus = FocusPane::Composer;
        }
    }

    /// Update a drag-to-select in the composer.
    pub fn update_composer_drag(&mut self, col: u16, row: u16) {
        if !self.composer_selection.is_active() || self.composer_selection.is_locked() {
            return;
        }
        if let Some(idx) = self.mouse_to_composer_index(col, row) {
            if self.composer_selection.word_snap() {
                self.composer_selection
                    .update_end_word_snap(idx, &self.composer.chars);
            } else {
                self.composer_selection.update_end(idx);
            }
        }
    }

    /// Finish a drag-to-select in the composer. If the selection is non-empty,
    /// copy it to the clipboard.
    pub fn finish_composer_drag(&mut self, col: u16, row: u16) {
        if !self.composer_selection.is_locked() {
            self.update_composer_drag(col, row);
        }
        if let Some((lo, hi)) = self.composer_selection.ordered_range() {
            if lo != hi {
                let text: String = self.composer.chars[lo..hi].iter().collect();
                let char_count = text.chars().count();
                match arboard::Clipboard::new() {
                    Ok(mut cb) => match cb.set_text(&text) {
                        Ok(_) => {
                            self.status_message = format!("✓ Copied {char_count} chars");
                            self.error_message = None;
                        }
                        Err(e) => {
                            self.error_message = Some(format!("Clipboard write failed: {e}"));
                        }
                    },
                    Err(e) => {
                        self.error_message = Some(format!("Clipboard unavailable: {e}"));
                    }
                }
                // Keep the selection visible (consistent with transcript drag-to-select).
                // It will be cleared by: next drag, key press, or Ctrl+C.
            } else {
                // Single click (no drag) — move cursor to click position, clear selection.
                self.composer.cursor = hi;
                self.composer_selection.clear();
            }
        }
    }

    /// Return the selected range (lo, hi) as char indices, if any.
    pub fn composer_selection_range(&self) -> Option<(usize, usize)> {
        let (lo, hi) = self.composer_selection.ordered_range()?;
        if lo == hi {
            return None;
        }
        Some((lo, hi))
    }

    /// Copy the selected composer text to the system clipboard.
    pub fn copy_composer_selection_to_clipboard(&mut self) {
        let Some((lo, hi)) = self.composer_selection_range() else {
            return;
        };
        let text: String = self.composer.chars[lo..hi].iter().collect();
        let char_count = text.chars().count();
        match arboard::Clipboard::new() {
            Ok(mut cb) => match cb.set_text(&text) {
                Ok(_) => {
                    self.status_message = format!("✓ Copied {char_count} chars");
                    self.error_message = None;
                }
                Err(e) => {
                    self.error_message = Some(format!("Clipboard write failed: {e}"));
                }
            },
            Err(e) => {
                self.error_message = Some(format!("Clipboard unavailable: {e}"));
            }
        }
    }

    /// Scroll the composer content by `delta` visual rows (negative = up).
    /// This adjusts the cursor position to keep the view anchored, since
    /// the composer currently auto-scrolls to keep the cursor visible.
    pub fn composer_scroll(&mut self, delta: i32) {
        let (first_w, cont_w) = self.composer_wrap_widths();
        let (cursor_row, cursor_col) = self.composer.visual_cursor_row_col(first_w, cont_w);
        let visible_rows = self.composer_area.height as usize;
        let total_visual_rows = {
            let text = self.composer.text();
            let logical_lines: Vec<&str> = if text.is_empty() {
                vec![""]
            } else {
                text.split('\n').collect()
            };
            let widths = (first_w, cont_w);
            let mut rows = 0usize;
            for (i, line) in logical_lines.iter().enumerate() {
                rows += super::types::wrapped_rows_for_line(line.chars().count(), i == 0, widths);
            }
            rows.max(1)
        };

        // Compute target visual row: move the "viewport anchor" by delta
        let current_scroll = if cursor_row >= visible_rows {
            cursor_row + 1 - visible_rows
        } else {
            0
        };
        let new_scroll = if delta < 0 {
            current_scroll.saturating_sub(delta.unsigned_abs() as usize)
        } else {
            (current_scroll + delta as usize).min(total_visual_rows.saturating_sub(visible_rows))
        };

        // Move cursor to the same column in the new scroll-relative row
        let target_row =
            new_scroll + (cursor_row - current_scroll).min(visible_rows.saturating_sub(1));
        let new_cursor =
            self.composer
                .cursor_for_visual_position(target_row, cursor_col, (first_w, cont_w));
        self.composer.cursor = new_cursor;
    }

    pub fn scroll_transcript(&mut self, delta: i32) {
        self.transcript_view.scroll_by(delta);
    }

    pub fn jump_transcript_to_bottom(&mut self) {
        self.transcript_view.jump_to_bottom();
    }

    pub fn composer_wrap_widths(&self) -> (usize, usize) {
        // Both first and continuation lines use the same prefix width (5 chars)
        // so both wrap at the same column — matching render_composer exactly.
        let w = self.composer_area.width.saturating_sub(5) as usize;
        (w, w)
    }

    /// Convert drag terminal coordinates into a `SelectionRange` (line + column)
    /// within the `transcript_lines` vec.  Normalises so start ≤ end.
    pub fn drag_selection_lines(&self) -> Option<super::types::SelectionRange> {
        self.current_selection_range()
    }

    /// Copy the selected transcript text to the system clipboard.
    /// For markdown entries the raw markdown source of the selected lines is copied.
    /// For other entry types the rendered visual text is used (gutter stripped).
    pub fn copy_selection_to_clipboard(&mut self) {
        let Some(sel) = self.current_selection_range() else {
            return;
        };

        let width = self.transcript_area.width;
        self.ensure_transcript_render_cache(width);

        // ── Header line (line 0) ────────────────────────────────────────────
        if sel.start_line == 0 {
            let header_text = self.header_plain_text();
            let chars: Vec<char> = header_text.chars().collect();
            let from = sel.start_col.min(chars.len());
            let to = (sel.end_col + 1).min(chars.len());
            if from < to {
                let selected: String = chars[from..to].iter().collect();
                if !selected.trim().is_empty() {
                    let char_count = selected.chars().count();
                    match arboard::Clipboard::new() {
                        Ok(mut cb) => match cb.set_text(selected) {
                            Ok(_) => {
                                self.status_message = format!("✓ Copied {char_count} chars");
                                self.error_message = None;
                            }
                            Err(e) => {
                                self.error_message = Some(format!("Clipboard write failed: {e}"));
                            }
                        },
                        Err(e) => {
                            self.error_message = Some(format!("Clipboard unavailable: {e}"));
                        }
                    }
                }
            }
            return;
        }

        // ── Rendered visual text ─────────────────────────────────────────────
        let (lines, total) = self.transcript_lines_all(width, None);
        let end_clamped = sel.end_line.min(total.saturating_sub(1));
        if sel.start_line > end_clamped {
            return;
        }

        let selected: String = lines[sel.start_line..=end_clamped]
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let full: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
                let chars: Vec<char> = full.chars().collect();
                let line_idx = sel.start_line + i;
                let from = if line_idx == sel.start_line {
                    sel.start_col
                } else {
                    0
                };
                let to = if line_idx == end_clamped {
                    sel.end_col + 1
                } else {
                    chars.len()
                };
                let from = from.min(chars.len());
                let to = to.min(chars.len());
                let text: String = chars[from..to].iter().collect();
                text.strip_prefix("  │  ").unwrap_or(&text).to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");

        if selected.trim().is_empty() {
            return;
        }
        let char_count = selected.chars().count();

        match arboard::Clipboard::new() {
            Ok(mut cb) => match cb.set_text(selected) {
                Ok(_) => {
                    self.status_message = format!("✓ Copied {char_count} chars");
                    self.error_message = None;
                }
                Err(e) => {
                    self.error_message = Some(format!("Clipboard write failed: {e}"));
                }
            },
            Err(e) => {
                self.error_message = Some(format!("Clipboard unavailable: {e}"));
            }
        }
    }

    pub async fn submit_composer(&mut self) -> Result<()> {
        let trimmed = self.composer.trimmed_text();
        if trimmed.is_empty() && self.composer.attachments.is_empty() {
            return Ok(());
        }
        let attachments = self.composer.take_attachments();
        let raw = self.composer.take_text();
        self.reset_composer_history_navigation();
        if self.try_handle_slash_command(raw.trim()).await? {
            return Ok(());
        }
        if let Some(cmd) = raw.trim().strip_prefix('!') {
            self.run_shell_command(cmd.trim()).await?;
            return Ok(());
        }
        if self.approval.has_pending() {
            self.error_message =
                Some("Resolve the approval request before sending more input".into());
            return Ok(());
        }

        self.error_message = None;
        // Detect inline image file paths in the text and promote them to attachments.
        // Lines that are bare image paths (or the only content) are extracted;
        // the remaining text becomes the textual part of the message.
        let (text_part, extra_attachments) = extract_image_paths(&raw);
        let all_attachments: Vec<&ComposerAttachment> =
            attachments.iter().chain(extra_attachments.iter()).collect();

        // If images are attached but the model doesn't support them, reject the submission.
        if !all_attachments.is_empty() && !self.current_model_supports_vision() {
            let ms = self.effective_model_settings();
            self.error_message = Some(format!(
                "This model ({}) does not support vision (image) input",
                ms.model_id
            ));
            // Put the text back in the composer so the user doesn't lose it.
            self.composer.set_text(raw);
            return Ok(());
        }

        // Bundle all images with the text into a single UserInput so the
        let images: Vec<super::super::domain::UserInputImage> = all_attachments
            .iter()
            .map(|a| super::super::domain::UserInputImage {
                path: a.path.clone(),
            })
            .collect();
        let mut input: Vec<UserInput> = Vec::new();
        if !text_part.is_empty() || !images.is_empty() {
            input.push(UserInput {
                text: if text_part.is_empty() {
                    None
                } else {
                    Some(super::super::domain::UserInputText {
                        text: text_part.clone(),
                    })
                },
                images,
            });
        }
        if input.is_empty() {
            return Ok(());
        }

        // ── Immediate visual feedback ─────────────────────────────────────────
        // Push the user's message into the transcript right away so "You" is
        // never blank.  Uses a well-known placeholder ID so that when the
        // runtime fires ItemCompleted for the real user message, we can replace
        // this placeholder with the persisted entry (avoiding duplicates).
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            let display_body = if all_attachments.is_empty() {
                raw.clone()
            } else {
                let names: Vec<String> = all_attachments
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("[#image {}]", i + 1))
                    .collect();
                format!("{}\n{}", raw, names.join("\n"))
            };
            self.upsert_transcript_entry(TranscriptEntry {
                item_id: USER_PENDING_PLACEHOLDER_ID.to_string(),
                turn_id: self.active_turn_id.clone(),
                tool_call_id: None,
                kind: EntryKind::User,
                title: LABEL_USER.into(),
                body: display_body,
                timestamp: Some(ts),
                completed_at: None,
                pending: true,
                collapsed: false,
            });
        }
        self.transcript_view.jump_to_bottom();

        if let Some(turn_id) = self.active_turn_id.clone() {
            self.runtime
                .steer_turn(TurnSteerParams {
                    thread_id: self.current_thread_id.clone(),
                    expected_turn_id: turn_id.clone(),
                    input,
                })
                .await?;
            self.status_message = format!("Queued input for {}", short_turn_id(&turn_id));
        } else {
            let turn = self
                .runtime
                .start_turn(
                    TurnStartParams {
                        thread_id: self.current_thread_id.clone(),
                        input,
                        cwd: None,
                        model_settings: self.model_settings_override.clone(),
                        approval_policy: self.current_approval_policy_override(),
                        permission_profile: None,
                        output_schema: None,
                        repo_map_verbosity: None,
                        agent_depth: 0,
                    },
                    super::super::domain::SurfaceKind::Interactive,
                )
                .await?;
            self.active_turn_id = Some(turn.turn.turn_id.clone());
            self.status_message = format!("Started {}", short_turn_id(&turn.turn.turn_id));
        }
        self.push_composer_history_entry(&raw);
        Ok(())
    }

    pub fn composer_history_prev(&mut self) -> bool {
        use super::composer_history::HistoryNavigation;
        let current = self.composer.text().to_string();
        match self.composer_history.prev(&current) {
            HistoryNavigation::Empty => false,
            HistoryNavigation::Set(text) | HistoryNavigation::Restore(text) => {
                self.composer.set_text(text);
                true
            }
        }
    }

    pub fn composer_history_next(&mut self) -> bool {
        use super::composer_history::HistoryNavigation;
        let current = self.composer.text().to_string();
        match self.composer_history.next(&current) {
            HistoryNavigation::Empty => false,
            HistoryNavigation::Set(text) | HistoryNavigation::Restore(text) => {
                self.composer.set_text(text);
                true
            }
        }
    }

    pub fn composer_history_active(&self) -> bool {
        self.composer_history.is_active()
    }

    pub fn reset_composer_history_navigation(&mut self) {
        self.composer_history.reset_navigation();
    }

    async fn build_composer_history(
        &self,
        persisted: &super::super::domain::PersistedThread,
    ) -> Result<Vec<String>> {
        Ok(extract_user_message_history(&persisted.items))
    }

    fn push_composer_history_entry(&mut self, raw: &str) {
        if raw.trim().is_empty() {
            return;
        }
        self.composer_history.push(raw.to_string());
    }

    pub async fn try_handle_slash_command(&mut self, command: &str) -> Result<bool> {
        if !command.starts_with('/') {
            return Ok(false);
        }

        // /speculate is a pass-through: recognised by the TUI (so it doesn't
        // show "Unknown command") but forwarded to the agent as plain text so
        // the executor's should_speculate() can detect the prefix.
        if command.starts_with("/speculate") {
            return Ok(false);
        }

        let handled = if matches!(command, "/quit" | "/exit") {
            self.should_quit = true;
            true
        } else if matches!(command, "/new") {
            self.create_new_thread().await?;
            true
        } else if matches!(command, "/fork") {
            self.fork_current_thread().await?;
            true
        } else if matches!(command, "/threads" | "/reload") {
            self.refresh_threads().await?;
            self.status_message = format!("Loaded {} threads", self.threads.len());
            self.error_message = None;
            true
        } else if matches!(command, "/interrupt") {
            self.interrupt_active_turn().await?;
            true
        } else if matches!(command, "/compact") {
            self.compact_current_thread();
            true
        } else if matches!(command, "/approve auto") {
            self.set_auto_approve_tools(true).await;
            true
        } else if matches!(command, "/approve ask" | "/approve manual") {
            self.set_auto_approve_tools(false).await;
            true
        } else if matches!(command, "/approve toggle") {
            self.toggle_auto_approve_tools().await;
            true
        } else if matches!(command, "/approve" | "/approvals") {
            self.status_message = format!("Approvals: {}", self.approval_mode_label());
            self.error_message = None;
            true
        } else if let Some(rest) = command.strip_prefix("/open ") {
            self.load_thread(rest.trim()).await?;
            true
        } else if let Some(rest) = command.strip_prefix("/resume ") {
            self.load_thread(rest.trim()).await?;
            true
        } else if matches!(command, "/model") {
            let ms = self.effective_model_settings();
            let reasoning = ms.reasoning_effort.as_str();
            self.status_message = format!(
                "Model: {}/{} · reasoning: {reasoning}",
                ms.provider_id, ms.model_id
            );
            self.error_message = None;
            true
        } else if let Some(rest) = command.strip_prefix("/model ") {
            let rest = rest.trim();
            let mut ms = self.effective_model_settings();
            if let Some((provider, model)) = rest.split_once('/') {
                ms.provider_id = provider.trim().to_string();
                ms.model_id = model.trim().to_string();
            } else {
                ms.model_id = rest.to_string();
            }
            self.runtime
                .set_thread_model_settings(ThreadModelSettingsParams {
                    thread_id: self.current_thread_id.clone(),
                    model_id: ms.model_id.clone(),
                    provider_id: ms.provider_id.clone(),
                })
                .await?;
            if let Some(meta) = self.current_thread_meta.as_mut() {
                meta.model_id = ms.model_id.clone();
                meta.provider_id = ms.provider_id.clone();
            }
            self.status_message = format!("Model set to {}/{}", ms.provider_id, ms.model_id);
            self.error_message = None;
            self.model_settings_override = Some(ms);
            info!(
                thread_id = %self.current_thread_id,
                provider_id = %self.effective_model_settings().provider_id,
                model_id = %self.effective_model_settings().model_id,
                "tui: model override updated via /model"
            );
            true
        } else if matches!(command, "/reasoning") {
            let effort = self
                .effective_model_settings()
                .reasoning_effort
                .as_str()
                .to_string();
            self.status_message = format!("Reasoning: {effort}");
            self.error_message = None;
            true
        } else if let Some(rest) = command.strip_prefix("/reasoning ") {
            let rest = rest.trim().to_ascii_lowercase();
            let Some(effort) = ReasoningEffort::parse(&rest) else {
                self.error_message = Some(format!(
                    "Invalid reasoning effort: {rest}. Use auto, off, low, medium, or high."
                ));
                return Ok(true);
            };
            let mut ms = self.effective_model_settings();
            ms.reasoning_effort = effort;
            let label = ms.reasoning_effort.as_str();
            self.status_message = format!("Reasoning set to {label}");
            self.error_message = None;
            self.model_settings_override = Some(ms);
            true
        } else if matches!(command, "/patterns") {
            match self.runtime.list_patterns() {
                Ok(patterns) => {
                    if patterns.is_empty() {
                        self.status_message = "No behavioural patterns learned yet".into();
                        self.error_message = None;
                    } else {
                        let mut lines: Vec<String> = Vec::new();
                        lines.push(format!("{} pattern(s) learned:", patterns.len()));
                        for p in &patterns {
                            let freq = p.frequency;
                            let confirmed = if freq >= 2 { "✓" } else { "?" };
                            lines.push(format!(
                                "  {confirmed} [{:?}] {} (×{freq}) — {}",
                                p.kind, p.signal, p.hint
                            ));
                        }
                        self.push_status_entry(
                            format!(
                                "patterns_{}",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs()
                            ),
                            EntryKind::Status,
                            "Patterns",
                            &lines.join("\n"),
                            None,
                        );
                        self.status_message = format!("{} pattern(s) listed", patterns.len());
                        self.error_message = None;
                    }
                    true
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to list patterns: {e}"));
                    true
                }
            }
        } else if matches!(command, "/patterns clear") || matches!(command, "/patterns reset") {
            match self.runtime.delete_all_patterns() {
                Ok(count) => {
                    self.status_message = format!("Deleted all {count} pattern(s)");
                    self.error_message = None;
                    true
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to delete patterns: {e}"));
                    true
                }
            }
        } else if let Some(rest) = command.strip_prefix("/patterns delete ") {
            let pattern_id = rest.trim();
            if pattern_id.is_empty() {
                self.error_message = Some("Usage: /patterns delete <pattern_id>".into());
                true
            } else {
                match self.runtime.delete_pattern(pattern_id) {
                    Ok(true) => {
                        self.status_message = format!("Deleted pattern {pattern_id}");
                        self.error_message = None;
                    }
                    Ok(false) => {
                        self.error_message = Some(format!("Pattern {pattern_id} not found"));
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Failed to delete pattern: {e}"));
                    }
                }
                true
            }
        } else if matches!(command, "/help") {
            self.status_message =
                "Keys: Tab/↑↓ autocomplete, Ctrl+A/E start/end-of-line, Ctrl+N new, Ctrl+F fork, \
                 Ctrl+C interrupt (double-tap clears composer), Ctrl+Q quit  ·  \
                 Approval: Y approve  U approve+auto  D deny  ·  \
                 Commands: /model [provider/model]  /reasoning [auto|off|low|medium|high]  \
                 /approve auto|ask|toggle  /compact  /new  /fork  /open <id>  /threads  \
                 /patterns [delete <id>|clear]  /speculate <task> (autocomplete)  ·  \
                 CLI: codezilla -r (resume last thread)"
                    .into();
            self.error_message = None;
            true
        } else {
            self.error_message = Some(format!("Unknown command: {command}"));
            true
        };

        Ok(handled)
    }

    async fn run_shell_command(&mut self, cmd: &str) -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let id_base = format!("shell-{ts}");

        self.push_status_entry(
            format!("{id_base}-cmd"),
            EntryKind::User,
            "Shell",
            &format!("$ {cmd}"),
            Some(ts as i64),
        );

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
            .await;

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                let mut body = String::new();
                if !stdout.is_empty() {
                    body.push_str(stdout.trim_end());
                }
                if !stderr.is_empty() {
                    if !body.is_empty() {
                        body.push('\n');
                    }
                    body.push_str(stderr.trim_end());
                }
                if body.is_empty() {
                    body = format!("(exit {})", out.status.code().unwrap_or(-1));
                } else if !out.status.success() {
                    body.push_str(&format!("\n(exit {})", out.status.code().unwrap_or(-1)));
                }
                let kind = if out.status.success() {
                    EntryKind::ToolResult
                } else {
                    EntryKind::Error
                };
                self.push_status_entry(
                    format!("{id_base}-out"),
                    kind,
                    "Output",
                    &body,
                    Some(ts as i64),
                );
            }
            Err(e) => {
                self.push_status_entry(
                    format!("{id_base}-out"),
                    EntryKind::Error,
                    "Shell error",
                    &e.to_string(),
                    Some(ts as i64),
                );
            }
        }
        Ok(())
    }

    /// Returns the model settings that will be used for the next turn:
    /// the session override if set, otherwise a snapshot from thread metadata + config defaults.
    pub fn effective_model_settings(&self) -> super::super::domain::ModelSettings {
        if let Some(ref ms) = self.model_settings_override {
            return ms.clone();
        }
        let cfg = self.runtime.effective_config();
        let (model_id, provider_id) = self
            .current_thread_meta
            .as_ref()
            .map(|m| (m.model_id.clone(), m.provider_id.clone()))
            .unwrap_or_else(|| {
                (
                    cfg.model_settings.model_id.clone(),
                    cfg.model_settings.provider_id.clone(),
                )
            });
        super::super::domain::ModelSettings {
            model_id,
            provider_id,
            reasoning_effort: cfg.model_settings.reasoning_effort,
            summary_mode: None,
            service_tier: None,
            web_search_enabled: false,
            context_window: cfg.model_settings.context_window,
        }
    }

    /// Returns true if the current model supports image input.
    pub fn current_model_supports_vision(&self) -> bool {
        let ms = self.effective_model_settings();
        let cfg = self.runtime.effective_config();
        cfg.models
            .iter()
            .find(|m| m.model_id == ms.model_id)
            .map(|m| m.supports_vision())
            .unwrap_or(false)
    }
    pub fn update_autocomplete(&mut self) {
        let text = self.composer.trimmed_text();
        if !text.starts_with('/') {
            self.autocomplete.clear();
            return;
        }

        let ms = self.effective_model_settings();
        let cur_reasoning = ms.reasoning_effort.as_str();
        let cur_model_key = format!("{}/{}", ms.provider_id, ms.model_id);
        let cfg_models = self.runtime.effective_config().models.clone();

        let mut all: Vec<AutocompleteItem> = Vec::new();

        // ── static commands ───────────────────────────────────────────────────
        for cmd in &[
            "/approve ask",
            "/approve auto",
            "/approve manual",
            "/approve toggle",
            "/approve",
            "/approvals",
            "/compact",
            "/exit",
            "/fork",
            "/help",
            "/interrupt",
            "/new",
            "/open ",
            "/patterns",
            "/patterns clear",
            "/patterns delete ",
            "/quit",
            "/reload",
            "/resume ",
            "/speculate ",
            "/threads",
        ] {
            all.push(AutocompleteItem::simple(*cmd));
        }

        all.push(AutocompleteItem::simple("/model"));
        for preset in &cfg_models {
            let key = format!("{}/{}", preset.provider_id, preset.model_id);
            let value = format!("/model {key}");
            let marker = if key == cur_model_key { "  ←" } else { "" };
            let mods_str = preset
                .modalities
                .iter()
                .map(|m| format!("{m:?}").to_lowercase())
                .collect::<Vec<_>>()
                .join(", ");
            let label = format!("{value}  [{mods_str}]{marker}");
            all.push(AutocompleteItem::labeled(value, label));
        }

        // ── /reasoning: sorted auto → off → low → medium → high ──────────────
        all.push(AutocompleteItem::simple("/reasoning"));
        for level in &["auto", "off", "low", "medium", "high"] {
            let value = format!("/reasoning {level}");
            let marker = if *level == cur_reasoning { "  ←" } else { "" };
            let label = format!("{value}{marker}");
            all.push(AutocompleteItem::labeled(value, label));
        }

        // ── /threads: list known threads as /resume <id> entries ─────────────
        for thread in self.threads.threads() {
            let title = thread_label(thread);
            let dir = thread
                .cwd
                .as_deref()
                .map(basename)
                .filter(|d| !d.is_empty())
                .map(|d| format!("  │  {d}"))
                .unwrap_or_default();
            let age = relative_time_ago(thread.updated_at);
            let id = thread.thread_id.clone();
            let marker = if id == self.current_thread_id {
                "  ←"
            } else {
                ""
            };
            let value = format!("/resume {id}");
            let display = format!("/threads  {title}{dir}  {age}{marker}");
            all.push(AutocompleteItem::labeled(value, display));
        }

        let filtered: Vec<AutocompleteItem> =
            super::autocomplete::filter_and_rank(all, text.as_str());
        self.autocomplete.set_suggestions(filtered);
    }

    /// Move selection down (or wrap), fill the composer.
    pub fn autocomplete_select_next(&mut self) {
        if let Some(value) = self.autocomplete.select_next() {
            self.composer.set_text(value);
        }
    }

    /// Move selection up (or wrap), fill the composer.
    pub fn autocomplete_select_prev(&mut self) {
        if let Some(value) = self.autocomplete.select_prev() {
            self.composer.set_text(value);
        }
    }

    pub async fn toggle_auto_approve_tools(&mut self) {
        let enable = !self.auto_approve_tools_enabled();
        self.set_auto_approve_tools(enable).await;
    }

    pub async fn set_auto_approve_tools(&mut self, enabled: bool) {
        self.approval
            .set_policy_override(super::approval::ApprovalState::override_for_auto(enabled));
        // Propagate immediately to any running turn so it takes effect on the
        // next tool-call evaluation without needing to restart.
        let _ = self
            .runtime
            .set_thread_approval_policy(
                &self.current_thread_id,
                self.approval.policy_override().cloned(),
            )
            .await;
        self.status_message = format!("Approvals: {}", self.approval_mode_label());
        self.error_message = None;
    }

    pub fn auto_approve_tools_enabled(&self) -> bool {
        self.approval
            .auto_enabled(&self.runtime.effective_config().approval_policy)
    }

    pub fn approval_mode_label(&self) -> &'static str {
        self.approval
            .mode_label(&self.runtime.effective_config().approval_policy)
    }

    fn current_approval_policy_override(&self) -> Option<ApprovalPolicy> {
        self.approval.policy_override().cloned()
    }

    pub async fn interrupt_active_turn(&mut self) -> Result<()> {
        let Some(turn_id) = self.active_turn_id.clone() else {
            self.status_message = "No active turn to interrupt".into();
            self.error_message = None;
            return Ok(());
        };
        self.runtime
            .interrupt_turn(TurnInterruptParams {
                thread_id: self.current_thread_id.clone(),
                turn_id: turn_id.clone(),
            })
            .await?;
        self.status_message = format!("Interrupt requested for {}", short_turn_id(&turn_id));
        self.error_message = None;
        Ok(())
    }

    pub async fn resolve_pending_approval(&mut self, decision: ApprovalDecision) -> Result<()> {
        let Some(approval) = self.approval.pending().cloned() else {
            return Ok(());
        };
        self.runtime
            .resolve_approval(&approval.approval.request.approval_id, decision, None)
            .await?;
        self.status_message = format!(
            "{decision:?} {}",
            approval.approval.request.title.to_lowercase()
        );
        self.error_message = None;
        self.approval.set_pending(None);
        Ok(())
    }

    /// Approve the current pending request AND enable auto-approve for future requests.
    pub async fn resolve_pending_approval_auto(&mut self) -> Result<()> {
        self.set_auto_approve_tools(true).await;
        self.resolve_pending_approval(ApprovalDecision::Approved)
            .await
    }

    pub async fn handle_runtime_event(&mut self, event: RuntimeEvent) -> Result<()> {
        match event.kind {
            RuntimeEventKind::ThreadStarted
            | RuntimeEventKind::TurnStarted
            | RuntimeEventKind::TurnCompleted
            | RuntimeEventKind::TurnFailed => {
                self.refresh_threads().await?;
            }
            RuntimeEventKind::TokenUsageUpdate => {
                // Live token usage update during streaming — replace the
                // streaming-turn accumulator (values are cumulative, not deltas).
                // Ignore late events for non-active turns so we never repopulate
                // the streaming bucket after TurnCompleted has folded the totals
                // into `token_usage`.
                if event.turn_id == self.active_turn_id {
                    let input = event
                        .payload
                        .get("inputTokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let output = event
                        .payload
                        .get("outputTokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let cached = event
                        .payload
                        .get("cachedTokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    if input > 0 || output > 0 {
                        self.streaming_turn_usage = TokenUsage {
                            input_tokens: input,
                            output_tokens: output,
                            cached_tokens: cached,
                        };
                    }
                }
            }
            _ => {}
        }

        if event.thread_id.as_deref() != Some(self.current_thread_id.as_str()) {
            if let Some(child_thread) = event.thread_id.as_deref() {
                if let Some(child) = self.activity.child_agent_for_thread(child_thread).cloned() {
                    match event.kind {
                        RuntimeEventKind::TurnCompleted => {
                            let status = match event.payload.get(KEY_STATUS).and_then(Value::as_str)
                            {
                                Some("Interrupted") => ChildAgentStatus::Interrupted,
                                _ => ChildAgentStatus::Completed,
                            };
                            self.activity.set_child_agent_status(child_thread, status);
                            self.upsert_sub_agent_transcript_status(
                                &child.parent_tool_call_id,
                                child_thread,
                                &child.label,
                                status,
                            );
                        }
                        RuntimeEventKind::TurnFailed => {
                            self.activity
                                .set_child_agent_status(child_thread, ChildAgentStatus::Failed);
                            self.upsert_sub_agent_transcript_status(
                                &child.parent_tool_call_id,
                                child_thread,
                                &child.label,
                                ChildAgentStatus::Failed,
                            );
                        }
                        _ => {}
                    }
                }
            }
            return Ok(());
        }

        match event.kind {
            RuntimeEventKind::ThreadStarted => {
                if let Ok(metadata) =
                    serde_json::from_value::<ThreadMetadata>(event.payload.clone())
                {
                    self.current_thread_meta = Some(metadata.clone());
                    self.status_message = format!("Started {}", thread_label(&metadata));
                    self.error_message = None;
                }
            }
            RuntimeEventKind::TurnStarted => {
                self.active_turn_id = event.turn_id.clone();
                self.streaming_turn_usage = TokenUsage::default();
                self.activity.start_turn(std::time::Instant::now());
                self.status_message = "Thinking…".into();
                self.error_message = None;
                if let Some(turn_id) = event.turn_id.as_deref() {
                    self.start_working_entry(turn_id);
                }
            }
            RuntimeEventKind::ItemStarted => self.handle_item_started(&event)?,
            RuntimeEventKind::ItemUpdated => self.handle_item_updated(&event),
            RuntimeEventKind::ItemCompleted => self.handle_item_completed(&event)?,
            RuntimeEventKind::ApprovalRequested => {
                let pending = serde_json::from_value::<PendingApproval>(event.payload.clone())?;
                let preview = format_approval_preview(&pending.request.action);
                let summary_line = preview
                    .lines()
                    .next()
                    .unwrap_or(&pending.request.justification)
                    .to_string();
                self.approval.set_pending(Some(PendingApprovalView {
                    approval: pending.clone(),
                    action_preview: preview,
                }));
                self.activity
                    .set_blocked(super::activity::BlockedReason::Approval);
                self.status_message = format!("Approval required: {}", pending.request.title);
                self.error_message = None;
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Status,
                    "Approval",
                    &format!("Requested: {summary_line}"),
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::ApprovalResolved => {
                let resolution =
                    serde_json::from_value::<ApprovalResolution>(event.payload.clone())?;
                self.remove_latest_approval_request_entry();
                let summary = self
                    .approval
                    .pending()
                    .map(|approval| {
                        format!(
                            "{:?}: {}",
                            resolution.decision,
                            approval
                                .action_preview
                                .lines()
                                .next()
                                .unwrap_or(&approval.approval.request.title)
                        )
                    })
                    .unwrap_or_else(|| format!("{:?}", resolution.decision));
                self.approval.set_pending(None);
                self.activity.clear_blocked();
                self.status_message = format!("Approval {:?}", resolution.decision);
                self.error_message = None;
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Status,
                    "Approval",
                    &summary,
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::TurnCompleted => {
                if let Some(turn_id) = event.turn_id.as_deref() {
                    self.finalize_working_entry(turn_id, "Done");
                }
                self.active_turn_id = None;
                self.approval.set_pending(None);
                self.activity.end_turn();
                self.remove_thinking_placeholder();
                if let Some(thread) = self.current_thread_meta.as_mut() {
                    thread.status = super::super::domain::ThreadStatus::Idle;
                }
                // Accumulate token usage from the completed turn. The payload
                // is a TurnCompleted projection, not a full TurnMetadata, so we
                // pull `tokenUsage` directly rather than deserializing the
                // whole struct (which would fail on missing fields).
                let actual_usage = event
                    .payload
                    .get("tokenUsage")
                    .and_then(|v| serde_json::from_value::<TokenUsage>(v.clone()).ok());
                let estimated_usage = event
                    .payload
                    .get("estimatedTokenUsage")
                    .and_then(|v| serde_json::from_value::<TokenUsage>(v.clone()).ok());
                if let Some(chosen_usage) = actual_usage
                    .filter(|u| u.input_tokens > 0 || u.output_tokens > 0 || u.cached_tokens > 0)
                    .or(estimated_usage)
                {
                    self.latest_prompt_input_tokens = chosen_usage.input_tokens;
                    self.token_usage.input_tokens += chosen_usage.input_tokens;
                    self.token_usage.output_tokens += chosen_usage.output_tokens;
                    self.token_usage.cached_tokens += chosen_usage.cached_tokens;
                }
                // Clear the streaming accumulator — the authoritative totals are
                // now in token_usage from the completed-turn metadata above.
                self.streaming_turn_usage = TokenUsage::default();
                self.status_message = "Ready".into();
                self.error_message = None;
            }
            RuntimeEventKind::TurnFailed => {
                if let Some(turn_id) = event.turn_id.as_deref() {
                    self.finalize_working_entry(turn_id, "Failed after");
                }
                self.active_turn_id = None;
                // Preserve the streaming turn's prompt token count so the
                // context-remaining % in the status bar stays accurate after
                // an interrupt (Ctrl+C). Without this, `streaming_turn_usage`
                // is zeroed and `latest_prompt_input_tokens` still holds the
                // previous completed turn's value, making ctx% jump backwards.
                if self.streaming_turn_usage.input_tokens > 0 {
                    self.latest_prompt_input_tokens = self.streaming_turn_usage.input_tokens;
                }
                self.streaming_turn_usage = TokenUsage::default();
                self.approval.set_pending(None);
                self.activity.end_turn();
                self.remove_thinking_placeholder();
                let kind_label = event
                    .payload
                    .get(KEY_KIND)
                    .and_then(|v| v.as_str())
                    .unwrap_or("Error");
                let reason = event
                    .payload
                    .get(KEY_REASON)
                    .and_then(|v| v.as_str())
                    .unwrap_or("turn failed");
                self.error_message = Some(format!("{kind_label}: {reason}"));
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Error,
                    kind_label,
                    reason,
                    Some(event.emitted_at / 1000),
                );
                if let Some(thread) = self.current_thread_meta.as_mut() {
                    thread.status = super::super::domain::ThreadStatus::Idle;
                }
            }
            RuntimeEventKind::Warning => {
                // Warnings are informational — render as Status, not Error,
                // and do NOT override the error_message field.
                let msg = cod_error::humanize_warning(&event.payload);
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Status,
                    "Warning",
                    &msg,
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::Disconnected => {
                let msg = cod_error::humanize_warning(&event.payload);
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Error,
                    "Disconnected",
                    &msg,
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::CompactionStatus => {
                if event.thread_id.as_deref() != Some(&self.current_thread_id) {
                    return Ok(());
                }
                let status = event
                    .payload
                    .get(KEY_STATUS)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let msg = event
                    .payload
                    .get(KEY_MESSAGE)
                    .and_then(|v| v.as_str())
                    .unwrap_or("Compacting…");

                if !is_background_compaction_status_event(&event, &self.current_thread_id) {
                    // Always update the status message (including "started")
                    // so the user sees "Compacting…" similar to "Thinking…".
                    self.status_message = msg.to_string();
                    return Ok(());
                }

                // Auto-compaction runs out-of-band and mutates persisted
                // thread items. Reload the current thread on completion so the
                // transcript and caches reflect the compacted state
                // immediately, with fresh token counts and ctx%.
                if status == "completed"
                    && event.thread_id.as_deref() == Some(&self.current_thread_id)
                {
                    self.load_thread(&self.current_thread_id.clone()).await?;
                    // Re-estimate context usage from the compacted items
                    // so ctx% reflects the reduced context.
                    self.recalculate_context_from_items().await;
                }
                // Always show status (including "started") so
                // "Compacting…" is visible like "Thinking…".
                self.status_message = msg.to_string();
                let kind = if status == "failed" {
                    EntryKind::Error
                } else {
                    EntryKind::Status
                };
                self.push_status_entry(
                    event.event_id,
                    kind,
                    "Compaction",
                    msg,
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::ChildAgentSpawned => {
                // Sub-agent visibility: register the child in the activity
                // tree so the panel can render its lifecycle. Subsequent
                // TurnCompleted/TurnFailed events on the child's thread_id
                // will update the status (handled at the top of this match).
                let parent = event
                    .payload
                    .get("parentThreadId")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if parent != self.current_thread_id {
                    return Ok(());
                }
                let parent_call = event
                    .payload
                    .get("parentToolCallId")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let child_tid = event
                    .payload
                    .get("childThreadId")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let child_turn = event
                    .payload
                    .get("childTurnId")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let label = event
                    .payload
                    .get("label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("agent")
                    .to_string();
                self.activity.start_child_agent(
                    parent_call.clone(),
                    child_tid.clone(),
                    child_turn,
                    label.clone(),
                    std::time::Instant::now(),
                );
                self.upsert_sub_agent_transcript_status(
                    &parent_call,
                    &child_tid,
                    &label,
                    ChildAgentStatus::Running,
                );
            }
            RuntimeEventKind::TokenUsageUpdate => {
                // Live streaming updates are handled in the first match above;
                // this arm exists only to keep the match exhaustive.
            }
            RuntimeEventKind::SpeculativeCandidateStarted => {
                let total = event
                    .payload
                    .get("totalCandidates")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let idx = event
                    .payload
                    .get("candidateIndex")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                self.status_message = format!("🔮 Exploring approach {}/{}…", idx + 1, total);
            }
            RuntimeEventKind::SpeculativeCandidateCompleted => {
                let label = event
                    .payload
                    .get("approachLabel")
                    .and_then(|v| v.as_str())
                    .unwrap_or("approach");
                let elapsed = event
                    .payload
                    .get("elapsedMs")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                self.push_status_entry(
                    event.event_id.clone(),
                    EntryKind::Status,
                    "Speculative",
                    &format!("✅ \"{}\" completed ({}ms)", label, elapsed),
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::SpeculativeJudgeStarted => {
                let status = event
                    .payload
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("judging");
                if status == "judging" {
                    let n = event
                        .payload
                        .get("candidateCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    self.status_message = format!("🏛️  Judging {} candidate approaches…", n);
                } else {
                    self.status_message = "🔮 Spawning candidate explorers…".into();
                }
            }
            RuntimeEventKind::SpeculativeJudgeCompleted => {
                let selected = event
                    .payload
                    .get("selectedCandidateId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let rationale = event
                    .payload
                    .get("rationale")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                self.status_message = format!("✨ Selected: {}", selected);
                self.push_status_entry(
                    event.event_id.clone(),
                    EntryKind::Status,
                    "Judge",
                    &format!("Selected {} — {}", selected, rationale),
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::CheckpointReviewStarted => {
                let files: Vec<String> = event
                    .payload
                    .get("filesReviewed")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                let file_count = files.len();
                self.status_message = format!(
                    "🔍 Reviewing {} file change{}…",
                    file_count,
                    if file_count == 1 { "" } else { "s" }
                );
            }
            RuntimeEventKind::CheckpointReviewCompleted => {
                let approved = event
                    .payload
                    .get("approved")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let issue_count = event
                    .payload
                    .get("issueCount")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                let (label, body) = if approved {
                    ("✅ Review", "Changes approved".to_string())
                } else {
                    (
                        "⚠️  Review",
                        format!(
                            "{} issue{} found — feedback injected",
                            issue_count,
                            if issue_count == 1 { "" } else { "s" }
                        ),
                    )
                };
                self.push_status_entry(
                    event.event_id.clone(),
                    EntryKind::Status,
                    label,
                    &body,
                    Some(event.emitted_at / 1000),
                );
                self.status_message = if approved {
                    "✅ Review passed".into()
                } else {
                    format!(
                        "⚠️  Review: {} issue{}",
                        issue_count,
                        if issue_count == 1 { "" } else { "s" }
                    )
                };
            }
        }

        Ok(())
    }

    fn handle_item_started(&mut self, event: &RuntimeEvent) -> Result<()> {
        use anyhow::anyhow;
        let item_id = event
            .payload
            .get("itemId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing itemId"))?;
        let kind = event
            .payload
            .get(KEY_KIND)
            .and_then(|v| v.as_str())
            .unwrap_or(EVENT_KIND_AGENT_MESSAGE);

        let entry_kind = match kind {
            "USER_MESSAGE" => EntryKind::User,
            k if k == EVENT_KIND_AGENT_MESSAGE => EntryKind::Assistant,
            k if k == EVENT_KIND_REASONING_TEXT || k == EVENT_KIND_REASONING_SUMMARY => {
                EntryKind::Reasoning
            }
            _ => EntryKind::Status,
        };
        let title = match entry_kind {
            EntryKind::User => LABEL_USER,
            EntryKind::Assistant => LABEL_ASSISTANT,
            EntryKind::Reasoning => LABEL_REASONING,
            _ => LABEL_RUNTIME,
        };
        // Once real content starts, the thinking placeholder is no longer needed.
        if entry_kind == EntryKind::Assistant {
            self.remove_thinking_placeholder();
        }
        // Skip creating a duplicate entry for USER_MESSAGE — it's already
        // pushed eagerly in submit_composer with the actual body text.
        if entry_kind != EntryKind::User {
            self.upsert_transcript_entry(TranscriptEntry {
                item_id: item_id.to_string(),
                turn_id: event.turn_id.clone(),
                tool_call_id: None,
                kind: entry_kind,
                title: title.into(),
                body: String::new(),
                timestamp: Some(event.emitted_at / 1000),
                completed_at: None,
                pending: true,
                collapsed: false,
            });
        }

        // ── Contextual status + structured activity tracking ───────────────
        // `activity` (the reducer) tracks tool calls in flight as a set with
        // start times; `status_message` drives the status bar.
        let status = match kind {
            "TOOL_CALL" => {
                let tool_name = event
                    .payload
                    .get(KEY_TOOL_NAME)
                    .and_then(|v| v.as_str())
                    .unwrap_or(LABEL_TOOL_FALLBACK);
                let tool_call_id = event
                    .payload
                    .get(KEY_TOOL_CALL_ID)
                    .and_then(|v| v.as_str())
                    .unwrap_or(item_id);
                // Best-effort short label: first arg (path/command) if present.
                let hint = event
                    .payload
                    .get(KEY_ARGUMENTS)
                    .and_then(|a| {
                        a.get("path")
                            .or_else(|| a.get("command"))
                            .or_else(|| a.get("pattern"))
                    })
                    .and_then(|v| v.as_str())
                    .map(|s| {
                        // Trim to a short display name (basename for paths)
                        let short = s.rsplit('/').next().unwrap_or(s);
                        if short.len() > 40 {
                            format!("{}…", &short[..37])
                        } else {
                            short.to_string()
                        }
                    });
                self.activity.start_tool(
                    tool_call_id,
                    tool_name,
                    hint.clone(),
                    std::time::Instant::now(),
                );
                let label = match &hint {
                    Some(h) => format!("⚙ {} {}", tool_name, h),
                    None => format!("⚙ {}", tool_name),
                };
                format!("{}…", label)
            }
            "COMMAND_EXECUTION" => {
                let cmd = event
                    .payload
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("cmd");
                let short_cmd = if cmd.len() > 50 {
                    format!("{}…", &cmd[..47])
                } else {
                    cmd.to_string()
                };
                format!("$ {}", short_cmd)
            }
            "FILE_CHANGE" => {
                let path = event
                    .payload
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("file");
                format!("✏  {path}")
            }
            _ => "Streaming response…".into(),
        };
        self.status_message = status;
        self.error_message = None;
        // Do NOT force auto_scroll here — if the user scrolled up, respect that.
        Ok(())
    }

    fn handle_item_updated(&mut self, event: &RuntimeEvent) {
        let Some(item_id) = event.payload.get("itemId").and_then(|v| v.as_str()) else {
            return;
        };
        let delta = event
            .payload
            .get("delta")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if let Some(index) = self.transcript_index.get(item_id).copied() {
            self.transcript[index].body.push_str(delta);
            self.invalidate_transcript_cache();
        } else {
            let kind = event
                .payload
                .get(KEY_KIND)
                .and_then(|v| v.as_str())
                .unwrap_or(EVENT_KIND_AGENT_MESSAGE);
            let (entry_kind, title) = match kind {
                k if k == EVENT_KIND_REASONING_TEXT || k == EVENT_KIND_REASONING_SUMMARY => {
                    (EntryKind::Reasoning, LABEL_THINKING)
                }
                _ => (EntryKind::Assistant, LABEL_ASSISTANT),
            };
            self.upsert_transcript_entry(TranscriptEntry {
                item_id: item_id.to_string(),
                turn_id: event.turn_id.clone(),
                tool_call_id: None,
                kind: entry_kind,
                title: title.into(),
                body: delta.to_string(),
                timestamp: Some(event.emitted_at / 1000),
                completed_at: None,
                pending: true,
                collapsed: false,
            });
        }
        self.status_message = "Streaming response…".into();
        // Mark the assistant as actively streaming so the header shows
        // "◆ generating…" when no tool is in flight (the reducer falls back
        // to the streaming label automatically — see ActivityState::header_line).
        self.activity.set_streaming(true);
        // Do NOT force auto_scroll here — if the user scrolled up, respect that.
    }

    fn handle_item_completed(&mut self, event: &RuntimeEvent) -> Result<()> {
        let item = serde_json::from_value::<ConversationItem>(event.payload.clone())?;
        if item.kind == ItemKind::ToolResult {
            let result = serde_json::from_value::<ToolResult>(item.payload.clone())?;
            self.apply_spawn_agent_result_status(&result);
            if tool_result_failed(&result) {
                self.finish_tool_call_entry(
                    &result.tool_call_id,
                    item.created_at,
                    Some(tool_result_status_suffix(&result)),
                );
                self.upsert_transcript_entry(self.failed_tool_result_entry(&item, &result));
            } else {
                // Merge the result into its matching ToolCall entry in-place so that
                // the call and result appear as a single cohesive transcript block.
                // This avoids the separate ToolResult entry and keeps the cache consistent.
                if let Some(call_idx) = self.transcript.iter().rposition(|e| {
                    e.kind == EntryKind::ToolCall
                        && e.tool_call_id.as_deref() == Some(result.tool_call_id.as_str())
                }) {
                    self.merge_tool_result_into_call(call_idx, &item, &result);
                } else {
                    // No matching call found — fall back to standalone result entry.
                    self.remove_latest_tool_context_entries();
                    self.upsert_transcript_entry(entry_from_item(&item));
                }
            }
            // Remove this tool from the in-flight tracker; the reducer keeps
            // any other tools that started in the same parallel batch alive.
            self.activity.finish_tool(&result.tool_call_id);
        } else if matches!(item.kind, ItemKind::ToolCall) {
            let mut entry = entry_from_item(&item);
            entry.pending = true;
            self.start_tool_activity_from_item(&item);
            self.upsert_transcript_entry(entry);
        } else if matches!(item.kind, ItemKind::AgentMessage) {
            self.upsert_transcript_entry(entry_from_item(&item));
            self.activity.set_streaming(false);
        } else if matches!(item.kind, ItemKind::UserMessage) {
            // Replace the eager placeholder from submit_composer with the
            // real persisted entry (correct item_id, non-pending).
            self.remove_user_pending_placeholder();
            self.upsert_transcript_entry(entry_from_item(&item));
        } else {
            self.upsert_transcript_entry(entry_from_item(&item));
        }
        // Do NOT force auto_scroll — respect user scroll position.
        Ok(())
    }

    fn upsert_loaded_item(&mut self, item: &ConversationItem) {
        if item.kind == ItemKind::ToolResult {
            if let Ok(result) = serde_json::from_value::<ToolResult>(item.payload.clone()) {
                if let Some(call_idx) = self.transcript.iter().rposition(|e| {
                    e.kind == EntryKind::ToolCall
                        && e.tool_call_id.as_deref() == Some(result.tool_call_id.as_str())
                }) {
                    if tool_result_failed(&result) {
                        self.finish_tool_call_entry(
                            &result.tool_call_id,
                            item.created_at,
                            Some(tool_result_status_suffix(&result)),
                        );
                        self.upsert_transcript_entry(self.failed_tool_result_entry(item, &result));
                    } else {
                        self.merge_tool_result_into_call(call_idx, item, &result);
                    }
                    return;
                }
            }
        }

        self.upsert_transcript_entry(entry_from_item(item));
    }

    fn merge_tool_result_into_call(
        &mut self,
        call_idx: usize,
        item: &ConversationItem,
        result: &ToolResult,
    ) {
        let result_body = format_tool_result(
            item.payload.get(KEY_OUTPUT),
            item.payload.get(KEY_ERROR_MESSAGE),
        );
        let base_suffix = if tool_result_not_spawned(result) {
            "skipped"
        } else {
            STATUS_DONE
        };

        // For collapsed entries, extract a brief summary from the result output
        // so the header reads e.g. "read_file  main.rs → done  (142 lines)".
        let suffix = if self.transcript.get(call_idx).is_some_and(|e| e.collapsed) {
            let output = item.payload.get(KEY_OUTPUT);
            let summary = collapsed_result_summary(output);
            if summary.is_empty() {
                base_suffix.to_string()
            } else {
                format!("{base_suffix}  ({summary})")
            }
        } else {
            base_suffix.to_string()
        };

        {
            let call = &mut self.transcript[call_idx];
            if !call.body.ends_with('\n') {
                call.body.push('\n');
            }
            call.body.push_str("─── result ───\n");
            call.body.push_str(&result_body);
            call.title = title_with_status_suffix(&call.title, &suffix);
            call.pending = false;
            call.completed_at = Some(item.created_at);
        }
        self.invalidate_transcript_cache();
        // Remove any Approval status entries that were inserted after the call.
        self.remove_approval_entries_after(call_idx);
    }

    fn finish_tool_call_entry(
        &mut self,
        tool_call_id: &str,
        completed_at: i64,
        suffix: Option<&str>,
    ) {
        let Some(call_idx) = self.transcript.iter().rposition(|e| {
            e.kind == EntryKind::ToolCall && e.tool_call_id.as_deref() == Some(tool_call_id)
        }) else {
            return;
        };

        let call = &mut self.transcript[call_idx];
        if let Some(suffix) = suffix {
            call.title = title_with_status_suffix(&call.title, suffix);
        }
        call.pending = false;
        call.completed_at = Some(completed_at);
        self.invalidate_transcript_cache();
    }

    fn start_tool_activity_from_item(&mut self, item: &ConversationItem) {
        let tool_name = item
            .payload
            .get(KEY_TOOL_NAME)
            .and_then(Value::as_str)
            .unwrap_or(LABEL_TOOL_FALLBACK);
        let tool_call_id = item
            .payload
            .get(KEY_TOOL_CALL_ID)
            .and_then(Value::as_str)
            .unwrap_or(item.item_id.as_str());
        let hint = tool_hint_from_arguments(item.payload.get(KEY_ARGUMENTS));

        self.activity.start_tool(
            tool_call_id,
            tool_name,
            hint.clone(),
            std::time::Instant::now(),
        );
        self.status_message = match hint {
            Some(h) => format!("⚙ {tool_name} {h}…"),
            None => format!("⚙ {tool_name}…"),
        };
        self.error_message = None;
    }

    pub fn upsert_transcript_entry(&mut self, mut entry: TranscriptEntry) {
        let is_working = entry.item_id.starts_with(WORKING_ENTRY_ID_PREFIX);
        if let Some(index) = self.transcript_index.get(&entry.item_id).copied() {
            let previous = &self.transcript[index];
            let should_reposition_completed_reasoning =
                previous.pending && !entry.pending && entry.kind == EntryKind::Reasoning;
            if previous.pending && !entry.pending {
                entry.completed_at = entry.completed_at.or(entry.timestamp);
                entry.timestamp = previous.timestamp.or(entry.timestamp);
            }
            self.transcript[index] = entry;
            if should_reposition_completed_reasoning {
                self.reposition_reasoning_before_same_turn_assistant(index);
            }
        } else {
            let index = self.transcript_insert_index(&entry);
            self.transcript.insert(index, entry);
            self.rebuild_transcript_index();
        }
        // Keep the active (pending) Working entry pinned to the bottom of the
        // transcript: when any other entry is added, slide the timer past it
        // so it always renders as the last row of the current turn.
        if !is_working {
            self.bump_pending_working_entry_to_end();
        }
        self.invalidate_transcript_cache();
    }

    fn transcript_insert_index(&self, entry: &TranscriptEntry) -> usize {
        if entry.pending || entry.kind != EntryKind::Reasoning {
            return self.transcript.len();
        }
        let Some(turn_id) = entry.turn_id.as_deref() else {
            return self.transcript.len();
        };
        self.transcript
            .iter()
            .position(|existing| {
                existing.turn_id.as_deref() == Some(turn_id)
                    && existing.kind == EntryKind::Assistant
            })
            .unwrap_or(self.transcript.len())
    }

    fn reposition_reasoning_before_same_turn_assistant(&mut self, index: usize) {
        if index >= self.transcript.len()
            || self.transcript[index].kind != EntryKind::Reasoning
            || self.transcript[index].pending
        {
            return;
        }
        let Some(turn_id) = self.transcript[index].turn_id.clone() else {
            return;
        };

        let entry = self.transcript.remove(index);
        let insert_at = self
            .transcript
            .iter()
            .position(|existing| {
                existing.turn_id.as_deref() == Some(turn_id.as_str())
                    && existing.kind == EntryKind::Assistant
            })
            .unwrap_or(self.transcript.len());
        self.transcript.insert(insert_at, entry);
        self.rebuild_transcript_index();
    }

    fn bump_pending_working_entry_to_end(&mut self) {
        let Some(turn_id) = self.active_turn_id.clone() else {
            return;
        };
        let id = working_entry_id(&turn_id);
        let Some(idx) = self.transcript_index.get(&id).copied() else {
            return;
        };
        if idx + 1 >= self.transcript.len() {
            return;
        }
        let entry = self.transcript.remove(idx);
        self.transcript.push(entry);
        self.rebuild_transcript_index();
    }

    pub fn push_status_entry(
        &mut self,
        item_id: String,
        kind: EntryKind,
        title: &str,
        body: &str,
        timestamp: Option<i64>,
    ) {
        self.upsert_transcript_entry(TranscriptEntry {
            item_id,
            turn_id: None,
            tool_call_id: None,
            kind,
            title: title.into(),
            body: body.into(),
            timestamp,
            completed_at: None,
            pending: false,
            collapsed: false,
        });
        // Do NOT force auto_scroll — respect user scroll position.
    }

    /// Insert the in-transcript "Working" timer entry on TurnStarted. One entry
    /// per turn (id keyed by turn_id) so prior turns' final durations stick
    /// around. The title is refreshed each tick via `update_working_entry`;
    /// on TurnCompleted/TurnFailed it is frozen with total duration.
    fn start_working_entry(&mut self, turn_id: &str) {
        self.upsert_transcript_entry(TranscriptEntry {
            item_id: working_entry_id(turn_id),
            turn_id: Some(turn_id.to_string()),
            tool_call_id: None,
            kind: EntryKind::Status,
            title: "Working".into(),
            body: String::new(),
            timestamp: None,
            completed_at: None,
            pending: true,
            collapsed: false,
        });
    }

    /// Refresh the active turn's working entry title (elapsed time) and body
    /// (a one-line description of what the agent is currently doing). No-op
    /// when no turn is active.
    pub fn update_working_entry(&mut self) {
        let Some(turn_id) = self.active_turn_id.clone() else {
            return;
        };
        let Some(elapsed) = self.activity.turn_elapsed(std::time::Instant::now()) else {
            return;
        };
        let id = working_entry_id(&turn_id);
        let Some(idx) = self.transcript_index.get(&id).copied() else {
            return;
        };
        let title = format!(
            "Working · {}",
            super::types::format_duration(elapsed.as_secs() as i64)
        );
        let body = self.working_status_description();
        let entry = &mut self.transcript[idx];
        let mut changed = false;
        if entry.title != title {
            entry.title = title;
            changed = true;
        }
        if entry.body != body {
            entry.body = body;
            changed = true;
        }
        if changed {
            self.invalidate_transcript_cache();
        }
    }

    /// Build a short description of what the agent is currently doing, derived
    /// from the activity tracker. Used as the body line of the Working entry.
    fn working_status_description(&self) -> String {
        if let Some(reason) = self.activity.blocked() {
            return reason.label().to_string();
        }
        // Sub-agents run alongside other tool activity; surface them first when
        // any are alive so the user sees parallel work.
        let live_children = self
            .activity
            .child_agents()
            .iter()
            .filter(|c| c.status == super::activity::ChildAgentStatus::Running)
            .count();

        let tools = self.activity.tools_in_flight();
        let primary = match tools.len() {
            0 => {
                if self.activity.is_streaming() {
                    "Generating response…".into()
                } else {
                    "Thinking…".into()
                }
            }
            1 => describe_tool_activity(&tools[0]),
            n => {
                let mut seen = std::collections::HashSet::new();
                let names: Vec<&str> = tools
                    .iter()
                    .map(|t| t.tool_name.as_str())
                    .filter(|name| seen.insert(*name))
                    .collect();
                format!("Running {n} tools: {}", names.join(", "))
            }
        };

        let combined = if live_children > 0 {
            let suffix = if live_children == 1 {
                "sub-agent running".to_string()
            } else {
                format!("{live_children} sub-agents running")
            };
            // If the primary already mentions a sub-agent (spawn_agent in flight),
            // don't duplicate the count — just keep the descriptive line.
            if primary.starts_with("Running sub-agent") {
                primary
            } else {
                format!("{primary} · {suffix}")
            }
        } else {
            primary
        };

        truncate_status_line(&combined, 80)
    }

    /// Freeze the working entry for `turn_id` with its final duration and clear
    /// pending. Must be called BEFORE `activity.end_turn()`.
    fn finalize_working_entry(&mut self, turn_id: &str, label: &str) {
        let id = working_entry_id(turn_id);
        let Some(idx) = self.transcript_index.get(&id).copied() else {
            return;
        };
        let title = match self.activity.turn_elapsed(std::time::Instant::now()) {
            Some(elapsed) => format!(
                "{label} · {}",
                super::types::format_duration(elapsed.as_secs() as i64)
            ),
            None => label.to_string(),
        };
        let summary = self.working_done_summary(turn_id);
        let entry = &mut self.transcript[idx];
        entry.title = title;
        entry.body = summary;
        entry.pending = false;
        self.invalidate_transcript_cache();
    }

    /// Summarize what the just-finished turn actually did — quote the
    /// commands run, patterns searched, files edited, etc. so the user sees
    /// the work, not just counts. Empty when the turn produced nothing
    /// beyond a plain text answer.
    fn working_done_summary(&self, turn_id: &str) -> String {
        let current = working_entry_id(turn_id);
        let mut files: Vec<String> = Vec::new();
        let mut commands: Vec<String> = Vec::new();
        let mut searches: Vec<String> = Vec::new();
        let mut fetches: Vec<String> = Vec::new();
        let mut sub_agents: Vec<String> = Vec::new();
        let mut other_tools: Vec<String> = Vec::new();
        for entry in self.transcript.iter().rev() {
            if entry.item_id != current && entry.item_id.starts_with(WORKING_ENTRY_ID_PREFIX) {
                break;
            }
            match entry.kind {
                EntryKind::FileChange => files.push(entry.title.clone()),
                EntryKind::ToolCall => {
                    let title_lower = entry.title.to_lowercase();
                    let name = entry.title.split_whitespace().next().unwrap_or("");
                    if title_lower.starts_with("sub-agent") {
                        sub_agents.push(
                            entry
                                .title
                                .split_once(':')
                                .map(|(_, s)| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .filter(|s| !s.is_empty())
                                .unwrap_or_else(|| "sub-agent".into()),
                        );
                    } else {
                        match name {
                            "bash_exec" | "shell_exec" => {
                                if let Some(cmd) = first_meaningful_line(&entry.body) {
                                    commands.push(cmd);
                                }
                            }
                            "grep_search" => {
                                if let Some(p) = extract_grep_pattern(&entry.body) {
                                    searches.push(p);
                                }
                            }
                            "web_fetch" => {
                                if let Some(u) = extract_first_url(&entry.body) {
                                    fetches.push(u);
                                }
                            }
                            // File-mutating tools already produce a FileChange
                            // entry, so don't double-count them as tool calls.
                            "write_file" | "patch_file" | "create_directory" | "remove_path"
                            | "copy_path" => {}
                            _ if !name.is_empty() => other_tools.push(name.to_string()),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        // Walked the transcript in reverse; flip back to chronological order.
        files.reverse();
        commands.reverse();
        searches.reverse();
        fetches.reverse();
        sub_agents.reverse();
        other_tools.reverse();

        let mut parts: Vec<String> = Vec::new();
        if !files.is_empty() {
            let mut seen = std::collections::HashSet::new();
            let unique: Vec<String> = files
                .into_iter()
                .filter(|f| seen.insert(f.clone()))
                .collect();
            parts.push(format!("Edited {}", join_with_more(&unique, 3)));
        }
        if !commands.is_empty() {
            let quoted: Vec<String> = commands
                .iter()
                .map(|c| format!("`{}`", trim_long(c, 40)))
                .collect();
            parts.push(format!("ran {}", join_with_more(&quoted, 2)));
        }
        if !searches.is_empty() {
            let quoted: Vec<String> = searches
                .iter()
                .map(|s| format!("`{}`", trim_long(s, 30)))
                .collect();
            parts.push(format!("searched {}", join_with_more(&quoted, 2)));
        }
        if !fetches.is_empty() {
            parts.push(format!("fetched {}", join_with_more(&fetches, 2)));
        }
        if !sub_agents.is_empty() {
            parts.push(format!(
                "{} sub-agent{}",
                sub_agents.len(),
                if sub_agents.len() == 1 { "" } else { "s" }
            ));
        }
        if parts.is_empty() && !other_tools.is_empty() {
            let mut seen = std::collections::HashSet::new();
            let unique: Vec<String> = other_tools
                .into_iter()
                .filter(|t| seen.insert(t.clone()))
                .collect();
            parts.push(format!("called {}", join_with_more(&unique, 2)));
        }
        truncate_status_line(&parts.join(", "), 120)
    }

    /// Remove the "thinking" placeholder from the transcript if it is still present.
    fn remove_thinking_placeholder(&mut self) {
        if let Some(index) = self.transcript_index.get(THINKING_PLACEHOLDER_ID).copied() {
            self.transcript.remove(index);
            self.rebuild_transcript_index();
        }
    }

    /// Remove the eager "You" placeholder injected by submit_composer.
    /// Called when the real UserMessage ItemCompleted event arrives so the
    /// placeholder is replaced by the persisted entry (avoiding duplicates).
    fn remove_user_pending_placeholder(&mut self) {
        if let Some(index) = self
            .transcript_index
            .get(USER_PENDING_PLACEHOLDER_ID)
            .copied()
        {
            self.transcript.remove(index);
            self.rebuild_transcript_index();
        }
    }

    fn remove_latest_approval_request_entry(&mut self) {
        let Some(index) = self.transcript.iter().rposition(|entry| {
            entry.kind == EntryKind::Status
                && entry.title == "Approval"
                && entry.body.starts_with("Requested:")
        }) else {
            return;
        };
        self.transcript.remove(index);
        self.rebuild_transcript_index();
    }

    fn remove_latest_tool_context_entries(&mut self) {
        let Some(tool_call_index) = self
            .transcript
            .iter()
            .rposition(|entry| entry.kind == EntryKind::ToolCall)
        else {
            return;
        };

        self.transcript = self
            .transcript
            .iter()
            .enumerate()
            .filter(|(index, entry)| {
                *index != tool_call_index
                    && !(*index > tool_call_index
                        && entry.kind == EntryKind::Status
                        && entry.title == "Approval")
            })
            .map(|(_, entry)| entry.clone())
            .collect();
        self.rebuild_transcript_index();
    }

    /// Remove Approval status entries that appear after `after_index` in the transcript.
    /// Called after merging a tool result into its call entry.
    fn remove_approval_entries_after(&mut self, after_index: usize) {
        let had_approvals = self.transcript[after_index + 1..]
            .iter()
            .any(|e| e.kind == EntryKind::Status && e.title == "Approval");
        if !had_approvals {
            return;
        }
        self.transcript = self
            .transcript
            .iter()
            .enumerate()
            .filter(|(idx, e)| {
                *idx <= after_index || !(e.kind == EntryKind::Status && e.title == "Approval")
            })
            .map(|(_, e)| e.clone())
            .collect();
        self.rebuild_transcript_index();
    }

    fn rebuild_transcript_index(&mut self) {
        self.transcript_index.clear();
        for (index, entry) in self.transcript.iter().enumerate() {
            self.transcript_index.insert(entry.item_id.clone(), index);
        }
        self.invalidate_transcript_cache();
    }

    pub fn transcript_total_lines(&mut self, width: u16) -> usize {
        self.ensure_transcript_render_cache(width);
        self.transcript_render_cache.total_lines
    }

    pub fn transcript_window_lines(
        &mut self,
        width: u16,
        start_line: usize,
        max_lines: usize,
        selection: Option<SelectionRange>,
    ) -> (Vec<Line<'static>>, usize) {
        self.ensure_transcript_render_cache(width);
        if self.transcript_render_cache.entries.is_empty() {
            return transcript_lines(
                &self.transcript,
                self.activity.spinner_tick(),
                width,
                selection,
            );
        }

        let end_line = start_line.saturating_add(max_lines);
        let body_width = (width as usize).saturating_sub(5).max(1);
        let now_ts = chrono::Utc::now().timestamp();
        let first_entry = self
            .transcript_render_cache
            .line_ends
            .partition_point(|&line_end| line_end <= start_line);

        let mut lines = Vec::new();
        let mut prev_timestamp: Option<i64> = None;
        let mut prev_user_timestamp: Option<i64> = None;
        for idx in first_entry..self.transcript_render_cache.entries.len() {
            let entry_end = self.transcript_render_cache.line_ends[idx];
            let entry = &self.transcript_render_cache.entries[idx];
            let entry_start = entry_end.saturating_sub(entry.line_count);
            if entry_start >= end_line {
                break;
            }
            append_cached_transcript_entry_lines(
                &mut lines,
                entry,
                self.activity.spinner_tick(),
                body_width,
                start_line,
                end_line,
                entry_start,
                prev_timestamp,
                prev_user_timestamp,
                now_ts,
            );
            // Track timestamps for duration/gap calculations
            if entry.timestamp.is_some() {
                prev_timestamp = entry.timestamp;
            }
            if entry.kind == EntryKind::User && entry.timestamp.is_some() {
                prev_user_timestamp = entry.timestamp;
            }
        }

        if let Some(sel) = selection {
            let visible_start = start_line;
            let visible_end = start_line.saturating_add(lines.len());
            let selection_start = sel.start_line.max(visible_start);
            let selection_end = sel.end_line.min(visible_end.saturating_sub(1));
            if selection_start <= selection_end {
                for actual_line_idx in selection_start..=selection_end {
                    let line_idx = actual_line_idx - visible_start;
                    let col_from = if actual_line_idx == sel.start_line {
                        sel.start_col
                    } else {
                        0
                    };
                    let col_to = if actual_line_idx == sel.end_line {
                        sel.end_col
                    } else {
                        usize::MAX / 2
                    };
                    let taken = std::mem::replace(&mut lines[line_idx], Line::from(""));
                    lines[line_idx] = apply_char_selection(taken, col_from, col_to);
                }
            }
        }

        (lines, self.transcript_render_cache.total_lines)
    }

    fn transcript_lines_all(
        &mut self,
        width: u16,
        selection: Option<SelectionRange>,
    ) -> (Vec<Line<'static>>, usize) {
        let total = self.transcript_total_lines(width);
        self.transcript_window_lines(width, 0, total.max(1), selection)
    }

    fn ensure_transcript_render_cache(&mut self, width: u16) {
        if !self.transcript_render_cache.dirty && self.transcript_render_cache.width == width {
            return;
        }

        self.transcript_render_cache = build_transcript_render_cache(&self.transcript, width);
    }

    /// Extract plain-text lines from the rendered transcript for word-snapping.
    fn transcript_plain_lines(&mut self, width: u16) -> Vec<String> {
        let (lines, _total) = self.transcript_lines_all(width, None);
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    fn invalidate_transcript_cache(&mut self) {
        self.transcript_render_cache.dirty = true;
    }

    fn failed_tool_result_entry(
        &self,
        item: &ConversationItem,
        result: &ToolResult,
    ) -> TranscriptEntry {
        let tool_name = self
            .transcript
            .iter()
            .rev()
            .find(|entry| {
                entry.kind == EntryKind::ToolCall
                    && entry.tool_call_id.as_deref() == Some(result.tool_call_id.as_str())
            })
            .or_else(|| {
                self.transcript
                    .iter()
                    .rev()
                    .find(|entry| entry.kind == EntryKind::ToolCall)
            })
            .map(|entry| entry.title.clone())
            .unwrap_or_else(|| "tool".to_string());

        let error_value = result.error_message.as_deref().map(Value::from);
        let body = format_tool_result(Some(&result.output), error_value.as_ref());

        TranscriptEntry {
            item_id: item.item_id.clone(),
            turn_id: Some(item.turn_id.clone()),
            tool_call_id: Some(result.tool_call_id.clone()),
            kind: EntryKind::Error,
            title: failed_tool_title(&tool_name, result),
            body,
            timestamp: Some(item.created_at),
            completed_at: None,
            pending: false,
            collapsed: false,
        }
    }

    fn apply_spawn_agent_result_status(&mut self, result: &ToolResult) {
        let Some(child_thread_id) = result
            .output
            .get(KEY_THREAD_ID_SNAKE)
            .or_else(|| result.output.get(KEY_THREAD_ID))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
        else {
            return;
        };
        let child_turn_id = result
            .output
            .get(KEY_TURN_ID_SNAKE)
            .or_else(|| result.output.get(KEY_TURN_ID))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        if !self.activity.is_known_child_thread(&child_thread_id) {
            self.activity.start_child_agent(
                result.tool_call_id.clone(),
                child_thread_id.clone(),
                child_turn_id,
                "sub-agent",
                std::time::Instant::now(),
            );
        }

        let error = result
            .output
            .get(KEY_ERROR)
            .and_then(Value::as_str)
            .or(result.error_message.as_deref());
        let status = if result.ok {
            ChildAgentStatus::Completed
        } else {
            match error {
                Some(STATUS_TIMEOUT) => ChildAgentStatus::TimedOut,
                Some(STATUS_INTERRUPTED) => ChildAgentStatus::Interrupted,
                _ => ChildAgentStatus::Failed,
            }
        };
        self.activity
            .set_child_agent_status(&child_thread_id, status);
        let label = self
            .activity
            .child_agent_for_thread(&child_thread_id)
            .map(|c| c.label.clone())
            .unwrap_or_else(|| "sub-agent".to_string());
        self.upsert_sub_agent_transcript_status(
            &result.tool_call_id,
            &child_thread_id,
            &label,
            status,
        );
    }

    fn upsert_sub_agent_transcript_status(
        &mut self,
        parent_tool_call_id: &str,
        child_thread_id: &str,
        label: &str,
        status: ChildAgentStatus,
    ) {
        let Some(call_idx) = self
            .transcript
            .iter()
            .rposition(|e| e.kind == EntryKind::ToolCall && e.title.starts_with("sub-agent:"))
            .or_else(|| {
                self.transcript.iter().rposition(|e| {
                    e.kind == EntryKind::ToolCall
                        && e.tool_call_id.as_deref() == Some(parent_tool_call_id)
                })
            })
        else {
            return;
        };

        let mut children = self.activity.child_agents().to_vec();
        if children.is_empty() {
            // Defensive fallback when event ordering races before activity
            // registration catches up.
            children.push(super::activity::ChildAgentActivity {
                parent_tool_call_id: parent_tool_call_id.to_string(),
                child_thread_id: child_thread_id.to_string(),
                child_turn_id: String::new(),
                label: label.to_string(),
                status,
                started_at: std::time::Instant::now(),
                finished_at: None,
            });
        }

        let now = std::time::Instant::now();

        let mut section_lines = vec!["sub-agents:".to_string()];

        for child in &children {
            if matches!(child.status, ChildAgentStatus::Completed) {
                continue;
            }
            let status_text = match child.status {
                ChildAgentStatus::Running => {
                    let frame = spinner_frame(self.activity.spinner_tick());
                    let elapsed = child.elapsed(now).as_secs();
                    if elapsed >= 45 {
                        section_lines.push(format!(
                            "- {} {} • {}s • stuck",
                            frame,
                            truncate_chars(&child.label, 64),
                            elapsed
                        ));
                    } else {
                        section_lines.push(format!(
                            "- {} {} • {}s",
                            frame,
                            truncate_chars(&child.label, 64),
                            elapsed
                        ));
                    }
                    continue;
                }
                ChildAgentStatus::Completed => STATUS_DONE,
                ChildAgentStatus::Failed => STATUS_FAILED,
                ChildAgentStatus::Interrupted => STATUS_INTERRUPTED,
                ChildAgentStatus::TimedOut => STATUS_TIMED_OUT,
            };
            let status_icon = match child.status {
                ChildAgentStatus::Running => "⠋",
                ChildAgentStatus::Completed => "✓",
                ChildAgentStatus::Failed => "✗",
                ChildAgentStatus::Interrupted => "◌",
                ChildAgentStatus::TimedOut => "⏱",
            };
            let elapsed_secs = child.elapsed(now).as_secs();
            section_lines.push(format!(
                "- {} {} • {} • {}s",
                status_icon,
                truncate_chars(&child.label, 64),
                status_text,
                elapsed_secs
            ));
        }

        if section_lines.len() == 1 {
            section_lines.push("- ✓ all sub-agents done".to_string());
        }

        let section = format!("{}\n", section_lines.join("\n"));
        let existing = self.transcript[call_idx].body.clone();
        let result_marker = "─── result ───";
        let start_marker = "sub-agents:";

        let mut rebuilt = if let Some(start) = existing.find(start_marker) {
            let end = existing[start..]
                .find(result_marker)
                .map(|rel| start + rel)
                .unwrap_or(existing.len());
            let mut s = String::new();
            s.push_str(&existing[..start]);
            s.push_str(&section);
            if end < existing.len() {
                if !s.ends_with('\n') {
                    s.push('\n');
                }
                s.push_str(existing[end..].trim_start_matches('\n'));
            }
            s
        } else if let Some(result_idx) = existing.find(result_marker) {
            let mut s = String::new();
            s.push_str(&existing[..result_idx]);
            if !s.is_empty() && !s.ends_with('\n') {
                s.push('\n');
            }
            s.push_str(&section);
            s.push_str(&existing[result_idx..]);
            s
        } else {
            let mut s = existing;
            if !s.is_empty() && !s.ends_with('\n') {
                s.push('\n');
            }
            s.push_str(&section);
            s
        };

        if !rebuilt.ends_with('\n') {
            rebuilt.push('\n');
        }
        self.transcript[call_idx].body = rebuilt;

        // Remove duplicate sub-agent sections from other tool-call entries.
        for (idx, entry) in self.transcript.iter_mut().enumerate() {
            if idx == call_idx || entry.kind != EntryKind::ToolCall {
                continue;
            }
            if let Some(start) = entry.body.find(start_marker) {
                let end = entry.body[start..]
                    .find(result_marker)
                    .map(|rel| start + rel)
                    .unwrap_or(entry.body.len());
                let mut trimmed = String::new();
                trimmed.push_str(&entry.body[..start]);
                if end < entry.body.len() {
                    if !trimmed.ends_with('\n') && !trimmed.is_empty() {
                        trimmed.push('\n');
                    }
                    trimmed.push_str(entry.body[end..].trim_start_matches('\n'));
                }
                entry.body = trimmed;
            }
        }
        self.invalidate_transcript_cache();
    }

    pub fn refresh_live_sub_agent_sections(&mut self) -> bool {
        let running = self
            .activity
            .child_agents()
            .iter()
            .any(|c| matches!(c.status, ChildAgentStatus::Running));
        if !running {
            return false;
        }

        let mut changed = false;
        let mut seen = std::collections::HashSet::new();
        let children = self.activity.child_agents().to_vec();
        for child in &children {
            if !seen.insert(child.parent_tool_call_id.clone()) {
                continue;
            }
            self.upsert_sub_agent_transcript_status(
                &child.parent_tool_call_id,
                &child.child_thread_id,
                &child.label,
                child.status,
            );
            changed = true;
        }
        changed
    }
}

fn failed_tool_title(tool_name: &str, result: &ToolResult) -> String {
    let tool_name = tool_name
        .split_once(" → ")
        .map(|(base, _)| base)
        .unwrap_or(tool_name);
    match result.output.get(KEY_ERROR).and_then(Value::as_str) {
        Some(STATUS_TIMEOUT) => format!("{tool_name} timed out"),
        Some(STATUS_INTERRUPTED) => format!("{tool_name} interrupted"),
        _ => format!("{tool_name} failed"),
    }
}

fn tool_result_status_suffix(result: &ToolResult) -> &'static str {
    match result.output.get(KEY_ERROR).and_then(Value::as_str) {
        Some(STATUS_TIMEOUT) => STATUS_TIMED_OUT,
        Some(STATUS_INTERRUPTED) => STATUS_INTERRUPTED,
        _ => STATUS_FAILED,
    }
}

fn title_with_status_suffix(title: &str, suffix: &str) -> String {
    let base = title
        .split_once(" → ")
        .map(|(base, _)| base)
        .unwrap_or(title);
    format!("{base} → {suffix}")
}

/// Extract a brief summary string from a tool result's raw output JSON.
///
/// Used only for collapsed entries so the header shows context, e.g.:
/// - `read_file  main.rs → done  (142 lines)`
/// - `list_dir  src → done  (42 entries)`
/// - `grep_search  /foo/ → done  (5 matches)`
fn collapsed_result_summary(output: Option<&Value>) -> String {
    let Some(output) = output else {
        return String::new();
    };

    // read_file: { path, content }
    if let Some(content) = output.get("content").and_then(Value::as_str) {
        let lines = content.lines().count();
        return format!("{lines} lines");
    }

    // list_dir: { entries: [...], count, truncated }
    if let Some(entries) = output.get("entries").and_then(Value::as_array) {
        let count = output
            .get("count")
            .and_then(Value::as_u64)
            .unwrap_or(entries.len() as u64);
        let truncated = output
            .get("truncated")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let suffix = if truncated { "+" } else { "" };
        return format!("{count}{suffix} entries");
    }

    // grep_search: { matches: [...] }
    if let Some(matches) = output.get("matches").and_then(Value::as_array) {
        let n = matches.len();
        return if n == 1 {
            "1 match".to_string()
        } else {
            format!("{n} matches")
        };
    }

    // web_fetch: { url, status, total_chars }
    if let Some(status) = output.get(KEY_STATUS).and_then(Value::as_u64) {
        if output.get("url").is_some() {
            let chars = output
                .get("total_chars")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            return format!("HTTP {status} · {chars} chars");
        }
    }

    // simple ok: { ok: true, path }
    if output.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        return "ok".to_string();
    }

    String::new()
}
fn tool_hint_from_arguments(arguments: Option<&Value>) -> Option<String> {
    arguments
        .and_then(|a| {
            a.get("path")
                .or_else(|| a.get("command"))
                .or_else(|| a.get("pattern"))
        })
        .and_then(Value::as_str)
        .map(|s| {
            let short = s.rsplit('/').next().unwrap_or(s);
            truncate_chars(short, 40)
        })
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

fn build_transcript_render_cache(entries: &[TranscriptEntry], width: u16) -> TranscriptRenderCache {
    if entries.is_empty() {
        return TranscriptRenderCache {
            width,
            dirty: false,
            entries: Vec::new(),
            line_ends: Vec::new(),
            total_lines: 3,
        };
    }

    let body_width = (width as usize).saturating_sub(5).max(1);
    let mut cached_entries: Vec<CachedTranscriptEntry> = Vec::with_capacity(entries.len());
    let mut line_ends = Vec::with_capacity(entries.len());
    let mut total_lines = 0usize;
    let mut tool_names_by_call_id: HashMap<String, String> = HashMap::new();

    for (i, entry) in entries.iter().enumerate() {
        // System prompt entries are persisted for debugging/auditing but should
        // not be rendered in the TUI — they are internal LLM context and would
        // otherwise appear as a huge empty-looking "◈ System" block.
        if entry.kind == EntryKind::System {
            continue;
        }

        // ToolResult entries that were successfully merged into their ToolCall are
        // suppressed: the ToolCall body already contains the result (mutated in
        // handle_item_completed), so rendering the result separately would be redundant.
        if entry.kind == EntryKind::ToolResult {
            continue;
        }

        let use_markdown = matches!(
            entry.kind,
            EntryKind::User | EntryKind::Assistant | EntryKind::Summary | EntryKind::Reasoning
        );

        let use_read_file = matches!(entry.kind, EntryKind::ToolCall | EntryKind::ToolResult)
            && is_read_file_body(&entry.body);
        let use_diff = matches!(entry.kind, EntryKind::ToolCall | EntryKind::ToolResult)
            && is_diff_body(&entry.body);

        let is_working_timer = entry.item_id.starts_with(WORKING_ENTRY_ID_PREFIX);
        let (raw_body, body_lines): (String, Vec<String>) =
            if entry.body.is_empty() && is_working_timer {
                // Working timer entry uses the title for everything; render no body row.
                (String::new(), Vec::new())
            } else if entry.body.is_empty() && entry.pending {
                // Build an animated working indicator that matches the visual render.
                let spinner = super::types::spinner_frame(0);
                let working_text = match entry.kind {
                    EntryKind::Assistant => "thinking",
                    EntryKind::ToolCall => "calling tool",
                    EntryKind::ToolResult => "waiting",
                    EntryKind::Reasoning => "reasoning",
                    _ => "working",
                };
                (String::new(), vec![format!("{spinner}  {working_text}…")])
            } else if use_markdown {
                // Render now to get the correct line count for scroll arithmetic.
                // The rendered Lines themselves are discarded; raw_body is kept so
                // append_cached_transcript_entry_lines can re-render with styles.
                let rendered_len = md_to_lines(&entry.body, Color::White, body_width).len();
                (
                    entry.body.clone(),
                    // Placeholders — count matters, content does not.
                    vec![String::new(); rendered_len.max(1)],
                )
            } else if use_read_file {
                let lang = super::types::read_file_lang_for_body(&entry.body);
                let rendered_len = render_read_file_body_lines(&entry.body, lang, body_width).len();
                (entry.body.clone(), vec![String::new(); rendered_len.max(1)])
            } else if use_diff {
                // Count lines the same way the renderer will so scroll arithmetic is exact.
                let line_count: usize = entry
                    .body
                    .split('\n')
                    .map(|l| split_at_width(l, body_width).len())
                    .sum();
                (entry.body.clone(), vec![String::new(); line_count.max(1)])
            } else {
                let plain: Vec<String> = entry
                    .body
                    .split('\n')
                    .flat_map(|body_line| split_at_width(body_line, body_width))
                    .collect();
                (String::new(), plain)
            };

        let mut title = entry.title.clone();
        if entry.kind == EntryKind::ToolCall {
            if let Some(call_id) = entry.tool_call_id.as_ref() {
                tool_names_by_call_id.insert(call_id.clone(), title.clone());
            }
        } else if entry.kind == EntryKind::ToolResult {
            if let Some(call_id) = entry.tool_call_id.as_ref() {
                if let Some(tool_name) = tool_names_by_call_id.get(call_id) {
                    title = format!("{tool_name} result");
                }
            }
        }

        let _ = i; // index no longer used after removing the pre-pass

        if let Some(prev) = cached_entries.last() {
            let same_kind = prev.kind == entry.kind;
            let same_title = prev.title == title;
            let same_body = prev.raw_body == raw_body && prev.body_lines == body_lines;
            if same_kind && same_title && same_body {
                title = format!("{title}  ↺ same as above");
            }
        }

        // Working timer with no body collapses to header + trailing-blank only;
        // every other entry guarantees at least one body row.
        let body_line_count = if entry.collapsed && !entry.pending {
            // Collapsed entries hide their body entirely.
            0
        } else if is_working_timer && body_lines.is_empty() {
            0
        } else {
            body_lines.len().max(1)
        };
        let line_count = 1 + body_line_count + 1;
        total_lines += line_count;
        line_ends.push(total_lines);
        cached_entries.push(CachedTranscriptEntry {
            kind: entry.kind,
            title,
            timestamp: entry.timestamp,
            completed_at: entry.completed_at,
            pending: entry.pending,
            is_working_timer,
            raw_body,
            body_lines,
            line_count,
            collapsed: entry.collapsed,
        });
    }

    TranscriptRenderCache {
        width,
        dirty: false,
        entries: cached_entries,
        line_ends,
        total_lines,
    }
}

fn extract_user_message_history(items: &[ConversationItem]) -> Vec<String> {
    items
        .iter()
        .filter(|item| item.kind == ItemKind::UserMessage)
        .filter_map(|item| item.payload.get(KEY_TEXT).and_then(Value::as_str))
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .filter(|text| !is_synthetic_sub_agent_prompt(text))
        .map(ToOwned::to_owned)
        .collect()
}

fn is_synthetic_sub_agent_prompt(text: &str) -> bool {
    text.starts_with("You are running as a bounded sub-agent.")
}

#[allow(clippy::too_many_arguments)]
fn append_cached_transcript_entry_lines(
    out: &mut Vec<Line<'static>>,
    entry: &CachedTranscriptEntry,
    spinner_tick: u64,
    body_width: usize,
    start_line: usize,
    end_line: usize,
    entry_start: usize,
    _prev_timestamp: Option<i64>,
    _prev_user_timestamp: Option<i64>,
    now_ts: i64,
) {
    let (sigil, sigil_color, body_color) = entry_style(entry.kind);
    let mut current_line = entry_start;

    // ── Header ───────────────────────────────────────────────────────────────
    if current_line >= start_line && current_line < end_line {
        let mut header_spans = vec![
            Span::raw("  "),
            Span::styled(
                sigil.to_string(),
                Style::default()
                    .fg(sigil_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                entry.title.clone(),
                Style::default()
                    .fg(sigil_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ];
        if let Some(ts) = entry.timestamp {
            header_spans.push(Span::raw("  "));
            header_spans.push(Span::styled(
                format_timestamp(ts),
                Style::default().fg(COLOR_MUTED),
            ));
            // Show this entry's own elapsed time. Pending entries count up;
            // completed entries keep their frozen completion duration.
            if let Some(elapsed) =
                entry_elapsed_secs(entry.timestamp, entry.completed_at, entry.pending, now_ts)
            {
                header_spans.push(Span::styled(
                    format!(" · {}", super::types::format_duration(elapsed)),
                    Style::default().fg(super::types::COLOR_DIM),
                ));
            }
        }
        // The pending spinner is only useful on the Working timer entry — the
        // global "we're busy" indicator. Other pending entries (user prompt
        // placeholders, in-flight tool calls, streaming assistant messages)
        // don't need their own spinner now that the Working entry exists.
        if entry.pending && entry.is_working_timer {
            header_spans.push(Span::raw("  "));
            header_spans.push(Span::styled(
                super::types::spinner_frame(spinner_tick).to_string(),
                Style::default()
                    .fg(COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        // Show collapse/expand chevron on collapsible (auto-collapsed) entries.
        if entry.collapsed && !entry.pending {
            header_spans.push(Span::styled("  ▸", Style::default().fg(COLOR_MUTED)));
        }
        out.push(Line::from(header_spans));
    }
    current_line += 1;

    // ── Body ─────────────────────────────────────────────────────────────────
    // Collapsed entries skip the body entirely — only the header is shown.
    if entry.collapsed && !entry.pending {
        if current_line >= start_line && current_line < end_line {
            out.push(Line::from(""));
        }
        return;
    }

    let use_markdown = matches!(
        entry.kind,
        EntryKind::User | EntryKind::Assistant | EntryKind::Summary | EntryKind::Reasoning
    );
    let use_read_file = matches!(entry.kind, EntryKind::ToolCall | EntryKind::ToolResult)
        && is_read_file_body(&entry.raw_body);
    let use_diff = matches!(entry.kind, EntryKind::ToolCall | EntryKind::ToolResult)
        && is_diff_body(&entry.raw_body);

    if use_markdown && !entry.raw_body.is_empty() {
        // Re-render from the raw Markdown source so all spans carry proper styles.
        // Use body_width (same value used in build_transcript_render_cache) so the
        // rendered line count exactly matches the cached line_count used for scroll
        // arithmetic — a mismatch was causing the bottom of long responses to be clipped.
        let md_rendered = md_to_lines(&entry.raw_body, body_color, body_width);
        for md_line in md_rendered {
            if current_line >= start_line && current_line < end_line {
                let mut spans = vec![Span::styled("  │  ", Style::default().fg(COLOR_MUTED))];
                spans.extend(md_line.spans);
                out.push(Line::from(spans));
            }
            current_line += 1;
        }
    } else if use_read_file && !entry.raw_body.is_empty() {
        let lang = super::types::read_file_lang_for_body(&entry.raw_body);
        let rendered = render_read_file_body_lines(&entry.raw_body, lang, body_width);
        for rendered_line in rendered {
            if current_line >= start_line && current_line < end_line {
                let mut spans = vec![Span::styled("  │  ", Style::default().fg(COLOR_MUTED))];
                spans.extend(rendered_line);
                out.push(Line::from(spans));
            }
            current_line += 1;
        }
    } else if use_diff && !entry.raw_body.is_empty() {
        let lang = super::types::diff_lang_for_body(&entry.raw_body);
        for body_line in entry.raw_body.split('\n') {
            for chunk in split_at_width(body_line, body_width) {
                if current_line >= start_line && current_line < end_line {
                    let spans = render_diff_chunk(&chunk, lang);
                    let mut line_spans =
                        vec![Span::styled("  │  ", Style::default().fg(COLOR_MUTED))];
                    line_spans.extend(spans);
                    out.push(Line::from(line_spans));
                }
                current_line += 1;
            }
        }
    } else {
        for body_line in &entry.body_lines {
            if current_line >= start_line && current_line < end_line {
                // Detect working-indicator lines and style them prominently.
                let is_working_indicator = entry.pending
                    && body_line.contains("  ")
                    && (body_line.contains("thinking")
                        || body_line.contains("calling tool")
                        || body_line.contains("waiting")
                        || body_line.contains("reasoning")
                        || body_line.contains("working"));
                // Style the result separator line subtly.
                let is_separator = body_line.trim() == "─── result ───";
                let body_span = if is_working_indicator {
                    Span::styled(
                        body_line.clone(),
                        Style::default()
                            .fg(COLOR_ACCENT)
                            .add_modifier(Modifier::BOLD),
                    )
                } else if is_separator {
                    Span::styled(
                        body_line.clone(),
                        Style::default().fg(COLOR_MUTED).add_modifier(Modifier::DIM),
                    )
                } else {
                    Span::styled(body_line.clone(), Style::default().fg(body_color))
                };
                out.push(Line::from(vec![
                    Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
                    body_span,
                ]));
            }
            current_line += 1;
        }
    }

    if current_line >= start_line && current_line < end_line {
        out.push(Line::from(""));
    }
}

fn apply_char_selection(line: Line<'static>, col_from: usize, col_to: usize) -> Line<'static> {
    let mut result: Vec<Span<'static>> = Vec::new();
    let mut cursor = 0usize;
    let col_end = col_to.saturating_add(1);

    for span in line.spans {
        let content = span.content.as_ref();
        let char_count = content.chars().count();
        let span_end = cursor + char_count;

        if span_end <= col_from || cursor >= col_end {
            result.push(span);
        } else if cursor >= col_from && span_end <= col_end {
            result.push(Span::styled(
                span.content.clone(),
                span.style.add_modifier(Modifier::REVERSED),
            ));
        } else {
            let local_start = col_from.saturating_sub(cursor);
            let local_end = col_end.saturating_sub(cursor).min(char_count);

            if local_start > 0 {
                let pre: String = content.chars().take(local_start).collect();
                result.push(Span::styled(pre, span.style));
            }
            let sel: String = content
                .chars()
                .skip(local_start)
                .take(local_end - local_start)
                .collect();
            result.push(Span::styled(
                sel,
                span.style.add_modifier(Modifier::REVERSED),
            ));
            if local_end < char_count {
                let post: String = content.chars().skip(local_end).collect();
                result.push(Span::styled(post, span.style));
            }
        }
        cursor = span_end;
    }
    Line::from(result)
}

fn format_approval_preview(action: &Value) -> String {
    let Ok(action) = serde_json::from_value::<ActionDescriptor>(action.clone()) else {
        return serde_json::to_string_pretty(action).unwrap_or_else(|_| action.to_string());
    };

    let mut lines = Vec::new();

    if let Some(command) = action
        .command
        .as_ref()
        .filter(|command| !command.is_empty())
    {
        lines.push(format!("command   {}", shell_command_preview(command)));
    }

    if let Some(primary_path) = action.paths.first() {
        if action.action_type == "command"
            || action.action_type == "shell_exec"
            || action.action_type == "bash_exec"
        {
            lines.push(format!("cwd       {primary_path}"));
        } else if action.paths.len() == 1 {
            lines.push(format!("path      {primary_path}"));
        }
    }

    if action.paths.len() > 1 {
        let label = if action.action_type == "copy_path" {
            "paths"
        } else {
            "targets"
        };
        lines.push(format!("{label:<10}{}", action.paths.join(", ")));
    }

    if !action.domains.is_empty() {
        lines.push(format!("domains   {}", action.domains.join(", ")));
    }

    if lines.is_empty() {
        lines.push(format!("action    {}", action.action_type));
    } else if action.action_type != "command"
        && action.action_type != "shell_exec"
        && action.action_type != "bash_exec"
    {
        lines.insert(0, format!("action    {}", action.action_type));
    }

    lines.join("\n")
}

fn shell_command_preview(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| shell_escape(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_escape(arg: &str) -> String {
    if !arg.is_empty()
        && arg
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | ':' | '='))
    {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', "'\"'\"'"))
    }
}

fn tool_result_failed(result: &ToolResult) -> bool {
    if tool_result_not_spawned(result) {
        return false;
    }
    !result.ok
        || result
            .error_message
            .as_ref()
            .is_some_and(|message| !message.trim().is_empty())
        || result
            .output
            .get(KEY_ERROR)
            .and_then(Value::as_str)
            .is_some()
}

fn tool_result_not_spawned(result: &ToolResult) -> bool {
    result.output.get(KEY_STATUS).and_then(Value::as_str) == Some(STATUS_NOT_SPAWNED)
}

fn is_background_compaction_status_event(event: &RuntimeEvent, current_thread_id: &str) -> bool {
    event.kind == RuntimeEventKind::CompactionStatus
        && event.thread_id.as_deref() == Some(current_thread_id)
        // This event kind is also used for turn-local status such as codebase
        // indexing. Only turn-less events are actual background compaction
        && event.turn_id.is_none()
}

/// Scan user input text for image file paths and extract them as attachments.
///
/// Detects paths in two forms:
/// 1. A line that is entirely a file path (possibly with leading/trailing whitespace).
/// 2. A path embedded in text that starts with `/`, `~/`, `./`, or `C:\` and ends
///    with an image extension. This handles macOS screenshot paths with spaces.
///
/// Extracted paths are replaced with `[#image N]` placeholders in the text
/// so the LLM knows an image was attached. The remaining text becomes the text
/// portion of the message.
fn extract_image_paths(text: &str) -> (String, Vec<ComposerAttachment>) {
    let image_exts = [
        "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif", "ico",
    ];
    let mut text_lines = Vec::new();
    let mut attachments = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Case 1: the entire line is a path — replace with placeholder
        if is_bare_image_path(trimmed, &image_exts) {
            let path = expand_home(trimmed);
            let mime = mime_from_ext(&path);
            attachments.push(ComposerAttachment {
                path: path.clone(),
                mime_type: mime.to_string(),
            });
            text_lines.push(format!("[#image {}]", attachments.len()));
            continue;
        }

        // Case 2: extract embedded paths (e.g. "what's in this image? /Users/me/photo.png")
        let mut remaining = line.to_string();
        let mut found = true;
        while found {
            found = false;
            if let Some(start) = find_path_start(&remaining) {
                let candidate = &remaining[start..];
                if let Some(end) = find_image_path_end(candidate, &image_exts) {
                    let path_str = &candidate[..end];
                    let path = expand_home(path_str);
                    let mime = mime_from_ext(&path);
                    attachments.push(ComposerAttachment {
                        path: path.clone(),
                        mime_type: mime.to_string(),
                    });
                    // Replace the path with a placeholder so the LLM knows an image was attached
                    remaining = format!(
                        "{}[#image {}]{}",
                        &remaining[..start],
                        attachments.len(),
                        &remaining[start + end..]
                    );
                    found = true;
                }
            }
        }

        let trimmed_remaining = remaining.trim();
        if !trimmed_remaining.is_empty() {
            text_lines.push(trimmed_remaining.to_string());
        }
    }

    (text_lines.join("\n"), attachments)
}

/// Check if a string is entirely a bare image file path.
fn is_bare_image_path(s: &str, exts: &[&str]) -> bool {
    if s.is_empty() || s.contains('\n') {
        return false;
    }
    let lower = s.to_lowercase();
    exts.iter().any(|ext| lower.ends_with(&format!(".{ext}")))
        && (s.starts_with('/') || s.starts_with('~') || s.starts_with("./") || s.starts_with(".."))
}

/// Find the start index of a path-like substring (starts with /, ~/, ./, or C:\).
fn find_path_start(s: &str) -> Option<usize> {
    // Look for ~/ first (highest priority, explicit home)
    if let Some(i) = s.find("~/") {
        return Some(i);
    }
    // Look for absolute paths /
    for (i, c) in s.char_indices() {
        if c == '/' && (i == 0 || !s[..i].ends_with(|c: char| c.is_alphanumeric() || c == '_')) {
            return Some(i);
        }
    }
    // Look for ./
    if let Some(i) = s.find("./") {
        return Some(i);
    }
    None
}

/// Given a string starting at a path, find the end index (exclusive) where the
/// image extension ends. Returns None if no image extension is found.
fn find_image_path_end(s: &str, exts: &[&str]) -> Option<usize> {
    let lower = s.to_lowercase();
    for ext in exts {
        let suffix = format!(".{ext}");
        if let Some(pos) = lower.find(&suffix) {
            let end = pos + suffix.len();
            // Make sure the character after the extension isn't alphanumeric
            // (avoid matching "photo.pngx" as "photo.png")
            if end >= lower.len() || !lower.as_bytes()[end].is_ascii_alphanumeric() {
                return Some(end);
            }
        }
    }
    None
}

/// Expand `~` at the start of a path to the home directory.
fn expand_home(s: &str) -> String {
    if let Some(rest) = s.strip_prefix('~') {
        if rest.is_empty() || rest.starts_with('/') {
            if let Ok(home) = std::env::var("HOME") {
                return format!("{}{}", home, rest);
            }
        }
    }
    s.to_string()
}

/// Return a MIME type string based on the file extension.
fn mime_from_ext(path: &str) -> &'static str {
    let lower = path.to_lowercase();
    if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else if lower.ends_with(".bmp") {
        "image/bmp"
    } else if lower.ends_with(".tiff") || lower.ends_with(".tif") {
        "image/tiff"
    } else if lower.ends_with(".ico") {
        "image/x-icon"
    } else {
        "application/octet-stream"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::domain::now_millis;
    use serde_json::json;

    fn compaction_event(thread_id: &str, turn_id: Option<&str>) -> RuntimeEvent {
        RuntimeEvent {
            event_id: "evt_test".into(),
            kind: RuntimeEventKind::CompactionStatus,
            thread_id: Some(thread_id.into()),
            turn_id: turn_id.map(str::to_string),
            sequence: 1,
            payload: json!({ "status": "started", "message": "Auto-compacting context..." }),
            emitted_at: now_millis(),
        }
    }

    #[test]
    fn background_compaction_event_has_no_turn_id() {
        let event = compaction_event("thread_a", None);
        assert!(is_background_compaction_status_event(&event, "thread_a"));
    }

    #[test]
    fn turn_scoped_status_is_not_compaction() {
        let event = compaction_event("thread_a", Some("turn_a"));
        assert!(!is_background_compaction_status_event(&event, "thread_a"));
    }

    #[test]
    fn other_thread_status_is_not_current_compaction() {
        let event = compaction_event("thread_b", None);
        assert!(!is_background_compaction_status_event(&event, "thread_a"));
    }
}
