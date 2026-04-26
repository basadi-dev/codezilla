use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::oneshot;

use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::markdown::{md_to_lines, md_to_lines_with_source_map};

use super::super::domain::{
    ActionDescriptor, ApprovalDecision, ApprovalPolicy, ApprovalPolicyKind, ApprovalResolution,
    ConversationItem, ItemKind, PendingApproval, RuntimeEvent, RuntimeEventKind, ThreadMetadata,
    ToolResult, TurnStatus, UserInput,
};
use super::super::error as cod_error;
use super::super::runtime::{
    ConversationRuntime, ThreadCompactParams, ThreadCompactResult, ThreadForkParams,
    ThreadListParams, ThreadReadParams, ThreadResumeParams, ThreadStartParams, TurnInterruptParams,
    TurnStartParams, TurnSteerParams,
};
use super::types::{
    basename, entry_from_item, entry_style, format_timestamp, format_tool_result,
    is_read_file_body, relative_time_ago, render_read_file_body_lines, short_turn_id,
    split_at_width, thread_label, transcript_lines, AutocompleteItem, ComposerState, EntryKind,
    FocusPane, PendingApprovalView, SelectionPoint, SelectionRange, TranscriptEntry, COLOR_ACCENT,
    COLOR_MUTED, THREAD_LIMIT,
};

/// Sentinel item-id for the "thinking" placeholder injected immediately after
/// the user submits a message.  Removed when the first real agent content arrives.
const THINKING_PLACEHOLDER_ID: &str = "__codezilla_thinking__";

#[derive(Debug, Clone)]
struct CachedTranscriptEntry {
    kind: EntryKind,
    title: String,
    timestamp: Option<i64>,
    pending: bool,
    /// For markdown entries (Assistant/Summary/Reasoning), stores the original
    /// raw body text so the renderer can call md_to_lines each frame.
    /// For plain entries, this is empty.
    raw_body: String,
    /// Pre-chunked plain-text lines (used for non-markdown entries and for
    /// counting lines in the scroll arithmetic for markdown entries).
    body_lines: Vec<String>,
    line_count: usize,
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
    pub threads: Vec<ThreadMetadata>,
    pub selected_thread_id: Option<String>,
    pub transcript: Vec<TranscriptEntry>,
    pub transcript_index: HashMap<String, usize>,
    pub transcript_scroll: u16,
    pub auto_scroll: bool,
    pub focus: FocusPane,
    pub composer: ComposerState,
    pub pending_approval: Option<PendingApprovalView>,
    pub approval_policy_override: Option<ApprovalPolicy>,
    pub active_turn_id: Option<String>,
    pub status_message: String,
    pub error_message: Option<String>,
    pub should_quit: bool,
    /// Monotonically increasing frame counter — drives the spinner animation.
    pub spinner_tick: u64,
    /// Whether mouse capture is active (wheel scroll on, native select off).
    pub mouse_capture_enabled: bool,
    /// Whether the user has pressed Ctrl+Q and is awaiting confirmation.
    pub quit_requested: bool,
    // ── drag-to-select ────────────────────────────────────────────────────────
    /// Whether the user has pressed Ctrl+C once while composer is focused and
    /// is awaiting a second Ctrl+C to clear the composer prompt.
    pub composer_clear_requested: bool,
    /// Fixed transcript positions where the left button started and ended.
    pub drag_start: Option<SelectionPoint>,
    /// Fixed transcript position of the current/final drag point.
    pub drag_end: Option<SelectionPoint>,
    /// Transcript area rect — written by the renderer each frame.
    pub transcript_area: Rect,
    /// Composer input area rect — written by the renderer each frame.
    pub composer_area: Rect,
    transcript_render_cache: TranscriptRenderCache,
    /// Slash-command autocomplete suggestions for the current composer text.
    pub autocomplete_suggestions: Vec<AutocompleteItem>,
    /// Index into `autocomplete_suggestions` of the highlighted entry.
    pub autocomplete_selected: usize,
    /// First visible row in the autocomplete list (scroll offset).
    pub autocomplete_scroll: usize,
    /// Per-session model/reasoning override (None = use thread metadata + config defaults).
    pub model_settings_override: Option<super::super::domain::ModelSettings>,
    composer_history: Vec<String>,
    composer_history_index: Option<usize>,
    composer_history_saved_input: Option<String>,
    /// Background compaction task result receiver. Set when /compact is running.
    pub(super) pending_compact: Option<oneshot::Receiver<anyhow::Result<ThreadCompactResult>>>,
    /// Live activity description for the header — e.g. "⚙ read_file src/main.rs".
    /// Set when a tool-call/command/file-change item starts; cleared on turn end.
    pub live_activity: Option<String>,
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
            threads: Vec::new(),
            selected_thread_id: Some(initial_thread_id),
            transcript: Vec::new(),
            transcript_index: HashMap::new(),
            transcript_scroll: 0,
            auto_scroll: true,
            focus: FocusPane::Composer,
            composer: ComposerState::default(),
            pending_approval: None,
            approval_policy_override: None,
            active_turn_id: None,
            status_message: "Ready".into(),
            error_message: None,
            should_quit: false,
            spinner_tick: 0,
            quit_requested: false,
            composer_clear_requested: false,
            mouse_capture_enabled: true,
            drag_start: None,
            drag_end: None,
            transcript_area: Rect::default(),
            composer_area: Rect::default(),
            transcript_render_cache: TranscriptRenderCache {
                dirty: true,
                ..TranscriptRenderCache::default()
            },
            autocomplete_suggestions: Vec::new(),
            autocomplete_selected: 0,
            autocomplete_scroll: 0,
            model_settings_override: None,
            composer_history: Vec::new(),
            composer_history_index: None,
            composer_history_saved_input: None,
            pending_compact: None,
            live_activity: None,
        };
        app.refresh_threads().await?;
        let current = app.current_thread_id.clone();
        app.load_thread(&current).await?;
        Ok(app)
    }

    pub async fn refresh_threads(&mut self) -> Result<()> {
        self.threads = self
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

        if self.threads.is_empty() {
            self.selected_thread_id = None;
            return Ok(());
        }

        let fallback = self.current_thread_id.clone();
        let desired = self
            .selected_thread_id
            .clone()
            .unwrap_or_else(|| fallback.clone());
        if self.threads.iter().any(|t| t.thread_id == desired) {
            self.selected_thread_id = Some(desired);
        } else {
            self.selected_thread_id = Some(fallback);
        }
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
        self.selected_thread_id = Some(thread_id.to_string());
        self.transcript.clear();
        self.transcript_index.clear();
        for item in &persisted.items {
            self.upsert_transcript_entry(entry_from_item(item));
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
        self.pending_approval = None;
        self.drag_start = None;
        self.drag_end = None;
        self.auto_scroll = true;
        self.error_message = None;
        self.reset_composer_history_navigation();
        self.composer_history = self.build_composer_history(&persisted).await?;
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
                        self.status_message = format!(
                            "✓ Compacted — {} item(s) replaced with summary",
                            r.items_removed
                        );
                        self.error_message = None;
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Compact failed: {e}"));
                    }
                }
                Ok(true)
            }
            Err(oneshot::error::TryRecvError::Empty) => Ok(false),
            Err(oneshot::error::TryRecvError::Closed) => {
                self.pending_compact = None;
                self.error_message = Some("Compaction task dropped unexpectedly".into());
                Ok(true)
            }
        }
    }

    // ── Drag-to-select helpers ────────────────────────────────────────────────

    pub fn begin_transcript_drag(&mut self, col: u16, row: u16) {
        let Some(point) = self.mouse_to_selection_point(col, row, false) else {
            self.clear_selection();
            return;
        };
        self.drag_start = Some(point);
        self.drag_end = Some(point);
        self.auto_scroll = false;
    }

    pub fn update_transcript_drag(&mut self, col: u16, row: u16) {
        if self.drag_start.is_none() {
            return;
        }
        if let Some(point) = self.mouse_to_selection_point(col, row, true) {
            self.drag_end = Some(point);
        }
    }

    pub fn finish_transcript_drag(&mut self, col: u16, row: u16) {
        self.update_transcript_drag(col, row);
        let moved = self.drag_start != self.drag_end;
        if moved {
            self.copy_selection_to_clipboard();
        } else {
            self.clear_selection();
        }
    }

    fn mouse_to_selection_point(
        &self,
        col: u16,
        row: u16,
        clamp_to_viewport: bool,
    ) -> Option<SelectionPoint> {
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
            line: self.transcript_scroll as usize + (row as usize - area.y as usize),
            col: col.saturating_sub(area.x) as usize,
        })
    }

    fn current_selection_range(&self) -> Option<super::types::SelectionRange> {
        use super::types::SelectionRange;

        let start = self.drag_start?;
        let end = self.drag_end?;
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

    pub fn clear_selection(&mut self) {
        self.drag_start = None;
        self.drag_end = None;
    }

    pub fn scroll_transcript(&mut self, delta: i32) {
        if delta < 0 {
            // Scrolling up — detach from the bottom.
            self.auto_scroll = false;
            self.transcript_scroll = self
                .transcript_scroll
                .saturating_sub(delta.unsigned_abs() as u16);
        } else {
            // Scrolling down — just advance; render.rs will detect if we've
            // reached max_scroll and re-engage auto-scroll there.
            self.transcript_scroll = self.transcript_scroll.saturating_add(delta as u16);
        }
    }

    pub fn jump_transcript_to_bottom(&mut self) {
        self.auto_scroll = true;
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

        // ── Try partial raw-markdown copy for markdown entries ────────────────
        if let Some(raw_text) = self.try_copy_partial_markdown(&sel, width) {
            let char_count = raw_text.chars().count();
            match arboard::Clipboard::new() {
                Ok(mut cb) => match cb.set_text(raw_text) {
                    Ok(_) => {
                        self.status_message = format!("✓ Copied {char_count} chars (markdown)");
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
            return;
        }

        // ── Fallback: rendered visual text (non-markdown entries) ─────────────
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

    /// For a selection that falls within a single markdown entry, extract the raw
    /// markdown source lines that correspond to the selected visual lines.
    ///
    /// Uses a source-line map built from the rendered output to find which raw
    /// markdown lines generated the selected visual lines, then expands to the
    /// nearest block boundaries (empty lines) so complete paragraphs/tables/code
    /// blocks are always returned rather than mid-block fragments.
    fn try_copy_partial_markdown(&self, sel: &SelectionRange, width: u16) -> Option<String> {
        let cache = &self.transcript_render_cache;
        if cache.entries.is_empty() {
            return None;
        }

        let start_idx = cache
            .line_ends
            .partition_point(|&end| end <= sel.start_line);
        let end_idx = cache.line_ends.partition_point(|&end| end <= sel.end_line);
        if start_idx >= cache.entries.len() {
            return None;
        }

        let entry = &cache.entries[start_idx];
        if !matches!(
            entry.kind,
            EntryKind::Assistant | EntryKind::Summary | EntryKind::Reasoning
        ) {
            return None;
        }
        if entry.raw_body.is_empty() {
            return None;
        }

        // The entry layout is: 1 header line + N body lines + 1 trailing blank.
        // body_start is the first visual line of the markdown body within the entry.
        let entry_vis_start = if start_idx == 0 {
            0
        } else {
            cache.line_ends[start_idx - 1]
        };
        let body_vis_start = entry_vis_start + 1;

        let bw = (width as usize).saturating_sub(5).max(10);
        let (_, source_map) =
            md_to_lines_with_source_map(&entry.raw_body, ratatui::style::Color::White, bw);
        if source_map.is_empty() {
            return None;
        }

        // Convert absolute visual lines → relative to body start.
        let rel_start = sel.start_line.saturating_sub(body_vis_start);
        let rel_end = if end_idx == start_idx {
            sel.end_line.saturating_sub(body_vis_start)
        } else {
            source_map.len().saturating_sub(1)
        }
        .min(source_map.len().saturating_sub(1));

        if rel_start >= source_map.len() {
            return None;
        }

        let src_a = *source_map.get(rel_start).unwrap_or(&0);
        let src_b = *source_map.get(rel_end).unwrap_or(&src_a);
        let (src_start, src_end) = (src_a.min(src_b), src_a.max(src_b));

        // Expand to nearest block boundaries (empty lines) so we always return
        // complete blocks: whole paragraphs, full tables, fenced code blocks, etc.
        let raw_lines: Vec<&str> = entry.raw_body.lines().collect();
        let n = raw_lines.len();
        if n == 0 {
            return None;
        }

        let is_empty = |i: usize| {
            raw_lines
                .get(i)
                .map(|l| l.trim().is_empty())
                .unwrap_or(true)
        };

        let block_start = (0..=src_start.min(n - 1))
            .rev()
            .find(|&i| i == 0 || is_empty(i.saturating_sub(1)))
            .unwrap_or(0);

        let block_end = (src_end.min(n - 1)..n)
            .find(|&i| is_empty(i + 1) || i + 1 >= n)
            .unwrap_or(n - 1);

        let excerpt = raw_lines[block_start..=block_end].join("\n");
        if excerpt.trim().is_empty() {
            None
        } else {
            Some(excerpt)
        }
    }

    pub async fn submit_composer(&mut self) -> Result<()> {
        let trimmed = self.composer.trimmed_text();
        if trimmed.is_empty() {
            return Ok(());
        }

        let raw = self.composer.take_text();
        self.reset_composer_history_navigation();
        if self.try_handle_slash_command(raw.trim()).await? {
            return Ok(());
        }
        if let Some(cmd) = raw.trim().strip_prefix('!') {
            self.run_shell_command(cmd.trim()).await?;
            return Ok(());
        }

        if self.pending_approval.is_some() {
            self.error_message =
                Some("Resolve the approval request before sending more input".into());
            return Ok(());
        }

        self.error_message = None;

        // ── Immediate visual feedback ─────────────────────────────────────────
        // Push a pending "thinking" placeholder into the transcript right now so
        // the user sees something happening in the chat area before the runtime
        // fires TurnStarted / ItemStarted.
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            self.upsert_transcript_entry(TranscriptEntry {
                item_id: THINKING_PLACEHOLDER_ID.to_string(),
                tool_call_id: None,
                kind: EntryKind::Assistant,
                title: "Thinking…".into(),
                body: String::new(),
                timestamp: Some(ts),
                pending: true,
            });
        }
        self.auto_scroll = true;

        if let Some(turn_id) = self.active_turn_id.clone() {
            self.runtime
                .steer_turn(TurnSteerParams {
                    thread_id: self.current_thread_id.clone(),
                    expected_turn_id: turn_id.clone(),
                    input: vec![UserInput::from_text(&raw)],
                })
                .await?;
            self.status_message = format!("Queued input for {}", short_turn_id(&turn_id));
        } else {
            let turn = self
                .runtime
                .start_turn(
                    TurnStartParams {
                        thread_id: self.current_thread_id.clone(),
                        input: vec![UserInput::from_text(&raw)],
                        cwd: None,
                        model_settings: self.model_settings_override.clone(),
                        approval_policy: self.current_approval_policy_override(),
                        permission_profile: None,
                        output_schema: None,
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
        if self.composer_history.is_empty() {
            return false;
        }

        let next_index = match self.composer_history_index {
            Some(index) => index.saturating_sub(1),
            None => self.composer_history.len() - 1,
        };

        if self.composer_history_index.is_none() {
            self.composer_history_saved_input = Some(self.composer.text().to_string());
        }

        self.composer_history_index = Some(next_index);
        self.composer
            .set_text(self.composer_history[next_index].clone());
        true
    }

    pub fn composer_history_next(&mut self) -> bool {
        if self.composer_history.is_empty() {
            return false;
        }

        let Some(index) = self.composer_history_index else {
            self.composer_history_saved_input = Some(self.composer.text().to_string());
            self.composer_history_index = Some(0);
            self.composer.set_text(self.composer_history[0].clone());
            return true;
        };

        if index + 1 < self.composer_history.len() {
            let next_index = index + 1;
            self.composer_history_index = Some(next_index);
            self.composer
                .set_text(self.composer_history[next_index].clone());
        } else {
            let restored = self.composer_history_saved_input.take().unwrap_or_default();
            self.composer_history_index = None;
            self.composer.set_text(restored);
        }
        true
    }

    pub fn composer_history_active(&self) -> bool {
        self.composer_history_index.is_some()
    }

    pub fn reset_composer_history_navigation(&mut self) {
        self.composer_history_index = None;
        self.composer_history_saved_input = None;
    }

    async fn build_composer_history(
        &self,
        persisted: &super::super::domain::PersistedThread,
    ) -> Result<Vec<String>> {
        let current = extract_user_message_history(&persisted.items);
        if !current.is_empty() {
            return Ok(current);
        }

        if let Some(previous_thread) = self
            .threads
            .iter()
            .find(|thread| thread.thread_id != persisted.metadata.thread_id)
        {
            let fallback = self
                .runtime
                .read_thread(ThreadReadParams {
                    thread_id: previous_thread.thread_id.clone(),
                })
                .await?
                .thread;
            let history = extract_user_message_history(&fallback.items);
            if !history.is_empty() {
                return Ok(history);
            }
        }

        Ok(Vec::new())
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
            let reasoning = ms.reasoning_effort.as_deref().unwrap_or("off");
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
            self.status_message = format!("Model set to {}/{}", ms.provider_id, ms.model_id);
            self.error_message = None;
            self.model_settings_override = Some(ms);
            true
        } else if matches!(command, "/reasoning") {
            let effort = self
                .effective_model_settings()
                .reasoning_effort
                .as_deref()
                .unwrap_or("off")
                .to_string();
            self.status_message = format!("Reasoning: {effort}");
            self.error_message = None;
            true
        } else if let Some(rest) = command.strip_prefix("/reasoning ") {
            let rest = rest.trim();
            let mut ms = self.effective_model_settings();
            ms.reasoning_effort = if rest == "off" {
                None
            } else {
                Some(rest.to_string())
            };
            let label = ms.reasoning_effort.as_deref().unwrap_or("off");
            self.status_message = format!("Reasoning set to {label}");
            self.error_message = None;
            self.model_settings_override = Some(ms);
            true
        } else if matches!(command, "/help") {
            self.status_message =
                "Keys: Tab/↑↓ autocomplete, Ctrl+A/E start/end-of-line, Ctrl+N new, Ctrl+F fork, \
                 Ctrl+C interrupt (double-tap clears composer), Ctrl+Q quit  ·  \
                 Commands: /model [provider/model]  /reasoning [low|medium|high|off]  \
                 /approve auto|ask|toggle  /compact  /new  /fork  /open <id>  /threads (autocomplete)  ·  \
                 CLI: codezilla -r (resume last thread)".into();
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
            reasoning_effort: cfg.model_settings.reasoning_effort.clone(),
            summary_mode: None,
            service_tier: None,
            web_search_enabled: false,
            context_window: cfg.model_settings.context_window,
        }
    }

    pub fn update_autocomplete(&mut self) {
        let text = self.composer.trimmed_text();
        if !text.starts_with('/') {
            self.autocomplete_suggestions.clear();
            self.autocomplete_selected = 0;
            self.autocomplete_scroll = 0;
            return;
        }

        let ms = self.effective_model_settings();
        let cur_reasoning = ms.reasoning_effort.as_deref().unwrap_or("off");
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
            "/quit",
            "/reload",
            "/resume ",
            "/threads",
        ] {
            all.push(AutocompleteItem::simple(*cmd));
        }

        // ── /model: current entry + presets from config ───────────────────────
        all.push(AutocompleteItem::simple("/model"));
        for preset in &cfg_models {
            let key = format!("{}/{}", preset.provider_id, preset.model_id);
            let value = format!("/model {key}");
            let marker = if key == cur_model_key { "  ←" } else { "" };
            let label = format!("{value}{marker}");
            all.push(AutocompleteItem::labeled(value, label));
        }

        // ── /reasoning: sorted off → low → medium → high with current marker ─
        all.push(AutocompleteItem::simple("/reasoning"));
        for level in &["off", "low", "medium", "high"] {
            let value = format!("/reasoning {level}");
            let marker = if *level == cur_reasoning { "  ←" } else { "" };
            let label = format!("{value}{marker}");
            all.push(AutocompleteItem::labeled(value, label));
        }

        // ── /threads: list known threads as /resume <id> entries ─────────────
        for thread in &self.threads {
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

        self.autocomplete_suggestions = all
            .into_iter()
            .filter(|item| {
                item.label.starts_with(text.as_str()) || item.value.starts_with(text.as_str())
            })
            .collect();
        self.autocomplete_selected = 0;
        self.autocomplete_scroll = 0;
    }

    /// Move selection down (or wrap), fill the composer, scroll to keep selected visible.
    pub fn autocomplete_select_next(&mut self) {
        if self.autocomplete_suggestions.is_empty() {
            return;
        }
        self.autocomplete_selected =
            (self.autocomplete_selected + 1) % self.autocomplete_suggestions.len();
        let v = self.autocomplete_suggestions[self.autocomplete_selected]
            .value
            .clone();
        self.composer.set_text(v);
        self.autocomplete_clamp_scroll();
    }

    /// Move selection up (or wrap), fill the composer, scroll to keep selected visible.
    pub fn autocomplete_select_prev(&mut self) {
        if self.autocomplete_suggestions.is_empty() {
            return;
        }
        let len = self.autocomplete_suggestions.len();
        self.autocomplete_selected = (self.autocomplete_selected + len - 1) % len;
        let v = self.autocomplete_suggestions[self.autocomplete_selected]
            .value
            .clone();
        self.composer.set_text(v);
        self.autocomplete_clamp_scroll();
    }

    fn autocomplete_clamp_scroll(&mut self) {
        const MAX_VISIBLE: usize = 8;
        let viewport = MAX_VISIBLE.min(self.autocomplete_suggestions.len());
        let sel = self.autocomplete_selected;
        if sel < self.autocomplete_scroll {
            self.autocomplete_scroll = sel;
        } else if sel >= self.autocomplete_scroll + viewport {
            self.autocomplete_scroll = sel + 1 - viewport;
        }
    }

    pub async fn toggle_auto_approve_tools(&mut self) {
        let enable = !self.auto_approve_tools_enabled();
        self.set_auto_approve_tools(enable).await;
    }

    pub async fn set_auto_approve_tools(&mut self, enabled: bool) {
        self.approval_policy_override = if enabled {
            Some(ApprovalPolicy {
                kind: ApprovalPolicyKind::Never,
                granular: None,
            })
        } else {
            None
        };
        // Propagate immediately to any running turn so it takes effect on the
        // next tool-call evaluation without needing to restart.
        let _ = self
            .runtime
            .set_thread_approval_policy(
                &self.current_thread_id,
                self.approval_policy_override.clone(),
            )
            .await;
        self.status_message = format!("Approvals: {}", self.approval_mode_label());
        self.error_message = None;
    }

    pub fn auto_approve_tools_enabled(&self) -> bool {
        matches!(
            self.effective_approval_policy().kind,
            ApprovalPolicyKind::Never
        )
    }

    pub fn approval_mode_label(&self) -> &'static str {
        if self.auto_approve_tools_enabled() {
            "auto"
        } else {
            "ask"
        }
    }

    fn current_approval_policy_override(&self) -> Option<ApprovalPolicy> {
        self.approval_policy_override.clone()
    }

    fn effective_approval_policy(&self) -> ApprovalPolicy {
        self.approval_policy_override
            .clone()
            .unwrap_or_else(|| self.runtime.effective_config().approval_policy.clone())
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
        let Some(approval) = self.pending_approval.clone() else {
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
        self.pending_approval = None;
        Ok(())
    }

    pub async fn handle_runtime_event(&mut self, event: RuntimeEvent) -> Result<()> {
        match event.kind {
            RuntimeEventKind::ThreadStarted
            | RuntimeEventKind::TurnStarted
            | RuntimeEventKind::TurnCompleted
            | RuntimeEventKind::TurnFailed => {
                self.refresh_threads().await?;
            }
            _ => {}
        }

        if event.thread_id.as_deref() != Some(self.current_thread_id.as_str()) {
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
                self.status_message = "Thinking…".into();
                self.error_message = None;
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Status,
                    "Turn",
                    "Assistant is preparing a response",
                    Some(event.emitted_at / 1000),
                );
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
                self.pending_approval = Some(PendingApprovalView {
                    approval: pending.clone(),
                    action_preview: preview,
                });
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
                    .pending_approval
                    .as_ref()
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
                self.pending_approval = None;
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
                self.active_turn_id = None;
                self.pending_approval = None;
                self.live_activity = None;
                self.remove_thinking_placeholder();
                if let Some(thread) = self.current_thread_meta.as_mut() {
                    thread.status = super::super::domain::ThreadStatus::Idle;
                }
                self.status_message = "Ready".into();
                self.error_message = None;
            }
            RuntimeEventKind::TurnFailed => {
                self.active_turn_id = None;
                self.pending_approval = None;
                self.live_activity = None;
                self.remove_thinking_placeholder();
                // Prefer "kind" (the structured label) as the title; fall back to "reason".
                let kind_label = event
                    .payload
                    .get("kind")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Error");
                let reason = event
                    .payload
                    .get("reason")
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
                let msg = event
                    .payload
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Compacting…");
                self.status_message = msg.to_string();
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
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("AGENT_MESSAGE");

        let entry_kind = match kind {
            "AGENT_MESSAGE" => EntryKind::Assistant,
            "REASONING_TEXT" | "REASONING_SUMMARY" => EntryKind::Reasoning,
            _ => EntryKind::Status,
        };
        let title = match entry_kind {
            EntryKind::Assistant => "Codezilla",
            EntryKind::Reasoning => "Reasoning",
            _ => "Runtime",
        };
        // Once real content starts, the thinking placeholder is no longer needed.
        if entry_kind == EntryKind::Assistant {
            self.remove_thinking_placeholder();
        }
        self.upsert_transcript_entry(TranscriptEntry {
            item_id: item_id.to_string(),
            tool_call_id: None,
            kind: entry_kind,
            title: title.into(),
            body: String::new(),
            timestamp: Some(event.emitted_at / 1000),
            pending: true,
        });

        // ── Contextual status + live activity messages ─────────────────────
        // `live_activity` drives the header; `status_message` drives the status bar.
        let (activity, status) = match kind {
            "TOOL_CALL" => {
                let tool_name = event
                    .payload
                    .get("toolName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tool");
                // Best-effort short label: first arg (path/command) if present.
                let arg_hint = event
                    .payload
                    .get("arguments")
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
                let activity_str = match arg_hint {
                    Some(hint) => format!("⚙ {} {}", tool_name, hint),
                    None => format!("⚙ {}", tool_name),
                };
                (activity_str.clone(), format!("{}…", activity_str))
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
                let activity_str = format!("$ {}", short_cmd);
                (activity_str.clone(), activity_str)
            }
            "FILE_CHANGE" => {
                let path = event
                    .payload
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("file");
                let short_path = path.rsplit('/').next().unwrap_or(path);
                let activity_str = format!("✏ {}", short_path);
                (activity_str.clone(), format!("✏  {path}"))
            }
            _ => (String::new(), "Streaming response…".into()),
        };
        self.live_activity = if activity.is_empty() {
            None
        } else {
            Some(activity)
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
            self.upsert_transcript_entry(TranscriptEntry {
                item_id: item_id.to_string(),
                tool_call_id: None,
                kind: EntryKind::Assistant,
                title: "Codezilla".into(),
                body: delta.to_string(),
                timestamp: Some(event.emitted_at / 1000),
                pending: true,
            });
        }
        self.status_message = "Streaming response…".into();
        // While streaming text, keep live_activity as-is (shows last tool) or set generic.
        if self.live_activity.is_none() {
            self.live_activity = Some("◆ generating…".to_string());
        }
        // Do NOT force auto_scroll here — if the user scrolled up, respect that.
    }

    fn handle_item_completed(&mut self, event: &RuntimeEvent) -> Result<()> {
        let item = serde_json::from_value::<ConversationItem>(event.payload.clone())?;
        if item.kind == ItemKind::ToolResult {
            let result = serde_json::from_value::<ToolResult>(item.payload.clone())?;
            if tool_result_failed(&result) {
                self.upsert_transcript_entry(self.failed_tool_result_entry(&item, &result));
            } else {
                self.remove_latest_tool_context_entries();
                self.upsert_transcript_entry(entry_from_item(&item));
            }
            // Clear tool activity once the result is in.
            self.live_activity = None;
        } else if matches!(item.kind, ItemKind::AgentMessage) {
            self.upsert_transcript_entry(entry_from_item(&item));
            self.live_activity = None;
        } else {
            self.upsert_transcript_entry(entry_from_item(&item));
        }
        // Do NOT force auto_scroll — respect user scroll position.
        Ok(())
    }

    pub fn upsert_transcript_entry(&mut self, entry: TranscriptEntry) {
        if let Some(index) = self.transcript_index.get(&entry.item_id).copied() {
            self.transcript[index] = entry;
        } else {
            let index = self.transcript.len();
            self.transcript_index.insert(entry.item_id.clone(), index);
            self.transcript.push(entry);
        }
        self.invalidate_transcript_cache();
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
            tool_call_id: None,
            kind,
            title: title.into(),
            body: body.into(),
            timestamp,
            pending: false,
        });
        // Do NOT force auto_scroll — respect user scroll position.
    }

    /// Remove the "thinking" placeholder from the transcript if it is still present.
    fn remove_thinking_placeholder(&mut self) {
        if let Some(index) = self.transcript_index.get(THINKING_PLACEHOLDER_ID).copied() {
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
            return transcript_lines(&self.transcript, self.spinner_tick, width, selection);
        }

        let end_line = start_line.saturating_add(max_lines);
        let body_width = (width as usize).saturating_sub(5).max(1);
        let first_entry = self
            .transcript_render_cache
            .line_ends
            .partition_point(|&line_end| line_end <= start_line);

        let mut lines = Vec::new();
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
                self.spinner_tick,
                body_width,
                start_line,
                end_line,
                entry_start,
            );
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

    fn invalidate_transcript_cache(&mut self) {
        self.transcript_render_cache.dirty = true;
    }

    fn failed_tool_result_entry(
        &self,
        item: &ConversationItem,
        result: &ToolResult,
    ) -> TranscriptEntry {
        let (tool_name, tool_input) = self
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
            .map(|entry| (entry.title.clone(), entry.body.clone()))
            .unwrap_or_else(|| ("tool".to_string(), String::new()));

        let mut body = Vec::new();
        if !tool_input.trim().is_empty() {
            body.push(format!(
                "input: {}",
                tool_input.lines().next().unwrap_or_default()
            ));
        }
        let error_value = result.error_message.as_deref().map(Value::from);
        body.push(format_tool_result(
            Some(&result.output),
            error_value.as_ref(),
        ));

        TranscriptEntry {
            item_id: item.item_id.clone(),
            tool_call_id: Some(result.tool_call_id.clone()),
            kind: EntryKind::Error,
            title: format!("{tool_name} failed"),
            body: body.join("\n"),
            timestamp: Some(item.created_at),
            pending: false,
        }
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

    for entry in entries {
        let use_markdown = matches!(
            entry.kind,
            EntryKind::Assistant | EntryKind::Summary | EntryKind::Reasoning
        );
        let use_read_file = entry.kind == EntryKind::ToolResult && is_read_file_body(&entry.body);

        let (raw_body, body_lines): (String, Vec<String>) =
            if entry.body.is_empty() && entry.pending {
                // Build an animated working indicator that matches the visual render
                let spinner = super::types::spinner_frame(0);
                let working_text = match entry.kind {
                    EntryKind::Assistant => "thinking",
                    EntryKind::ToolCall => "calling tool",
                    EntryKind::ToolResult => "waiting",
                    EntryKind::Reasoning => "reasoning",
                    _ => "working",
                };
                (String::new(), vec![format!("{spinner}  {working_text}…")])
            } else if use_markdown || use_read_file {
                // Render now to get the correct line count for scroll arithmetic.
                // The rendered Lines themselves are discarded; raw_body is kept so
                // append_cached_transcript_entry_lines can re-render with styles.
                let rendered_len = if use_markdown {
                    md_to_lines(&entry.body, Color::White, body_width).len()
                } else {
                    let lang = super::types::read_file_lang_for_body(&entry.body);
                    render_read_file_body_lines(&entry.body, lang, body_width).len()
                };
                (
                    entry.body.clone(),
                    // Placeholders — count matters, content does not.
                    vec![String::new(); rendered_len.max(1)],
                )
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

        if let Some(prev) = cached_entries.last() {
            let same_kind = prev.kind == entry.kind;
            let same_title = prev.title == title;
            let same_body = prev.raw_body == raw_body && prev.body_lines == body_lines;
            if same_kind && same_title && same_body {
                title = format!("{title}  ↺ same as above");
            }
        }

        let line_count = 1 + body_lines.len().max(1) + 1;
        total_lines += line_count;
        line_ends.push(total_lines);
        cached_entries.push(CachedTranscriptEntry {
            kind: entry.kind,
            title,
            timestamp: entry.timestamp,
            pending: entry.pending,
            raw_body,
            body_lines,
            line_count,
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
        .filter_map(|item| item.payload.get("text").and_then(Value::as_str))
        .filter(|text| !text.trim().is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn append_cached_transcript_entry_lines(
    out: &mut Vec<Line<'static>>,
    entry: &CachedTranscriptEntry,
    spinner_tick: u64,
    body_width: usize,
    start_line: usize,
    end_line: usize,
    entry_start: usize,
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
        }
        if entry.pending {
            header_spans.push(Span::raw("  "));
            header_spans.push(Span::styled(
                super::types::spinner_frame(spinner_tick).to_string(),
                Style::default()
                    .fg(COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        out.push(Line::from(header_spans));
    }
    current_line += 1;

    // ── Body ─────────────────────────────────────────────────────────────────
    let use_markdown = matches!(
        entry.kind,
        EntryKind::Assistant | EntryKind::Summary | EntryKind::Reasoning
    );
    let use_read_file = entry.kind == EntryKind::ToolResult && is_read_file_body(&entry.raw_body);

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
    } else {
        for body_line in &entry.body_lines {
            if current_line >= start_line && current_line < end_line {
                // Detect working-indicator lines and style them prominently
                let is_working_indicator = entry.pending
                    && body_line.contains("  ")
                    && (body_line.contains("thinking")
                        || body_line.contains("calling tool")
                        || body_line.contains("waiting")
                        || body_line.contains("reasoning")
                        || body_line.contains("working"));
                let body_span = if is_working_indicator {
                    Span::styled(
                        body_line.clone(),
                        Style::default()
                            .fg(COLOR_ACCENT)
                            .add_modifier(Modifier::BOLD),
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
    !result.ok
        || result
            .error_message
            .as_ref()
            .is_some_and(|message| !message.trim().is_empty())
        || result.output.get("error").and_then(Value::as_str).is_some()
}
