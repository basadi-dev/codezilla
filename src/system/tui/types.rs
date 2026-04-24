use serde_json::Value;
use std::path::Path;

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use crate::system::domain::PendingApproval;
use crate::system::domain::{ConversationItem, ItemKind, ThreadMetadata};

// ─── Colour palette ── (Claude Code / Codex CLI inspired) ────────────────────

/// Background of focused input border / streaming accent
pub const COLOR_ACCENT: Color = Color::Rgb(100, 200, 163); // soft mint-green
/// Dimmed muted text, timestamps, hints
pub const COLOR_MUTED: Color = Color::Rgb(110, 118, 135);
/// User message sigil + text
pub const COLOR_USER: Color = Color::Rgb(255, 195, 100); // warm amber
/// Assistant message sigil + text
pub const COLOR_ASSISTANT: Color = Color::Rgb(140, 200, 255); // sky blue
/// Tool call sigil
pub const COLOR_TOOL: Color = Color::Rgb(220, 140, 255); // soft violet
/// Tool result sigil
pub const COLOR_TOOL_RESULT: Color = Color::Rgb(120, 220, 160); // soft green
/// Error sigil + text
pub const COLOR_ERROR: Color = Color::Rgb(255, 100, 100); // soft red
/// Reasoning sigil
pub const COLOR_REASONING: Color = Color::Rgb(180, 160, 255); // lavender
/// Approval modal border
pub const COLOR_APPROVAL: Color = Color::Rgb(255, 210, 80);
/// Summary sigil
pub const COLOR_SUMMARY: Color = Color::Rgb(200, 180, 255);
/// Status / runtime events
pub const COLOR_STATUS: Color = Color::Rgb(140, 200, 140);
/// Subtle border for approval modal
pub const COLOR_BORDER: Color = Color::Rgb(60, 68, 84);
/// Input prompt glyph color
pub const COLOR_PROMPT: Color = Color::Rgb(100, 200, 163);

// ─── Spinner frames ───────────────────────────────────────────────────────────

pub const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn spinner_frame(tick: u64) -> &'static str {
    SPINNER_FRAMES[(tick as usize) % SPINNER_FRAMES.len()]
}

// ─── Layout constants ─────────────────────────────────────────────────────────

pub const THREAD_LIMIT: i32 = 40;
pub const COMPOSER_MIN_HEIGHT: u16 = 4;
pub const COMPOSER_MAX_HEIGHT: u16 = 7; // sep(1) + margin(1) + input(4) + margin(1)

// ─── Enums ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPane {
    Transcript,
    Composer,
}

/// A single autocomplete suggestion.
/// `value` is inserted into the composer; `label` is displayed (may include markers).
#[derive(Debug, Clone)]
pub struct AutocompleteItem {
    pub value: String,
    pub label: String,
}

impl AutocompleteItem {
    pub fn simple(s: impl Into<String>) -> Self {
        let s = s.into();
        Self { label: s.clone(), value: s }
    }
    pub fn labeled(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self { value: value.into(), label: label.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    User,
    Assistant,
    ToolCall,
    ToolResult,
    Summary,
    Status,
    Error,
    Attachment,
    Reasoning,
}

// ─── Structs ──────────────────────────────────────────────────────────────────

/// Character-level selection range within the transcript_lines vec.
/// All coordinates are zero-based content positions (scroll-adjusted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionRange {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize, // inclusive
}

/// A fixed character position in the rendered transcript line list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SelectionPoint {
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct TranscriptEntry {
    pub item_id: String,
    pub kind: EntryKind,
    pub title: String,
    pub body: String,
    pub timestamp: Option<i64>,
    pub pending: bool,
}

#[derive(Debug, Clone)]
pub struct PendingApprovalView {
    pub approval: PendingApproval,
    pub action_preview: String,
}

// ─── ComposerState ────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct ComposerState {
    pub chars: Vec<char>,
    pub cursor: usize,
}

impl ComposerState {
    pub fn set_text(&mut self, text: String) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub fn trimmed_text(&self) -> String {
        self.text().trim().to_string()
    }

    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    pub fn take_text(&mut self) -> String {
        let text = self.text();
        self.chars.clear();
        self.cursor = 0;
        text
    }

    pub fn insert_char(&mut self, ch: char) {
        self.chars.insert(self.cursor, ch);
        self.cursor += 1;
    }

    pub fn insert_str(&mut self, text: &str) {
        let chars = text.chars().collect::<Vec<_>>();
        let len = chars.len();
        self.chars.splice(self.cursor..self.cursor, chars);
        self.cursor += len;
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor -= 1;
        self.chars.remove(self.cursor);
    }

    pub fn delete(&mut self) {
        if self.cursor >= self.chars.len() {
            return;
        }
        self.chars.remove(self.cursor);
    }

    pub fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub fn move_right(&mut self) {
        if self.cursor < self.chars.len() {
            self.cursor += 1;
        }
    }

    pub fn move_word_left(&mut self) {
        if self.cursor == 0 {
            return;
        }

        while self.cursor > 0 && !is_word_char(self.chars[self.cursor - 1]) {
            self.cursor -= 1;
        }
        while self.cursor > 0 && is_word_char(self.chars[self.cursor - 1]) {
            self.cursor -= 1;
        }
    }

    pub fn move_word_right(&mut self) {
        while self.cursor < self.chars.len() && !is_word_char(self.chars[self.cursor]) {
            self.cursor += 1;
        }
        while self.cursor < self.chars.len() && is_word_char(self.chars[self.cursor]) {
            self.cursor += 1;
        }
    }

    pub fn move_home(&mut self) {
        while self.cursor > 0 && self.chars[self.cursor - 1] != '\n' {
            self.cursor -= 1;
        }
    }

    pub fn move_end(&mut self) {
        while self.cursor < self.chars.len() && self.chars[self.cursor] != '\n' {
            self.cursor += 1;
        }
    }

    pub fn delete_to_line_start(&mut self) {
        let start = self.current_line_start();
        if start >= self.cursor {
            return;
        }
        self.chars.drain(start..self.cursor);
        self.cursor = start;
    }

    /// Delete the word to the left of the cursor (Alt+Backspace / M-DEL).
    pub fn delete_word_left(&mut self) {
        let original = self.cursor;
        self.move_word_left();
        if self.cursor < original {
            self.chars.drain(self.cursor..original);
        }
    }

    /// Delete the word to the right of the cursor (Alt+d / M-d).
    pub fn delete_word_right(&mut self) {
        let original = self.cursor;
        self.move_word_right();
        if self.cursor > original {
            self.chars.drain(original..self.cursor);
            self.cursor = original;
        }
    }

    pub fn move_visual_up(&mut self, first_width: usize, continuation_width: usize) {
        let widths = normalized_composer_widths(first_width, continuation_width);
        let (row, col) = self.visual_cursor_row_col(widths.0, widths.1);
        if row == 0 {
            return;
        }
        self.cursor = self.cursor_for_visual_position(row - 1, col, widths);
    }

    pub fn move_visual_down(&mut self, first_width: usize, continuation_width: usize) {
        let widths = normalized_composer_widths(first_width, continuation_width);
        let (row, col) = self.visual_cursor_row_col(widths.0, widths.1);
        let total_rows = self.visual_line_count(widths.0, widths.1);
        if row + 1 >= total_rows {
            return;
        }
        self.cursor = self.cursor_for_visual_position(row + 1, col, widths);
    }

    pub fn cursor_row_col(&self) -> (usize, usize) {
        let mut row = 0;
        let mut col = 0;
        for (index, ch) in self.chars.iter().enumerate() {
            if index == self.cursor {
                break;
            }
            if *ch == '\n' {
                row += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (row, col)
    }

    fn current_line_start(&self) -> usize {
        let mut start = self.cursor;
        while start > 0 && self.chars[start - 1] != '\n' {
            start -= 1;
        }
        start
    }

    pub fn visual_line_count(&self, first_width: usize, continuation_width: usize) -> usize {
        let widths = normalized_composer_widths(first_width, continuation_width);
        let text = self.text();
        let logical_lines = if text.is_empty() {
            vec![String::new()]
        } else {
            text.split('\n').map(ToOwned::to_owned).collect::<Vec<_>>()
        };

        logical_lines
            .iter()
            .enumerate()
            .map(|(index, line)| wrapped_rows_for_line(line.chars().count(), index == 0, widths))
            .sum()
    }

    pub fn visual_cursor_row_col(
        &self,
        first_width: usize,
        continuation_width: usize,
    ) -> (usize, usize) {
        let widths = normalized_composer_widths(first_width, continuation_width);
        let (line_index, col) = self.cursor_row_col();
        let text = self.text();
        let logical_lines = if text.is_empty() {
            vec![String::new()]
        } else {
            text.split('\n').map(ToOwned::to_owned).collect::<Vec<_>>()
        };

        let mut visual_row = 0usize;
        for (index, line) in logical_lines.iter().enumerate().take(line_index) {
            visual_row += wrapped_rows_for_line(line.chars().count(), index == 0, widths);
        }

        let (row_in_line, col_in_line) = wrapped_position_in_line(col, line_index == 0, widths);
        (visual_row + row_in_line, col_in_line)
    }

    fn cursor_for_visual_position(
        &self,
        target_row: usize,
        desired_col: usize,
        widths: (usize, usize),
    ) -> usize {
        let text = self.text();
        let logical_lines = if text.is_empty() {
            vec![String::new()]
        } else {
            text.split('\n').map(ToOwned::to_owned).collect::<Vec<_>>()
        };

        let mut visual_row = 0usize;
        let mut absolute_offset = 0usize;

        for (index, line) in logical_lines.iter().enumerate() {
            let line_len = line.chars().count();
            let rows = wrapped_rows_for_line(line_len, index == 0, widths);
            if target_row < visual_row + rows {
                let local_row = target_row - visual_row;
                let line_col = line_col_for_wrapped_position(
                    line_len,
                    local_row,
                    desired_col,
                    index == 0,
                    widths,
                );
                return absolute_offset + line_col.min(line_len);
            }

            visual_row += rows;
            absolute_offset += line_len;
            if index + 1 < logical_lines.len() {
                absolute_offset += 1;
            }
        }

        self.chars.len()
    }
}

fn normalized_composer_widths(first_width: usize, continuation_width: usize) -> (usize, usize) {
    (first_width.max(1), continuation_width.max(1))
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

fn wrapped_rows_for_line(len: usize, is_first_line: bool, widths: (usize, usize)) -> usize {
    let (first_width, continuation_width) = widths;
    let initial_width = if is_first_line {
        first_width
    } else {
        continuation_width
    };

    if len <= initial_width {
        return 1;
    }

    let remaining = len - initial_width;
    1 + remaining.div_ceil(continuation_width)
}

fn wrapped_position_in_line(
    col: usize,
    is_first_line: bool,
    widths: (usize, usize),
) -> (usize, usize) {
    let (first_width, continuation_width) = widths;
    let initial_width = if is_first_line {
        first_width
    } else {
        continuation_width
    };

    if col < initial_width {
        return (0, col);
    }

    let remaining = col - initial_width;
    (
        1 + (remaining / continuation_width),
        remaining % continuation_width,
    )
}

fn line_col_for_wrapped_position(
    len: usize,
    row: usize,
    desired_col: usize,
    is_first_line: bool,
    widths: (usize, usize),
) -> usize {
    let (first_width, continuation_width) = widths;
    let initial_width = if is_first_line {
        first_width
    } else {
        continuation_width
    };

    if row == 0 {
        return desired_col.min(len.min(initial_width));
    }

    let row_start = initial_width + (row - 1) * continuation_width;
    if row_start >= len {
        return len;
    }
    let available = (len - row_start).min(continuation_width);
    row_start + desired_col.min(available)
}

// ─── Sigils ───────────────────────────────────────────────────────────────────

/// Returns (sigil_str, sigil_color, body_color) for each entry kind.
pub fn entry_style(kind: EntryKind) -> (&'static str, Color, Color) {
    match kind {
        EntryKind::User => ("▶", COLOR_USER, Color::White),
        EntryKind::Assistant => ("◆", COLOR_ASSISTANT, Color::White),
        EntryKind::ToolCall => ("⚙", COLOR_TOOL, Color::Rgb(240, 218, 255)),
        EntryKind::ToolResult => ("✓", COLOR_TOOL_RESULT, Color::Rgb(210, 240, 220)),
        EntryKind::Summary => ("◈", COLOR_SUMMARY, Color::Rgb(230, 222, 255)),
        EntryKind::Status => ("·", COLOR_STATUS, Color::Rgb(200, 225, 200)),
        EntryKind::Error => ("✗", COLOR_ERROR, Color::Rgb(255, 210, 210)),
        EntryKind::Attachment => ("⊞", COLOR_MUTED, Color::Rgb(200, 210, 225)),
        EntryKind::Reasoning => ("⋯", COLOR_REASONING, Color::Rgb(220, 215, 255)),
    }
}

// ─── Utility functions shared across tui modules ──────────────────────────────

pub fn entry_from_item(item: &ConversationItem) -> TranscriptEntry {
    match item.kind {
        ItemKind::UserMessage => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::User,
            title: "You".into(),
            body: item
                .payload
                .get("text")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default()
                .to_string(),
            timestamp: Some(item.created_at),
            pending: false,
        },
        ItemKind::AgentMessage => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::Assistant,
            title: "Codezilla".into(),
            body: item
                .payload
                .get("text")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default()
                .to_string(),
            timestamp: Some(item.created_at),
            pending: false,
        },
        ItemKind::ToolCall => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::ToolCall,
            title: format!(
                "{}",
                item.payload
                    .get("toolName")
                    .and_then(|v: &serde_json::Value| v.as_str())
                    .unwrap_or("tool")
            ),
            body: format_tool_call(
                item.payload
                    .get("toolName")
                    .and_then(|v: &serde_json::Value| v.as_str())
                    .unwrap_or("tool"),
                item.payload
                    .get("arguments")
                    .unwrap_or(&serde_json::Value::Null),
            ),
            timestamp: Some(item.created_at),
            pending: false,
        },
        ItemKind::ToolResult => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::ToolResult,
            title: "result".into(),
            body: format_tool_result(item.payload.get("output"), item.payload.get("errorMessage")),
            timestamp: Some(item.created_at),
            pending: false,
        },
        ItemKind::ReasoningSummary => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::Summary,
            title: "summary".into(),
            body: item
                .payload
                .get("summary")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_else(|| item.payload.as_str().unwrap_or_default())
                .to_string(),
            timestamp: Some(item.created_at),
            pending: false,
        },
        ItemKind::ReasoningText => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::Reasoning,
            title: "thinking".into(),
            body: item
                .payload
                .get("text")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default()
                .to_string(),
            timestamp: Some(item.created_at),
            pending: false,
        },
        ItemKind::UserAttachment => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::Attachment,
            title: "attachment".into(),
            body: item
                .payload
                .get("path")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default()
                .to_string(),
            timestamp: Some(item.created_at),
            pending: false,
        },
        ItemKind::Error => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::Error,
            title: "error".into(),
            body: pretty_json_or_text(Some(&item.payload), None),
            timestamp: Some(item.created_at),
            pending: false,
        },
        _ => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::Status,
            title: format!("{:?}", item.kind).to_lowercase(),
            body: pretty_json_or_text(Some(&item.payload), None),
            timestamp: Some(item.created_at),
            pending: false,
        },
    }
}

/// Render transcript as ratatui Lines, stream-style (no box borders).
/// `width` is the viewport width so body lines are pre-wrapped here
/// (avoids ratatui's own wrapping which breaks drag selection coordinates).
/// `selection` applies character-level highlight to the given range.
pub fn transcript_lines(
    entries: &[TranscriptEntry],
    spinner_tick: u64,
    width: u16,
    selection: Option<SelectionRange>,
) -> (Vec<Line<'static>>, usize) {
    transcript_window_lines(entries, spinner_tick, width, selection, 0, usize::MAX)
}

/// Render only a window of transcript lines while still returning the total
/// rendered line count for scroll calculations.
pub fn transcript_window_lines(
    entries: &[TranscriptEntry],
    spinner_tick: u64,
    width: u16,
    selection: Option<SelectionRange>,
    start_line: usize,
    max_lines: usize,
) -> (Vec<Line<'static>>, usize) {
    if entries.is_empty() {
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No messages yet — type below and press Enter",
                Style::default().fg(COLOR_MUTED),
            )),
            Line::from(""),
        ];
        let n = lines.len();
        return (lines, n);
    }

    let mut lines: Vec<Line<'static>> = Vec::new();
    let body_width = (width as usize).saturating_sub(5).max(1); // "  │  " = 5 chars
    let end_line = start_line.saturating_add(max_lines);
    let mut line_index = 0usize;

    for entry in entries {
        let entry_line_count = transcript_entry_line_count(entry, body_width);
        let entry_start = line_index;
        let entry_end = line_index + entry_line_count;

        if entry_end > start_line && entry_start < end_line {
            append_transcript_entry_lines(
                &mut lines,
                entry,
                spinner_tick,
                body_width,
                start_line,
                end_line,
                entry_start,
            );
        }

        line_index = entry_end;
    }

    // ── Apply character-level selection highlight ─────────────────────────────
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

    (lines, line_index)
}

/// Split `s` into chunks of at most `width` chars (hard-wrap, no word break).
/// Always returns at least one element even for empty strings.
pub fn split_at_width(s: &str, width: usize) -> Vec<String> {
    if s.is_empty() {
        return vec![String::new()];
    }
    let chars: Vec<char> = s.chars().collect();
    chars
        .chunks(width.max(1))
        .map(|c| c.iter().collect())
        .collect()
}

fn transcript_entry_line_count(entry: &TranscriptEntry, body_width: usize) -> usize {
    let body_lines = if entry.body.is_empty() && entry.pending {
        1
    } else {
        entry
            .body
            .split('\n')
            .map(|body_line| split_at_width(body_line, body_width).len())
            .sum()
    };
    1 + body_lines + 1
}

fn append_transcript_entry_lines(
    out: &mut Vec<Line<'static>>,
    entry: &TranscriptEntry,
    spinner_tick: u64,
    body_width: usize,
    start_line: usize,
    end_line: usize,
    entry_start: usize,
) {
    let (sigil, sigil_color, body_color) = entry_style(entry.kind);
    let mut current_line = entry_start;

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
                spinner_frame(spinner_tick).to_string(),
                Style::default()
                    .fg(COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        out.push(Line::from(header_spans));
    }
    current_line += 1;

    if entry.body.is_empty() && entry.pending {
        if current_line >= start_line && current_line < end_line {
            out.push(Line::from(vec![
                Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
                Span::styled("…".to_string(), Style::default().fg(COLOR_MUTED)),
            ]));
        }
        current_line += 1;
    } else {
        for body_line in entry.body.split('\n') {
            for chunk in split_at_width(body_line, body_width) {
                if current_line >= start_line && current_line < end_line {
                    out.push(Line::from(vec![
                        Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
                        Span::styled(chunk, Style::default().fg(body_color)),
                    ]));
                }
                current_line += 1;
            }
        }
    }

    if current_line >= start_line && current_line < end_line {
        out.push(Line::from(""));
    }
}

/// Split spans at column boundaries so that [col_from, col_to] (inclusive)
/// gets `Modifier::REVERSED` while the rest keeps its original style.
fn apply_char_selection(line: Line<'static>, col_from: usize, col_to: usize) -> Line<'static> {
    let mut result: Vec<Span<'static>> = Vec::new();
    let mut cursor = 0usize;
    let col_end = col_to.saturating_add(1); // exclusive

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

pub fn composer_height(composer: &ComposerState, width: u16) -> u16 {
    // Both lines wrap at (width - 5): the unified prefix width used in render_composer.
    let text_width = width.saturating_sub(5) as usize;
    let rows = composer.visual_line_count(text_width, text_width) as u16;
    // +3 = separator(1) + top-margin(1) + bottom-margin(1) inside render_composer
    (rows + 3).clamp(COMPOSER_MIN_HEIGHT, COMPOSER_MAX_HEIGHT)
}

pub fn truncate_lines(text: &str, max_lines: usize) -> String {
    let mut lines = text.lines().take(max_lines).collect::<Vec<_>>();
    if text.lines().count() > max_lines {
        lines.push("…");
    }
    lines.join("\n")
}

pub fn format_timestamp(timestamp: i64) -> String {
    use chrono::TimeZone;
    chrono::Local
        .timestamp_opt(timestamp, 0)
        .single()
        .map(|v| v.format("%H:%M").to_string())
        .unwrap_or_else(|| timestamp.to_string())
}

pub fn basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|v| v.to_str())
        .filter(|v| !v.is_empty())
        .unwrap_or(path)
        .to_string()
}

pub fn current_state_label(has_active_turn: bool, awaiting_approval: bool) -> &'static str {
    if awaiting_approval {
        "approval"
    } else if has_active_turn {
        "streaming"
    } else {
        "ready"
    }
}

pub fn short_thread_id(thread_id: &str) -> String {
    thread_id
        .strip_prefix("thread_")
        .unwrap_or(thread_id)
        .chars()
        .take(8)
        .collect()
}

pub fn short_turn_id(turn_id: &str) -> String {
    turn_id
        .strip_prefix("turn_")
        .unwrap_or(turn_id)
        .chars()
        .take(8)
        .collect()
}

pub fn thread_label(thread: &ThreadMetadata) -> String {
    if let Some(title) = &thread.title {
        if !title.trim().is_empty() {
            return title.clone();
        }
    }
    format!("thread {}", short_thread_id(&thread.thread_id))
}

pub fn pretty_json_or_text(primary: Option<&Value>, secondary: Option<&Value>) -> String {
    if let Some(text) = primary.and_then(|v| v.as_str()) {
        return text.to_string();
    }
    if let Some(value) = primary {
        if !value.is_null() {
            return serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
        }
    }
    if let Some(text) = secondary.and_then(|v| v.as_str()) {
        return text.to_string();
    }
    if let Some(value) = secondary {
        if !value.is_null() {
            return serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
        }
    }
    String::new()
}

fn format_tool_call(tool_name: &str, arguments: &Value) -> String {
    match tool_name {
        "shell_exec" => {
            let mut lines = Vec::new();
            if let Some(argv) = arguments.get("argv").and_then(Value::as_array) {
                let command = argv
                    .iter()
                    .filter_map(Value::as_str)
                    .map(shell_escape)
                    .collect::<Vec<_>>()
                    .join(" ");
                if !command.is_empty() {
                    lines.push(command);
                }
            }
            if let Some(cwd) = arguments.get("cwd").and_then(Value::as_str) {
                lines.push(format!("cwd: {cwd}"));
            }
            if lines.is_empty() {
                pretty_json_or_text(Some(arguments), None)
            } else {
                lines.join("\n")
            }
        }
        "read_file" | "write_file" | "create_directory" | "remove_path" => {
            let mut lines = Vec::new();
            if let Some(path) = arguments.get("path").and_then(Value::as_str) {
                lines.push(path.to_string());
            }
            if tool_name == "write_file" {
                if let Some(content) = arguments.get("content").and_then(Value::as_str) {
                    let line_count = content.lines().count().max(1);
                    lines.push(format!("content: {line_count} lines"));
                }
            }
            if lines.is_empty() {
                pretty_json_or_text(Some(arguments), None)
            } else {
                lines.join("\n")
            }
        }
        "copy_path" => {
            let source = arguments.get("source").and_then(Value::as_str);
            let target = arguments.get("target").and_then(Value::as_str);
            match (source, target) {
                (Some(source), Some(target)) => format!("{source}\n-> {target}"),
                _ => pretty_json_or_text(Some(arguments), None),
            }
        }
        _ => pretty_json_or_text(Some(arguments), None),
    }
}

pub fn format_tool_result(output: Option<&Value>, error_message: Option<&Value>) -> String {
    let Some(output) = output else {
        return pretty_json_or_text(None, error_message);
    };

    if let Some(formatted) = format_shell_result(output) {
        return formatted;
    }

    pretty_json_or_text(Some(output), error_message)
}

fn format_shell_result(output: &Value) -> Option<String> {
    let command = output.get("command")?.as_array()?;
    let command = command
        .iter()
        .filter_map(Value::as_str)
        .map(shell_escape)
        .collect::<Vec<_>>()
        .join(" ");

    let mut lines = Vec::new();
    if !command.is_empty() {
        lines.push(format!("$ {command}"));
    }
    let cwd = output.get("cwd").and_then(Value::as_str);
    let exit_code = output.get("exitCode").and_then(Value::as_i64);
    match (cwd, exit_code) {
        (Some(cwd), Some(exit_code)) => lines.push(format!("{cwd}  ·  exit {exit_code}")),
        (Some(cwd), None) => lines.push(cwd.to_string()),
        (None, Some(exit_code)) => lines.push(format!("exit {exit_code}")),
        (None, None) => {}
    }

    let stdout = output
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim_end();
    let stderr = output
        .get("stderr")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim_end();

    if !stdout.is_empty() {
        lines.push(String::new());
        lines.push(stdout.to_string());
    }
    if !stderr.is_empty() {
        lines.push(String::new());
        lines.push("stderr:".to_string());
        lines.push(stderr.to_string());
    }
    if output
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        lines.push(String::new());
        lines.push("output truncated".to_string());
    }

    Some(lines.join("\n"))
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
