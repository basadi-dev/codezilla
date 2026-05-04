use serde_json::Value;
use std::path::Path;

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::markdown::{highlight_code_line, lang_for_path, md_to_lines};

use crate::system::domain::PendingApproval;
use crate::system::domain::{
    ConversationItem, ItemKind, ThreadMetadata, KEY_ERROR, LABEL_ASSISTANT, LABEL_ERROR,
    LABEL_SYSTEM, LABEL_USER, STATUS_COMPLETED, STATUS_FAILED, STATUS_INTERRUPTED,
    STATUS_TIMED_OUT, STATUS_TIMEOUT,
};

// ─── Colour palette ── (Claude Code / Codex CLI inspired) ────────────────────

/// Diff added-line background (subtle green tint)
pub const BG_DIFF_ADD: Color = Color::Rgb(20, 60, 30);
/// Diff removed-line background (subtle red tint)
pub const BG_DIFF_REMOVE: Color = Color::Rgb(60, 20, 20);

/// Background of focused input border / streaming accent
pub const COLOR_ACCENT: Color = Color::Rgb(100, 200, 163); // soft mint-green
/// Dimmed muted text, timestamps, hints
pub const COLOR_MUTED: Color = Color::Rgb(110, 118, 135);
/// Even dimmer text for secondary metadata (durations, gaps)
pub const COLOR_DIM: Color = Color::Rgb(80, 86, 100);
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
/// Warning sigil + text (informational, not a hard failure)
pub const COLOR_WARNING: Color = Color::Rgb(255, 200, 80); // warm amber-yellow
/// Reasoning sigil
pub const COLOR_REASONING: Color = Color::Rgb(180, 160, 255); // lavender
/// System message sigil + text
pub const COLOR_SYSTEM: Color = Color::Rgb(160, 180, 200); // cool slate
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
        Self {
            label: s.clone(),
            value: s,
        }
    }
    pub fn labeled(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    User,
    System,
    Assistant,
    ToolCall,
    ToolResult,
    Summary,
    Status,
    #[allow(dead_code)]
    Warning,
    Error,
    Attachment,
    Reasoning,
    FileChange,
    Command,
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
    pub turn_id: Option<String>,
    pub tool_call_id: Option<String>,
    pub kind: EntryKind,
    pub title: String,
    pub body: String,
    pub timestamp: Option<i64>,
    pub completed_at: Option<i64>,
    pub pending: bool,
    /// When true, only the header line is shown in the transcript; the body
    /// is hidden. Read-only tool calls auto-collapse to reduce noise.
    pub collapsed: bool,
}

#[derive(Debug, Clone)]
pub struct PendingApprovalView {
    pub approval: PendingApproval,
    pub action_preview: String,
}

// ─── ComposerState ────────────────────────────────────────────────────────────

/// Pastes larger than this many characters are stored out-of-band and shown
/// as a `[pasted N chars]` placeholder so the composer stays responsive.
const PASTE_PLACEHOLDER_THRESHOLD: usize = 500;

/// The placeholder text injected into the composer for large pastes.
/// Brackets make it behave as a single token for Alt+Backspace deletion.
fn paste_placeholder(char_count: usize) -> String {
    format!("[pasted {char_count} chars]")
}

#[derive(Debug, Default, Clone)]
pub struct ComposerState {
    pub chars: Vec<char>,
    pub cursor: usize,
    /// Holds the full text of a large paste that was replaced by a placeholder.
    /// `take_text()` splices it back before returning the final string.
    pasted_text: Option<String>,
}

impl ComposerState {
    pub fn set_text(&mut self, text: String) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
        self.pasted_text = None;
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

    /// Clears the composer and returns the final text, expanding any paste
    /// placeholder back to the original full content before returning.
    pub fn take_text(&mut self) -> String {
        let displayed = self.text();
        self.chars.clear();
        self.cursor = 0;

        let text = if let Some(real) = self.pasted_text.take() {
            // Replace the placeholder token with the stored real text.
            let placeholder = paste_placeholder(real.chars().count());
            displayed.replacen(&placeholder, &real, 1)
        } else {
            displayed
        };

        text
    }

    /// Returns true when a large paste placeholder is currently active.
    #[allow(dead_code)]
    pub fn has_paste_placeholder(&self) -> bool {
        self.pasted_text.is_some()
    }

    pub fn insert_char(&mut self, ch: char) {
        self.chars.insert(self.cursor, ch);
        self.cursor += 1;
    }

    /// Insert a string into the composer.  If the string is longer than
    /// [`PASTE_PLACEHOLDER_THRESHOLD`] characters **and** the composer is
    /// currently empty (a fresh paste into a blank input), the real text is
    /// stored in `pasted_text` and only a compact placeholder is inserted so
    /// the composer widget stays snappy.
    pub fn insert_str(&mut self, text: &str) {
        let char_count = text.chars().count();
        if char_count >= PASTE_PLACEHOLDER_THRESHOLD
            && self.pasted_text.is_none()
            && self.chars.is_empty()
        {
            // Store the full paste and show a compact stand-in instead.
            self.pasted_text = Some(text.to_string());
            let placeholder = paste_placeholder(char_count);
            let ph_chars: Vec<char> = placeholder.chars().collect();
            let ph_len = ph_chars.len();
            self.chars.splice(self.cursor..self.cursor, ph_chars);
            self.cursor += ph_len;
        } else {
            let chars = text.chars().collect::<Vec<_>>();
            let len = chars.len();
            self.chars.splice(self.cursor..self.cursor, chars);
            self.cursor += len;
        }
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

    /// Return the character index of the start of the line containing `pos`.
    pub fn line_start(&self, pos: usize) -> usize {
        let pos = pos.min(self.chars.len());
        let mut i = pos;
        while i > 0 && self.chars[i - 1] != '\n' {
            i -= 1;
        }
        i
    }

    /// Return the character index just past the end of the line containing `pos`
    /// (excludes the trailing newline, if any).
    pub fn line_end(&self, pos: usize) -> usize {
        let pos = pos.min(self.chars.len());
        let mut i = pos;
        while i < self.chars.len() && self.chars[i] != '\n' {
            i += 1;
        }
        i
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

    pub fn cursor_for_visual_position(
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

    /// Convert a visual (row, col) position back to a character index in
    /// `self.chars`. This is the inverse of `visual_cursor_row_col` and is
    /// used for mouse hit-testing in the composer.
    pub fn index_for_visual_position(
        &self,
        target_row: usize,
        desired_col: usize,
        first_width: usize,
        continuation_width: usize,
    ) -> usize {
        let widths = normalized_composer_widths(first_width, continuation_width);
        let text = self.text();
        let logical_lines: Vec<&str> = if text.is_empty() {
            vec![""]
        } else {
            text.split('\n').collect()
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
                absolute_offset += 1; // the '\n'
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

pub(crate) fn wrapped_rows_for_line(
    len: usize,
    is_first_line: bool,
    widths: (usize, usize),
) -> usize {
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
        EntryKind::System => ("◈", COLOR_SYSTEM, COLOR_SYSTEM),
        EntryKind::Assistant => ("◆", COLOR_ASSISTANT, Color::White),
        EntryKind::ToolCall => ("⚙", COLOR_TOOL, Color::Rgb(240, 218, 255)),
        EntryKind::ToolResult => ("✓", COLOR_TOOL_RESULT, Color::Rgb(210, 240, 220)),
        EntryKind::Summary => ("◈", COLOR_SUMMARY, Color::Rgb(230, 222, 255)),
        EntryKind::Status => ("·", COLOR_STATUS, Color::Rgb(200, 225, 200)),
        EntryKind::Warning => ("⚠", COLOR_WARNING, Color::Rgb(255, 240, 200)),
        EntryKind::Error => ("✗", COLOR_ERROR, Color::Rgb(255, 210, 210)),
        EntryKind::Attachment => ("⊞", COLOR_MUTED, Color::Rgb(200, 210, 225)),
        EntryKind::Reasoning => ("⋯", COLOR_REASONING, Color::Rgb(220, 215, 255)),
        EntryKind::FileChange => ("✏", COLOR_WARNING, Color::Rgb(255, 230, 190)),
        EntryKind::Command => ("$", COLOR_WARNING, Color::Rgb(200, 220, 200)),
    }
}

// ─── Utility functions shared across tui modules ──────────────────────────────

/// Returns true for tools whose transcript output should be collapsed by default.
///
/// Read-only / low-signal tools collapse to a header-only line to reduce
/// transcript noise. Action tools (bash, write, patch, spawn_agent) remain
/// expanded because users need to see their output (diffs, command results).
pub fn should_auto_collapse_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file"
            | "list_dir"
            | "grep_search"
            | "web_fetch"
            | "image_metadata"
            | "create_directory"
            | "remove_path"
            | "copy_path"
    )
}

pub fn entry_from_item(item: &ConversationItem) -> TranscriptEntry {
    let tool_call_id = item
        .payload
        .get("toolCallId")
        .and_then(|v: &serde_json::Value| v.as_str())
        .map(ToOwned::to_owned);

    match item.kind {
        ItemKind::UserMessage => TranscriptEntry {
            item_id: item.item_id.clone(),
            turn_id: Some(item.turn_id.clone()),
            tool_call_id: None,
            kind: EntryKind::User,
            title: LABEL_USER.into(),
            body: item
                .payload
                .get("text")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default()
                .to_string(),
            timestamp: Some(item.created_at),
            completed_at: None,
            pending: false,
            collapsed: false,
        },
        ItemKind::SystemMessage => TranscriptEntry {
            item_id: item.item_id.clone(),
            turn_id: Some(item.turn_id.clone()),
            tool_call_id: None,
            kind: EntryKind::System,
            title: LABEL_SYSTEM.into(),
            body: item
                .payload
                .get("text")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default()
                .to_string(),
            timestamp: Some(item.created_at),
            completed_at: None,
            pending: false,
            collapsed: false,
        },
        ItemKind::AgentMessage => TranscriptEntry {
            item_id: item.item_id.clone(),
            turn_id: Some(item.turn_id.clone()),
            tool_call_id: None,
            kind: EntryKind::Assistant,
            title: LABEL_ASSISTANT.into(),
            body: item
                .payload
                .get("text")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default()
                .to_string(),
            timestamp: Some(item.created_at),
            completed_at: None,
            pending: false,
            collapsed: false,
        },
        ItemKind::ToolCall => {
            let tool_name = item
                .payload
                .get("toolName")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("tool");
            let arguments = item
                .payload
                .get("arguments")
                .unwrap_or(&serde_json::Value::Null);
            TranscriptEntry {
                item_id: item.item_id.clone(),
                turn_id: Some(item.turn_id.clone()),
                tool_call_id,
                kind: EntryKind::ToolCall,
                title: format_tool_call_title(tool_name, arguments),
                body: format_tool_call(tool_name, arguments),
                timestamp: Some(item.created_at),
                completed_at: None,
                pending: false,
                collapsed: should_auto_collapse_tool(tool_name),
            }
        }
        ItemKind::ToolResult => TranscriptEntry {
            item_id: item.item_id.clone(),
            turn_id: Some(item.turn_id.clone()),
            tool_call_id,
            kind: EntryKind::ToolResult,
            title: "result".into(),
            body: format_tool_result(item.payload.get("output"), item.payload.get("errorMessage")),
            timestamp: Some(item.created_at),
            completed_at: None,
            pending: false,
            collapsed: false,
        },
        ItemKind::ReasoningSummary => TranscriptEntry {
            item_id: item.item_id.clone(),
            turn_id: Some(item.turn_id.clone()),
            tool_call_id: None,
            kind: EntryKind::Summary,
            title: "summary".into(),
            body: item
                .payload
                .get("text")
                .or_else(|| item.payload.get("summary"))
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_else(|| item.payload.as_str().unwrap_or_default())
                .to_string(),
            timestamp: Some(item.created_at),
            completed_at: None,
            pending: false,
            collapsed: false,
        },
        ItemKind::ReasoningText => TranscriptEntry {
            item_id: item.item_id.clone(),
            turn_id: Some(item.turn_id.clone()),
            tool_call_id: None,
            kind: EntryKind::Reasoning,
            title: "thinking".into(),
            body: item
                .payload
                .get("text")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default()
                .to_string(),
            timestamp: Some(item.created_at),
            completed_at: None,
            pending: false,
            collapsed: false,
        },
        ItemKind::UserAttachment => TranscriptEntry {
            item_id: item.item_id.clone(),
            turn_id: Some(item.turn_id.clone()),
            tool_call_id: None,
            kind: EntryKind::Attachment,
            title: "attachment".into(),
            body: item
                .payload
                .get("path")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default()
                .to_string(),
            timestamp: Some(item.created_at),
            completed_at: None,
            pending: false,
            collapsed: false,
        },
        ItemKind::Error => {
            // Prefer structured { kind, message } payload produced by the new error pipeline.
            let kind_label = item
                .payload
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or(LABEL_ERROR)
                .to_string();
            let fallback_body = pretty_json_or_text(Some(&item.payload), None);
            let message = item
                .payload
                .get("message")
                .and_then(|v| v.as_str())
                .or_else(|| item.payload.get("reason").and_then(|v| v.as_str()))
                .map(|s| s.to_string())
                .unwrap_or(fallback_body);
            TranscriptEntry {
                item_id: item.item_id.clone(),
                turn_id: Some(item.turn_id.clone()),
                tool_call_id: None,
                kind: EntryKind::Error,
                title: kind_label,
                body: message,
                timestamp: Some(item.created_at),
                completed_at: None,
                pending: false,
                collapsed: false,
            }
        }
        ItemKind::FileChange => {
            let path = item
                .payload
                .get("path")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("");
            let change_kind = item
                .payload
                .get("changeKind")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("modified");
            let diff = item
                .payload
                .get("diff")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default();
            let summary = item
                .payload
                .get("summary")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default();

            // Format body as a unified-diff-like text so the existing diff
            // highlighter in append_transcript_entry_lines renders it.
            let body = if !diff.is_empty() {
                format!("--- a/{path}\n+++ b/{path}\n{diff}")
            } else if !summary.is_empty() {
                summary.to_string()
            } else {
                format!("{change_kind} {path}")
            };

            TranscriptEntry {
                item_id: item.item_id.clone(),
                turn_id: Some(item.turn_id.clone()),
                tool_call_id: None,
                kind: EntryKind::FileChange,
                title: path.rsplit('/').next().unwrap_or(path).to_string(),
                body,
                timestamp: Some(item.created_at),
                completed_at: None,
                pending: false,
                collapsed: false,
            }
        }
        ItemKind::CommandExecution => {
            let cmd_parts: Vec<String> = item
                .payload
                .get("command")
                .and_then(|v: &serde_json::Value| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let cmd_str = if cmd_parts.is_empty() {
                item.payload
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            } else {
                cmd_parts.join(" ")
            };
            let exit_code = item.payload.get("exitCode").and_then(|v| v.as_i64());
            let mut body = String::new();
            if !cmd_str.is_empty() {
                body.push_str(&cmd_str);
            }
            if let Some(code) = exit_code {
                if !body.is_empty() {
                    body.push('\n');
                }
                body.push_str(&format!("exit code: {code}"));
            }

            TranscriptEntry {
                item_id: item.item_id.clone(),
                turn_id: Some(item.turn_id.clone()),
                tool_call_id: None,
                kind: EntryKind::Command,
                title: cmd_str
                    .split_whitespace()
                    .next()
                    .unwrap_or("cmd")
                    .to_string(),
                body,
                timestamp: Some(item.created_at),
                completed_at: None,
                pending: false,
                collapsed: false,
            }
        }
        ItemKind::CommandOutput => {
            let stdout = item
                .payload
                .get("stdout")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default();
            let stderr = item
                .payload
                .get("stderr")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or_default();
            let exit_code = item.payload.get("exitCode").and_then(|v| v.as_i64());
            let mut body = String::new();
            if !stdout.is_empty() {
                body.push_str(stdout);
            }
            if !stderr.is_empty() {
                if !body.is_empty() {
                    body.push('\n');
                }
                body.push_str(&format!("stderr:\n{}", stderr));
            }
            if let Some(code) = exit_code {
                body.push_str(&format!("\nexit code: {code}"));
            }

            TranscriptEntry {
                item_id: item.item_id.clone(),
                turn_id: Some(item.turn_id.clone()),
                tool_call_id: None,
                kind: EntryKind::Command,
                title: "output".into(),
                body,
                timestamp: Some(item.created_at),
                completed_at: None,
                pending: false,
                collapsed: false,
            }
        }
        _ => TranscriptEntry {
            item_id: item.item_id.clone(),
            turn_id: Some(item.turn_id.clone()),
            tool_call_id: None,
            collapsed: false,
            kind: EntryKind::Status,
            title: format!("{:?}", item.kind).to_lowercase(),
            body: pretty_json_or_text(Some(&item.payload), None),
            timestamp: Some(item.created_at),
            completed_at: None,
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
    let now_ts = chrono::Utc::now().timestamp();
    let mut line_index = 0usize;
    let mut prev_timestamp: Option<i64> = None;
    let mut prev_user_timestamp: Option<i64> = None;

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
                prev_timestamp,
                prev_user_timestamp,
                now_ts,
            );
        }

        // Track timestamps for duration/gap calculations
        if entry.timestamp.is_some() {
            prev_timestamp = entry.timestamp;
        }
        if entry.kind == EntryKind::User && entry.timestamp.is_some() {
            prev_user_timestamp = entry.timestamp;
        }

        line_index = entry_end;
    }

    // Apply character-level selection highlighting (drag-to-select)
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
    let is_working_timer = entry.item_id.starts_with("__codezilla_working__");
    let body_lines = if entry.collapsed && !entry.pending {
        // Collapsed entries show header + trailing blank only (no body).
        0
    } else if entry.body.is_empty() && is_working_timer {
        0
    } else if entry.body.is_empty() && entry.pending {
        1
    } else if matches!(
        entry.kind,
        EntryKind::Assistant | EntryKind::Summary | EntryKind::Reasoning
    ) {
        // Use the actual rendered markdown line count for accurate scroll math.
        md_to_lines(&entry.body, Color::White, body_width)
            .len()
            .max(1)
    } else {
        entry
            .body
            .split('\n')
            .map(|body_line| split_at_width(body_line, body_width).len())
            .sum::<usize>()
            .max(1)
    };
    1 + body_lines + 1
}

#[allow(clippy::too_many_arguments)]
fn append_transcript_entry_lines(
    out: &mut Vec<Line<'static>>,
    entry: &TranscriptEntry,
    spinner_tick: u64,
    body_width: usize,
    start_line: usize,
    end_line: usize,
    entry_start: usize,
    _prev_timestamp: Option<i64>,
    prev_user_timestamp: Option<i64>,
    now_ts: i64,
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
            // Show this entry's own elapsed time. Pending entries count up;
            // completed entries keep their frozen completion duration.
            if let Some(elapsed) =
                entry_elapsed_secs(entry.timestamp, entry.completed_at, entry.pending, now_ts)
            {
                header_spans.push(Span::styled(
                    format!(" · {}", format_duration(elapsed)),
                    Style::default().fg(COLOR_DIM),
                ));
            }
            // Show time since last user message (if this isn't a user message itself)
            if entry.kind != EntryKind::User {
                if let Some(user_ts) = prev_user_timestamp {
                    let since_user = ts - user_ts;
                    if since_user > 0 {
                        header_spans.push(Span::styled(
                            format!(" · +{}", format_duration(since_user)),
                            Style::default().fg(COLOR_DIM),
                        ));
                    }
                }
            }
        }
        // Spinner is reserved for the Working timer entry; see app.rs
        // append_cached_transcript_entry_lines for the rationale.
        let header_is_working_timer = entry.item_id.starts_with("__codezilla_working__");
        if entry.pending && header_is_working_timer {
            header_spans.push(Span::raw("  "));
            header_spans.push(Span::styled(
                spinner_frame(spinner_tick).to_string(),
                Style::default()
                    .fg(COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        // Show collapse/expand chevron on collapsible entries.
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
        EntryKind::Assistant | EntryKind::Summary | EntryKind::Reasoning
    );
    // The Working timer entry uses the header (title + spinner) for everything;
    // skip the body section entirely when its body is empty (both the pending
    // placeholder fallback and the trailing-empty-line fall-through).
    let is_working_timer = entry.item_id.starts_with("__codezilla_working__");
    if is_working_timer && entry.body.is_empty() {
        if current_line >= start_line && current_line < end_line {
            out.push(Line::from(""));
        }
        return;
    }
    if entry.body.is_empty() && entry.pending && !is_working_timer {
        if current_line >= start_line && current_line < end_line {
            // Show a clear animated "working" indicator in the transcript body
            let spinner = spinner_frame(spinner_tick);
            let working_text = match entry.kind {
                EntryKind::Assistant => "thinking",
                EntryKind::ToolCall => "calling tool",
                EntryKind::ToolResult => "waiting",
                EntryKind::Reasoning => "reasoning",
                _ => "working",
            };
            out.push(Line::from(vec![
                Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
                Span::styled(
                    format!("{spinner}  {working_text}…"),
                    Style::default()
                        .fg(COLOR_ACCENT)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
        }
        current_line += 1;
    } else if use_markdown {
        // Render via pulldown-cmark for rich Markdown (headings, tables, code, etc.)
        let md_rendered = md_to_lines(&entry.body, body_color, body_width);
        for md_line in md_rendered {
            if current_line >= start_line && current_line < end_line {
                // Prepend gutter to each rendered Markdown line.
                let mut spans = vec![Span::styled("  │  ", Style::default().fg(COLOR_MUTED))];
                spans.extend(md_line.spans);
                out.push(Line::from(spans));
            }
            current_line += 1;
        }
    } else if entry.kind == EntryKind::FileChange && is_diff_body(&entry.body) {
        // FileChange entries with diff content — use diff highlighter.
        let lang = diff_lang_for_body(&entry.body);
        for body_line in entry.body.split('\n') {
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
    } else if (entry.kind == EntryKind::ToolResult || entry.kind == EntryKind::ToolCall)
        && is_read_file_body(&entry.body)
    {
        let lang = read_file_lang_for_body(&entry.body);
        for rendered_line in render_read_file_body_lines(&entry.body, lang, body_width) {
            if current_line >= start_line && current_line < end_line {
                let mut spans = vec![Span::styled("  │  ", Style::default().fg(COLOR_MUTED))];
                spans.extend(rendered_line);
                out.push(Line::from(spans));
            }
            current_line += 1;
        }
    } else if (entry.kind == EntryKind::ToolResult || entry.kind == EntryKind::ToolCall)
        && is_diff_body(&entry.body)
    {
        // Unified diff / write_file result. Reuse the source highlighter when we
        // can infer a language from the changed path, but keep the diff markers
        // colored so additions/removals still read as a diff.
        let lang = diff_lang_for_body(&entry.body);
        for body_line in entry.body.split('\n') {
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
    } else if entry.kind == EntryKind::Command {
        // Command entries: first line is the command ($ prefix), rest is output.
        // If the output portion looks like a unified diff, render it with
        // diff highlighting (green/red backgrounds, syntax colouring).
        let mut lines_iter = entry.body.split('\n');
        let cmd_line = lines_iter.next().unwrap_or("");
        let output_rest: Vec<&str> = lines_iter.collect();
        let output_body: String = output_rest.join("\n");
        let diff_output = is_diff_body(&output_body);
        let diff_lang = if diff_output {
            diff_lang_for_body(&output_body)
        } else {
            ""
        };

        // Render the command line ($ prefix)
        for chunk in split_at_width(cmd_line, body_width) {
            if current_line >= start_line && current_line < end_line {
                out.push(Line::from(vec![
                    Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
                    Span::styled(
                        "$ ",
                        Style::default()
                            .fg(COLOR_WARNING)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(chunk, Style::default().fg(body_color)),
                ]));
            }
            current_line += 1;
        }

        // Render output lines — diff-highlighted or plain
        for body_line in &output_rest {
            for chunk in split_at_width(body_line, body_width) {
                if current_line >= start_line && current_line < end_line {
                    if diff_output {
                        let spans = render_diff_chunk(&chunk, diff_lang);
                        let mut line_spans =
                            vec![Span::styled("  │  ", Style::default().fg(COLOR_MUTED))];
                        line_spans.extend(spans);
                        out.push(Line::from(line_spans));
                    } else {
                        out.push(Line::from(vec![
                            Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
                            Span::styled("  ", Style::default()),
                            Span::styled(chunk, Style::default().fg(body_color)),
                        ]));
                    }
                }
                current_line += 1;
            }
        }
    } else if entry.kind == EntryKind::FileChange {
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
    } else {
        for body_line in entry.body.split('\n') {
            for chunk in split_at_width(body_line, body_width) {
                if current_line >= start_line && current_line < end_line {
                    // Style the result separator line subtly
                    let is_separator = chunk.trim() == "─── result ───";
                    let chunk_style = if is_separator {
                        Style::default().fg(COLOR_MUTED).add_modifier(Modifier::DIM)
                    } else {
                        Style::default().fg(body_color)
                    };
                    out.push(Line::from(vec![
                        Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
                        Span::styled(chunk, chunk_style),
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

/// Returns true when `body` looks like a unified diff output (to trigger colourised rendering).
///
/// Note: deliberately excludes the `  ·  ` table-separator that tool-result tables use,
/// since that pattern is NOT a diff header and caused false-positive diff rendering.
pub fn is_diff_body(body: &str) -> bool {
    // Require at least one actual diff marker in the first few non-empty lines
    // to distinguish a real unified diff from, e.g., a search-result table.
    for line in body.lines().filter(|l| !l.trim().is_empty()).take(5) {
        if line.starts_with("--- ") || line.starts_with("+++ ") || line.starts_with("@@ ") {
            return true;
        }
    }
    false
}

pub fn is_read_file_body(body: &str) -> bool {
    body.lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.starts_with("📄 "))
        .unwrap_or(false)
}

/// Pick a ratatui colour for a single diff line based on its prefix character.
fn diff_line_color(line: &str) -> Color {
    if line.starts_with("---") || line.starts_with("+++") {
        COLOR_MUTED // file header — muted (must be checked before single +/-)
    } else if line.starts_with('+') {
        Color::Rgb(100, 220, 120) // green — addition
    } else if line.starts_with('-') {
        Color::Rgb(255, 100, 100) // red — removal
    } else if line.starts_with("@@") {
        Color::Rgb(100, 200, 240) // cyan — hunk header
    } else if line.starts_with('▲') {
        COLOR_WARNING // truncation notice
    } else {
        Color::Rgb(190, 190, 190) // context lines — slightly dimmed
    }
}

pub fn diff_lang_for_body(body: &str) -> &'static str {
    diff_path_for_body(body).map(lang_for_path).unwrap_or("")
}

pub fn read_file_lang_for_body(body: &str) -> &'static str {
    read_file_path_for_body(body)
        .map(lang_for_path)
        .unwrap_or("")
}

fn diff_path_for_body(body: &str) -> Option<&str> {
    let mut non_empty_lines = body.lines().filter(|l| !l.trim().is_empty());
    let first = non_empty_lines.next()?;

    if let Some((path, _status)) = first.split_once("  ·  ") {
        let path = path.trim();
        if !path.is_empty() {
            return Some(path);
        }
    }

    if let Some(path) = diff_header_path(first) {
        if path != "/dev/null" {
            return Some(path);
        }
    }

    if first.starts_with("--- ") {
        if let Some(second) = non_empty_lines.next() {
            if let Some(path) = diff_header_path(second) {
                if path != "/dev/null" {
                    return Some(path);
                }
            }
        }
    }

    None
}

fn diff_header_path(line: &str) -> Option<&str> {
    let line = line
        .strip_prefix("--- ")
        .or_else(|| line.strip_prefix("+++ "))?;
    let path = line.split_whitespace().next()?;
    Some(
        path.strip_prefix("a/")
            .or_else(|| path.strip_prefix("b/"))
            .unwrap_or(path),
    )
}

fn read_file_path_for_body(body: &str) -> Option<&str> {
    let first = body.lines().find(|l| !l.trim().is_empty())?;
    if !first.starts_with("📄 ") {
        return None;
    }
    let path = first.trim_start_matches("📄 ").split("  (").next()?.trim();
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

pub fn render_read_file_body_lines(
    body: &str,
    lang: &str,
    width: usize,
) -> Vec<Vec<Span<'static>>> {
    let mut out = Vec::new();
    let mut lines = body.lines();
    let header = lines.next().unwrap_or_default();
    let _blank = lines.next();

    out.push(vec![Span::styled(
        header.to_string(),
        Style::default().fg(COLOR_MUTED),
    )]);
    out.push(Vec::new());

    for body_line in lines {
        let chunks = split_at_width(body_line, width);
        for chunk in chunks {
            if chunk.trim() == "…" || chunk.starts_with("▲ ") {
                out.push(vec![Span::styled(
                    chunk,
                    Style::default().fg(COLOR_WARNING),
                )]);
                continue;
            }

            if lang.is_empty() {
                out.push(vec![Span::styled(
                    chunk,
                    Style::default().fg(Color::Rgb(190, 190, 190)),
                )]);
                continue;
            }

            out.push(highlight_code_line(&chunk, lang));
        }
    }

    out
}

pub fn render_diff_chunk(chunk: &str, lang: &str) -> Vec<Span<'static>> {
    if chunk.starts_with("--- ") || chunk.starts_with("+++ ") {
        return vec![Span::styled(
            chunk.to_string(),
            Style::default().fg(COLOR_MUTED),
        )];
    }
    if chunk.starts_with("@@") {
        return vec![Span::styled(
            chunk.to_string(),
            Style::default().fg(Color::Rgb(100, 200, 240)),
        )];
    }
    if chunk.starts_with('▲') {
        return vec![Span::styled(
            chunk.to_string(),
            Style::default().fg(COLOR_WARNING),
        )];
    }

    // Helper: apply a background colour to every span in a list.
    fn with_bg(spans: Vec<Span<'static>>, bg: Color) -> Vec<Span<'static>> {
        spans
            .into_iter()
            .map(|s| Span::styled(s.content, s.style.bg(bg)))
            .collect()
    }

    if lang.is_empty() {
        let color = diff_line_color(chunk);
        let bg = if !chunk.starts_with("+++") && !chunk.starts_with("---") {
            if chunk.starts_with('+') {
                BG_DIFF_ADD
            } else if chunk.starts_with('-') {
                BG_DIFF_REMOVE
            } else {
                Color::Reset
            }
        } else {
            Color::Reset
        };
        return vec![Span::styled(
            chunk.to_string(),
            Style::default().fg(color).bg(bg),
        )];
    }

    // File headers (--- / +++) — render muted with no background.
    if chunk.starts_with("---") || chunk.starts_with("+++") {
        return vec![Span::styled(
            chunk.to_string(),
            Style::default().fg(COLOR_MUTED),
        )];
    }

    if let Some(rest) = chunk.strip_prefix('+') {
        let mut spans = vec![Span::styled(
            "+".to_string(),
            Style::default()
                .fg(Color::Rgb(100, 220, 120))
                .bg(BG_DIFF_ADD),
        )];
        spans.extend(with_bg(highlight_code_line(rest, lang), BG_DIFF_ADD));
        return spans;
    }

    if let Some(rest) = chunk.strip_prefix('-') {
        let mut spans = vec![Span::styled(
            "-".to_string(),
            Style::default()
                .fg(Color::Rgb(255, 100, 100))
                .bg(BG_DIFF_REMOVE),
        )];
        spans.extend(with_bg(highlight_code_line(rest, lang), BG_DIFF_REMOVE));
        return spans;
    }

    if let Some(rest) = chunk.strip_prefix(' ') {
        return highlight_code_line(rest, lang);
    }

    highlight_code_line(chunk, lang)
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

/// Format a duration in seconds into a compact human-readable string.
/// e.g. "3s", "1m12s", "2h5m"
pub fn format_duration(secs: i64) -> String {
    if secs < 0 {
        return String::new();
    }
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("{m}m")
        } else {
            format!("{m}m{s:02}s")
        }
    } else {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        if m == 0 {
            format!("{h}h")
        } else {
            format!("{h}h{m:02}m")
        }
    }
}

pub fn entry_elapsed_secs(
    timestamp: Option<i64>,
    completed_at: Option<i64>,
    _pending: bool,
    _now: i64,
) -> Option<i64> {
    // Only return a duration once the entry has completed — the per-entry
    // ticker during pending was redundant with the in-transcript Working
    // entry, which shows the whole turn's elapsed time.
    let start = timestamp?;
    let end = completed_at?;
    Some((end - start).max(0))
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

/// Returns a compact relative-time string for a Unix-seconds timestamp,
/// e.g. "just now", "45s ago", "12m ago", "3h ago", "2d ago", "4w ago".
pub fn relative_time_ago(ts: i64) -> String {
    let now = chrono::Utc::now().timestamp();
    let secs = (now - ts).max(0) as u64;
    match secs {
        0..=59 => "just now".into(),
        60..=3599 => format!("{}m ago", secs / 60),
        3600..=86399 => format!("{}h ago", secs / 3600),
        86400..=604799 => format!("{}d ago", secs / 86400),
        604800..=2591999 => format!("{}w ago", secs / 604800),
        _ => format!("{}mo ago", secs / 2592000),
    }
}

pub fn pretty_json_or_text(primary: Option<&Value>, secondary: Option<&Value>) -> String {
    if let Some(text) = primary.and_then(|v| v.as_str()) {
        return summarise_long_output(text, TOOL_OUTPUT_MAX_LINES, TOOL_OUTPUT_MAX_CHARS);
    }
    if let Some(value) = primary {
        if !value.is_null() {
            let s = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
            return summarise_long_output(&s, TOOL_OUTPUT_MAX_LINES, TOOL_OUTPUT_MAX_CHARS);
        }
    }
    if let Some(text) = secondary.and_then(|v| v.as_str()) {
        return summarise_long_output(text, TOOL_OUTPUT_MAX_LINES, TOOL_OUTPUT_MAX_CHARS);
    }
    if let Some(value) = secondary {
        if !value.is_null() {
            let s = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
            return summarise_long_output(&s, TOOL_OUTPUT_MAX_LINES, TOOL_OUTPUT_MAX_CHARS);
        }
    }
    String::new()
}

fn format_tool_call_title(tool_name: &str, arguments: &Value) -> String {
    match tool_name {
        "spawn_agent" => spawn_agent_label(arguments)
            .map(|label| format!("sub-agent: {label}"))
            .unwrap_or_else(|| "sub-agent".to_string()),
        // Read-only / collapsible tools: embed the key argument in the title
        // so collapsed entries show context (e.g. "read_file  main.rs").
        "read_file" | "write_file" | "patch_file" | "create_directory" | "remove_path" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .map(basename)
                .unwrap_or_default();
            if path.is_empty() {
                tool_name.to_string()
            } else {
                format!("{tool_name}  {path}")
            }
        }
        "list_dir" => {
            let path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
            let short = basename(path);
            format!("{tool_name}  {short}")
        }
        "grep_search" => {
            let pattern = arguments
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("?");
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .map(basename)
                .unwrap_or_else(|| ".".into());
            format!("{tool_name}  /{pattern}/  in {path}")
        }
        "web_fetch" => {
            let url = arguments.get("url").and_then(Value::as_str).unwrap_or("?");
            // Show just the domain for brevity.
            let short = url
                .strip_prefix("https://")
                .or_else(|| url.strip_prefix("http://"))
                .unwrap_or(url)
                .split('/')
                .next()
                .unwrap_or(url);
            format!("{tool_name}  {short}")
        }
        "copy_path" => {
            let source = arguments
                .get("source")
                .and_then(Value::as_str)
                .map(basename);
            let target = arguments
                .get("target")
                .and_then(Value::as_str)
                .map(basename);
            match (source, target) {
                (Some(s), Some(t)) => format!("{tool_name}  {s} → {t}"),
                (Some(s), None) => format!("{tool_name}  {s}"),
                _ => tool_name.to_string(),
            }
        }

        _ => tool_name.to_string(),
    }
}

fn format_tool_call(tool_name: &str, arguments: &Value) -> String {
    match tool_name {
        // Keep spawn_agent call bodies terse; the title already carries the task label.
        "spawn_agent" => String::new(),
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
        "bash_exec" => {
            let mut lines = Vec::new();
            if let Some(cmd) = arguments.get("command").and_then(Value::as_str) {
                lines.push(cmd.to_string());
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
        "list_dir" => {
            let path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
            let depth = arguments.get("depth").and_then(Value::as_u64).unwrap_or(1);
            format!("{path}  (depth {depth})")
        }
        "grep_search" => {
            let pattern = arguments
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("?");
            let path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
            format!("/{pattern}/  in {path}")
        }
        "patch_file" => {
            // Produce a unified-diff-like body so is_diff_body() triggers
            // and render_diff_chunk() applies syntax highlighting.
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("file");
            let start_line = arguments
                .get("start_line")
                .and_then(Value::as_u64)
                .unwrap_or(1);
            let end_line = arguments
                .get("end_line")
                .and_then(Value::as_u64)
                .unwrap_or(1);
            let content = arguments
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("");
            let content_lines: Vec<&str> = content.lines().collect();
            let new_count = content_lines.len().max(1);
            let mut lines = Vec::new();
            lines.push(format!("{path}  ·  editing lines {start_line}–{end_line}"));
            lines.push(String::new());
            lines.push(format!("--- a/{path}"));
            lines.push(format!("+++ b/{path}"));
            lines.push(format!(
                "@@ -{start_line},{} +{start_line},{new_count} @@",
                end_line.saturating_sub(start_line) + 1
            ));
            for line in &content_lines {
                lines.push(format!("+{line}"));
            }
            lines.join("\n")
        }
        "write_file" => {
            // Produce a 📄-prefixed body so is_read_file_body() triggers
            // and render_read_file_body_lines() applies syntax highlighting.
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("file");
            let content = arguments.get("content").and_then(Value::as_str);
            let line_count = content.map(|c| c.lines().count()).unwrap_or(0).max(1);
            let mut lines = Vec::new();
            lines.push(format!("📄 {path}  ({line_count} lines · writing)"));
            lines.push(String::new());
            if let Some(content) = content {
                lines.push(summarise_long_output(
                    content,
                    TOOL_OUTPUT_MAX_LINES,
                    TOOL_OUTPUT_MAX_CHARS,
                ));
            }
            lines.join("\n")
        }
        "read_file" | "create_directory" | "remove_path" => {
            let mut lines = Vec::new();
            if let Some(path) = arguments.get("path").and_then(Value::as_str) {
                lines.push(path.to_string());
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

fn spawn_agent_label(arguments: &Value) -> Option<String> {
    let prompt = arguments.get("prompt").and_then(Value::as_str)?.trim();
    if prompt.is_empty() {
        return None;
    }
    let first_line = prompt.lines().find(|line| !line.trim().is_empty())?.trim();
    Some(truncate_chars(first_line, 96))
}

pub fn format_tool_result(output: Option<&Value>, error_message: Option<&Value>) -> String {
    let Some(output) = output else {
        return pretty_json_or_text(None, error_message);
    };

    // write_file: prefer the diff view if present
    if output.get("diff").is_some() {
        return format_write_file_result(output);
    }

    // shell_exec / bash_exec results
    if let Some(formatted) = format_shell_result(output) {
        return formatted;
    }

    // list_dir results: {root, entries, count, truncated}
    if let Some(formatted) = format_list_dir_result(output) {
        return formatted;
    }

    // read_file results: {path, content}
    if let Some(formatted) = format_read_file_result(output) {
        return formatted;
    }

    // grep_search results: {matches, source}
    if let Some(formatted) = format_grep_result(output) {
        return formatted;
    }

    // Simple ok results: {ok, path} or {ok, source, target}
    if let Some(formatted) = format_simple_ok_result(output) {
        return formatted;
    }

    // spawn_agent deterministic-task redirect: no child thread is created.
    if let Some(formatted) = format_spawn_agent_not_spawned_result(output) {
        return formatted;
    }

    // spawn_agent results: {thread_id, turn_id, result, error?}
    if let Some(formatted) = format_spawn_agent_result(output, error_message) {
        return formatted;
    }

    pretty_json_or_text(Some(output), error_message)
}

fn format_spawn_agent_not_spawned_result(output: &Value) -> Option<String> {
    if output.get("status").and_then(Value::as_str) != Some("not_spawned") {
        return None;
    }

    let reason = output
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("This task should use direct tools instead of a sub-agent.");
    let mut lines = vec![
        "status: not spawned".to_string(),
        format!("reason: {reason}"),
    ];

    if let Some(tool) = output.get("suggested_tool").and_then(Value::as_str) {
        lines.push(String::new());
        lines.push(format!("suggested tool: {tool}"));
    }
    if let Some(args) = output.get("suggested_arguments") {
        lines.push(format!(
            "arguments: {}",
            serde_json::to_string(args).unwrap_or_else(|_| args.to_string())
        ));
    }
    if let Some(next) = output.get("next_step").and_then(Value::as_str) {
        lines.push(format!("next: {next}"));
    }

    Some(lines.join("\n"))
}

fn format_spawn_agent_result(output: &Value, error_message: Option<&Value>) -> Option<String> {
    let thread_id = output
        .get("thread_id")
        .or_else(|| output.get("threadId"))
        .and_then(Value::as_str)?;
    let turn_id = output
        .get("turn_id")
        .or_else(|| output.get("turnId"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let error = output.get(KEY_ERROR).and_then(Value::as_str);
    let status = match error {
        Some(STATUS_TIMEOUT) => STATUS_TIMED_OUT,
        Some(STATUS_INTERRUPTED) => STATUS_INTERRUPTED,
        Some(_) => STATUS_FAILED,
        None => STATUS_COMPLETED,
    };

    let mut lines = vec![
        format!("status: {status}"),
        format!(
            "thread: {}  turn: {}",
            short_thread_id(thread_id),
            short_turn_id(turn_id),
        ),
    ];

    if let Some(message) = error_message.and_then(Value::as_str) {
        if !message.trim().is_empty() {
            lines.push(format!("error: {}", message.trim()));
        }
    }

    let result = output
        .get("result")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if !result.is_empty() {
        lines.push(String::new());
        if error == Some(STATUS_TIMEOUT) {
            lines.push("partial output before timeout:".to_string());
        }
        lines.push(summarise_long_output(
            result,
            TOOL_OUTPUT_MAX_LINES,
            TOOL_OUTPUT_MAX_CHARS,
        ));
    }

    Some(lines.join("\n"))
}

/// Format the result of a `write_file` call that carries a unified diff.
fn format_write_file_result(output: &Value) -> String {
    let path = output.get("path").and_then(Value::as_str).unwrap_or("?");
    let is_new = output
        .get("is_new_file")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let added = output
        .get("lines_added")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let removed = output
        .get("lines_removed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let diff = output.get("diff").and_then(Value::as_str).unwrap_or("");

    let status = if is_new {
        format!("new file  +{added}")
    } else {
        format!("+{added}  -{removed}")
    };

    let mut lines = vec![format!("{path}  ·  {status}")];
    if !diff.is_empty() {
        lines.push(String::new());
        // Truncate diff to avoid flooding the viewport.
        let diff_lines: Vec<&str> = diff.lines().collect();
        let keep = diff_lines.len().min(TOOL_OUTPUT_MAX_LINES);
        if keep < diff_lines.len() {
            lines.extend(diff_lines[..keep].iter().map(|l| l.to_string()));
            lines.push(format!("▲ {} lines hidden", diff_lines.len() - keep));
        } else {
            lines.extend(diff_lines.iter().map(|l| l.to_string()));
        }
    }
    lines.join("\n")
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
        lines.push(summarise_long_output(
            stdout,
            TOOL_OUTPUT_MAX_LINES,
            TOOL_OUTPUT_MAX_CHARS,
        ));
    }
    if !stderr.is_empty() {
        lines.push(String::new());
        lines.push("stderr:".to_string());
        lines.push(summarise_long_output(
            stderr,
            TOOL_OUTPUT_MAX_LINES,
            TOOL_OUTPUT_MAX_CHARS,
        ));
    }
    if output
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        lines.push(String::new());
        lines.push("[backend truncated output]".to_string());
    }

    Some(lines.join("\n"))
}

/// Format a `list_dir` result as a compact visual tree.
fn format_list_dir_result(output: &Value) -> Option<String> {
    let entries = output.get("entries")?.as_array()?;
    let root = output.get("root").and_then(Value::as_str).unwrap_or(".");
    let truncated = output
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut lines: Vec<String> = Vec::new();
    let count = entries.len();

    // Header
    lines.push(format!("📁 {root}  ({count} entries)"));

    // Cap display to avoid flooding the viewport
    let max_show = TOOL_OUTPUT_MAX_LINES.saturating_sub(2);
    let shown = entries.len().min(max_show);
    let hidden = entries.len().saturating_sub(shown);

    for entry in &entries[..shown] {
        let path = entry.get("path").and_then(Value::as_str).unwrap_or("");
        let is_dir = entry
            .get("type")
            .and_then(Value::as_str)
            .map(|t| t == "dir")
            .unwrap_or(false);
        let size = entry.get("size_bytes").and_then(Value::as_u64);

        // Indent based on path depth
        let depth = path
            .chars()
            .filter(|&c| c == '/' || c == std::path::MAIN_SEPARATOR)
            .count();
        let indent = "  ".repeat(depth);
        let icon = if is_dir { "▸" } else { " " };
        let name = path.rsplit('/').next().unwrap_or(path);
        let size_str = if is_dir {
            String::new()
        } else if let Some(b) = size {
            format!("  {}", human_bytes(b))
        } else {
            String::new()
        };
        lines.push(format!("{indent}{icon} {name}{size_str}"));
    }

    if hidden > 0 {
        lines.push(format!("  … {hidden} more entries"));
    } else if truncated {
        lines.push("  … (listing truncated)".to_string());
    }

    Some(lines.join("\n"))
}

/// Format a `read_file` result as a compact header + content preview.
fn format_read_file_result(output: &Value) -> Option<String> {
    // Must have both "path" and "content" keys — unique to read_file
    let path = output.get("path")?.as_str()?;
    let content = output.get("content")?.as_str()?;
    // Exclude write_file results which also carry a "path" but have "diff"
    if output.get("diff").is_some() || output.get("ok").is_some() {
        return None;
    }

    let line_count = content.lines().count();
    let byte_count = content.len();
    let header = format!(
        "📄 {path}  ({line_count} lines · {})",
        human_bytes(byte_count as u64)
    );

    let mut lines = vec![header, String::new()];
    lines.push(summarise_long_output(
        content,
        TOOL_OUTPUT_MAX_LINES,
        TOOL_OUTPUT_MAX_CHARS,
    ));
    Some(lines.join("\n"))
}

/// Format a `grep_search` result as a match count header + list.
fn format_grep_result(output: &Value) -> Option<String> {
    let matches = output.get("matches")?.as_array()?;
    let source = output.get("source").and_then(Value::as_str).unwrap_or("?");
    let count = matches.len();

    if count == 0 {
        return Some(format!("🔍 No matches  (via {source})"));
    }

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("🔍 {count} matches  (via {source})"));
    lines.push(String::new());

    let max_show = TOOL_OUTPUT_MAX_LINES.saturating_sub(2);
    let shown = matches.len().min(max_show);
    let hidden = matches.len().saturating_sub(shown);

    for m in &matches[..shown] {
        let formatted = if let (Some(file), Some(content)) = (
            m.get("file").and_then(Value::as_str),
            m.get("content").and_then(Value::as_str),
        ) {
            // Structured match: { file, line, content }
            let file = file.trim_start_matches("./");
            let content = content.trim();
            if let Some(line_num) = m.get("line").and_then(Value::as_u64) {
                format!("{file}:{line_num}  {content}")
            } else {
                format!("{file}  {content}")
            }
        } else if let Some(s) = m.as_str() {
            // Legacy flat string fallback
            s.to_string()
        } else {
            continue; // unknown shape — skip
        };

        // Trim very long lines
        let trimmed = if formatted.chars().count() > 120 {
            let s: String = formatted.chars().take(120).collect();
            format!("{s}…")
        } else {
            formatted
        };
        lines.push(trimmed);
    }
    if hidden > 0 {
        lines.push(format!("  … {hidden} more matches"));
    }

    Some(lines.join("\n"))
}

/// Format simple `{ok: true, path: ...}` style results (create_directory, remove_path, copy_path).
fn format_simple_ok_result(output: &Value) -> Option<String> {
    // Only handle objects where ok == true and there's no content/entries/matches
    let obj = output.as_object()?;
    if !obj.get("ok")?.as_bool().unwrap_or(false) {
        return None;
    }
    // Bail if it looks like a richer result already handled above
    if obj.contains_key("entries")
        || obj.contains_key("content")
        || obj.contains_key("matches")
        || obj.contains_key("diff")
        || obj.contains_key("stdout")
    {
        return None;
    }

    let path = obj.get("path").and_then(Value::as_str);
    let source = obj.get("source").and_then(Value::as_str);
    let target = obj.get("target").and_then(Value::as_str);

    let msg = match (path, source, target) {
        (Some(p), None, None) => format!("✓  {p}"),
        (None, Some(src), Some(tgt)) => format!("✓  {src}  →  {tgt}"),
        _ => return None,
    };
    Some(msg)
}

/// Human-readable byte size: "1.2 KB", "34 B", "2.1 MB", etc.
fn human_bytes(b: u64) -> String {
    match b {
        0 => "0 B".into(),
        1..=1023 => format!("{b} B"),
        1024..=1_048_575 => format!("{:.1} KB", b as f64 / 1024.0),
        1_048_576..=1_073_741_823 => format!("{:.1} MB", b as f64 / 1_048_576.0),
        _ => format!("{:.1} GB", b as f64 / 1_073_741_824.0),
    }
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

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

// ─── Output summarisation ─────────────────────────────────────────────────────

/// Maximum lines shown for a single tool output block in the TUI.
const TOOL_OUTPUT_MAX_LINES: usize = 20;
/// Maximum total characters shown before summarising.
const TOOL_OUTPUT_MAX_CHARS: usize = 1_500;

/// Truncate `text` if it exceeds `max_lines` lines **or** `max_chars` characters.
///
/// When truncation is needed the function returns the *tail* of the output
/// (most useful for command output) preceded by a single "▲ N lines hidden"
/// notice.  If the text fits within both limits it is returned unchanged.
pub fn summarise_long_output(text: &str, max_lines: usize, max_chars: usize) -> String {
    let line_count = text.lines().count();
    let char_count = text.chars().count();

    let needs_truncation = line_count > max_lines || char_count > max_chars;
    if !needs_truncation {
        return text.to_string();
    }

    // Collect lines and keep only the tail window.
    let all_lines: Vec<&str> = text.lines().collect();
    let keep = max_lines.min(all_lines.len());
    let skipped = all_lines.len().saturating_sub(keep);

    // Further trim the kept tail if it still exceeds max_chars.
    let tail_lines = &all_lines[skipped..];
    let mut tail = tail_lines.join("\n");
    if tail.chars().count() > max_chars {
        // Trim from the start of the tail string to fit.
        let chars: Vec<char> = tail.chars().collect();
        let start = chars.len().saturating_sub(max_chars);
        tail = chars[start..].iter().collect();
        // Re-align to a line boundary so we don't split mid-line.
        if let Some(nl) = tail.find('\n') {
            tail = tail[nl + 1..].to_string();
        }
    }

    let notice = format!(
        "▲ {} lines hidden",
        line_count.saturating_sub(tail.lines().count())
    );
    format!("{notice}\n{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn spawn_agent_call_formats_as_sub_agent_task() {
        let args = json!({
            "prompt": "List the contents of `src` and summarize it.\nUse concise output.",
            "timeout_secs": 60
        });

        assert_eq!(
            format_tool_call_title("spawn_agent", &args),
            "sub-agent: List the contents of `src` and summarize it."
        );
        let body = format_tool_call("spawn_agent", &args);
        assert!(
            body.is_empty(),
            "spawn_agent body should stay terse: {body}"
        );
    }

    #[test]
    fn spawn_agent_result_formats_without_raw_json() {
        let output = json!({
            "thread_id": "thread_74a00930da5c485bb792e91bd85622c1",
            "turn_id": "turn_8e6444c1fc7c4ac18e39353731b9dbe3",
            "result": "The output was truncated at 300 entries.",
            "error": "timeout"
        });
        let error = Value::from("sub-agent timed out after 120s");

        let body = format_tool_result(Some(&output), Some(&error));
        assert!(body.contains("status: timed out"));
        assert!(body.contains("thread: 74a00930"));
        assert!(body.contains("turn: 8e6444c1"));
        assert!(body.contains("The output was truncated"));
        assert!(
            !body.contains("\"thread_id\""),
            "body should not be raw JSON: {body}"
        );
    }

    #[test]
    fn spawn_agent_not_spawned_formats_as_redirect() {
        let output = json!({
            "status": "not_spawned",
            "error": "deterministic_directory_inventory",
            "reason": "Directory inventory is deterministic and should use direct file tools, not a model sub-agent.",
            "suggested_tool": "list_dir",
            "suggested_arguments": { "path": "bin", "depth": 3, "max_entries": 300 },
            "next_step": "Call list_dir directly."
        });

        let body = format_tool_result(Some(&output), None);
        assert!(body.contains("status: not spawned"));
        assert!(body.contains("suggested tool: list_dir"));
        assert!(body.contains("\"path\":\"bin\""));
        assert!(
            !body.contains("thread:"),
            "no child thread should be shown: {body}"
        );
    }

    #[test]
    fn entry_elapsed_runs_for_pending_and_freezes_when_completed() {
        // Pending entry with no completed_at → None (elapsed shown via Working timer instead).
        assert_eq!(entry_elapsed_secs(Some(10), None, true, 14), None);
        // Completed entry → frozen duration from start to completed_at.
        assert_eq!(entry_elapsed_secs(Some(10), Some(12), false, 99), Some(2));
        // No start timestamp → None.
        assert_eq!(entry_elapsed_secs(Some(10), None, false, 99), None);
    }

    #[test]
    fn transcript_header_shows_entry_elapsed_not_previous_gap() {
        let entries = vec![
            TranscriptEntry {
                item_id: "u1".into(),
                turn_id: Some("turn_1".into()),
                tool_call_id: None,
                kind: EntryKind::User,
                title: "You".into(),
                body: "go".into(),
                timestamp: Some(10),
                completed_at: None,
                pending: false,
                collapsed: false,
            },
            TranscriptEntry {
                item_id: "a1".into(),
                turn_id: Some("turn_1".into()),
                tool_call_id: None,
                kind: EntryKind::Assistant,
                title: "Codezilla".into(),
                body: "done".into(),
                timestamp: Some(15),
                completed_at: Some(18),
                pending: false,
                collapsed: false,
            },
        ];

        let (lines, _) = transcript_window_lines(&entries, 0, 120, None, 0, 10);
        let assistant_header = lines
            .iter()
            .map(|line| {
                let mut text = String::new();
                for span in &line.spans {
                    text.push_str(span.content.as_ref());
                }
                text
            })
            .find(|line| line.contains("Codezilla"))
            .expect("assistant header should render");

        assert!(assistant_header.contains(" · 3s"));
        assert!(assistant_header.contains(" · +5s"));
        assert!(!assistant_header.contains(" · 5s · +5s"));
    }
}
