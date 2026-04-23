use serde_json::Value;
use std::path::Path;

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
};

use crate::system::domain::{ConversationItem, ItemKind, ThreadMetadata};
use crate::system::domain::PendingApproval;

// ─── Colour palette ───────────────────────────────────────────────────────────

pub const COLOR_BORDER: Color = Color::Rgb(58, 70, 88);
pub const COLOR_ACCENT: Color = Color::Rgb(86, 196, 181);
pub const COLOR_MUTED: Color = Color::Rgb(140, 149, 165);
pub const COLOR_USER: Color = Color::Rgb(242, 186, 92);
pub const COLOR_ASSISTANT: Color = Color::Rgb(163, 218, 255);
pub const COLOR_TOOL: Color = Color::Rgb(245, 132, 91);
pub const COLOR_STATUS: Color = Color::Rgb(160, 208, 120);
pub const COLOR_ERROR: Color = Color::Rgb(255, 112, 112);
pub const COLOR_APPROVAL: Color = Color::Rgb(255, 210, 94);
pub const COLOR_SUMMARY: Color = Color::Rgb(190, 164, 255);

// ─── Layout constants ─────────────────────────────────────────────────────────

pub const THREAD_LIMIT: i32 = 40;
pub const COMPOSER_MIN_HEIGHT: u16 = 5;
pub const COMPOSER_MAX_HEIGHT: u16 = 9;

// ─── Enums ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPane {
    Threads,
    Transcript,
    Composer,
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

    pub fn line_count(&self) -> usize {
        self.chars.iter().filter(|&&ch| ch == '\n').count() + 1
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

    pub fn move_up(&mut self) {
        let start = self.current_line_start();
        if start == 0 {
            return;
        }
        let previous_end = start - 1;
        let mut previous_start = previous_end;
        while previous_start > 0 && self.chars[previous_start - 1] != '\n' {
            previous_start -= 1;
        }
        let column = self.cursor.saturating_sub(start);
        let previous_len = previous_end.saturating_sub(previous_start);
        self.cursor = previous_start + column.min(previous_len);
    }

    pub fn move_down(&mut self) {
        let start = self.current_line_start();
        let end = self.line_end_from(start);
        if end >= self.chars.len() {
            return;
        }
        let next_start = end + 1;
        let next_end = self.line_end_from(next_start);
        let column = self.cursor.saturating_sub(start);
        let next_len = next_end.saturating_sub(next_start);
        self.cursor = next_start + column.min(next_len);
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

    fn line_end_from(&self, start: usize) -> usize {
        let mut end = start;
        while end < self.chars.len() && self.chars[end] != '\n' {
            end += 1;
        }
        end
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
                "Tool · {}",
                item.payload
                    .get("toolName")
                    .and_then(|v: &serde_json::Value| v.as_str())
                    .unwrap_or("call")
            ),
            body: serde_json::to_string_pretty(
                item.payload
                    .get("arguments")
                    .unwrap_or(&serde_json::Value::Null),
            )
            .unwrap_or_else(|_| item.payload.to_string()),
            timestamp: Some(item.created_at),
            pending: false,
        },
        ItemKind::ToolResult => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::ToolResult,
            title: "Tool result".into(),
            body: pretty_json_or_text(item.payload.get("output"), item.payload.get("errorMessage")),
            timestamp: Some(item.created_at),
            pending: false,
        },
        ItemKind::ReasoningSummary => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::Summary,
            title: "Summary".into(),
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
            title: "Reasoning".into(),
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
            title: "Attachment".into(),
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
            title: "Error".into(),
            body: pretty_json_or_text(Some(&item.payload), None),
            timestamp: Some(item.created_at),
            pending: false,
        },
        _ => TranscriptEntry {
            item_id: item.item_id.clone(),
            kind: EntryKind::Status,
            title: format!("{:?}", item.kind),
            body: pretty_json_or_text(Some(&item.payload), None),
            timestamp: Some(item.created_at),
            pending: false,
        },
    }
}

pub fn transcript_text(entries: &[TranscriptEntry]) -> (Text<'static>, usize) {
    if entries.is_empty() {
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "No messages yet. Start typing in the composer below.",
                Style::default().fg(COLOR_MUTED),
            )),
        ];
        return (Text::from(lines), 2);
    }

    let mut lines = Vec::new();
    for entry in entries {
        let (label_color, body_color) = match entry.kind {
            EntryKind::User => (COLOR_USER, Color::White),
            EntryKind::Assistant => (COLOR_ASSISTANT, Color::White),
            EntryKind::ToolCall => (COLOR_TOOL, Color::Rgb(244, 220, 205)),
            EntryKind::ToolResult => (Color::Rgb(255, 165, 130), Color::Rgb(248, 228, 214)),
            EntryKind::Summary => (COLOR_SUMMARY, Color::Rgb(232, 221, 255)),
            EntryKind::Status => (COLOR_STATUS, Color::Rgb(210, 230, 190)),
            EntryKind::Error => (COLOR_ERROR, Color::Rgb(255, 215, 215)),
            EntryKind::Attachment => (COLOR_MUTED, Color::Rgb(205, 212, 224)),
            EntryKind::Reasoning => (Color::Rgb(155, 136, 255), Color::Rgb(222, 217, 255)),
        };

        let mut header = vec![Span::styled(
            entry.title.clone(),
            Style::default().fg(label_color).add_modifier(Modifier::BOLD),
        )];
        if let Some(ts) = entry.timestamp {
            header.push(Span::raw("  "));
            header.push(Span::styled(
                format_timestamp(ts),
                Style::default().fg(COLOR_MUTED),
            ));
        }
        if entry.pending {
            header.push(Span::raw("  "));
            header.push(Span::styled("streaming", Style::default().fg(COLOR_ACCENT)));
        }
        lines.push(Line::from(header));

        for body_line in entry.body.split('\n') {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(body_line.to_string(), Style::default().fg(body_color)),
            ]));
        }
        lines.push(Line::from(""));
    }

    let count = lines.len();
    (Text::from(lines), count)
}

pub fn composer_height(composer: &ComposerState) -> u16 {
    let lines = composer.line_count() as u16 + 2;
    lines.clamp(COMPOSER_MIN_HEIGHT, COMPOSER_MAX_HEIGHT)
}

pub fn panel_block<'a>(title: &'a str, focused: bool) -> ratatui::widgets::Block<'a> {
    use ratatui::widgets::{Block, Borders};
    let border_style = if focused {
        Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(COLOR_BORDER)
    };
    Block::default()
        .title(format!(" {title} "))
        .borders(Borders::ALL)
        .border_style(border_style)
}

pub fn centered_rect(percent_x: u16, percent_y: u16, area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    use ratatui::layout::{Constraint, Direction, Layout};
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
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
        "awaiting approval"
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

pub fn pretty_json_or_text(
    primary: Option<&Value>,
    secondary: Option<&Value>,
) -> String {
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
