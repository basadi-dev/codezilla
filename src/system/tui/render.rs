use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::app::InteractiveApp;
use super::types::{
    basename, composer_height, current_state_label, spinner_frame, split_at_width, thread_label,
    truncate_lines, COLOR_ACCENT, COLOR_APPROVAL, COLOR_BORDER, COLOR_ERROR, COLOR_MUTED,
    COLOR_PROMPT, COLOR_USER,
};

// ─── Spinner tick (bumped each frame by the caller side; we read it from app) ─

/// Draw the complete TUI frame using a stream-style layout:
///   ┌────────────────────────────────────────┐
///   │  header (1 line, no box)               │
///   │  separator (1 line)                    │
///   │  transcript (scrolling stream)         │
///   │  composer input                        │
///   │  status bar (1 line)                   │
///   └────────────────────────────────────────┘
pub fn draw(app: &mut InteractiveApp, frame: &mut Frame) {
    let ch = composer_height(&app.composer, frame.area().width);
    let ac_h = if app.autocomplete_suggestions.is_empty() {
        0
    } else {
        (app.autocomplete_suggestions.len() as u16).min(8)
    };

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),       // header
            Constraint::Length(1),       // separator
            Constraint::Min(4),          // transcript
            Constraint::Length(ac_h),    // autocomplete list (0 when hidden)
            Constraint::Length(ch),      // composer
            Constraint::Length(1),       // status bar
        ])
        .split(frame.area());

    render_header(app, frame, outer[0]);
    render_separator(frame, outer[1]);
    render_transcript(app, frame, outer[2]);
    if ac_h > 0 {
        render_autocomplete(app, frame, outer[3]);
    }
    render_composer(app, frame, outer[4]);
    render_status_bar(app, frame, outer[5]);
}

// ─── Header ───────────────────────────────────────────────────────────────────

fn render_header(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    let meta = app.current_thread_meta.as_ref();
    let thread = meta.map(thread_label).unwrap_or_else(|| "no thread".into());
    let cwd = meta
        .and_then(|t| t.cwd.as_ref())
        .map(|c| basename(c))
        .unwrap_or_else(|| "~".into());
    let (model, reasoning) = {
        let ms = app.effective_model_settings();
        let model = format!("{}/{}", ms.provider_id, ms.model_id);
        let reasoning = ms.reasoning_effort.unwrap_or_default();
        (model, reasoning)
    };
    let approval_mode = app.approval_mode_label();
    let state = current_state_label(app.active_turn_id.is_some(), app.pending_approval.is_some());

    let state_color = if app.error_message.is_some() {
        COLOR_ERROR
    } else if app.active_turn_id.is_some() {
        COLOR_ACCENT
    } else if app.pending_approval.is_some() {
        COLOR_APPROVAL
    } else {
        COLOR_MUTED
    };

    // State indicator: spinning dot or static
    let state_sigil = if app.active_turn_id.is_some() {
        spinner_frame(app.spinner_tick)
    } else {
        "●"
    };

    let line = Line::from(vec![
        Span::styled(
            "  codezilla",
            Style::default()
                .fg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ·  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(cwd, Style::default().fg(Color::White)),
        Span::styled("  /  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(thread, Style::default().fg(COLOR_USER)),
        Span::styled("  ·  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(model, Style::default().fg(COLOR_MUTED)),
        Span::styled("  ·  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            if reasoning.is_empty() {
                "reasoning:off".into()
            } else {
                format!("reasoning:{reasoning}")
            },
            Style::default().fg(COLOR_MUTED),
        ),
        Span::styled("  ·  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            format!("approve:{approval_mode}"),
            Style::default().fg(COLOR_MUTED),
        ),
        Span::styled("  ", Style::default()),
        Span::styled(
            state_sigil.to_string(),
            Style::default()
                .fg(state_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(state.to_string(), Style::default().fg(state_color)),
    ]);

    frame.render_widget(Paragraph::new(line), area);
}

// ─── Separator ────────────────────────────────────────────────────────────────

fn render_separator(frame: &mut Frame, area: Rect) {
    // A single dim horizontal rule using box-drawing character repeated
    let width = area.width as usize;
    let rule = "─".repeat(width);
    frame.render_widget(
        Paragraph::new(Span::styled(rule, Style::default().fg(COLOR_BORDER))),
        area,
    );
}

// ─── Transcript ───────────────────────────────────────────────────────────────

fn render_transcript(app: &mut InteractiveApp, frame: &mut Frame, area: Rect) {
    let transcript_area = if app.pending_approval.is_some() && area.height >= 10 {
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(4), Constraint::Length(8)])
            .split(area);
        render_approval_panel(app, frame, sections[1]);
        sections[0]
    } else {
        area
    };

    let (content_area, indicator_area) = if transcript_area.width >= 16 {
        let transcript_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(8), Constraint::Length(5)])
            .split(transcript_area);
        (transcript_layout[0], Some(transcript_layout[1]))
    } else {
        (transcript_area, None)
    };

    // Store the area so drag coordinate mapping is always current.
    app.transcript_area = content_area;

    let viewport_height = content_area.height as usize;
    let selection = app.drag_selection_lines();
    let total_lines = app.transcript_total_lines(content_area.width);
    let max_scroll = (total_lines.saturating_sub(viewport_height)) as u16;

    if app.auto_scroll || app.transcript_scroll >= max_scroll {
        // At the bottom (or already in follow mode) — clamp and re-engage auto-scroll
        // so that manually scrolling down to the last line re-attaches follow mode.
        app.transcript_scroll = max_scroll;
        app.auto_scroll = true;
    }

    // Pass area.width so body lines are hard-wrapped here rather than by ratatui.
    // This guarantees 1 Line in the vec == 1 terminal row, making drag coords exact.
    let overscan = 8usize;
    let start_line = app.transcript_scroll as usize;
    let window_start = start_line.saturating_sub(overscan);
    let window_height = viewport_height.saturating_add(overscan * 2);
    let (lines, _) =
        app.transcript_window_lines(content_area.width, window_start, window_height, selection);

    // No Wrap — lines are pre-wrapped above so ratatui doesn't need to.
    let paragraph =
        Paragraph::new(Text::from(lines)).scroll(((start_line - window_start) as u16, 0));
    frame.render_widget(paragraph, content_area);

    if let Some(indicator_area) = indicator_area {
        render_scroll_indicator(
            frame,
            indicator_area,
            start_line,
            viewport_height,
            total_lines,
        );
    }
}

fn render_scroll_indicator(
    frame: &mut Frame,
    area: Rect,
    start_line: usize,
    viewport_height: usize,
    total_lines: usize,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let max_scroll = total_lines.saturating_sub(viewport_height);
    let percent = if max_scroll == 0 {
        100
    } else {
        ((start_line * 100) / max_scroll).min(100)
    };

    let bar_height = area.height.saturating_sub(1) as usize;
    let thumb_height = if total_lines == 0 {
        bar_height.max(1)
    } else {
        ((viewport_height * bar_height) / total_lines).clamp(1, bar_height.max(1))
    };
    let thumb_offset = if max_scroll == 0 || bar_height <= thumb_height {
        0
    } else {
        (start_line * (bar_height - thumb_height)) / max_scroll
    };

    let mut lines = Vec::with_capacity(area.height as usize);
    lines.push(Line::from(Span::styled(
        format!("{percent:>3}%"),
        Style::default().fg(COLOR_MUTED),
    )));

    for idx in 0..bar_height {
        let is_thumb = idx >= thumb_offset && idx < thumb_offset + thumb_height;
        lines.push(Line::from(Span::styled(
            if is_thumb { "  █" } else { "  │" },
            Style::default().fg(if is_thumb { COLOR_ACCENT } else { COLOR_BORDER }),
        )));
    }

    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}

// ─── Composer ─────────────────────────────────────────────────────────────────

fn render_composer(app: &mut InteractiveApp, frame: &mut Frame, area: Rect) {
    // Top edge: a slim separator
    let comp_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // separator
            Constraint::Length(1), // top margin
            Constraint::Min(1),    // input
            Constraint::Length(1), // bottom margin
        ])
        .split(area);

    // separator
    let width = area.width as usize;
    let rule = "─".repeat(width);
    frame.render_widget(
        Paragraph::new(Span::styled(rule, Style::default().fg(COLOR_BORDER))),
        comp_layout[0],
    );

    frame.render_widget(Paragraph::new(" "), comp_layout[1]);
    frame.render_widget(Paragraph::new(" "), comp_layout[3]);

    let input_area = comp_layout[2];
    app.composer_area = input_area;

    // All rows share the same prefix width (5 chars: "  ❯  " or "     ").
    // This guarantees first_width == continuation_width so the text grid is
    // perfectly rectangular and visual Up/Down cursor navigation is exact.
    let prefix: u16 = 5;
    let text_width = input_area.width.saturating_sub(prefix) as usize;
    let text_width = text_width.max(1);

    // Build input lines — prefix first line with prompt glyph
    let text_str = app.composer.text();
    let raw_lines: Vec<&str> = if text_str.is_empty() {
        vec![""]
    } else {
        text_str.lines().collect()
    };

    let mut rendered_lines: Vec<Line<'static>> = Vec::new();
    for (i, raw_line) in raw_lines.iter().enumerate() {
        // Determine prompt glyph for the very first chunk only.
        let (first_glyph, glyph_style) = if app.active_turn_id.is_some() {
            ("  ⋯  ".to_string(), Style::default().fg(COLOR_MUTED))
        } else if text_str.is_empty() {
            ("  ❯  ".to_string(), Style::default().fg(COLOR_MUTED))
        } else {
            (
                "  ❯  ".to_string(),
                Style::default()
                    .fg(COLOR_PROMPT)
                    .add_modifier(Modifier::BOLD),
            )
        };
        // Continuation prefix: same width, no glyph.
        let cont_prefix = "     ".to_string(); // 5 spaces

        if text_str.is_empty() {
            rendered_lines.push(Line::from(vec![
                Span::styled(first_glyph, glyph_style),
                Span::styled(
                    "Message codezilla… (4 lines max · Shift+Enter newline · Ctrl+U kill line · Ctrl/Alt+←/→ word jump · ↑/↓ move line)",
                    Style::default().fg(COLOR_MUTED),
                ),
            ]));
            break;
        }

        let mut chunks = split_at_width(raw_line, text_width);
        if chunks.is_empty() {
            chunks.push(String::new());
        }

        for (chunk_index, chunk) in chunks.into_iter().enumerate() {
            if i == 0 && chunk_index == 0 {
                rendered_lines.push(Line::from(vec![
                    Span::styled(first_glyph.clone(), glyph_style),
                    Span::raw(chunk),
                ]));
            } else {
                rendered_lines.push(Line::from(vec![
                    Span::styled(cont_prefix.clone(), Style::default()),
                    Span::raw(chunk),
                ]));
            }
        }
    }

    let (row, col) = app
        .composer
        .visual_cursor_row_col(text_width, text_width);
    let visible_rows = input_area.height as usize;
    let composer_scroll = if row >= visible_rows {
        row + 1 - visible_rows
    } else {
        0
    };

    let paragraph = Paragraph::new(Text::from(rendered_lines)).scroll((composer_scroll as u16, 0));
    frame.render_widget(paragraph, input_area);

    // Cursor position — x_offset is always `prefix` (5) because every row
    // starts at the same column.
    if app.pending_approval.is_none() {
        let visible_row = row.saturating_sub(composer_scroll);
        let cursor_x = input_area.x
            + prefix
            + (col as u16).min(input_area.width.saturating_sub(prefix + 1));
        let cursor_y = input_area.y + (visible_row as u16).min(input_area.height.saturating_sub(1));
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

// ─── Status bar ───────────────────────────────────────────────────────────────

fn render_status_bar(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    let message = app
        .error_message
        .clone()
        .unwrap_or_else(|| app.status_message.clone());

    let (msg_style, prefix) = if app.error_message.is_some() {
        (Style::default().fg(COLOR_ERROR), "✗  ")
    } else {
        (Style::default().fg(COLOR_MUTED), "   ")
    };

    let mouse_mode = if app.mouse_capture_enabled {
        "scroll"
    } else {
        "select"
    };
    let hints = format!(
        "^A·approve:{}  ^M·mouse:{mouse_mode}  scroll: wheel·PgUp/Dn·^U/D  ^N·new  ^F·fork  ^C·interrupt  ^Q·quit",
        app.approval_mode_label()
    );

    let line = Line::from(vec![
        Span::styled(prefix.to_string(), msg_style),
        Span::styled(message, msg_style),
        Span::styled(format!("  ·  {hints}"), Style::default().fg(COLOR_MUTED)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

// ─── Approval panel ───────────────────────────────────────────────────────────

fn render_approval_panel(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    let Some(approval) = &app.pending_approval else {
        return;
    };
    let body_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title + metadata
            Constraint::Min(2),    // action preview
            Constraint::Length(1), // key hint
        ])
        .split(area);

    frame.render_widget(
        Block::default()
            .title(Span::styled(
                " approval required ",
                Style::default()
                    .fg(COLOR_APPROVAL)
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(COLOR_APPROVAL)),
        area,
    );

    let header = Text::from(vec![
        Line::from(Span::styled(
            approval.approval.request.title.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("category  ", Style::default().fg(COLOR_MUTED)),
            Span::raw(format!("{:?}", approval.approval.request.category)),
            Span::styled("  ·  ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                "A",
                Style::default()
                    .fg(COLOR_APPROVAL)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" approve  ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                "D / Esc",
                Style::default()
                    .fg(COLOR_ERROR)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" deny", Style::default().fg(COLOR_MUTED)),
        ]),
        Line::from(vec![
            Span::styled("reason    ", Style::default().fg(COLOR_MUTED)),
            Span::raw(approval.approval.request.justification.clone()),
        ]),
    ]);
    let header_area = inset(body_layout[0], 1, 0);
    frame.render_widget(Paragraph::new(header), header_area);

    let preview = truncate_lines(
        &approval.action_preview,
        body_layout[1].height.saturating_sub(2) as usize,
    );
    frame.render_widget(
        Paragraph::new(preview).wrap(Wrap { trim: false }).block(
            Block::default()
                .title(Span::styled(" action ", Style::default().fg(COLOR_MUTED)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(COLOR_BORDER)),
        ),
        body_layout[1],
    );

    let hint_area = inset(body_layout[2], 1, 0);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " pending approval blocks new input until resolved",
                Style::default().fg(COLOR_MUTED),
            ),
            Span::styled("  ·  ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                "A",
                Style::default()
                    .fg(COLOR_APPROVAL)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" approve", Style::default().fg(COLOR_MUTED)),
            Span::styled("  ·  ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                "D",
                Style::default()
                    .fg(COLOR_ERROR)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" deny", Style::default().fg(COLOR_MUTED)),
        ])),
        hint_area,
    );
}

// ─── Autocomplete section ─────────────────────────────────────────────────────

fn render_autocomplete(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    if area.height == 0 {
        return;
    }
    let suggestions = &app.autocomplete_suggestions;
    let selected = app.autocomplete_selected;

    let scroll = app.autocomplete_scroll;
    let lines: Vec<Line> = suggestions
        .iter()
        .enumerate()
        .skip(scroll)
        .take(area.height as usize)
        .map(|(i, item)| {
            if i == selected {
                Line::from(vec![
                    Span::styled(
                        "  ❯  ",
                        Style::default()
                            .fg(COLOR_ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        item.label.trim_end().to_owned(),
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::styled("     ", Style::default()),
                    Span::styled(item.label.trim_end().to_owned(), Style::default().fg(COLOR_MUTED)),
                ])
            }
        })
        .collect();

    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}

/// Shrink a rect by `left` columns and `top` rows (simple inset helper).
fn inset(r: Rect, left: u16, top: u16) -> Rect {
    Rect {
        x: r.x + left,
        y: r.y + top,
        width: r.width.saturating_sub(left),
        height: r.height.saturating_sub(top),
    }
}
