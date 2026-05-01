use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::Paragraph,
    Frame,
};

use super::app::InteractiveApp;
use super::types::{
    basename, composer_height, current_state_label, spinner_frame, split_at_width, thread_label,
    truncate_lines, COLOR_ACCENT, COLOR_APPROVAL, COLOR_BORDER, COLOR_ERROR, COLOR_MUTED,
    COLOR_PROMPT, COLOR_USER,
};

/// Dim colour used for right-aligned status-bar key hints.
const COLOR_HINT: Color = Color::Rgb(60, 65, 75);

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
    let ac_h = if !app.autocomplete.is_active() {
        0
    } else {
        (app.autocomplete.suggestions().len() as u16).min(8)
    };

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),    // header
            Constraint::Length(1),    // separator
            Constraint::Min(4),       // transcript
            Constraint::Length(ac_h), // autocomplete list (0 when hidden)
            Constraint::Length(ch),   // composer
            Constraint::Length(1),    // status bar
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

// ─── Header ────────────────────────────────────────────────────────────────────
//
// Layout:
//   Left:  ◈ codezilla  ⠋ Streaming…  thread-title
//   Right: cwd │ model │ reasoning │ 🔓
//
// - `reasoning` is only shown when explicitly set (not "off").
// - Approval mode shown as a compact icon (🔓 auto / 🔒 ask).
// - Thread title is truncated with ellipsis if it would overflow.

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
    let approval_icon = if app.auto_approve_tools_enabled() {
        "🔓"
    } else {
        "🔒"
    };
    let state = current_state_label(app.active_turn_id.is_some(), app.approval.has_pending());

    let state_color = if app.error_message.is_some() {
        COLOR_ERROR
    } else if app.active_turn_id.is_some() {
        COLOR_ACCENT
    } else if app.approval.has_pending() {
        COLOR_APPROVAL
    } else {
        COLOR_MUTED
    };

    // State indicator: spinning dot or static
    let state_sigil = if app.active_turn_id.is_some() {
        spinner_frame(app.activity.spinner_tick())
    } else {
        "●"
    };

    // Live activity label: prefer the structured header (tool name + elapsed
    // time, multi-tool summary when parallel), fall back to the state label.
    let live_label: String = if app.active_turn_id.is_some() {
        app.activity
            .header_line(std::time::Instant::now(), "◆ generating…")
            .unwrap_or_else(|| state.to_string())
    } else {
        state.to_string()
    };

    // ── Left side: brand + live state ────────────────────────────────────────
    let mut left_spans: Vec<Span<'static>> = vec![
        Span::styled(
            " ◈ ",
            Style::default()
                .fg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "codezilla",
            Style::default()
                .fg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            state_sigil.to_string(),
            Style::default()
                .fg(state_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {live_label}"), Style::default().fg(state_color)),
    ];

    // ── Right side: context info ─────────────────────────────────────────────
    let mut right_spans: Vec<Span<'static>> = vec![];
    right_spans.push(Span::styled(cwd.clone(), Style::default().fg(Color::White)));
    right_spans.push(Span::styled(" │ ", Style::default().fg(COLOR_BORDER)));
    right_spans.push(Span::styled(model, Style::default().fg(COLOR_MUTED)));
    if !reasoning.is_empty() {
        right_spans.push(Span::styled(" │ ", Style::default().fg(COLOR_BORDER)));
        right_spans.push(Span::styled(
            format!("reasoning:{reasoning}"),
            Style::default().fg(COLOR_MUTED),
        ));
    }
    right_spans.push(Span::styled(" │ ", Style::default().fg(COLOR_BORDER)));
    right_spans.push(Span::styled(
        approval_icon.to_string(),
        Style::default().fg(COLOR_MUTED),
    ));

    // ── Calculate widths and fit thread title in between ─────────────────────
    let left_width: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
    let right_width: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
    let available = area.width as usize;
    // 2 chars for "  " gap between left and thread
    let max_thread = available.saturating_sub(left_width + right_width + 2);
    let thread_display = if thread.chars().count() > max_thread && max_thread > 3 {
        // Truncate with ellipsis
        let chars: Vec<char> = thread.chars().take(max_thread.saturating_sub(1)).collect();
        format!("{}…", chars.iter().collect::<String>())
    } else {
        thread
    };
    let thread_display_width = thread_display.chars().count();

    left_spans.push(Span::styled("  ", Style::default()));
    left_spans.push(Span::styled(
        thread_display,
        Style::default().fg(COLOR_USER),
    ));

    // Padding between left+thread and right-aligned context
    let used = left_width + 2 + thread_display_width + right_width;
    let padding = available.saturating_sub(used);
    left_spans.push(Span::styled(" ".repeat(padding), Style::default()));
    left_spans.extend(right_spans);

    frame.render_widget(Paragraph::new(Line::from(left_spans)), area);
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
    let transcript_area = if app.approval.has_pending() && area.height >= 10 {
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

    let effective_scroll = app.transcript_view.settle_at(max_scroll);

    // Pass area.width so body lines are hard-wrapped here rather than by ratatui.
    // This guarantees 1 Line in the vec == 1 terminal row, making drag coords exact.
    let overscan = 8usize;
    let start_line = effective_scroll as usize;
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

    // Selection range in char indices, if any
    let sel_range = app.composer_selection_range();

    // Track absolute char offset as we walk through logical lines,
    // so we can map selection ranges onto visual chunks.
    let mut char_offset = 0usize;

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
                    "Message codezilla… (Shift+Enter for newline)",
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
            let chunk_char_count = chunk.chars().count();
            let abs_start = char_offset;
            let abs_end = char_offset + chunk_char_count;

            // Determine if this chunk's text is fully or partially selected
            let (prefix_spans, text_spans) = if let Some((sel_lo, sel_hi)) = sel_range {
                let rel_lo = sel_lo.saturating_sub(abs_start);
                let rel_hi = sel_hi.min(abs_end).saturating_sub(abs_start);
                let chunk_chars: Vec<char> = chunk.chars().collect();
                let chunk_len = chunk_chars.len();

                if rel_hi == 0 || rel_lo >= chunk_len {
                    // No selection in this chunk
                    let prefix = if i == 0 && chunk_index == 0 {
                        vec![Span::styled(first_glyph.clone(), glyph_style)]
                    } else {
                        vec![Span::styled(cont_prefix.clone(), Style::default())]
                    };
                    (prefix, vec![Span::raw(chunk)])
                } else if rel_lo == 0 && rel_hi >= chunk_len {
                    // Entire chunk is selected — also highlight the prefix
                    let sel_style = Style::default()
                        .fg(Color::Black)
                        .bg(Color::Rgb(100, 160, 220));
                    let prefix = if i == 0 && chunk_index == 0 {
                        vec![Span::styled(first_glyph.clone(), sel_style)]
                    } else {
                        vec![Span::styled(cont_prefix.clone(), sel_style)]
                    };
                    let text: String = chunk_chars.iter().collect();
                    (prefix, vec![Span::styled(text, sel_style)])
                } else {
                    // Partial selection
                    let lo = rel_lo.min(chunk_len);
                    let hi = rel_hi.min(chunk_len);
                    let prefix = if i == 0 && chunk_index == 0 {
                        vec![Span::styled(first_glyph.clone(), glyph_style)]
                    } else {
                        vec![Span::styled(cont_prefix.clone(), Style::default())]
                    };
                    let mut spans = Vec::new();
                    if lo > 0 {
                        let before: String = chunk_chars[..lo].iter().collect();
                        spans.push(Span::raw(before));
                    }
                    let sel_text: String = chunk_chars[lo..hi].iter().collect();
                    spans.push(Span::styled(
                        sel_text,
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Rgb(100, 160, 220)),
                    ));
                    if hi < chunk_len {
                        let after: String = chunk_chars[hi..].iter().collect();
                        spans.push(Span::raw(after));
                    }
                    (prefix, spans)
                }
            } else {
                let prefix = if i == 0 && chunk_index == 0 {
                    vec![Span::styled(first_glyph.clone(), glyph_style)]
                } else {
                    vec![Span::styled(cont_prefix.clone(), Style::default())]
                };
                (prefix, vec![Span::raw(chunk)])
            };

            rendered_lines.push(Line::from(
                prefix_spans
                    .into_iter()
                    .chain(text_spans.into_iter())
                    .collect::<Vec<_>>(),
            ));
            char_offset = abs_end;
        }

        // Account for the '\n' between logical lines (except the last)
        if i + 1 < raw_lines.len() {
            char_offset += 1; // the newline char
        }
    }

    let (row, col) = app.composer.visual_cursor_row_col(text_width, text_width);
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
    if !app.approval.has_pending() {
        let visible_row = row.saturating_sub(composer_scroll);
        let cursor_x =
            input_area.x + prefix + (col as u16).min(input_area.width.saturating_sub(prefix + 1));
        let cursor_y = input_area.y + (visible_row as u16).min(input_area.height.saturating_sub(1));
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

// ─── Status bar ───────────────────────────────────────────────────────────────
//
// Layout:
//   Left:  status message (or error)
//   Right: essential key hints (dimmed, right-aligned)
//
// Only 3 essential shortcuts are shown permanently. Full list available via /help.

fn render_status_bar(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    // ── Quit-confirm mode: take over the entire status bar ────────────────────
    if app.quit_requested {
        let spans = vec![
            Span::styled(
                "  ⚠ ",
                Style::default()
                    .fg(COLOR_APPROVAL)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Press ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                "^Q",
                Style::default()
                    .fg(COLOR_APPROVAL)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" again to quit", Style::default().fg(COLOR_MUTED)),
            Span::styled("   ", Style::default()),
            Span::styled(
                "Esc",
                Style::default()
                    .fg(COLOR_ERROR)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to cancel", Style::default().fg(COLOR_MUTED)),
        ];
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
        return;
    }

    let message = app
        .error_message
        .clone()
        .unwrap_or_else(|| app.status_message.clone());

    let (msg_style, prefix) = if app.error_message.is_some() {
        (Style::default().fg(COLOR_ERROR), "✗ ")
    } else {
        (Style::default().fg(COLOR_MUTED), "  ")
    };

    // Build context info: token usage + context remaining %
    let usage = &app.token_usage;
    let has_tokens = usage.input_tokens > 0 || usage.output_tokens > 0;

    // Context remaining percentage (always compute)
    let context_window = app
        .effective_model_settings()
        .context_window
        .unwrap_or(100_000);
    let prompt_budget = context_window.saturating_sub(8_192);
    let used_pct = (usage.input_tokens as f64 / prompt_budget as f64) * 100.0;
    let remaining_pct = (100.0 - used_pct).max(0.0);

    let token_part = if has_tokens {
        let in_k = usage.input_tokens as f64 / 1000.0;
        let out_k = usage.output_tokens as f64 / 1000.0;
        let cached_k = usage.cached_tokens as f64 / 1000.0;
        if usage.cached_tokens > 0 {
            format!("↑{in_k:.1}k ↓{out_k:.1}k ⚡{cached_k:.1}k  ")
        } else {
            format!("↑{in_k:.1}k ↓{out_k:.1}k  ")
        }
    } else {
        String::new()
    };

    let pct_str = format!("ctx {remaining_pct:.0}%");

    // Context width for layout calculation
    let context_width = token_part.chars().count() + pct_str.chars().count() + 2; // +2 for trailing padding

    // Essential keys only — right-aligned, always dim
    let essential_keys = "^N new  ^C stop  ^Q quit";
    let keys_width = essential_keys.chars().count();
    let msg_width = prefix.chars().count() + message.chars().count();

    let mut spans = vec![
        Span::styled(prefix.to_string(), msg_style),
        Span::styled(message, msg_style),
    ];

    let available = area.width as usize;
    let used = msg_width;
    let padding = available.saturating_sub(used + context_width + keys_width);
    if padding > 0 {
        spans.push(Span::styled(" ".repeat(padding), Style::default()));
    }
    if has_tokens {
        spans.push(Span::styled(token_part, Style::default().fg(COLOR_HINT)));
    }
    // Always show context remaining percentage
    let pct_color = if remaining_pct > 50.0 {
        COLOR_HINT // dim — plenty of room
    } else if remaining_pct > 20.0 {
        Color::Rgb(255, 200, 80) // yellow — getting low
    } else {
        COLOR_ERROR // red — critical
    };
    spans.push(Span::styled(pct_str, Style::default().fg(pct_color)));
    spans.push(Span::styled("  ", Style::default()));
    // Only show key hints if there's room
    if available.saturating_sub(used + context_width) >= keys_width {
        spans.push(Span::styled(
            essential_keys,
            Style::default().fg(COLOR_HINT),
        ));
    }

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ─── Approval panel ───────────────────────────────────────────────────────────
//
// Stream-style inline panel (no borders) matching the autocomplete aesthetic.
// Uses the same gutter prefix pattern as transcript entries.

fn render_approval_panel(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    let Some(approval) = app.approval.pending() else {
        return;
    };

    let body_width = (area.width as usize).saturating_sub(5).max(1);

    // Build lines exactly like the autocomplete: prefix + content, no borders.
    let mut lines: Vec<Line<'static>> = Vec::new();

    // ── Header line: sigil + title ──────────────────────────────────────────
    lines.push(Line::from(vec![
        Span::styled(
            "  ⚠  ",
            Style::default()
                .fg(COLOR_APPROVAL)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            approval.approval.request.title.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    // ── Category + key hints (same line, compact) ─────────────────────────────
    lines.push(Line::from(vec![
        Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            format!("{:?}", approval.approval.request.category),
            Style::default().fg(COLOR_MUTED),
        ),
        Span::styled("  ·  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            "A",
            Style::default()
                .fg(COLOR_APPROVAL)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" approve  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            "U",
            Style::default()
                .fg(COLOR_APPROVAL)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" auto-approve  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            "D",
            Style::default()
                .fg(COLOR_ERROR)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" deny  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            "Esc",
            Style::default()
                .fg(COLOR_ERROR)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" cancel", Style::default().fg(COLOR_MUTED)),
    ]));
    // ── Justification / reason ────────────────────────────────────────────────
    if !approval.approval.request.justification.is_empty() {
        for chunk in split_at_width(&approval.approval.request.justification, body_width) {
            lines.push(Line::from(vec![
                Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
                Span::styled(chunk, Style::default().fg(Color::White)),
            ]));
        }
    }

    // ── Separator before action preview ─────────────────────────────────────────
    lines.push(Line::from(vec![
        Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
        Span::styled("─".repeat(body_width), Style::default().fg(COLOR_BORDER)),
    ]));

    // ── Action preview (hard-wrapped, gutter-prefixed) ─────────────────────────
    let preview = truncate_lines(
        &approval.action_preview,
        area.height.saturating_sub(lines.len() as u16 + 1) as usize,
    );
    for preview_line in preview.lines() {
        for chunk in split_at_width(preview_line, body_width) {
            lines.push(Line::from(vec![
                Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
                Span::styled(chunk, Style::default().fg(Color::White)),
            ]));
        }
    }

    // ── Bottom hint line ──────────────────────────────────────────────────────
    lines.push(Line::from(vec![
        Span::styled("  │  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            "pending approval blocks new input until resolved",
            Style::default().fg(COLOR_MUTED),
        ),
    ]));

    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}

// ─── Autocomplete section ─────────────────────────────────────────────────────

fn render_autocomplete(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    if area.height == 0 {
        return;
    }
    let suggestions = app.autocomplete.suggestions();
    let selected = app.autocomplete.selected_index();
    let scroll = app.autocomplete.scroll_offset();
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
                    Span::styled(
                        item.label.trim_end().to_owned(),
                        Style::default().fg(COLOR_MUTED),
                    ),
                ])
            }
        })
        .collect();

    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}
