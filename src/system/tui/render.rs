use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use super::app::InteractiveApp;
use super::types::{
    basename, centered_rect, composer_height, current_state_label, panel_block, thread_label,
    transcript_text, truncate_lines, FocusPane, COLOR_ACCENT, COLOR_APPROVAL,
    COLOR_BORDER, COLOR_ERROR, COLOR_MUTED,
};

/// Draw the complete TUI frame. Pure — only reads from `app`, mutates only
/// `app.transcript_scroll` to clamp it to the actual viewport size.
pub fn draw(app: &mut InteractiveApp, frame: &mut Frame) {
    let ch = composer_height(&app.composer);
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(12),
            Constraint::Length(ch),
            Constraint::Length(1),
        ])
        .split(frame.area());

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(30), Constraint::Min(20)])
        .split(outer[1]);

    render_header(app, frame, outer[0]);
    render_threads(app, frame, body[0]);
    render_transcript(app, frame, body[1]);
    render_composer(app, frame, outer[2]);
    render_status_bar(app, frame, outer[3]);

    if app.pending_approval.is_some() {
        render_approval_modal(app, frame);
    }
}

fn render_header(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    let block = panel_block("Codezilla", false);
    let meta = app.current_thread_meta.as_ref();
    let title = meta.map(thread_label).unwrap_or_else(|| "No thread".into());
    let cwd = meta
        .and_then(|t| t.cwd.as_ref())
        .map(|c| basename(c))
        .unwrap_or_else(|| "unknown cwd".into());
    let model = meta
        .map(|t| format!("{} via {}", t.model_id, t.provider_id))
        .unwrap_or_else(|| "model unavailable".into());
    let state = current_state_label(
        app.active_turn_id.is_some(),
        app.pending_approval.is_some(),
    );
    let text = Text::from(vec![
        Line::from(vec![
            Span::styled(" Thread ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                title,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(" State ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                state,
                Style::default()
                    .fg(COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Path ", Style::default().fg(COLOR_MUTED)),
            Span::raw(cwd),
            Span::raw("  "),
            Span::styled(" Model ", Style::default().fg(COLOR_MUTED)),
            Span::raw(model),
        ]),
    ]);
    frame.render_widget(Paragraph::new(text).block(block), area);
}

fn render_threads(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    let selected_index = app
        .selected_thread_id
        .as_ref()
        .and_then(|selected| {
            app.threads
                .iter()
                .position(|t| &t.thread_id == selected)
        })
        .or_else(|| (!app.threads.is_empty()).then_some(0));

    let items = if app.threads.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "No threads",
            Style::default().fg(COLOR_MUTED),
        )))]
    } else {
        app.threads
            .iter()
            .map(|t| {
                let is_current = t.thread_id == app.current_thread_id;
                let marker = if is_current { "●" } else { "○" };
                let status = format!("{:?}", t.status).replace('"', "");
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            marker,
                            Style::default().fg(if is_current { COLOR_ACCENT } else { COLOR_MUTED }),
                        ),
                        Span::raw(" "),
                        Span::styled(
                            thread_label(t),
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]),
                    Line::from(vec![
                        Span::styled(status, Style::default().fg(COLOR_MUTED)),
                        Span::raw("  "),
                        Span::styled(
                            basename(t.cwd.as_deref().unwrap_or("")),
                            Style::default().fg(COLOR_MUTED),
                        ),
                    ]),
                ])
            })
            .collect::<Vec<_>>()
    };

    let mut state = ListState::default();
    state.select(selected_index);
    let list = List::new(items)
        .block(panel_block("Threads", app.focus == FocusPane::Threads))
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(26, 35, 49))
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ");
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_transcript(app: &mut InteractiveApp, frame: &mut Frame, area: Rect) {
    let (text, total_lines) = transcript_text(&app.transcript);
    let viewport_height = area.height.saturating_sub(2) as usize;
    let max_scroll = total_lines.saturating_sub(viewport_height) as u16;
    if app.auto_scroll || app.transcript_scroll > max_scroll {
        app.transcript_scroll = max_scroll;
    }

    let title = if app.transcript.is_empty() {
        "Transcript"
    } else {
        "Transcript · live"
    };
    let paragraph = Paragraph::new(text)
        .block(panel_block(title, app.focus == FocusPane::Transcript))
        .scroll((app.transcript_scroll, 0))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn render_composer(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    let mut lines = app
        .composer
        .text()
        .lines()
        .map(|line| Line::from(line.to_string()))
        .collect::<Vec<_>>();
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "Type a prompt, /help, or use Ctrl+N for a new thread",
            Style::default().fg(COLOR_MUTED),
        )));
    }

    let title = if app.active_turn_id.is_some() {
        "Composer · Enter queues into active turn"
    } else {
        "Composer · Enter sends"
    };

    let paragraph = Paragraph::new(Text::from(lines))
        .block(panel_block(title, app.focus == FocusPane::Composer))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);

    if app.focus == FocusPane::Composer && app.pending_approval.is_none() {
        let (row, col) = app.composer.cursor_row_col();
        let cursor_x = area.x + 1 + (col as u16).min(area.width.saturating_sub(3));
        let cursor_y = area.y + 1 + (row as u16).min(area.height.saturating_sub(3));
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

fn render_status_bar(app: &InteractiveApp, frame: &mut Frame, area: Rect) {
    let focus = match app.focus {
        FocusPane::Threads => "threads",
        FocusPane::Transcript => "transcript",
        FocusPane::Composer => "composer",
    };
    let message = app
        .error_message
        .clone()
        .unwrap_or_else(|| app.status_message.clone());
    let style = if app.error_message.is_some() {
        Style::default().fg(COLOR_ERROR).bg(Color::Rgb(28, 18, 18))
    } else {
        Style::default().fg(Color::Black).bg(COLOR_ACCENT)
    };
    let text = format!(
        " {}  |  Tab focus  Ctrl+N new  Ctrl+F fork  Ctrl+C interrupt  Ctrl+Q quit  |  {} ",
        focus, message
    );
    frame.render_widget(Paragraph::new(text).style(style), area);
}

fn render_approval_modal(app: &InteractiveApp, frame: &mut Frame) {
    let Some(approval) = &app.pending_approval else {
        return;
    };
    let area = centered_rect(68, 56, frame.area());
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(8),
            Constraint::Length(2),
        ])
        .split(area);

    frame.render_widget(Clear, area);
    frame.render_widget(
        Block::default()
            .title(" Approval Required ")
            .borders(Borders::ALL)
            .border_style(
                Style::default()
                    .fg(COLOR_APPROVAL)
                    .add_modifier(Modifier::BOLD),
            ),
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
            Span::styled("Category: ", Style::default().fg(COLOR_MUTED)),
            Span::raw(format!("{:?}", approval.approval.request.category)),
        ]),
        Line::from(vec![
            Span::styled("Justification: ", Style::default().fg(COLOR_MUTED)),
            Span::raw(approval.approval.request.justification.clone()),
        ]),
    ]);
    frame.render_widget(Paragraph::new(header), body[0]);

    let preview = truncate_lines(
        &approval.action_preview,
        body[1].height.saturating_sub(2) as usize,
    );
    frame.render_widget(
        Paragraph::new(preview).wrap(Wrap { trim: false }).block(
            Block::default()
                .title(" Action ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(COLOR_BORDER)),
        ),
        body[1],
    );

    frame.render_widget(
        Paragraph::new("Press A to approve or D / Esc to deny").style(
            Style::default()
                .fg(COLOR_APPROVAL)
                .add_modifier(Modifier::BOLD),
        ),
        body[2],
    );
}
