pub mod app;
pub mod input;
pub mod markdown;
pub mod render;
pub mod types;

use anyhow::Result;
use crossterm::{
    cursor::{Hide, Show},
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste,
        EnableMouseCapture, Event, MouseButton, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{
    io,
    time::{Duration, Instant},
};
use tokio::sync::broadcast::error::TryRecvError;
use tracing::warn;

use super::runtime::{ConversationRuntime, EventFilter};
use app::InteractiveApp;
use types::FocusPane;

const ACTIVE_POLL_INTERVAL: Duration = Duration::from_millis(40);
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(250);

struct TerminalSession {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalSession {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste,
            Hide
        )?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok(Self { terminal })
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            DisableMouseCapture,
            DisableBracketedPaste,
            Show,
            LeaveAlternateScreen
        );
        let _ = self.terminal.show_cursor();
    }
}

/// Public entry point. Runs the full interactive TUI until the user quits.
pub async fn run_interactive_tui(
    runtime: ConversationRuntime,
    initial_thread_id: String,
    initial_prompt: Option<String>,
) -> Result<i32> {
    let mut session = TerminalSession::enter()?;
    let subscriber_id = format!("interactive_ui_{}", uuid::Uuid::new_v4().simple());
    let mut subscription = runtime
        .event_bus()
        .subscribe(subscriber_id.clone(), EventFilter { thread_id: None });
    let mut app = InteractiveApp::bootstrap(runtime.clone(), initial_thread_id).await?;
    let mut dirty = true;
    let mut last_draw = Instant::now();

    if let Some(prompt) = initial_prompt {
        app.composer.set_text(prompt);
        app.submit_composer().await?;
        dirty = true;
    }

    // Track the last-applied mouse capture state so we can react to changes.
    let mut mouse_capture_active = true; // matches EnableMouseCapture called in TerminalSession::enter

    loop {
        let spinner_active = app.active_turn_id.is_some() || app.pending_compact.is_some();
        if spinner_active && last_draw.elapsed() >= ACTIVE_POLL_INTERVAL {
            app.spinner_tick = app.spinner_tick.wrapping_add(1);
            dirty = true;
        }

        // Poll background compaction task (non-blocking try_recv under the hood).
        if app.pending_compact.is_some() {
            if app.poll_compact_result().await? {
                dirty = true;
            }
        }

        if dirty {
            session
                .terminal
                .draw(|frame| render::draw(&mut app, frame))?;
            last_draw = Instant::now();
            dirty = false;
        }

        // Sync mouse capture with app preference whenever it changes.
        if app.mouse_capture_enabled != mouse_capture_active {
            mouse_capture_active = app.mouse_capture_enabled;
            dirty = true;
            if mouse_capture_active {
                let _ = execute!(session.terminal.backend_mut(), EnableMouseCapture);
            } else {
                let _ = execute!(session.terminal.backend_mut(), DisableMouseCapture);
            }
        }

        // Drain all pending runtime events before polling keyboard input
        loop {
            match subscription.receiver.try_recv() {
                Ok(event) => {
                    app.handle_runtime_event(event).await?;
                    dirty = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Lagged(count)) => {
                    warn!(dropped = count, "tui lagged runtime events");
                    app.error_message =
                        Some(format!("UI dropped {count} runtime events while rendering"));
                    dirty = true;
                    break;
                }
                Err(TryRecvError::Closed) => {
                    warn!("tui runtime event stream closed");
                    app.error_message = Some("Runtime event stream closed".into());
                    dirty = true;
                    break;
                }
            }
        }

        if app.should_quit {
            break;
        }

        let poll_interval = if app.active_turn_id.is_some() {
            ACTIVE_POLL_INTERVAL
        } else {
            IDLE_POLL_INTERVAL
        };

        if event::poll(poll_interval)? {
            match event::read()? {
                Event::Key(key) => {
                    input::handle_key(&mut app, key).await?;
                    dirty = true;
                }
                Event::Mouse(mouse) if mouse_capture_active => {
                    match mouse.kind {
                        // ── Wheel scroll ─────────────────────────────────────
                        MouseEventKind::ScrollUp => {
                            app.scroll_transcript(-3);
                            dirty = true;
                        }
                        MouseEventKind::ScrollDown => {
                            app.scroll_transcript(3);
                            dirty = true;
                        }
                        // ── Drag-to-select (left button) ─────────────────────
                        MouseEventKind::Down(MouseButton::Left) => {
                            app.begin_transcript_drag(mouse.column, mouse.row);
                            dirty = true;
                        }
                        MouseEventKind::Drag(MouseButton::Left) => {
                            app.update_transcript_drag(mouse.column, mouse.row);
                            dirty = true;
                        }
                        MouseEventKind::Up(MouseButton::Left) => {
                            app.finish_transcript_drag(mouse.column, mouse.row);
                            dirty = true;
                        }
                        _ => {}
                    }
                }
                Event::Paste(text) => {
                    if app.pending_approval.is_none() && app.focus == FocusPane::Composer {
                        app.jump_transcript_to_bottom();
                        // Normalize line endings: convert \r\n and \r to \n so the
                        // composer counts and renders lines consistently.
                        let text = text.replace("\r\n", "\n").replace('\r', "\n");
                        // Strip trailing newlines to prevent accidental submission
                        // when copying from code blocks or terminal output.
                        let text = text.trim_end_matches('\n');
                        app.composer.insert_str(&text);
                        dirty = true;
                    }
                }
                Event::Resize(_, _) => {
                    dirty = true;
                }
                _ => {}
            }
        }
    }

    runtime.event_bus().unsubscribe(&subscriber_id);
    Ok(0)
}
