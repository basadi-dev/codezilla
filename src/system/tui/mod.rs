pub mod app;
pub mod input;
pub mod render;
pub mod types;

use anyhow::Result;
use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{io, time::Duration};
use tokio::sync::broadcast::error::TryRecvError;

use super::runtime::{ConversationRuntime, EventFilter};
use app::InteractiveApp;
use types::FocusPane;

const POLL_INTERVAL: Duration = Duration::from_millis(40);

struct TerminalSession {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalSession {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, Hide)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok(Self { terminal })
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), Show, LeaveAlternateScreen);
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

    if let Some(prompt) = initial_prompt {
        app.composer.set_text(prompt);
        app.submit_composer().await?;
    }

    loop {
        session.terminal.draw(|frame| render::draw(&mut app, frame))?;

        // Drain all pending runtime events before polling keyboard input
        loop {
            match subscription.receiver.try_recv() {
                Ok(event) => app.handle_runtime_event(event).await?,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Lagged(count)) => {
                    app.error_message =
                        Some(format!("UI dropped {count} runtime events while rendering"));
                    break;
                }
                Err(TryRecvError::Closed) => {
                    app.error_message = Some("Runtime event stream closed".into());
                    break;
                }
            }
        }

        if app.should_quit {
            break;
        }

        if event::poll(POLL_INTERVAL)? {
            match event::read()? {
                Event::Key(key) => input::handle_key(&mut app, key).await?,
                Event::Paste(text) => {
                    if app.pending_approval.is_none() && app.focus == FocusPane::Composer {
                        app.composer.insert_str(&text);
                    }
                }
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }

    runtime.event_bus().unsubscribe(&subscriber_id);
    Ok(0)
}
