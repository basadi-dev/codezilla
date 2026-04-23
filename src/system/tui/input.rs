use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::super::domain::ApprovalDecision;
use super::app::InteractiveApp;
use super::types::FocusPane;

pub async fn handle_key(app: &mut InteractiveApp, key: KeyEvent) -> Result<()> {
    if key.kind == KeyEventKind::Release {
        return Ok(());
    }

    if app.pending_approval.is_some() {
        return handle_approval_key(app, key).await;
    }

    match (key.code, key.modifiers) {
        (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
            app.should_quit = true;
        }
        (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
            app.create_new_thread().await?;
        }
        (KeyCode::Char('f'), KeyModifiers::CONTROL) => {
            app.fork_current_thread().await?;
        }
        (KeyCode::Char('r'), KeyModifiers::CONTROL) => {
            app.refresh_threads().await?;
            app.status_message = format!("Loaded {} threads", app.threads.len());
            app.error_message = None;
        }
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
            app.interrupt_active_turn().await?;
        }
        (KeyCode::Tab, _) => app.next_focus(),
        (KeyCode::BackTab, _) => app.previous_focus(),
        _ => match app.focus {
            FocusPane::Threads => handle_threads_key(app, key).await?,
            FocusPane::Transcript => handle_transcript_key(app, key),
            FocusPane::Composer => handle_composer_key(app, key).await?,
        },
    }
    Ok(())
}

async fn handle_approval_key(app: &mut InteractiveApp, key: KeyEvent) -> Result<()> {
    match key.code {
        KeyCode::Char('a') | KeyCode::Char('A') => {
            app.resolve_pending_approval(ApprovalDecision::Approved).await?;
        }
        KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Esc => {
            app.resolve_pending_approval(ApprovalDecision::Denied).await?;
        }
        _ => {}
    }
    Ok(())
}

async fn handle_threads_key(app: &mut InteractiveApp, key: KeyEvent) -> Result<()> {
    match key.code {
        KeyCode::Up => app.select_thread_delta(-1),
        KeyCode::Down => app.select_thread_delta(1),
        KeyCode::Enter => app.open_selected_thread().await?,
        KeyCode::Char('j') => app.select_thread_delta(1),
        KeyCode::Char('k') => app.select_thread_delta(-1),
        _ => {}
    }
    Ok(())
}

fn handle_transcript_key(app: &mut InteractiveApp, key: KeyEvent) {
    app.auto_scroll = false;
    match key.code {
        KeyCode::Up => app.transcript_scroll = app.transcript_scroll.saturating_sub(1),
        KeyCode::Down => app.transcript_scroll = app.transcript_scroll.saturating_add(1),
        KeyCode::PageUp => app.transcript_scroll = app.transcript_scroll.saturating_sub(8),
        KeyCode::PageDown => app.transcript_scroll = app.transcript_scroll.saturating_add(8),
        KeyCode::Home => app.transcript_scroll = 0,
        KeyCode::End => app.auto_scroll = true,
        _ => {}
    }
}

async fn handle_composer_key(app: &mut InteractiveApp, key: KeyEvent) -> Result<()> {
    match key.code {
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
            app.composer.insert_char('\n');
        }
        KeyCode::Enter => app.submit_composer().await?,
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.composer.insert_char(ch);
        }
        KeyCode::Backspace => app.composer.backspace(),
        KeyCode::Delete => app.composer.delete(),
        KeyCode::Left => app.composer.move_left(),
        KeyCode::Right => app.composer.move_right(),
        KeyCode::Up => app.composer.move_up(),
        KeyCode::Down => app.composer.move_down(),
        KeyCode::Home => app.composer.move_home(),
        KeyCode::End => app.composer.move_end(),
        KeyCode::Esc => {
            if !app.composer.is_empty() {
                app.composer = super::types::ComposerState::default();
            }
        }
        _ => {}
    }
    Ok(())
}
