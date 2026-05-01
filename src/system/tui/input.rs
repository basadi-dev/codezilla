use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::super::domain::ApprovalDecision;
use super::app::InteractiveApp;
use super::types::FocusPane;

pub async fn handle_key(app: &mut InteractiveApp, key: KeyEvent) -> Result<()> {
    if key.kind == KeyEventKind::Release {
        return Ok(());
    }

    // ── Quit confirmation: second ^Q quits, anything else cancels ─────────
    if app.quit_requested {
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
                app.should_quit = true;
            }
            _ => {
                app.quit_requested = false;
                app.status_message = "Ready".into();
            }
        }
        return Ok(());
    }

    // ── Composer clear confirmation: second ^C clears composer, anything else cancels ──
    if app.composer_clear_requested {
        match (key.code, key.modifiers) {
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                app.composer = super::types::ComposerState::default();
                app.autocomplete.clear();
                app.reset_composer_history_navigation();
                app.composer_clear_requested = false;
                app.status_message = "Composer cleared".into();
                app.error_message = None;
            }
            _ => {
                app.composer_clear_requested = false;
                // Fall through to normal handling — don't return early.
            }
        }
        // If we just cleared the composer, we're done. Otherwise fall through.
        if !app.composer_clear_requested && app.composer.is_empty() {
            return Ok(());
        }
        // If composer_clear_requested was cancelled (any other key), fall through
        // to normal handling below — but only if we didn't already handle the key.
        if !app.composer_clear_requested {
            // Fall through to the main match below.
        } else {
            return Ok(());
        }
    }

    if app.approval.has_pending() {
        return handle_approval_key(app, key).await;
    }

    match (key.code, key.modifiers) {
        (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
            app.quit_requested = true;
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
            if app.transcript_selection.is_active() {
                app.copy_selection_to_clipboard();
                app.clear_selection();
            } else if app.composer_selection.is_active() {
                app.copy_composer_selection_to_clipboard();
                app.clear_composer_selection();
            } else if app.focus == FocusPane::Composer && !app.composer.is_empty() {
                app.composer_clear_requested = true;
                app.status_message = "Press Ctrl+C again to clear composer".into();
                app.error_message = None;
            } else {
                app.interrupt_active_turn().await?;
            }
        }
        // Ctrl+A — start of line when in composer.
        (KeyCode::Char('a'), KeyModifiers::CONTROL) => {
            if app.focus == FocusPane::Composer {
                app.jump_transcript_to_bottom();
                app.composer.move_home();
            }
        }
        // Ctrl+M — toggle mouse capture.
        //   ON  → wheel scrolls transcript (default)
        //   OFF → terminal handles mouse natively; drag-to-select works
        (KeyCode::Char('m'), KeyModifiers::CONTROL) => {
            app.mouse_capture_enabled = !app.mouse_capture_enabled;
            app.status_message = if app.mouse_capture_enabled {
                "Mouse: scroll mode  (Ctrl+M to switch back to select mode)".into()
            } else {
                "Mouse: select mode  (Ctrl+M to switch back to scroll mode)".into()
            };
            app.error_message = None;
        }
        (KeyCode::Char('u'), KeyModifiers::CONTROL) if app.focus == FocusPane::Composer => {
            handle_composer_key(app, key).await?;
        }

        // ── Global scroll — works in ANY focus pane ───────────────────────
        // Ctrl+U / Ctrl+D  →  half-page scroll (12 lines)
        (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
            app.scroll_transcript(-12);
        }
        (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
            app.scroll_transcript(12);
        }
        // PageUp / PageDown  →  8-line scroll
        (KeyCode::PageUp, _) => {
            app.scroll_transcript(-8);
        }
        (KeyCode::PageDown, _) => {
            app.scroll_transcript(8);
        }
        // Ctrl+Up / Ctrl+Down  →  1-line scroll
        (KeyCode::Up, KeyModifiers::CONTROL) => {
            app.scroll_transcript(-1);
        }
        (KeyCode::Down, KeyModifiers::CONTROL) => {
            app.scroll_transcript(1);
        }
        // Ctrl+End  →  jump to bottom + re-enable auto-scroll
        (KeyCode::End, KeyModifiers::CONTROL) => {
            app.transcript_view.jump_to_bottom();
        }
        // Ctrl+Home  →  jump to top
        (KeyCode::Home, KeyModifiers::CONTROL) => {
            app.transcript_view.jump_to_top();
        }

        // Tab / Shift+Tab: autocomplete when suggestions are live, else switch pane
        (KeyCode::Tab, _) => {
            if app.autocomplete.is_active() {
                app.autocomplete_select_next();
            } else {
                app.focus = match app.focus {
                    FocusPane::Composer => FocusPane::Transcript,
                    _ => FocusPane::Composer,
                };
            }
        }
        (KeyCode::BackTab, _) => {
            if app.autocomplete.is_active() {
                app.autocomplete_select_prev();
            } else {
                app.focus = match app.focus {
                    FocusPane::Transcript => FocusPane::Composer,
                    _ => FocusPane::Transcript,
                };
            }
        }

        _ => match app.focus {
            FocusPane::Transcript => handle_transcript_key(app, key),
            FocusPane::Composer => handle_composer_key(app, key).await?,
        },
    }
    Ok(())
}

async fn handle_approval_key(app: &mut InteractiveApp, key: KeyEvent) -> Result<()> {
    match (key.code, key.modifiers) {
        (KeyCode::Char('a') | KeyCode::Char('A'), _) => {
            app.resolve_pending_approval(ApprovalDecision::Approved)
                .await?;
        }
        (KeyCode::Char('u') | KeyCode::Char('U'), _) => {
            app.resolve_pending_approval_auto().await?;
        }
        (KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Esc, _) => {
            app.resolve_pending_approval(ApprovalDecision::Denied)
                .await?;
        }
        _ => {}
    }
    Ok(())
}

fn handle_transcript_key(app: &mut InteractiveApp, key: KeyEvent) {
    match key.code {
        KeyCode::Up => app.scroll_transcript(-1),
        KeyCode::Down => app.scroll_transcript(1),
        KeyCode::PageUp => app.scroll_transcript(-8),
        KeyCode::PageDown => app.scroll_transcript(8),
        KeyCode::Home => app.transcript_view.jump_to_top(),
        KeyCode::End => app.transcript_view.jump_to_bottom(),
        _ => {}
    }
}

async fn handle_composer_key(app: &mut InteractiveApp, key: KeyEvent) -> Result<()> {
    // Any key press in the composer clears an active drag selection.
    app.clear_composer_selection();

    match (key.code, key.modifiers) {
        (KeyCode::Enter, modifiers) if modifiers.contains(KeyModifiers::SHIFT) => {
            app.jump_transcript_to_bottom();
            app.reset_composer_history_navigation();
            app.composer.insert_char('\n');
        }
        (KeyCode::Enter, _) => {
            app.jump_transcript_to_bottom();
            app.autocomplete.clear();
            app.submit_composer().await?
        }

        // ── Alt/Option word navigation ─────────────────────────────────────
        // macOS terminals translate Option+Left → Esc+b  and  Option+Right → Esc+f
        // Crossterm decodes those as Char('b'/'f') with ALT modifier.
        // We handle these *before* the generic char catcher so they don't type.
        (KeyCode::Char('b'), m) if m.contains(KeyModifiers::ALT) => {
            app.jump_transcript_to_bottom();
            app.composer.move_word_left();
        }
        (KeyCode::Char('f'), m) if m.contains(KeyModifiers::ALT) => {
            app.jump_transcript_to_bottom();
            app.composer.move_word_right();
        }
        // Alt+d  →  delete word right  (emacs M-d)
        (KeyCode::Char('d'), m) if m.contains(KeyModifiers::ALT) => {
            app.jump_transcript_to_bottom();
            app.composer.delete_word_right();
        }
        // Alt+Backspace  →  delete word left  (emacs M-DEL)
        (KeyCode::Backspace, m) if m.contains(KeyModifiers::ALT) => {
            app.jump_transcript_to_bottom();
            app.composer.delete_word_left();
        }
        // Ctrl+W  →  delete word left  (emacs/readline C-w)
        (KeyCode::Char('w'), KeyModifiers::CONTROL) => {
            app.jump_transcript_to_bottom();
            app.reset_composer_history_navigation();
            app.composer.delete_word_left();
            app.update_autocomplete();
        }
        // Ctrl+A  →  start of line  (emacs/readline C-a)
        // Handled in the global match above when focus == Composer.
        // Ctrl+E  →  end of line  (emacs/readline C-e)
        (KeyCode::Char('e'), KeyModifiers::CONTROL) => {
            app.jump_transcript_to_bottom();
            app.composer.move_end();
        }

        // Generic printable character — exclude CONTROL and ALT so modifier
        // combos don't fall through and type raw letters.
        (KeyCode::Char(ch), modifiers)
            if !modifiers.contains(KeyModifiers::CONTROL)
                && !modifiers.contains(KeyModifiers::ALT) =>
        {
            app.jump_transcript_to_bottom();
            app.reset_composer_history_navigation();
            app.composer.insert_char(ch);
            app.update_autocomplete();
        }
        (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
            app.jump_transcript_to_bottom();
            app.reset_composer_history_navigation();
            app.composer.delete_to_line_start();
            app.update_autocomplete();
        }
        (KeyCode::Backspace, _) => {
            app.jump_transcript_to_bottom();
            app.reset_composer_history_navigation();
            app.composer.backspace();
            app.update_autocomplete();
        }
        (KeyCode::Delete, _) => {
            app.jump_transcript_to_bottom();
            app.reset_composer_history_navigation();
            app.composer.delete();
            app.update_autocomplete();
        }
        // Ctrl+Left / Ctrl+Right or Alt+Left / Alt+Right  →  word jump
        // (covers terminals that send Alt+Arrow directly rather than Esc+b/f)
        (KeyCode::Left, m)
            if m.contains(KeyModifiers::CONTROL) || m.contains(KeyModifiers::ALT) =>
        {
            app.jump_transcript_to_bottom();
            app.composer.move_word_left();
        }
        (KeyCode::Right, m)
            if m.contains(KeyModifiers::CONTROL) || m.contains(KeyModifiers::ALT) =>
        {
            app.jump_transcript_to_bottom();
            app.composer.move_word_right();
        }
        (KeyCode::Left, _) => {
            app.jump_transcript_to_bottom();
            app.composer.move_left();
        }
        (KeyCode::Right, _) => {
            app.jump_transcript_to_bottom();
            app.composer.move_right();
        }
        (KeyCode::Up, _) => {
            if app.autocomplete.is_active() {
                app.autocomplete_select_prev();
            } else if app.composer_history_active()
                || app.composer.is_empty()
                || composer_cursor_on_first_visual_row(app)
            {
                app.jump_transcript_to_bottom();
                app.composer_history_prev();
            } else {
                app.jump_transcript_to_bottom();
                let (first_width, continuation_width) = app.composer_wrap_widths();
                app.composer
                    .move_visual_up(first_width.max(1), continuation_width.max(1));
            }
        }
        (KeyCode::Down, _) => {
            if app.autocomplete.is_active() {
                app.autocomplete_select_next();
            } else if app.composer_history_active() {
                app.jump_transcript_to_bottom();
                app.composer_history_next();
            } else {
                app.jump_transcript_to_bottom();
                let (first_width, continuation_width) = app.composer_wrap_widths();
                app.composer
                    .move_visual_down(first_width.max(1), continuation_width.max(1));
            }
        }
        (KeyCode::Home, _) => {
            app.jump_transcript_to_bottom();
            app.composer.move_home();
        }
        (KeyCode::End, _) => {
            app.jump_transcript_to_bottom();
            app.composer.move_end();
        }
        (KeyCode::Esc, _) => {
            app.autocomplete.clear();
            app.reset_composer_history_navigation();
            if !app.composer.is_empty() {
                app.jump_transcript_to_bottom();
                app.composer = super::types::ComposerState::default();
            }
        }
        _ => {}
    }
    Ok(())
}

fn composer_cursor_on_first_visual_row(app: &InteractiveApp) -> bool {
    let (first_width, continuation_width) = app.composer_wrap_widths();
    let (row, _) = app
        .composer
        .visual_cursor_row_col(first_width.max(1), continuation_width.max(1));
    row == 0
}
