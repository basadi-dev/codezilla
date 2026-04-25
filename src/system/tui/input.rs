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
        (KeyCode::Char('a'), KeyModifiers::CONTROL) => {
            app.toggle_auto_approve_tools();
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
            app.auto_scroll = true;
        }
        // Ctrl+Home  →  jump to top
        (KeyCode::Home, KeyModifiers::CONTROL) => {
            app.auto_scroll = false;
            app.transcript_scroll = 0;
        }

        // Tab / Shift+Tab: autocomplete when suggestions are live, else switch pane
        (KeyCode::Tab, _) => {
            if !app.autocomplete_suggestions.is_empty() {
                app.autocomplete_select_next();
            } else {
                app.focus = match app.focus {
                    FocusPane::Composer => FocusPane::Transcript,
                    _ => FocusPane::Composer,
                };
            }
        }
        (KeyCode::BackTab, _) => {
            if !app.autocomplete_suggestions.is_empty() {
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
        (KeyCode::Char('a'), KeyModifiers::CONTROL) => {
            app.toggle_auto_approve_tools();
        }
        (KeyCode::Char('a') | KeyCode::Char('A'), _) => {
            app.resolve_pending_approval(ApprovalDecision::Approved)
                .await?;
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
        KeyCode::Home => {
            app.auto_scroll = false;
            app.transcript_scroll = 0;
        }
        KeyCode::End => app.auto_scroll = true,
        _ => {}
    }
}

async fn handle_composer_key(app: &mut InteractiveApp, key: KeyEvent) -> Result<()> {
    match (key.code, key.modifiers) {
        (KeyCode::Enter, modifiers) if modifiers.contains(KeyModifiers::SHIFT) => {
            app.jump_transcript_to_bottom();
            app.reset_composer_history_navigation();
            app.composer.insert_char('\n');
        }
        (KeyCode::Enter, _) => {
            app.jump_transcript_to_bottom();
            app.autocomplete_suggestions.clear();
            app.autocomplete_selected = 0;
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
        (KeyCode::Left, m) if m.contains(KeyModifiers::CONTROL) || m.contains(KeyModifiers::ALT) => {
            app.jump_transcript_to_bottom();
            app.composer.move_word_left();
        }
        (KeyCode::Right, m) if m.contains(KeyModifiers::CONTROL) || m.contains(KeyModifiers::ALT) => {
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
            if !app.autocomplete_suggestions.is_empty() {
                app.autocomplete_select_prev();
            } else if app.composer_history_active() || app.composer.is_empty() {
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
            if !app.autocomplete_suggestions.is_empty() {
                app.autocomplete_select_next();
            } else if app.composer_history_active() || app.composer.is_empty() {
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
            app.autocomplete_suggestions.clear();
            app.autocomplete_selected = 0;
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
