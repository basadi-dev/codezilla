pub mod activity;
pub mod app;
pub mod approval;
pub mod autocomplete;
pub mod composer_history;
pub mod input;
pub mod markdown;
pub mod render;
pub mod selection;
pub mod threads;
pub mod transcript_view;
pub mod types;

use anyhow::Result;
use crossterm::{
    cursor::{Hide, Show},
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, MouseButton, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{
    io,
    time::{Duration, Instant},
};
use tokio::sync::mpsc::error::TryRecvError;
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
            app.activity.tick();
            app.update_working_entry();
            let _ = app.refresh_live_sub_agent_sections();
            dirty = true;
        }

        // Poll background compaction task (non-blocking try_recv under the hood).
        if app.pending_compact.is_some() && app.poll_compact_result().await? {
            dirty = true;
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
                Err(TryRecvError::Disconnected) => {
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
                    // Helper: is the mouse inside the composer area?
                    let in_composer = {
                        let a = app.composer_area;
                        a.width > 0
                            && a.height > 0
                            && mouse.column >= a.x
                            && mouse.row >= a.y
                            && mouse.column < a.x + a.width
                            && mouse.row < a.y + a.height
                    };
                    // Helper: is the mouse inside the status bar area?
                    let in_status_bar = {
                        let a = app.status_bar_area;
                        a.width > 0
                            && a.height > 0
                            && mouse.column >= a.x
                            && mouse.row >= a.y
                            && mouse.column < a.x + a.width
                            && mouse.row < a.y + a.height
                    };

                    // When the user clicks in the status bar, temporarily
                    // disable mouse capture so the terminal's native
                    // drag-to-select can work on the status bar text.
                    // Re-enable on mouse-up.
                    if in_status_bar {
                        match mouse.kind {
                            MouseEventKind::Down(MouseButton::Left) => {
                                app.mouse_capture_enabled = false;
                                dirty = true;
                            }
                            MouseEventKind::Up(MouseButton::Left) => {
                                app.mouse_capture_enabled = true;
                                dirty = true;
                            }
                            _ => {}
                        }
                        // Don't process status bar clicks as transcript drags.
                        continue;
                    }

                    match mouse.kind {
                        // ── Wheel scroll ─────────────────────────────────────
                        MouseEventKind::ScrollUp => {
                            if in_composer {
                                app.composer_scroll(-1);
                            } else {
                                app.scroll_transcript(-3);
                            }
                            dirty = true;
                        }
                        MouseEventKind::ScrollDown => {
                            if in_composer {
                                app.composer_scroll(1);
                            } else {
                                app.scroll_transcript(3);
                            }
                            dirty = true;
                        }
                        // ── Drag-to-select (left button) ─────────────────────
                        MouseEventKind::Down(MouseButton::Left) => {
                            if in_composer {
                                app.begin_composer_drag(mouse.column, mouse.row);
                            } else {
                                app.begin_transcript_drag(mouse.column, mouse.row);
                            }
                            dirty = true;
                        }
                        MouseEventKind::Drag(MouseButton::Left) => {
                            if app.composer_selection.is_active() {
                                app.update_composer_drag(mouse.column, mouse.row);
                            } else {
                                app.update_transcript_drag(mouse.column, mouse.row);
                            }
                            dirty = true;
                        }
                        MouseEventKind::Up(MouseButton::Left) => {
                            if app.composer_selection.is_active() {
                                app.finish_composer_drag(mouse.column, mouse.row);
                            } else {
                                app.finish_transcript_drag(mouse.column, mouse.row);
                            }
                            dirty = true;
                        }
                        _ => {}
                    }
                }
                Event::Paste(text) => {
                    if !app.approval.has_pending() && app.focus == FocusPane::Composer {
                        app.jump_transcript_to_bottom();
                        // Normalize line endings: convert \r\n and \r to \n so the
                        // composer counts and renders lines consistently.
                        let text = text.replace("\r\n", "\n").replace('\r', "\n");
                        // Strip trailing newlines to prevent accidental submission
                        // when copying from code blocks or terminal output.
                        let text = text.trim_end_matches('\n');

                        // Priority 1: dragged file path (quoted / escaped / file://).
                        if let Some((path, mime)) = detect_dragged_image_path(text) {
                            if !app.current_model_supports_vision() {
                                let ms = app.effective_model_settings();
                                app.error_message = Some(format!(
                                    "This model ({}) does not support vision (image) input",
                                    ms.model_id
                                ));
                            } else {
                                let fname = std::path::Path::new(&path)
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string();
                                tracing::info!(path = %path, mime = %mime, "tui: attaching dragged image");
                                app.composer.add_attachment(path, mime.to_string());
                                app.status_message = format!("Attached image: {}", fname);
                                app.error_message = None;
                            }
                        // Priority 2: empty/whitespace bracketed paste — likely Cmd+V on
                        // macOS with image content in the clipboard. The terminal can't
                        // serialize image bytes into a paste event, so it sends nothing
                        // useful. Pull the image out of arboard ourselves.
                        } else if text.trim().is_empty() {
                            tracing::info!(
                                len = text.len(),
                                "tui: empty Event::Paste received — falling back to arboard image read"
                            );
                            // try_attach_clipboard_image always sets a status message,
                            // success or failure, so the user sees what happened.
                            try_attach_clipboard_image(&mut app);
                        } else {
                            app.composer.insert_str(text);
                        }
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

/// Try to attach an image from the system clipboard via `arboard`. Returns
/// `true` on success. Used as a fallback when `Event::Paste` arrives empty
/// (typical for Cmd+V on macOS terminals when the clipboard holds image data
fn try_attach_clipboard_image(app: &mut InteractiveApp) -> bool {
    if !app.current_model_supports_vision() {
        let ms = app.effective_model_settings();
        app.error_message = Some(format!(
            "This model ({}) does not support vision (image) input",
            ms.model_id
        ));
        return false;
    }
    let mut clipboard = match arboard::Clipboard::new() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "clipboard: failed to open system clipboard");
            app.status_message = format!("Clipboard open failed: {e}");
            return false;
        }
    };
    let img_data = match clipboard.get_image() {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "clipboard: get_image returned no image (clipboard may hold text-only or an unsupported format)"
            );
            app.status_message = format!("No image in clipboard ({e})");
            return false;
        }
    };
    tracing::info!(
        width = img_data.width,
        height = img_data.height,
        bytes = img_data.bytes.len(),
        "clipboard: read image data"
    );

    let dir = std::env::temp_dir().join("codezilla_pasted_images");
    let _ = std::fs::create_dir_all(&dir);
    let filename = format!("paste_{}.png", uuid::Uuid::new_v4().simple());
    let path = dir.join(&filename);

    let img = match image::RgbaImage::from_raw(
        img_data.width as u32,
        img_data.height as u32,
        img_data.bytes.as_ref().to_vec(),
    ) {
        Some(i) => i,
        None => {
            tracing::warn!(
                width = img_data.width,
                height = img_data.height,
                bytes = img_data.bytes.len(),
                "clipboard: RgbaImage::from_raw rejected the buffer (size mismatch?)"
            );
            app.status_message = "Failed to decode clipboard image (buffer size mismatch)".into();
            return false;
        }
    };
    if let Err(e) = img.save(&path) {
        tracing::warn!(error = %e, "clipboard: failed to write temp PNG");
        app.status_message = format!("Failed to save pasted image: {e}");
        return false;
    }
    let path_str = path.to_string_lossy().to_string();
    tracing::info!(path = %path_str, "tui: attaching pasted clipboard image");
    app.composer
        .add_attachment(path_str, "image/png".to_string());
    app.status_message = format!("Attached image: {}", filename);
    app.error_message = None;
    true
}

/// Try to interpret pasted text as a draggable image file path.
pub(super) fn detect_dragged_image_path(raw: &str) -> Option<(String, &'static str)> {
    let candidate = clean_pasted_path(raw);
    if candidate.is_empty() || candidate.contains('\n') {
        return None;
    }
    let lower = candidate.to_lowercase();
    let mime = if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else if lower.ends_with(".bmp") {
        "image/bmp"
    } else if lower.ends_with(".tiff") || lower.ends_with(".tif") {
        "image/tiff"
    } else if lower.ends_with(".ico") {
        "image/x-icon"
    } else {
        return None;
    };
    // Require the file to actually exist — otherwise we'd silently attach a
    // bogus path and the LLM would receive text-only with no image.
    if !std::path::Path::new(&candidate).is_file() {
        tracing::warn!(
            path = %candidate,
            "tui: dragged image path does not exist on disk; pasting as text instead"
        );
        return None;
    }
    Some((candidate, mime))
}

/// Normalize a pasted file path: strip surrounding quotes, decode `file://`
/// URIs, percent-decode, unescape backslash-escaped characters, and expand
/// a leading `~`. Returns the cleaned path string.
fn clean_pasted_path(raw: &str) -> String {
    let mut s = raw.trim().to_string();

    // Strip matching surrounding quotes (single or double).
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        s = s[1..s.len() - 1].to_string();
    }

    // Strip file:// URI prefix. RFC 8089 allows "file:" or "file:///"
    // (third slash is the absolute-path separator).
    if let Some(rest) = s.strip_prefix("file://") {
        s = rest.to_string();
    } else if let Some(rest) = s.strip_prefix("file:") {
        s = rest.to_string();
    }

    // Percent-decode (e.g. %20 → space). Keep it simple — only decode
    // valid two-hex-digit escapes; leave malformed ones as-is.
    s = percent_decode(&s);

    // Unescape backslash-escaped characters. macOS terminals escape spaces
    // and shell metacharacters when dragging files: `/foo\ bar.png`.
    s = unescape_backslashes(&s);

    // Expand leading `~` to $HOME.
    if let Some(rest) = s.strip_prefix('~') {
        if rest.is_empty() || rest.starts_with('/') {
            if let Ok(home) = std::env::var("HOME") {
                s = format!("{home}{rest}");
            }
        }
    }

    s
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

fn unescape_backslashes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
                continue;
            }
        }
        out.push(c);
    }
    out
}
