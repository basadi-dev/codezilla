//! Drag-selection reducers — transcript and composer.
//!
//! Both panes implement the same behaviour: mouse-down starts a drag, motion
//! extends it, mouse-up either copies a non-empty selection or clears.
//! Double-click within 400 ms at the same coords selects the word under the
//! cursor and enters word-snap mode: subsequent drag motion snaps the endpoint
//! to word boundaries so the selection extends word-by-word.
//!
//! The two reducers carry the same shape but differ in what a "point" is —
//! `SelectionPoint` (line + col) for the transcript, `usize` (char index)
//! for the composer.
//!
//! What lives here: the drag start/end pair, the locked/word_snap flags, and
//! the click-history fields used to detect double-clicks. What stays on
//! `InteractiveApp`: the actual `mouse_to_selection_point` mapping and the
//! clipboard copy — those need access to the rendered transcript / composer
//! text, which the reducer deliberately doesn't own.
//!
//! A few accessors (`drag_end`, composer `is_moved`) are not yet read in
//! production but are part of the deliberate API surface and covered by
//! tests; the module-scope allow keeps them tidy.
#![allow(dead_code)]

use std::time::{Duration, Instant};

use super::types::SelectionPoint;

/// Time window within which a second click counts as a double-click.
const DOUBLE_CLICK_WINDOW: Duration = Duration::from_millis(400);

// ─── TranscriptSelectionState ────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct TranscriptSelectionState {
    drag_start: Option<SelectionPoint>,
    drag_end: Option<SelectionPoint>,
    /// When set (e.g. after a double-click), drag updates are ignored so the
    /// already-locked range survives mouse motion.
    locked: bool,
    /// When set (after a double-click selects a word), subsequent drag updates
    /// snap the endpoint to word boundaries so the selection extends
    /// word-by-word instead of character-by-character.
    word_snap: bool,
    /// The anchor point of the initial word selection (the opposite end from
    /// `drag_start`). When dragging backward past the anchor, `drag_start`
    /// is moved to the anchor so the original word stays selected.
    word_snap_anchor: Option<SelectionPoint>,
    last_click: Option<Instant>,
    last_click_col: u16,
    last_click_row: u16,
}

impl TranscriptSelectionState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn drag_start(&self) -> Option<SelectionPoint> {
        self.drag_start
    }

    pub fn drag_end(&self) -> Option<SelectionPoint> {
        self.drag_end
    }

    pub fn is_active(&self) -> bool {
        self.drag_start.is_some()
    }

    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// True when the selection was initiated by a double-click and subsequent
    /// drag motion should snap to word boundaries.
    pub fn word_snap(&self) -> bool {
        self.word_snap
    }

    /// Has the drag moved away from its starting point?
    pub fn is_moved(&self) -> bool {
        self.drag_start.is_some() && self.drag_start != self.drag_end
    }

    /// Begin a fresh drag at `point`. The drag is unlocked — subsequent
    /// motion updates can extend `drag_end`.
    pub fn start(&mut self, point: SelectionPoint) {
        self.drag_start = Some(point);
        self.drag_end = Some(point);
        self.locked = false;
        self.word_snap = false;
        self.word_snap_anchor = None;
    }

    /// Lock the selection to the supplied range (used by double-click to
    /// pin a word span). Subsequent drag motion is ignored until the
    /// next mouse-down.
    pub fn lock(&mut self, start: SelectionPoint, end: SelectionPoint) {
        self.drag_start = Some(start);
        self.drag_end = Some(end);
        self.locked = true;
    }

    /// Enter word-snap mode: the selection is not locked (drag can extend it),
    /// but the endpoint snaps to word boundaries. The anchor stores the
    /// opposite end of the initial word so that dragging backward preserves
    /// the originally-selected word.
    pub fn start_word_snap(&mut self, start: SelectionPoint, end: SelectionPoint) {
        self.drag_start = Some(start);
        self.drag_end = Some(end);
        self.word_snap_anchor = Some(end);
        self.locked = false;
        self.word_snap = true;
    }

    /// Update the drag end point. No-op when the selection is locked or
    /// when no drag is in progress.
    pub fn update_end(&mut self, point: SelectionPoint) {
        if self.locked || self.drag_start.is_none() {
            return;
        }
        self.drag_end = Some(point);
    }

    /// Update the drag end point with word-snapping. When `word_snap` is set,
    /// the endpoint is snapped to the nearest word boundary using the provided
    /// text lines. When dragging backward (endpoint before anchor), the anchor
    /// becomes `drag_start` so the original word stays selected; when dragging
    /// forward, `drag_start` stays at the word start. When not in word-snap
    /// mode, behaves like `update_end`.
    pub fn update_end_word_snap(&mut self, point: SelectionPoint, lines: &[String]) {
        if self.locked || self.drag_start.is_none() {
            return;
        }
        if self.word_snap {
            let anchor = self
                .word_snap_anchor
                .unwrap_or_else(|| self.drag_start.unwrap());
            let snapped = if point < anchor {
                // Dragging backward: snap to word-start boundaries
                snap_point_to_word_start(point, lines)
            } else {
                // Dragging forward: snap to word-end boundaries (inclusive)
                snap_point_to_word(point, lines)
            };
            // When dragging backward, anchor becomes drag_start so the
            // original word stays selected.
            self.drag_start = Some(anchor);
            self.drag_end = Some(snapped);
        } else {
            self.drag_end = Some(point);
        }
    }

    /// Clear the active selection (drag points + lock/word_snap flags). Click
    /// history is preserved so a follow-up click still counts as a double-click.
    pub fn clear(&mut self) {
        self.drag_start = None;
        self.drag_end = None;
        self.locked = false;
        self.word_snap = false;
        self.word_snap_anchor = None;
    }

    /// Reset the click-history window. Use after a successful double-click
    /// so a triple-click doesn't extend the same range, and after pane
    /// switches so a click in another pane doesn't accidentally pair with
    /// this one.
    pub fn forget_click(&mut self) {
        self.last_click = None;
    }

    /// Returns true if a click at `(col, row)` at time `now` counts as a
    /// double-click against the previously-recorded click.
    pub fn is_double_click(&self, now: Instant, col: u16, row: u16) -> bool {
        self.last_click.is_some_and(|t| {
            now.duration_since(t) < DOUBLE_CLICK_WINDOW
                && col == self.last_click_col
                && row == self.last_click_row
        })
    }

    /// Record a click for double-click detection on the next mouse-down.
    pub fn record_click(&mut self, now: Instant, col: u16, row: u16) {
        self.last_click = Some(now);
        self.last_click_col = col;
        self.last_click_row = row;
    }
}

// ─── ComposerSelectionState ──────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct ComposerSelectionState {
    drag_start: Option<usize>,
    drag_end: Option<usize>,
    locked: bool,
    word_snap: bool,
    /// The anchor index of the initial word selection (the opposite end from
    /// `drag_start`). When dragging backward past the anchor, `drag_start`
    /// is moved to the anchor so the original word stays selected.
    word_snap_anchor: Option<usize>,
    last_click: Option<Instant>,
    last_click_col: u16,
    last_click_row: u16,
}

impl ComposerSelectionState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn drag_start(&self) -> Option<usize> {
        self.drag_start
    }

    pub fn drag_end(&self) -> Option<usize> {
        self.drag_end
    }

    pub fn is_active(&self) -> bool {
        self.drag_start.is_some()
    }

    pub fn is_locked(&self) -> bool {
        self.locked
    }

    pub fn word_snap(&self) -> bool {
        self.word_snap
    }

    pub fn is_moved(&self) -> bool {
        self.drag_start.is_some() && self.drag_start != self.drag_end
    }

    pub fn start(&mut self, char_index: usize) {
        self.drag_start = Some(char_index);
        self.drag_end = Some(char_index);
        self.locked = false;
        self.word_snap = false;
        self.word_snap_anchor = None;
    }

    pub fn lock(&mut self, start: usize, end: usize) {
        self.drag_start = Some(start);
        self.drag_end = Some(end);
        self.locked = true;
    }

    /// Enter word-snap mode: the selection is not locked (drag can extend it),
    /// but the endpoint snaps to word boundaries. The anchor stores the
    /// opposite end of the initial word so that dragging backward preserves
    /// the originally-selected word.
    pub fn start_word_snap(&mut self, start: usize, end: usize) {
        self.drag_start = Some(start);
        self.drag_end = Some(end);
        self.word_snap_anchor = Some(end);
        self.locked = false;
        self.word_snap = true;
    }

    pub fn update_end(&mut self, char_index: usize) {
        if self.locked || self.drag_start.is_none() {
            return;
        }
        self.drag_end = Some(char_index);
    }

    /// Update the drag end point with word-snapping. When `word_snap` is set,
    /// the endpoint is snapped to the nearest word boundary using the provided
    /// character slice. When dragging backward (endpoint before anchor), the
    /// anchor becomes `drag_start` so the original word stays selected; when
    /// dragging forward, `drag_start` stays at the word start. When not in
    /// word-snap mode, behaves like `update_end`.
    pub fn update_end_word_snap(&mut self, char_index: usize, chars: &[char]) {
        if self.locked || self.drag_start.is_none() {
            return;
        }
        if self.word_snap {
            let anchor = self
                .word_snap_anchor
                .unwrap_or_else(|| self.drag_start.unwrap());
            let snapped = if char_index < anchor {
                // Dragging backward: snap to word-start boundaries
                snap_char_to_word_start(char_index, chars)
            } else {
                // Dragging forward: snap to word-end boundaries (exclusive)
                snap_char_to_word(char_index, chars)
            };
            // Always set drag_start to the anchor so the original word stays
            // selected regardless of drag direction.
            self.drag_start = Some(anchor);
            self.drag_end = Some(snapped);
        } else {
            self.drag_end = Some(char_index);
        }
    }

    pub fn clear(&mut self) {
        self.drag_start = None;
        self.drag_end = None;
        self.locked = false;
        self.word_snap = false;
        self.word_snap_anchor = None;
    }

    pub fn forget_click(&mut self) {
        self.last_click = None;
    }

    pub fn is_double_click(&self, now: Instant, col: u16, row: u16) -> bool {
        self.last_click.is_some_and(|t| {
            now.duration_since(t) < DOUBLE_CLICK_WINDOW
                && col == self.last_click_col
                && row == self.last_click_row
        })
    }

    pub fn record_click(&mut self, now: Instant, col: u16, row: u16) {
        self.last_click = Some(now);
        self.last_click_col = col;
        self.last_click_row = row;
    }

    /// Sorted (start, end) range — useful when copying or for display where
    /// the selection might be backwards (drag right-to-left).
    pub fn ordered_range(&self) -> Option<(usize, usize)> {
        match (self.drag_start, self.drag_end) {
            (Some(a), Some(b)) if a <= b => Some((a, b)),
            (Some(a), Some(b)) => Some((b, a)),
            _ => None,
        }
    }
}

// ─── Word-snap helpers ──────────────────────────────���────────────────────────

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// Snap a char index to the nearest word-end boundary (for forward dragging).
/// Returns an **exclusive** index (one past the last word char) since composer
/// selections use Rust slice semantics `[lo..hi]`.
/// If the index is on a word char, extend to the end of that word.
/// If the index is on a non-word char, find the nearest word and snap to its
/// end; if no word is nearby, return the raw index (no snap).
fn snap_char_to_word(idx: usize, chars: &[char]) -> usize {
    if idx >= chars.len() {
        return chars.len();
    }
    if is_word_char(chars[idx]) {
        // Extend forward to the end of the word
        let mut end = idx;
        while end < chars.len() && is_word_char(chars[end]) {
            end += 1;
        }
        end
    } else {
        // Non-word char: look for the nearest word and snap to its end.
        // Search forward first, then backward.
        let mut forward = idx + 1;
        while forward < chars.len() && !is_word_char(chars[forward]) {
            forward += 1;
        }
        if forward < chars.len() {
            let mut end = forward;
            while end < chars.len() && is_word_char(chars[end]) {
                end += 1;
            }
            return end;
        }
        let mut backward = idx;
        while backward > 0 && !is_word_char(chars[backward - 1]) {
            backward -= 1;
        }
        if backward > 0 && is_word_char(chars[backward - 1]) {
            // Snap to end of the word before us
            let mut end = backward - 1;
            while end > 0 && is_word_char(chars[end - 1]) {
                end -= 1;
            }
            // end is now the start; find the actual end
            let mut word_end = backward;
            while word_end < chars.len() && is_word_char(chars[word_end]) {
                word_end += 1;
            }
            return word_end;
        }
        // No word found nearby — no snap
        idx
    }
}

/// Snap a char index to the nearest word-start boundary (for backward dragging).
/// If the index is on a word char, snap to the start of that word.
/// If the index is on a non-word char, find the nearest word and snap to its
/// start; if no word is nearby, return the raw index (no snap).
fn snap_char_to_word_start(idx: usize, chars: &[char]) -> usize {
    if idx >= chars.len() {
        return chars.len();
    }
    if is_word_char(chars[idx]) {
        // Snap backward to the start of the word
        let mut start = idx;
        while start > 0 && is_word_char(chars[start - 1]) {
            start -= 1;
        }
        start
    } else {
        // Non-word char: look for the nearest word and snap to its start.
        // Search backward first, then forward.
        if idx > 0 && is_word_char(chars[idx - 1]) {
            let mut start = idx - 1;
            while start > 0 && is_word_char(chars[start - 1]) {
                start -= 1;
            }
            return start;
        }
        let mut forward = idx + 1;
        while forward < chars.len() && !is_word_char(chars[forward]) {
            forward += 1;
        }
        if forward < chars.len() {
            return forward;
        }
        // No word found nearby — no snap
        idx
    }
}

/// Snap a `SelectionPoint` to the nearest word-end boundary (for forward dragging)
/// in the rendered transcript lines. Returns an **inclusive** column index
/// (the last char of the word), since `SelectionRange.end_col` is inclusive.
fn snap_point_to_word(point: SelectionPoint, lines: &[String]) -> SelectionPoint {
    let line_idx = point.line;
    if line_idx >= lines.len() {
        return SelectionPoint {
            line: lines.len().saturating_sub(1),
            col: lines
                .last()
                .map(|l| l.chars().count().saturating_sub(1))
                .unwrap_or(0),
        };
    }
    let line_str = &lines[line_idx];
    let line_chars: Vec<char> = line_str.chars().collect();
    let col = point.col.min(line_chars.len());

    if col >= line_chars.len() {
        return SelectionPoint {
            line: line_idx,
            col: line_chars.len().saturating_sub(1),
        };
    }

    if is_word_char(line_chars[col]) {
        // Extend forward to the last char of the word (inclusive end)
        let mut end = col;
        while end + 1 < line_chars.len() && is_word_char(line_chars[end + 1]) {
            end += 1;
        }
        SelectionPoint {
            line: line_idx,
            col: end,
        }
    } else {
        // Non-word char: find nearest word and snap to its last char (inclusive)
        // Search forward first
        let mut forward = col + 1;
        while forward < line_chars.len() && !is_word_char(line_chars[forward]) {
            forward += 1;
        }
        if forward < line_chars.len() {
            let mut end = forward;
            while end + 1 < line_chars.len() && is_word_char(line_chars[end + 1]) {
                end += 1;
            }
            return SelectionPoint {
                line: line_idx,
                col: end,
            };
        }
        // Search backward
        if col > 0 {
            let mut back = col;
            while back > 0 && !is_word_char(line_chars[back - 1]) {
                back -= 1;
            }
            if back > 0 && is_word_char(line_chars[back - 1]) {
                // Find the end of this word (inclusive)
                let mut end = back - 1;
                while end + 1 < line_chars.len() && is_word_char(line_chars[end + 1]) {
                    end += 1;
                }
                return SelectionPoint {
                    line: line_idx,
                    col: end,
                };
            }
        }
        // No word found — no snap
        SelectionPoint {
            line: line_idx,
            col,
        }
    }
}

/// Snap a `SelectionPoint` to the nearest word-start boundary (for backward dragging)
/// in the rendered transcript lines.
fn snap_point_to_word_start(point: SelectionPoint, lines: &[String]) -> SelectionPoint {
    let line_idx = point.line;
    if line_idx >= lines.len() {
        return SelectionPoint {
            line: lines.len().saturating_sub(1),
            col: 0,
        };
    }
    let line_str = &lines[line_idx];
    let line_chars: Vec<char> = line_str.chars().collect();
    let col = point.col.min(line_chars.len());

    if col >= line_chars.len() {
        return SelectionPoint {
            line: line_idx,
            col: line_chars.len(),
        };
    }

    if is_word_char(line_chars[col]) {
        // Snap backward to the start of the word
        let mut start = col;
        while start > 0 && is_word_char(line_chars[start - 1]) {
            start -= 1;
        }
        SelectionPoint {
            line: line_idx,
            col: start,
        }
    } else {
        // Non-word char: find nearest word and snap to its start
        // Search backward first
        if col > 0 && is_word_char(line_chars[col - 1]) {
            let mut start = col - 1;
            while start > 0 && is_word_char(line_chars[start - 1]) {
                start -= 1;
            }
            return SelectionPoint {
                line: line_idx,
                col: start,
            };
        }
        // Search forward
        let mut forward = col + 1;
        while forward < line_chars.len() && !is_word_char(line_chars[forward]) {
            forward += 1;
        }
        if forward < line_chars.len() {
            return SelectionPoint {
                line: line_idx,
                col: forward,
            };
        }
        // No word found — no snap
        SelectionPoint {
            line: line_idx,
            col,
        }
    }
}

/// Find the start of the word at or before the given char index in composer text.
/// Used to determine the word-start anchor when double-clicking.
fn word_start_at(idx: usize, chars: &[char]) -> usize {
    if idx >= chars.len() {
        return chars.len();
    }
    if is_word_char(chars[idx]) {
        let mut start = idx;
        while start > 0 && is_word_char(chars[start - 1]) {
            start -= 1;
        }
        start
    } else {
        idx
    }
}

/// Find the end of the word at or after the given char index in composer text.
/// Returns an **exclusive** index (one past the last word char) since composer
/// selections use Rust slice semantics.
fn word_end_at(idx: usize, chars: &[char]) -> usize {
    if idx >= chars.len() {
        return chars.len();
    }
    if is_word_char(chars[idx]) {
        let mut end = idx;
        while end < chars.len() && is_word_char(chars[end]) {
            end += 1;
        }
        end
    } else {
        idx + 1
    }
}

/// Find the start of the word at or before the given column in a transcript line.
fn word_start_col_at(col: usize, line: &[char]) -> usize {
    if col >= line.len() {
        return 0;
    }
    if is_word_char(line[col]) {
        let mut start = col;
        while start > 0 && is_word_char(line[start - 1]) {
            start -= 1;
        }
        start
    } else {
        col
    }
}

/// Find the end of the word at or after the given column in a transcript line.
/// Returns an **inclusive** column index (the last char of the word) since
/// `SelectionRange.end_col` is inclusive.
fn word_end_col_at(col: usize, line: &[char]) -> usize {
    if col >= line.len() {
        return line.len().saturating_sub(1);
    }
    if is_word_char(line[col]) {
        let mut end = col;
        while end + 1 < line.len() && is_word_char(line[end + 1]) {
            end += 1;
        }
        end
    } else {
        col
    }
}

/// Compute the word range (start_point, end_point) for a double-click at the
/// given point in the transcript rendered lines. The end column is **inclusive**
/// (matching `SelectionRange.end_col` semantics).
pub fn transcript_word_range_at(
    point: SelectionPoint,
    lines: &[String],
) -> (SelectionPoint, SelectionPoint) {
    if point.line >= lines.len() {
        let last = lines.len().saturating_sub(1);
        let col_max = lines
            .last()
            .map(|l| l.chars().count().saturating_sub(1))
            .unwrap_or(0);
        return (
            SelectionPoint { line: last, col: 0 },
            SelectionPoint {
                line: last,
                col: col_max,
            },
        );
    }
    let line_chars: Vec<char> = lines[point.line].chars().collect();
    let col = point.col.min(line_chars.len().saturating_sub(1));

    let start_col = word_start_col_at(col, &line_chars);
    let end_col = word_end_col_at(col, &line_chars);

    (
        SelectionPoint {
            line: point.line,
            col: start_col,
        },
        SelectionPoint {
            line: point.line,
            col: end_col,
        },
    )
}

/// Compute the word range (start_idx, end_idx) for a double-click at the
/// given char index in the composer text. The end index is **exclusive**
/// (one past the last word char) since composer selections use Rust slice
/// semantics `[lo..hi]`.
pub fn composer_word_range_at(idx: usize, chars: &[char]) -> (usize, usize) {
    let idx = idx.min(chars.len().saturating_sub(1));
    (word_start_at(idx, chars), word_end_at(idx, chars))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(line: usize, col: usize) -> SelectionPoint {
        SelectionPoint { line, col }
    }

    // ── Transcript ───────────────────────────────────────────────────────────

    #[test]
    fn transcript_default_is_inactive() {
        let s = TranscriptSelectionState::new();
        assert!(!s.is_active());
        assert!(!s.is_locked());
        assert!(!s.word_snap());
        assert!(!s.is_moved());
        assert!(s.drag_start().is_none());
    }

    #[test]
    fn transcript_start_makes_collapsed_selection() {
        let mut s = TranscriptSelectionState::new();
        s.start(pt(2, 3));
        assert!(s.is_active());
        assert!(!s.is_locked());
        assert!(!s.word_snap());
        assert!(!s.is_moved(), "fresh drag is collapsed at start");
    }

    #[test]
    fn transcript_update_end_extends_when_unlocked() {
        let mut s = TranscriptSelectionState::new();
        s.start(pt(2, 3));
        s.update_end(pt(2, 10));
        assert_eq!(s.drag_end(), Some(pt(2, 10)));
        assert!(s.is_moved());
    }

    #[test]
    fn transcript_lock_freezes_endpoint() {
        let mut s = TranscriptSelectionState::new();
        s.lock(pt(2, 0), pt(2, usize::MAX / 2));
        assert!(s.is_locked());
        s.update_end(pt(2, 5));
        assert_eq!(
            s.drag_end(),
            Some(pt(2, usize::MAX / 2)),
            "locked range should not move"
        );
    }

    #[test]
    fn transcript_word_snap_extends_on_drag() {
        let mut s = TranscriptSelectionState::new();
        // Double-click "hello" (cols 0–4 inclusive)
        s.start_word_snap(pt(0, 0), pt(0, 4));
        assert!(!s.is_locked());
        assert!(s.word_snap());
        // Drag forward into "world" — should snap to last char of "world" (col 10)
        let lines = vec!["hello world".to_string()];
        s.update_end_word_snap(pt(0, 9), &lines);
        assert_eq!(s.drag_end(), Some(pt(0, 10)));
        // drag_start should be the anchor (word end = col 4)
        assert_eq!(s.drag_start(), Some(pt(0, 4)));
    }

    #[test]
    fn transcript_word_snap_backward_drag_preserves_original_word() {
        let mut s = TranscriptSelectionState::new();
        // Double-click "world" (cols 6–10 inclusive)
        s.start_word_snap(pt(0, 6), pt(0, 10));
        let lines = vec!["hello world".to_string()];
        // Drag backward past "hello" — should snap to start of "hello" (col 0)
        s.update_end_word_snap(pt(0, 2), &lines);
        assert_eq!(s.drag_end(), Some(pt(0, 0)));
        // drag_start should be the anchor (word end = col 10) so the original
        // word stays selected: selection is [0..10] inclusive.
        assert_eq!(s.drag_start(), Some(pt(0, 10)));
    }

    #[test]
    fn transcript_word_snap_no_snap_when_not_set() {
        let mut s = TranscriptSelectionState::new();
        s.start(pt(0, 3));
        assert!(!s.word_snap());
        let lines = vec!["hello world".to_string()];
        s.update_end_word_snap(pt(0, 9), &lines);
        // Without word_snap, should pass through the raw point
        assert_eq!(s.drag_end(), Some(pt(0, 9)));
    }

    #[test]
    fn transcript_clear_resets_drag_but_preserves_click_history() {
        let mut s = TranscriptSelectionState::new();
        let now = Instant::now();
        s.record_click(now, 4, 7);
        s.start(pt(2, 0));
        s.clear();
        assert!(!s.is_active());
        assert!(!s.word_snap());
        assert!(s.is_double_click(now, 4, 7), "click history must persist");
    }

    #[test]
    fn transcript_double_click_window_is_400ms() {
        let mut s = TranscriptSelectionState::new();
        let t0 = Instant::now();
        s.record_click(t0, 5, 6);
        assert!(s.is_double_click(t0 + Duration::from_millis(399), 5, 6));
        assert!(!s.is_double_click(t0 + Duration::from_millis(401), 5, 6));
    }

    #[test]
    fn transcript_double_click_requires_same_coords() {
        let mut s = TranscriptSelectionState::new();
        let t0 = Instant::now();
        s.record_click(t0, 5, 6);
        assert!(!s.is_double_click(t0 + Duration::from_millis(50), 6, 6));
        assert!(!s.is_double_click(t0 + Duration::from_millis(50), 5, 7));
    }

    #[test]
    fn transcript_forget_click_breaks_double_click_pairing() {
        let mut s = TranscriptSelectionState::new();
        let t0 = Instant::now();
        s.record_click(t0, 5, 6);
        s.forget_click();
        assert!(!s.is_double_click(t0 + Duration::from_millis(10), 5, 6));
    }

    // ── Composer ─────────────────────────────────────────────────────────────

    #[test]
    fn composer_ordered_range_handles_backward_drag() {
        let mut s = ComposerSelectionState::new();
        s.start(10);
        s.update_end(3);
        assert_eq!(s.ordered_range(), Some((3, 10)));
    }

    #[test]
    fn composer_lock_and_update_end_no_op() {
        let mut s = ComposerSelectionState::new();
        s.lock(0, 8);
        s.update_end(5);
        assert_eq!(s.drag_end(), Some(8));
    }

    #[test]
    fn composer_clear_resets_drag_only() {
        let mut s = ComposerSelectionState::new();
        s.record_click(Instant::now(), 1, 2);
        s.start(4);
        s.clear();
        assert!(!s.is_active());
    }

    #[test]
    fn composer_no_range_when_inactive() {
        let s = ComposerSelectionState::new();
        assert_eq!(s.ordered_range(), None);
    }

    #[test]
    fn composer_word_snap_extends_on_drag() {
        let mut s = ComposerSelectionState::new();
        // Double-click "hello" (chars 0..5 exclusive)
        s.start_word_snap(0, 5);
        assert!(!s.is_locked());
        assert!(s.word_snap());
        // Drag forward into "world" — should snap to end of "world" (char 11 exclusive)
        let chars: Vec<char> = "hello world".chars().collect();
        s.update_end_word_snap(9, &chars);
        assert_eq!(s.drag_end(), Some(11));
        // drag_start should be the anchor (word end = 5)
        assert_eq!(s.drag_start(), Some(5));
    }

    #[test]
    fn composer_word_snap_backward_drag_preserves_original_word() {
        let mut s = ComposerSelectionState::new();
        // Double-click "world" (chars 6..11 exclusive)
        s.start_word_snap(6, 11);
        let chars: Vec<char> = "hello world".chars().collect();
        // Drag backward past "hello" — should snap to start of "hello" (char 0)
        s.update_end_word_snap(2, &chars);
        assert_eq!(s.drag_end(), Some(0));
        // drag_start should be the anchor (word end = 11) so the original
        // word stays selected: selection is [0..11] exclusive.
        assert_eq!(s.drag_start(), Some(11));
    }

    // ── Word-snap helpers ────────────────────────────────────────────────────

    #[test]
    fn snap_char_to_word_on_word() {
        let chars: Vec<char> = "hello world".chars().collect();
        // On 'l' in "hello" — should extend to end of "hello" (exclusive: 5)
        assert_eq!(snap_char_to_word(2, &chars), 5);
        // On 'o' in "hello" — should extend to end of "hello" (exclusive: 5)
        assert_eq!(snap_char_to_word(4, &chars), 5);
        // On 'w' in "world" — should extend to end of "world" (exclusive: 11)
        assert_eq!(snap_char_to_word(6, &chars), 11);
    }

    #[test]
    fn snap_char_to_word_on_space() {
        let chars: Vec<char> = "hello world".chars().collect();
        // On the space between words — should snap to end of "world" (exclusive: 11)
        assert_eq!(snap_char_to_word(5, &chars), 11);
    }

    #[test]
    fn snap_char_to_word_at_end() {
        let chars: Vec<char> = "hello".chars().collect();
        // Past the end — should return chars.len()
        assert_eq!(snap_char_to_word(5, &chars), 5);
        // On 'o' — should extend to end (exclusive: 5)
        assert_eq!(snap_char_to_word(4, &chars), 5);
    }

    #[test]
    fn snap_char_to_word_start_on_word() {
        let chars: Vec<char> = "hello world".chars().collect();
        // On 'l' in "hello" — should snap to start of "hello" (0)
        assert_eq!(snap_char_to_word_start(2, &chars), 0);
        // On 'o' in "world" — should snap to start of "world" (6)
        assert_eq!(snap_char_to_word_start(10, &chars), 6);
    }

    #[test]
    fn snap_char_to_word_start_on_space() {
        let chars: Vec<char> = "hello world".chars().collect();
        // On the space — should snap to start of "hello" (0)
        assert_eq!(snap_char_to_word_start(5, &chars), 0);
    }

    #[test]
    fn word_range_at_composer() {
        let chars: Vec<char> = "hello world".chars().collect();
        // Click on 'l' in "hello" (idx 2)
        let (lo, hi) = composer_word_range_at(1, &chars);
        assert_eq!(lo, 0);
        assert_eq!(hi, 5); // exclusive end

        // Click on 'o' in "world" (idx 10)
        let (lo, hi) = composer_word_range_at(8, &chars);
        assert_eq!(lo, 6);
        assert_eq!(hi, 11); // exclusive end
    }

    #[test]
    fn word_range_at_transcript() {
        let lines = vec!["hello world".to_string()];
        let point = SelectionPoint { line: 0, col: 2 };
        let (start, end) = transcript_word_range_at(point, &lines);
        assert_eq!(start, SelectionPoint { line: 0, col: 0 });
        // "hello" occupies cols 0–4, so inclusive end is col 4
        assert_eq!(end, SelectionPoint { line: 0, col: 4 });
    }
}
