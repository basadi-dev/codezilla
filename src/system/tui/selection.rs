//! Drag-selection reducers — transcript and composer.
//!
//! Both panes implement the same behaviour: mouse-down starts a drag, motion
//! extends it, mouse-up either copies a non-empty selection or clears.
//! Double-click within 400 ms at the same coords selects the whole line/word.
//! The two reducers carry the same shape but differ in what a "point" is —
//! `SelectionPoint` (line + col) for the transcript, `usize` (char index)
//! for the composer.
//!
//! What lives here: the drag start/end pair, the locked flag, and the
//! click-history fields used to detect double-clicks. What stays on
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
    }

    /// Lock the selection to the supplied range (used by double-click to
    /// pin a whole-line span). Subsequent drag motion is ignored until the
    /// next mouse-down.
    pub fn lock(&mut self, start: SelectionPoint, end: SelectionPoint) {
        self.drag_start = Some(start);
        self.drag_end = Some(end);
        self.locked = true;
    }

    /// Update the drag end point. No-op when the selection is locked or
    /// when no drag is in progress.
    pub fn update_end(&mut self, point: SelectionPoint) {
        if self.locked || self.drag_start.is_none() {
            return;
        }
        self.drag_end = Some(point);
    }

    /// Clear the active selection (drag points + lock flag). Click history
    /// is preserved so a follow-up click still counts as a double-click.
    pub fn clear(&mut self) {
        self.drag_start = None;
        self.drag_end = None;
        self.locked = false;
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

    pub fn is_moved(&self) -> bool {
        self.drag_start.is_some() && self.drag_start != self.drag_end
    }

    pub fn start(&mut self, char_index: usize) {
        self.drag_start = Some(char_index);
        self.drag_end = Some(char_index);
        self.locked = false;
    }

    pub fn lock(&mut self, start: usize, end: usize) {
        self.drag_start = Some(start);
        self.drag_end = Some(end);
        self.locked = true;
    }

    pub fn update_end(&mut self, char_index: usize) {
        if self.locked || self.drag_start.is_none() {
            return;
        }
        self.drag_end = Some(char_index);
    }

    pub fn clear(&mut self) {
        self.drag_start = None;
        self.drag_end = None;
        self.locked = false;
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
        assert!(!s.is_moved());
        assert!(s.drag_start().is_none());
    }

    #[test]
    fn transcript_start_makes_collapsed_selection() {
        let mut s = TranscriptSelectionState::new();
        s.start(pt(2, 3));
        assert!(s.is_active());
        assert!(!s.is_locked());
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
    fn transcript_clear_resets_drag_but_preserves_click_history() {
        let mut s = TranscriptSelectionState::new();
        let now = Instant::now();
        s.record_click(now, 4, 7);
        s.start(pt(2, 0));
        s.clear();
        assert!(!s.is_active());
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
}
