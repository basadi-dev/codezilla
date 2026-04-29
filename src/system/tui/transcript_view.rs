//! Transcript scroll cursor + auto-follow flag.
//!
//! This is the smallest cohesive piece of transcript state — it does not own
//! the transcript itself (entries live on `InteractiveApp::transcript`) nor
//! the render cache, only the user's "where am I looking" cursor and whether
//! the view should snap to the bottom on new content.
//!
//! The renderer calls [`settle_at`] each frame with the maximum scroll
//! position; if the user has scrolled exactly to the bottom, auto-follow
//! re-engages so subsequent streaming text stays visible. That bit of
//! "looks-stateful-but-belongs-to-the-renderer" logic used to be inlined
//! in `render.rs`; pulling it into the reducer makes the invariant explicit
//! and gives it test coverage.

#[derive(Debug)]
pub struct TranscriptViewState {
    scroll: u16,
    auto_scroll: bool,
}

impl Default for TranscriptViewState {
    fn default() -> Self {
        Self {
            scroll: 0,
            auto_scroll: true,
        }
    }
}

impl TranscriptViewState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn scroll(&self) -> u16 {
        self.scroll
    }

    #[allow(dead_code)] // public API surface; covered by tests, no live consumer yet
    pub fn auto_scroll(&self) -> bool {
        self.auto_scroll
    }

    #[allow(dead_code)] // public API surface; covered by tests, no live consumer yet
    pub fn set_scroll(&mut self, value: u16) {
        self.scroll = value;
    }

    pub fn set_auto_scroll(&mut self, value: bool) {
        self.auto_scroll = value;
    }

    /// Apply a relative scroll delta. Negative deltas detach from the bottom
    /// (auto-scroll off); positive deltas move forward but do *not* re-engage
    /// auto-scroll on their own — that happens in [`settle_at`] when the
    /// renderer sees we hit the maximum.
    pub fn scroll_by(&mut self, delta: i32) {
        if delta < 0 {
            self.auto_scroll = false;
            self.scroll = self.scroll.saturating_sub(delta.unsigned_abs() as u16);
        } else {
            self.scroll = self.scroll.saturating_add(delta as u16);
        }
    }

    /// Re-engage auto-scroll. Equivalent to "press End": the next render will
    /// pin the view to the bottom regardless of the current scroll value.
    pub fn jump_to_bottom(&mut self) {
        self.auto_scroll = true;
    }

    /// Detach from the bottom and rewind to the top.
    pub fn jump_to_top(&mut self) {
        self.auto_scroll = false;
        self.scroll = 0;
    }

    /// Renderer hook: called every frame with the current maximum scroll
    /// (transcript height − viewport height). Two effects:
    ///   1. If auto-scroll is on, snap to the bottom.
    ///   2. If the user has manually scrolled all the way down, re-engage
    ///      auto-scroll so streaming content keeps following.
    ///
    /// Returns the effective scroll position to use for this frame.
    pub fn settle_at(&mut self, max_scroll: u16) -> u16 {
        if self.auto_scroll || self.scroll >= max_scroll {
            self.scroll = max_scroll;
            self.auto_scroll = true;
        }
        self.scroll
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_top_with_autoscroll_on() {
        let s = TranscriptViewState::new();
        assert_eq!(s.scroll(), 0);
        assert!(s.auto_scroll());
    }

    #[test]
    fn scroll_up_detaches_autoscroll() {
        let mut s = TranscriptViewState::new();
        s.set_scroll(10); // emulate sitting somewhere in the middle
        s.scroll_by(-3);
        assert_eq!(s.scroll(), 7);
        assert!(!s.auto_scroll(), "scrolling up should detach from bottom");
    }

    #[test]
    fn scroll_down_does_not_re_engage_autoscroll_on_its_own() {
        let mut s = TranscriptViewState::new();
        s.scroll_by(-5);
        assert!(!s.auto_scroll());
        s.scroll_by(2);
        assert!(
            !s.auto_scroll(),
            "settle_at, not scroll_by, owns auto-scroll re-engagement"
        );
    }

    #[test]
    fn settle_at_max_pins_and_engages_autoscroll() {
        let mut s = TranscriptViewState::new();
        s.scroll_by(-10);
        s.set_scroll(20); // user is exactly at max
        let used = s.settle_at(20);
        assert_eq!(used, 20);
        assert_eq!(s.scroll(), 20);
        assert!(s.auto_scroll(), "reaching max should re-engage autoscroll");
    }

    #[test]
    fn settle_at_with_autoscroll_snaps_even_when_below_max() {
        let mut s = TranscriptViewState::new();
        s.set_scroll(3);
        // auto_scroll is still true → renderer should snap to max regardless
        let used = s.settle_at(15);
        assert_eq!(used, 15);
        assert_eq!(s.scroll(), 15);
        assert!(s.auto_scroll());
    }

    #[test]
    fn settle_at_below_max_keeps_user_position_when_detached() {
        let mut s = TranscriptViewState::new();
        s.scroll_by(-5);
        s.set_scroll(7);
        let used = s.settle_at(20);
        assert_eq!(used, 7, "should respect detached position");
        assert!(!s.auto_scroll());
    }

    #[test]
    fn jump_to_bottom_engages_autoscroll() {
        let mut s = TranscriptViewState::new();
        s.scroll_by(-5);
        s.jump_to_bottom();
        assert!(s.auto_scroll());
    }

    #[test]
    fn jump_to_top_detaches_and_rewinds() {
        let mut s = TranscriptViewState::new();
        s.set_scroll(50);
        s.jump_to_top();
        assert_eq!(s.scroll(), 0);
        assert!(!s.auto_scroll());
    }

    #[test]
    fn scroll_up_saturates_at_zero() {
        let mut s = TranscriptViewState::new();
        s.scroll_by(-1000);
        assert_eq!(s.scroll(), 0);
    }
}
