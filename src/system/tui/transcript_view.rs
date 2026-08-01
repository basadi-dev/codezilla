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
