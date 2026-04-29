//! Slash-command autocomplete reducer — pure state, no rendering, no runtime.
//!
//! This is the first step of the Phase 6 TUI split: peel a self-contained
//! piece of UI state out of the `InteractiveApp` god-object and give it a
//! reducer-style API that can be unit-tested without spinning up a terminal.
//!
//! The reducer holds the current suggestion list, the selected index, and the
//! scroll offset. Mutations are limited to a small set of high-level
//! operations (`set_suggestions`, `select_next`, `select_prev`, `clear`),
//! which keeps the call sites in `app.rs` short and makes the invariants
//! (selection always in-range, scroll always covers the selection) easy to
//! enforce in one place.
//!
//! Computing the candidate list itself still lives in `app.rs` because it
//! needs runtime/config context (model presets, threads list, current model).
//! That responsibility can move out separately once we have a clean way to
//! express the inputs without coupling back to `InteractiveApp`.

use super::types::AutocompleteItem;

/// How many suggestions render at once. Used to clamp the scroll offset.
pub const AUTOCOMPLETE_VIEWPORT: usize = 8;

#[derive(Debug, Default)]
pub struct AutocompleteState {
    suggestions: Vec<AutocompleteItem>,
    selected: usize,
    scroll: usize,
}

impl AutocompleteState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn suggestions(&self) -> &[AutocompleteItem] {
        &self.suggestions
    }

    pub fn selected_index(&self) -> usize {
        self.selected
    }

    pub fn scroll_offset(&self) -> usize {
        self.scroll
    }

    pub fn is_active(&self) -> bool {
        !self.suggestions.is_empty()
    }

    /// Replace the suggestion list; reset selection and scroll to the top.
    pub fn set_suggestions(&mut self, suggestions: Vec<AutocompleteItem>) {
        self.suggestions = suggestions;
        self.selected = 0;
        self.scroll = 0;
    }

    /// Drop all suggestions and reset state.
    pub fn clear(&mut self) {
        self.suggestions.clear();
        self.selected = 0;
        self.scroll = 0;
    }

    /// Move the selection forward (wrapping). Returns the newly-selected
    /// item's `value`, which the caller typically writes into the composer.
    /// Returns `None` if the list is empty.
    pub fn select_next(&mut self) -> Option<String> {
        if self.suggestions.is_empty() {
            return None;
        }
        self.selected = (self.selected + 1) % self.suggestions.len();
        self.clamp_scroll();
        Some(self.suggestions[self.selected].value.clone())
    }

    /// Move the selection backward (wrapping). See `select_next`.
    pub fn select_prev(&mut self) -> Option<String> {
        if self.suggestions.is_empty() {
            return None;
        }
        let len = self.suggestions.len();
        self.selected = (self.selected + len - 1) % len;
        self.clamp_scroll();
        Some(self.suggestions[self.selected].value.clone())
    }

    /// The currently-selected item, if any.
    #[allow(dead_code)] // public API surface; covered by tests, no live consumer yet
    pub fn selected_item(&self) -> Option<&AutocompleteItem> {
        self.suggestions.get(self.selected)
    }

    fn clamp_scroll(&mut self) {
        let viewport = AUTOCOMPLETE_VIEWPORT.min(self.suggestions.len());
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if viewport > 0 && self.selected >= self.scroll + viewport {
            self.scroll = self.selected + 1 - viewport;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(n: usize) -> Vec<AutocompleteItem> {
        (0..n)
            .map(|i| AutocompleteItem::simple(format!("/cmd-{i}")))
            .collect()
    }

    #[test]
    fn empty_state_does_nothing() {
        let mut s = AutocompleteState::new();
        assert!(!s.is_active());
        assert_eq!(s.select_next(), None);
        assert_eq!(s.select_prev(), None);
        assert!(s.selected_item().is_none());
    }

    #[test]
    fn set_suggestions_resets_selection_and_scroll() {
        let mut s = AutocompleteState::new();
        s.set_suggestions(items(20));
        assert_eq!(s.selected_index(), 0);
        assert_eq!(s.scroll_offset(), 0);
        // Move selection deep into the list, then re-set; should reset.
        for _ in 0..15 {
            s.select_next();
        }
        assert!(s.selected_index() > 0);
        s.set_suggestions(items(3));
        assert_eq!(s.selected_index(), 0);
        assert_eq!(s.scroll_offset(), 0);
    }

    #[test]
    fn clear_drops_all_state() {
        let mut s = AutocompleteState::new();
        s.set_suggestions(items(5));
        s.select_next();
        s.clear();
        assert!(!s.is_active());
        assert_eq!(s.selected_index(), 0);
        assert_eq!(s.scroll_offset(), 0);
    }

    #[test]
    fn select_next_wraps_at_end() {
        let mut s = AutocompleteState::new();
        s.set_suggestions(items(3));
        assert_eq!(s.select_next().as_deref(), Some("/cmd-1"));
        assert_eq!(s.select_next().as_deref(), Some("/cmd-2"));
        assert_eq!(s.select_next().as_deref(), Some("/cmd-0"));
    }

    #[test]
    fn select_prev_wraps_at_start() {
        let mut s = AutocompleteState::new();
        s.set_suggestions(items(3));
        assert_eq!(s.select_prev().as_deref(), Some("/cmd-2"));
        assert_eq!(s.select_prev().as_deref(), Some("/cmd-1"));
        assert_eq!(s.select_prev().as_deref(), Some("/cmd-0"));
    }

    #[test]
    fn scroll_follows_selection_past_viewport() {
        let mut s = AutocompleteState::new();
        s.set_suggestions(items(20));
        // Stepping forward should keep scroll = 0 until we exceed the viewport.
        for i in 0..AUTOCOMPLETE_VIEWPORT - 1 {
            s.select_next();
            assert_eq!(s.scroll_offset(), 0, "step {i} should not scroll");
        }
        // The next step pushes selection past the viewport — scroll moves.
        s.select_next();
        assert_eq!(s.selected_index(), AUTOCOMPLETE_VIEWPORT);
        assert_eq!(s.scroll_offset(), 1);
    }

    #[test]
    fn scroll_follows_selection_when_paging_back() {
        let mut s = AutocompleteState::new();
        s.set_suggestions(items(20));
        // Wrap back from index 0 → 19, scroll should jump to keep it visible.
        s.select_prev();
        assert_eq!(s.selected_index(), 19);
        assert_eq!(s.scroll_offset(), 19 + 1 - AUTOCOMPLETE_VIEWPORT);
    }

    #[test]
    fn selected_item_tracks_selection() {
        let mut s = AutocompleteState::new();
        s.set_suggestions(items(3));
        assert_eq!(s.selected_item().map(|i| i.value.as_str()), Some("/cmd-0"));
        s.select_next();
        assert_eq!(s.selected_item().map(|i| i.value.as_str()), Some("/cmd-1"));
    }
}
