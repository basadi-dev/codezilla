//! Composer history reducer — Up/Down arrows recall prior submitted prompts.
//!
//! Holds the recalled-prompts list, the cursor position within it, and the
//! "saved" current input so navigation can restore the user's in-progress
//! draft if they arrow away and back. The actual list comes from the
//! persisted thread (the parent supplies it via `replace_history(...)`).

#[derive(Debug, Default)]
pub struct ComposerHistoryState {
    /// Prior submitted prompts, oldest → newest.
    history: Vec<String>,
    /// Cursor into `history` while navigating; `None` while live editing.
    index: Option<usize>,
    /// User's draft at the moment they entered history navigation. Restored
    /// when they navigate past the newest entry.
    saved_input: Option<String>,
}

/// Outcome of an arrow-key navigation step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryNavigation {
    /// No-op: history is empty.
    Empty,
    /// Replace the composer text with the supplied string.
    Set(String),
    /// Restore the user's in-progress draft (they navigated past the newest
    /// entry). Empty string if no draft was captured.
    Restore(String),
}

impl ComposerHistoryState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn replace_history(&mut self, history: Vec<String>) {
        self.history = history;
        self.reset_navigation();
    }

    /// Append a freshly-submitted entry. Empty/whitespace inputs are dropped
    /// by the caller (matches the prior `push_composer_history_entry`
    /// guard); duplicates of the most recent entry are dropped here so the
    /// history doesn't fill with `↑↑↑` repeats.
    pub fn push(&mut self, entry: String) {
        if self.history.last().map(String::as_str) == Some(entry.as_str()) {
            return;
        }
        self.history.push(entry);
        self.reset_navigation();
    }

    pub fn is_active(&self) -> bool {
        self.index.is_some()
    }

    pub fn reset_navigation(&mut self) {
        self.index = None;
        self.saved_input = None;
    }

    /// Up-arrow: move toward older entries. Saves `current_input` on the
    /// first step so a later wrap-around can restore it.
    pub fn prev(&mut self, current_input: &str) -> HistoryNavigation {
        if self.history.is_empty() {
            return HistoryNavigation::Empty;
        }
        let next_index = match self.index {
            Some(i) => i.saturating_sub(1),
            None => {
                self.saved_input = Some(current_input.to_string());
                self.history.len() - 1
            }
        };
        self.index = Some(next_index);
        HistoryNavigation::Set(self.history[next_index].clone())
    }

    /// Down-arrow: move toward newer entries after history navigation has
    /// started. Past the newest entry restores the saved draft and exits
    /// navigation mode.
    pub fn next(&mut self, _current_input: &str) -> HistoryNavigation {
        if self.history.is_empty() {
            return HistoryNavigation::Empty;
        }
        let Some(index) = self.index else {
            return HistoryNavigation::Empty;
        };
        if index + 1 < self.history.len() {
            self.index = Some(index + 1);
            HistoryNavigation::Set(self.history[index + 1].clone())
        } else {
            let restored = self.saved_input.take().unwrap_or_default();
            self.index = None;
            HistoryNavigation::Restore(restored)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn empty_history_returns_empty_for_both_directions() {
        let mut s = ComposerHistoryState::new();
        assert_eq!(s.prev("draft"), HistoryNavigation::Empty);
        assert_eq!(s.next("draft"), HistoryNavigation::Empty);
        assert!(!s.is_active());
    }

    #[test]
    fn prev_starts_from_newest_then_walks_back() {
        let mut s = ComposerHistoryState::new();
        s.replace_history(h(&["one", "two", "three"]));
        assert_eq!(s.prev("draft"), HistoryNavigation::Set("three".into()));
        assert_eq!(s.prev(""), HistoryNavigation::Set("two".into()));
        assert_eq!(s.prev(""), HistoryNavigation::Set("one".into()));
        // Saturating: another prev stays at oldest.
        assert_eq!(s.prev(""), HistoryNavigation::Set("one".into()));
    }

    #[test]
    fn next_from_idle_is_noop() {
        let mut s = ComposerHistoryState::new();
        s.replace_history(h(&["one", "two", "three"]));
        assert_eq!(s.next("draft"), HistoryNavigation::Empty);
        assert!(!s.is_active());
    }

    #[test]
    fn next_past_newest_restores_saved_draft() {
        let mut s = ComposerHistoryState::new();
        s.replace_history(h(&["one", "two"]));
        s.prev("my draft"); // index=1 (newest), captures "my draft", Set("two")
        s.prev(""); // index=0, Set("one")
        s.next(""); // index=1, Set("two")
                    // One more next walks past the newest entry, restoring the saved draft.
        assert_eq!(s.next(""), HistoryNavigation::Restore("my draft".into()));
        assert!(!s.is_active(), "navigation mode should exit");
    }

    #[test]
    fn first_prev_captures_current_draft() {
        let mut s = ComposerHistoryState::new();
        s.replace_history(h(&["one"]));
        s.prev("hello"); // captures "hello" as saved, index=0, Set("one")
                         // Single-entry list → next from newest immediately restores.
        assert_eq!(s.next(""), HistoryNavigation::Restore("hello".into()));
    }

    #[test]
    fn replace_history_resets_navigation() {
        let mut s = ComposerHistoryState::new();
        s.replace_history(h(&["one"]));
        s.prev("");
        assert!(s.is_active());
        s.replace_history(h(&["alpha", "beta"]));
        assert!(!s.is_active());
        // Fresh navigation starts at the new newest.
        assert_eq!(s.prev(""), HistoryNavigation::Set("beta".into()));
    }

    #[test]
    fn reset_navigation_drops_saved_input_too() {
        let mut s = ComposerHistoryState::new();
        s.replace_history(h(&["one"]));
        s.prev("draft");
        s.reset_navigation();
        // After reset, navigation past the end has nothing to restore.
        s.prev("");
        assert_eq!(s.next(""), HistoryNavigation::Restore(String::new()));
    }
}
