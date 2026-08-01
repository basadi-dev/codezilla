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
