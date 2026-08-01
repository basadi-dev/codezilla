//! Thread-list reducer: cached thread metadata + selection cursor.
//!
//! The async fetch lives on `InteractiveApp::refresh_threads` (it needs the
//! runtime); this reducer just owns the resulting list and the selection
//! invariant. After the list is replaced, callers run
//! `reconcile_selection(fallback)` to make sure the cursor points at a thread
//! that still exists.

use crate::system::domain::ThreadMetadata;

#[derive(Debug, Default)]
pub struct ThreadListState {
    threads: Vec<ThreadMetadata>,
    selected: Option<String>,
}

impl ThreadListState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn threads(&self) -> &[ThreadMetadata] {
        &self.threads
    }

    #[allow(dead_code)] // public API surface; covered by tests, no live consumer yet
    pub fn selected_id(&self) -> Option<&str> {
        self.selected.as_deref()
    }

    pub fn set_selected(&mut self, id: Option<String>) {
        self.selected = id;
    }

    #[allow(dead_code)] // public API surface; covered by tests, no live consumer yet
    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }

    pub fn len(&self) -> usize {
        self.threads.len()
    }

    /// Replace the cached thread list. Callers should follow this with
    /// `reconcile_selection(...)` to fix up the cursor for the new list.
    pub fn set_threads(&mut self, threads: Vec<ThreadMetadata>) {
        self.threads = threads;
    }

    /// Returns true if the list contains a thread with the given id.
    pub fn contains(&self, id: &str) -> bool {
        self.threads.iter().any(|t| t.thread_id == id)
    }

    /// Reconcile the selection against the current list:
    ///   - If the list is empty, selection clears.
    ///   - If the current selection is still present, keep it.
    ///   - Otherwise fall back to `fallback` (typically the
    ///     `current_thread_id` the TUI is open on).
    ///
    /// Note: matches the prior in-place logic exactly, including the case
    /// where `fallback` itself isn't in the list — that's preserved as an
    /// observed invariant of the existing callers.
    pub fn reconcile_selection(&mut self, fallback: Option<String>) {
        if self.threads.is_empty() {
            self.selected = None;
            return;
        }
        let desired = self.selected.clone().or_else(|| fallback.clone());
        if let Some(id) = &desired {
            if self.contains(id) {
                self.selected = desired;
                return;
            }
        }
        self.selected = fallback;
    }
}
