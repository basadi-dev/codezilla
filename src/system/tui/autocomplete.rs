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

// ─── Matching ─────────────────────────────────────────────────────────────────

/// Score a candidate label against a query. Higher = better. `None` = no match.
fn match_score(label: &str, query: &str) -> Option<u32> {
    let label_lower = label.to_lowercase();
    let query_lower = query.to_lowercase();

    // Exact match
    if label_lower == query_lower {
        return Some(1000);
    }
    // Prefix match
    if label_lower.starts_with(&query_lower) {
        return Some(900);
    }
    // Word-boundary prefix (e.g. "mod" matches "/model", "rea" matches "/reasoning")
    if let Some(pos) = label_lower.find(&query_lower) {
        // Earlier substring = higher score
        return Some(800u32.saturating_sub(pos as u32));
    }
    // Fuzzy: all query chars appear in order
    if fuzzy_match(&label_lower, &query_lower) {
        return Some(500);
    }
    None
}

/// True when every char in `query` appears in `label` in order (not necessarily contiguous).
fn fuzzy_match(label: &str, query: &str) -> bool {
    let mut chars = label.chars();
    for qc in query.chars() {
        loop {
            match chars.next() {
                Some(lc) if lc == qc => break,
                Some(_) => continue,
                None => return false,
            }
        }
    }
    true
}

/// Filter and rank candidates by query. Returns items sorted best-match-first.
pub fn filter_and_rank(candidates: Vec<AutocompleteItem>, query: &str) -> Vec<AutocompleteItem> {
    if query.is_empty() {
        return candidates;
    }
    let mut scored: Vec<(u32, AutocompleteItem)> = candidates
        .into_iter()
        .filter_map(|item| {
            // Match against both label (display) and value (inserted text).
            let score =
                match_score(&item.label, query).or_else(|| match_score(&item.value, query))?;
            Some((score, item))
        })
        .collect();
    scored.sort_by(|(s1, _), (s2, _)| s2.cmp(s1));
    scored.into_iter().map(|(_, item)| item).collect()
}
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
