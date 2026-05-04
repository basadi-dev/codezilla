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

    // ── matching tests ────────────────────────────────────────────────────

    #[test]
    fn filter_and_rank_exact_match_first() {
        let candidates = vec![
            AutocompleteItem::simple("/model"),
            AutocompleteItem::simple("/model gpt-4"),
            AutocompleteItem::simple("/approve"),
        ];
        let result = filter_and_rank(candidates, "/model");
        assert_eq!(result[0].value, "/model");
    }

    #[test]
    fn filter_and_rank_prefix_match() {
        let candidates = vec![
            AutocompleteItem::simple("/approve"),
            AutocompleteItem::simple("/approve auto"),
            AutocompleteItem::simple("/model"),
        ];
        let result = filter_and_rank(candidates, "/approve");
        assert_eq!(result[0].value, "/approve");
        assert_eq!(result[1].value, "/approve auto");
    }

    #[test]
    fn filter_and_rank_substring_match() {
        let candidates = vec![
            AutocompleteItem::simple("/approve"),
            AutocompleteItem::simple("/model"),
            AutocompleteItem::simple("/reasoning"),
        ];
        // "mod" is a substring of "/model"
        let result = filter_and_rank(candidates, "mod");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, "/model");
    }

    #[test]
    fn filter_and_rank_fuzzy_match() {
        let candidates = vec![
            AutocompleteItem::simple("/approve"),
            AutocompleteItem::simple("/model"),
            AutocompleteItem::simple("/reasoning"),
        ];
        // "mdl" fuzzy-matches "/model"
        let result = filter_and_rank(candidates, "mdl");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, "/model");
    }

    #[test]
    fn filter_and_rank_no_match_returns_empty() {
        let candidates = vec![
            AutocompleteItem::simple("/approve"),
            AutocompleteItem::simple("/model"),
        ];
        let result = filter_and_rank(candidates, "xyz");
        assert!(result.is_empty());
    }

    #[test]
    fn filter_and_rank_scores_better_match_higher() {
        let candidates = vec![
            AutocompleteItem::simple("/reasoning"),
            AutocompleteItem::simple("/model"),
            AutocompleteItem::simple("/approve"),
        ];
        // "re" is prefix of "/reasoning", substring of "/approve"
        let result = filter_and_rank(candidates, "re");
        assert_eq!(result[0].value, "/reasoning");
    }

    #[test]
    fn filter_and_rank_empty_query_returns_all() {
        let candidates = vec![
            AutocompleteItem::simple("/model"),
            AutocompleteItem::simple("/approve"),
        ];
        let result = filter_and_rank(candidates, "");
        assert_eq!(result.len(), 2);
    }
}
