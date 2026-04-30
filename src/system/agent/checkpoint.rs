//! File-edit checkpoint store — Phase 8 of the agent architecture plan.
//!
//! Goal: snapshot the *prior* contents of every file a write/patch/remove
//! tool call is about to touch, keyed by `tool_call_id`. The runtime can
//! then `undo_tool_call(id)` to restore exactly the bytes that existed
//! before the call ran.
//!
//! The store is in-memory only — it does not survive process restart. That's
//! deliberate: rollback is a "I just made a bad edit, take it back" feature,
//! not a durable history. A bounded total-bytes budget keeps a runaway
//! benchmark or stuck loop from exhausting RAM with snapshots.
//!
//! Restoration itself happens in `runtime.undo_tool_call` — this module only
//! owns the data. Keeps the store testable without filesystem access and
//! lets the runtime decide how to apply the bytes (sandbox, intel cache
//! invalidation, etc.).
//!
//! The non-`take_snapshots` accessors (`snapshots_for`, `most_recent_call_id`,
//! `entry_count`, `total_bytes`, `clear`) are public-by-design — they let
//! benchmarks and future TUI surfaces inspect the store without consuming
//! it. Tests cover them; production paths haven't picked them up yet.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;

/// Cap on total snapshot bytes. Beyond this, snapshots from the oldest
/// recorded tool calls are evicted to make room. Sized to comfortably hold
/// dozens of small file edits without limiting normal use.
pub const MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024; // 64 MB

/// Cap on how many distinct tool-call snapshots are retained.
pub const MAX_ENTRIES: usize = 1000;

/// Bytes captured for one file path *before* a tool call modifies it.
#[derive(Debug, Clone)]
pub struct FileSnapshot {
    pub path: PathBuf,
    /// `None` ⇒ the file did not exist; restoring should delete the file
    /// the tool created.
    pub prior_content: Option<Vec<u8>>,
}

impl FileSnapshot {
    pub fn size(&self) -> usize {
        self.prior_content.as_ref().map(Vec::len).unwrap_or(0)
    }
}

#[derive(Debug)]
pub struct CheckpointStore {
    inner: Mutex<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    /// tool_call_id → snapshots taken for that call.
    by_call: HashMap<String, Vec<FileSnapshot>>,
    /// Insertion order — used for FIFO eviction and "the most recent call".
    order: VecDeque<String>,
    total_bytes: usize,
}

impl Default for CheckpointStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
        }
    }

    /// Record the prior bytes of `path` for `tool_call_id`. Multiple calls
    /// for the same `(call_id, path)` are deduplicated — only the *first*
    /// snapshot wins, since that's the "before-the-call" state we want to
    /// restore to. Subsequent writes within the same tool call won't lose
    /// the original content.
    pub fn snapshot_before(
        &self,
        tool_call_id: &str,
        path: PathBuf,
        prior_content: Option<Vec<u8>>,
    ) {
        let mut inner = self.inner.lock().unwrap();

        let entry = inner.by_call.entry(tool_call_id.to_string()).or_default();
        if entry.iter().any(|s| s.path == path) {
            // Already snapshotted within this tool call — keep the first.
            return;
        }
        let added = prior_content.as_ref().map(Vec::len).unwrap_or(0);
        entry.push(FileSnapshot {
            path,
            prior_content,
        });
        inner.total_bytes += added;
        if !inner.order.contains(&tool_call_id.to_string()) {
            inner.order.push_back(tool_call_id.to_string());
        }
        inner.evict_until_within_caps();
    }

    /// Snapshots recorded for a given tool call, in the order they were
    /// captured. Returns `None` if no snapshots exist for that call.
    pub fn snapshots_for(&self, tool_call_id: &str) -> Option<Vec<FileSnapshot>> {
        self.inner
            .lock()
            .unwrap()
            .by_call
            .get(tool_call_id)
            .cloned()
    }

    /// Take the snapshots for a tool call out of the store. Use this when
    /// applying an undo so the snapshots aren't accidentally restored twice.
    pub fn take_snapshots(&self, tool_call_id: &str) -> Option<Vec<FileSnapshot>> {
        let mut inner = self.inner.lock().unwrap();
        let removed = inner.by_call.remove(tool_call_id);
        if let Some(ref snapshots) = removed {
            let freed: usize = snapshots.iter().map(|s| s.size()).sum();
            inner.total_bytes = inner.total_bytes.saturating_sub(freed);
            if let Some(pos) = inner.order.iter().position(|id| id == tool_call_id) {
                inner.order.remove(pos);
            }
        }
        removed
    }

    /// The most recent tool_call_id we have snapshots for, if any.
    pub fn most_recent_call_id(&self) -> Option<String> {
        self.inner.lock().unwrap().order.back().cloned()
    }

    /// Total tool calls currently checkpointed.
    pub fn entry_count(&self) -> usize {
        self.inner.lock().unwrap().by_call.len()
    }

    /// Total bytes retained across all snapshots.
    pub fn total_bytes(&self) -> usize {
        self.inner.lock().unwrap().total_bytes
    }

    /// Drop all snapshots. Used when starting a fresh thread / between tests.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.by_call.clear();
        inner.order.clear();
        inner.total_bytes = 0;
    }
}

impl Inner {
    /// FIFO-evict oldest tool-call snapshots until under both caps.
    fn evict_until_within_caps(&mut self) {
        while (self.total_bytes > MAX_TOTAL_BYTES || self.by_call.len() > MAX_ENTRIES)
            && !self.order.is_empty()
        {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(snapshots) = self.by_call.remove(&oldest) {
                let freed: usize = snapshots.iter().map(|s| s.size()).sum();
                self.total_bytes = self.total_bytes.saturating_sub(freed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn empty_store_returns_none_and_zero_counts() {
        let s = CheckpointStore::new();
        assert!(s.snapshots_for("c1").is_none());
        assert!(s.most_recent_call_id().is_none());
        assert_eq!(s.entry_count(), 0);
        assert_eq!(s.total_bytes(), 0);
    }

    #[test]
    fn snapshot_before_records_path_and_content() {
        let s = CheckpointStore::new();
        s.snapshot_before("c1", p("a.txt"), Some(b"old".to_vec()));
        let got = s.snapshots_for("c1").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].path, p("a.txt"));
        assert_eq!(got[0].prior_content.as_deref(), Some(b"old".as_slice()));
        assert_eq!(s.total_bytes(), 3);
    }

    #[test]
    fn first_snapshot_wins_for_duplicate_path_within_call() {
        let s = CheckpointStore::new();
        s.snapshot_before("c1", p("a.txt"), Some(b"original".to_vec()));
        s.snapshot_before("c1", p("a.txt"), Some(b"intermediate".to_vec()));
        let got = s.snapshots_for("c1").unwrap();
        assert_eq!(got.len(), 1, "duplicate path should not double-record");
        assert_eq!(
            got[0].prior_content.as_deref(),
            Some(b"original".as_slice()),
            "the first (true 'before') content must be preserved"
        );
    }

    #[test]
    fn snapshot_records_missing_file_as_none() {
        let s = CheckpointStore::new();
        s.snapshot_before("c1", p("new.txt"), None);
        let got = s.snapshots_for("c1").unwrap();
        assert_eq!(got.len(), 1);
        assert!(got[0].prior_content.is_none());
        assert_eq!(s.total_bytes(), 0);
    }

    #[test]
    fn most_recent_tracks_insertion_order() {
        let s = CheckpointStore::new();
        s.snapshot_before("c1", p("a"), Some(b"x".to_vec()));
        s.snapshot_before("c2", p("b"), Some(b"y".to_vec()));
        s.snapshot_before("c3", p("c"), Some(b"z".to_vec()));
        assert_eq!(s.most_recent_call_id().as_deref(), Some("c3"));
    }

    #[test]
    fn take_snapshots_removes_and_returns() {
        let s = CheckpointStore::new();
        s.snapshot_before("c1", p("a"), Some(b"abc".to_vec()));
        let taken = s.take_snapshots("c1").unwrap();
        assert_eq!(taken.len(), 1);
        assert!(s.snapshots_for("c1").is_none());
        assert_eq!(s.entry_count(), 0);
        assert_eq!(s.total_bytes(), 0);
    }

    #[test]
    fn take_snapshots_for_unknown_call_is_none() {
        let s = CheckpointStore::new();
        assert!(s.take_snapshots("ghost").is_none());
    }

    #[test]
    fn clear_drops_everything() {
        let s = CheckpointStore::new();
        s.snapshot_before("c1", p("a"), Some(b"a".to_vec()));
        s.snapshot_before("c2", p("b"), Some(b"bb".to_vec()));
        s.clear();
        assert_eq!(s.entry_count(), 0);
        assert_eq!(s.total_bytes(), 0);
        assert!(s.most_recent_call_id().is_none());
    }

    #[test]
    fn multiple_paths_in_same_call_are_independent_entries() {
        let s = CheckpointStore::new();
        s.snapshot_before("c1", p("a"), Some(b"a".to_vec()));
        s.snapshot_before("c1", p("b"), Some(b"bb".to_vec()));
        let got = s.snapshots_for("c1").unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(s.total_bytes(), 3);
    }

    #[test]
    fn fifo_eviction_when_entry_cap_exceeded() {
        let s = CheckpointStore::new();
        // Force the entry cap by simulating MAX_ENTRIES + 5 distinct calls.
        // Use small content so the byte cap doesn't trigger first.
        for i in 0..(MAX_ENTRIES + 5) {
            s.snapshot_before(&format!("c{i}"), p(&format!("f{i}")), Some(b"x".to_vec()));
        }
        assert!(s.entry_count() <= MAX_ENTRIES);
        // Oldest five (c0..c4) should be evicted; newest should still be there.
        assert!(s.snapshots_for("c0").is_none());
        assert!(s.snapshots_for(&format!("c{}", MAX_ENTRIES + 4)).is_some());
    }
}
