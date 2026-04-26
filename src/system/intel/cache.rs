//! SHA2-keyed in-process LRU cache for symbol extraction results.
//!
//! Keyed on `(absolute_path, SHA-256 of file content)` so the cached
//! symbols are always consistent with the file on disk — no stale reads.
//! Eviction is simple FIFO at 200 entries; the working set of a repo is
//! almost always well under that.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use super::symbols::Symbol;

// ─── CacheKey ─────────────────────────────────────────────────────────────────

#[derive(Hash, Eq, PartialEq, Clone)]
struct CacheKey {
    path: PathBuf,
    content_hash: [u8; 32],
}

// ─── IntelCache ───────────────────────────────────────────────────────────────

/// Thread-safe cache mapping `(path, content_hash)` → extracted symbols.
///
/// The cache is shared for the entire process lifetime (stored in
/// `RuntimeInner`) so symbol results persist across turns and threads.
pub struct IntelCache {
    inner: Mutex<CacheInner>,
}

struct CacheInner {
    map: HashMap<CacheKey, Vec<Symbol>>,
    /// Insertion-order queue for FIFO eviction.
    lru: VecDeque<CacheKey>,
    capacity: usize,
}

impl IntelCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                map: HashMap::new(),
                lru: VecDeque::new(),
                capacity,
            }),
        }
    }

    /// Compute the SHA-256 of `content` and look up the cache.
    /// Returns `Some` if there is a fresh cache hit, `None` on miss.
    pub fn get(&self, path: &Path, content: &str) -> Option<Vec<Symbol>> {
        let hash = sha256(content);
        let key = CacheKey {
            path: path.to_path_buf(),
            content_hash: hash,
        };
        let inner = self.inner.lock().unwrap();
        inner.map.get(&key).cloned()
    }

    /// Store symbols for the given path + content.
    pub fn put(&self, path: &Path, content: &str, symbols: Vec<Symbol>) {
        let hash = sha256(content);
        let key = CacheKey {
            path: path.to_path_buf(),
            content_hash: hash,
        };
        let mut inner = self.inner.lock().unwrap();

        // If the key already exists, just update the value — don't push a
        // duplicate into the eviction queue.
        #[allow(clippy::map_entry)]
        if inner.map.contains_key(&key) {
            inner.map.insert(key, symbols);
            return;
        }

        // Evict if at capacity (FIFO — oldest first).
        while inner.map.len() >= inner.capacity {
            if let Some(old_key) = inner.lru.pop_front() {
                inner.map.remove(&old_key);
            } else {
                break;
            }
        }

        inner.lru.push_back(key.clone());
        inner.map.insert(key, symbols);
    }

    /// Invalidate all cache entries for `path` (regardless of content hash).
    /// Called after `write_file` / `patch_file` so stale symbols are evicted.
    pub fn invalidate(&self, path: &Path) {
        let mut inner = self.inner.lock().unwrap();
        inner.lru.retain(|k| k.path != *path);
        inner.map.retain(|k, _| k.path != path);
    }
}

fn sha256(content: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher.finalize().into()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::intel::symbols::{Symbol, SymbolKind};

    fn sym(name: &str) -> Symbol {
        Symbol {
            name: name.into(),
            kind: SymbolKind::Function,
            line: 1,
        }
    }

    #[test]
    fn hit_after_put() {
        let cache = IntelCache::new(10);
        let path = Path::new("/foo/bar.rs");
        let content = "pub fn hello() {}";
        cache.put(path, content, vec![sym("hello")]);
        let result = cache.get(path, content);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0].name, "hello");
    }

    #[test]
    fn miss_on_changed_content() {
        let cache = IntelCache::new(10);
        let path = Path::new("/foo/bar.rs");
        cache.put(path, "pub fn hello() {}", vec![sym("hello")]);
        // Different content → hash mismatch → miss
        assert!(cache.get(path, "pub fn world() {}").is_none());
    }

    #[test]
    fn invalidate_removes_entry() {
        let cache = IntelCache::new(10);
        let path = Path::new("/foo/bar.rs");
        let content = "pub fn hello() {}";
        cache.put(path, content, vec![sym("hello")]);
        cache.invalidate(path);
        assert!(cache.get(path, content).is_none());
    }

    #[test]
    fn evicts_at_capacity() {
        let cache = IntelCache::new(3);
        for i in 0..4u32 {
            let content = format!("pub fn f{i}() {{}}");
            cache.put(
                Path::new(&format!("/f{i}.rs")),
                &content,
                vec![sym(&format!("f{i}"))],
            );
        }
        let inner = cache.inner.lock().unwrap();
        assert_eq!(inner.map.len(), 3);
    }
}
