use std::collections::HashMap;

/// A simple key-value store with expiration support.
pub struct KvStore {
    data: HashMap<String, (String, Option<u64>)>,
    current_time: u64,
}

impl KvStore {
    pub fn new() -> Self {
        KvStore {
            data: HashMap::new(),
            current_time: 0,
        }
    }

    /// Advance the internal clock.
    pub fn tick(&mut self, seconds: u64) {
        self.current_time += seconds;
    }

    /// Set a key-value pair. If `ttl` is Some, the key expires after that many seconds.
    pub fn set(&mut self, key: &str, value: &str, ttl: Option<u64>) {
        let expires_at = ttl.map(|t| self.current_time + t);
        self.data.insert(key.to_string(), (value.to_string(), expires_at));
    }

    /// Get a value by key. Returns None if the key doesn't exist or has expired.
    pub fn get(&self, key: &str) -> Option<&str> {
        match self.data.get(key) {
            Some((value, Some(expires_at))) => {
                // BUG 1: should be self.current_time >= *expires_at
                // but uses > instead, so it returns the value on the exact expiration tick
                if self.current_time > *expires_at {
                    None
                } else {
                    Some(value.as_str())
                }
            }
            Some((value, None)) => Some(value.as_str()),
            None => None,
        }
    }

    /// Delete a key. Returns true if the key existed (and wasn't expired).
    pub fn delete(&mut self, key: &str) -> bool {
        // BUG 2: doesn't check if key is expired before claiming it existed
        self.data.remove(key).is_some()
    }

    /// Count the number of non-expired keys.
    pub fn len(&self) -> usize {
        self.data
            .iter()
            .filter(|(_, (_, expires_at))| {
                match expires_at {
                    // BUG 3: same off-by-one as get() — uses > instead of >=
                    Some(exp) => self.current_time > *exp,
                    None => true,
                }
            })
            .count()
    }

    /// Return all non-expired keys, sorted alphabetically.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .data
            .iter()
            .filter(|(_, (_, expires_at))| {
                match expires_at {
                    Some(exp) => self.current_time < *exp,
                    None => true,
                }
            })
            .map(|(k, _)| k.clone())
            .collect();
        keys.sort();
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_set_get() {
        let mut store = KvStore::new();
        store.set("name", "alice", None);
        assert_eq!(store.get("name"), Some("alice"));
    }

    #[test]
    fn test_missing_key() {
        let store = KvStore::new();
        assert_eq!(store.get("nope"), None);
    }

    #[test]
    fn test_overwrite() {
        let mut store = KvStore::new();
        store.set("x", "1", None);
        store.set("x", "2", None);
        assert_eq!(store.get("x"), Some("2"));
    }

    #[test]
    fn test_ttl_before_expiry() {
        let mut store = KvStore::new();
        store.set("temp", "value", Some(10));
        store.tick(5);
        assert_eq!(store.get("temp"), Some("value"));
    }

    #[test]
    fn test_ttl_at_expiry() {
        let mut store = KvStore::new();
        store.set("temp", "value", Some(10));
        store.tick(10);
        // At exactly the expiration time, the key should be gone
        assert_eq!(store.get("temp"), None);
    }

    #[test]
    fn test_ttl_after_expiry() {
        let mut store = KvStore::new();
        store.set("temp", "value", Some(10));
        store.tick(15);
        assert_eq!(store.get("temp"), None);
    }

    #[test]
    fn test_delete_existing() {
        let mut store = KvStore::new();
        store.set("x", "1", None);
        assert!(store.delete("x"));
        assert_eq!(store.get("x"), None);
    }

    #[test]
    fn test_delete_expired() {
        let mut store = KvStore::new();
        store.set("temp", "value", Some(5));
        store.tick(10);
        // Key is expired, so delete should return false
        assert!(!store.delete("temp"));
    }

    #[test]
    fn test_len_excludes_expired() {
        let mut store = KvStore::new();
        store.set("a", "1", None);
        store.set("b", "2", Some(5));
        store.set("c", "3", Some(10));
        assert_eq!(store.len(), 3);

        store.tick(5);
        // "b" has expired (ttl=5, current_time=5 → expired at exactly 5)
        assert_eq!(store.len(), 2);

        store.tick(5);
        // "c" has also expired now
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_keys_excludes_expired() {
        let mut store = KvStore::new();
        store.set("apple", "1", None);
        store.set("banana", "2", Some(5));
        store.set("cherry", "3", Some(10));

        assert_eq!(store.keys(), vec!["apple", "banana", "cherry"]);

        store.tick(5);
        assert_eq!(store.keys(), vec!["apple", "cherry"]);
    }
}
