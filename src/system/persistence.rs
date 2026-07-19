use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::domain::{
    now_seconds, ConversationItem, ItemKind, MemoryMode, PersistedThread, ThreadFilter,
    ThreadMetadata, ThreadStatus, TurnMetadata, TurnStatus,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationMemoryRecord {
    pub memory_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub kind: String,
    pub scope: String,
    pub content: String,
    pub importance: f32,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
}

pub struct PersistenceManager {
    conn: Mutex<Connection>,
    memories_root: PathBuf,
    #[allow(dead_code)]
    logs_root: PathBuf,
}

impl PersistenceManager {
    pub fn new(
        state_root: impl AsRef<Path>,
        memories_root: impl AsRef<Path>,
        logs_root: impl AsRef<Path>,
    ) -> Result<Self> {
        let state_root = state_root.as_ref();
        fs::create_dir_all(state_root)?;
        fs::create_dir_all(memories_root.as_ref())?;
        fs::create_dir_all(logs_root.as_ref())?;

        let db_path = state_root.join("codezilla.sqlite3");
        let conn = Connection::open(&db_path)
            .with_context(|| format!("opening sqlite database {}", db_path.display()))?;
        let manager = Self {
            conn: Mutex::new(conn),
            memories_root: memories_root.as_ref().to_path_buf(),
            logs_root: logs_root.as_ref().to_path_buf(),
        };
        manager.init()?;
        manager.recover_incomplete_turns()?;
        Ok(manager)
    }

    pub fn create_thread(&self, metadata: &ThreadMetadata) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                r#"
            INSERT INTO threads (
                thread_id, title, created_at, updated_at, cwd, model_id, provider_id,
                status, forked_from_id, archived, ephemeral, memory_mode, last_sequence
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, 0)
            "#,
                params![
                    metadata.thread_id,
                    metadata.title,
                    metadata.created_at,
                    metadata.updated_at,
                    metadata.cwd,
                    metadata.model_id,
                    metadata.provider_id,
                    enum_json(&metadata.status)?,
                    metadata.forked_from_id,
                    metadata.archived as i64,
                    metadata.ephemeral as i64,
                    enum_json(&metadata.memory_mode)?,
                ],
            )?;
            Ok(())
        })
    }

    pub fn create_turn(&self, metadata: &TurnMetadata) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                r#"
            INSERT INTO turns (
                turn_id, thread_id, created_at, updated_at, status, started_by_surface,
                token_usage_json, token_usage_estimated_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
                params![
                    metadata.turn_id,
                    metadata.thread_id,
                    metadata.created_at,
                    metadata.updated_at,
                    enum_json(&metadata.status)?,
                    enum_json(&metadata.started_by_surface)?,
                    serde_json::to_string(&metadata.token_usage)?,
                    serde_json::to_string(&metadata.estimated_token_usage)?,
                ],
            )?;
            conn.execute(
                "UPDATE threads SET updated_at = ?2, status = ?3 WHERE thread_id = ?1",
                params![
                    metadata.thread_id,
                    metadata.updated_at,
                    enum_json(&ThreadStatus::Running)?,
                ],
            )?;
            Ok(())
        })
    }

    pub fn append_item(&self, item: &ConversationItem) -> Result<()> {
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            let next_order = next_item_order(&tx, &item.thread_id)?;
            tx.execute(
                r#"
            INSERT INTO items (
                item_id, thread_id, turn_id, created_at, kind, payload_json, item_order, tombstoned
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0)
            "#,
                params![
                    item.item_id,
                    item.thread_id,
                    item.turn_id,
                    item.created_at,
                    enum_json(&item.kind)?,
                    serde_json::to_string(&item.payload)?,
                    next_order,
                ],
            )?;
            tx.execute(
                "UPDATE threads SET updated_at = ?2 WHERE thread_id = ?1",
                params![item.thread_id, item.created_at],
            )?;
            tx.commit()?;
            Ok(())
        })
    }

    pub fn update_turn(&self, turn: &TurnMetadata) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                r#"
            UPDATE turns
            SET updated_at = ?3, status = ?4, token_usage_json = ?5, token_usage_estimated_json = ?6
            WHERE turn_id = ?1 AND thread_id = ?2
            "#,
                params![
                    turn.turn_id,
                    turn.thread_id,
                    turn.updated_at,
                    enum_json(&turn.status)?,
                    serde_json::to_string(&turn.token_usage)?,
                    serde_json::to_string(&turn.estimated_token_usage)?,
                ],
            )?;
            Ok(())
        })
    }

    pub fn update_thread(&self, thread: &ThreadMetadata) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                r#"
            UPDATE threads
            SET title = ?2, updated_at = ?3, cwd = ?4, model_id = ?5, provider_id = ?6,
                status = ?7, forked_from_id = ?8, archived = ?9, ephemeral = ?10, memory_mode = ?11
            WHERE thread_id = ?1
            "#,
                params![
                    thread.thread_id,
                    thread.title,
                    thread.updated_at,
                    thread.cwd,
                    thread.model_id,
                    thread.provider_id,
                    enum_json(&thread.status)?,
                    thread.forked_from_id,
                    thread.archived as i64,
                    thread.ephemeral as i64,
                    enum_json(&thread.memory_mode)?,
                ],
            )?;
            Ok(())
        })
    }

    pub fn archive_thread(&self, thread_id: &str) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                "UPDATE threads SET archived = 1, status = ?2, updated_at = ?3 WHERE thread_id = ?1",
                params![
                    thread_id,
                    enum_json(&ThreadStatus::Archived)?,
                    now_seconds()
                ],
            )?;
            Ok(())
        })
    }

    pub fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            tx.execute("DELETE FROM items WHERE thread_id = ?1", params![thread_id])?;
            tx.execute("DELETE FROM turns WHERE thread_id = ?1", params![thread_id])?;
            tx.execute(
                "DELETE FROM threads WHERE thread_id = ?1",
                params![thread_id],
            )?;
            tx.commit()?;
            Ok(())
        })
    }

    /// Mark every non-tombstoned item in `thread_id` as deleted.
    /// Returns the number of rows affected.
    pub fn tombstone_all_items(&self, thread_id: &str) -> Result<usize> {
        self.with_conn(|conn| {
            let count = conn.execute(
                "UPDATE items SET tombstoned = 1 WHERE thread_id = ?1 AND tombstoned = 0",
                params![thread_id],
            )?;
            Ok(count)
        })
    }

    pub fn read_thread(&self, thread_id: &str) -> Result<PersistedThread> {
        self.with_conn(|conn| {
            let metadata = read_thread_metadata(conn, thread_id)?
                .with_context(|| format!("thread_not_found: {thread_id}"))?;

            let turns = {
                let mut stmt = conn.prepare(
                    "SELECT turn_id, thread_id, created_at, updated_at, status, started_by_surface, token_usage_json, token_usage_estimated_json
                     FROM turns WHERE thread_id = ?1 ORDER BY created_at ASC",
                )?;
                let mut rows = stmt.query(params![thread_id])?;
                let mut turns = Vec::new();
                while let Some(row) = rows.next()? {
                    turns.push(TurnMetadata {
                        turn_id: row.get(0)?,
                        thread_id: row.get(1)?,
                        created_at: row.get(2)?,
                        updated_at: row.get(3)?,
                        status: serde_json::from_str(&row.get::<_, String>(4)?)?,
                        started_by_surface: serde_json::from_str(&row.get::<_, String>(5)?)?,
                        token_usage: serde_json::from_str(&row.get::<_, String>(6)?)?,
                        estimated_token_usage: serde_json::from_str(&row.get::<_, String>(7)?)?,
                    });
                }
                turns
            };

            let items = {
                let mut stmt = conn.prepare(
                    "SELECT item_id, thread_id, turn_id, created_at, kind, payload_json
                     FROM items WHERE thread_id = ?1 AND tombstoned = 0 ORDER BY item_order ASC",
                )?;
                let mut rows = stmt.query(params![thread_id])?;
                let mut items = Vec::new();
                while let Some(row) = rows.next()? {
                    items.push(ConversationItem {
                        item_id: row.get(0)?,
                        thread_id: row.get(1)?,
                        turn_id: row.get(2)?,
                        created_at: row.get(3)?,
                        kind: serde_json::from_str(&row.get::<_, String>(4)?)?,
                        payload: serde_json::from_str(&row.get::<_, String>(5)?)?,
                    });
                }
                items
            };

            Ok(PersistedThread { metadata, turns, items })
        })
    }

    pub fn list_threads(&self, filter: ThreadFilter) -> Result<Vec<ThreadMetadata>> {
        self.with_conn(|conn| {
            let limit = if filter.limit > 0 { filter.limit } else { 20 };
            let mut stmt = conn.prepare(
                "SELECT thread_id, title, created_at, updated_at, cwd, model_id, provider_id,
                        status, forked_from_id, archived, ephemeral, memory_mode
                 FROM threads ORDER BY updated_at DESC LIMIT ?1",
            )?;
            let mut rows = stmt.query(params![limit])?;
            let mut results = Vec::new();
            while let Some(row) = rows.next()? {
                let metadata = ThreadMetadata {
                    thread_id: row.get(0)?,
                    title: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                    cwd: row.get(4)?,
                    model_id: row.get(5)?,
                    provider_id: row.get(6)?,
                    status: serde_json::from_str(&row.get::<_, String>(7)?)?,
                    forked_from_id: row.get(8)?,
                    archived: row.get::<_, i64>(9)? != 0,
                    ephemeral: row.get::<_, i64>(10)? != 0,
                    memory_mode: serde_json::from_str(&row.get::<_, String>(11)?)?,
                };
                if let Some(archived) = filter.archived {
                    if metadata.archived != archived {
                        continue;
                    }
                }
                if let Some(cwd) = &filter.cwd {
                    if metadata.cwd.as_deref() != Some(cwd.as_str()) {
                        continue;
                    }
                }
                if let Some(search) = &filter.search_term {
                    let haystack = format!(
                        "{} {}",
                        metadata.thread_id,
                        metadata.title.clone().unwrap_or_default()
                    );
                    if !haystack.contains(search) {
                        continue;
                    }
                }
                results.push(metadata);
            }
            Ok(results)
        })
    }

    pub fn bump_thread_sequence(&self, thread_id: &str) -> Result<i64> {
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            let current: i64 = tx
                .query_row(
                    "SELECT COALESCE(last_sequence, 0) FROM threads WHERE thread_id = ?1",
                    params![thread_id],
                    |row| row.get(0),
                )
                .optional()?
                .unwrap_or(0);
            let next = current + 1;
            tx.execute(
                "UPDATE threads SET last_sequence = ?2 WHERE thread_id = ?1",
                params![thread_id, next],
            )?;
            tx.commit()?;
            Ok(next)
        })
    }

    pub fn reset_memories(&self) -> Result<()> {
        if self.memories_root.exists() {
            for entry in fs::read_dir(&self.memories_root)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    fs::remove_file(path)?;
                } else if path.is_dir() {
                    fs::remove_dir_all(path)?;
                }
            }
        }
        self.with_conn(|conn| {
            conn.execute("DELETE FROM conversation_memories", [])?;
            Ok(())
        })?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn append_log(&self, level: &str, message: &str) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO logs (created_at, level, message) VALUES (?1, ?2, ?3)",
                params![now_seconds(), level, message],
            )?;
            Ok(())
        })
    }

    #[allow(dead_code)]
    pub fn save_memory_record(&self, thread_id: &str, payload: &Value) -> Result<()> {
        let path = self.memories_root.join(format!("{thread_id}.json"));
        fs::write(path, serde_json::to_vec_pretty(payload)?)?;
        Ok(())
    }

    pub fn append_conversation_memory(
        &self,
        thread_id: &str,
        turn_id: &str,
        kind: &str,
        scope: &str,
        content: &str,
        importance: f32,
    ) -> Result<String> {
        let content = content.trim();
        if content.is_empty() {
            anyhow::bail!("empty_memory_content");
        }
        let memory_id = format!("mem_{}", uuid::Uuid::new_v4().simple());
        self.with_conn(|conn| {
            conn.execute(
                r#"
                INSERT INTO conversation_memories (
                    memory_id, thread_id, turn_id, kind, scope, content, importance, created_at, last_used_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL)
                "#,
                params![
                    memory_id,
                    thread_id,
                    turn_id,
                    kind,
                    scope,
                    content,
                    importance,
                    now_seconds(),
                ],
            )?;
            Ok(())
        })?;
        Ok(memory_id)
    }

    pub fn search_conversation_memories(
        &self,
        thread_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ConversationMemoryRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query_terms = memory_terms(query);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored = self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                r#"
                SELECT memory_id, thread_id, turn_id, kind, scope, content, importance, created_at, last_used_at
                FROM conversation_memories
                WHERE thread_id = ?1 OR scope = 'global'
                ORDER BY created_at DESC
                LIMIT 1000
                "#,
            )?;
            let mut rows = stmt.query(params![thread_id])?;
            let mut records = Vec::new();
            while let Some(row) = rows.next()? {
                let record = ConversationMemoryRecord {
                    memory_id: row.get(0)?,
                    thread_id: row.get(1)?,
                    turn_id: row.get(2)?,
                    kind: row.get(3)?,
                    scope: row.get(4)?,
                    content: row.get(5)?,
                    importance: row.get::<_, f64>(6)? as f32,
                    created_at: row.get(7)?,
                    last_used_at: row.get(8)?,
                };
                let content_terms = memory_terms(&record.content);
                let overlap = query_terms.intersection(&content_terms).count() as f32;
                if overlap > 0.0 {
                    let recency = 1.0 / (1.0 + ((now_seconds() - record.created_at).max(0) as f32 / 86_400.0));
                    let score = overlap + record.importance + recency;
                    records.push((score, record));
                }
            }
            Ok(records)
        })?;

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        let records: Vec<_> = scored.into_iter().map(|(_, record)| record).collect();
        self.touch_conversation_memories(records.iter().map(|r| r.memory_id.as_str()))?;
        Ok(records)
    }

    pub fn list_conversation_memories(
        &self,
        thread_id: &str,
        limit: usize,
    ) -> Result<Vec<ConversationMemoryRecord>> {
        let limit = limit.clamp(1, 200);
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                r#"
                SELECT memory_id, thread_id, turn_id, kind, scope, content, importance, created_at, last_used_at
                FROM conversation_memories
                WHERE thread_id = ?1 OR scope = 'global'
                ORDER BY created_at DESC
                LIMIT ?2
                "#,
            )?;
            let mut rows = stmt.query(params![thread_id, limit as i64])?;
            let mut records = Vec::new();
            while let Some(row) = rows.next()? {
                records.push(ConversationMemoryRecord {
                    memory_id: row.get(0)?,
                    thread_id: row.get(1)?,
                    turn_id: row.get(2)?,
                    kind: row.get(3)?,
                    scope: row.get(4)?,
                    content: row.get(5)?,
                    importance: row.get::<_, f64>(6)? as f32,
                    created_at: row.get(7)?,
                    last_used_at: row.get(8)?,
                });
            }
            Ok(records)
        })
    }

    pub fn update_conversation_memory(
        &self,
        memory_id: &str,
        thread_id: &str,
        kind: Option<&str>,
        scope: Option<&str>,
        content: Option<&str>,
        importance: Option<f32>,
    ) -> Result<bool> {
        self.with_conn(|conn| {
            let n = conn.execute(
                r#"
                UPDATE conversation_memories
                SET kind = COALESCE(?2, kind),
                    scope = COALESCE(?3, scope),
                    content = COALESCE(?4, content),
                    importance = COALESCE(?5, importance)
                WHERE memory_id = ?1 AND (thread_id = ?6 OR scope = 'global')
                "#,
                params![memory_id, kind, scope, content, importance, thread_id],
            )?;
            Ok(n > 0)
        })
    }

    pub fn delete_conversation_memory(&self, memory_id: &str, thread_id: &str) -> Result<bool> {
        self.with_conn(|conn| {
            let n = conn.execute(
                "DELETE FROM conversation_memories WHERE memory_id = ?1 AND (thread_id = ?2 OR scope = 'global')",
                params![memory_id, thread_id],
            )?;
            Ok(n > 0)
        })
    }

    fn touch_conversation_memories<'a>(&self, ids: impl Iterator<Item = &'a str>) -> Result<()> {
        let ids: Vec<String> = ids.map(str::to_string).collect();
        if ids.is_empty() {
            return Ok(());
        }
        self.with_conn(|conn| {
            for id in ids {
                conn.execute(
                    "UPDATE conversation_memories SET last_used_at = ?2 WHERE memory_id = ?1",
                    params![id, now_seconds()],
                )?;
            }
            Ok(())
        })
    }

    #[allow(dead_code)]
    pub fn rebuild_metadata(&self) -> Result<()> {
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            tx.execute("DELETE FROM threads", [])?;

            let rows = {
                let mut stmt = tx.prepare(
                "SELECT thread_id, MIN(created_at), MAX(created_at) FROM items GROUP BY thread_id",
            )?;
                let rows = stmt.query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })?;
                let mut collected = Vec::new();
                for row in rows {
                    collected.push(row?);
                }
                collected
            };

            for (thread_id, created_at, updated_at) in rows {
                tx.execute(
                    r#"
                INSERT INTO threads (
                    thread_id, created_at, updated_at, model_id, provider_id, status,
                    archived, ephemeral, memory_mode, last_sequence
                ) VALUES (?1, ?2, ?3, '', '', ?4, 0, 0, ?5, 0)
                "#,
                    params![
                        thread_id,
                        created_at,
                        updated_at,
                        enum_json(&ThreadStatus::Idle)?,
                        enum_json(&MemoryMode::Enabled)?,
                    ],
                )?;
            }
            tx.commit()?;
            Ok(())
        })
    }

    fn init(&self) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute_batch(
                r#"
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                title TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                cwd TEXT,
                model_id TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                status TEXT NOT NULL,
                forked_from_id TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                ephemeral INTEGER NOT NULL DEFAULT 0,
                memory_mode TEXT NOT NULL,
                last_sequence INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_by_surface TEXT NOT NULL,
                token_usage_json TEXT NOT NULL,
                token_usage_estimated_json TEXT NOT NULL DEFAULT '{"inputTokens":0,"outputTokens":0,"cachedTokens":0}'
            );
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                item_order INTEGER NOT NULL,
                tombstoned INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_items_thread_order ON items(thread_id, item_order);
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_memories (
                memory_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                created_at INTEGER NOT NULL,
                last_used_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_conversation_memories_thread ON conversation_memories(thread_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_conversation_memories_scope ON conversation_memories(scope);
            "#,
            )?;
            conn.execute(
                "ALTER TABLE turns ADD COLUMN token_usage_estimated_json TEXT NOT NULL DEFAULT '{\"inputTokens\":0,\"outputTokens\":0,\"cachedTokens\":0}'",
                [],
            )
            .ok();
            conn.execute(
                "UPDATE turns
                 SET token_usage_estimated_json = token_usage_json
                 WHERE token_usage_estimated_json IS NULL
                    OR token_usage_estimated_json = ''
                    OR token_usage_estimated_json = '{\"inputTokens\":0,\"outputTokens\":0,\"cachedTokens\":0}'",
                [],
            )?;
            Ok(())
        })
    }

    fn recover_incomplete_turns(&self) -> Result<()> {
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            let mut stmt =
                tx.prepare("SELECT turn_id, thread_id FROM turns WHERE status IN (?1, ?2)")?;
            let mut rows = stmt.query(params![
                enum_json(&TurnStatus::Running)?,
                enum_json(&TurnStatus::WaitingForApproval)?,
            ])?;
            let mut stuck = Vec::new();
            while let Some(row) = rows.next()? {
                stuck.push((row.get::<_, String>(0)?, row.get::<_, String>(1)?));
            }
            drop(rows);
            drop(stmt);

            for (turn_id, thread_id) in stuck {
                tx.execute(
                    "UPDATE turns SET status = ?3, updated_at = ?4 WHERE turn_id = ?1 AND thread_id = ?2",
                    params![
                        turn_id,
                        thread_id,
                        enum_json(&TurnStatus::Interrupted)?,
                        now_seconds(),
                    ],
                )?;

                let next_order = next_item_order(&tx, &thread_id)?;
                tx.execute(
                    r#"
                INSERT INTO items (item_id, thread_id, turn_id, created_at, kind, payload_json, item_order, tombstoned)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0)
                "#,
                    params![
                        format!("recovery-{}-{}", thread_id, turn_id),
                        thread_id,
                        turn_id,
                        now_seconds(),
                        enum_json(&ItemKind::Status)?,
                        json_text(json!({
                            "message": "Recovered interrupted turn after restart",
                            "recovery": true
                        }))?,
                        next_order,
                    ],
                )?;
                tx.execute(
                    "UPDATE threads SET status = ?2, updated_at = ?3 WHERE thread_id = ?1",
                    params![
                        thread_id,
                        enum_json(&ThreadStatus::Interrupted)?,
                        now_seconds(),
                    ],
                )?;
            }
            tx.commit()?;
            Ok(())
        })
    }

    fn with_conn<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let conn = self.conn.lock().expect("sqlite mutex poisoned");
        f(&conn)
    }
}

fn memory_terms(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter_map(|term| {
            let term = term.trim().to_ascii_lowercase();
            if term.len() >= 3 && !MEMORY_STOP_WORDS.contains(&term.as_str()) {
                Some(term)
            } else {
                None
            }
        })
        .collect()
}

const MEMORY_STOP_WORDS: &[&str] = &[
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "were", "have",
    "has", "had", "but", "not", "can", "could", "would", "should", "from", "about", "into", "when",
    "where", "what", "how", "why", "then", "than", "them", "they", "there", "their", "use",
    "using", "used", "need", "want", "like",
];

fn next_item_order(tx: &Transaction<'_>, thread_id: &str) -> Result<i64> {
    let next = tx.query_row(
        "SELECT COALESCE(MAX(item_order), 0) + 1 FROM items WHERE thread_id = ?1",
        params![thread_id],
        |row| row.get(0),
    )?;
    Ok(next)
}

fn read_thread_metadata(conn: &Connection, thread_id: &str) -> Result<Option<ThreadMetadata>> {
    conn.query_row(
        "SELECT thread_id, title, created_at, updated_at, cwd, model_id, provider_id,
                status, forked_from_id, archived, ephemeral, memory_mode
         FROM threads WHERE thread_id = ?1",
        params![thread_id],
        |row| {
            Ok(ThreadMetadata {
                thread_id: row.get(0)?,
                title: row.get(1)?,
                created_at: row.get(2)?,
                updated_at: row.get(3)?,
                cwd: row.get(4)?,
                model_id: row.get(5)?,
                provider_id: row.get(6)?,
                status: serde_json::from_str(&row.get::<_, String>(7)?)
                    .unwrap_or(ThreadStatus::Idle),
                forked_from_id: row.get(8)?,
                archived: row.get::<_, i64>(9)? != 0,
                ephemeral: row.get::<_, i64>(10)? != 0,
                memory_mode: serde_json::from_str(&row.get::<_, String>(11)?)
                    .unwrap_or(MemoryMode::Enabled),
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

fn enum_json<T: serde::Serialize>(value: &T) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

fn json_text(value: Value) -> Result<String> {
    Ok(serde_json::to_string(&value)?)
}
