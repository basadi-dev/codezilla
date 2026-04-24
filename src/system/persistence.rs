use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::domain::{
    now_seconds, ConversationItem, ItemKind, MemoryMode, PersistedThread, ThreadFilter,
    ThreadMetadata, ThreadStatus, TurnMetadata, TurnStatus,
};

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
                token_usage_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
                params![
                    metadata.turn_id,
                    metadata.thread_id,
                    metadata.created_at,
                    metadata.updated_at,
                    enum_json(&metadata.status)?,
                    enum_json(&metadata.started_by_surface)?,
                    serde_json::to_string(&metadata.token_usage)?,
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
            SET updated_at = ?3, status = ?4, token_usage_json = ?5
            WHERE turn_id = ?1 AND thread_id = ?2
            "#,
                params![
                    turn.turn_id,
                    turn.thread_id,
                    turn.updated_at,
                    enum_json(&turn.status)?,
                    serde_json::to_string(&turn.token_usage)?,
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

    pub fn read_thread(&self, thread_id: &str) -> Result<PersistedThread> {
        self.with_conn(|conn| {
            let metadata = read_thread_metadata(conn, thread_id)?
                .with_context(|| format!("thread_not_found: {thread_id}"))?;

            let turns = {
                let mut stmt = conn.prepare(
                    "SELECT turn_id, thread_id, created_at, updated_at, status, started_by_surface, token_usage_json
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
                token_usage_json TEXT NOT NULL
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
            "#,
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
