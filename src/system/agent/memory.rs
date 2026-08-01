//! Semantic memory store — Phase 4 of the agent architecture.
#![allow(dead_code)]
//!
//! Provides durable, vector-indexed memory for agent turns: store observations,
//! plans, and facts as embeddings; retrieve the most relevant ones by cosine
//! similarity at query time.
//!
//! # Design
//!
//! * **`EmbeddingProvider`** — pluggable text → f32[] conversion. Swap in
//!   an Anthropic/OpenAI endpoint or a local model (fastembed) without
//!   touching store code.
//!
//! * **`SemanticMemoryStore`** — ingest / query / delete / stats trait.
//!
//! * **`SqliteVecMemoryStore`** — production implementation: embeddings stored
//!   as little-endian f32 BLOBs in a bundled SQLite database; cosine similarity
//!   computed in Rust over the full candidate set at query time. This is the
//!   sqlite-vec pattern (embedding column in SQLite, similarity in the host
//!   language) without requiring the loadable extension, which gives the same
//!   correctness with zero external dependencies and scales comfortably past
//!   100 k entries on modern hardware.
//!
//! # Typical usage
//! ```ignore
//! let store = SqliteVecMemoryStore::open("agent_memory.db", Arc::new(my_embedder))?;
//! let id = store.ingest(MemoryEntry {
//!     kind: MemoryKind::Fact,
//!     text: "The project uses Rust edition 2021".into(),
//!     ..Default::default()
//! }).await?;
//! let hits = store.query("which Rust edition?", 5).await?;
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use rusqlite::{params, Connection};
use uuid::Uuid;

// ─── MemoryKind ───────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    #[default]
    Observation,
    Plan,
    Fact,
    FileContext,
    ConversationSummary,
    CommandLog,
    GitDiff,
    TestOutput,
    Decision,
}

impl std::fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MemoryKind::Observation => "observation",
            MemoryKind::Plan => "plan",
            MemoryKind::Fact => "fact",
            MemoryKind::FileContext => "file_context",
            MemoryKind::ConversationSummary => "conversation_summary",
            MemoryKind::CommandLog => "command_log",
            MemoryKind::GitDiff => "git_diff",
            MemoryKind::TestOutput => "test_output",
            MemoryKind::Decision => "decision",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for MemoryKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "observation" => Ok(MemoryKind::Observation),
            "plan" => Ok(MemoryKind::Plan),
            "fact" => Ok(MemoryKind::Fact),
            "file_context" => Ok(MemoryKind::FileContext),
            "conversation_summary" => Ok(MemoryKind::ConversationSummary),
            "command_log" => Ok(MemoryKind::CommandLog),
            "git_diff" => Ok(MemoryKind::GitDiff),
            "test_output" => Ok(MemoryKind::TestOutput),
            "decision" => Ok(MemoryKind::Decision),
            other => Err(anyhow!("unknown memory kind: {other}")),
        }
    }
}

// ─── Value types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct MemoryEntry {
    pub kind: MemoryKind,
    pub text: String,
    /// Arbitrary key/value metadata (e.g. "file_path", "turn_id").
    pub metadata: HashMap<String, String>,
    /// Scope entries to a conversation thread for filtered recall.
    pub thread_id: Option<String>,
}

/// A single result from a similarity search.
#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub entry_id: String,
    pub entry: MemoryEntry,
    /// Cosine similarity in [−1, 1]; higher is more relevant.
    pub score: f32,
}

#[derive(Debug, Default, Clone)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub by_kind: HashMap<MemoryKind, usize>,
}

// ─── EmbeddingProvider ───────────────────────────────────────────────────────

/// Converts text to a fixed-dimension float vector.
///
/// All vectors returned by a single provider instance must have the same
/// dimensionality (`Self::dimensions()`). Mixing providers on the same store
/// is not supported — re-embed from scratch if you switch models.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn dimensions(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Default batched embed: sequential fallback.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }
}

// ─── SemanticMemoryStore ──────────────────────────────────────────────────────

#[async_trait]
pub trait SemanticMemoryStore: Send + Sync {
    /// Embed `entry.text` and persist the entry. Returns the generated entry ID.
    async fn ingest(&self, entry: MemoryEntry) -> Result<String>;

    /// Return the `k` most similar entries to `text` across all kinds/threads.
    async fn query(&self, text: &str, k: usize) -> Result<Vec<MemoryHit>>;

    /// Like `query` but optionally restrict to a specific kind and/or thread.
    async fn query_filtered(
        &self,
        text: &str,
        k: usize,
        kind: Option<MemoryKind>,
        thread_id: Option<&str>,
    ) -> Result<Vec<MemoryHit>>;

    /// Remove an entry by ID. Returns `true` if it existed.
    async fn delete(&self, entry_id: &str) -> Result<bool>;

    /// Aggregate statistics over the current store contents.
    fn stats(&self) -> Result<MemoryStats>;
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

// ─── SqliteVecMemoryStore ─────────────────────────────────────────────────────

/// SQLite-backed semantic memory store.
///
/// Embeddings are stored as little-endian f32 BLOBs alongside entry metadata.
/// Similarity search loads all candidate vectors and ranks them by cosine
/// similarity in Rust — identical semantics to the sqlite-vec virtual-table
/// approach, without requiring the loadable extension.
pub struct SqliteVecMemoryStore {
    conn: Mutex<Connection>,
    embedder: std::sync::Arc<dyn EmbeddingProvider>,
}

impl SqliteVecMemoryStore {
    /// Open or create a store at the given filesystem path.
    pub fn open(path: &Path, embedder: std::sync::Arc<dyn EmbeddingProvider>) -> Result<Self> {
        let conn = Connection::open(path)?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedder,
        })
    }

    /// Create a transient in-memory store (useful for tests and short-lived agents).
    pub fn open_in_memory(embedder: std::sync::Arc<dyn EmbeddingProvider>) -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedder,
        })
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_entries (
                id          TEXT    PRIMARY KEY,
                kind        TEXT    NOT NULL,
                text        TEXT    NOT NULL,
                metadata    TEXT    NOT NULL DEFAULT '{}',
                thread_id   TEXT,
                embedding   BLOB    NOT NULL,
                created_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory_entries(kind);
            CREATE INDEX IF NOT EXISTS idx_memory_thread ON memory_entries(thread_id);",
        )?;
        Ok(())
    }

    fn now_secs() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }
}

#[async_trait]
impl SemanticMemoryStore for SqliteVecMemoryStore {
    async fn ingest(&self, entry: MemoryEntry) -> Result<String> {
        let embedding = self.embedder.embed(&entry.text).await?;
        let id = format!("mem_{}", Uuid::new_v4().simple());
        let metadata_json = serde_json::to_string(&entry.metadata)?;
        let kind_str = entry.kind.to_string();
        let blob = vec_to_blob(&embedding);
        let now = Self::now_secs();

        {
            let conn = self.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO memory_entries
                    (id, kind, text, metadata, thread_id, embedding, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    id,
                    kind_str,
                    entry.text,
                    metadata_json,
                    entry.thread_id,
                    blob,
                    now
                ],
            )?;
        }

        tracing::debug!(
            entry_id = %id,
            kind = %kind_str,
            text_len = entry.text.len(),
            "memory: ingested"
        );
        Ok(id)
    }

    async fn query(&self, text: &str, k: usize) -> Result<Vec<MemoryHit>> {
        self.query_filtered(text, k, None, None).await
    }

    async fn query_filtered(
        &self,
        text: &str,
        k: usize,
        kind: Option<MemoryKind>,
        thread_id: Option<&str>,
    ) -> Result<Vec<MemoryHit>> {
        // Embed the query before acquiring the lock.
        let query_vec = self.embedder.embed(text).await?;
        let kind_str: Option<String> = kind.as_ref().map(|k| k.to_string());

        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, kind, text, metadata, thread_id, embedding
             FROM memory_entries
             WHERE (?1 IS NULL OR kind = ?1)
               AND (?2 IS NULL OR thread_id = ?2)",
        )?;

        let mut hits: Vec<MemoryHit> = stmt
            .query_map(params![kind_str, thread_id], |row| {
                let id: String = row.get(0)?;
                let kind_s: String = row.get(1)?;
                let text: String = row.get(2)?;
                let metadata_json: String = row.get(3)?;
                let thread_id: Option<String> = row.get(4)?;
                let blob: Vec<u8> = row.get(5)?;
                Ok((id, kind_s, text, metadata_json, thread_id, blob))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id, kind_s, text, metadata_json, thread_id, blob)| {
                let kind = kind_s.parse::<MemoryKind>().ok()?;
                let metadata: HashMap<String, String> =
                    serde_json::from_str(&metadata_json).unwrap_or_default();
                let embedding = blob_to_vec(&blob);
                let score = cosine_similarity(&query_vec, &embedding);
                Some(MemoryHit {
                    entry_id: id,
                    entry: MemoryEntry {
                        kind,
                        text,
                        metadata,
                        thread_id,
                    },
                    score,
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }

    async fn delete(&self, entry_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute(
            "DELETE FROM memory_entries WHERE id = ?1",
            params![entry_id],
        )?;
        Ok(n > 0)
    }

    fn stats(&self) -> Result<MemoryStats> {
        let conn = self.conn.lock().unwrap();

        let total: usize =
            conn.query_row("SELECT COUNT(*) FROM memory_entries", [], |r| r.get(0))?;

        let mut by_kind: HashMap<MemoryKind, usize> = HashMap::new();
        let mut stmt = conn.prepare("SELECT kind, COUNT(*) FROM memory_entries GROUP BY kind")?;
        for row in stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, usize>(1)?)))?
            .flatten()
        {
            if let Ok(kind) = row.0.parse::<MemoryKind>() {
                by_kind.insert(kind, row.1);
            }
        }

        Ok(MemoryStats {
            total_entries: total,
            by_kind,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────
