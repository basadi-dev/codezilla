//! Typed memory-operation schema for a post-trained memory controller.
//!
//! A controller model should emit this JSON shape. Runtime code can validate it
//! and apply the operations deterministically instead of letting free-form model
//! text mutate memory.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::system::persistence::{ConversationMemoryRecord, PersistenceManager};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MemoryControllerPlan {
    #[serde(default)]
    pub search: Vec<MemorySearchOp>,
    #[serde(default)]
    pub store: Vec<MemoryStoreOp>,
    #[serde(default)]
    pub update: Vec<MemoryUpdateOp>,
    #[serde(default)]
    pub forget: Vec<MemoryForgetOp>,
    #[serde(default)]
    pub answer_strategy: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemorySearchOp {
    pub query: String,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreOp {
    pub kind: String,
    #[serde(default = "default_scope")]
    pub scope: String,
    pub content: String,
    #[serde(default = "default_importance")]
    pub importance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryUpdateOp {
    pub memory_id: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryForgetOp {
    pub memory_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AppliedMemoryPlan {
    #[serde(default)]
    pub search_results: Vec<MemorySearchResult>,
    #[serde(default)]
    pub stored_ids: Vec<String>,
    pub updated: usize,
    pub forgotten: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemorySearchResult {
    pub query: String,
    pub memories: Vec<ConversationMemoryRecord>,
}

pub fn apply_memory_plan(
    persistence: &Arc<PersistenceManager>,
    thread_id: &str,
    turn_id: &str,
    plan: &MemoryControllerPlan,
) -> Result<AppliedMemoryPlan> {
    validate_plan(plan)?;
    let mut applied = AppliedMemoryPlan::default();

    for search in &plan.search {
        let memories = persistence.search_conversation_memories(
            thread_id,
            &search.query,
            search.limit.clamp(1, 50),
        )?;
        applied.search_results.push(MemorySearchResult {
            query: search.query.clone(),
            memories,
        });
    }

    for store in &plan.store {
        let id = persistence.append_conversation_memory(
            thread_id,
            turn_id,
            &store.kind,
            &store.scope,
            &store.content,
            store.importance.clamp(0.0, 1.0),
        )?;
        applied.stored_ids.push(id);
    }

    for update in &plan.update {
        if persistence.update_conversation_memory(
            &update.memory_id,
            thread_id,
            update.kind.as_deref(),
            update.scope.as_deref(),
            update.content.as_deref(),
            update.importance.map(|v| v.clamp(0.0, 1.0)),
        )? {
            applied.updated += 1;
        }
    }

    for forget in &plan.forget {
        if persistence.delete_conversation_memory(&forget.memory_id, thread_id)? {
            applied.forgotten += 1;
        }
    }

    Ok(applied)
}

fn validate_plan(plan: &MemoryControllerPlan) -> Result<()> {
    for search in &plan.search {
        require_non_empty("search.query", &search.query)?;
    }
    for store in &plan.store {
        require_non_empty("store.kind", &store.kind)?;
        require_scope(&store.scope)?;
        require_non_empty("store.content", &store.content)?;
    }
    for update in &plan.update {
        require_non_empty("update.memoryId", &update.memory_id)?;
        if let Some(scope) = &update.scope {
            require_scope(scope)?;
        }
    }
    for forget in &plan.forget {
        require_non_empty("forget.memoryId", &forget.memory_id)?;
    }
    Ok(())
}

fn require_non_empty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(anyhow!("{field} must not be empty"))
    } else {
        Ok(())
    }
}

fn require_scope(scope: &str) -> Result<()> {
    if matches!(scope, "thread" | "global") {
        Ok(())
    } else {
        Err(anyhow!("scope must be \"thread\" or \"global\""))
    }
}

fn default_search_limit() -> usize {
    6
}

fn default_scope() -> String {
    "thread".into()
}

fn default_importance() -> f32 {
    0.7
}
