//! Typed payload structs for [`RuntimeEvent`].
//!
//! `RuntimeEvent.payload` is `serde_json::Value` for wire-format flexibility
//! and to keep the broadcast channel type-erased. Consumers (TUI, server,
//! benchmarks, supervisor) historically reach into the JSON by hand. This
//! module defines a typed view over each payload shape so consumers can
//! deserialize once and pattern-match on a real Rust enum.
//!
//! Producers continue to publish JSON; this is a *consumer-side* type layer.
//! That keeps the change non-invasive and avoids touching every emit site.
//!
//! Several variants are not consumed yet — TUI / server still parse JSON
//! directly. The `dead_code` allow at module scope reflects that this is a
//! deliberate API surface for future migration, not unfinished work.
#![allow(dead_code)]

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::domain::{
    ApprovalResolution, FileChangeSummary, ItemKind, PendingApproval, RuntimeEvent,
    RuntimeEventKind, ThreadId, ThreadMetadata, TokenUsage, TurnId, TurnMetadata, TurnStatus,
};

/// Typed view of every event payload the runtime emits.
///
/// Construct via [`RuntimeEventPayload::from_event`] to decode the JSON in
/// [`RuntimeEvent::payload`]. Item-level payloads (`ItemStarted` /
/// `ItemUpdated` / `ItemCompleted`) carry polymorphic content (text deltas,
/// tool calls, tool results, …) so they remain partly typed: a typed envelope
/// plus a `Value` for the kind-specific body.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RuntimeEventPayload {
    ThreadStarted(ThreadMetadata),
    TurnStarted(TurnMetadata),
    ItemStarted(ItemEnvelope),
    ItemUpdated(ItemUpdate),
    ItemCompleted(ItemEnvelope),
    TurnCompleted(TurnCompletedPayload),
    TurnFailed(TurnFailedPayload),
    ApprovalRequested(PendingApproval),
    ApprovalResolved(ApprovalResolution),
    Warning(WarningPayload),
    Disconnected(DisconnectedPayload),
    CompactionStatus(CompactionStatusPayload),
    ChildAgentSpawned(ChildAgentSpawnedPayload),
}

/// Payload of `ItemStarted` and `ItemCompleted` events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemEnvelope {
    pub item_id: String,
    pub kind: ItemKind,
    /// Item-specific body: text for `AgentMessage`, a `ToolCall` for
    /// `ToolCall`, a `ToolResult` for `ToolResult`, etc.
    #[serde(default)]
    pub payload: Value,
}

/// Payload of `ItemUpdated` events. The runtime currently emits two shapes:
/// streaming text appends (`delta` + `mode = "append"`) and full payload
/// replacements (no `delta`, just `payload`). Both are represented here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemUpdate {
    pub item_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Used for replace-style updates (e.g. when ToolResult content lands
    /// after the tool finished running).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnCompletedPayload {
    pub turn_id: TurnId,
    pub thread_id: ThreadId,
    pub status: TurnStatus,
    pub token_usage: TokenUsage,
    /// `TurnMetrics` carries `agent_iterations`, `tool_call_count`,
    /// `elapsed_ms`, and `file_changes`. Kept as `Value` here because the
    /// metrics struct is internal to the executor; consumers that need the
    /// metrics field can `serde_json::from_value` it themselves.
    #[serde(default)]
    pub metrics: Value,
    /// Convenience: pre-extracted file changes from `metrics.fileChanges`,
    /// since benchmarks read this directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_changes: Option<Vec<FileChangeSummary>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnFailedPayload {
    /// Stable error label (e.g. `loop_limit`, `empty_response`,
    /// `context_overflow`). Use this for programmatic decisions.
    pub kind: String,
    /// Human-readable explanation of the failure.
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WarningPayload {
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisconnectedPayload {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionStatusPayload {
    /// One of: "started", "completed", "failed".
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Payload for `ChildAgentSpawned` events. Ties a child thread/turn to the
/// parent's `tool_call_id` so consumers (TUI, benchmarks) can render the
/// agent tree without inferring the relationship from `agent_depth`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChildAgentSpawnedPayload {
    pub parent_thread_id: ThreadId,
    pub parent_turn_id: TurnId,
    pub parent_tool_call_id: String,
    pub child_thread_id: ThreadId,
    pub child_turn_id: TurnId,
    /// User-visible label — typically the first line of the spawn prompt.
    pub label: String,
}

impl RuntimeEventPayload {
    /// Decode `event.payload` according to `event.kind`.
    pub fn from_event(event: &RuntimeEvent) -> Result<Self> {
        match event.kind {
            RuntimeEventKind::ThreadStarted => Ok(Self::ThreadStarted(decode(
                "ThreadStarted",
                &event.payload,
            )?)),
            RuntimeEventKind::TurnStarted => {
                Ok(Self::TurnStarted(decode("TurnStarted", &event.payload)?))
            }
            RuntimeEventKind::ItemStarted => {
                Ok(Self::ItemStarted(decode("ItemStarted", &event.payload)?))
            }
            RuntimeEventKind::ItemUpdated => {
                Ok(Self::ItemUpdated(decode("ItemUpdated", &event.payload)?))
            }
            RuntimeEventKind::ItemCompleted => Ok(Self::ItemCompleted(decode(
                "ItemCompleted",
                &event.payload,
            )?)),
            RuntimeEventKind::TurnCompleted => {
                let mut payload: TurnCompletedPayload = decode("TurnCompleted", &event.payload)?;
                if payload.file_changes.is_none() {
                    payload.file_changes = payload.metrics.get("fileChanges").and_then(|v| {
                        serde_json::from_value::<Vec<FileChangeSummary>>(v.clone()).ok()
                    });
                }
                Ok(Self::TurnCompleted(payload))
            }
            RuntimeEventKind::TurnFailed => {
                Ok(Self::TurnFailed(decode("TurnFailed", &event.payload)?))
            }
            RuntimeEventKind::ApprovalRequested => Ok(Self::ApprovalRequested(decode(
                "ApprovalRequested",
                &event.payload,
            )?)),
            RuntimeEventKind::ApprovalResolved => Ok(Self::ApprovalResolved(decode(
                "ApprovalResolved",
                &event.payload,
            )?)),
            RuntimeEventKind::Warning => Ok(Self::Warning(decode("Warning", &event.payload)?)),
            RuntimeEventKind::Disconnected => {
                Ok(Self::Disconnected(decode("Disconnected", &event.payload)?))
            }
            RuntimeEventKind::CompactionStatus => Ok(Self::CompactionStatus(decode(
                "CompactionStatus",
                &event.payload,
            )?)),
            RuntimeEventKind::ChildAgentSpawned => Ok(Self::ChildAgentSpawned(decode(
                "ChildAgentSpawned",
                &event.payload,
            )?)),
        }
    }
}

impl RuntimeEvent {
    /// Decode this event's JSON payload into a [`RuntimeEventPayload`].
    pub fn parsed_payload(&self) -> Result<RuntimeEventPayload> {
        RuntimeEventPayload::from_event(self)
    }
}

fn decode<T: for<'de> Deserialize<'de>>(label: &str, value: &Value) -> Result<T> {
    serde_json::from_value(value.clone())
        .map_err(|e| anyhow!("event_payload_decode_failed: {label}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::domain::now_millis;
    use serde_json::json;

    fn evt(kind: RuntimeEventKind, payload: Value) -> RuntimeEvent {
        RuntimeEvent {
            event_id: "e1".into(),
            kind,
            thread_id: Some("t1".into()),
            turn_id: Some("u1".into()),
            sequence: 1,
            payload,
            emitted_at: now_millis(),
        }
    }

    #[test]
    fn warning_payload_roundtrips() {
        let e = evt(RuntimeEventKind::Warning, json!({ "message": "heads up" }));
        match e.parsed_payload().unwrap() {
            RuntimeEventPayload::Warning(p) => assert_eq!(p.message, "heads up"),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn turn_failed_payload_decodes() {
        let e = evt(
            RuntimeEventKind::TurnFailed,
            json!({ "kind": "empty_response", "reason": "no text" }),
        );
        match e.parsed_payload().unwrap() {
            RuntimeEventPayload::TurnFailed(p) => {
                assert_eq!(p.kind, "empty_response");
                assert_eq!(p.reason, "no text");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn turn_completed_extracts_file_changes_from_metrics() {
        let e = evt(
            RuntimeEventKind::TurnCompleted,
            json!({
                "turnId": "u1",
                "threadId": "t1",
                "status": "COMPLETED",
                "tokenUsage": { "inputTokens": 0, "outputTokens": 0, "cachedTokens": 0 },
                "metrics": {
                    "agentIterations": 1,
                    "toolCallCount": 0,
                    "elapsedMs": 10,
                    "fileChanges": [
                        { "path": "x.txt", "kind": "create", "linesAdded": 1, "linesRemoved": 0, "diff": "" }
                    ]
                }
            }),
        );
        match e.parsed_payload().unwrap() {
            RuntimeEventPayload::TurnCompleted(p) => {
                let changes = p.file_changes.expect("should pre-extract file changes");
                assert_eq!(changes.len(), 1);
                assert_eq!(changes[0].path, "x.txt");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn item_update_handles_streaming_delta_shape() {
        let e = evt(
            RuntimeEventKind::ItemUpdated,
            json!({ "itemId": "item_x", "delta": "hi", "mode": "append" }),
        );
        match e.parsed_payload().unwrap() {
            RuntimeEventPayload::ItemUpdated(u) => {
                assert_eq!(u.item_id, "item_x");
                assert_eq!(u.delta.as_deref(), Some("hi"));
                assert_eq!(u.mode.as_deref(), Some("append"));
                assert!(u.payload.is_none());
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn unknown_payload_shape_returns_decode_error() {
        let e = evt(RuntimeEventKind::Warning, json!({ "wrong": true }));
        let err = e.parsed_payload().unwrap_err().to_string();
        assert!(err.contains("Warning"), "error should mention kind: {err}");
    }
}
