use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::llm::{self, LlmClient, StreamChunk};
use crate::system::domain::{
    ConversationItem, ItemKind, ModelId, ModelSettings, ToolCall, ToolDefinition, ToolProviderKind,
    TokenUsage,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDescription {
    pub model_id: ModelId,
    pub provider_id: String,
    pub display_name: String,
    pub hidden: bool,
    pub is_default: bool,
    #[serde(default)]
    pub reasoning_efforts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRequest {
    pub model_settings: ModelSettings,
    #[serde(default)]
    pub system_instructions: Vec<String>,
    #[serde(default)]
    pub conversation_items: Vec<ConversationItem>,
    #[serde(default)]
    pub tool_definitions: Vec<ToolDefinition>,
    pub output_schema: Option<Value>,
}

#[derive(Debug, Clone)]
pub enum ModelStreamEvent {
    AssistantDelta(String),
    #[allow(dead_code)]
    ReasoningDelta(String),
    ToolCalls(Vec<ToolCall>),
    Completed(TokenUsage),
    Failed(String),
}

pub struct ModelGateway {
    client: Arc<dyn LlmClient>,
}

impl ModelGateway {
    pub fn new(client: Arc<dyn LlmClient>) -> Self {
        Self { client }
    }

    pub fn list_models(&self, settings: &ModelSettings) -> Vec<ModelDescription> {
        vec![ModelDescription {
            model_id: settings.model_id.clone(),
            provider_id: settings.provider_id.clone(),
            display_name: settings.model_id.clone(),
            hidden: false,
            is_default: true,
            reasoning_efforts: vec!["low".into(), "medium".into(), "high".into()],
        }]
    }

    pub async fn start_response(
        &self,
        request: ModelRequest,
    ) -> Result<mpsc::Receiver<ModelStreamEvent>> {
        let messages = build_llm_messages(&request.system_instructions, &request.conversation_items);
        let tools = request.tool_definitions.iter().map(|t| llm::ToolDefinition {
            typ: "function".into(),
            function: llm::FunctionDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        }).collect::<Vec<_>>();

        let (tx, rx) = mpsc::channel(256);
        let client = self.client.clone();
        let model_settings = request.model_settings.clone();

        tokio::spawn(async move {
            match client.stream(
                &messages, &tools, &model_settings.model_id, 0.2,
                model_settings.reasoning_effort.as_deref(), 8192,
            ).await {
                Ok(mut stream) => {
                    let mut usage = TokenUsage::default();
                    let mut partial_tools: HashMap<usize, (Option<String>, Option<String>, String)> = HashMap::new();
                    while let Some(chunk) = stream.recv().await {
                        match chunk {
                            StreamChunk::Text(text) => {
                                let _ = tx.send(ModelStreamEvent::AssistantDelta(text)).await;
                            }
                            StreamChunk::ToolCallDelta { index, id, name, arguments_delta } => {
                                let entry = partial_tools.entry(index).or_default();
                                if let Some(id) = id { entry.0 = Some(id); }
                                if let Some(name) = name { entry.1 = Some(name); }
                                if let Some(args) = arguments_delta { entry.2.push_str(&args); }
                            }
                            StreamChunk::Usage(tokens) => {
                                usage = TokenUsage {
                                    input_tokens: tokens.prompt_tokens as i64,
                                    output_tokens: tokens.completion_tokens as i64,
                                    cached_tokens: 0,
                                };
                            }
                            StreamChunk::Done => break,
                        }
                    }
                    if !partial_tools.is_empty() {
                        let calls = partial_tools.into_iter().map(|(_, (id, name, args))| ToolCall {
                            tool_call_id: id.unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple())),
                            provider_kind: ToolProviderKind::Builtin,
                            tool_name: name.unwrap_or_default(),
                            arguments: serde_json::from_str(&args).unwrap_or_else(|_| serde_json::json!({})),
                        }).collect::<Vec<_>>();
                        let _ = tx.send(ModelStreamEvent::ToolCalls(calls)).await;
                    }
                    let _ = tx.send(ModelStreamEvent::Completed(usage)).await;
                }
                Err(stream_err) => {
                    match client.complete(
                        &messages, &tools, &model_settings.model_id, 0.2,
                        model_settings.reasoning_effort.as_deref(), 8192,
                    ).await {
                        Ok(response) => {
                            if !response.content.is_empty() {
                                let _ = tx.send(ModelStreamEvent::AssistantDelta(response.content)).await;
                            }
                            if !response.tool_calls.is_empty() {
                                let calls = response.tool_calls.into_iter().map(|c| ToolCall {
                                    tool_call_id: c.id,
                                    provider_kind: ToolProviderKind::Builtin,
                                    tool_name: c.function.name,
                                    arguments: serde_json::from_str(&c.function.arguments).unwrap_or_else(|_| serde_json::json!({})),
                                }).collect::<Vec<_>>();
                                let _ = tx.send(ModelStreamEvent::ToolCalls(calls)).await;
                            }
                            let usage = response.usage.unwrap_or_default();
                            let _ = tx.send(ModelStreamEvent::Completed(TokenUsage {
                                input_tokens: usage.prompt_tokens as i64,
                                output_tokens: usage.completion_tokens as i64,
                                cached_tokens: 0,
                            })).await;
                        }
                        Err(complete_err) => {
                            let _ = tx.send(ModelStreamEvent::Failed(format!(
                                "{stream_err}; fallback complete failed: {complete_err}"
                            ))).await;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
}

fn build_llm_messages(
    system_instructions: &[String],
    items: &[ConversationItem],
) -> Vec<llm::Message> {
    let mut messages = Vec::new();
    for instruction in system_instructions {
        messages.push(llm::Message::system(instruction.clone()));
    }
    for item in items {
        match item.kind {
            ItemKind::UserMessage => {
                if let Some(text) = item.payload.get("text").and_then(Value::as_str) {
                    messages.push(llm::Message::user(text.to_string()));
                }
            }
            ItemKind::UserAttachment => {
                if let Some(path) = item.payload.get("path").and_then(Value::as_str) {
                    messages.push(llm::Message::user(format!("[image:{}]", path)));
                }
            }
            ItemKind::AgentMessage | ItemKind::ReasoningSummary => {
                if let Some(text) = item.payload.get("text").and_then(Value::as_str) {
                    messages.push(llm::Message::assistant(text.to_string()));
                }
            }
            ItemKind::ToolCall => {
                if let Ok(tc) = serde_json::from_value::<ToolCall>(item.payload.clone()) {
                    messages.push(llm::Message {
                        role: llm::Role::Assistant,
                        content: String::new(),
                        tool_calls: vec![llm::ToolCall {
                            id: tc.tool_call_id,
                            function: llm::FunctionCall { name: tc.tool_name, arguments: tc.arguments.to_string() },
                        }],
                        tool_result: None,
                        think_content: String::new(),
                        timestamp: chrono::Utc::now(),
                    });
                }
            }
            ItemKind::ToolResult => {
                if let Ok(tr) = serde_json::from_value::<crate::system::domain::ToolResult>(item.payload.clone()) {
                    messages.push(llm::Message {
                        role: llm::Role::Tool,
                        content: String::new(),
                        tool_calls: Vec::new(),
                        tool_result: Some(llm::ToolResult {
                            tool_call_id: tr.tool_call_id,
                            name: String::new(),
                            result: Some(tr.output.to_string()),
                            error: tr.error_message,
                        }),
                        think_content: String::new(),
                        timestamp: chrono::Utc::now(),
                    });
                }
            }
            _ => {}
        }
    }
    messages
}
