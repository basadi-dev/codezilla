package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"codezilla/internal/tools"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// ---------------------------------------------------------------------------
// Tool schema conversion
// ---------------------------------------------------------------------------

// buildLLMTools converts the registered tool specs into anyllm.Tool format so
// they can be submitted to the LLM API as a native function-calling schema.
func (a *agent) buildLLMTools() []anyllm.Tool {
	if a.toolRegistry == nil {
		return nil
	}
	specs := a.toolRegistry.GetToolSpecs()
	result := make([]anyllm.Tool, 0, len(specs))
	for _, spec := range specs {
		result = append(result, anyllm.Tool{
			Type: "function",
			Function: anyllm.Function{
				Name:        spec.Name,
				Description: spec.Description,
				Parameters:  toolJSONSchemaToMap(spec.ParameterSchema),
			},
		})
	}
	return result
}

// toolJSONSchemaToMap converts the internal JSONSchema type to the
// map[string]any format expected by the any-llm-go API.
func toolJSONSchemaToMap(schema tools.JSONSchema) map[string]any {
	m := map[string]any{"type": schema.Type}
	if schema.Description != "" {
		m["description"] = schema.Description
	}
	if len(schema.Properties) > 0 {
		props := map[string]any{}
		for k, v := range schema.Properties {
			props[k] = toolJSONSchemaToMap(v)
		}
		m["properties"] = props
	}
	if len(schema.Required) > 0 {
		m["required"] = schema.Required
	}
	if schema.Items != nil {
		m["items"] = toolJSONSchemaToMap(*schema.Items)
	}
	if len(schema.Enum) > 0 {
		m["enum"] = schema.Enum
	}
	return m
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// extractCompletionContent returns the text content from a completion message.
// For thinking/reasoning models (e.g. gpt-oss, qwen3) the response lives in
// the Reasoning field when Content is empty.
func extractCompletionContent(msg anyllm.Message) string {
	content := strings.TrimSpace(msg.ContentString())
	if content != "" {
		return content
	}
	if msg.Reasoning != nil {
		content = strings.TrimSpace(msg.Reasoning.Content)
	}
	return content
}

// parseToolCallArguments unmarshals the JSON arguments string inside a native
// tool call into a plain map.
func parseToolCallArguments(argsJSON string) map[string]interface{} {
	params := make(map[string]interface{})
	if argsJSON != "" {
		_ = json.Unmarshal([]byte(argsJSON), &params)
	}
	return params
}

// ---------------------------------------------------------------------------
// Core LLM call
// ---------------------------------------------------------------------------

// generateCompletion is the low-level method that sends a request to the LLM
// with native tool schemas included. All higher-level methods call this.
func (a *agent) generateCompletion(ctx context.Context, llmTools []anyllm.Tool) (*anyllm.ChatCompletion, error) {
	systemPrompt := a.buildSystemPrompt()
	chatMessages := a.buildChatMessages()

	// Ensure system prompt is first
	if len(chatMessages) > 0 && chatMessages[0].Role != "system" {
		chatMessages = append([]anyllm.Message{{Role: "system", Content: systemPrompt}}, chatMessages...)
	} else if len(chatMessages) == 0 {
		chatMessages = []anyllm.Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: "Hello"},
		}
	}

	a.logger.Debug("Sending Chat request to LLM Provider",
		"provider", a.config.Provider,
		"model", a.config.Model,
		"messages", len(chatMessages),
		"tools", len(llmTools),
		"temperature", a.config.Temperature)

	startTime := time.Now()
	response, err := a.llmClient.Complete(ctx, a.config.Provider, a.config.Model, chatMessages, a.config.Temperature, llmTools)
	duration := time.Since(startTime)

	if err != nil {
		a.logger.Error("Failed to get response from LLM API", "error", err, "duration", duration.String())
		return nil, fmt.Errorf("failed to get response from LLM API: %w", err)
	}

	nativeToolCalls := 0
	if len(response.Choices) > 0 {
		nativeToolCalls = len(response.Choices[0].Message.ToolCalls)
	}
	a.logger.Debug("Received Chat response",
		"duration", duration.String(),
		"model", a.config.Model,
		"choiceCount", len(response.Choices),
		"nativeToolCalls", nativeToolCalls)

	return response, nil
}

// generateResponse is a convenience wrapper around generateCompletion that
// returns plain text. It includes retry logic for models that initially return
// empty content (e.g. thinking models on the first call).
func (a *agent) generateResponse(ctx context.Context) (string, error) {
	llmTools := a.buildLLMTools()

	completion, err := a.generateCompletion(ctx, llmTools)
	if err != nil {
		return "", err
	}

	content := ""
	if len(completion.Choices) > 0 {
		content = extractCompletionContent(completion.Choices[0].Message)
	}

	// Retry once with a nudge if model returned nothing
	if content == "" {
		a.logger.Warn("Empty response from model, retrying with nudge",
			"provider", a.config.Provider, "model", a.config.Model)

		// Build messages for the nudge call without touching agent context
		sysPrompt := a.buildSystemPrompt()
		msgs := a.buildChatMessages()
		if len(msgs) > 0 && msgs[0].Role != "system" {
			msgs = append([]anyllm.Message{{Role: "system", Content: sysPrompt}}, msgs...)
		}
		nudgeMsgs := append(msgs, anyllm.Message{
			Role:    "user",
			Content: "Please respond to my previous message.",
		})

		retryStart := time.Now()
		retryResp, retryErr := a.llmClient.Complete(ctx, a.config.Provider, a.config.Model, nudgeMsgs, a.config.Temperature, llmTools)
		retryDuration := time.Since(retryStart)

		if retryErr == nil && len(retryResp.Choices) > 0 {
			content = extractCompletionContent(retryResp.Choices[0].Message)
			if content != "" {
				a.logger.Info("Retry succeeded after empty initial response", "duration", retryDuration.String())
				return content, nil
			}
		}

		a.logger.Warn("Retry also returned empty, using fallback message",
			"retryError", retryErr, "retryDuration", retryDuration.String())
		return "I'm sorry, I wasn't able to generate a proper response. Could you please try again or rephrase your question?", nil
	}

	return content, nil
}

// ---------------------------------------------------------------------------
// Completion dispatch
// ---------------------------------------------------------------------------

// handleCompletion processes a completed API response. It dispatches to:
//  1. Native tool calls (structured ToolCalls field) — highest priority
//  2. Text-pattern tool calls (XML/JSON embedded in response text) — fallback
//  3. Plain text response — final answer
func (a *agent) handleCompletion(ctx context.Context, completion *anyllm.ChatCompletion, llmTools []anyllm.Tool) (string, error) {
	if len(completion.Choices) == 0 {
		return "I'm sorry, I wasn't able to generate a proper response. Please try again.", nil
	}
	msg := completion.Choices[0].Message

	// 1. Native structured tool calls
	if len(msg.ToolCalls) > 0 {
		a.logger.Info("Processing native tool calls from completion", "count", len(msg.ToolCalls))
		return a.executeNativeToolCalls(ctx, msg.ToolCalls, llmTools)
	}

	// 2. Extract text (handles Content + Reasoning for thinking models)
	content := extractCompletionContent(msg)
	if content == "" {
		return "I'm sorry, I wasn't able to generate a proper response. Please try again.", nil
	}

	// 3. Text-pattern tool calls (fallback for models that embed XML/JSON in text)
	return a.runToolLoop(ctx, content)
}

// ---------------------------------------------------------------------------
// Native tool call executor
// ---------------------------------------------------------------------------

// executeNativeToolCalls executes a slice of native API ToolCalls, injects the
// results into the conversation context, then generates the follow-up response.
// Loops if the follow-up also returns tool calls (up to maxIter times).
func (a *agent) executeNativeToolCalls(ctx context.Context, nativeCalls []anyllm.ToolCall, llmTools []anyllm.Tool) (string, error) {
	const maxIter = 10
	for i := 0; i < maxIter; i++ {
		if len(nativeCalls) == 0 {
			break
		}

		a.logger.Debug("Executing native tool calls", "count", len(nativeCalls), "iteration", i)

		// Record the assistant's tools execution intent
		var thinkContent string
		if len(nativeCalls) > 0 && nativeCalls[0].Function.Name != "" {
			thinkContent = fmt.Sprintf("I am calling %d tools natively...", len(nativeCalls))
		}
		a.context.AddNativeToolCallsMessage(thinkContent, nativeCalls)

		for _, tc := range nativeCalls {
			params := parseToolCallArguments(tc.Function.Arguments)
			a.logger.Info("Executing native tool call", "tool", tc.Function.Name)

			if a.config.OnToolExecution != nil {
				a.config.OnToolExecution(tc.Function.Name, params)
			}

			result, err := a.ExecuteTool(ctx, tc.Function.Name, params)
			if err != nil {
				a.logger.Error("Native tool call failed", "tool", tc.Function.Name, "error", err)
				a.context.AddNativeToolResultMessage(tc.ID, nil, fmt.Errorf("tool `%s` failed: %w", tc.Function.Name, err))
			} else {
				a.logger.Debug("Native tool executed successfully", "tool", tc.Function.Name)
				a.context.AddNativeToolResultMessage(tc.ID, result, nil)
			}
		}

		// Generate the follow-up response
		completion, err := a.generateCompletion(ctx, llmTools)
		if err != nil {
			return "", fmt.Errorf("failed to generate follow-up after tool calls: %w", err)
		}
		if len(completion.Choices) == 0 {
			return "No response after tool execution.", nil
		}

		msg := completion.Choices[0].Message

		// More native tool calls?
		if len(msg.ToolCalls) > 0 {
			nativeCalls = msg.ToolCalls
			continue
		}

		// Final text response
		content := extractCompletionContent(msg)
		if content == "" {
			return "I wasn't able to generate a response after tool execution.", nil
		}
		return content, nil
	}
	return "Reached maximum tool call iterations.", nil
}

// ---------------------------------------------------------------------------
// Text-pattern tool loop (fallback)
// ---------------------------------------------------------------------------

// runToolLoop handles tool calls that are embedded as XML/JSON markup in the
// response text. This is the backward-compat path for models that don't
// support native function calling or that mix tool calls into their text.
func (a *agent) runToolLoop(ctx context.Context, initialResponse string) (string, error) {
	llmTools := a.buildLLMTools()
	finalResponse := initialResponse
	maxIterations := 10

	for i := 0; i < maxIterations; i++ {
		toolCalls := a.extractAllToolCalls(finalResponse)
		if len(toolCalls) == 0 {
			a.logger.Debug("No text-pattern tool calls detected", "iteration", i)
			break
		}

		a.logger.Debug("Processing text-pattern tool calls", "count", len(toolCalls), "iteration", i)

		for _, tc := range toolCalls {
			a.logger.Info("Executing text-pattern tool call",
				"tool", tc.toolCall.ToolName, "iteration", i)

			if a.config.OnToolExecution != nil {
				a.config.OnToolExecution(tc.toolCall.ToolName, tc.toolCall.Params)
			}
			a.context.AddToolCallMessage(tc.toolCall.ToolName, tc.toolCall.Params)

			result, err := a.ExecuteTool(ctx, tc.toolCall.ToolName, tc.toolCall.Params)
			if err != nil {
				a.logger.Error("Text-pattern tool call failed",
					"tool", tc.toolCall.ToolName, "error", err)
				a.context.AddToolResultMessage(nil, err)
				a.context.AddUserMessage(fmt.Sprintf(
					"Tool '%s' failed: %v. Please try a different approach.", tc.toolCall.ToolName, err))
			} else {
				a.logger.Debug("Text-pattern tool executed successfully", "tool", tc.toolCall.ToolName)
				a.context.AddToolResultMessage(result, nil)
			}
		}

		// Generate follow-up (with native tool support — handles cross-over)
		completion, err := a.generateCompletion(ctx, llmTools)
		if err != nil {
			return "", fmt.Errorf("failed to generate follow-up response: %w", err)
		}
		if len(completion.Choices) == 0 {
			break
		}

		msg := completion.Choices[0].Message

		// If the follow-up uses native tool calls, hand off to native executor
		if len(msg.ToolCalls) > 0 {
			return a.executeNativeToolCalls(ctx, msg.ToolCalls, llmTools)
		}

		finalResponse = extractCompletionContent(msg)
	}

	return finalResponse, nil
}

// ---------------------------------------------------------------------------
// System prompt / message builders
// ---------------------------------------------------------------------------

// buildSystemPrompt assembles the full system prompt.
// Tool names and descriptions are listed for model context; the format
// instructions (XML/JSON) have been removed because tool calling is now handled
// via the structured Tools API parameter.
func (a *agent) buildSystemPrompt() string {
	messages := a.context.GetFormattedMessages()
	var systemParts []string
	for _, msg := range messages {
		if msg.Role == "system" && msg.ContentString() != "" {
			systemParts = append(systemParts, msg.ContentString())
		}
	}
	systemPrompt := strings.Join(systemParts, "\n\n")

	// Append tool usage instructions. We provide code-block/XML fallbacks here 
	// for models that do not support or properly generate structured native Function calling.
	if a.toolRegistry != nil && len(a.toolRegistry.ListTools()) > 0 {
		if !strings.Contains(systemPrompt, "You have access to the following tools") {
			var sb strings.Builder
			sb.WriteString("You have access to the following tools:\n\n")
			for _, tool := range a.toolRegistry.ListTools() {
				sb.WriteString(fmt.Sprintf("- **%s**: %s\n", tool.Name(), tool.Description()))
			}
			sb.WriteString("\nWhen you need to use a tool, you should call it using the native function calling API if supported. Otherwise, format your response in one of these fallback ways:\n\n")
			sb.WriteString("1. XML format:\n<tool>\n  <name>toolName</name>\n  <params>\n    <param1>value1</param1>\n  </params>\n</tool>\n\n")
			sb.WriteString("2. JSON format:\n```json\n{\"tool\": \"toolName\", \"params\": {\"param1\": \"value1\"}}\n```\n\n")
			sb.WriteString("3. For shell commands, use code blocks:\n```bash\ncommand here\n```\n\n")
			sb.WriteString("Call these tools whenever the user asks you to read files, list files, execute commands, or perform file operations. Do not just describe what you plan to do — explicitly output the tool call formats above.")
			systemPrompt = systemPrompt + "\n\n" + sb.String()
		}
	}

	return systemPrompt
}

// buildChatMessages returns the conversation history as anyllm.Message slices.
func (a *agent) buildChatMessages() []anyllm.Message {
	return a.context.GetFormattedMessages()
}

// ---------------------------------------------------------------------------
// Markup helpers
// ---------------------------------------------------------------------------

// stripToolCallMarkup removes XML/JSON/bash tool call markup from response text.
func (a *agent) stripToolCallMarkup(response string) string {
	stripped := response
	jsonPattern := regexp.MustCompile("(?s)```json\\s*\\n(.*?)\\n?```")
	bashPattern := regexp.MustCompile("(?s)```(bash|sh|shell|terminal|console)\\s*\\n(.*?)\\n?```")
	xmlPattern := regexp.MustCompile(`(?s)<tool>[\s\n]*(.*?)[\s\n]*</tool>`)
	stripped = jsonPattern.ReplaceAllString(stripped, "")
	stripped = bashPattern.ReplaceAllString(stripped, "")
	stripped = xmlPattern.ReplaceAllString(stripped, "")
	return strings.TrimSpace(stripped)
}
