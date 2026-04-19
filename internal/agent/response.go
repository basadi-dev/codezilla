package agent

import (
	"context"
	"fmt"
	"strings"
	"time"

	"codezilla/internal/tools"
	anyllm "github.com/mozilla-ai/any-llm-go"
)

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

func (a *agent) generateCompletion(ctx context.Context, modelOverride string, llmTools []anyllm.Tool) (*anyllm.ChatCompletion, error) {
	systemPrompt := a.buildSystemPrompt()
	rawMessages := a.buildChatMessages()

	var chatMessages []anyllm.Message
	if systemPrompt != "" {
		chatMessages = append(chatMessages, anyllm.Message{Role: "system", Content: systemPrompt})
	}

	for _, msg := range rawMessages {
		if msg.Role != "system" {
			chatMessages = append(chatMessages, msg)
		}
	}

	if len(chatMessages) == 0 || (len(chatMessages) == 1 && chatMessages[0].Role == "system") {
		chatMessages = append(chatMessages, anyllm.Message{
			Role: "user", Content: "Hello",
		})
	}

	targetModel := a.config.Model
	if modelOverride != "" {
		targetModel = modelOverride
	}

	a.logger.Debug("Sending Chat request to LLM Provider",
		"provider", a.config.Provider,
		"model", targetModel,
		"messages", len(chatMessages),
		"tools", len(llmTools),
		"temperature", a.config.Temperature)

	startTime := time.Now()
	response, err := a.llmClient.Complete(ctx, a.config.Provider, targetModel, chatMessages, a.config.Temperature, a.config.ReasoningEffort, llmTools)
	duration := time.Since(startTime)

	if err != nil {
		a.logger.Error("Failed to get response from LLM API", "error", err, "duration", duration.String())
		return nil, fmt.Errorf("failed to get response from LLM API: %w", err)
	}

	return response, nil
}

func (a *agent) buildSystemPrompt() string {
	messages := a.context.GetFormattedMessages()
	var systemParts []string
	for _, msg := range messages {
		if msg.Role == "system" && msg.ContentString() != "" {
			systemParts = append(systemParts, msg.ContentString())
		}
	}
	systemPrompt := strings.Join(systemParts, "\n\n")

	// We still append the prompt descriptions, but without the hardcoded XML XML/Bash rules if desired by Strategy.
	// We'll keep the basic description for backward compatibility logic inside the prompt.go strategy.
	if a.toolRegistry != nil && len(a.toolRegistry.ListTools()) > 0 {
		if !strings.Contains(systemPrompt, "You have access to the following tools") {
			var sb strings.Builder
			sb.WriteString("You have access to the following tools:\n\n")
			for _, tool := range a.toolRegistry.ListTools() {
				sb.WriteString(fmt.Sprintf("- **%s**: %s\n", tool.Name(), tool.Description()))
			}
			systemPrompt = systemPrompt + "\n\n" + sb.String()
		}
	}

	return systemPrompt
}

func (a *agent) buildChatMessages() []anyllm.Message {
	return a.context.GetFormattedMessages()
}
