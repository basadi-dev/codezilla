package agent

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// generateResponse generates a response from the LLM using the Chat API
func (a *agent) generateResponse(ctx context.Context) (string, error) {
	systemPrompt := a.buildSystemPrompt()
	chatMessages := a.buildChatMessages()

	a.logger.Debug("Sending Chat request to LLM Provider",
		"provider", a.config.Provider,
		"model", a.config.Model,
		"messages", len(chatMessages),
		"temperature", a.config.Temperature,
		"systemLen", len(systemPrompt))

	// Ensure system prompt is the first message if needed
	if len(chatMessages) > 0 && chatMessages[0].Role != "system" {
		chatMessages = append([]anyllm.Message{
			{Role: "system", Content: systemPrompt},
		}, chatMessages...)
	} else if len(chatMessages) == 0 {
		chatMessages = append(chatMessages, anyllm.Message{Role: "system", Content: systemPrompt})
		chatMessages = append(chatMessages, anyllm.Message{Role: "user", Content: "Hello"})
	}

	startTime := time.Now()
	response, err := a.llmClient.Complete(ctx, a.config.Provider, a.config.Model, chatMessages, a.config.Temperature)
	duration := time.Since(startTime)

	if err != nil {
		a.logger.Error("Failed to get response from LLM API", "error", err, "duration", duration.String())
		return "", fmt.Errorf("failed to get response from LLM API: %w", err)
	}

	a.logger.Debug("Received Chat response",
		"duration", duration.String(),
		"model", a.config.Model)

	cleanResponse := ""
	if len(response.Choices) > 0 {
		cleanResponse = strings.TrimSpace(response.Choices[0].Message.ContentString())
	}

	if cleanResponse == "" {
		a.logger.Warn("Empty response from model, using fallback")
		cleanResponse = "I'm sorry, I wasn't able to generate a proper response. Could you please try again or rephrase your question?"
	}

	return cleanResponse, nil
}

// buildSystemPrompt assembles the full system prompt including tool info.
func (a *agent) buildSystemPrompt() string {
	messages := a.context.GetFormattedMessages()
	var systemParts []string

	for _, msg := range messages {
		if msg.Role == "system" && msg.ContentString() != "" {
			systemParts = append(systemParts, msg.ContentString())
		}
	}

	systemPrompt := strings.Join(systemParts, "\n\n")

	// Append tool info if not already present
	if a.toolRegistry != nil && len(a.toolRegistry.ListTools()) > 0 {
		if !strings.Contains(systemPrompt, "You have access to the following tools") {
			var toolsInfo strings.Builder
			toolsInfo.WriteString("You have access to the following tools:\n\n")
			for _, tool := range a.toolRegistry.ListTools() {
				toolsInfo.WriteString(fmt.Sprintf("- %s: %s\n", tool.Name(), tool.Description()))
			}
			toolsInfo.WriteString("\nWhen you need to use a tool, format your response in one of these ways:\n\n")
			toolsInfo.WriteString("1. XML format:\n<tool>\n  <name>toolName</name>\n  <params>\n    <param1>value1</param1>\n  </params>\n</tool>\n\n")
			toolsInfo.WriteString("2. JSON format:\n```json\n{\"tool\": \"toolName\", \"params\": {\"param1\": \"value1\"}}\n```\n\n")
			toolsInfo.WriteString("3. For shell commands, use code blocks:\n```bash\ncommand here\n```\n\n")
			systemPrompt = systemPrompt + "\n\n" + toolsInfo.String()
		}
	}

	return systemPrompt
}

// buildChatMessages returns the cached slice of `anyllm.Message`
func (a *agent) buildChatMessages() []anyllm.Message {
	return a.context.GetFormattedMessages()
}

// runToolLoop executes tool calls found in a response and returns the follow-up
// response after all tool calls complete.  It mutates agent context in-place.
// Returns the final response text (no tool calls remain).
func (a *agent) runToolLoop(ctx context.Context, initialResponse string) (string, error) {
	finalResponse := initialResponse
	maxIterations := 10
	iterations := 0

	for iterations < maxIterations {
		iterations++

		toolCalls := a.extractAllToolCalls(finalResponse)
		if len(toolCalls) == 0 {
			a.logger.Debug("No more tool calls detected, reached final response", "iterations", iterations)
			break
		}

		// Strip tool call markup; keep any surrounding text
		narration := a.stripToolCallMarkup(finalResponse)

		a.logger.Debug("Processing tool calls",
			"count", len(toolCalls),
			"iteration", iterations)

		for _, tc := range toolCalls {
			a.logger.Info("Executing tool from response",
				"tool", tc.toolCall.ToolName,
				"iteration", iterations)

			// Add tool call to context
			a.context.AddToolCallMessage(tc.toolCall.ToolName, tc.toolCall.Params)

			result, err := a.ExecuteTool(ctx, tc.toolCall.ToolName, tc.toolCall.Params)
			if err != nil {
				errMsg := fmt.Sprintf("Tool execution failed: %v", err)
				a.logger.Error("Tool execution failed during tool loop",
					"tool", tc.toolCall.ToolName,
					"error", err)
				a.context.AddToolResultMessage(nil, err)

				// Let the LLM know about the failure and try again
				a.context.AddUserMessage(fmt.Sprintf("The tool '%s' failed with error: %s. Please try a different approach or explain the issue.", tc.toolCall.ToolName, errMsg))
			} else {
				a.logger.Debug("Tool executed successfully",
					"tool", tc.toolCall.ToolName,
					"resultType", fmt.Sprintf("%T", result))
				a.context.AddToolResultMessage(result, nil)
			}
		}

		// Generate follow-up response
		var err error
		a.logger.Debug("Generating follow-up response after tool execution",
			"iteration", iterations,
			"narration", narration)

		finalResponse, err = a.generateResponse(ctx)
		if err != nil {
			return "", fmt.Errorf("failed to generate follow-up response: %w", err)
		}
	}

	return finalResponse, nil
}

// stripToolCallMarkup removes tool call markup from response text, leaving user-facing prose.
func (a *agent) stripToolCallMarkup(response string) string {
	// Remove XML, JSON and bash code blocks that are tool calls
	stripped := response

	// These are the same patterns used in extractToolCall
	jsonPattern := regexp.MustCompile("(?s)```json\\s*\\n(.*?)\\n?```")
	bashPattern := regexp.MustCompile("(?s)```(bash|sh|shell|terminal|console)\\s*\\n(.*?)\\n?```")
	xmlPattern := regexp.MustCompile(`(?s)<tool>[\s\n]*(.*?)[\s\n]*</tool>`)

	stripped = jsonPattern.ReplaceAllString(stripped, "")
	stripped = bashPattern.ReplaceAllString(stripped, "")
	stripped = xmlPattern.ReplaceAllString(stripped, "")

	return strings.TrimSpace(stripped)
}
