package agent

import (
	"fmt"
	"sync"

	"codezilla/llm"
	"codezilla/pkg/logger"
)

// TokenCounter interface for counting tokens
type TokenCounter interface {
	Count(messages []llm.Message) int
	CountString(s string) int
}

// ContextManager manages conversation context with token limits
type ContextManager struct {
	messages     []llm.Message
	maxTokens    int
	tokenCounter TokenCounter
	logger       *logger.Logger
	mu           sync.RWMutex
}

// NewContextManager creates a new context manager
func NewContextManager(maxTokens int, counter TokenCounter, log *logger.Logger) *ContextManager {
	if counter == nil {
		counter = &HeuristicCounter{}
	}
	if log == nil {
		// Create a default logger
		log, _ = logger.New(logger.Config{Silent: true})
	}

	return &ContextManager{
		messages:     []llm.Message{},
		maxTokens:    maxTokens,
		tokenCounter: counter,
		logger:       log,
	}
}

// AddSystemMessage adds a system message
func (cm *ContextManager) AddSystemMessage(content string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.messages = append(cm.messages, llm.Message{
		Role:    "system",
		Content: content,
	})
	cm.compress()
}

// AddUserMessage adds a user message
func (cm *ContextManager) AddUserMessage(content string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.messages = append(cm.messages, llm.Message{
		Role:    "user",
		Content: content,
	})
	cm.compress()
}

// AddAssistantMessage adds an assistant message
func (cm *ContextManager) AddAssistantMessage(content string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.messages = append(cm.messages, llm.Message{
		Role:    "assistant",
		Content: content,
	})
	cm.compress()
}

// AddToolResultMessage adds a tool result as a message
func (cm *ContextManager) AddToolResultMessage(toolName string, result interface{}, err error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	var content string
	if err != nil {
		content = "Tool execution failed: " + err.Error()
	} else {
		content = formatToolResultSimple(result)
	}

	cm.messages = append(cm.messages, llm.Message{
		Role:    "tool",
		Content: content,
	})
	cm.compress()
}

// GetMessages returns a copy of all messages
func (cm *ContextManager) GetMessages() []llm.Message {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Return a copy to prevent external modification
	messagesCopy := make([]llm.Message, len(cm.messages))
	copy(messagesCopy, cm.messages)
	return messagesCopy
}

// Clear removes all non-system messages
func (cm *ContextManager) Clear() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	var systemMessages []llm.Message
	for _, msg := range cm.messages {
		if msg.Role == "system" {
			systemMessages = append(systemMessages, msg)
		}
	}
	cm.messages = systemMessages
}

// compress removes old messages if we exceed token limit
func (cm *ContextManager) compress() {
	totalTokens := cm.tokenCounter.Count(cm.messages)

	if totalTokens <= cm.maxTokens {
		return
	}

	cm.logger.Debug("Compressing context", "currentTokens", totalTokens, "maxTokens", cm.maxTokens)

	// Keep system messages and recent messages
	var systemMessages []llm.Message
	var otherMessages []llm.Message

	for _, msg := range cm.messages {
		if msg.Role == "system" {
			systemMessages = append(systemMessages, msg)
		} else {
			otherMessages = append(otherMessages, msg)
		}
	}

	// Start with system messages
	newMessages := systemMessages

	// Add recent messages until we reach token limit
	for i := len(otherMessages) - 1; i >= 0; i-- {
		testMessages := append(newMessages, otherMessages[i])
		testTokens := cm.tokenCounter.Count(testMessages)

		if testTokens > cm.maxTokens {
			break
		}

		newMessages = append([]llm.Message{otherMessages[i]}, newMessages...)
	}

	// Ensure proper order (system messages first, then chronological)
	finalMessages := systemMessages
	for _, msg := range newMessages {
		if msg.Role != "system" {
			finalMessages = append(finalMessages, msg)
		}
	}

	cm.messages = finalMessages
	cm.logger.Debug("Context compressed", "oldCount", len(cm.messages), "newCount", len(finalMessages), "newTokens", cm.tokenCounter.Count(finalMessages))
}

// HeuristicCounter provides simple token counting
type HeuristicCounter struct{}

func (h *HeuristicCounter) Count(messages []llm.Message) int {
	total := 0
	for _, msg := range messages {
		total += h.CountString(msg.Content)
	}
	return total
}

func (h *HeuristicCounter) CountString(s string) int {
	// Rough estimate: 4 characters per token
	return len(s) / 4
}

// formatToolResultSimple formats a tool result for display
func formatToolResultSimple(result interface{}) string {
	switch v := result.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}
