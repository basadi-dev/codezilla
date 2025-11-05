package core

import (
	"context"

	"codezilla/internal/tools"
	"codezilla/llm"
)

// LLMProviderAdapter adapts llm.Provider to tools.LLMClient
type LLMProviderAdapter struct {
	provider llm.Provider
}

// NewLLMProviderAdapter creates a new adapter
func NewLLMProviderAdapter(provider llm.Provider) *LLMProviderAdapter {
	return &LLMProviderAdapter{provider: provider}
}

// GenerateResponse adapts the GenerateResponse call
func (a *LLMProviderAdapter) GenerateResponse(ctx context.Context, messages []tools.LLMMessage) (string, error) {
	// Convert tools.LLMMessage to llm.Message
	llmMessages := make([]llm.Message, len(messages))
	for i, msg := range messages {
		llmMessages[i] = llm.Message{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// Use the provider to generate a response
	resp, err := a.provider.Generate(ctx, llm.GenerateRequest{
		Model:       "qwen3:14b", // Default model for analysis
		Messages:    llmMessages,
		Temperature: 0.7,
		MaxTokens:   4000,
		Stream:      false,
	})

	if err != nil {
		return "", err
	}

	return resp.Content, nil
}
