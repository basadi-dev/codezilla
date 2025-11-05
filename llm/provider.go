package llm

import (
	"context"
	"time"
)

// Provider is the universal interface for ANY LLM provider
type Provider interface {
	// Generate generates a response for the given request
	Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error)

	// StreamGenerate generates a streaming response
	StreamGenerate(ctx context.Context, req GenerateRequest) (<-chan StreamChunk, error)

	// ListModels returns available models for this provider
	ListModels(ctx context.Context) ([]ModelInfo, error)

	// Name returns the provider name (e.g., "ollama", "openai", "anthropic")
	Name() string

	// CountTokens counts tokens for the given text (optional, may return estimate)
	CountTokens(text string) int
}

// GenerateRequest is provider-agnostic request structure
type GenerateRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	MaxTokens   int       `json:"max_tokens"`
	Stream      bool      `json:"stream"`
	System      string    `json:"system,omitempty"`
}

// GenerateResponse is provider-agnostic response structure
type GenerateResponse struct {
	Content    string      `json:"content"`
	Model      string      `json:"model"`
	TokensUsed TokenUsage  `json:"tokens_used"`
	Done       bool        `json:"done"`
	Duration   time.Duration `json:"duration,omitempty"`
}

// StreamChunk represents a chunk in a streaming response
type StreamChunk struct {
	Content string
	Done    bool
	Error   error
}

// Message represents a conversation message
type Message struct {
	Role    string `json:"role"`    // "system", "user", "assistant", "tool"
	Content string `json:"content"`
}

// TokenUsage tracks token consumption
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ModelInfo contains information about a model
type ModelInfo struct {
	Name        string    `json:"name"`
	Size        int64     `json:"size,omitempty"`
	ModifiedAt  time.Time `json:"modified_at,omitempty"`
	Description string    `json:"description,omitempty"`
}

// ProviderConfig contains configuration for any provider
type ProviderConfig struct {
	Name     string
	BaseURL  string
	APIKey   string
	Username string
	Password string
	Headers  map[string]string
	Timeout  time.Duration
}
