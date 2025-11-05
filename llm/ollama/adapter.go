package ollama

import (
	"context"
	"time"

	"codezilla/llm"
)

// OllamaAdapter adapts the Ollama client to the Provider interface
type OllamaAdapter struct {
	client Client
}

// NewOllamaAdapter creates a new Ollama adapter
func NewOllamaAdapter(options ...func(*ClientOptions)) llm.Provider {
	return &OllamaAdapter{
		client: NewClient(options...),
	}
}

// Generate implements llm.Provider
func (a *OllamaAdapter) Generate(ctx context.Context, req llm.GenerateRequest) (*llm.GenerateResponse, error) {
	// Convert to Ollama chat format
	messages := make([]Message, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = Message{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	chatReq := ChatRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   false,
		Options: map[string]interface{}{
			"temperature": req.Temperature,
			"num_predict": req.MaxTokens,
		},
	}

	start := time.Now()
	resp, err := a.client.Chat(ctx, chatReq)
	if err != nil {
		return nil, err
	}
	duration := time.Since(start)

	// Convert response
	return &llm.GenerateResponse{
		Content: resp.Message.Content,
		Model:   resp.Model,
		TokensUsed: llm.TokenUsage{
			PromptTokens:     resp.PromptEvalCount,
			CompletionTokens: resp.EvalCount,
			TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
		},
		Done:     resp.Done,
		Duration: duration,
	}, nil
}

// StreamGenerate implements llm.Provider
func (a *OllamaAdapter) StreamGenerate(ctx context.Context, req llm.GenerateRequest) (<-chan llm.StreamChunk, error) {
	messages := make([]Message, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = Message{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	chatReq := ChatRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   true,
		Options: map[string]interface{}{
			"temperature": req.Temperature,
			"num_predict": req.MaxTokens,
		},
	}

	streamChan, err := a.client.StreamChat(ctx, chatReq)
	if err != nil {
		return nil, err
	}

	// Convert Ollama stream to provider stream
	outChan := make(chan llm.StreamChunk)
	go func() {
		defer close(outChan)
		for chunk := range streamChan {
			outChan <- llm.StreamChunk{
				Content: chunk.Message.Content,
				Done:    chunk.Done,
				Error:   chunk.Error,
			}
		}
	}()

	return outChan, nil
}

// ListModels implements llm.Provider
func (a *OllamaAdapter) ListModels(ctx context.Context) ([]llm.ModelInfo, error) {
	resp, err := a.client.ListModels(ctx)
	if err != nil {
		return nil, err
	}

	models := make([]llm.ModelInfo, len(resp.Models))
	for i, model := range resp.Models {
		// Parse the modified_at time string
		modifiedAt, _ := time.Parse(time.RFC3339, model.ModifiedAt)

		models[i] = llm.ModelInfo{
			Name:       model.Name,
			Size:       model.Size,
			ModifiedAt: modifiedAt,
		}
	}

	return models, nil
}

// Name implements llm.Provider
func (a *OllamaAdapter) Name() string {
	return "ollama"
}

// CountTokens implements llm.Provider
func (a *OllamaAdapter) CountTokens(text string) int {
	// Use simple heuristic (4 chars per token)
	// Ollama doesn't provide a token counting API
	return len(text) / 4
}
