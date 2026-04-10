package agent

import (
	"context"

	"codezilla/pkg/logger"
	anyllm "github.com/mozilla-ai/any-llm-go"
)

// StreamProcessor abstracts the messy select channel logic
type StreamProcessor struct {
	logger *logger.Logger
}

func NewStreamProcessor(logger *logger.Logger) *StreamProcessor {
	return &StreamProcessor{logger: logger}
}

// ProcessChannel reads streams and fires events. It returns the raw concatenated text and/or extracted native tools
func (sp *StreamProcessor) ProcessChannel(
	ctx context.Context,
	streamCh <-chan anyllm.ChatCompletionChunk,
	errCh <-chan error,
	onTextToken func(string),
) (string, []anyllm.ToolCall, error) {

	var fullResponse string
	var streamedToolCalls []anyllm.ToolCall
	var streamErr error
	var totalChunks, contentChunks, reasoningChunks int

	// Loop until channels are closed
	for {
		select {
		case chunk, ok := <-streamCh:
			if !ok {
				streamCh = nil
			} else {
				totalChunks++
				if len(chunk.Choices) > 0 {
					delta := chunk.Choices[0].Delta
					if delta.Content != "" {
						contentChunks++
						fullResponse += delta.Content
						if onTextToken != nil {
							onTextToken(delta.Content)
						}
					} else if delta.Reasoning != nil && delta.Reasoning.Content != "" {
						reasoningChunks++
						fullResponse += delta.Reasoning.Content
						if onTextToken != nil {
							onTextToken(delta.Reasoning.Content)
						}
					} else if len(delta.ToolCalls) > 0 {
						streamedToolCalls = append(streamedToolCalls, delta.ToolCalls...)
					}
				}
			}
		case errVal, ok := <-errCh:
			if ok && errVal != nil {
				sp.logger.Error("Stream processing error", "error", errVal)
				streamErr = errVal
			}
			errCh = nil
		}

		if streamCh == nil && errCh == nil {
			break
		}
	}

	sp.logger.Debug("Stream finished",
		"total", totalChunks,
		"content", contentChunks,
		"reasoning", reasoningChunks,
		"nativeTools", len(streamedToolCalls))

	return fullResponse, streamedToolCalls, streamErr
}
