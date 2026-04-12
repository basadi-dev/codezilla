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

// ProcessChannel reads streams and fires events. It returns the raw concatenated text and/or extracted native tools.
// onToolPreparing is called the first time each distinct tool name appears in the stream,
// so the caller can update a spinner/status with what's being invoked.
func (sp *StreamProcessor) ProcessChannel(
	ctx context.Context,
	streamCh <-chan anyllm.ChatCompletionChunk,
	errCh <-chan error,
	onTextToken func(string),
	onToolPreparing func(toolName string),
) (string, []anyllm.ToolCall, error) {

	var fullResponse string
	var streamedToolCalls []anyllm.ToolCall
	var streamErr error
	var totalChunks, contentChunks, reasoningChunks int
	var inReasoning bool
	var sentFirstTool bool
	seenToolNames := map[string]bool{}

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

					// Close reasoning if switching to content or tools
					if inReasoning && (delta.Content != "" || len(delta.ToolCalls) > 0) {
						inReasoning = false
						if onTextToken != nil {
							onTextToken("</think>\n")
						}
						fullResponse += "</think>\n"
					}

					if delta.Content != "" {
						contentChunks++
						fullResponse += delta.Content
						// Don't forward content to UI once tool calls have started;
						// such content is usually LLM preamble noise.
						if onTextToken != nil && !sentFirstTool {
							onTextToken(delta.Content)
						}
					} else if delta.Reasoning != nil && delta.Reasoning.Content != "" {
						if !inReasoning {
							inReasoning = true
							if onTextToken != nil {
								onTextToken("<think>\n")
							}
							fullResponse += "<think>\n"
						}
						reasoningChunks++
						fullResponse += delta.Reasoning.Content
						if onTextToken != nil {
							onTextToken(delta.Reasoning.Content)
						}
					} else if len(delta.ToolCalls) > 0 {
						sentFirstTool = true
						for _, tc := range delta.ToolCalls {
							if tc.Function.Name != "" && !seenToolNames[tc.Function.Name] {
								seenToolNames[tc.Function.Name] = true
								if onToolPreparing != nil {
									onToolPreparing(tc.Function.Name)
								}
							}
						}
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
			if inReasoning {
				if onTextToken != nil {
					onTextToken("</think>\n")
				}
				fullResponse += "</think>\n"
			}
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
