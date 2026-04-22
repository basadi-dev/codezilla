package agent

import (
	"context"
	"fmt"
	"testing"

	"codezilla/pkg/logger"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// ──────────────────────────────────────────────────────────────────────────────
// Stream Processor Tests — C1/C2 scenarios
// ──────────────────────────────────────────────────────────────────────────────

func newTestStreamProcessor() *StreamProcessor {
	log, _ := logger.New(logger.Config{Silent: true})
	return NewStreamProcessor(log)
}

func sendChunks(chunks []anyllm.ChatCompletionChunk) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	chunkCh := make(chan anyllm.ChatCompletionChunk, len(chunks))
	errCh := make(chan error, 1)
	for _, c := range chunks {
		chunkCh <- c
	}
	close(chunkCh)
	close(errCh)
	return chunkCh, errCh
}

func sendChunksWithError(chunks []anyllm.ChatCompletionChunk, err error) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	chunkCh := make(chan anyllm.ChatCompletionChunk, len(chunks))
	errCh := make(chan error, 1)
	for _, c := range chunks {
		chunkCh <- c
	}
	close(chunkCh)
	errCh <- err
	close(errCh)
	return chunkCh, errCh
}

func contentChunk(s string) anyllm.ChatCompletionChunk {
	return anyllm.ChatCompletionChunk{
		Choices: []anyllm.ChunkChoice{
			{Delta: anyllm.ChunkDelta{Content: s}},
		},
	}
}

func reasoningChunk(s string) anyllm.ChatCompletionChunk {
	return anyllm.ChatCompletionChunk{
		Choices: []anyllm.ChunkChoice{
			{Delta: anyllm.ChunkDelta{Reasoning: &anyllm.Reasoning{Content: s}}},
		},
	}
}

func toolCallChunk(name, args string) anyllm.ChatCompletionChunk {
	return anyllm.ChatCompletionChunk{
		Choices: []anyllm.ChunkChoice{
			{
				Delta: anyllm.ChunkDelta{
					ToolCalls: []anyllm.ToolCall{
						MakeToolCall("call_1", name, map[string]interface{}{"path": "/test"}),
					},
				},
			},
		},
	}
}

func usageChunk(prompt, completion, total int) anyllm.ChatCompletionChunk {
	return anyllm.ChatCompletionChunk{
		Usage: MakeUsage(prompt, completion, total),
	}
}

func TestStreamProcessor_PlainContent(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	chunks := []anyllm.ChatCompletionChunk{
		contentChunk("Hello "),
		contentChunk("World!"),
	}
	chunkCh, errCh := sendChunks(chunks)

	var tokens []string
	fullResp, tools, usage, err := sp.ProcessChannel(ctx, chunkCh, errCh, func(s string) {
		tokens = append(tokens, s)
	}, nil)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fullResp != "Hello World!" {
		t.Errorf("fullResp = %q, want %q", fullResp, "Hello World!")
	}
	if len(tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(tools))
	}
	if usage != nil {
		t.Errorf("expected nil usage, got %v", usage)
	}
	if len(tokens) != 2 {
		t.Errorf("expected 2 token callbacks, got %d", len(tokens))
	}
}

func TestStreamProcessor_ReasoningToContent(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	// Simulate: reasoning → content transition
	chunks := []anyllm.ChatCompletionChunk{
		reasoningChunk("Let me think..."),
		reasoningChunk(" about this."),
		contentChunk("Here's the answer."),
	}
	chunkCh, errCh := sendChunks(chunks)

	var tokens []string
	fullResp, _, _, err := sp.ProcessChannel(ctx, chunkCh, errCh, func(s string) {
		tokens = append(tokens, s)
	}, nil)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should contain <think>...</think> wrapping around reasoning
	if fullResp == "" {
		t.Fatal("fullResp should not be empty")
	}

	// Verify the response includes think block markers
	if !containsString(fullResp, "<think>") {
		t.Error("expected <think> tag in response")
	}
	if !containsString(fullResp, "</think>") {
		t.Error("expected </think> tag in response")
	}
	if !containsString(fullResp, "Here's the answer.") {
		t.Error("expected content text in response")
	}

	// Tokens should include think markers sent to UI
	if len(tokens) == 0 {
		t.Error("expected token callbacks")
	}
}

func TestStreamProcessor_ToolCallDetection(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	chunks := []anyllm.ChatCompletionChunk{
		contentChunk("Let me read that file."),
		toolCallChunk("fileRead", `{"path":"/test"}`),
	}
	chunkCh, errCh := sendChunks(chunks)

	var preparedTools []string
	fullResp, tools, _, err := sp.ProcessChannel(ctx, chunkCh, errCh, nil, func(toolName string) {
		preparedTools = append(preparedTools, toolName)
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !containsString(fullResp, "Let me read that file.") {
		t.Error("expected content text in response")
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tools))
	}
	if tools[0].Function.Name != "fileRead" {
		t.Errorf("tool name = %q, want %q", tools[0].Function.Name, "fileRead")
	}
	if len(preparedTools) != 1 || preparedTools[0] != "fileRead" {
		t.Errorf("OnToolPreparing not called correctly: got %v", preparedTools)
	}
}

func TestStreamProcessor_ContentSuppressedAfterTool(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	// Content before tool → should be forwarded
	// Content after tool → should NOT be forwarded (it's LLM preamble noise)
	chunks := []anyllm.ChatCompletionChunk{
		contentChunk("Before tool. "),
		toolCallChunk("fileRead", `{}`),
		contentChunk("After tool noise."),
	}
	chunkCh, errCh := sendChunks(chunks)

	var tokens []string
	_, _, _, err := sp.ProcessChannel(ctx, chunkCh, errCh, func(s string) {
		tokens = append(tokens, s)
	}, nil)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Only "Before tool. " should have been sent to UI
	combined := ""
	for _, tok := range tokens {
		combined += tok
	}
	if containsString(combined, "After tool noise") {
		t.Error("content after tool call should be suppressed from UI")
	}
	if !containsString(combined, "Before tool") {
		t.Error("content before tool call should be forwarded to UI")
	}
}

func TestStreamProcessor_UsageCapture(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	chunks := []anyllm.ChatCompletionChunk{
		contentChunk("Response."),
		usageChunk(100, 50, 150),
	}
	chunkCh, errCh := sendChunks(chunks)

	_, _, usage, err := sp.ProcessChannel(ctx, chunkCh, errCh, nil, nil)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if usage == nil {
		t.Fatal("expected non-nil usage")
	}
	if usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", usage.PromptTokens)
	}
	if usage.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", usage.CompletionTokens)
	}
	if usage.TotalTokens != 150 {
		t.Errorf("TotalTokens = %d, want 150", usage.TotalTokens)
	}
}

func TestStreamProcessor_ErrorChannel(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	testErr := fmt.Errorf("connection reset")
	chunks := []anyllm.ChatCompletionChunk{
		contentChunk("Partial"),
	}
	chunkCh, errCh := sendChunksWithError(chunks, testErr)

	_, _, _, err := sp.ProcessChannel(ctx, chunkCh, errCh, nil, nil)

	if err == nil {
		t.Fatal("expected error from error channel")
	}
	if err.Error() != "connection reset" {
		t.Errorf("error = %q, want %q", err.Error(), "connection reset")
	}
}

func TestStreamProcessor_EmptyStream(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	chunkCh, errCh := sendChunks(nil)

	fullResp, tools, _, err := sp.ProcessChannel(ctx, chunkCh, errCh, nil, nil)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fullResp != "" {
		t.Errorf("expected empty response, got %q", fullResp)
	}
	if len(tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(tools))
	}
}

func TestStreamProcessor_ReasoningOnly(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	// Only reasoning, no content — should auto-close think block
	chunks := []anyllm.ChatCompletionChunk{
		reasoningChunk("Deep reasoning about the problem."),
	}
	chunkCh, errCh := sendChunks(chunks)

	fullResp, _, _, err := sp.ProcessChannel(ctx, chunkCh, errCh, nil, nil)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !containsString(fullResp, "<think>") {
		t.Error("expected <think> tag")
	}
	if !containsString(fullResp, "</think>") {
		t.Error("expected </think> auto-close tag")
	}
}

func TestStreamProcessor_MultipleToolCalls(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	chunks := []anyllm.ChatCompletionChunk{
		toolCallChunk("fileRead", `{"path":"/a"}`),
		{
			Choices: []anyllm.ChunkChoice{
				{
					Delta: anyllm.ChunkDelta{
						ToolCalls: []anyllm.ToolCall{
							MakeToolCall("call_2", "grepSearch", map[string]interface{}{"query": "test"}),
						},
					},
				},
			},
		},
	}
	chunkCh, errCh := sendChunks(chunks)

	var preparedTools []string
	_, tools, _, err := sp.ProcessChannel(ctx, chunkCh, errCh, nil, func(name string) {
		preparedTools = append(preparedTools, name)
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(tools))
	}
	// OnToolPreparing should be called once per distinct tool name
	if len(preparedTools) != 2 {
		t.Errorf("expected 2 tool preparings, got %d: %v", len(preparedTools), preparedTools)
	}
}

func TestStreamProcessor_SpecialTokenStripping(t *testing.T) {
	sp := newTestStreamProcessor()
	ctx := context.Background()

	// Content with leaked special tokens that should be stripped
	chunks := []anyllm.ChatCompletionChunk{
		contentChunk("<|im_start|>assistant\nHello there!"),
	}
	chunkCh, errCh := sendChunks(chunks)

	var tokens []string
	_, _, _, err := sp.ProcessChannel(ctx, chunkCh, errCh, func(s string) {
		tokens = append(tokens, s)
	}, nil)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	combined := ""
	for _, tok := range tokens {
		combined += tok
	}
	if containsString(combined, "<|im_start|>") {
		t.Error("special tokens should be stripped from UI output")
	}
	if !containsString(combined, "Hello there!") {
		t.Error("actual content should be preserved")
	}
}

// helper
func containsString(haystack, needle string) bool {
	return len(haystack) >= len(needle) && (haystack == needle || len(needle) == 0 ||
		func() bool {
			for i := 0; i <= len(haystack)-len(needle); i++ {
				if haystack[i:i+len(needle)] == needle {
					return true
				}
			}
			return false
		}())
}
