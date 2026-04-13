package agent

import (
	"fmt"
	"sync"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// TokenUsage holds token counts for a single LLM call or an accumulated session.
type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	ReasoningTokens  int
	TotalTokens      int
}

// String returns a human-readable summary like "1,240 in → 386 out (1,626 total)".
func (u TokenUsage) String() string {
	s := fmt.Sprintf("%s in → %s out (%s total)",
		FormatNumber(u.PromptTokens),
		FormatNumber(u.CompletionTokens),
		FormatNumber(u.TotalTokens))
	if u.ReasoningTokens > 0 {
		s += fmt.Sprintf(" [%s reasoning]", FormatNumber(u.ReasoningTokens))
	}
	return s
}

// TokenTracker accumulates LLM token usage across calls within a session.
// Safe for concurrent use.
type TokenTracker struct {
	mu      sync.Mutex
	turns   []TokenUsage // per-LLM-call usage
	session TokenUsage   // running cumulative total
}

// NewTokenTracker creates a new empty tracker.
func NewTokenTracker() *TokenTracker {
	return &TokenTracker{}
}

// Record adds the usage from a single LLM call and updates the session total.
// Nil or zero-valued usage is silently ignored.
func (t *TokenTracker) Record(usage *anyllm.Usage) {
	if usage == nil || usage.TotalTokens == 0 {
		return
	}

	turn := TokenUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		ReasoningTokens:  usage.ReasoningTokens,
		TotalTokens:      usage.TotalTokens,
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	t.turns = append(t.turns, turn)
	t.session.PromptTokens += turn.PromptTokens
	t.session.CompletionTokens += turn.CompletionTokens
	t.session.ReasoningTokens += turn.ReasoningTokens
	t.session.TotalTokens += turn.TotalTokens
}

// LastTurn returns the usage from the most recent LLM call.
// Returns a zero-valued TokenUsage if no calls have been recorded.
func (t *TokenTracker) LastTurn() TokenUsage {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.turns) == 0 {
		return TokenUsage{}
	}
	return t.turns[len(t.turns)-1]
}

// SessionTotal returns the cumulative usage across all LLM calls.
func (t *TokenTracker) SessionTotal() TokenUsage {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.session
}

// TurnCount returns the number of LLM calls recorded.
func (t *TokenTracker) TurnCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()

	return len(t.turns)
}

// Reset clears all recorded usage data.
func (t *TokenTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.turns = nil
	t.session = TokenUsage{}
}

// FormatNumber formats an integer with comma separators (e.g., 1234 → "1,234").
func FormatNumber(n int) string {
	if n < 0 {
		return "-" + FormatNumber(-n)
	}
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}

	// Insert commas from right to left
	var result []byte
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			result = append(result, ',')
		}
		result = append(result, byte(c))
	}
	return string(result)
}
