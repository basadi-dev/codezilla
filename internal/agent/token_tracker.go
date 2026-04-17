package agent

import (
	"fmt"
	"sort"
	"strings"
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

// String returns a compact status-line summary like "19.5k in · 197 out".
func (u TokenUsage) String() string {
	s := fmt.Sprintf("%s in · %s out",
		CompactNumber(u.PromptTokens),
		CompactNumber(u.CompletionTokens))
	if u.ReasoningTokens > 0 {
		s += fmt.Sprintf(" · %s reasoning", CompactNumber(u.ReasoningTokens))
	}
	return s
}

// DetailedString returns a full comma-separated summary like "19,455 in → 197 out (19,652 total)".
func (u TokenUsage) DetailedString() string {
	s := fmt.Sprintf("%s in → %s out (%s total)",
		FormatNumber(u.PromptTokens),
		FormatNumber(u.CompletionTokens),
		FormatNumber(u.TotalTokens))
	if u.ReasoningTokens > 0 {
		s += fmt.Sprintf(" [%s reasoning]", FormatNumber(u.ReasoningTokens))
	}
	return s
}

// ModelUsage associates a model name with its token usage breakdown.
type ModelUsage struct {
	Model string
	Usage TokenUsage
}

// FormatModelBreakdown renders a per-model breakdown with in/out arrows.
//
// Single model:  "gemma4:31b 2k↑ 200↓"
// Multi model:   "gemma4:31b 2k↑ 200↓ · gemma3:12b 500↑ 50↓"
func FormatModelBreakdown(models map[string]TokenUsage) string {
	if len(models) == 0 {
		return "—"
	}

	// Sort by total tokens descending so the primary model comes first
	entries := make([]ModelUsage, 0, len(models))
	for model, usage := range models {
		entries = append(entries, ModelUsage{Model: model, Usage: usage})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Usage.TotalTokens > entries[j].Usage.TotalTokens
	})

	parts := make([]string, len(entries))
	for i, e := range entries {
		parts[i] = fmt.Sprintf("%s %s↑ %s↓",
			e.Model,
			CompactNumber(e.Usage.PromptTokens),
			CompactNumber(e.Usage.CompletionTokens))
	}
	return strings.Join(parts, " · ")
}

// TokenTracker accumulates LLM token usage across calls within a session.
// Tracks per-model breakdowns for both the current turn and the overall session.
// Safe for concurrent use.
type TokenTracker struct {
	mu      sync.Mutex
	turns   []TokenUsage // per-LLM-call usage
	session TokenUsage   // running cumulative total

	// Per-model breakdown: model name → usage with in/out split
	turnModels    map[string]TokenUsage // reset each turn group via ResetTurn()
	sessionModels map[string]TokenUsage // cumulative across entire session
}

// NewTokenTracker creates a new empty tracker.
func NewTokenTracker() *TokenTracker {
	return &TokenTracker{
		turnModels:    make(map[string]TokenUsage),
		sessionModels: make(map[string]TokenUsage),
	}
}

// Record adds the usage from a single LLM call and updates the session total.
// model identifies which model produced this usage. Nil or zero-valued usage is silently ignored.
func (t *TokenTracker) Record(model string, usage *anyllm.Usage) {
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

	// Per-model tracking (accumulate in/out separately)
	m := t.turnModels[model]
	m.PromptTokens += turn.PromptTokens
	m.CompletionTokens += turn.CompletionTokens
	m.ReasoningTokens += turn.ReasoningTokens
	m.TotalTokens += turn.TotalTokens
	t.turnModels[model] = m

	sm := t.sessionModels[model]
	sm.PromptTokens += turn.PromptTokens
	sm.CompletionTokens += turn.CompletionTokens
	sm.ReasoningTokens += turn.ReasoningTokens
	sm.TotalTokens += turn.TotalTokens
	t.sessionModels[model] = sm
}

// ResetTurn clears the per-turn model breakdown (called at the start of each user turn).
func (t *TokenTracker) ResetTurn() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.turnModels = make(map[string]TokenUsage)
	t.turns = nil
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

// TurnModelBreakdown returns a copy of the per-model token breakdown for the current turn.
func (t *TokenTracker) TurnModelBreakdown() map[string]TokenUsage {
	t.mu.Lock()
	defer t.mu.Unlock()

	cp := make(map[string]TokenUsage, len(t.turnModels))
	for k, v := range t.turnModels {
		cp[k] = v
	}
	return cp
}

// SessionModelBreakdown returns a copy of the per-model token breakdown for the entire session.
func (t *TokenTracker) SessionModelBreakdown() map[string]TokenUsage {
	t.mu.Lock()
	defer t.mu.Unlock()

	cp := make(map[string]TokenUsage, len(t.sessionModels))
	for k, v := range t.sessionModels {
		cp[k] = v
	}
	return cp
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
	t.turnModels = make(map[string]TokenUsage)
	t.sessionModels = make(map[string]TokenUsage)
}

// CompactNumber formats a token count into a short human-friendly label.
//
//	0       → "0"
//	850     → "850"
//	1000    → "1k"
//	1500    → "1.5k"
//	19455   → "19.5k"
//	100000  → "100k"
//	1234567 → "1.2m"
func CompactNumber(n int) string {
	if n < 0 {
		return "-" + CompactNumber(-n)
	}
	switch {
	case n < 1000:
		return fmt.Sprintf("%d", n)
	case n < 10_000:
		v := float64(n) / 1000
		if n%1000 == 0 {
			return fmt.Sprintf("%dk", n/1000)
		}
		return fmt.Sprintf("%.1fk", v)
	case n < 1_000_000:
		v := float64(n) / 1000
		if n%1000 < 100 {
			return fmt.Sprintf("%dk", n/1000)
		}
		return fmt.Sprintf("%.1fk", v)
	default:
		v := float64(n) / 1_000_000
		if n%1_000_000 < 100_000 {
			return fmt.Sprintf("%dm", n/1_000_000)
		}
		return fmt.Sprintf("%.1fm", v)
	}
}

// FormatNumber formats an integer with comma separators (e.g., 1234 → "1,234").
// Used for detailed displays like /tokens.
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
