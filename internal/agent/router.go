package agent

import (
	"regexp"
	"strings"
)

// RequestTier represents the model tier for routing.
type RequestTier int

const (
	// TierFast is for simple Q&A, greetings, short explanations.
	TierFast RequestTier = iota
	// TierDefault is for standard coding tasks.
	TierDefault
	// TierHeavy is for complex multi-file refactors and deep reasoning.
	TierHeavy
)

const (
	// RoutingThreshold is the score required to shift a request out of the Default tier
	RoutingThreshold = 3
	
	// Thresholds for text analysis
	MaxLengthFast  = 50
	MinLengthHeavy = 300
	MaxExplainLen  = 200
)

// String returns a human-readable label for the tier.
func (t RequestTier) String() string {
	switch t {
	case TierFast:
		return "fast"
	case TierDefault:
		return "default"
	case TierHeavy:
		return "heavy"
	default:
		return "unknown"
	}
}

// ModelRouter classifies user requests and selects the appropriate model tier.
// It uses purely heuristic rules (no LLM call) to keep routing instant and free.
type ModelRouter struct {
	Enabled      bool
	FastModel    string // lightweight model for simple Q&A
	DefaultModel string // standard coding tasks
	HeavyModel   string // complex multi-file refactors (future)
}

// NewModelRouter creates a router with the given model assignments.
// If fastModel is empty, the router still classifies but TierFast falls back to defaultModel.
func NewModelRouter(enabled bool, fastModel, defaultModel, heavyModel string) *ModelRouter {
	return &ModelRouter{
		Enabled:      enabled,
		FastModel:    fastModel,
		DefaultModel: defaultModel,
		HeavyModel:   heavyModel,
	}
}

// ModelForTier returns the model name for a given tier, falling back to DefaultModel
// when a tier-specific model is not configured.
func (r *ModelRouter) ModelForTier(tier RequestTier) string {
	switch tier {
	case TierFast:
		if r.FastModel != "" {
			return r.FastModel
		}
	case TierHeavy:
		if r.HeavyModel != "" {
			return r.HeavyModel
		}
	}
	return r.DefaultModel
}

// Classify examines the user message and conversation state to determine which
// model tier should handle the request. It returns the selected tier and a short
// human-readable reason string for logging/UI display.
//
// prevToolCount is the number of tools called in the previous agent turn (0 for
// the first turn of a conversation).
func (r *ModelRouter) Classify(message string, prevToolCount int) (RequestTier, string) {
	if !r.Enabled {
		return TierDefault, "routing disabled"
	}

	lower := strings.ToLower(strings.TrimSpace(message))
	length := len(lower)

	// --- Score accumulators ---
	var fastScore, heavyScore int
	var fastReason, heavyReason string

	// ---- Fast-tier signals ----

	// Greetings / acknowledgements
	if isGreeting(lower) {
		fastScore += 5
		fastReason = "greeting/acknowledgement"
	}

	// Very short, no code keywords
	if length < MaxLengthFast && !containsCodeKeyword(lower) {
		fastScore += RoutingThreshold
		if fastReason == "" {
			fastReason = "short simple query"
		}
	}

	// Explanation-seeking questions
	if containsExplainKeyword(lower) && length < MaxExplainLen && !containsCodeKeyword(lower) {
		fastScore += RoutingThreshold
		if fastReason == "" {
			fastReason = "simple explanation"
		}
	}

	// ---- Heavy-tier signals ----

	// Complex action keywords (accumulate score for each match)
	for _, kw := range complexKeywords {
		if strings.Contains(lower, kw) {
			heavyScore += RoutingThreshold
			if heavyReason == "" {
				heavyReason = kw + " request"
			}
		}
	}

	// Long input (likely detailed spec)
	if length > MinLengthHeavy {
		heavyScore++
		if heavyReason == "" {
			heavyReason = "detailed request"
		}
	}

	// Numbered lists in the message (multi-step plan)
	if numberedListPattern.MatchString(message) {
		heavyScore += 2
		if heavyReason == "" {
			heavyReason = "multi-step task"
		}
	}

	// Multiple file paths
	if filePathPattern.FindAllString(message, -1) != nil && len(filePathPattern.FindAllString(message, -1)) >= 2 {
		heavyScore++
		if heavyReason == "" {
			heavyReason = "multi-file operation"
		}
	}

	// Contains code fences (paste of code to work with)
	if strings.Contains(message, "```") {
		heavyScore++
		if heavyReason == "" {
			heavyReason = "code-heavy request"
		}
	}

	// Previous turn was tool-heavy
	if prevToolCount >= 3 {
		heavyScore++
	}

	// ---- Decision ----

	// Fast path: only if we have a fast model configured and fast signals are strong
	if fastScore >= RoutingThreshold && r.FastModel != "" {
		return TierFast, fastReason
	}

	// Heavy path
	if heavyScore >= RoutingThreshold {
		return TierHeavy, heavyReason
	}

	return TierDefault, "standard request"
}

// ----- Keyword sets & patterns -----

var greetingPhrases = []string{
	"hello", "hi", "hey", "good morning", "good afternoon", "good evening",
	"thanks", "thank you", "thx", "ty", "ok", "okay", "sure", "cool",
	"got it", "understood", "yes", "no", "yep", "nope", "bye", "goodbye",
	"gm", "gn", "sup", "cheers", "👋", "🙏", "👍",
}

func isGreeting(lower string) bool {
	// Exact match or message is just a greeting with minimal extras
	trimmed := strings.TrimRight(lower, " .!?")
	for _, g := range greetingPhrases {
		if trimmed == g {
			return true
		}
	}
	return false
}

var codeKeywords = []string{
	"implement", "refactor", "debug", "fix", "test", "deploy",
	"function", "class", "module", "api", "endpoint", "database",
	"error", "bug", "crash", "build", "compile", "lint",
	"file", "read", "write", "edit", "delete", "create",
	"search", "grep", "run", "execute", "command",
}

func containsCodeKeyword(lower string) bool {
	for _, kw := range codeKeywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

var explainKeywords = []string{
	"explain", "what is", "what are", "how does", "how do",
	"why is", "why does", "tell me about", "describe",
	"what's the difference", "summarize", "summary",
}

func containsExplainKeyword(lower string) bool {
	for _, kw := range explainKeywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

var complexKeywords = []string{
	"implement", "refactor", "migrate", "integrate",
	"redesign", "architect", "rewrite", "overhaul",
	"set up", "configure and deploy", "step by step",
	"plan", "architecture", "design",
}



var (
	// Matches numbered lists in any position: "1. foo", "  2) bar", inline "1. design 2. write"
	numberedListPattern = regexp.MustCompile(`\d+[\.)]\s`)
	filePathPattern     = regexp.MustCompile(`(?:\./|/[a-zA-Z])[^\s]+\.[a-zA-Z0-9]+`)
)
