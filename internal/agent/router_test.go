package agent

import (
	"testing"
)

func TestModelRouter_Classify(t *testing.T) {
	router := NewModelRouter(true, "gemma3:12b", "gpt-oss:120b", "")

	tests := []struct {
		name          string
		message       string
		prevToolCount int
		wantTier      RequestTier
		wantReason    string // partial match on reason
	}{
		// ---- TierFast cases ----
		{
			name:     "greeting hello",
			message:  "hello",
			wantTier: TierFast,
		},
		{
			name:     "greeting thanks",
			message:  "thanks!",
			wantTier: TierFast,
		},
		{
			name:     "greeting ok",
			message:  "ok",
			wantTier: TierFast,
		},
		{
			name:     "greeting with emoji",
			message:  "👋",
			wantTier: TierFast,
		},
		{
			name:     "simple explanation",
			message:  "what is a goroutine?",
			wantTier: TierFast,
		},
		{
			name:     "explain keyword",
			message:  "explain how channels work",
			wantTier: TierFast,
		},
		{
			name:     "short yes",
			message:  "yes",
			wantTier: TierFast,
		},

		// ---- TierDefault cases ----
		{
			name:     "read a file",
			message:  "read the file main.go",
			wantTier: TierDefault,
		},
		{
			name:     "fix a bug",
			message:  "fix the bug in handler.go where the response is nil",
			wantTier: TierDefault,
		},
		{
			name:     "general coding question",
			message:  "how do I write a middleware that logs request durations?",
			wantTier: TierDefault,
		},
		{
			name:     "medium length question",
			message:  "can you look at the session module and tell me if there's a race condition?",
			wantTier: TierDefault,
		},

		// ---- TierHeavy cases ----
		{
			name:     "implement with steps",
			message:  "implement a REST API with authentication. 1. design the schema 2. write the migration 3. add the API endpoints",
			wantTier: TierHeavy,
		},
		{
			name:     "refactor with numbered list",
			message:  "refactor the session module:\n1. extract the recorder into its own package\n2. add compression\n3. update all callers",
			wantTier: TierHeavy,
		},
		{
			name:     "migrate request",
			message:  "migrate the database from SQLite to PostgreSQL with proper connection pooling and implement the new repository layer step by step",
			wantTier: TierHeavy,
		},
		{
			name:     "long detailed spec",
			message:  "I need you to implement a complete WebSocket server that supports multiple rooms, authentication via JWT tokens, message persistence to a database, and real-time presence tracking. The server should handle reconnection gracefully and include rate limiting per client. Also add comprehensive test coverage for the connection lifecycle. Here's the detailed specification of what each component should do and how they interact with each other in the system architecture...",
			wantTier: TierHeavy,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tier, reason := router.Classify(tt.message, tt.prevToolCount)
			if tier != tt.wantTier {
				t.Errorf("Classify(%q) = tier %s, want %s (reason: %q)", tt.message, tier, tt.wantTier, reason)
			}
		})
	}
}

func TestModelRouter_Classify_Disabled(t *testing.T) {
	router := NewModelRouter(false, "gemma3:12b", "gpt-oss:120b", "")

	tier, reason := router.Classify("hello", 0)
	if tier != TierDefault {
		t.Errorf("disabled router should always return TierDefault, got %s", tier)
	}
	if reason != "routing disabled" {
		t.Errorf("disabled router reason should be 'routing disabled', got %q", reason)
	}
}

func TestModelRouter_ModelForTier(t *testing.T) {
	tests := []struct {
		name      string
		fast      string
		deflt     string
		heavy     string
		tier      RequestTier
		wantModel string
	}{
		{
			name:      "fast tier with fast model",
			fast:      "gemma3:12b",
			deflt:     "gpt-oss:120b",
			tier:      TierFast,
			wantModel: "gemma3:12b",
		},
		{
			name:      "fast tier without fast model falls back",
			fast:      "",
			deflt:     "gpt-oss:120b",
			tier:      TierFast,
			wantModel: "gpt-oss:120b",
		},
		{
			name:      "default tier",
			fast:      "gemma3:12b",
			deflt:     "gpt-oss:120b",
			tier:      TierDefault,
			wantModel: "gpt-oss:120b",
		},
		{
			name:      "heavy tier with heavy model",
			fast:      "gemma3:12b",
			deflt:     "gpt-oss:120b",
			heavy:     "qwen3-coder:480b",
			tier:      TierHeavy,
			wantModel: "qwen3-coder:480b",
		},
		{
			name:      "heavy tier without heavy model falls back",
			fast:      "gemma3:12b",
			deflt:     "gpt-oss:120b",
			heavy:     "",
			tier:      TierHeavy,
			wantModel: "gpt-oss:120b",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := NewModelRouter(true, tt.fast, tt.deflt, tt.heavy)
			got := router.ModelForTier(tt.tier)
			if got != tt.wantModel {
				t.Errorf("ModelForTier(%s) = %q, want %q", tt.tier, got, tt.wantModel)
			}
		})
	}
}

func TestRequestTier_String(t *testing.T) {
	if TierFast.String() != "fast" {
		t.Errorf("TierFast.String() = %q, want 'fast'", TierFast.String())
	}
	if TierDefault.String() != "default" {
		t.Errorf("TierDefault.String() = %q, want 'default'", TierDefault.String())
	}
	if TierHeavy.String() != "heavy" {
		t.Errorf("TierHeavy.String() = %q, want 'heavy'", TierHeavy.String())
	}
}
