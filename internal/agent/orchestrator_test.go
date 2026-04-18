package agent

import (
	"strings"
	"testing"
)

func TestStripThinkBlocks(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "no think block",
			in:   "Hello world",
			want: "Hello world",
		},
		{
			name: "single think block",
			in:   "<think>reasoning here</think>\n\nActual response",
			want: "Actual response",
		},
		{
			name: "think block in the middle",
			in:   "Before <think>internal reasoning</think> After",
			want: "Before  After",
		},
		{
			name: "multiple think blocks",
			in:   "<think>first</think>middle<think>second</think>end",
			want: "middleend",
		},
		{
			name: "empty think block",
			in:   "<think></think>response",
			want: "response",
		},
		{
			name: "only a think block",
			in:   "<think>just thinking, no response</think>",
			want: "",
		},
		{
			name: "unclosed think tag preserved",
			in:   "<think>unclosed reasoning\nSome response text",
			want: "<think>unclosed reasoning\nSome response text",
		},
		{
			name: "mismatched tags preserved",
			in:   "</think>Some text<think>",
			want: "</think>Some text<think>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripThinkBlocks(tt.in)
			if got != tt.want {
				t.Errorf("stripThinkBlocks(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestExtractAndStripThinkBlocks(t *testing.T) {
	tests := []struct {
		name      string
		in        string
		wantText  string
		wantThink string
	}{
		{
			name:      "no think block",
			in:        "Hello world",
			wantText:  "Hello world",
			wantThink: "",
		},
		{
			name:      "single think block",
			in:        "<think>\nreasoning here\n</think>\n\nActual response",
			wantText:  "Actual response",
			wantThink: "reasoning here",
		},
		{
			name:      "multiple think blocks joined with separator",
			in:        "<think>first thought</think>middle<think>second thought</think>end",
			wantText:  "middleend",
			wantThink: "first thought\n---\nsecond thought",
		},
		{
			name:      "only think block leaves empty response",
			in:        "<think>just thinking</think>",
			wantText:  "",
			wantThink: "just thinking",
		},
		{
			name:      "preserves unclosed tags unchanged",
			in:        "<think>unclosed",
			wantText:  "<think>unclosed",
			wantThink: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotText, gotThink := extractAndStripThinkBlocks(tt.in)
			if gotText != tt.wantText {
				t.Errorf("extractAndStripThinkBlocks(%q) text = %q, want %q", tt.in, gotText, tt.wantText)
			}
			// Normalise separator whitespace for comparison
			gotThink = strings.TrimSpace(gotThink)
			if gotThink != tt.wantThink {
				t.Errorf("extractAndStripThinkBlocks(%q) think = %q, want %q", tt.in, gotThink, tt.wantThink)
			}
		})
	}
}
