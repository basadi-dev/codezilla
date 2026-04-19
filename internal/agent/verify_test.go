package agent

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectProjectType(t *testing.T) {
	tests := []struct {
		name     string
		files    []string
		expected ProjectType
	}{
		{
			name:     "Go project",
			files:    []string{"go.mod"},
			expected: ProjectGo,
		},
		{
			name:     "Node project",
			files:    []string{"package.json"},
			expected: ProjectNode,
		},
		{
			name:     "Rust project",
			files:    []string{"Cargo.toml"},
			expected: ProjectRust,
		},
		{
			name:     "Python project with pyproject.toml",
			files:    []string{"pyproject.toml"},
			expected: ProjectPython,
		},
		{
			name:     "Python project with setup.py",
			files:    []string{"setup.py"},
			expected: ProjectPython,
		},
		{
			name:     "Unknown project",
			files:    []string{"README.md"},
			expected: ProjectUnknown,
		},
		{
			name:     "Go takes precedence over Node",
			files:    []string{"go.mod", "package.json"},
			expected: ProjectGo,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			for _, f := range tt.files {
				if err := os.WriteFile(filepath.Join(tmpDir, f), []byte(""), 0644); err != nil {
					t.Fatal(err)
				}
			}

			result := DetectProjectType(tmpDir)
			if result != tt.expected {
				t.Errorf("DetectProjectType() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestDefaultVerifyCommands(t *testing.T) {
	tests := []struct {
		pt       ProjectType
		wantNil  bool
		wantMin  int
	}{
		{ProjectGo, false, 2},       // go build + go vet
		{ProjectNode, false, 1},     // tsc
		{ProjectRust, false, 1},     // cargo check
		{ProjectPython, false, 1},   // py_compile
		{ProjectUnknown, true, 0},   // no commands
	}

	for _, tt := range tests {
		t.Run(string(tt.pt), func(t *testing.T) {
			cmds := DefaultVerifyCommands(tt.pt)
			if tt.wantNil && cmds != nil {
				t.Errorf("expected nil commands for %s, got %v", tt.pt, cmds)
			}
			if !tt.wantNil && len(cmds) < tt.wantMin {
				t.Errorf("expected at least %d commands for %s, got %d", tt.wantMin, tt.pt, len(cmds))
			}
		})
	}
}

func TestIsFileModifyingTool(t *testing.T) {
	tests := []struct {
		name   string
		tool   string
		params map[string]interface{}
		want   bool
	}{
		{"fileEdit always modifies", "fileEdit", nil, true},
		{"multiReplace always modifies", "multiReplace", nil, true},
		{"fileManage with write", "fileManage", map[string]interface{}{"action": "write"}, true},
		{"fileManage with edit", "fileManage", map[string]interface{}{"action": "edit"}, true},
		{"fileManage with read", "fileManage", map[string]interface{}{"action": "read"}, false},
		{"fileManage with list", "fileManage", map[string]interface{}{"action": "list"}, false},
		{"grepSearch is read-only", "grepSearch", nil, false},
		{"execute is not tracked", "execute", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isFileModifyingTool(tt.tool, tt.params)
			if got != tt.want {
				t.Errorf("isFileModifyingTool(%q, %v) = %v, want %v", tt.tool, tt.params, got, tt.want)
			}
		})
	}
}

func TestReasoningEffortForRetry(t *testing.T) {
	tests := []struct {
		current string
		retry   int
		want    string
	}{
		{"", 1, "medium"},
		{"", 2, "high"},
		{"low", 1, "medium"},
		{"low", 2, "high"},
		{"medium", 1, "medium"},  // already at medium, no downgrade
		{"medium", 2, "high"},
		{"high", 1, "high"},     // already at high
		{"high", 2, "high"},
		{"high", 3, "high"},     // cap at high
	}

	for _, tt := range tests {
		t.Run(tt.current+"_retry"+string(rune('0'+tt.retry)), func(t *testing.T) {
			got := reasoningEffortForRetry(tt.current, tt.retry)
			if got != tt.want {
				t.Errorf("reasoningEffortForRetry(%q, %d) = %q, want %q", tt.current, tt.retry, got, tt.want)
			}
		})
	}
}
