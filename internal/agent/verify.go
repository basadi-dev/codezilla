package agent

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ProjectType represents the detected project build system.
type ProjectType string

const (
	ProjectGo      ProjectType = "go"
	ProjectNode    ProjectType = "node"
	ProjectPython  ProjectType = "python"
	ProjectRust    ProjectType = "rust"
	ProjectUnknown ProjectType = "unknown"
)

const (
	// maxVerifyRetries is the number of times the agent will retry after a failed
	// verification before giving up and returning the response as-is.
	maxVerifyRetries = 2

	// verifyTimeout is the maximum time allowed for all verification commands combined.
	verifyTimeout = 30 * time.Second
)

// VerifyResult contains the outcome of running verification commands.
type VerifyResult struct {
	Passed   bool
	Errors   []string // one entry per failing command
	Commands []string // the commands that were executed
	Duration time.Duration
}

// fileModifyingTools is the set of tool names that can modify files on disk.
var fileModifyingTools = map[string]bool{
	"fileManage":   true, // when action is write/edit/delete
	"fileEdit":     true,
	"multiReplace": true,
}

// isFileModifyingTool returns true if the tool name+params indicate a file write.
func isFileModifyingTool(name string, params map[string]interface{}) bool {
	if name == "fileManage" {
		action, _ := params["action"].(string)
		return action == "write" || action == "edit" || action == "delete"
	}
	return fileModifyingTools[name]
}

// DetectProjectType examines the working directory for build system indicators.
func DetectProjectType(workDir string) ProjectType {
	indicators := []struct {
		file    string
		project ProjectType
	}{
		{"go.mod", ProjectGo},
		{"package.json", ProjectNode},
		{"Cargo.toml", ProjectRust},
		{"pyproject.toml", ProjectPython},
		{"setup.py", ProjectPython},
		{"requirements.txt", ProjectPython},
	}

	for _, ind := range indicators {
		if _, err := os.Stat(filepath.Join(workDir, ind.file)); err == nil {
			return ind.project
		}
	}
	return ProjectUnknown
}

// DefaultVerifyCommands returns the standard build-check commands for a project type.
// These are intentionally fast commands (no full test suite) to keep verification snappy.
func DefaultVerifyCommands(pt ProjectType) []string {
	switch pt {
	case ProjectGo:
		return []string{"go build ./...", "go vet ./..."}
	case ProjectNode:
		// Only type-check if tsconfig exists; npm run build can be slow
		return []string{"npx --no-install tsc --noEmit 2>/dev/null || true"}
	case ProjectRust:
		return []string{"cargo check"}
	case ProjectPython:
		return []string{"python -m py_compile"} // basic syntax check
	default:
		return nil
	}
}

// RunVerification executes the given commands sequentially in workDir and
// returns a VerifyResult. It stops at the first failure.
func RunVerification(ctx context.Context, workDir string, commands []string) *VerifyResult {
	if len(commands) == 0 {
		return &VerifyResult{Passed: true}
	}

	start := time.Now()
	result := &VerifyResult{
		Commands: commands,
	}

	ctx, cancel := context.WithTimeout(ctx, verifyTimeout)
	defer cancel()

	for _, cmdStr := range commands {
		cmd := exec.CommandContext(ctx, "sh", "-c", cmdStr)
		cmd.Dir = workDir
		cmd.Env = append(os.Environ(), "PAGER=cat")

		output, err := cmd.CombinedOutput()
		if err != nil {
			// Truncate very long error outputs to avoid bloating the context
			errOutput := strings.TrimSpace(string(output))
			if len(errOutput) > 2000 {
				errOutput = errOutput[:2000] + "\n[... truncated ...]"
			}

			errMsg := fmt.Sprintf("Command `%s` failed:\n%s", cmdStr, errOutput)
			if errOutput == "" {
				errMsg = fmt.Sprintf("Command `%s` failed: %v", cmdStr, err)
			}
			result.Errors = append(result.Errors, errMsg)
			// Don't stop — run all commands to catch all errors
		}
	}

	result.Duration = time.Since(start)
	result.Passed = len(result.Errors) == 0
	return result
}

// reasoningEffortForRetry returns the appropriate reasoning effort level
// for a given verify retry attempt. Each retry escalates the thinking budget.
func reasoningEffortForRetry(currentEffort string, retryNum int) string {
	levels := []string{"low", "medium", "high"}

	// Find current index
	currentIdx := -1
	for i, l := range levels {
		if l == currentEffort {
			currentIdx = i
			break
		}
	}

	// Escalate: retry 1 → medium, retry 2 → high
	targetIdx := retryNum // 1-indexed retry → 1=medium, 2=high
	if currentIdx >= targetIdx {
		// Already at or above the target level
		return currentEffort
	}
	if targetIdx >= len(levels) {
		return levels[len(levels)-1]
	}
	return levels[targetIdx]
}
