package tools

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// GrepSearchTool allows searching text inside files
type GrepSearchTool struct{}

// NewGrepSearchTool creates a new search tool
func NewGrepSearchTool() *GrepSearchTool {
	return &GrepSearchTool{}
}

// Name returns the tool name
func (t *GrepSearchTool) Name() string {
	return "grepSearch"
}

// Description returns the tool description
func (t *GrepSearchTool) Description() string {
	return "Searches for a specific text pattern or regular expression across files in a directory. Returns the occurrences of the pattern with line numbers."
}

// ParameterSchema returns the JSON schema for this tool's parameters
func (t *GrepSearchTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"query": {
				Type:        "string",
				Description: "The search query, pattern, or regular expression.",
			},
			"path": {
				Type:        "string",
				Description: "The file or directory path to search in.",
			},
			"is_regex": {
				Type:        "boolean",
				Description: "Set to true if the query is a regular expression.",
				Default:     false,
			},
			"case_sensitive": {
				Type:        "boolean",
				Description: "Set to true to make the search case-sensitive.",
				Default:     false,
			},
		},
		Required: []string{"query", "path"},
	}
}

// Execute performs the search operation
func (t *GrepSearchTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	query, _ := params["query"].(string)
	searchPath, _ := params["path"].(string)

	isRegex := false
	if r, ok := params["is_regex"].(bool); ok {
		isRegex = r
	}

	caseSensitive := false
	if c, ok := params["case_sensitive"].(bool); ok {
		caseSensitive = c
	}

	if query == "" {
		return nil, &ErrInvalidToolParams{
			ToolName: t.Name(),
			Message:  "query cannot be empty",
		}
	}

	// Expand ~ to home directory
	if len(searchPath) > 0 && searchPath[0] == '~' {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message:  "failed to expand home directory",
				Err:      err,
			}
		}
		searchPath = filepath.Join(homeDir, searchPath[1:])
	}

	if _, err := os.Stat(searchPath); os.IsNotExist(err) {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("path does not exist: %s", searchPath),
		}
	}

	// Build grep argument array
	args := []string{"-rnI"} // recursive, line numbers, ignore binaries

	if !caseSensitive {
		args = append(args, "-i")
	}

	if isRegex {
		args = append(args, "-E")
	} else {
		args = append(args, "-F") // fixed strings
	}

	args = append(args, query, searchPath)

	cmd := exec.CommandContext(ctx, "grep", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	output := stdout.String()

	// grep exit status: 0 if selected lines are found, 1 if not found, 2 if an error occurred.
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			if exitError.ExitCode() == 1 {
				// No lines matched, this is not a technical error, just an empty result
				return map[string]interface{}{
					"success": true,
					"matches": []string{},
				}, nil
			}
		}
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("grep returned error: %v", stderr.String()),
			Err:      err,
		}
	}

	// Format results
	lines := strings.Split(strings.TrimSpace(output), "\n")

	// Limit massive results to prevent context overflow
	if len(lines) > 50 {
		lines = lines[:50]
		lines = append(lines, fmt.Sprintf("... (truncated %d more matches)", len(lines)-50))
	}

	return map[string]interface{}{
		"success": true,
		"matches": lines,
	}, nil
}
