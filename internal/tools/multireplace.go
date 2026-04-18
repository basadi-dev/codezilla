package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// MultiReplaceTool allows multiple targeted replacements of text within a file
type MultiReplaceTool struct{}

// NewMultiReplaceTool creates a new multi-replace tool
func NewMultiReplaceTool() *MultiReplaceTool {
	return &MultiReplaceTool{}
}

// Name returns the tool name
func (t *MultiReplaceTool) Name() string {
	return "multiReplace"
}

// Description returns the tool description
func (t *MultiReplaceTool) Description() string {
	return "Replaces multiple specific targeted blocks of text within an existing file sequentially. Use this to modify multiple disconnected areas of a large file."
}

// ParameterSchema returns the JSON schema for this tool's parameters
func (t *MultiReplaceTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"file_path": {
				Type:        "string",
				Description: "The path to the file to edit",
			},
			"replacements": {
				Type: "array",
				Items: &JSONSchema{
					Type: "object",
					Properties: map[string]JSONSchema{
						"target_content": {
							Type:        "string",
							Description: "The exact literal string block to search for and replace. Must match exactly.",
						},
						"replacement_content": {
							Type:        "string",
							Description: "The new content to drop in place of the target_content.",
						},
					},
					Required: []string{"target_content", "replacement_content"},
				},
				Description: "List of replacements to apply to the file one by one",
			},
		},
		Required: []string{"file_path", "replacements"},
	}
}

// Execute performs the targeted replacements sequentially
func (t *MultiReplaceTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	filePath, _ := params["file_path"].(string)

	// Expand ~ to home directory
	if len(filePath) > 0 && filePath[0] == '~' {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message:  "failed to expand home directory",
				Err:      err,
			}
		}
		filePath = filepath.Join(homeDir, filePath[1:])
	}

	// Acquire per-file lock before the read-modify-write sequence.
	// See filelock.go for rationale.
	unlock := acquireFileLock(filePath)
	defer unlock()

	// Enforce read-before-write (same policy as fileEdit).
	if err := EnforceReadBeforeWrite(filePath, t.Name()); err != nil {
		return nil, &ErrToolExecution{ToolName: t.Name(), Message: err.Error()}
	}

	// Read existing content
	existingBytes, err := os.ReadFile(filePath)
	if err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("failed to read file: %s", filePath),
			Err:      err,
		}
	}
	content := string(existingBytes)

	// Push a single backup of the original content. One undo entry per
	// multiReplace call, regardless of how many chunks it contains.
	PushBackup(filePath, content)

	replacementsRaw, ok := params["replacements"].([]interface{})
	if !ok {
		return nil, &ErrInvalidToolParams{
			ToolName: t.Name(),
			Message:  "replacements must be an array of objects",
		}
	}

	originalContent := content
	appliedCount := 0

	for i, repRaw := range replacementsRaw {
		rep, ok := repRaw.(map[string]interface{})
		if !ok {
			return nil, &ErrInvalidToolParams{
				ToolName: t.Name(),
				Message:  fmt.Sprintf("replacement at index %d is not an object", i),
			}
		}

		targetContent, _ := rep["target_content"].(string)
		replacementContent, _ := rep["replacement_content"].(string)

		if targetContent == "" {
			return nil, &ErrInvalidToolParams{
				ToolName: t.Name(),
				Message:  fmt.Sprintf("target_content at index %d cannot be empty", i),
			}
		}

		occurrences := strings.Count(content, targetContent)
		if occurrences == 0 {
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message:  fmt.Sprintf("target_content at index %d not found in the file.", i),
			}
		} else if occurrences > 1 {
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message:  fmt.Sprintf("target_content at index %d found %d times in the file. Must be unique.", i, occurrences),
			}
		}

		content = strings.Replace(content, targetContent, replacementContent, 1)
		appliedCount++
	}

	// Generate diff directly
	diffOutput := GenerateDiff(originalContent, content, 3)

	// Print the diff directly to stderr for immediate visibility
	fmt.Fprintf(os.Stderr, "\nMULTI REPLACE DIFF\n")
	fmt.Fprintf(os.Stderr, "File: %s\n", filePath)
	fmt.Fprintf(os.Stderr, "Applied %d replacements\n", appliedCount)
	fmt.Fprintf(os.Stderr, "%s\n", diffOutput)
	fmt.Fprintf(os.Stderr, "\n")

	// Get original file permissions
	var fileMode os.FileMode = 0644
	if info, err := os.Stat(filePath); err == nil {
		fileMode = info.Mode()
	}

	// Write content
	if err := os.WriteFile(filePath, []byte(content), fileMode); err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("failed to save file: %s", filePath),
			Err:      err,
		}
	}

	result := map[string]interface{}{
		"success":      true,
		"file_path":    filePath,
		"replacements": appliedCount,
		"diff":         diffOutput,
	}

	return result, nil
}
