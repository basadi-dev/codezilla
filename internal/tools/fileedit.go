package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// firstLine returns the first non-empty line of s, truncated to 80 chars.
// Used to provide diagnostic hints in error messages.
func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			if len(line) > 80 {
				return line[:80] + "…"
			}
			return line
		}
	}
	return "(empty)"
}

// FileEditTool allows targeted replacement of text within a file
type FileEditTool struct{}

// NewFileEditTool creates a new file edit tool
func NewFileEditTool() *FileEditTool {
	return &FileEditTool{}
}

// Name returns the tool name
func (t *FileEditTool) Name() string {
	return "fileEdit"
}

// Description returns the tool description
func (t *FileEditTool) Description() string {
	return "Replaces a specific targeted block of text within an existing file. " +
		"IMPORTANT: Always read the file with fileManage (action:read) BEFORE calling this tool so that target_content matches exactly. " +
		"Use multiReplace when making more than one edit to the same file in a single turn."
}

// ParameterSchema returns the JSON schema for this tool's parameters
func (t *FileEditTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"file_path": {
				Type:        "string",
				Description: "The path to the file to edit",
			},
			"target_content": {
				Type:        "string",
				Description: "The exact literal string block to search for and replace. Must match the file text spacing and indentation exactly.",
			},
			"replacement_content": {
				Type:        "string",
				Description: "The new content to drop in place of the target_content.",
			},
		},
		Required: []string{"file_path", "target_content", "replacement_content"},
	}
}

// Execute performs the targeted replacement
func (t *FileEditTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	filePath, _ := params["file_path"].(string)
	targetContent, _ := params["target_content"].(string)
	replacementContent, _ := params["replacement_content"].(string)

	if targetContent == "" {
		return nil, &ErrInvalidToolParams{
			ToolName: t.Name(),
			Message:  "target_content cannot be empty",
		}
	}

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
	// The orchestrator runs tool calls concurrently; without this lock, two
	// simultaneous fileEdit calls on the same file would race and the second
	// write would silently overwrite the first edit.
	unlock := acquireFileLock(filePath)
	defer unlock()

	// Enforce read-before-write: reject the edit if the LLM hasn't read
	// this file yet in the current session, or if it was modified externally
	// since the last read (which would make any target_content stale).
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

	// Save a backup before mutating so the user can /undo this edit.
	PushBackup(filePath, content)

	// Check if target content exists in the file
	occurrences := strings.Count(content, targetContent)
	if occurrences == 0 {
		// Provide a diagnostic hint: show the first line the LLM was searching for,
		// and the first line of the actual file so the model can spot the mismatch.
		wantedFirstLine := firstLine(targetContent)
		fileFirstLine := firstLine(content)
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message: fmt.Sprintf(
				"target_content not found in the file. "+
					"The file must be read with fileManage (action:read) before editing to ensure exact whitespace and indentation match. "+
					"First line you searched for: %q. First line of the actual file: %q.",
				wantedFirstLine, fileFirstLine),
		}
	} else if occurrences > 1 {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("target_content found %d times in the file. Please provide a more unique block of text to replace to avoid ambiguity.", occurrences),
		}
	}

	// Perform replacement
	newContent := strings.Replace(content, targetContent, replacementContent, 1)

	// Generate diff directly
	diffOutput := GenerateDiff(content, newContent, 3)

	// Print the diff directly to stderr for immediate visibility
	fmt.Fprintf(os.Stderr, "\nFILE EDIT DIFF\n")
	fmt.Fprintf(os.Stderr, "File: %s\n", filePath)
	fmt.Fprintf(os.Stderr, "%s\n", diffOutput)
	fmt.Fprintf(os.Stderr, "\n")

	// Get original file permissions
	var fileMode os.FileMode = 0644
	if info, err := os.Stat(filePath); err == nil {
		fileMode = info.Mode()
	}

	// Write content
	if err := os.WriteFile(filePath, []byte(newContent), fileMode); err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("failed to save file: %s", filePath),
			Err:      err,
		}
	}

	result := map[string]interface{}{
		"success":   true,
		"file_path": filePath,
		"diff":      diffOutput,
	}

	return result, nil
}
