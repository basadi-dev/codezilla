package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

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
	return "Replaces a specific targeted block of text within an existing file. Use this instead of fileWrite to modify large files without rewriting them entirely."
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

	// Check if target content exists
	occurrences := strings.Count(content, targetContent)
	if occurrences == 0 {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  "target_content not found in the file. Ensure the spaces and indentation match exactly.",
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
