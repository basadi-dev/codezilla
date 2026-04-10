package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"codezilla/pkg/style"

	"github.com/charmbracelet/lipgloss"
)

// FileWriteTool allows writing content to a file
type FileWriteTool struct{}

// NewFileWriteTool creates a new file write tool
func NewFileWriteTool() *FileWriteTool {
	return &FileWriteTool{}
}

// Name returns the tool name
func (t *FileWriteTool) Name() string {
	return "fileWrite"
}

// Description returns the tool description
func (t *FileWriteTool) Description() string {
	return "Writes content to a file on the local filesystem with diff preview for existing files"
}

// ParameterSchema returns the JSON schema for this tool's parameters
func (t *FileWriteTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"file_path": {
				Type:        "string",
				Description: "The path to the file to write",
			},
			"content": {
				Type:        "string",
				Description: "The content to write to the file",
			},
			"append": {
				Type:        "boolean",
				Description: "Whether to append to the file instead of overwriting it",
				Default:     false,
			},
			"skip_diff": {
				Type:        "boolean",
				Description: "Skip showing diff for existing files",
				Default:     false,
			},
		},
		Required: []string{"file_path", "content"},
	}
}

// Execute writes the content to the specified file
func (t *FileWriteTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	// Validate parameters
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	// Extract parameters
	filePath, ok := params["file_path"].(string)
	if !ok {
		return nil, &ErrInvalidToolParams{
			ToolName: t.Name(),
			Message:  "file_path must be a string",
		}
	}

	content, ok := params["content"].(string)
	if !ok {
		return nil, &ErrInvalidToolParams{
			ToolName: t.Name(),
			Message:  "content must be a string",
		}
	}

	// Default append to false
	append := false
	if appendParam, ok := params["append"].(bool); ok {
		append = appendParam
	}

	// Optional parameters for diff control
	skipDiff := false
	if skipDiffParam, ok := params["skip_diff"].(bool); ok {
		skipDiff = skipDiffParam
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

	// Ensure directory exists
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("failed to create directory: %s", dir),
			Err:      err,
		}
	}

	// Check if file exists and generate diff if needed
	fileExists := false
	var existingContent string
	var diffOutput string

	if !append {
		// Only check for diff if we're not appending
		fileInfo, err := os.Stat(filePath)
		if err == nil && !fileInfo.IsDir() {
			fileExists = true

			// Only generate diff if not skipped
			if !skipDiff {
				// Read existing content
				existingBytes, err := os.ReadFile(filePath)
				if err == nil {
					existingContent = string(existingBytes)

					// Generate diff if content is different
					if existingContent != content {
						// Generate diff directly
						diffOutput = GenerateDiff(existingContent, content, 3)
					} else {
						diffOutput = "No changes detected."
					}
				}
			}
		}
	}

	// If file exists and changes detected, add diff info to param map for permission request
	// and also display the diff directly to the user in the terminal
	if fileExists && !skipDiff && diffOutput != "No changes detected." {
		// Add diff to params with special "_" prefix to indicate it's internal
		params["_fileDiff"] = diffOutput

		title := "File Diff Preview: " + filePath
		if style.UseColors {
			title = lipgloss.NewStyle().Foreground(lipgloss.Color("#00D7D7")).Bold(true).Render(title)
		}
		fmt.Fprintf(os.Stderr, "\n%s\n%s\n", title, diffOutput)
	}

	// Determine flags based on append mode
	flag := os.O_WRONLY | os.O_CREATE
	if append {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}

	// Get original file permissions if file exists
	var fileMode os.FileMode = 0644
	if info, err := os.Stat(filePath); err == nil {
		fileMode = info.Mode()
	}

	// Open file
	file, err := os.OpenFile(filePath, flag, fileMode)
	if err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("failed to open file for writing: %s", filePath),
			Err:      err,
		}
	}
	defer file.Close()

	// Write content
	_, err = file.WriteString(content)
	if err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("failed to write to file: %s", filePath),
			Err:      err,
		}
	}

	// If a diff was generated, show it again after writing (so user can see what was changed)
	if fileExists && !skipDiff && diffOutput != "No changes detected." {
		var modeStr string
		if append {
			modeStr = "Append"
		} else {
			modeStr = "Overwrite"
		}

		// Create styled box for success message
		var output strings.Builder
		output.WriteString(fmt.Sprintf("File updated: %s\n", filePath))
		output.WriteString(fmt.Sprintf("Bytes written: %d\n", len(content)))
		output.WriteString(fmt.Sprintf("Mode: %s", modeStr))

		boxStyle := lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			Padding(0, 1)

		title := "Changes Successfully Written"
		if style.UseColors {
			boxStyle = boxStyle.BorderForeground(lipgloss.Color("#00D787"))
			title = lipgloss.NewStyle().Foreground(lipgloss.Color("#00D787")).Bold(true).Render(title)
		}

		var finalOutput string
		if style.UseColors {
			finalOutput = boxStyle.Render(output.String())
		} else {
			finalOutput = "==== CHANGES WRITTEN ====\n" + output.String() + "\n========================="
		}

		fmt.Fprintf(os.Stderr, "\n%s\n%s\n\n", title, finalOutput)
	}

	result := map[string]interface{}{
		"success":     true,
		"file_path":   filePath,
		"bytes":       len(content),
		"appended":    append,
		"file_exists": fileExists,
	}

	return result, nil
}
