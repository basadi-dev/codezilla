package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

// FileManageTool allows moving, renaming, deleting, and creating directories
type FileManageTool struct{}

// NewFileManageTool creates a new file manage tool
func NewFileManageTool() *FileManageTool {
	return &FileManageTool{}
}

// Name returns the tool name
func (t *FileManageTool) Name() string {
	return "fileManage"
}

// Description returns the tool description
func (t *FileManageTool) Description() string {
	return "Performs basic file system operations such as creating directories (mkdir), moving/renaming items (move), or deleting items (delete)."
}

// ParameterSchema returns the JSON schema for this tool's parameters
func (t *FileManageTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"action": {
				Type:        "string",
				Description: "The action to perform: 'mkdir', 'move', or 'delete'.",
				Enum:        []interface{}{"mkdir", "move", "delete"},
			},
			"path": {
				Type:        "string",
				Description: "The primary file or directory path to operate on.",
			},
			"destination_path": {
				Type:        "string",
				Description: "The destination path, required ONLY for 'move' actions.",
			},
		},
		Required: []string{"action", "path"},
	}
}

// Execute performs the targeted replacement
func (t *FileManageTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	action, _ := params["action"].(string)
	targetPath, _ := params["path"].(string)

	// Expand ~ to home directory
	if len(targetPath) > 0 && targetPath[0] == '~' {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message:  "failed to expand home directory",
				Err:      err,
			}
		}
		targetPath = filepath.Join(homeDir, targetPath[1:])
	}

	switch action {
	case "mkdir":
		if err := os.MkdirAll(targetPath, 0755); err != nil {
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message:  fmt.Sprintf("failed to create directory: %s", targetPath),
				Err:      err,
			}
		}
		return map[string]interface{}{"success": true, "message": "Directory created successfully"}, nil

	case "delete":
		if err := os.RemoveAll(targetPath); err != nil {
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message:  fmt.Sprintf("failed to delete path: %s", targetPath),
				Err:      err,
			}
		}
		return map[string]interface{}{"success": true, "message": "Path deleted successfully"}, nil

	case "move":
		destPath, ok := params["destination_path"].(string)
		if !ok || destPath == "" {
			return nil, &ErrInvalidToolParams{
				ToolName: t.Name(),
				Message:  "destination_path is required for move action",
			}
		}

		if len(destPath) > 0 && destPath[0] == '~' {
			homeDir, _ := os.UserHomeDir()
			destPath = filepath.Join(homeDir, destPath[1:])
		}

		// Ensure parent directory of destination exists
		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message:  fmt.Sprintf("failed to create destination directory: %s", filepath.Dir(destPath)),
				Err:      err,
			}
		}

		if err := os.Rename(targetPath, destPath); err != nil {
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message:  fmt.Sprintf("failed to move/rename path from %s to %s", targetPath, destPath),
				Err:      err,
			}
		}
		return map[string]interface{}{"success": true, "message": "Moved successfully"}, nil

	default:
		return nil, &ErrInvalidToolParams{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("unknown action: %s", action),
		}
	}
}
