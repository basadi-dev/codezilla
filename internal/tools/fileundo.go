package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

// FileUndoTool restores a file to its state before the most recent fileEdit,
// multiReplace, or fileManage-edit operation performed in the current session.
type FileUndoTool struct{}

// NewFileUndoTool creates a new FileUndoTool.
func NewFileUndoTool() *FileUndoTool { return &FileUndoTool{} }

func (t *FileUndoTool) Name() string { return "fileUndo" }

func (t *FileUndoTool) Description() string {
	return "Reverts a file to its state before the most recent fileEdit or multiReplace call in this session. " +
		"Each call undoes exactly one edit operation. Call again to undo further."
}

func (t *FileUndoTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"file_path": {
				Type:        "string",
				Description: "Absolute or relative path (with ~) to the file to undo.",
			},
		},
		Required: []string{"file_path"},
	}
}

// Execute pops the most-recent backup for the given file and writes it back to disk.
func (t *FileUndoTool) Execute(_ context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	filePath, _ := params["file_path"].(string)

	// Expand ~ to home directory
	if len(filePath) > 0 && filePath[0] == '~' {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, &ErrToolExecution{ToolName: t.Name(), Message: "failed to expand home directory", Err: err}
		}
		filePath = filepath.Join(homeDir, filePath[1:])
	}

	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return nil, &ErrToolExecution{ToolName: t.Name(), Message: "failed to resolve path", Err: err}
	}

	// Acquire the per-file lock before reading the backup and writing.
	unlock := acquireFileLock(absPath)
	defer unlock()

	prev, ok := PopBackup(absPath)
	if !ok {
		remaining := UndoDepth(absPath)
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message: fmt.Sprintf(
				"no undo history for %q in this session (history depth: %d). "+
					"Only edits made with fileEdit or multiReplace during the current session can be undone.",
				absPath, remaining),
		}
	}

	// Read the current content so we can show what was restored.
	currentBytes, _ := os.ReadFile(absPath)
	currentContent := string(currentBytes)

	// Preserve original file permissions.
	var fileMode os.FileMode = 0644
	if info, statErr := os.Stat(absPath); statErr == nil {
		fileMode = info.Mode()
	}

	if err := os.WriteFile(absPath, []byte(prev), fileMode); err != nil {
		// If the write failed, push the backup back so the user can retry.
		PushBackup(absPath, prev)
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("failed to restore file: %s", absPath),
			Err:      err,
		}
	}

	// After a successful undo, update the read tracker so subsequent edits
	// don't see a "file modified since last read" error.
	RecordFileRead(absPath)

	diff := GenerateDiff(currentContent, prev, 3)

	return map[string]interface{}{
		"success":        true,
		"file_path":      absPath,
		"restored_bytes": len(prev),
		"remaining_undo": UndoDepth(absPath),
		"diff":           diff,
		"message": fmt.Sprintf(
			"Restored %q to previous version (%d→%d bytes). %d undo step(s) remaining.",
			absPath, len(currentContent), len(prev), UndoDepth(absPath)),
	}, nil
}
