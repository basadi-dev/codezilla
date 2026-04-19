package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// MultiReplaceTool makes one or many targeted text replacements in a file.
type MultiReplaceTool struct{}

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

func NewMultiReplaceTool() *MultiReplaceTool { return &MultiReplaceTool{} }

func (t *MultiReplaceTool) Name() string { return "multiReplace" }

func (t *MultiReplaceTool) Description() string {
	return "Surgically replaces one or more distinct text blocks inside an existing file. " +
		"Each replacement is identified by its exact literal `target_content` string (must be unique in the file). " +
		"IMPORTANT: Always read the file with fileManage (action:read) BEFORE calling this so that target_content matches the current file content exactly. " +
		"Use `allow_multiple: true` on a chunk to replace all occurrences of that pattern. " +
		"All replacements are applied atomically — one file write, one undo entry."
}

func (t *MultiReplaceTool) ParameterSchema() JSONSchema {
	chunkSchema := JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"target_content": {
				Type:        "string",
				Description: "The exact literal block to find and replace. Must match file content including whitespace/indentation.",
			},
			"replacement_content": {
				Type:        "string",
				Description: "The new content that replaces target_content.",
			},
			"allow_multiple": {
				Type:        "boolean",
				Description: "If true, replace all occurrences of target_content. Default false (errors if not unique).",
			},
		},
		Required: []string{"target_content", "replacement_content"},
	}

	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"file_path": {
				Type:        "string",
				Description: "Path to the file to edit.",
			},
			"replacements": {
				Type:        "array",
				Items:       &chunkSchema,
				Description: "One or more replacement operations to apply in order.",
			},
		},
		Required: []string{"file_path", "replacements"},
	}
}

func (t *MultiReplaceTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	filePath, _ := params["file_path"].(string)
	if filePath == "" {
		return nil, &ErrInvalidToolParams{ToolName: t.Name(), Message: "file_path is required"}
	}

	// Expand ~
	if len(filePath) > 0 && filePath[0] == '~' {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, &ErrToolExecution{ToolName: t.Name(), Message: "failed to expand ~", Err: err}
		}
		filePath = filepath.Join(homeDir, filePath[1:])
	}
	filePath = filepath.Clean(filePath)
	if !filepath.IsAbs(filePath) {
		cwd, _ := os.Getwd()
		filePath = filepath.Join(cwd, filePath)
	}

	// Acquire per-file lock — ensures atomicity when the orchestrator runs
	// multiple tool calls concurrently and two end up targeting the same file.
	unlock := acquireFileLock(filePath)
	defer unlock()

	// Enforce read-before-write: the LLM must have read the file in this session
	// so that target_content strings are guaranteed to match the live content.
	if err := EnforceReadBeforeWrite(filePath, t.Name()); err != nil {
		return nil, &ErrToolExecution{ToolName: t.Name(), Message: err.Error()}
	}

	existingBytes, err := os.ReadFile(filePath)
	if err != nil {
		return nil, &ErrToolExecution{ToolName: t.Name(), Message: fmt.Sprintf("failed to read %s", filePath), Err: err}
	}
	original := string(existingBytes)

	// One backup per call regardless of chunk count, so /undo is one step.
	PushBackup(filePath, original)

	replacementsRaw, ok := params["replacements"].([]interface{})
	if !ok {
		return nil, &ErrInvalidToolParams{ToolName: t.Name(), Message: "replacements must be an array"}
	}

	content := original
	for i, raw := range replacementsRaw {
		chunk, ok := raw.(map[string]interface{})
		if !ok {
			return nil, &ErrInvalidToolParams{ToolName: t.Name(), Message: fmt.Sprintf("replacement[%d] is not an object", i)}
		}

		target, _ := chunk["target_content"].(string)
		replacement, _ := chunk["replacement_content"].(string)
		allowMultiple, _ := chunk["allow_multiple"].(bool)

		if target == "" {
			return nil, &ErrInvalidToolParams{ToolName: t.Name(), Message: fmt.Sprintf("replacement[%d].target_content is empty", i)}
		}

		occurrences := strings.Count(content, target)
		switch {
		case occurrences == 0:
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message: fmt.Sprintf(
					"replacement[%d]: target_content not found in file. "+
						"First line searched: %q. First line of file: %q. "+
						"Re-read the file first to ensure exact whitespace match.",
					i, firstLine(target), firstLine(content)),
			}
		case occurrences > 1 && !allowMultiple:
			return nil, &ErrToolExecution{
				ToolName: t.Name(),
				Message: fmt.Sprintf(
					"replacement[%d]: target_content appears %d times — make the match more specific or set allow_multiple:true to replace all occurrences",
					i, occurrences),
			}
		case allowMultiple:
			content = strings.ReplaceAll(content, target, replacement)
		default:
			content = strings.Replace(content, target, replacement, 1)
		}
	}

	diffOutput := GenerateDiff(original, content, 3)

	fmt.Fprintf(os.Stderr, "\nMULTI REPLACE DIFF\nFile: %s\nApplied %d replacement(s)\n%s\n\n",
		filePath, len(replacementsRaw), diffOutput)

	var fileMode os.FileMode = 0644
	if info, err := os.Stat(filePath); err == nil {
		fileMode = info.Mode()
	}
	if err := os.WriteFile(filePath, []byte(content), fileMode); err != nil {
		return nil, &ErrToolExecution{ToolName: t.Name(), Message: fmt.Sprintf("failed to write %s", filePath), Err: err}
	}

	return map[string]interface{}{
		"success":      true,
		"file_path":    filePath,
		"replacements": len(replacementsRaw),
		"diff":         diffOutput,
	}, nil
}
