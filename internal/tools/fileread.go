package tools

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
)

const maxFullReadLines = 500 // auto-truncate files larger than this

// FileReadTool allows reading file contents
type FileReadTool struct{}

// NewFileReadTool creates a new file read tool
func NewFileReadTool() *FileReadTool {
	return &FileReadTool{}
}

// Name returns the tool name
func (t *FileReadTool) Name() string {
	return "fileRead"
}

// Description returns the tool description
func (t *FileReadTool) Description() string {
	return "Read a file. For large files use line_start/line_end to read a range."
}

// ParameterSchema returns the JSON schema for this tool's parameters
func (t *FileReadTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"file_path": {
				Type:        "string",
				Description: "Path to the file to read",
			},
			"line_start": {
				Type:        "integer",
				Description: "First line to read (1-based, optional)",
			},
			"line_end": {
				Type:        "integer",
				Description: "Last line to read (1-based, inclusive, optional)",
			},
		},
		Required: []string{"file_path"},
	}
}

// Execute reads the file and returns its contents
func (t *FileReadTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	// Validate parameters
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	// Extract file path
	filePath, ok := params["file_path"].(string)
	if !ok {
		return nil, &ErrInvalidToolParams{
			ToolName: t.Name(),
			Message:  "file_path must be a string",
		}
	}

	// Validate and clean the path
	filePath, err := ValidateAndCleanPath(filePath)
	if err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  "invalid file path",
			Err:      err,
		}
	}

	// Make sure the file exists
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("failed to access file: %s", filePath),
			Err:      err,
		}
	}

	// Make sure it's a regular file (not a directory)
	if fileInfo.IsDir() {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("path is a directory, not a file: %s", filePath),
		}
	}

	// Check file size to prevent loading very large files
	if fileInfo.Size() > 10*1024*1024 { // 10MB limit
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("file too large (>10MB): %s", filePath),
		}
	}

	// Parse optional line range
	lineStart := 0
	lineEnd := 0
	if v, ok := toInt(params["line_start"]); ok && v > 0 {
		lineStart = v
	}
	if v, ok := toInt(params["line_end"]); ok && v > 0 {
		lineEnd = v
	}

	// If a line range was requested, read only those lines
	if lineStart > 0 || lineEnd > 0 {
		return readLineRange(filePath, lineStart, lineEnd)
	}

	// Full read — but auto-truncate if too large
	return readWithTruncation(filePath)
}

// readLineRange reads specific lines [start, end] (1-based, inclusive).
func readLineRange(filePath string, start, end int) (interface{}, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	if start <= 0 {
		start = 1
	}

	var sb strings.Builder
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 256*1024), 1024*1024) // handle long lines
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		if lineNum < start {
			continue
		}
		if end > 0 && lineNum > end {
			break
		}
		fmt.Fprintf(&sb, "%d: %s\n", lineNum, scanner.Text())
	}

	totalLines := lineNum
	// If we stopped early count remaining lines
	if end > 0 && end < totalLines {
		for scanner.Scan() {
			lineNum++
		}
		totalLines = lineNum
	}

	header := fmt.Sprintf("[%s] lines %d–%d of %d total\n", filePath, start, end, totalLines)
	if end <= 0 || end >= totalLines {
		header = fmt.Sprintf("[%s] lines %d–%d (end of file)\n", filePath, start, totalLines)
	}
	return header + sb.String(), nil
}

// readWithTruncation reads the full file but truncates if it exceeds maxFullReadLines.
func readWithTruncation(filePath string) (interface{}, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 256*1024), 1024*1024)

	var sb strings.Builder
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		if lineNum <= maxFullReadLines {
			sb.WriteString(scanner.Text())
			sb.WriteByte('\n')
		}
	}

	if lineNum <= maxFullReadLines {
		return sb.String(), nil
	}

	// File was truncated — return partial content with a hint
	hint := fmt.Sprintf(
		"\n\n[TRUNCATED] File has %d lines, showing first %d. Use line_start/line_end to read specific sections.",
		lineNum, maxFullReadLines,
	)
	return sb.String() + hint, nil
}

// toInt converts an interface{} (float64 or int from JSON) to int.
func toInt(v interface{}) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	}
	return 0, false
}
