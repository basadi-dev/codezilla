package tools

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// maxFullReadLines is the max lines before auto-truncating a read.
const maxFullReadLines = 500

// FileManageTool handles all standard filesystem actions (read, write, edit, list, mkdir, move, delete)
type FileManageTool struct{}

func NewFileManageTool() *FileManageTool {
	return &FileManageTool{}
}

func (t *FileManageTool) Name() string { return "fileManage" }

func (t *FileManageTool) Description() string {
	return "Unified file system operations: read, write, edit, list, mkdir, move, delete."
}

func (t *FileManageTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"action": {
				Type:        "string",
				Description: "Action to perform: 'read', 'write', 'edit', 'list', 'mkdir', 'move', 'delete'",
				Enum:        []interface{}{"read", "write", "edit", "list", "mkdir", "move", "delete"},
			},
			"path": {
				Type:        "string",
				Description: "Target file or directory path",
			},
			"content": {
				Type:        "string",
				Description: "(write) Content to write",
			},
			"line_start": {
				Type:        "integer",
				Description: "(read) First line to read (1-based)",
			},
			"line_end": {
				Type:        "integer",
				Description: "(read) Last line to read (inclusive)",
			},
			"replacement_chunks": {
				Type:        "array",
				Description: "(edit) Array of replacements",
				Items: &JSONSchema{
					Type: "object",
					Properties: map[string]JSONSchema{
						"start_line":          {Type: "integer"},
						"end_line":            {Type: "integer"},
						"target_content":      {Type: "string"},
						"replacement_content": {Type: "string"},
						"allow_multiple":      {Type: "boolean"},
					},
					Required: []string{"start_line", "end_line", "target_content", "replacement_content"},
				},
			},
			"pattern": {
				Type:        "string",
				Description: "(list) Glob pattern to filter files (e.g. '*.go')",
			},
			"maxDepth": {
				Type:        "integer",
				Description: "(list) Max recursion depth (0 for unlimited)",
			},
			"includeHidden": {Type: "boolean"},
			"readContents":  {Type: "boolean"},
			"destination_path": {
				Type:        "string",
				Description: "(move) Destination path",
			},
		},
		Required: []string{"action", "path"},
	}
}

func (t *FileManageTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	action, _ := params["action"].(string)
	targetPath, _ := params["path"].(string)

	if len(targetPath) > 0 && targetPath[0] == '~' {
		homeDir, _ := os.UserHomeDir()
		targetPath = filepath.Join(homeDir, targetPath[1:])
	}
	var err error
	if action != "mkdir" && action != "write" && action != "move" && action != "delete" && action != "list" {
		targetPath, err = ValidateAndCleanPath(targetPath)
		if err != nil {
			return nil, &ErrToolExecution{ToolName: t.Name(), Message: "invalid path", Err: err}
		}
	} else if action == "list" && targetPath == "" {
		targetPath, _ = os.Getwd()
	} else {
		// Clean the path locally avoiding strict must-exist checks for write/mkdir
		targetPath = filepath.Clean(targetPath)
		if !filepath.IsAbs(targetPath) {
			cwd, _ := os.Getwd()
			targetPath = filepath.Join(cwd, targetPath)
		}
	}

	switch action {
	case "read":
		return t.execRead(targetPath, params)
	case "write":
		return t.execWrite(targetPath, params)
	case "edit":
		return t.execEdit(targetPath, params)
	case "list":
		return t.execList(targetPath, params)
	case "mkdir":
		if err := os.MkdirAll(targetPath, 0755); err != nil {
			return nil, err
		}
		return map[string]interface{}{"success": true, "message": "Directory created"}, nil
	case "delete":
		if err := os.RemoveAll(targetPath); err != nil {
			return nil, err
		}
		return map[string]interface{}{"success": true, "message": "Path deleted"}, nil
	case "move":
		destPath, ok := params["destination_path"].(string)
		if !ok || destPath == "" {
			return nil, fmt.Errorf("destination_path required for move")
		}
		if len(destPath) > 0 && destPath[0] == '~' {
			homeDir, _ := os.UserHomeDir()
			destPath = filepath.Join(homeDir, destPath[1:])
		}
		destPath = filepath.Clean(destPath)
		if !filepath.IsAbs(destPath) {
			cwd, _ := os.Getwd()
			destPath = filepath.Join(cwd, destPath)
		}
		os.MkdirAll(filepath.Dir(destPath), 0755)
		if err := os.Rename(targetPath, destPath); err != nil {
			return nil, err
		}
		return map[string]interface{}{"success": true, "message": "Moved successfully"}, nil
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

func (t *FileManageTool) execRead(filePath string, params map[string]interface{}) (interface{}, error) {
	info, err := os.Stat(filePath)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		return nil, fmt.Errorf("path is a directory")
	}
	if info.Size() > 10*1024*1024 {
		return nil, fmt.Errorf("file too large (>10MB)")
	}

	lineStart, _ := toInt(params["line_start"])
	lineEnd, _ := toInt(params["line_end"])

	if lineStart > 0 || lineEnd > 0 {
		return readLineRange(filePath, lineStart, lineEnd)
	}
	return readWithTruncation(filePath)
}

func (t *FileManageTool) execWrite(filePath string, params map[string]interface{}) (interface{}, error) {
	content, _ := params["content"].(string)
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		return nil, err
	}
	return fmt.Sprintf("Successfully wrote %d bytes to %s", len(content), filePath), nil
}

func (t *FileManageTool) execEdit(filePath string, params map[string]interface{}) (interface{}, error) {
	chunks, ok := params["replacement_chunks"].([]interface{})
	if !ok || len(chunks) == 0 {
		return nil, fmt.Errorf("replacement_chunks must be a non-empty array")
	}

	editTool := NewFileEditTool()
	var results []interface{}

	for _, rawChunk := range chunks {
		chunk, ok := rawChunk.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid chunk format in replacement_chunks")
		}

		res, err := editTool.Execute(context.Background(), map[string]interface{}{
			"file_path":           filePath,
			"target_content":      chunk["target_content"],
			"replacement_content": chunk["replacement_content"],
		})
		if err != nil {
			return nil, err
		}
		results = append(results, res)
	}

	// Just return success message or the last result
	return map[string]interface{}{
		"success": true,
		"message": fmt.Sprintf("Successfully applied %d edits to %s", len(chunks), filePath),
		"results": results,
	}, nil
}

func (t *FileManageTool) execList(dir string, params map[string]interface{}) (interface{}, error) {
	listTool := NewListFilesTool()
	p := map[string]interface{}{"dir": dir}
	for k, v := range params {
		if k != "action" && k != "path" {
			p[k] = v
		}
	}
	return listTool.Execute(context.Background(), p)
}

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
	scanner.Buffer(make([]byte, 256*1024), 1024*1024)
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
	hint := fmt.Sprintf("\n\n[TRUNCATED] File has %d lines, showing first %d. Use line_start/line_end.", lineNum, maxFullReadLines)
	return sb.String() + hint, nil
}
