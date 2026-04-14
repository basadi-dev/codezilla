package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

// RepoMapGeneratorTool scans a codebase and returns a highly compressed structural outline
// using language-specific pseudo-AST regular expressions.
type RepoMapGeneratorTool struct{}

// NewRepoMapGeneratorTool creates a new RepoMapGeneratorTool
func NewRepoMapGeneratorTool() *RepoMapGeneratorTool {
	return &RepoMapGeneratorTool{}
}

// Name returns the tool name
func (t *RepoMapGeneratorTool) Name() string {
	return "repoMapGenerator"
}

// Description returns the tool description
func (t *RepoMapGeneratorTool) Description() string {
	return "Scans the target directory and outputs an ultra-condensed repo map (classes, functions, interfaces, structs) for supported files, helping to quickly locate logical components without reading whole files."
}

// ParameterSchema returns the JSON schema for this tool's parameters
func (t *RepoMapGeneratorTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"dir": {
				Type:        "string",
				Description: "Base directory to build the repo map for (default: current directory).",
			},
			"specificDirs": {
				Type: "array",
				Items: &JSONSchema{
					Type: "string",
				},
				Description: "Specific directories relative to base dir to focus the map on.",
			},
			"excludePatterns": {
				Type: "array",
				Items: &JSONSchema{
					Type: "string",
				},
				Description: "Additional glob patterns to exclude (defaults like vendor, node_modules are already excluded).",
			},
		},
	}
}

// languageHeuristics holds regular expressions for extracting pseudo-ASTs
var languageHeuristics = map[string]*regexp.Regexp{
	"go": regexp.MustCompile(`^(?P<indent>\s*)(?:type\s+\w+\s+(?:struct|interface)|func\s+(?:\([^)]+\)\s+)?\w+\s*\()`),
	"py": regexp.MustCompile(`^(?P<indent>\s*)(?:class|def)\s+\w+\s*(?:\([^)]*\))?:`),
	"js": regexp.MustCompile(`^(?P<indent>\s*)(?:(?:export\s+)?(?:class|function)\s+\w+|(?:export\s+const\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|[^\s=]+)\s*=>))`),
	"ts": regexp.MustCompile(`^(?P<indent>\s*)(?:(?:export\s+)?(?:class|function|interface|type)\s+\w+|(?:export\s+const\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|[^\s=]+)\s*=>))`),
	"rs": regexp.MustCompile(`^(?P<indent>\s*)(?:(?:pub\s+)?(?:struct|enum|trait|fn|impl(?:<[^>]*>)?)\s+\w+)`),
}

// RepoMapResult holds the structural map
type RepoMapResult struct {
	Directory string   `json:"directory"`
	MapLines  []string `json:"map_lines"`
}

// Execute performs the repo map generation
func (t *RepoMapGeneratorTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	dir, _ := params["dir"].(string)
	if dir == "" {
		var err error
		dir, err = os.Getwd()
		if err != nil {
			return nil, fmt.Errorf("failed to get current directory: %w", err)
		}
	}

	dir, err := ValidateAndCleanPath(dir)
	if err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  "invalid directory path",
			Err:      err,
		}
	}

	excludePatterns := getDefaultExcludePatterns()
	if customExcludes, ok := params["excludePatterns"].([]string); ok {
		excludePatterns = append(excludePatterns, customExcludes...)
	}

	var specificDirs []string
	if dirs, ok := params["specificDirs"].([]interface{}); ok {
		for _, d := range dirs {
			if dirStr, ok := d.(string); ok {
				specificDirs = append(specificDirs, dirStr)
			}
		}
	}

	// 1. Gather all target files
	var targetFiles []string
	if len(specificDirs) > 0 {
		for _, sd := range specificDirs {
			targetDir := filepath.Join(dir, sd)
			_ = filepath.Walk(targetDir, func(path string, info os.FileInfo, err error) error {
				if err == nil && !info.IsDir() && !matchesAnyPattern(path, excludePatterns) {
					targetFiles = append(targetFiles, path)
				}
				if info != nil && info.IsDir() && matchesAnyPattern(path, excludePatterns) {
					return filepath.SkipDir
				}
				return nil
			})
		}
	} else {
		_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() && !matchesAnyPattern(path, excludePatterns) {
				targetFiles = append(targetFiles, path)
			}
			if info != nil && info.IsDir() && matchesAnyPattern(path, excludePatterns) {
				return filepath.SkipDir
			}
			return nil
		})
	}

	// 2. Process concurrently
	var mu sync.Mutex
	mapLines := make([]string, 0)
	var wg sync.WaitGroup
	sem := make(chan struct{}, 10) // up to 10 files at once

	for _, path := range targetFiles {
		ext := strings.TrimPrefix(filepath.Ext(path), ".")
		re, supported := languageHeuristics[ext]
		if !supported {
			continue
		}

		wg.Add(1)
		go func(filePath string, parser *regexp.Regexp) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			content, err := os.ReadFile(filePath)
			if err != nil {
				return
			}

			relPath, _ := filepath.Rel(dir, filePath)
			var localLines []string

			lines := strings.Split(string(content), "\n")
			for i, line := range lines {
				if parser.MatchString(line) {
					cleanLine := strings.TrimRight(line, " {\n\r:")
					localLines = append(localLines, fmt.Sprintf("%d: %s", i+1, cleanLine))
				}
			}

			if len(localLines) > 0 {
				mu.Lock()
				mapLines = append(mapLines, fmt.Sprintf("File: %s", relPath))
				mapLines = append(mapLines, localLines...)
				mapLines = append(mapLines, "") // separator
				mu.Unlock()
			}
		}(path, re)
	}

	wg.Wait()

	return &RepoMapResult{
		Directory: dir,
		MapLines:  mapLines,
	}, nil
}
