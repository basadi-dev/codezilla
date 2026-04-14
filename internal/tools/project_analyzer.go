package tools

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"codezilla/internal/core/llm"
	"codezilla/pkg/logger"
	anyllm "github.com/mozilla-ai/any-llm-go"
)

// ================================
// Core Interfaces and Types
// ================================

// FileAnalyzer interface for analyzing individual files
type FileAnalyzer interface {
	AnalyzeFile(ctx context.Context, filePath string, content string, userQuery string) (*FileAnalysis, error)
}

// FileAnalysis represents the analysis result for a single file
type FileAnalysis struct {
	Summary      string            `json:"summary"`
	KeyFindings  []string          `json:"key_findings"`
	Relevance    float64           `json:"relevance"`
	Issues       []string          `json:"issues,omitempty"`
	Dependencies []string          `json:"dependencies,omitempty"`
	CodeSmells   []string          `json:"code_smells,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// FileCategory represents the category of a file
type FileCategory string

const (
	CategorySource        FileCategory = "source"
	CategoryTest          FileCategory = "test"
	CategoryConfig        FileCategory = "config"
	CategoryDocumentation FileCategory = "documentation"
	CategoryData          FileCategory = "data"
	CategoryBuild         FileCategory = "build"
	CategoryAsset         FileCategory = "asset"
	CategoryOther         FileCategory = "other"
)

// FileTypeInfo contains information about a file type
type FileTypeInfo struct {
	Category    FileCategory
	Extensions  []string
	Keywords    []string
	Importance  float64
	Description string
}

// Note: AnalyzerFactory is defined in analyzer_factory.go

// ================================
// Default File Analyzer
// ================================

// DefaultFileAnalyzer provides basic analysis without LLM
type DefaultFileAnalyzer struct{}

// NewDefaultFileAnalyzer creates a new default analyzer
func NewDefaultFileAnalyzer() *DefaultFileAnalyzer {
	return &DefaultFileAnalyzer{}
}

// AnalyzeFile performs basic file analysis
func (a *DefaultFileAnalyzer) AnalyzeFile(ctx context.Context, filePath string, content string, userQuery string) (*FileAnalysis, error) {
	lines := strings.Split(content, "\n")

	// Simple keyword-based relevance
	relevance := 0.5
	queryLower := strings.ToLower(userQuery)
	contentLower := strings.ToLower(content)

	if strings.Contains(contentLower, queryLower) {
		relevance = 0.8
	}

	return &FileAnalysis{
		Summary:     fmt.Sprintf("File with %d lines", len(lines)),
		KeyFindings: []string{"Basic analysis without LLM"},
		Relevance:   relevance,
		Metadata: map[string]string{
			"analyzer": "default",
			"lines":    fmt.Sprintf("%d", len(lines)),
		},
	}, nil
}

// ================================
// LLM File Analyzer
// ================================

// LLMFileAnalyzer uses an LLM to analyze files
type LLMFileAnalyzer struct {
	llmClient *llm.Client
	provider  string
	model     string
	logger    *logger.Logger
}

// NewLLMFileAnalyzer creates a new LLM-based file analyzer
func NewLLMFileAnalyzer(llmClient *llm.Client, provider, model string, logger *logger.Logger) *LLMFileAnalyzer {
	return &LLMFileAnalyzer{
		llmClient: llmClient,
		provider:  provider,
		model:     model,
		logger:    logger,
	}
}

// AnalyzeFile analyzes a file using LLM
func (a *LLMFileAnalyzer) AnalyzeFile(ctx context.Context, filePath string, content string, userQuery string) (*FileAnalysis, error) {
	// Check if LLM client is available
	if a.llmClient == nil {
		return a.fallbackAnalysis(filePath, content, userQuery), nil
	}
	// Truncate content if too large (keep first and last parts)
	maxContentSize := 4000 // Conservative limit for context
	truncatedContent := content
	if len(content) > maxContentSize {
		halfSize := maxContentSize / 2
		truncatedContent = content[:halfSize] + "\n\n[... content truncated ...]\n\n" + content[len(content)-halfSize:]
	}

	prompt := fmt.Sprintf(`Analyze the following file with respect to this query: "%s"

File: %s

Content:
%s

Provide a structured analysis with:
1. A brief summary (1-2 sentences)
2. Key findings relevant to the query (list 2-5 items)
3. Relevance score (0-1) based on how well this file matches the query
4. Any issues, problems, or code smells found (if applicable)
5. Key dependencies or relationships with other parts of the codebase (if identifiable)

Format your response as JSON with these fields:
- summary: string
- key_findings: array of strings
- relevance: number between 0 and 1
- issues: array of strings (optional)
- dependencies: array of strings (optional)
- code_smells: array of strings (optional)`, userQuery, filePath, truncatedContent)

	messages := []anyllm.Message{
		{
			Role:    "system",
			Content: "You are a code analysis assistant. Provide concise, relevant analysis focused on the user's query. Return valid JSON only.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	response, err := a.llmClient.Complete(ctx, a.provider, a.model, messages, 0.2, nil)
	if err != nil {
		a.logger.Error("LLM analysis failed for %s: %v", filePath, err)
		// Fall back to basic analysis
		return a.fallbackAnalysis(filePath, content, userQuery), nil
	}

	cleanResponse := ""
	if len(response.Choices) > 0 {
		cleanResponse = strings.TrimSpace(response.Choices[0].Message.ContentString())
	}

	// Parse LLM response
	analysis, err := a.parseAnalysisResponse(cleanResponse)
	if err != nil {
		a.logger.Warn("Failed to parse LLM response for %s: %v", filePath, err)
		return a.fallbackAnalysis(filePath, content, userQuery), nil
	}

	return analysis, nil
}

func (a *LLMFileAnalyzer) parseAnalysisResponse(response string) (*FileAnalysis, error) {
	// Try to extract JSON from the response
	// LLMs sometimes wrap JSON in markdown code blocks
	jsonStr := response
	if idx := strings.Index(response, "```json"); idx >= 0 {
		start := idx + 7
		if end := strings.Index(response[start:], "```"); end >= 0 {
			jsonStr = response[start : start+end]
		}
	} else if idx := strings.Index(response, "{"); idx >= 0 {
		// Find the matching closing brace
		if end := strings.LastIndex(response, "}"); end >= idx {
			jsonStr = response[idx : end+1]
		}
	}

	var result struct {
		Summary      string   `json:"summary"`
		KeyFindings  []string `json:"key_findings"`
		Relevance    float64  `json:"relevance"`
		Issues       []string `json:"issues,omitempty"`
		Dependencies []string `json:"dependencies,omitempty"`
		CodeSmells   []string `json:"code_smells,omitempty"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Ensure relevance is within bounds
	if result.Relevance < 0 {
		result.Relevance = 0
	} else if result.Relevance > 1 {
		result.Relevance = 1
	}

	return &FileAnalysis{
		Summary:      result.Summary,
		KeyFindings:  result.KeyFindings,
		Relevance:    result.Relevance,
		Issues:       result.Issues,
		Dependencies: result.Dependencies,
		CodeSmells:   result.CodeSmells,
		Metadata: map[string]string{
			"analyzer": "llm",
		},
	}, nil
}

func (a *LLMFileAnalyzer) fallbackAnalysis(filePath string, content string, userQuery string) *FileAnalysis {
	lines := strings.Split(content, "\n")

	// Simple keyword-based relevance
	relevance := 0.5
	queryLower := strings.ToLower(userQuery)
	contentLower := strings.ToLower(content)

	if strings.Contains(contentLower, queryLower) {
		relevance = 0.8
	}

	return &FileAnalysis{
		Summary:     fmt.Sprintf("File with %d lines (fallback analysis)", len(lines)),
		KeyFindings: []string{"LLM analysis failed, using fallback"},
		Relevance:   relevance,
		Metadata: map[string]string{
			"analyzer": "fallback",
			"lines":    fmt.Sprintf("%d", len(lines)),
		},
	}
}

// ================================
// Category-Specific Analyzers
// ================================

// ConfigFileAnalyzer specializes in configuration file analysis
type ConfigFileAnalyzer struct {
	baseAnalyzer FileAnalyzer
}

func (a *ConfigFileAnalyzer) AnalyzeFile(ctx context.Context, filePath string, content string, userQuery string) (*FileAnalysis, error) {
	// Add configuration-specific context
	enhancedQuery := fmt.Sprintf("%s [This is a configuration file. Look for settings, parameters, environment variables, and configuration options.]", userQuery)

	analysis, err := a.baseAnalyzer.AnalyzeFile(ctx, filePath, content, enhancedQuery)
	if err != nil {
		return nil, err
	}

	// Boost relevance for config files when searching for settings
	if strings.Contains(strings.ToLower(userQuery), "config") ||
		strings.Contains(strings.ToLower(userQuery), "setting") {
		analysis.Relevance = min(1.0, analysis.Relevance*1.2)
	}

	return analysis, nil
}

// TestFileAnalyzer specializes in test file analysis
type TestFileAnalyzer struct {
	baseAnalyzer FileAnalyzer
}

func (a *TestFileAnalyzer) AnalyzeFile(ctx context.Context, filePath string, content string, userQuery string) (*FileAnalysis, error) {
	// Add test-specific context
	enhancedQuery := fmt.Sprintf("%s [This is a test file. Look for test cases, assertions, test coverage, and testing patterns.]", userQuery)

	analysis, err := a.baseAnalyzer.AnalyzeFile(ctx, filePath, content, enhancedQuery)
	if err != nil {
		return nil, err
	}

	// Boost relevance for test files when searching for tests
	if strings.Contains(strings.ToLower(userQuery), "test") {
		analysis.Relevance = min(1.0, analysis.Relevance*1.2)
	}

	return analysis, nil
}

// DocumentationAnalyzer specializes in documentation analysis
type DocumentationAnalyzer struct {
	baseAnalyzer FileAnalyzer
}

func (a *DocumentationAnalyzer) AnalyzeFile(ctx context.Context, filePath string, content string, userQuery string) (*FileAnalysis, error) {
	// Add documentation-specific context
	enhancedQuery := fmt.Sprintf("%s [This is a documentation file. Look for explanations, guides, API documentation, and usage examples.]", userQuery)

	analysis, err := a.baseAnalyzer.AnalyzeFile(ctx, filePath, content, enhancedQuery)
	if err != nil {
		return nil, err
	}

	// Boost relevance for docs when searching for documentation
	if strings.Contains(strings.ToLower(userQuery), "doc") ||
		strings.Contains(strings.ToLower(userQuery), "guide") ||
		strings.Contains(strings.ToLower(userQuery), "example") {
		analysis.Relevance = min(1.0, analysis.Relevance*1.2)
	}

	return analysis, nil
}

// ================================
// Enhanced Project Scan Analyzer
// ================================

// EnhancedProjectScanAnalyzer adds file categorization and specialized analysis
type EnhancedProjectScanAnalyzer struct {
	*ProjectScanAnalyzerTool
	fileTypeRegistry  map[string]FileTypeInfo
	categoryAnalyzers map[FileCategory]FileAnalyzer
	analysisCache     *AnalysisCache
	mu                sync.RWMutex
}

// NewEnhancedProjectScanAnalyzer creates an enhanced analyzer with categorization
func NewEnhancedProjectScanAnalyzer(baseAnalyzer FileAnalyzer) *EnhancedProjectScanAnalyzer {
	analyzer := &EnhancedProjectScanAnalyzer{
		ProjectScanAnalyzerTool: NewProjectScanAnalyzerTool(),
		fileTypeRegistry:        make(map[string]FileTypeInfo),
		categoryAnalyzers:       make(map[FileCategory]FileAnalyzer),
		analysisCache:           NewAnalysisCache(100, 15*time.Minute),
	}

	// Initialize file type registry
	analyzer.initializeFileTypes()

	// Initialize category-specific analyzers
	analyzer.categoryAnalyzers[CategoryConfig] = &ConfigFileAnalyzer{baseAnalyzer: baseAnalyzer}
	analyzer.categoryAnalyzers[CategoryTest] = &TestFileAnalyzer{baseAnalyzer: baseAnalyzer}
	analyzer.categoryAnalyzers[CategoryDocumentation] = &DocumentationAnalyzer{baseAnalyzer: baseAnalyzer}

	// Use base analyzer for other categories
	analyzer.categoryAnalyzers[CategorySource] = baseAnalyzer
	analyzer.categoryAnalyzers[CategoryData] = baseAnalyzer
	analyzer.categoryAnalyzers[CategoryBuild] = baseAnalyzer
	analyzer.categoryAnalyzers[CategoryAsset] = baseAnalyzer
	analyzer.categoryAnalyzers[CategoryOther] = baseAnalyzer

	return analyzer
}

func (a *EnhancedProjectScanAnalyzer) initializeFileTypes() {
	// Source code files
	a.registerFileType("go", FileTypeInfo{
		Category:    CategorySource,
		Keywords:    []string{"func", "package", "import", "type", "struct"},
		Importance:  0.9,
		Description: "Go source code",
	})

	a.registerFileType("js", FileTypeInfo{
		Category:    CategorySource,
		Keywords:    []string{"function", "const", "let", "var", "class", "export", "import"},
		Importance:  0.9,
		Description: "JavaScript source code",
	})

	a.registerFileType("ts", FileTypeInfo{
		Category:    CategorySource,
		Keywords:    []string{"interface", "type", "class", "export", "import"},
		Importance:  0.9,
		Description: "TypeScript source code",
	})

	a.registerFileType("py", FileTypeInfo{
		Category:    CategorySource,
		Keywords:    []string{"def", "class", "import", "from", "__init__"},
		Importance:  0.9,
		Description: "Python source code",
	})

	// Test files
	a.registerFileType("test.go", FileTypeInfo{
		Category:    CategoryTest,
		Keywords:    []string{"testing.T", "Test", "assert", "require"},
		Importance:  0.8,
		Description: "Go test files",
	})

	a.registerFileType("_test.go", FileTypeInfo{
		Category:    CategoryTest,
		Keywords:    []string{"testing.T", "Test", "assert", "require"},
		Importance:  0.8,
		Description: "Go test files",
	})

	a.registerFileType("test.js", FileTypeInfo{
		Category:    CategoryTest,
		Keywords:    []string{"describe", "it", "test", "expect"},
		Importance:  0.8,
		Description: "JavaScript test files",
	})

	// Configuration files
	a.registerFileType("json", FileTypeInfo{
		Category:    CategoryConfig,
		Keywords:    []string{},
		Importance:  0.7,
		Description: "JSON configuration",
	})

	a.registerFileType("yaml", FileTypeInfo{
		Category:    CategoryConfig,
		Keywords:    []string{},
		Importance:  0.7,
		Description: "YAML configuration",
	})

	a.registerFileType("yml", FileTypeInfo{
		Category:    CategoryConfig,
		Keywords:    []string{},
		Importance:  0.7,
		Description: "YAML configuration",
	})

	a.registerFileType("toml", FileTypeInfo{
		Category:    CategoryConfig,
		Keywords:    []string{},
		Importance:  0.7,
		Description: "TOML configuration",
	})

	// Documentation files
	a.registerFileType("md", FileTypeInfo{
		Category:    CategoryDocumentation,
		Keywords:    []string{"#", "##", "###"},
		Importance:  0.6,
		Description: "Markdown documentation",
	})

	a.registerFileType("rst", FileTypeInfo{
		Category:    CategoryDocumentation,
		Keywords:    []string{},
		Importance:  0.6,
		Description: "reStructuredText documentation",
	})

	// Build files
	a.registerFileType("Makefile", FileTypeInfo{
		Category:    CategoryBuild,
		Keywords:    []string{"target:", "$(", "${"},
		Importance:  0.8,
		Description: "Make build file",
	})

	a.registerFileType("Dockerfile", FileTypeInfo{
		Category:    CategoryBuild,
		Keywords:    []string{"FROM", "RUN", "CMD", "COPY"},
		Importance:  0.8,
		Description: "Docker build file",
	})

	// Data files
	a.registerFileType("csv", FileTypeInfo{
		Category:    CategoryData,
		Keywords:    []string{},
		Importance:  0.5,
		Description: "CSV data file",
	})

	a.registerFileType("sql", FileTypeInfo{
		Category:    CategoryData,
		Keywords:    []string{"SELECT", "INSERT", "UPDATE", "CREATE", "TABLE"},
		Importance:  0.7,
		Description: "SQL file",
	})
}

func (a *EnhancedProjectScanAnalyzer) registerFileType(pattern string, info FileTypeInfo) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.fileTypeRegistry[pattern] = info
}

func (a *EnhancedProjectScanAnalyzer) categorizeFile(filePath string) (FileCategory, FileTypeInfo) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	fileName := filepath.Base(filePath)
	ext := filepath.Ext(filePath)
	if ext != "" {
		ext = ext[1:] // Remove the dot
	}

	// Check for exact filename matches first (e.g., Makefile, Dockerfile)
	if info, exists := a.fileTypeRegistry[fileName]; exists {
		return info.Category, info
	}

	// Check for pattern matches (e.g., _test.go, test.js)
	for pattern, info := range a.fileTypeRegistry {
		if strings.HasSuffix(fileName, pattern) {
			return info.Category, info
		}
	}

	// Check by extension
	if info, exists := a.fileTypeRegistry[ext]; exists {
		return info.Category, info
	}

	// Default to "other" category
	return CategoryOther, FileTypeInfo{
		Category:    CategoryOther,
		Importance:  0.5,
		Description: "Other file type",
	}
}

// ================================
// Analysis Cache
// ================================

// AnalysisCache provides LRU caching for analysis results
type AnalysisCache struct {
	mu      sync.RWMutex
	cache   map[string]*CachedAnalysis
	maxSize int
	ttl     time.Duration
}

// CachedAnalysis represents a cached analysis result
type CachedAnalysis struct {
	Analysis  *FileAnalysis
	Timestamp time.Time
}

// NewAnalysisCache creates a new analysis cache
func NewAnalysisCache(maxSize int, ttl time.Duration) *AnalysisCache {
	return &AnalysisCache{
		cache:   make(map[string]*CachedAnalysis),
		maxSize: maxSize,
		ttl:     ttl,
	}
}

func (c *AnalysisCache) Get(key string) (*FileAnalysis, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if cached, exists := c.cache[key]; exists {
		if time.Since(cached.Timestamp) < c.ttl {
			return cached.Analysis, true
		}
	}
	return nil, false
}

func (c *AnalysisCache) Set(key string, analysis *FileAnalysis) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Simple eviction: remove oldest entry if at capacity
	if len(c.cache) >= c.maxSize {
		var oldestKey string
		var oldestTime time.Time
		for k, v := range c.cache {
			if oldestKey == "" || v.Timestamp.Before(oldestTime) {
				oldestKey = k
				oldestTime = v.Timestamp
			}
		}
		delete(c.cache, oldestKey)
	}

	c.cache[key] = &CachedAnalysis{
		Analysis:  analysis,
		Timestamp: time.Now(),
	}
}

// ================================
// Error Handling
// ================================

// ErrorHandler provides resilient error handling with circuit breaker pattern
type ErrorHandler struct {
	maxRetries       int
	retryDelay       time.Duration
	fallbackAnalyzer FileAnalyzer
	circuitBreaker   *CircuitBreaker
}

// NewErrorHandler creates a new error handler
func NewErrorHandler(maxRetries int, retryDelay time.Duration, fallbackAnalyzer FileAnalyzer) *ErrorHandler {
	return &ErrorHandler{
		maxRetries:       maxRetries,
		retryDelay:       retryDelay,
		fallbackAnalyzer: fallbackAnalyzer,
		circuitBreaker:   NewCircuitBreaker(5, 1*time.Minute),
	}
}

// HandleAnalysisError handles errors during file analysis with retry logic.
// IMPORTANT: the first attempt is made immediately with NO delay. Backoff
// sleep only applies on genuine failures between retry attempts.
func (h *ErrorHandler) HandleAnalysisError(ctx context.Context, filePath string, content string, userQuery string, analyzer FileAnalyzer, err error) (*FileAnalysis, error) {
	// Check circuit breaker
	if !h.circuitBreaker.Allow() {
		return h.fallbackAnalyzer.AnalyzeFile(ctx, filePath, content, userQuery)
	}

	// ── First attempt: call immediately, no delay ────────────────────────────
	analysisCtx, cancel := context.WithTimeout(ctx, 45*time.Second)
	result, firstErr := analyzer.AnalyzeFile(analysisCtx, filePath, content, userQuery)
	cancel()
	if firstErr == nil {
		h.circuitBreaker.Success()
		return result, nil
	}

	// ── Retries with backoff — only reached on genuine failure ───────────────
	for attempt := 1; attempt <= h.maxRetries; attempt++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(h.retryDelay * time.Duration(attempt)):
			// exponential backoff between retries
		}

		analysisCtx, cancel = context.WithTimeout(ctx, 45*time.Second)
		result, retryErr := analyzer.AnalyzeFile(analysisCtx, filePath, content, userQuery)
		cancel()
		if retryErr == nil {
			h.circuitBreaker.Success()
			return result, nil
		}

		if logger, ok := analyzer.(*LLMFileAnalyzer); ok && logger.logger != nil {
			logger.logger.Warn("Retry %d/%d failed for %s: %v", attempt, h.maxRetries, filePath, retryErr)
		}
	}

	h.circuitBreaker.Failure()
	return h.fallbackAnalyzer.AnalyzeFile(ctx, filePath, content, userQuery)
}

// CircuitBreaker implements a simple circuit breaker pattern
type CircuitBreaker struct {
	mu           sync.Mutex
	failureCount int
	threshold    int
	lastFailTime time.Time
	cooldown     time.Duration
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(threshold int, cooldown time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		threshold: threshold,
		cooldown:  cooldown,
	}
}

func (cb *CircuitBreaker) Allow() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	// Reset if cooldown period has passed
	if time.Since(cb.lastFailTime) > cb.cooldown {
		cb.failureCount = 0
	}

	return cb.failureCount < cb.threshold
}

func (cb *CircuitBreaker) Success() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.failureCount = 0
}

func (cb *CircuitBreaker) Failure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.failureCount++
	cb.lastFailTime = time.Now()
}

// ================================
// Progress Reporting Adapter
// ================================

// SimpleProgressReporter provides a simple progress reporting interface
type SimpleProgressReporter interface {
	StartAnalysis(totalFiles int)
	UpdateFile(fileName string, current, total int)
	FileCompleted(fileName string, success bool, relevance float64)
	AnalysisComplete(duration time.Duration, successful, failed int)
}

// NullSimpleProgressReporter is a no-op implementation
type NullSimpleProgressReporter struct{}

func (r *NullSimpleProgressReporter) StartAnalysis(totalFiles int)                   {}
func (r *NullSimpleProgressReporter) UpdateFile(fileName string, current, total int) {}
func (r *NullSimpleProgressReporter) FileCompleted(fileName string, success bool, relevance float64) {
}
func (r *NullSimpleProgressReporter) AnalysisComplete(duration time.Duration, successful, failed int) {
}

// ================================
// Analysis Results
// ================================

// EnhancedProjectScanResult includes categorized results
type EnhancedProjectScanResult struct {
	*ProjectAnalysisResult
	FileCategories map[string]FileCategory         `json:"file_categories"`
	CategoryStats  map[FileCategory]*CategoryStats `json:"category_stats"`
	Timeline       []TimelineEvent                 `json:"timeline,omitempty"`
}

// CategoryStats provides statistics for each file category
type CategoryStats struct {
	FileCount        int     `json:"file_count"`
	RelevantCount    int     `json:"relevant_count"`
	AvgRelevance     float64 `json:"avg_relevance"`
	TotalIssues      int     `json:"total_issues"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// TimelineEvent represents an event during analysis
type TimelineEvent struct {
	Timestamp  time.Time `json:"timestamp"`
	Event      string    `json:"event"`
	FilePath   string    `json:"file_path,omitempty"`
	Category   string    `json:"category,omitempty"`
	Success    bool      `json:"success"`
	DurationMs int64     `json:"duration_ms,omitempty"`
	ErrorMsg   string    `json:"error,omitempty"`
}

// ================================
// Analysis Metrics
// ================================

// AnalysisMetrics tracks performance metrics
type AnalysisMetrics struct {
	mu               sync.Mutex
	fileMetrics      map[string]*FileMetrics
	categoryMetrics  map[FileCategory]*CategoryMetrics
	overallStartTime time.Time
	overallEndTime   time.Time
}

// FileMetrics tracks metrics for individual files
type FileMetrics struct {
	StartTime        time.Time
	EndTime          time.Time
	ReadDuration     time.Duration
	AnalysisDuration time.Duration
	FileSize         int64
	Relevance        float64
	IssueCount       int
}

// CategoryMetrics tracks metrics by file category
type CategoryMetrics struct {
	FileCount       int
	TotalSize       int64
	AvgRelevance    float64
	TotalIssues     int
	AvgAnalysisTime time.Duration
}

// ================================
// Main Project Scan Analyzer
// ================================

// ProjectScanAnalyzer is an enhanced version with all improvements
type ProjectScanAnalyzer struct {
	*EnhancedProjectScanAnalyzer
	errorHandler       *ErrorHandler
	progressReporter   SimpleProgressReporter
	analysisMetrics    *AnalysisMetrics
	toolExecutionCache sync.Map // map[string]*EnhancedProjectScanResult

	// printFunc is called for all progress output. Override via SetPrintFunc to
	// coordinate with terminal UI (e.g. hide/show spinner around each line).
	printFunc     func(format string, args ...interface{})
	statusUpdater func(status string)
}

// SetPrintFunc replaces the progress output function used by the analyzer.
func (a *ProjectScanAnalyzer) SetPrintFunc(fn func(format string, args ...interface{})) {
	if fn != nil {
		a.printFunc = fn
	}
}

// SetStatusUpdater configures a callback to natively update a UI spinner label.
func (a *ProjectScanAnalyzer) SetStatusUpdater(fn func(status string)) {
	if fn != nil {
		a.statusUpdater = fn
	}
}

// NewProjectScanAnalyzer creates the enhanced analyzer
func NewProjectScanAnalyzer(llmClient *llm.Client, provider, model string, logger *logger.Logger) *ProjectScanAnalyzer {
	// Create base LLM analyzer
	llmAnalyzer := NewLLMFileAnalyzer(llmClient, provider, model, logger)

	// Create enhanced analyzer with categorization
	enhancedAnalyzer := NewEnhancedProjectScanAnalyzer(llmAnalyzer)

	// Create error handler with fallback
	errorHandler := NewErrorHandler(3, 1*time.Second, NewDefaultFileAnalyzer())

	return &ProjectScanAnalyzer{
		EnhancedProjectScanAnalyzer: enhancedAnalyzer,
		errorHandler:                errorHandler,
		progressReporter:            &NullSimpleProgressReporter{},
		analysisMetrics: &AnalysisMetrics{
			fileMetrics:     make(map[string]*FileMetrics),
			categoryMetrics: make(map[FileCategory]*CategoryMetrics),
		},
		printFunc: func(format string, args ...interface{}) {
			fmt.Fprintf(os.Stderr, format, args...)
		},
	}
}

// Name returns the tool name
func (a *ProjectScanAnalyzer) Name() string {
	return "projectScanAnalyzer"
}

// Description returns the tool description
func (a *ProjectScanAnalyzer) Description() string {
	return "Enhanced project scanner with file categorization, custom analyzers, progress reporting, error recovery, and caching. Analyzes each file individually with context awareness for better insights."
}

// ParameterSchema returns the JSON schema for this tool's parameters
func (a *ProjectScanAnalyzer) ParameterSchema() JSONSchema {
	// Use the same schema as the base analyzer with additional parameters
	baseSchema := a.EnhancedProjectScanAnalyzer.ParameterSchema()

	// Add enhanced parameters
	baseSchema.Properties["showDetails"] = JSONSchema{
		Type:        "boolean",
		Description: "Show detailed progress information for each file (default: true)",
		Default:     true,
	}

	baseSchema.Properties["specificDirs"] = JSONSchema{
		Type: "array",
		Items: &JSONSchema{
			Type: "string",
		},
		Description: "Specific directories to search in, relative to the base dir (e.g., ['internal/', 'cmd/', 'pkg/']). If provided, only these directories will be scanned instead of the entire project. Useful for focusing analysis on specific parts of large codebases.",
	}

	baseSchema.Properties["onlyInSpecificDirs"] = JSONSchema{
		Type:        "boolean",
		Description: "When true and specificDirs is provided, only scan files directly in those directories (no subdirectories). Useful when you want to analyze only the top-level files in specific folders. When false, scan all subdirectories within specificDirs. Default: false",
		Default:     false,
	}

	// Override timeout to 45 seconds
	baseSchema.Properties["analysisTimeout"] = JSONSchema{
		Type:        "integer",
		Description: "Timeout per file analysis in seconds (default: 45)",
		Default:     45,
	}

	// Update concurrency description for sequential processing
	baseSchema.Properties["concurrency"] = JSONSchema{
		Type:        "integer",
		Description: "Number of files to analyze concurrently (default: 5)",
		Default:     5,
	}

	baseSchema.Properties["maxAnalysisFiles"] = JSONSchema{
		Type:        "integer",
		Description: "Maximum number of files to send to LLM analysis after local pre-filtering (default: 25). Lower values are faster; raise for broader coverage on large repos.",
		Default:     25,
	}

	return baseSchema
}

// getCacheKey generates a hash for tool parameters
func (a *ProjectScanAnalyzer) getCacheKey(params map[string]interface{}) string {
	b, _ := json.Marshal(params)
	hash := sha256.Sum256(b)
	return hex.EncodeToString(hash[:])
}

// scoreAndRankFiles replaces the old binary pre-filter with a two-pass ranked scorer.
//
// Pass 1 — path scoring (zero I/O, runs in microseconds):
//
//	Every component of the file path (dir names + basename without extension) is
//	scored against the query tokens. Weights:
//	  +10  exact base-name match  (e.g. query token "router" matches "router.go")
//	  +3   directory-component match
//	  +1   substring anywhere in the path
//
// Pass 2 — streaming content scoring (I/O only for files that scored 0 in pass 1):
//
//	Reads lines via bufio.Scanner. Stops after maxContentHits token hits to avoid
//	reading huge files entirely. Binary files are skipped entirely (null byte in
//	first 512 bytes). Each content token hit adds +1 to the score.
//
// Files scoring 0 across both passes are dropped.
// The result is sorted descending by score, so the maxAnalysisFiles cap (Fix 3)
// always passes the most-relevant files to the LLM, not just the first N alphabetically.
func (a *ProjectScanAnalyzer) scoreAndRankFiles(files []string, userQuery string) []string {
	queryTokens := extractKeywords(userQuery)
	if len(queryTokens) == 0 {
		// No extractable tokens — can't score; pass everything through unchanged.
		return files
	}

	const maxContentHits = 3 // stop scanning content after this many hits per file

	type scored struct {
		path  string
		score int
	}

	results := make([]scored, 0, len(files))

	for _, filePath := range files {
		score := 0

		// ── Pass 1: path scoring (zero I/O) ────────────────────────────────
		base := strings.ToLower(strings.TrimSuffix(filepath.Base(filePath), filepath.Ext(filePath)))
		pathLower := strings.ToLower(filePath)

		for _, token := range queryTokens {
			// Exact basename match is the strongest signal
			if base == token {
				score += 10
				continue
			}
			// Token appears in basename
			if strings.Contains(base, token) {
				score += 5
				continue
			}
			// Token appears somewhere in the directory path
			if strings.Contains(pathLower, token) {
				score += 3
			}
		}

		// ── Pass 2: streaming content scoring (only for zero-score files) ───────
		if score == 0 {
			f, err := os.Open(filePath)
			if err != nil {
				// Can't read: include with score 0 so the LLM can attempt it
				results = append(results, scored{filePath, 0})
				continue
			}

			// Check for binary file: read first 512 bytes
			header := make([]byte, 512)
			n, _ := f.Read(header)
			if isBinaryContent(header[:n]) {
				f.Close()
				continue // skip binary files
			}

			// Seek back to start and stream lines
			_, _ = f.Seek(0, 0)
			scanner := bufio.NewScanner(f)
			hits := 0
			for scanner.Scan() && hits < maxContentHits {
				lineLower := strings.ToLower(scanner.Text())
				for _, token := range queryTokens {
					if strings.Contains(lineLower, token) {
						score++
						hits++
						break // one token match per line is enough
					}
				}
			}
			f.Close()
		}

		if score > 0 {
			results = append(results, scored{filePath, score})
		}
	}

	if len(results) == 0 {
		return nil
	}

	// Sort descending by score so the hard cap picks the best files
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	paths := make([]string, len(results))
	for i, r := range results {
		paths[i] = r.path
	}

	dropped := len(files) - len(paths)
	if dropped > 0 {
		a.printFunc("🎯 Ranked pre-filter: kept %d/%d files (dropped %d zero-score)\n", len(paths), len(files), dropped)
	}
	return paths
}

// isBinaryContent returns true if the byte slice contains a null byte,
// which is a reliable heuristic for binary (non-text) files.
func isBinaryContent(b []byte) bool {
	for _, c := range b {
		if c == 0 {
			return true
		}
	}
	return false
}

func extractKeywords(query string) []string {
	// Generic English stopwords + instruction verbs that carry no content signal.
	// NOTE: "code", "project", "file" are intentionally NOT in this list —
	// they are meaningful domain tokens in a coding assistant context.
	stopwords := map[string]bool{
		// Articles / conjunctions / prepositions
		"a": true, "an": true, "the": true, "and": true, "or": true, "but": true,
		"in": true, "on": true, "at": true, "to": true, "from": true, "with": true,
		"of": true, "by": true, "as": true, "into": true, "via": true,
		// Auxiliary verbs
		"is": true, "are": true, "was": true, "were": true, "be": true, "been": true,
		"do": true, "does": true, "did": true, "have": true, "has": true, "had": true,
		"can": true, "could": true, "will": true, "would": true, "should": true, "may": true,
		// Question words
		"how": true, "what": true, "where": true, "when": true, "why": true, "who": true,
		// Instruction verbs — tell the assistant what to do, not what to find
		"show": true, "find": true, "search": true, "look": true, "get": true,
		"give": true, "tell": true, "explain": true, "list": true, "describe": true,
		"print": true, "display": true, "output": true, "return": true,
		// Filler pronouns / determiners
		"this": true, "that": true, "these": true, "those": true, "all": true,
		"any": true, "some": true, "each": true, "every": true, "which": true,
		"there": true, "here": true, "then": true, "just": true, "also": true,
		"not": true, "its": true, "their": true,
	}

	query = strings.ToLower(query)
	for _, p := range []string{",", ".", "!", "?", ";", ":", `"`, "'", "(", ")", "[", "]", "{", "}"} {
		query = strings.ReplaceAll(query, p, " ")
	}

	words := strings.Fields(query)
	var keywords []string
	seen := make(map[string]bool)
	for _, w := range words {
		if !stopwords[w] && len(w) > 2 && !seen[w] {
			seen[w] = true
			keywords = append(keywords, w)
		}
	}
	return keywords
}

// Execute performs the enhanced file-by-file analysis
func (a *ProjectScanAnalyzer) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	cacheKey := a.getCacheKey(params)
	if cached, ok := a.toolExecutionCache.Load(cacheKey); ok {
		a.printFunc("🎯 Using cached project scan results for identical parameters\n")
		return cached, nil
	}

	a.analysisMetrics.overallStartTime = time.Now()
	defer func() {
		a.analysisMetrics.overallEndTime = time.Now()
	}()

	// Get parameters
	dir, _ := params["dir"].(string)
	if dir == "" {
		var err error
		dir, err = os.Getwd()
		if err != nil {
			return nil, fmt.Errorf("failed to get current directory: %w", err)
		}
	}

	userQuery, ok := params["userQuery"].(string)
	if !ok || userQuery == "" {
		return nil, fmt.Errorf("userQuery is required")
	}

	// Setup progress reporter
	enableProgress := getBoolParam(params, "showProgress", true)
	showDetails := getBoolParam(params, "showDetails", true)

	if enableProgress {
		if a.statusUpdater != nil {
			a.progressReporter = &SpinnerProgressReporter{
				updateStatus: a.statusUpdater,
			}
		} else {
			enhancedReporter := NewEnhancedProgressReporter(
				a.printFunc,
				showDetails,
			)
			a.progressReporter = enhancedReporter
			defer enhancedReporter.PrintSummary()
		}
	} else {
		a.progressReporter = &NullSimpleProgressReporter{}
	}

	// Validate directory
	dir, err := ValidateAndCleanPath(dir)
	if err != nil {
		return nil, &ErrToolExecution{
			ToolName: a.Name(),
			Message:  "invalid directory path",
			Err:      err,
		}
	}

	// Get analysis parameters
	pattern, _ := params["pattern"].(string)
	includeHidden := getBoolParam(params, "includeHidden", false)
	maxDepth := getIntParam(params, "maxDepth", 0)
	maxFileSize := getIntParam(params, "maxFileSize", 1024*1024) // 1MB default
	relevanceThreshold := getFloatParam(params, "relevanceThreshold", 0.3)
	timeout := time.Duration(getIntParam(params, "analysisTimeout", 45)) * time.Second
	concurrency := getIntParam(params, "concurrency", 5)

	// Get exclude patterns
	excludePatterns := getDefaultExcludePatterns()
	if customExcludes, ok := params["excludePatterns"].([]string); ok {
		excludePatterns = append(excludePatterns, customExcludes...)
	}

	// Get specific directories to search
	var specificDirs []string
	if dirs, ok := params["specificDirs"].([]interface{}); ok {
		for _, d := range dirs {
			if dirStr, ok := d.(string); ok {
				specificDirs = append(specificDirs, dirStr)
			}
		}
	}

	// Get whether to only scan files in specific directories (no subdirectories)
	onlyInSpecificDirs := getBoolParam(params, "onlyInSpecificDirs", false)

	// Scan for files
	if len(specificDirs) > 0 {
		if onlyInSpecificDirs {
			a.printFunc("🎯 Scanning only files in specific directories (no subdirectories): %v\n", specificDirs)
		} else {
			a.printFunc("🎯 Scanning specific directories (including subdirectories): %v\n", specificDirs)
		}
	}
	files, err := a.scanFiles(dir, pattern, excludePatterns, includeHidden, maxDepth, specificDirs, onlyInSpecificDirs)
	if err != nil {
		return nil, &ErrToolExecution{
			ToolName: a.Name(),
			Message:  "failed to scan directory",
			Err:      err,
		}
	}

	// RANKED PRE-FILTER: score files by path + content relevance, drop zero-score.
	// Returns files sorted by relevance descending so the cap below picks the best.
	files = a.scoreAndRankFiles(files, userQuery)

	// HARD CAP: limit LLM queue to prevent runaway scan times on large repos.
	// At 5 workers and ~4s/file, 25 files = ~20s wall clock.
	maxFiles := getIntParam(params, "maxAnalysisFiles", 25)
	if maxFiles > 0 && len(files) > maxFiles {
		a.printFunc("🎯 Capping LLM analysis to top %d files (from %d candidates after pre-filter)\n", maxFiles, len(files))
		files = files[:maxFiles]
	}

	if len(files) == 0 {
		return &EnhancedProjectScanResult{
			ProjectAnalysisResult: &ProjectAnalysisResult{
				Summary:     "No files found matching the criteria",
				TotalFiles:  0,
				FileResults: []FileResult{},
			},
			FileCategories: make(map[string]FileCategory),
			CategoryStats:  make(map[FileCategory]*CategoryStats),
		}, nil
	}

	// Sort files by path for consistent ordering
	sort.Strings(files)

	// Categorize files
	fileCategories := make(map[string]FileCategory)
	for _, filePath := range files {
		category, _ := a.categorizeFile(filePath)
		fileCategories[filePath] = category
	}

	// Initialize result
	result := &EnhancedProjectScanResult{
		ProjectAnalysisResult: &ProjectAnalysisResult{
			Directory:   dir,
			UserQuery:   userQuery,
			TotalFiles:  len(files),
			FileResults: []FileResult{},
		},
		FileCategories: fileCategories,
		CategoryStats:  make(map[FileCategory]*CategoryStats),
		Timeline:       []TimelineEvent{},
	}

	// Initialize category stats
	for cat := range a.categoryAnalyzers {
		result.CategoryStats[cat] = &CategoryStats{}
	}

	// Start progress reporting
	a.progressReporter.StartAnalysis(len(files))

	// Analyze files concurrently
	err = a.analyzeFilesConcurrently(ctx, files, fileCategories, userQuery, relevanceThreshold, timeout, concurrency, result)
	if err != nil {
		return nil, err
	}

	// Generate summary
	a.generateEnhancedSummary(result, fileCategories, maxFileSize)

	// Report completion
	duration := time.Since(a.analysisMetrics.overallStartTime)
	successful := result.AnalyzedFiles
	failed := result.SkippedFiles
	a.progressReporter.AnalysisComplete(duration, successful, failed)

	// Save to cache
	a.toolExecutionCache.Store(cacheKey, result)

	return result, nil
}

// analyzeFilesConcurrently performs concurrent analysis of files
func (a *ProjectScanAnalyzer) analyzeFilesConcurrently(ctx context.Context, files []string,
	fileCategories map[string]FileCategory, userQuery string,
	relevanceThreshold float64, timeout time.Duration, concurrency int, result *EnhancedProjectScanResult) error {

	if concurrency < 1 {
		concurrency = 1
	}

	var mu sync.Mutex
	sem := make(chan struct{}, concurrency)
	errCh := make(chan error, 1)

	var wg sync.WaitGroup
	for idx, filePath := range files {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-errCh:
			return err
		case sem <- struct{}{}:
		}

		wg.Add(1)
		go func(i int, path string) {
			defer wg.Done()
			defer func() { <-sem }()

			// Update progress
			a.progressReporter.UpdateFile(filepath.Base(path), i+1, len(files))

			// Create timeout context for this file
			fileCtx, cancel := context.WithTimeout(ctx, timeout)

			// Analyze the file
			fileResult, timelineEvent := a.analyzeFileWithMetrics(fileCtx, path, fileCategories[path], userQuery)

			cancel() // Clean up the context

			mu.Lock()
			defer mu.Unlock()

			// Record timeline event
			result.Timeline = append(result.Timeline, timelineEvent)

			// Update category stats
			category := fileCategories[path]
			stats := result.CategoryStats[category]
			stats.FileCount++

			if fileResult.Error == "" {
				result.AnalyzedFiles++

				// Only include results above threshold
				if fileResult.Analysis.Relevance >= relevanceThreshold {
					result.FileResults = append(result.FileResults, fileResult)
					stats.RelevantCount++
				}

				// Update stats
				stats.AvgRelevance = (stats.AvgRelevance*float64(stats.FileCount-1) + fileResult.Analysis.Relevance) / float64(stats.FileCount)
				stats.TotalIssues += len(fileResult.Analysis.Issues)
				stats.ProcessingTimeMs += timelineEvent.DurationMs

				// Report completion
				a.progressReporter.FileCompleted(filepath.Base(path), true, fileResult.Analysis.Relevance)
			} else {
				result.SkippedFiles++
				result.Errors = append(result.Errors, fmt.Sprintf("%s: %s", path, fileResult.Error))

				// Report failure
				a.progressReporter.FileCompleted(filepath.Base(path), false, 0)
			}
		}(idx, filePath)
	}

	wg.Wait()

	select {
	case err := <-errCh:
		return err
	default:
	}

	return nil
}

// analyzeFileWithMetrics analyzes a single file and records metrics
func (a *ProjectScanAnalyzer) analyzeFileWithMetrics(ctx context.Context, filePath string,
	category FileCategory, userQuery string) (FileResult, TimelineEvent) {

	startTime := time.Now()
	event := TimelineEvent{
		Timestamp: startTime,
		Event:     "file_analysis",
		FilePath:  filePath,
		Category:  string(category),
		Success:   false,
	}

	// Initialize metrics for this file
	metrics := &FileMetrics{
		StartTime: startTime,
	}

	// Check cache first
	cacheKey := fmt.Sprintf("%s:%s", filePath, userQuery)
	if cached, found := a.analysisCache.Get(cacheKey); found {
		event.Success = true
		event.DurationMs = 0 // Cached result
		return FileResult{
			Path:     filePath,
			Analysis: *cached,
		}, event
	}

	// Read file content
	readStart := time.Now()
	content, err := os.ReadFile(filePath)
	if err != nil {
		event.ErrorMsg = fmt.Sprintf("failed to read: %v", err)
		event.DurationMs = time.Since(startTime).Milliseconds()
		return FileResult{
			Path:  filePath,
			Error: event.ErrorMsg,
		}, event
	}
	metrics.ReadDuration = time.Since(readStart)
	metrics.FileSize = int64(len(content))

	// Get appropriate analyzer for this category
	analyzer, exists := a.categoryAnalyzers[category]
	if !exists {
		analyzer = a.categoryAnalyzers[CategoryOther]
	}

	// Analyze with error handling
	analysisStart := time.Now()
	analysis, err := a.errorHandler.HandleAnalysisError(ctx, filePath, string(content), userQuery, analyzer, err)
	if err != nil {
		event.ErrorMsg = fmt.Sprintf("analysis failed: %v", err)
		event.DurationMs = time.Since(startTime).Milliseconds()
		return FileResult{
			Path:  filePath,
			Error: event.ErrorMsg,
		}, event
	}
	metrics.AnalysisDuration = time.Since(analysisStart)
	metrics.Relevance = analysis.Relevance
	metrics.IssueCount = len(analysis.Issues)

	// Cache the result
	a.analysisCache.Set(cacheKey, analysis)

	// Update metrics
	metrics.EndTime = time.Now()
	a.updateMetrics(filePath, category, metrics)

	event.Success = true
	event.DurationMs = time.Since(startTime).Milliseconds()

	return FileResult{
		Path:     filePath,
		Analysis: *analysis,
	}, event
}

// updateMetrics updates the analysis metrics
func (a *ProjectScanAnalyzer) updateMetrics(filePath string, category FileCategory, metrics *FileMetrics) {
	a.analysisMetrics.mu.Lock()
	defer a.analysisMetrics.mu.Unlock()

	// Store file metrics
	a.analysisMetrics.fileMetrics[filePath] = metrics

	// Update category metrics
	catMetrics, exists := a.analysisMetrics.categoryMetrics[category]
	if !exists {
		catMetrics = &CategoryMetrics{}
		a.analysisMetrics.categoryMetrics[category] = catMetrics
	}

	catMetrics.FileCount++
	catMetrics.TotalSize += metrics.FileSize
	catMetrics.AvgRelevance = (catMetrics.AvgRelevance*float64(catMetrics.FileCount-1) + metrics.Relevance) / float64(catMetrics.FileCount)
	catMetrics.TotalIssues += metrics.IssueCount

	// Update average analysis time
	totalTime := catMetrics.AvgAnalysisTime * time.Duration(catMetrics.FileCount-1)
	catMetrics.AvgAnalysisTime = (totalTime + metrics.AnalysisDuration) / time.Duration(catMetrics.FileCount)
}

// ================================
// Base Types and Structures
// ================================

// ProjectScanAnalyzerTool is the base analyzer struct
type ProjectScanAnalyzerTool struct {
	analyzer         FileAnalyzer
	progressReporter SimpleProgressReporter
}

// NewProjectScanAnalyzerTool creates a new base analyzer
func NewProjectScanAnalyzerTool() *ProjectScanAnalyzerTool {
	return &ProjectScanAnalyzerTool{
		analyzer:         NewDefaultFileAnalyzer(),
		progressReporter: &NullSimpleProgressReporter{},
	}
}

// ParameterSchema returns the JSON schema for base analyzer
func (t *ProjectScanAnalyzerTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"dir": {
				Type:        "string",
				Description: "Directory path to scan (defaults to current directory if empty)",
			},
			"specificDirs": {
				Type: "array",
				Items: &JSONSchema{
					Type: "string",
				},
				Description: "Specific directories to search in, relative to the base dir (e.g., ['internal/', 'cmd/', 'pkg/']). If provided, only these directories will be scanned instead of the entire project. Useful for focusing analysis on specific parts of large codebases.",
			},
			"onlyInSpecificDirs": {
				Type:        "boolean",
				Description: "When true and specificDirs is provided, only scan files directly in those directories (no subdirectories). Useful when you want to analyze only the top-level files in specific folders. When false, scan all subdirectories within specificDirs. Default: false",
				Default:     false,
			},
			"userQuery": {
				Type:        "string",
				Description: "The user's query or analysis criteria to apply to each file",
			},
			"pattern": {
				Type:        "string",
				Description: "Optional glob pattern to include only specific files (e.g., '*.go', '*.{js,ts}')",
			},
			"maxDepth": {
				Type:        "integer",
				Description: "Maximum recursion depth (0 for unlimited)",
				Default:     0,
			},
			"includeHidden": {
				Type:        "boolean",
				Description: "Whether to include hidden files and directories",
				Default:     false,
			},
			"maxFileSize": {
				Type:        "integer",
				Description: "Maximum size per file to analyze in bytes (default: 1MB)",
				Default:     1024 * 1024,
			},
			"excludePatterns": {
				Type: "array",
				Items: &JSONSchema{
					Type: "string",
				},
				Description: "Additional glob patterns to exclude beyond defaults",
			},
			"concurrency": {
				Type:        "integer",
				Description: "Number of files to analyze concurrently (default: 5)",
				Default:     5,
			},
			"relevanceThreshold": {
				Type:        "number",
				Description: "Minimum relevance score (0-1) to include in results (default: 0.3)",
				Default:     0.3,
			},
			"analysisTimeout": {
				Type:        "integer",
				Description: "Timeout per file analysis in seconds (default: 30)",
				Default:     30,
			},
			"showProgress": {
				Type:        "boolean",
				Description: "Show progress for each file being analyzed (default: true)",
				Default:     true,
			},
		},
		Required: []string{"userQuery"},
	}
}

// ProjectAnalysisResult represents the basic scan result
type ProjectAnalysisResult struct {
	Directory     string       `json:"directory"`
	UserQuery     string       `json:"user_query"`
	TotalFiles    int          `json:"total_files"`
	AnalyzedFiles int          `json:"analyzed_files"`
	SkippedFiles  int          `json:"skipped_files"`
	FileResults   []FileResult `json:"file_results"`
	Summary       string       `json:"summary"`
	Errors        []string     `json:"errors,omitempty"`
}

// FileResult represents a single file analysis result
type FileResult struct {
	Path     string       `json:"path"`
	Analysis FileAnalysis `json:"analysis,omitempty"`
	Error    string       `json:"error,omitempty"`
}

// scanFiles scans the directory for files matching criteria
func (a *ProjectScanAnalyzer) scanFiles(dir string, pattern string, excludePatterns []string,
	includeHidden bool, maxDepth int, specificDirs []string, onlyInSpecificDirs bool) ([]string, error) {

	var files []string

	// If specific directories are provided, scan only those
	if len(specificDirs) > 0 {
		for _, specificDir := range specificDirs {
			// Make path relative to base dir
			targetDir := filepath.Join(dir, specificDir)

			// Check if directory exists
			if info, err := os.Stat(targetDir); err != nil || !info.IsDir() {
				continue // Skip non-existent or non-directory paths
			}

			if onlyInSpecificDirs {
				// Only scan files directly in this directory, no subdirectories
				entries, err := os.ReadDir(targetDir)
				if err != nil {
					continue
				}

				for _, entry := range entries {
					if entry.IsDir() {
						continue // Skip subdirectories
					}

					filePath := filepath.Join(targetDir, entry.Name())
					info, err := entry.Info()
					if err != nil {
						continue
					}

					// Process the file (apply filters)
					if a.shouldIncludeFile(filePath, info, targetDir, pattern, excludePatterns, includeHidden) {
						files = append(files, filePath)
					}
				}
			} else {
				// Scan this specific directory and its subdirectories
				err := filepath.Walk(targetDir, func(path string, info os.FileInfo, err error) error {
					return a.processFile(path, info, err, dir, pattern, excludePatterns, includeHidden, maxDepth, &files)
				})

				if err != nil {
					return nil, err
				}
			}
		}
		return files, nil
	}

	// Default behavior: scan entire directory
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		return a.processFile(path, info, err, dir, pattern, excludePatterns, includeHidden, maxDepth, &files)
	})

	return files, err
}

// processFile handles the logic for processing individual files during directory traversal
func (a *ProjectScanAnalyzer) processFile(path string, info os.FileInfo, err error, baseDir string,
	pattern string, excludePatterns []string, includeHidden bool, maxDepth int, files *[]string) error {
	if err != nil {
		return nil // Skip files we can't access
	}

	// Calculate depth
	relPath, _ := filepath.Rel(baseDir, path)
	depth := strings.Count(relPath, string(filepath.Separator))

	// Check max depth
	if maxDepth > 0 && depth > maxDepth {
		if info.IsDir() {
			return filepath.SkipDir
		}
		return nil
	}

	// Skip hidden files/dirs if requested
	if !includeHidden && strings.HasPrefix(filepath.Base(path), ".") {
		if info.IsDir() {
			return filepath.SkipDir
		}
		return nil
	}

	// Skip excluded patterns
	if matchesAnyPattern(path, excludePatterns) {
		if info.IsDir() {
			return filepath.SkipDir
		}
		return nil
	}

	// Skip directories
	if info.IsDir() {
		return nil
	}

	// Apply pattern filter
	if pattern != "" {
		matched, _ := filepath.Match(pattern, filepath.Base(path))
		if !matched {
			return nil
		}
	}

	*files = append(*files, path)
	return nil
}

// shouldIncludeFile checks if a file should be included based on filters
func (a *ProjectScanAnalyzer) shouldIncludeFile(filePath string, info os.FileInfo, baseDir string,
	pattern string, excludePatterns []string, includeHidden bool) bool {

	// Skip directories
	if info.IsDir() {
		return false
	}

	// Skip hidden files if requested
	if !includeHidden && strings.HasPrefix(filepath.Base(filePath), ".") {
		return false
	}

	// Skip excluded patterns
	if matchesAnyPattern(filePath, excludePatterns) {
		return false
	}

	// Apply pattern filter
	if pattern != "" {
		matched, _ := filepath.Match(pattern, filepath.Base(filePath))
		if !matched {
			return false
		}
	}

	return true
}

// generateEnhancedSummary creates an enhanced summary
func (a *ProjectScanAnalyzer) generateEnhancedSummary(result *EnhancedProjectScanResult,
	fileCategories map[string]FileCategory, maxFileSize int) {

	// Calculate overall metrics
	totalRelevantFiles := 0
	totalIssues := 0
	categoryBreakdown := make(map[FileCategory]int)

	for _, stats := range result.CategoryStats {
		totalRelevantFiles += stats.RelevantCount
		totalIssues += stats.TotalIssues
	}

	for _, category := range fileCategories {
		categoryBreakdown[category]++
	}

	// Build enhanced summary
	var summaryParts []string

	summaryParts = append(summaryParts, fmt.Sprintf("Analyzed %d files in %s",
		result.AnalyzedFiles, result.Directory))

	if totalRelevantFiles > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("Found %d relevant files", totalRelevantFiles))
	}

	// Add category breakdown
	if len(categoryBreakdown) > 1 {
		catParts := []string{}
		for cat, count := range categoryBreakdown {
			if count > 0 {
				catParts = append(catParts, fmt.Sprintf("%s: %d", cat, count))
			}
		}
		summaryParts = append(summaryParts, fmt.Sprintf("Categories: %s", strings.Join(catParts, ", ")))
	}

	if totalIssues > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("Identified %d total issues", totalIssues))
	}

	if result.SkippedFiles > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("Skipped %d files", result.SkippedFiles))
	}

	// Add performance summary
	duration := a.analysisMetrics.overallEndTime.Sub(a.analysisMetrics.overallStartTime)
	summaryParts = append(summaryParts, fmt.Sprintf("Completed in %.2fs", duration.Seconds()))

	result.Summary = strings.Join(summaryParts, ". ") + "."
}

// Helper function to get float parameter with default
func getFloatParam(params map[string]interface{}, key string, defaultValue float64) float64 {
	if val, ok := params[key].(float64); ok {
		return val
	}
	return defaultValue
}

// Helper function to check if path matches any exclude pattern
func matchesAnyPattern(path string, patterns []string) bool {
	for _, pattern := range patterns {
		if matched, _ := filepath.Match(pattern, filepath.Base(path)); matched {
			return true
		}
		// Also check if any part of the path contains the pattern
		if strings.Contains(path, pattern) {
			return true
		}
	}
	return false
}

// Helper function to calculate min
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// ================================
// Enhanced Progress Reporter
// ================================

// SpinnerProgressReporter pushes clean status strings to a UI spinner
type SpinnerProgressReporter struct {
	updateStatus func(string)
}

func (r *SpinnerProgressReporter) StartAnalysis(totalFiles int) {
	if r.updateStatus != nil {
		r.updateStatus(fmt.Sprintf("🔍 Analyzing 0/%d files", totalFiles))
	}
}

func (r *SpinnerProgressReporter) UpdateFile(fileName string, current, total int) {
	if r.updateStatus != nil {
		r.updateStatus(fmt.Sprintf("🔍 Analyzing %d/%d: %s", current, total, truncateFileName(fileName, 40)))
	}
}

func (r *SpinnerProgressReporter) FileCompleted(fileName string, success bool, relevance float64) {}

func (r *SpinnerProgressReporter) AnalysisComplete(duration time.Duration, successful, failed int) {
	if r.updateStatus != nil {
		r.updateStatus(fmt.Sprintf("✅ Analysis complete: %d files in %.1fs", successful, duration.Seconds()))
	}
}

// EnhancedProgressReporter provides detailed progress reporting with statistics
type EnhancedProgressReporter struct {
	print            func(format string, args ...interface{})
	startTime        time.Time
	stats            ProgressStats
	mu               sync.Mutex
	showDetails      bool
	useColors        bool
	progressBarWidth int
}

// ProgressStats tracks overall analysis statistics
type ProgressStats struct {
	TotalFiles      int
	FilesProcessed  int
	FilesAnalyzed   int
	FilesSkipped    int
	FilesWithErrors int
	TotalBytes      int64
	TotalIssues     int
	StartTime       time.Time
}

// NewEnhancedProgressReporter creates an enhanced progress reporter
func NewEnhancedProgressReporter(printFunc func(format string, args ...interface{}), showDetails bool) *EnhancedProgressReporter {
	if printFunc == nil {
		printFunc = func(format string, args ...interface{}) {
			fmt.Printf(format, args...)
		}
	}

	return &EnhancedProgressReporter{
		print:            printFunc,
		startTime:        time.Now(),
		showDetails:      showDetails,
		useColors:        true,
		progressBarWidth: 30,
		stats: ProgressStats{
			StartTime: time.Now(),
		},
	}
}

// Color codes for terminal output
const (
	ColorReset  = "\033[0m"
	ColorRed    = "\033[31m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorCyan   = "\033[36m"
	ColorGray   = "\033[37m"
	ColorBold   = "\033[1m"
)

func (r *EnhancedProgressReporter) StartAnalysis(totalFiles int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stats.TotalFiles = totalFiles
	r.stats.StartTime = time.Now()

	if r.showDetails {
		r.print("\n%s🔍 Starting Enhanced Analysis%s\n", r.color(ColorBold), r.color(ColorReset))
		r.print("Total files to analyze: %s%d%s\n", r.color(ColorCyan), totalFiles, r.color(ColorReset))
		r.print("%s\n", strings.Repeat("─", 50))
	}
}

func (r *EnhancedProgressReporter) UpdateFile(fileName string, current, total int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stats.FilesProcessed = current

	if r.showDetails {
		// Create progress bar
		progress := float64(current) / float64(total)
		filled := int(progress * float64(r.progressBarWidth))
		bar := strings.Repeat("█", filled) + strings.Repeat("░", r.progressBarWidth-filled)

		// Clear line and print progress
		r.print("\r%s[%s] %d/%d%s %s%-40s%s",
			r.color(ColorGreen),
			bar,
			current,
			total,
			r.color(ColorReset),
			r.color(ColorCyan),
			truncateFileName(fileName, 40),
			r.color(ColorReset))
	} else {
		// Simple progress indicator
		r.print("\rAnalyzing files... %d/%d", current, total)
	}
}

func (r *EnhancedProgressReporter) FileCompleted(fileName string, success bool, relevance float64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if success {
		r.stats.FilesAnalyzed++

		if r.showDetails {
			relevanceColor := r.getRelevanceColor(relevance)
			status := fmt.Sprintf("%s✓%s", r.color(ColorGreen), r.color(ColorReset))
			r.print("\r\033[K%s %-40s %s[%.2f]%s\n",
				status,
				truncateFileName(fileName, 40),
				r.color(relevanceColor),
				relevance,
				r.color(ColorReset))
		}
	} else {
		r.stats.FilesSkipped++
		r.stats.FilesWithErrors++

		if r.showDetails {
			status := fmt.Sprintf("%s✗%s", r.color(ColorRed), r.color(ColorReset))
			r.print("\r\033[K%s %-40s %s[ERROR]%s\n",
				status,
				truncateFileName(fileName, 40),
				r.color(ColorRed),
				r.color(ColorReset))
		}
	}
}

func (r *EnhancedProgressReporter) AnalysisComplete(duration time.Duration, successful, failed int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Clear the progress line
	r.print("\r" + strings.Repeat(" ", 80) + "\r")
}

// PrintSummary prints a detailed summary of the analysis
func (r *EnhancedProgressReporter) PrintSummary() {
	r.mu.Lock()
	defer r.mu.Unlock()

	duration := time.Since(r.stats.StartTime)

	if r.showDetails {
		r.print("\n%s\n", strings.Repeat("─", 50))
		r.print("%s📊 Analysis Summary%s\n", r.color(ColorBold), r.color(ColorReset))
		r.print("%s\n", strings.Repeat("─", 50))

		// File statistics
		r.print("Files processed:  %s%d%s\n", r.color(ColorCyan), r.stats.FilesProcessed, r.color(ColorReset))
		r.print("Files analyzed:   %s%d%s\n", r.color(ColorGreen), r.stats.FilesAnalyzed, r.color(ColorReset))
		r.print("Files skipped:    %s%d%s\n", r.color(ColorYellow), r.stats.FilesSkipped, r.color(ColorReset))
		if r.stats.FilesWithErrors > 0 {
			r.print("Files with errors: %s%d%s\n", r.color(ColorRed), r.stats.FilesWithErrors, r.color(ColorReset))
		}

		// Performance metrics
		r.print("\nTime elapsed:     %s%.2fs%s\n", r.color(ColorPurple), duration.Seconds(), r.color(ColorReset))
		if r.stats.FilesAnalyzed > 0 {
			avgTime := duration.Seconds() / float64(r.stats.FilesAnalyzed)
			r.print("Avg time/file:    %s%.3fs%s\n", r.color(ColorPurple), avgTime, r.color(ColorReset))
		}

		// Success rate
		if r.stats.FilesProcessed > 0 {
			successRate := float64(r.stats.FilesAnalyzed) / float64(r.stats.FilesProcessed) * 100
			rateColor := ColorGreen
			if successRate < 80 {
				rateColor = ColorYellow
			}
			if successRate < 50 {
				rateColor = ColorRed
			}
			r.print("Success rate:     %s%.1f%%%s\n", r.color(rateColor), successRate, r.color(ColorReset))
		}

		r.print("%s\n", strings.Repeat("─", 50))
	} else {
		r.print("\nAnalysis complete: %d/%d files analyzed in %.2fs\n",
			r.stats.FilesAnalyzed, r.stats.TotalFiles, duration.Seconds())
	}
}

func (r *EnhancedProgressReporter) color(code string) string {
	if r.useColors {
		return code
	}
	return ""
}

func (r *EnhancedProgressReporter) getRelevanceColor(relevance float64) string {
	switch {
	case relevance >= 0.8:
		return ColorGreen
	case relevance >= 0.5:
		return ColorYellow
	case relevance >= 0.3:
		return ColorCyan
	default:
		return ColorGray
	}
}

func truncateFileName(fileName string, maxLen int) string {
	if len(fileName) <= maxLen {
		return fileName
	}

	// Try to keep the extension
	ext := filepath.Ext(fileName)
	base := filepath.Base(fileName)

	if len(base) <= maxLen {
		return base
	}

	// Truncate and add ellipsis
	keepLen := maxLen - 3 - len(ext) // 3 for "..."
	if keepLen > 0 {
		return base[:keepLen] + "..." + ext
	}

	return base[:maxLen-3] + "..."
}

// ================================
// Helper Functions
// ================================

// getDefaultExcludePatterns returns default patterns to exclude
func getDefaultExcludePatterns() []string {
	return []string{
		".git", ".svn", ".hg", ".bzr",
		"node_modules", "vendor", ".vendor",
		"build", "dist", "target", ".build",
		"*.log", "*.tmp", "*.temp",
		".DS_Store", "Thumbs.db",
		"*.pyc", "*.pyo", "__pycache__",
		"*.class", "*.jar",
		"*.o", "*.so", "*.dll", "*.exe",
	}
}

// getBoolParam gets a boolean parameter with default
func getBoolParam(params map[string]interface{}, key string, defaultValue bool) bool {
	if val, ok := params[key].(bool); ok {
		return val
	}
	return defaultValue
}

// getIntParam gets an integer parameter with default
func getIntParam(params map[string]interface{}, key string, defaultValue int) int {
	if val, ok := params[key].(int); ok {
		return val
	}
	if val, ok := params[key].(float64); ok {
		return int(val)
	}
	return defaultValue
}
