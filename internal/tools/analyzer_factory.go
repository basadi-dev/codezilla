package tools

import (
	"codezilla/internal/core/llm"
	"codezilla/pkg/logger"
)

// AnalyzerFactory creates file analyzers based on configuration
type AnalyzerFactory struct {
	llmClient *llm.Client
	provider  string
	model     string
	logger    *logger.Logger
}

// NewAnalyzerFactory creates a new analyzer factory
func NewAnalyzerFactory(llmClient *llm.Client, provider, model string, logger *logger.Logger) *AnalyzerFactory {
	return &AnalyzerFactory{
		llmClient: llmClient,
		provider:  provider,
		model:     model,
		logger:    logger,
	}
}

// CreateAnalyzer creates an appropriate analyzer based on availability
func (f *AnalyzerFactory) CreateAnalyzer(useLLM bool) FileAnalyzer {
	if useLLM && f.llmClient != nil {
		return NewLLMFileAnalyzer(f.llmClient, f.provider, f.model, f.logger)
	}
	return NewDefaultFileAnalyzer()
}

// CreateProjectScanAnalyzer creates the enhanced project scan analyzer with default stderr printing.
func (f *AnalyzerFactory) CreateProjectScanAnalyzer() *ProjectScanAnalyzer {
	return NewProjectScanAnalyzer(f.llmClient, f.provider, f.model, f.logger)
}

// CreateProjectScanAnalyzerWithPrint creates the enhanced project scan analyzer with a custom
// printFunc. Use this to wrap progress output with UI hooks (e.g. hide/show spinner) so that
// the analyzer's progress lines don't interleave with the terminal's thinking indicator.
func (f *AnalyzerFactory) CreateProjectScanAnalyzerWithPrint(printFunc func(format string, args ...interface{})) *ProjectScanAnalyzer {
	a := NewProjectScanAnalyzer(f.llmClient, f.provider, f.model, f.logger)
	a.SetPrintFunc(printFunc)
	return a
}

