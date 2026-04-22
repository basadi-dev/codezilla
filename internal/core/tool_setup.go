package core

import (
	"fmt"
	"os"

	"codezilla/internal/config"
	"codezilla/internal/core/llm"
	"codezilla/internal/tools"
	"codezilla/internal/ui"
	"codezilla/pkg/logger"
)

// registerTools registers all available tools with the registry and sets up
// their permission levels. This is separated from app.go for maintainability.
func registerTools(registry tools.ToolRegistry, llmClient *llm.Client, cfg *config.Config, logger *logger.Logger, permissionMgr tools.ToolPermissionManager, todoMgr *tools.TodoManager, appUI ui.UI) {
	// Unified File operation tool
	registry.RegisterTool(tools.NewFileManageTool())
	registry.RegisterTool(tools.NewFileUndoTool())
	registry.RegisterTool(tools.NewGrepSearchTool())
	registry.RegisterTool(tools.NewMultiReplaceTool())
	registry.RegisterTool(tools.NewRepoMapGeneratorTool())

	// Create analyzer factory and register analyzer tool
	analyzerModel := cfg.LLM.Models.Fast
	if analyzerModel == "" {
		analyzerModel = cfg.LLM.Models.Default
	}
	analyzerFactory := tools.NewAnalyzerFactory(llmClient, cfg.LLM.Provider, analyzerModel, logger)

	// Wrap the progress printFunc so it hides the spinner before writing and
	// restores it afterwards, preventing interleaving with the thinking indicator.
	analyzerPrint := func(format string, args ...interface{}) {
		appUI.HideThinking()
		fmt.Fprintf(os.Stderr, format, args...)
		appUI.ShowThinking()
	}

	// Register the analyzer (formerly V2)
	projectAnalyzer := analyzerFactory.CreateProjectScanAnalyzerWithPrint(analyzerPrint)
	projectAnalyzer.SetStatusUpdater(func(status string) {
		appUI.UpdateThinkingStatus(status)
	})
	registry.RegisterTool(projectAnalyzer)

	// All tools are auto-approved (NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("fileManage", tools.NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("fileEdit", tools.NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("fileUndo", tools.NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("multiReplace", tools.NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("grepSearch", tools.NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("projectScanAnalyzer", tools.NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("repoMapGenerator", tools.NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("execute", tools.NeverAsk)

	registry.RegisterTool(tools.NewExecuteTool(30))
	registry.RegisterTool(tools.NewJobOutputTool())
	registry.RegisterTool(tools.NewJobKillTool())
	permissionMgr.SetDefaultPermissionLevel("jobOutput", tools.NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("jobKill", tools.NeverAsk)

	// Todo management tools — manager is passed in from NewApp so the UI hook can access it
	for _, tool := range tools.GetTodoTools(todoMgr) {
		registry.RegisterTool(tool)
	}

	// Web search – embedded metasearch engine (DDG HTML + Wikipedia + optional Bing), no API key needed
	msConfig := tools.MetasearchConfig{
		EnableBing:     cfg.Metasearch.EnableBing,
		TimeoutSeconds: cfg.Metasearch.TimeoutSeconds,
		MaxResults:     cfg.Metasearch.MaxResults,
		JitterMs:       cfg.Metasearch.JitterMs,
	}
	if msConfig.TimeoutSeconds <= 0 {
		msConfig.TimeoutSeconds = 8
	}
	if msConfig.MaxResults <= 0 {
		msConfig.MaxResults = 5
	}
	registry.RegisterTool(tools.NewWebSearchTool(msConfig))
	permissionMgr.SetDefaultPermissionLevel("webSearch", tools.NeverAsk)

	// Fetch URL – converts any webpage to clean markdown via Jina AI Reader (free, no key)
	registry.RegisterTool(tools.NewFetchURLTool())
	permissionMgr.SetDefaultPermissionLevel("fetchURL", tools.NeverAsk)
}
