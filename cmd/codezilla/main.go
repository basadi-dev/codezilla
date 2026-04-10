package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"codezilla/internal/config"
	"codezilla/internal/core"
	"codezilla/internal/ui"
)

func main() {
	// Parse command line flags
	var (
		configPath  = flag.String("config", "", "Path to config file")
		uiType      = flag.String("ui", "fancy", "UI type: minimal or fancy")
		noColors    = flag.Bool("no-colors", false, "Disable colored output")
		provider    = flag.String("provider", "", "Override LLM provider (ollama, openai, anthropic, gemini)")
		model       = flag.String("model", "", "Override default model")
		ollamaURL   = flag.String("ollama-url", "", "Override Ollama API URL")
		temperature = flag.Float64("temperature", -1, "Override temperature (0.0-1.0)")
		maxTokens   = flag.Int("max-tokens", 0, "Override max tokens")
		version     = flag.Bool("version", false, "Show version")
		help        = flag.Bool("help", false, "Show help")
	)
	flag.Parse()

	// Handle version
	if *version {
		fmt.Println("Codezilla v2.0.0 - Modular Architecture")
		os.Exit(0)
	}

	// Handle help
	if *help {
		printHelp()
		os.Exit(0)
	}

	// Get config path
	if *configPath == "" {
		*configPath = getDefaultConfigPath()
	}

	// Load configuration
	cfg, err := config.LoadConfig(*configPath)
	if err != nil {
		cfg = config.DefaultConfig()
		fmt.Printf("Note: Using default configuration\n")
	}

	// Apply CLI overrides
	if *provider != "" {
		cfg.LLM.Provider = *provider
	}
	if *model != "" {
		cfg.LLM.Models.Default = *model
	}
	if *ollamaURL != "" {
		cfg.LLM.Ollama.BaseURL = *ollamaURL
	}
	if *temperature >= 0 && *temperature <= 1 {
		cfg.Temperature = float32(*temperature)
	}
	if *maxTokens > 0 {
		cfg.MaxTokens = *maxTokens
	}

	// Apply color settings
	if *noColors {
		cfg.NoColor = true
	}

	// Get history file path
	historyPath, _ := ui.GetDefaultHistoryFilePath()

	// Create UI based on selection
	var appUI ui.UI
	switch *uiType {
	case "minimal":
		appUI, err = ui.NewMinimalUI(historyPath)
	default:
		// Default to fancy UI
		appUI, err = ui.NewFancyUI(historyPath)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize UI: %v\n", err)
		os.Exit(1)
	}

	// Disable colors if requested
	if *noColors {
		appUI.DisableColors()
	}

	// Create the core application
	app, err := core.NewApp(cfg, appUI)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize application: %v\n", err)
		os.Exit(1)
	}
	defer app.Close()

	// Setup signal handling
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		appUI.Info("\nShutting down...")
		cancel()
	}()

	// Run the application
	if err := app.Run(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func getDefaultConfigPath() string {
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".config", "codezilla", "config.yaml")
	}
	return "config.yaml"
}

func printHelp() {
	fmt.Print(`Codezilla - Modular AI-powered coding assistant

Usage:
  codezilla [options]

Options:
  -config string       Path to configuration file
  -provider string     Override LLM provider (ollama, openai, anthropic, gemini)
  -model string        Override default model (e.g., "qwen3:14b")
  -ollama-url string   Override Ollama API URL (e.g., "http://localhost:11434")
  -temperature float   Override temperature (0.0-1.0)
  -max-tokens int      Override max tokens
  -ui string           UI type: fancy (default) or minimal
  -no-colors           Disable colored output
  -version             Show version information
  -help                Show this help message

UI Types:
  fancy     - Enhanced UI with animations and emoji (default)
  minimal   - Minimal UI with no colors or special formatting

Examples:
  # Run with default fancy UI
  codezilla

  # Run with minimal UI
  codezilla -ui minimal

  # Run without colors
  codezilla -no-colors

  # Use custom config
  codezilla -config /path/to/config.yaml

  # Override model
  codezilla -model "llama3:latest"

  # Override provider
  codezilla -provider openai -model "gpt-4o"

  # Override Ollama URL
  codezilla -ollama-url "http://192.168.1.100:11434"

  # Override temperature
  codezilla -temperature 0.8

The modular architecture allows easy switching between different UI implementations
while keeping the core functionality unchanged.
`)
}
