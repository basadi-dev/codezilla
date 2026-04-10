package ui

// Theme defines the colors and styles for the UI
type Theme struct {
	// Colors
	ColorReset  string
	ColorRed    string
	ColorGreen  string
	ColorYellow string
	ColorBlue   string
	ColorPurple string
	ColorCyan   string
	ColorBold   string
	ColorDim    string

	// Icons
	IconSuccess string
	IconError   string
	IconWarning string
	IconInfo    string
	IconPrompt  string
}

// UI defines the interface for user interaction
type UI interface {
	// Display methods
	Clear()
	ShowBanner()
	ShowWelcome(model, ollamaURL string, contextEnabled bool)
	ShowPrompt() string

	// Output methods
	Print(format string, args ...interface{})
	Println(format string, args ...interface{})
	Success(format string, args ...interface{})
	Error(format string, args ...interface{})
	Warning(format string, args ...interface{})
	Info(format string, args ...interface{})

	// Formatted output
	ShowThinking()
	HideThinking()
	ShowResponse(response string)
	ShowResponseStream(ch <-chan string)
	ShowCode(language, code string)

	// Structured displays
	ShowHelp()
	ShowModels(models []string, current string)
	ShowTools(tools []ToolInfo)
	ShowContext(context string)

	// Input methods
	ReadLine() (string, error)
	ReadPassword(prompt string) (string, error)
	Confirm(prompt string) (bool, error)

	// Theme management
	GetTheme() Theme
	SetTheme(theme Theme)
	DisableColors()
}

// ToolInfo represents information about a tool
type ToolInfo struct {
	Name        string
	Description string
	Permission  string
}

// Factory function type for creating UI instances
type Factory func(historyFile string) (UI, error)
