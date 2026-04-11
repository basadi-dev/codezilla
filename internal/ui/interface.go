package ui

import "github.com/charmbracelet/lipgloss"

// Theme defines the colors and styles for the UI
type Theme struct {
	// Styles
	StyleReset   lipgloss.Style
	StyleRed     lipgloss.Style
	StyleGreen   lipgloss.Style
	StyleYellow  lipgloss.Style
	StyleBlue    lipgloss.Style
	StylePurple  lipgloss.Style
	StyleCyan    lipgloss.Style
	StyleBold    lipgloss.Style
	StyleDim     lipgloss.Style

	// Icons
	IconSuccess string
	IconError   string
	IconWarning string
	IconInfo    string
	IconPrompt  string

	// Autocomplete Theme
	ACTheme AutocompleteTheme
}

// AutocompleteTheme defines the colors for the popup interactive autocomplete
type AutocompleteTheme struct {
	Cmd       lipgloss.Style
	Desc      lipgloss.Style
	HiCmd     lipgloss.Style
	HiDesc    lipgloss.Style
	HiPrefix  lipgloss.Style
	Separator lipgloss.Style
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
	ShowModels(models []string, current, planner, subAgent string)
	ShowTools(tools []ToolInfo)
	ShowContext(context string)

	// Input methods
	ReadLine() (string, error)
	ReadPassword(prompt string) (string, error)
	Confirm(prompt string) (bool, error)

	// Theme management
	GetTheme() Theme
	SetTheme(theme Theme)
	SetModel(model string)
	DisableColors()
}

// ToolInfo represents information about a tool
type ToolInfo struct {
	Name        string
	Description string
	Permission  string
}

// Completion represents an auto-complete candidate with an optional description
type Completion struct {
	Text        string // The literal text inserted into the input line
	Display     string // Optional display presentation (e.g. "/help, /h")
	Description string
}

// Completer is an optional interface that UI implementations may satisfy to
// support Tab auto-completion. App uses a type assertion to wire completions.
type Completer interface {
	SetCompleter(fn func(line string) []Completion)
}

// HistoryProvider is an optional interface that UI implementations may satisfy
// to expose command history for viewing, searching, and clearing.
type HistoryProvider interface {
	// GetHistory returns the most recent N history entries (newest last).
	// If n <= 0, returns all entries.
	GetHistory(n int) []string
	// SearchHistory returns history entries that contain the query substring.
	SearchHistory(query string) []string
	// ClearHistory removes all history entries and deletes the history file.
	ClearHistory() error
}

// Factory function type for creating UI instances
type Factory func(historyFile string) (UI, error)
