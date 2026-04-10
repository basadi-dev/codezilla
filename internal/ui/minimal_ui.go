package ui

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"golang.org/x/term"
)

// MinimalUI implements a minimal UI with no colors or fancy formatting
type MinimalUI struct {
	reader *FixedInput
}

// NewMinimalUI creates a minimal UI implementation
func NewMinimalUI(historyFile string) (UI, error) {
	reader, err := NewFixedInput("> ", historyFile)
	if err != nil {
		return nil, err
	}

	return &MinimalUI{reader: reader}, nil
}

func (ui *MinimalUI) Clear() {
	// Simple clear - just add some blank lines
	fmt.Print("\n\n\n")
}

func (ui *MinimalUI) ShowBanner() {
	fmt.Println()
	fmt.Println("CODEZILLA - AI-Powered Coding Assistant")
	fmt.Println("========================================")
	fmt.Println()
}

func (ui *MinimalUI) ShowWelcome(model, ollamaURL string, contextEnabled bool) {
	fmt.Println("Welcome! Type /help for commands.")
	fmt.Printf("Model: %s\n", model)
	fmt.Printf("Context: %s\n", map[bool]string{true: "enabled", false: "disabled"}[contextEnabled])
	fmt.Println()
}

func (ui *MinimalUI) ShowPrompt() string {
	return "> "
}

func (ui *MinimalUI) Print(format string, args ...interface{}) {
	fmt.Printf(format, args...)
}

func (ui *MinimalUI) Println(format string, args ...interface{}) {
	fmt.Printf(format+"\n", args...)
}

func (ui *MinimalUI) Success(format string, args ...interface{}) {
	fmt.Printf("[OK] "+format+"\n", args...)
}

func (ui *MinimalUI) Error(format string, args ...interface{}) {
	fmt.Printf("[ERROR] "+format+"\n", args...)
}

func (ui *MinimalUI) Warning(format string, args ...interface{}) {
	fmt.Printf("[WARN] "+format+"\n", args...)
}

func (ui *MinimalUI) Info(format string, args ...interface{}) {
	fmt.Printf("[INFO] "+format+"\n", args...)
}

func (ui *MinimalUI) ShowThinking() {
	fmt.Print("Thinking...")
}

func (ui *MinimalUI) HideThinking() {
	fmt.Print("\r            \r")
}

func (ui *MinimalUI) ShowResponse(response string) {
	fmt.Println("\nAssistant:")
	fmt.Println(response)
	fmt.Println()
}

func (ui *MinimalUI) ShowResponseStream(ch <-chan string) {
	fmt.Println("\nAssistant:")
	for token := range ch {
		fmt.Print(token)
	}
	fmt.Println()
}

func (ui *MinimalUI) ShowCode(language, code string) {
	fmt.Printf("--- %s ---\n", language)
	fmt.Print(code)
	if !strings.HasSuffix(code, "\n") {
		fmt.Println()
	}
	fmt.Println("--- end ---")
}

func (ui *MinimalUI) ShowHelp() {
	fmt.Println("\nCommands:")
	fmt.Println("  /help            - Show help")
	fmt.Println("  /exit            - Exit")
	fmt.Println("  /clear           - Clear screen")
	fmt.Println("  /models          - List models")
	fmt.Println("  /model           - Show/change model")
	fmt.Println("  /context         - Manage context")
	fmt.Println("  /tools           - Show tools")
	fmt.Println("  /save <filename> - Save conversation")
	fmt.Println("  /load <filename> - Load conversation")
	fmt.Println()
}

func (ui *MinimalUI) ShowModels(models []string, current string) {
	fmt.Println("\nModels:")
	for _, model := range models {
		if model == current {
			fmt.Printf("  * %s\n", model)
		} else {
			fmt.Printf("    %s\n", model)
		}
	}
	fmt.Println()
}

func (ui *MinimalUI) ShowTools(tools []ToolInfo) {
	fmt.Println("\nTools:")
	for _, tool := range tools {
		fmt.Printf("  - %s: %s (%s)\n", tool.Name, tool.Description, tool.Permission)
	}
	fmt.Println()
}

func (ui *MinimalUI) ShowContext(context string) {
	fmt.Println("\nContext:")
	if context == "" {
		fmt.Println("  (empty)")
	} else {
		fmt.Println(context)
	}
	fmt.Println()
}

func (ui *MinimalUI) ReadLine() (string, error) {
	return ui.reader.ReadLine()
}

func (ui *MinimalUI) ReadPassword(prompt string) (string, error) {
	fmt.Print(prompt)
	// Try to read password securely
	fd := int(os.Stdin.Fd())
	if term.IsTerminal(fd) {
		// Read password without echo
		passBytes, err := term.ReadPassword(fd)
		fmt.Println() // Add newline after password input
		if err != nil {
			return "", err
		}
		return string(passBytes), nil
	}
	// Fallback for non-terminal (e.g., piped input)
	reader := bufio.NewReader(os.Stdin)
	pass, err := reader.ReadString('\n')
	return strings.TrimSpace(pass), err
}

func (ui *MinimalUI) Confirm(prompt string) (bool, error) {
	fmt.Printf("%s (y/n): ", prompt)
	reader := bufio.NewReader(os.Stdin)
	response, err := reader.ReadString('\n')
	if err != nil {
		return false, err
	}
	response = strings.ToLower(strings.TrimSpace(response))
	return response == "y" || response == "yes", nil
}

func (ui *MinimalUI) GetTheme() Theme {
	return Theme{} // Empty theme
}

func (ui *MinimalUI) SetTheme(theme Theme) {
	// No-op for minimal UI
}

func (ui *MinimalUI) DisableColors() {
	// No-op - minimal UI has no colors
}
