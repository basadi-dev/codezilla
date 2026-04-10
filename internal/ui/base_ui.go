package ui

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/term"
)

// BaseUI implements the UI interface with a base interface
type BaseUI struct {
	theme        Theme
	reader       *FixedInput
	writer       *bufio.Writer
	spinnerStop  chan bool
	spinnerMutex sync.Mutex
	width        int
}

// NewBaseUI creates a new base UI
func NewBaseUI(historyFile string) (UI, error) {
	// Get terminal width
	width, _, _ := term.GetSize(int(os.Stdout.Fd()))
	if width == 0 {
		width = 80
	}

	// Create input reader
	reader, err := NewFixedInput(
		"", // Prompt will be set by theme
		historyFile,
	)
	if err != nil {
		return nil, err
	}

	ui := &BaseUI{
		theme:  DefaultTheme(),
		reader: reader,
		writer: bufio.NewWriter(os.Stdout),
		width:  width,
	}

	return ui, nil
}

// DefaultTheme returns the default terminal theme
func DefaultTheme() Theme {
	return Theme{
		ColorReset:  "\033[0m",
		ColorRed:    "\033[31m",
		ColorGreen:  "\033[32m",
		ColorYellow: "\033[33m",
		ColorBlue:   "\033[34m",
		ColorPurple: "\033[35m",
		ColorCyan:   "\033[36m",
		ColorBold:   "\033[1m",
		ColorDim:    "\033[2m",

		IconSuccess: "✓",
		IconError:   "✗",
		IconWarning: "⚠",
		IconInfo:    "ℹ",
		IconPrompt:  ">",
	}
}

// Clear clears the terminal screen
func (ui *BaseUI) Clear() {
	fmt.Fprint(ui.writer, "\033[2J\033[H")
	ui.writer.Flush()
}

// ShowBanner displays the application banner
func (ui *BaseUI) ShowBanner() {
	banner := `
   ____          _           _ _ _       
  / ___|___   __| | ___ ____(_) | | __ _ 
 | |   / _ \ / _` + "`" + ` |/ _ \_  /| | | |/ _` + "`" + ` |
 | |__| (_) | (_| |  __// / | | | | (_| |
  \____\___/ \__,_|\___/___|_|_|_|\__,_|
                                         
`
	fmt.Fprint(ui.writer, ui.theme.ColorCyan+banner+ui.theme.ColorReset)
	fmt.Fprintln(ui.writer, ui.theme.ColorBold+"AI-Powered Coding Assistant"+ui.theme.ColorReset)
	fmt.Fprintln(ui.writer, strings.Repeat("─", ui.width))
	ui.writer.Flush()
}

// ShowWelcome displays the welcome message
func (ui *BaseUI) ShowWelcome(model, ollamaURL string, contextEnabled bool) {
	ui.Print("%sWelcome!%s Type %s/help%s for commands or start chatting.\n",
		ui.theme.ColorBold, ui.theme.ColorReset,
		ui.theme.ColorYellow, ui.theme.ColorReset)
	ui.Print("Press %sEnter%s to submit your message.\n",
		ui.theme.ColorYellow, ui.theme.ColorReset)
	ui.Print("Using model: %s%s%s\n",
		ui.theme.ColorYellow, model, ui.theme.ColorReset)
	ui.Print("Ollama URL: %s%s%s\n",
		ui.theme.ColorDim, ollamaURL, ui.theme.ColorReset)

	if contextEnabled {
		ui.Print("Context retention: %sEnabled%s\n",
			ui.theme.ColorGreen, ui.theme.ColorReset)
	} else {
		ui.Print("Context retention: %sDisabled%s (use /context on to enable)\n",
			ui.theme.ColorDim, ui.theme.ColorReset)
	}
	ui.Println("")
}

// ShowPrompt returns the prompt string
func (ui *BaseUI) ShowPrompt() string {
	return fmt.Sprintf("%scodezilla%s 🤖 ",
		ui.theme.ColorBlue, ui.theme.ColorReset)
}

// Print outputs formatted text
func (ui *BaseUI) Print(format string, args ...interface{}) {
	fmt.Fprintf(ui.writer, format, args...)
	ui.writer.Flush()
}

// Println outputs formatted text with newline
func (ui *BaseUI) Println(format string, args ...interface{}) {
	fmt.Fprintf(ui.writer, format+"\n", args...)
	ui.writer.Flush()
}

// Success shows a success message
func (ui *BaseUI) Success(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Println("%s%s%s %s", ui.theme.ColorGreen, ui.theme.IconSuccess, ui.theme.ColorReset, msg)
}

// Error shows an error message
func (ui *BaseUI) Error(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Println("%s%s%s %s", ui.theme.ColorRed, ui.theme.IconError, ui.theme.ColorReset, msg)
}

// Warning shows a warning message
func (ui *BaseUI) Warning(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Println("%s%s%s %s", ui.theme.ColorYellow, ui.theme.IconWarning, ui.theme.ColorReset, msg)
}

// Info shows an info message
func (ui *BaseUI) Info(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Println("%s%s%s %s", ui.theme.ColorBlue, ui.theme.IconInfo, ui.theme.ColorReset, msg)
}

// ShowThinking shows a thinking/loading indicator
func (ui *BaseUI) ShowThinking() {
	ui.spinnerMutex.Lock()
	if ui.spinnerStop != nil {
		ui.spinnerMutex.Unlock()
		return
	}

	ui.spinnerStop = make(chan bool)
	ui.spinnerMutex.Unlock()

	go func() {
		chars := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
		i := 0
		for {
			select {
			case <-ui.spinnerStop:
				// Clear spinner line
				ui.Print("\r%s\r", strings.Repeat(" ", 20))
				return
			default:
				ui.Print("\r%s%s Thinking...%s",
					ui.theme.ColorCyan, chars[i%len(chars)], ui.theme.ColorReset)
				i++
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()
}

// HideThinking hides the thinking indicator
func (ui *BaseUI) HideThinking() {
	ui.spinnerMutex.Lock()
	defer ui.spinnerMutex.Unlock()

	if ui.spinnerStop != nil {
		close(ui.spinnerStop)
		ui.spinnerStop = nil
	}
}

// ShowResponse displays an AI response
func (ui *BaseUI) ShowResponse(response string) {
	ui.Println("\n%sAssistant:%s", ui.theme.ColorGreen, ui.theme.ColorReset)

	// Process response for code blocks
	lines := strings.Split(response, "\n")
	inCode := false

	for _, line := range lines {
		if strings.HasPrefix(line, "```") {
			inCode = !inCode
			if inCode {
				lang := strings.TrimPrefix(line, "```")
				ui.Print("%s```%s%s\n", ui.theme.ColorPurple, lang, ui.theme.ColorReset)
			} else {
				ui.Print("%s```%s\n", ui.theme.ColorPurple, ui.theme.ColorReset)
			}
		} else {
			ui.Println(line)
		}
	}
	ui.Println("")
}

// ShowResponseStream displays a streaming AI response token by token.
func (ui *BaseUI) ShowResponseStream(ch <-chan string) {
	ui.Println("\n%sAssistant:%s", ui.theme.ColorGreen, ui.theme.ColorReset)
	for token := range ch {
		fmt.Fprint(ui.writer, token)
		ui.writer.Flush()
	}
	ui.Println("")
	ui.writer.Flush()
}

// ShowCode displays a code block
func (ui *BaseUI) ShowCode(language, code string) {
	ui.Println("%s```%s%s", ui.theme.ColorPurple, language, ui.theme.ColorReset)
	ui.Print("%s", code)
	if !strings.HasSuffix(code, "\n") {
		ui.Println("")
	}
	ui.Println("%s```%s", ui.theme.ColorPurple, ui.theme.ColorReset)
}

// ShowHelp displays help information
func (ui *BaseUI) ShowHelp() {
	ui.Println("\n%sAvailable Commands:%s", ui.theme.ColorBold, ui.theme.ColorReset)

	commands := []struct {
		cmd  string
		desc string
	}{
		{"/help, /h", "Show this help"},
		{"/exit, /quit, /q", "Exit the application"},
		{"/clear, /c", "Clear the screen"},
		{"/models", "List available models"},
		{"/model [name]", "Show or change model"},
		{"/context [on|off|clear|show]", "Manage context"},
		{"/tools", "Show available tools"},
		{"/reset", "Reset conversation"},
		{"/save <filename>", "Save conversation to JSON file"},
		{"/load <filename>", "Load conversation from JSON file"},
	}

	for _, cmd := range commands {
		ui.Print("  %s%-30s%s %s\n",
			ui.theme.ColorYellow, cmd.cmd, ui.theme.ColorReset, cmd.desc)
	}
	ui.Println("")
}

// ShowModels displays available models
func (ui *BaseUI) ShowModels(models []string, current string) {
	ui.Println("\n%sAvailable Models:%s", ui.theme.ColorBold, ui.theme.ColorReset)

	for _, model := range models {
		if model == current {
			ui.Print("  %s*%s %s (current)\n",
				ui.theme.ColorGreen, ui.theme.ColorReset, model)
		} else {
			ui.Println("    %s", model)
		}
	}
	ui.Println("")
}

// ShowTools displays available tools
func (ui *BaseUI) ShowTools(tools []ToolInfo) {
	ui.Println("\n%sAvailable Tools:%s", ui.theme.ColorBold, ui.theme.ColorReset)

	for _, tool := range tools {
		permColor := ui.theme.ColorYellow
		if tool.Permission == "never_ask" {
			permColor = ui.theme.ColorGreen
		} else if tool.Permission == "always_ask" {
			permColor = ui.theme.ColorRed
		}

		ui.Print("  • %s%-15s%s %s %s(%s)%s\n",
			ui.theme.ColorYellow, tool.Name, ui.theme.ColorReset,
			tool.Description,
			permColor, tool.Permission, ui.theme.ColorReset)
	}
	ui.Println("")
}

// ShowContext displays conversation context
func (ui *BaseUI) ShowContext(context string) {
	ui.Println("\n%sConversation Context:%s", ui.theme.ColorBold, ui.theme.ColorReset)
	if context == "" {
		ui.Println("No context stored")
	} else {
		ui.Println(context)
	}
	ui.Println("")
}

// ReadLine reads a line of input (single-line mode)
func (ui *BaseUI) ReadLine() (string, error) {
	ui.reader.SetPrompt(ui.ShowPrompt())
	return ui.reader.ReadLine()
}

// ReadPassword reads a password without echoing
func (ui *BaseUI) ReadPassword(prompt string) (string, error) {
	ui.Print("%s", prompt)

	fd := int(os.Stdin.Fd())
	if term.IsTerminal(fd) {
		password, err := term.ReadPassword(fd)
		ui.Println("")
		return string(password), err
	}

	// Fallback to regular reading
	return ui.ReadLine()
}

// Confirm asks for yes/no confirmation
func (ui *BaseUI) Confirm(prompt string) (bool, error) {
	for {
		ui.Print("%s (y/n): ", prompt)
		response, err := ui.ReadLine()
		if err != nil {
			return false, err
		}

		response = strings.ToLower(strings.TrimSpace(response))
		switch response {
		case "y", "yes":
			return true, nil
		case "n", "no":
			return false, nil
		default:
			ui.Warning("Please answer 'y' or 'n'")
		}
	}
}

// GetTheme returns the current theme
func (ui *BaseUI) GetTheme() Theme {
	return ui.theme
}

// SetTheme sets a new theme
func (ui *BaseUI) SetTheme(theme Theme) {
	ui.theme = theme
}

// DisableColors disables all colors in the theme
func (ui *BaseUI) DisableColors() {
	ui.theme = Theme{
		ColorReset:  "",
		ColorRed:    "",
		ColorGreen:  "",
		ColorYellow: "",
		ColorBlue:   "",
		ColorPurple: "",
		ColorCyan:   "",
		ColorBold:   "",
		ColorDim:    "",

		IconSuccess: ui.theme.IconSuccess,
		IconError:   ui.theme.IconError,
		IconWarning: ui.theme.IconWarning,
		IconInfo:    ui.theme.IconInfo,
		IconPrompt:  ui.theme.IconPrompt,
	}
}
