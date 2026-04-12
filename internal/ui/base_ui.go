package ui

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/term"
	"github.com/charmbracelet/lipgloss"
)

// BaseUI implements the UI interface with a base interface
type BaseUI struct {
	theme          Theme
	reader         *FixedInput
	writer         *bufio.Writer
	spinnerStop    chan bool
	spinnerMutex   sync.Mutex
	spinnerStatus  string // updated via UpdateThinkingStatus
	width          int
	currentModel   string
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
		theme:  ThemeTokyoNight(),
		reader: reader,
		writer: bufio.NewWriter(os.Stdout),
		width:  width,
	}
	
	// Apply initial theme to reader
	ui.reader.SetTheme(ui.theme)

	return ui, nil
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
	fmt.Fprint(ui.writer, ui.theme.StyleCyan.Render(banner))
	title := ui.theme.StyleBold.Render("AI-Powered Coding Assistant")
	fmt.Fprintln(ui.writer, title)
	separator := ui.theme.StyleDim.Render(strings.Repeat("─", ui.width))
	fmt.Fprintln(ui.writer, separator)
	ui.writer.Flush()
}

// SetModel sets the active model name shown in the UI.
func (ui *BaseUI) SetModel(model string) {
	ui.currentModel = model
}

// ShowWelcome displays the welcome message
func (ui *BaseUI) ShowWelcome(model, ollamaURL string, contextEnabled bool) {
	ui.Print("%s Type %s for commands or start chatting.\n",
		ui.theme.StyleBold.Render("Welcome!"),
		ui.theme.StyleYellow.Render("/help"))
	ui.Print("Press %s to submit your message.\n",
		ui.theme.StyleYellow.Render("Enter"))
	ui.Print("Using model: %s\n",
		ui.theme.StyleYellow.Render(model))
	ui.Print("Ollama URL: %s\n",
		ui.theme.StyleDim.Render(ollamaURL))

	if contextEnabled {
		ui.Print("Context retention: %s\n",
			ui.theme.StyleGreen.Render("Enabled"))
	} else {
		ui.Print("Context retention: %s (use /context on to enable)\n",
			ui.theme.StyleDim.Render("Disabled"))
	}
	ui.Println("")
}

// ShowPrompt returns the prompt string
func (ui *BaseUI) ShowPrompt() string {
	return fmt.Sprintf("%s 🤖 ", ui.theme.StyleBlue.Render("codezilla"))
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
	ui.Println("%s %s", ui.theme.StyleGreen.Render(ui.theme.IconSuccess), msg)
}

// Error shows an error message
func (ui *BaseUI) Error(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Println("%s %s", ui.theme.StyleRed.Render(ui.theme.IconError), msg)
}

// Warning shows a warning message
func (ui *BaseUI) Warning(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Println("%s %s", ui.theme.StyleYellow.Render(ui.theme.IconWarning), msg)
}

// Info shows an info message
func (ui *BaseUI) Info(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Println("%s %s", ui.theme.StyleBlue.Render(ui.theme.IconInfo), msg)
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
				ui.Print("\r%s Thinking...", ui.theme.StyleCyan.Render(chars[i%len(chars)]))
				i++
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()
}

// UpdateThinkingStatus updates the label shown in the active spinner.
// BaseUI no-op; FancyUI overrides.
func (ui *BaseUI) UpdateThinkingStatus(label string) {
	ui.spinnerMutex.Lock()
	ui.spinnerStatus = label
	ui.spinnerMutex.Unlock()
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
	ui.Println("\n%s", ui.theme.StyleGreen.Render("Assistant:"))

	// Process response for code blocks
	lines := strings.Split(response, "\n")
	inCode := false

	for _, line := range lines {
		if strings.HasPrefix(line, "```") {
			inCode = !inCode
			if inCode {
				lang := strings.TrimPrefix(line, "```")
				ui.Print("%s\n", ui.theme.StylePurple.Render("```"+lang))
			} else {
				ui.Print("%s\n", ui.theme.StylePurple.Render("```"))
			}
		} else {
			ui.Println(line)
		}
	}
	ui.Println("")
}

// ShowResponseStream displays a streaming AI response token by token.
func (ui *BaseUI) ShowResponseStream(ch <-chan string) {
	ui.Println("\n%s", ui.theme.StyleGreen.Render("Assistant:"))
	for token := range ch {
		fmt.Fprint(ui.writer, token)
		ui.writer.Flush()
	}
	ui.Println("")
	ui.writer.Flush()
}

// ShowCode displays a code block
func (ui *BaseUI) ShowCode(language, code string) {
	ui.Println("%s", ui.theme.StylePurple.Render("```"+language))
	ui.Print("%s", code)
	if !strings.HasSuffix(code, "\n") {
		ui.Println("")
	}
	ui.Println("%s", ui.theme.StylePurple.Render("```"))
}

// ShowHelp displays help information
func (ui *BaseUI) ShowHelp() {
	ui.Println("\n%s", ui.theme.StyleBold.Render("Available Commands:"))

	// Use lipgloss-width-aware padding to avoid ANSI escape byte miscount
	cmdStyle := ui.theme.StyleYellow.Width(32)
	descStyle := ui.theme.StyleDim

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
		{"/skill list", "List all available skills"},
		{"/skill <name>", "Toggle a skill on/off"},
		{"/skill off <name>", "Deactivate a skill"},
		{"/skill reset", "Deactivate all skills"},
		{"/skill info <name>", "Show skill details"},
		{"/reset", "Reset conversation"},
		{"/save <filename>", "Save conversation to JSON file"},
		{"/load <filename>", "Load conversation from JSON file"},
	}

	for _, cmd := range commands {
		ui.Print("  %s%s\n", cmdStyle.Render(cmd.cmd), descStyle.Render(cmd.desc))
	}
	ui.Println("")
}

// ShowModels displays available models
func (ui *BaseUI) ShowModels(models []string, current, planner, subAgent, summariser string) {
	ui.Println("\n%s", ui.theme.StyleBold.Render("Available Models:"))

	for _, model := range models {
		var tags []string
		if model == current {
			tags = append(tags, "default")
		}
		if model == planner {
			tags = append(tags, "planner")
		}
		if model == subAgent {
			tags = append(tags, "sub-agent")
		}
		if model == summariser {
			tags = append(tags, "summariser")
		}

		if len(tags) > 0 {
			tagStr := strings.Join(tags, ", ")
			ui.Println("  %s %s",
				ui.theme.StyleGreen.Render("●"),
				ui.theme.StyleGreen.Bold(true).Render(model+" ("+tagStr+")"))
		} else {
			ui.Println("  %s %s", ui.theme.StyleDim.Render("○"), ui.theme.StyleDim.Render(model))
		}
	}
	ui.Println("")
}

// ShowTools displays available tools
func (ui *BaseUI) ShowTools(tools []ToolInfo) {
	ui.Println("\n%s", ui.theme.StyleBold.Render("Available Tools:"))

	// Use lipgloss-width-aware padding so columns align correctly
	nameStyle := ui.theme.StyleYellow.Width(20)

	for _, tool := range tools {
		permStyle := ui.theme.StyleYellow
		permIcon := "◆"
		if tool.Permission == "never_ask" {
			permStyle = ui.theme.StyleGreen
			permIcon = "✓"
		} else if tool.Permission == "always_ask" {
			permStyle = ui.theme.StyleRed
			permIcon = "!"
		}

		ui.Print("  %s %s %s\n",
			nameStyle.Render(tool.Name),
			ui.theme.StyleDim.Render(tool.Description),
			permStyle.Render("["+permIcon+" "+tool.Permission+"]"))
	}
	ui.Println("")
}

// ShowContext displays conversation context
func (ui *BaseUI) ShowContext(context string) {
	ui.Println("\n%s", ui.theme.StyleBold.Render("Conversation Context:"))
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

// SetCompleter wires a Tab-completion callback into the underlying input reader.
// BaseUI satisfies the ui.Completer interface.
func (ui *BaseUI) SetCompleter(fn func(line string) []Completion) {
	ui.reader.SetCompleter(fn)
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
	ui.reader.SetTheme(theme)
}

// DisableColors disables all colors in the theme
func (ui *BaseUI) DisableColors() {
	emptyStyle := lipgloss.NewStyle()
	ui.theme = Theme{
		StyleReset:  emptyStyle,
		StyleRed:    emptyStyle,
		StyleGreen:  emptyStyle,
		StyleYellow: emptyStyle,
		StyleBlue:   emptyStyle,
		StylePurple: emptyStyle,
		StyleCyan:   emptyStyle,
		StyleBold:   emptyStyle,
		StyleDim:    emptyStyle,

		IconSuccess: ui.theme.IconSuccess,
		IconError:   ui.theme.IconError,
		IconWarning: ui.theme.IconWarning,
		IconInfo:    ui.theme.IconInfo,
		IconPrompt:  ui.theme.IconPrompt,
	}
}
