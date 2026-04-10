package ui

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
)

// FancyUI implements a fancy UI with animations and extra visual elements
type FancyUI struct {
	*BaseUI   // Embed BaseUI and override specific methods
	spinnerWg sync.WaitGroup
}

// NewFancyUI creates a fancy UI implementation
func NewFancyUI(historyFile string) (UI, error) {
	baseUI, err := NewBaseUI(historyFile)
	if err != nil {
		return nil, err
	}

	// Cast to get access to the concrete type
	tui := baseUI.(*BaseUI)

	// Customize the theme with more fancy elements
	theme := tui.GetTheme()
	theme.IconSuccess = "✨"
	theme.IconError = "💥"
	theme.IconWarning = "🔥"
	theme.IconInfo = "💡"
	theme.IconPrompt = "🤖"
	tui.SetTheme(theme)

	return &FancyUI{BaseUI: tui}, nil
}

// ShowBanner displays an animated banner
func (ui *FancyUI) ShowBanner() {
	// Animated banner reveal
	banner := []string{
		"   ____          _           _ _ _       ",
		"  / ___|___   __| | ___ ____(_) | | __ _ ",
		" | |   / _ \\ / _` |/ _ \\_  /| | | |/ _` |",
		" | |__| (_) | (_| |  __// / | | | | (_| |",
		"  \\____\\___/ \\__,_|\\___/___|_|_|_|\\__,_|",
	}

	// Clear and animate
	ui.Clear()

	// Reveal banner line by line
	for i, line := range banner {
		ui.Print("%s%s%s\n", ui.theme.ColorCyan, line, ui.theme.ColorReset)
		time.Sleep(100 * time.Millisecond)

		// Add sparkles on last line
		if i == len(banner)-1 {
			ui.Print("\n%s✨ ", ui.theme.ColorYellow)
			time.Sleep(50 * time.Millisecond)
			ui.Print("AI-Powered ")
			time.Sleep(50 * time.Millisecond)
			ui.Print("Coding ")
			time.Sleep(50 * time.Millisecond)
			ui.Print("Assistant ")
			time.Sleep(50 * time.Millisecond)
			ui.Println("✨%s", ui.theme.ColorReset)
		}
	}

	// Animated gradient separator
	colors := []string{ui.theme.ColorPurple, ui.theme.ColorBlue, ui.theme.ColorCyan, ui.theme.ColorBlue, ui.theme.ColorPurple}
	separatorChars := []string{"═", "╪", "╬", "╪", "═"}

	for i := 0; i < ui.width; i++ {
		colorIndex := (i * len(colors)) / ui.width
		charIndex := (i * len(separatorChars)) / ui.width
		if charIndex >= len(separatorChars) {
			charIndex = len(separatorChars) - 1
		}
		ui.Print("%s%s", colors[colorIndex], separatorChars[charIndex])
		if i%3 == 0 {
			time.Sleep(5 * time.Millisecond)
		}
	}
	ui.Print("%s\n", ui.theme.ColorReset)
}

// ShowWelcome displays an enhanced welcome message
func (ui *FancyUI) ShowWelcome(model, ollamaURL string, contextEnabled bool) {
	// Animated welcome
	welcome := "Welcome to Codezilla!"
	ui.Print("%s", ui.theme.ColorBold)
	for _, char := range welcome {
		ui.Print("%c", char)
		time.Sleep(30 * time.Millisecond)
	}
	ui.Println("%s", ui.theme.ColorReset)

	ui.Print("Type %s/help%s for commands or start chatting.\n",
		ui.theme.ColorYellow, ui.theme.ColorReset)

	// Model info with icon
	ui.Print("🧠 Model: %s%s%s\n",
		ui.theme.ColorYellow, model, ui.theme.ColorReset)

	// Connection info with icon
	ui.Print("🔌 Ollama: %s%s%s\n",
		ui.theme.ColorDim, ollamaURL, ui.theme.ColorReset)

	// Context status with appropriate icon
	if contextEnabled {
		ui.Print("💾 Context: %sEnabled%s %s\n",
			ui.theme.ColorGreen, ui.theme.ColorReset, "✓")
	} else {
		ui.Print("💾 Context: %sDisabled%s\n",
			ui.theme.ColorDim, ui.theme.ColorReset)
	}

	// Working directory info
	cwd, _ := os.Getwd()
	ui.Print("📁 Working Directory: %s%s%s\n",
		ui.theme.ColorCyan, cwd, ui.theme.ColorReset)

	ui.Println("")
}

// ShowThinking shows an enhanced thinking animation
func (ui *FancyUI) ShowThinking() {
	ui.spinnerMutex.Lock()
	if ui.spinnerStop != nil {
		ui.spinnerMutex.Unlock()
		return
	}

	ui.spinnerStop = make(chan bool, 1) // Buffered channel
	ui.spinnerWg.Add(1)
	ui.spinnerMutex.Unlock()

	go func() {
		defer ui.spinnerWg.Done()

		frames := []string{
			"🤔 Thinking",
			"🤔 Thinking.",
			"🤔 Thinking..",
			"🤔 Thinking...",
			"💭 Processing",
			"💭 Processing.",
			"💭 Processing..",
			"💭 Processing...",
			"🧠 Analyzing",
			"🧠 Analyzing.",
			"🧠 Analyzing..",
			"🧠 Analyzing...",
		}

		i := 0
		ticker := time.NewTicker(150 * time.Millisecond)
		defer ticker.Stop()

		// Print initial frame
		ui.Print("\r%s%s%s",
			ui.theme.ColorCyan, frames[0], ui.theme.ColorReset)
		ui.writer.Flush()

		for {
			select {
			case <-ui.spinnerStop:
				// Clear the entire line
				ui.Print("\r\033[K") // ANSI escape code to clear line
				ui.writer.Flush()
				return
			case <-ticker.C:
				i++
				ui.Print("\r%s%s%s",
					ui.theme.ColorCyan, frames[i%len(frames)], ui.theme.ColorReset)
				ui.writer.Flush()
			}
		}
	}()
}

// HideThinking hides the thinking indicator and clears the line
func (ui *FancyUI) HideThinking() {
	ui.spinnerMutex.Lock()

	if ui.spinnerStop != nil {
		// Send stop signal
		select {
		case ui.spinnerStop <- true:
		default:
		}
		close(ui.spinnerStop)
		ui.spinnerStop = nil
		ui.spinnerMutex.Unlock()

		// Wait for the goroutine to finish cleaning up
		ui.spinnerWg.Wait()

		// Extra insurance - clear the line again
		ui.Print("\r\033[K")
		ui.writer.Flush()
	} else {
		ui.spinnerMutex.Unlock()
	}
}

// ShowResponse displays response formatted with Glamour
func (ui *FancyUI) ShowResponse(response string) {
	// Move to a new line first to avoid overwriting issues
	ui.Println("")
	ui.Println("%s🤖 Assistant:%s", ui.theme.ColorGreen, ui.theme.ColorReset)

	// Pre-process markdown to add spacing between table rows
	response = addTableSpacing(response)

	// Ensure we have a reasonable width for terminal wrapping
	width := ui.width - 4
	if width < 40 {
		width = 80 // fallback
	}

	renderer, err := glamour.NewTermRenderer(
		glamour.WithStyles(styles.DarkStyleConfig),
		glamour.WithWordWrap(width),
	)

	var out string
	if err == nil {
		out, err = renderer.Render(response)
	}

	if err == nil {
		fmt.Fprint(ui.writer, out)
	} else {
		// Fallback if Glamour fails
		fmt.Fprint(ui.writer, response)
		ui.Println("")
	}

	ui.Println("")

	// Ensure the buffer is flushed so the prompt appears
	ui.writer.Flush()
}

// addTableSpacing pre-processes markdown tables by inserting an empty
// padding row between data rows for better readability.
func addTableSpacing(md string) string {
	lines := strings.Split(md, "\n")
	var out []string

	isSeparator := func(s string) bool {
		s = strings.TrimSpace(s)
		if !strings.HasPrefix(s, "|") || !strings.HasSuffix(s, "|") {
			return false
		}
		cleaned := strings.ReplaceAll(s, " ", "")
		for _, c := range cleaned {
			if c != '|' && c != '-' && c != ':' {
				return false
			}
		}
		return true
	}

	isTableRow := func(s string) bool {
		s = strings.TrimSpace(s)
		return strings.HasPrefix(s, "|") && strings.HasSuffix(s, "|")
	}

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		out = append(out, line)

		if isTableRow(line) {
			if isSeparator(line) {
				continue
			}
			if i+1 < len(lines) && isSeparator(lines[i+1]) {
				continue
			}

			// It's a data row! Append an empty row for padding.
			trimmed := strings.TrimSpace(line)
			pipes := strings.Count(trimmed, "|")
			if pipes >= 2 {
				emptyCols := strings.Repeat("   |", pipes-1)
				out = append(out, "|"+emptyCols)
			}
		}
	}
	return strings.Join(out, "\n")
}

// ShowResponseStream streams tokens as they arrive with a header.
func (ui *FancyUI) ShowResponseStream(ch <-chan string) {
	ui.Println("")
	ui.Println("%s🤖 Assistant:%s", ui.theme.ColorGreen, ui.theme.ColorReset)
	for token := range ch {
		fmt.Fprint(ui.writer, token)
		ui.writer.Flush()
	}
	ui.Println("")
	ui.writer.Flush()
}

// Success shows success with animation
func (ui *FancyUI) Success(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Print("%s%s %s", ui.theme.ColorGreen, ui.theme.IconSuccess, msg)

	// Add a subtle animation
	for i := 0; i < 3; i++ {
		time.Sleep(50 * time.Millisecond)
		ui.Print(".")
	}
	ui.Println(" %s✅%s", ui.theme.ColorGreen, ui.theme.ColorReset)
	ui.writer.Flush()
}

// Error shows error with emphasis
func (ui *FancyUI) Error(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Println("%s%s %s%s", ui.theme.ColorRed, ui.theme.IconError, msg, ui.theme.ColorReset)
	// Flash effect
	time.Sleep(100 * time.Millisecond)
}
