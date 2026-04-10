package ui

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
	"github.com/charmbracelet/lipgloss"
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
		ui.Println("%s", ui.theme.StyleCyan.Render(line))
		time.Sleep(100 * time.Millisecond)

		// Add sparkles on last line
		if i == len(banner)-1 {
			ui.Print("\n%s", ui.theme.StyleYellow.Render("✨ "))
			time.Sleep(50 * time.Millisecond)
			ui.Print("AI-Powered ")
			time.Sleep(50 * time.Millisecond)
			ui.Print("Coding ")
			time.Sleep(50 * time.Millisecond)
			ui.Print("Assistant ")
			time.Sleep(50 * time.Millisecond)
			ui.Println("✨")
		}
	}

	// Animated gradient separator
	colors := []lipgloss.Style{ui.theme.StylePurple, ui.theme.StyleBlue, ui.theme.StyleCyan, ui.theme.StyleBlue, ui.theme.StylePurple}
	separatorChars := []string{"═", "╪", "╬", "╪", "═"}

	for i := 0; i < ui.width; i++ {
		colorIndex := (i * len(colors)) / ui.width
		charIndex := (i * len(separatorChars)) / ui.width
		if charIndex >= len(separatorChars) {
			charIndex = len(separatorChars) - 1
		}
		ui.Print("%s", colors[colorIndex].Render(separatorChars[charIndex]))
		if i%3 == 0 {
			time.Sleep(3 * time.Millisecond)
		}
	}
	ui.Println("")

	// Subtitle
	subtitle := ui.theme.StyleBold.Render("✨ AI-Powered Coding Assistant ✨")
	ui.Println("%s", subtitle)
}

// ShowWelcome displays an enhanced welcome message
func (ui *FancyUI) ShowWelcome(model, ollamaURL string, contextEnabled bool) {
	// Type-writer animation on the welcome text as a whole styled string
	welcome := "Welcome to Codezilla!"
	for i := 1; i <= len(welcome); i++ {
		ui.Print("\r%s", ui.theme.StyleBold.Foreground(lipgloss.Color("#FFD700")).Render(welcome[:i]))
		time.Sleep(28 * time.Millisecond)
	}
	ui.Println("")

	// Status info panel
	keyStyle := ui.theme.StyleDim.Width(22)

	ui.Print("%s %s\n", keyStyle.Render("  🧠 Model:"), ui.theme.StyleYellow.Render(model))
	ui.Print("%s %s\n", keyStyle.Render("  🔌 Provider:"), ui.theme.StyleDim.Render(ollamaURL))

	contextVal := ui.theme.StyleRed.Render("Disabled")
	if contextEnabled {
		contextVal = ui.theme.StyleGreen.Render("Enabled ✓")
	}
	ui.Print("%s %s\n", keyStyle.Render("  💾 Context:"), contextVal)

	cwd, _ := os.Getwd()
	ui.Print("%s %s\n", keyStyle.Render("  📁 Directory:"), ui.theme.StyleCyan.Render(cwd))

	ui.Print("\n  Type %s for commands or just start chatting.\n\n",
		ui.theme.StyleYellow.Render("/help"))
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
		ui.Print("\r%s", ui.theme.StyleCyan.Render(frames[0]))
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
				ui.Print("\r%s", ui.theme.StyleCyan.Render(frames[i%len(frames)]))
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
	ui.Println("%s", ui.theme.StyleGreen.Render("🤖 Assistant:"))

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
// It is careful to skip any content inside fenced code blocks (``` ... ```),
// which prevents the spacing logic from corrupting code blocks placed inside
// table cells by the LLM.
func addTableSpacing(md string) string {
	lines := strings.Split(md, "\n")
	var out []string
	inFence := false

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

		// Track fenced code block entry/exit so we never mutate their contents.
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
		}

		out = append(out, line)

		// Only add padding outside of fenced code blocks.
		if !inFence && isTableRow(line) {
			if isSeparator(line) {
				continue
			}
			if i+1 < len(lines) && isSeparator(lines[i+1]) {
				continue
			}

			// It's a data row — append an empty row for padding.
			pipes := strings.Count(strings.TrimSpace(line), "|")
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
	ui.Println("%s", ui.theme.StyleGreen.Render("🤖 Assistant:"))
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
	ui.Print("%s %s", ui.theme.StyleGreen.Render(ui.theme.IconSuccess), msg)

	// Add a subtle animation
	for i := 0; i < 3; i++ {
		time.Sleep(50 * time.Millisecond)
		ui.Print(".")
	}
	ui.Println(" %s", ui.theme.StyleGreen.Render("✅"))
	ui.writer.Flush()
}

// Error shows error with emphasis
func (ui *FancyUI) Error(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.Println("%s %s", ui.theme.StyleRed.Render(ui.theme.IconError), msg)
	// Flash effect
	time.Sleep(100 * time.Millisecond)
}
