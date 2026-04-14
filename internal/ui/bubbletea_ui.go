package ui

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
	"github.com/charmbracelet/lipgloss"
)

// ──────────────────────────────────────────────────────────────────────────────
// Message types for thread-safe communication with the BubbleTea program.
// Any goroutine can send these via tea.Program.Send().
// ──────────────────────────────────────────────────────────────────────────────

type appendOutputMsg struct{ text string }
type setSpinnerMsg struct {
	active bool
	label  string
}
type setTokenUsageMsg struct{ usage string }
type enableInputMsg struct{}
type setModelMsg struct{ model string }
type appQuitMsg struct{ err error }

// ──────────────────────────────────────────────────────────────────────────────
// appModel — the core BubbleTea Model
// ──────────────────────────────────────────────────────────────────────────────

type appModel struct {
	viewport viewport.Model
	input    textarea.Model
	spinner  spinner.Model

	// Layout
	width  int
	height int
	ready  bool // set after first WindowSizeMsg

	// Content buffer for the viewport
	outputLines []string

	// Spinner / status bar state
	spinnerActive bool
	spinnerLabel  string
	tokenUsage    string
	activeModel   string
	taskStart     time.Time
	stepStart     time.Time

	// Input state
	inputEnabled bool
	inputChan    chan string // signals ReadLine
	eofChan      chan struct{}

	// History
	history      []string
	historyIndex int
	savedInput   string

	// Completer
	completer func(line string) []Completion

	// Theme
	theme Theme

	// Prompt string
	prompt string
}

func newAppModel(theme Theme, prompt string) appModel {
	ti := textarea.New()
	ti.Placeholder = prompt
	// Textarea uses slightly different styling properties
	ti.Prompt = " " // We don't want a default internal prompt since we'll draw it ourselves or use placeholder
	
	// Ensure transparent background for clean integration
	ti.FocusedStyle.Prompt = lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#2E3C64", Dark: "#7AA2F7"}).Background(lipgloss.NoColor{})
	ti.FocusedStyle.CursorLine = lipgloss.NewStyle().Background(lipgloss.NoColor{})
	ti.FocusedStyle.Base = lipgloss.NewStyle().Background(lipgloss.NoColor{})
	ti.FocusedStyle.Text = lipgloss.NewStyle().Background(lipgloss.NoColor{})
	ti.FocusedStyle.Placeholder = lipgloss.NewStyle().Background(lipgloss.NoColor{})
	ti.BlurredStyle.Base = lipgloss.NewStyle().Background(lipgloss.NoColor{})
	ti.BlurredStyle.CursorLine = lipgloss.NewStyle().Background(lipgloss.NoColor{})

	ti.ShowLineNumbers = false
	ti.CharLimit = 20000
	ti.Focus()

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#7DCFFF", Dark: "#7DCFFF"})

	return appModel{
		input:        ti,
		spinner:      sp,
		inputChan:    make(chan string, 1),
		eofChan:      make(chan struct{}, 1),
		theme:        theme,
		prompt:       prompt,
		inputEnabled: true,
		history:      nil,
		historyIndex: 0,
		taskStart:    time.Now(),
		stepStart:    time.Now(),
	}
}

func (m appModel) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, m.spinner.Tick)
}

func (m appModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m = m.resizeViews(true)

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			if !m.inputEnabled {
				// Task running — Ctrl+C is handled by the signal handler in app.go.
				// We do nothing here so BubbleTea doesn't quit the program.
				return m, nil
			}
			// At prompt — signal EOF to ReadLine
			select {
			case m.eofChan <- struct{}{}:
			default:
			}
			return m, nil

		case tea.KeyEsc:
			if !m.inputEnabled {
				// ESC during task: send interrupt (same as Ctrl+C during task)
				// Handled by signal handler, nothing to do here
			}
			return m, nil

		case tea.KeyEnter:
			if m.inputEnabled {
				if msg.Alt {
					// Give back to textarea to insert newline
					break
				}
				value := strings.TrimSpace(m.input.Value())
				m.input.Reset()

				// Add to history
				if value != "" {
					m.history = append(m.history, value)
				}
				m.historyIndex = len(m.history)
				m.savedInput = ""

				// Echo the input to the viewport
				m.appendOutput(m.prompt + value)

				// Send to ReadLine
				select {
				case m.inputChan <- value:
				default:
				}

				m.inputEnabled = false
				m.taskStart = time.Now() // Record exact moment user hands off control
				m.stepStart = time.Now() // Also reset step timer
				m = m.resizeViews(false)
				return m, nil
			}

		case tea.KeyUp:
			if m.inputEnabled && len(m.history) > 0 {
				isNavigating := m.historyIndex < len(m.history)
				hasTyped := strings.TrimSpace(m.input.Value()) != ""

				if !hasTyped || isNavigating {
					if m.historyIndex == len(m.history) {
						m.savedInput = m.input.Value()
					}
					if m.historyIndex > 0 {
						m.historyIndex--
						m.input.SetValue(m.history[m.historyIndex])
						m.input.CursorEnd()
					}
					m = m.resizeViews(false)
					return m, nil
				}
			}

		case tea.KeyDown:
			if m.inputEnabled {
				isNavigating := m.historyIndex < len(m.history)

				if isNavigating {
					if m.historyIndex < len(m.history)-1 {
						m.historyIndex++
						m.input.SetValue(m.history[m.historyIndex])
						m.input.CursorEnd()
					} else if m.historyIndex == len(m.history)-1 {
						m.historyIndex = len(m.history)
						m.input.SetValue(m.savedInput)
						m.input.CursorEnd()
					}
					m = m.resizeViews(false)
					return m, nil
				}
			}
		}

		// Tab completion: update suggestions before delegating
		if m.inputEnabled && m.completer != nil {
			val := m.input.Value()
			// textarea handles its own keys, so we don't have SetSuggestions right now.
			// Let's just bypass autocomplete suggestions for textarea since it doesn't support them natively like textinput maybe.
			_ = val
		}

	case appendOutputMsg:
		m.appendOutput(msg.text)
		return m, nil

	case setSpinnerMsg:
		if m.spinnerLabel != msg.label || (!m.spinnerActive && msg.active) {
			m.stepStart = time.Now()
		}
		
		m.spinnerActive = msg.active
		m.spinnerLabel = msg.label
		if msg.active {
			// Do not reset taskStart here, so timer reflects total duration since user input
			return m, m.spinner.Tick
		}
		return m, nil

	case setTokenUsageMsg:
		m.tokenUsage = msg.usage
		return m, nil

	case enableInputMsg:
		m.inputEnabled = true
		m.historyIndex = len(m.history)
		m.input.Reset()
		cmds = append(cmds, m.input.Focus())
		return m, tea.Batch(cmds...)

	case setModelMsg:
		m.activeModel = msg.model
		return m, nil

	case spinner.TickMsg:
		if m.spinnerActive {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}
	case appQuitMsg:
		// App finished (or error) — quit the tea program
		return m, tea.Quit

	}

	// Delegate to sub-models
	if m.inputEnabled {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		cmds = append(cmds, cmd)
		// Resize dynamically if the user added/removed lines
		m = m.resizeViews(false)
	}

	// Viewport always updates (scroll support)
	{
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		cmds = append(cmds, cmd)
	}

	return m, tea.Batch(cmds...)
}

func (m *appModel) appendOutput(text string) {
	lines := strings.Split(text, "\n")
	if len(lines) == 0 {
		return
	}

	if len(m.outputLines) > 0 {
		// Append the first segment to the end of the existing last line
		m.outputLines[len(m.outputLines)-1] += lines[0]
	} else {
		m.outputLines = append(m.outputLines, lines[0])
	}
	
	// Any subsequent segments are new lines
	if len(lines) > 1 {
		m.outputLines = append(m.outputLines, lines[1:]...)
	}

	if m.ready {
		// Wrap text to terminal width to prevent truncation and format cleanly
		width := m.width
		if width > 0 {
			// Sub margin for perfect wrapping
			width -= 2 
		}
		
		content := strings.Join(m.outputLines, "\n")
		// Fast ANSI-aware wrap
		wrapped := lipgloss.NewStyle().Width(width).Render(content)
		m.viewport.SetContent(wrapped)
		m.viewport.GotoBottom()
	}
}

func (m appModel) View() string {
	if !m.ready {
		return "Loading..."
	}

	// Viewport (scrollable output)
	vpView := m.viewport.View()

	// Separator
	sepWidth := m.width
	if sepWidth > 1 {
		sepWidth--
	}
	separator := lipgloss.NewStyle().Faint(true).Render(strings.Repeat("─", sepWidth))

	// Input line
	var inputView string
	baseStyle := lipgloss.NewStyle().Border(lipgloss.NormalBorder(), false, false, false, true).PaddingLeft(1)
	if m.inputEnabled {
		inputView = baseStyle.BorderForeground(lipgloss.Color("240")).Render(m.input.View())
	} else {
		// During processing, keep the box structure but dim it
		inputView = baseStyle.BorderForeground(lipgloss.Color("236")).Faint(true).Render(m.input.View())
	}

	// Status bar
	statusContent := m.buildStatusBar()

	return lipgloss.JoinVertical(lipgloss.Left,
		vpView,
		separator,
		inputView,
		separator,
		statusContent,
	)
}

func (m appModel) getRequiredInputHeight() int {
	text := m.input.Value()
	lines := strings.Split(text, "\n")
	height := 0
	width := m.width - 4 // border and padding margins
	if width < 1 {
		width = 1
	}

	for _, line := range lines {
		l := lipgloss.Width(line)
		if l == 0 {
			height += 1
		} else {
			height += (l + width - 1) / width
		}
	}
	
	if height < 1 {
		height = 1
	}
	maxHeight := m.height / 3 // cap at 33% of window height
	if maxHeight < 4 {
		maxHeight = 4
	}
	if height > maxHeight {
		height = maxHeight
	}
	return height
}

func (m appModel) resizeViews(forceReflow bool) appModel {
	m.input.SetWidth(m.width)
	m.input.SetHeight(m.getRequiredInputHeight())

	headerHeight := 0
	inputHeight := m.input.Height()
	if inputHeight < 1 {
		inputHeight = 1
	}

	// Calculate visible wrapper height (adding padding/margins)
	// Border top+bottom + padding
	inputWrapperHeight := inputHeight + 0 // our border style doesn't add vertical height currently
	statusHeight := 2 // separator + status line
	
	vpHeight := m.height - headerHeight - inputWrapperHeight - statusHeight
	if vpHeight < 1 {
		vpHeight = 1
	}

	wrapped := lipgloss.NewStyle().Width(m.width - 2).Render(strings.Join(m.outputLines, "\n"))

	if !m.ready {
		m.viewport = viewport.New(m.width, vpHeight)
		m.viewport.SetContent(wrapped)
		m.viewport.GotoBottom()
		m.ready = true
	} else {
		oldHeight := m.viewport.Height
		m.viewport.Width = m.width
		m.viewport.Height = vpHeight
		if forceReflow {
			m.viewport.SetContent(wrapped)
		} else if oldHeight != vpHeight {
			// If height changed, we might need to jump back to bottom if we were pinned
			m.viewport.GotoBottom()
		}
	}
	return m
}

func (m appModel) buildStatusBar() string {
	model := m.activeModel
	if model == "" {
		model = "..."
	}

	if !m.inputEnabled {
		label := m.spinnerLabel
		if label == "" {
			label = "streaming"
		}

		// Truncate overly long labels to prevent unwanted line wrapping that breaks layout math
		maxLabel := m.width - 50 // Reserve ~50 chars for model, tokens, time
		if maxLabel < 15 {
			maxLabel = 15
		}
		// Strip newlines or ansi just in case, though label should be clean
		label = strings.ReplaceAll(label, "\n", " ")
		if len(label) > maxLabel {
			label = label[:maxLabel-3] + "..."
		}

		totalElapsed := time.Since(m.taskStart).Round(time.Second)
		stepElapsed := time.Since(m.stepStart).Round(time.Second)

		statusIcon := "💬"
		if m.spinnerActive {
			statusIcon = m.spinner.View()
		}

		msg := fmt.Sprintf("%s 🧠 %s  |  ⏳ %s (%s)  |  ⏱️ %s",
			statusIcon, model, label, stepElapsed, totalElapsed)
		if m.tokenUsage != "" {
			msg += fmt.Sprintf("  |  📊 %s", m.tokenUsage)
		}
		return m.theme.StyleCyan.Render(msg)
	}

	// Idle status
	tokenInfo := m.tokenUsage
	if tokenInfo == "" {
		tokenInfo = "0 tokens"
	}
	msg := fmt.Sprintf("🧠 %s  |  ⏱️ idle  |  📊 %s", model, tokenInfo)
	return lipgloss.NewStyle().Faint(true).Render(msg)
}

// ──────────────────────────────────────────────────────────────────────────────
// BubbleTeaUI — implements ui.UI interface
// ──────────────────────────────────────────────────────────────────────────────

// BubbleTeaUI wraps a BubbleTea program and satisfies the ui.UI interface.
type BubbleTeaUI struct {
	program *tea.Program
	model   *appModel // pointer to initial model (program owns a copy after Run)

	theme        Theme
	currentModel string
	historyFile  string

	// ReadLine synchronization — these channels are in the appModel
	inputChan chan string
	eofChan   chan struct{}
}

// TUIRunner is an optional interface satisfied by BubbleTeaUI.
// main.go detects this and uses it to run the tea program on the main goroutine.
type TUIRunner interface {
	// RunTUI starts the BubbleTea program on the calling goroutine and runs
	// appFn concurrently. Blocks until the program exits.
	RunTUI(ctx context.Context, appFn func(context.Context) error) error
}

// NewBubbleTeaUI creates a new BubbleTea-based UI implementation.
func NewBubbleTeaUI(historyFile string) (UI, error) {
	theme := ThemeTokyoNight()
	// Use fancy icons
	theme.IconSuccess = "✨"
	theme.IconError = "💥"
	theme.IconWarning = "🔥"
	theme.IconInfo = "💡"
	theme.IconPrompt = "🤖"

	prompt := "codezilla " + theme.IconPrompt + " "

	model := newAppModel(theme, prompt)

	// Load history from file
	if historyFile != "" {
		if entries, err := loadHistoryFile(historyFile); err == nil {
			model.history = entries
			model.historyIndex = len(entries)
		}
	}

	ui := &BubbleTeaUI{
		theme:       theme,
		historyFile: historyFile,
		inputChan:   model.inputChan,
		eofChan:     model.eofChan,
		model:       &model,
	}

	return ui, nil
}

// RunTUI starts the BubbleTea program on the calling goroutine and runs
// appFn concurrently. Blocks until the tea program exits.
func (ui *BubbleTeaUI) RunTUI(ctx context.Context, appFn func(context.Context) error) error {
	ui.program = tea.NewProgram(
		*ui.model,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	// Run the application logic in a background goroutine.
	// When it finishes, send appQuitMsg to shut down the tea program.
	go func() {
		err := appFn(ctx)
		// Save history before quitting
		if ui.historyFile != "" && ui.model != nil {
			_ = saveHistoryFile(ui.historyFile, ui.model.history)
		}
		if ui.program != nil {
			ui.program.Send(appQuitMsg{err: err})
		}
	}()

	// Run on the calling goroutine (BubbleTea requires this)
	_, err := ui.program.Run()
	return err
}

// ──────────────────────────────────────────────────────────────────────────────
// ui.UI interface implementation
// ──────────────────────────────────────────────────────────────────────────────

func (ui *BubbleTeaUI) Clear() {
	if ui.program != nil {
		ui.program.Send(appendOutputMsg{text: ""})
	}
}



func (ui *BubbleTeaUI) ShowPrompt() string {
	return "codezilla " + ui.theme.IconPrompt + " "
}

func (ui *BubbleTeaUI) UpdateModel(model string) {
	ui.currentModel = model
	if ui.program != nil {
		ui.program.Send(setModelMsg{model: model})
	}
}

// ── Output methods ───────────────────────────────────────────────────────────

func (ui *BubbleTeaUI) Print(format string, args ...interface{}) {
	text := fmt.Sprintf(format, args...)
	ui.appendToViewport(text)
}

func (ui *BubbleTeaUI) Println(format string, args ...interface{}) {
	text := fmt.Sprintf(format, args...)
	ui.appendToViewport(text + "\n")
}

func (ui *BubbleTeaUI) Success(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.appendToViewport(ui.theme.IconSuccess + " " + ui.theme.StyleGreen.Render(msg) + "\n")
}

func (ui *BubbleTeaUI) Error(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.appendToViewport(ui.theme.IconError + " " + ui.theme.StyleRed.Render(msg) + "\n")
}

func (ui *BubbleTeaUI) Warning(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.appendToViewport(ui.theme.IconWarning + " " + ui.theme.StyleYellow.Render(msg) + "\n")
}

func (ui *BubbleTeaUI) Info(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ui.appendToViewport(ui.theme.IconInfo + " " + msg + "\n")
}

// ── Thinking / Spinner ───────────────────────────────────────────────────────

func (ui *BubbleTeaUI) ShowThinking() {
	if ui.program != nil {
		ui.program.Send(setSpinnerMsg{active: true, label: ""})
	}
}

func (ui *BubbleTeaUI) HideThinking() {
	if ui.program != nil {
		ui.program.Send(setSpinnerMsg{active: false, label: ""})
	}
}

func (ui *BubbleTeaUI) RestartThinking() {
	if ui.program != nil {
		ui.program.Send(setSpinnerMsg{active: true, label: ""})
	}
}

func (ui *BubbleTeaUI) UpdateThinkingStatus(label string) {
	if ui.program != nil {
		ui.program.Send(setSpinnerMsg{active: true, label: label})
	}
}

func (ui *BubbleTeaUI) UpdateTokenUsage(usage string) {
	if ui.program != nil {
		ui.program.Send(setTokenUsageMsg{usage: usage})
	}
}

func (ui *BubbleTeaUI) SetPromptFooter(fn func() string) {
	// No-op: BubbleTea handles the footer natively in View()
}

// ── Response display ─────────────────────────────────────────────────────────

func (ui *BubbleTeaUI) ShowResponse(response string) {
	// Render with glamour for markdown formatting
	width := 80 // default
	if ui.model != nil && ui.model.width > 0 {
		width = ui.model.width - 4
	}

	renderer, err := glamour.NewTermRenderer(
		glamour.WithStyles(styles.DarkStyleConfig),
		glamour.WithWordWrap(width),
	)
	if err != nil {
		ui.appendToViewport("\n" + response + "\n")
		return
	}

	rendered, err := renderer.Render(response)
	if err != nil {
		ui.appendToViewport("\n" + response + "\n")
		return
	}

	ui.appendToViewport("\n" + ui.theme.StyleGreen.Render("🤖 Assistant:") + "\n" + rendered)
}

func (ui *BubbleTeaUI) ShowResponseStream(ch <-chan string) {
	ui.appendToViewport("\n" + ui.theme.StyleGreen.Render("🤖 Assistant:") + "\n")
	for token := range ch {
		ui.appendToViewport(token)
	}
	ui.appendToViewport("\n")
}

func (ui *BubbleTeaUI) ShowCode(language, code string) {
	ui.appendToViewport(ui.theme.StylePurple.Render("```"+language) + "\n")
	ui.appendToViewport(code)
	if !strings.HasSuffix(code, "\n") {
		ui.appendToViewport("\n")
	}
	ui.appendToViewport(ui.theme.StylePurple.Render("```") + "\n")
}

// ── Startup & Banners ────────────────────────────────────────────────────────

func (ui *BubbleTeaUI) ShowBanner() {
	banner := []string{
		"   ____          _           _ _ _       ",
		"  / ___|___   __| | ___ ____(_) | | __ _ ",
		" | |   / _ \\ / _` |/ _ \\_  /| | | |/ _` |",
		" | |__| (_) | (_| |  __// / | | | | (_| |",
		"  \\____\\___/ \\__,_|\\___/___|_|_|_|\\__,_|",
	}

	for i, line := range banner {
		ui.appendToViewport(ui.theme.StyleCyan.Render(line) + "\n")
		time.Sleep(100 * time.Millisecond)

		if i == len(banner)-1 {
			ui.appendToViewport("\n" + ui.theme.StyleYellow.Render("✨ ") + "AI-Powered Coding Assistant ✨\n")
		}
	}

	// Animated gradient separator
	colors := []lipgloss.Style{ui.theme.StylePurple, ui.theme.StyleBlue, ui.theme.StyleCyan, ui.theme.StyleBlue, ui.theme.StylePurple}
	separatorChars := []string{"═", "╪", "╬", "╪", "═"}

	var sep strings.Builder
	for i := 0; i < 80; i++ {
		colorIndex := (i * len(colors)) / 80
		charIndex := (i * len(separatorChars)) / 80
		if charIndex >= len(separatorChars) {
			charIndex = len(separatorChars) - 1
		}
		sep.WriteString(colors[colorIndex].Render(separatorChars[charIndex]))
	}
	ui.appendToViewport(sep.String() + "\n")
	time.Sleep(100 * time.Millisecond)
}

func (ui *BubbleTeaUI) ShowWelcome(model, ollamaURL string, contextEnabled bool) {
	ui.ShowWelcomeWithModels(model, "", "", ollamaURL, contextEnabled)
}

func (ui *BubbleTeaUI) ShowWelcomeWithModels(defaultModel, fastModel, heavyModel, ollamaURL string, contextEnabled bool) {
	welcome := ui.theme.StyleBold.Foreground(lipgloss.Color("#FFD700")).Render("Welcome to Codezilla!")
	ui.appendToViewport(welcome + "\n\n")

	keyStyle := ui.theme.StyleDim.Width(16)

	ui.appendToViewport("  🧠 " + keyStyle.Render("Model:") + " " + ui.theme.StyleYellow.Render(defaultModel) + "\n")
	if fastModel != "" {
		ui.appendToViewport("  ⚡ " + keyStyle.Render("Fast:") + " " + ui.theme.StyleYellow.Render(fastModel) + "\n")
	}
	if heavyModel != "" {
		ui.appendToViewport("  🏋️  " + keyStyle.Render("Heavy:") + " " + ui.theme.StyleYellow.Render(heavyModel) + "\n")
	}
	ui.appendToViewport("  🔌 " + keyStyle.Render("Provider:") + " " + ui.theme.StyleDim.Render(ollamaURL) + "\n")

	contextVal := ui.theme.StyleRed.Render("Disabled")
	if contextEnabled {
		contextVal = ui.theme.StyleGreen.Render("Enabled ✓")
	}
	ui.appendToViewport("  💾 " + keyStyle.Render("Context:") + " " + contextVal + "\n")

	cwd, _ := os.Getwd()
	ui.appendToViewport("  📁 " + keyStyle.Render("Directory:") + " " + ui.theme.StyleCyan.Render(cwd) + "\n\n")

	ui.appendToViewport("  Type " + ui.theme.StyleYellow.Render("/help") + " for commands or just start chatting.\n\n")
}

// ── Structured displays ──────────────────────────────────────────────────────

func (ui *BubbleTeaUI) ShowHelp() {
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
		{"/tokens", "Show session token usage"},
	}

	out := "\n" + ui.theme.StyleBold.Render("Available Commands:") + "\n"
	for _, cmd := range commands {
		out += "  " + cmdStyle.Render(cmd.cmd) + descStyle.Render(cmd.desc) + "\n"
	}
	out += "\n"
	ui.appendToViewport(out)
}

func (ui *BubbleTeaUI) ShowModels(models []string, current, fast, heavy string) {
	out := "\n" + ui.theme.StyleBold.Render("Available Models:") + "\n"
	for _, model := range models {
		var tags []string
		if model == current {
			tags = append(tags, "default")
		}
		if model == fast {
			tags = append(tags, "fast")
		}
		if model == heavy {
			tags = append(tags, "heavy")
		}

		if len(tags) > 0 {
			tagStr := strings.Join(tags, ", ")
			out += fmt.Sprintf("  %s %s\n",
				ui.theme.StyleGreen.Render("●"),
				ui.theme.StyleGreen.Bold(true).Render(model+" ("+tagStr+")"))
		} else {
			out += fmt.Sprintf("  %s %s\n", ui.theme.StyleDim.Render("○"), ui.theme.StyleDim.Render(model))
		}
	}
	out += "\n"
	ui.appendToViewport(out)
}

func (ui *BubbleTeaUI) ShowTools(tools []ToolInfo) {
	nameStyle := ui.theme.StyleYellow.Width(20)
	out := "\n" + ui.theme.StyleBold.Render("Available Tools:") + "\n"
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
		out += fmt.Sprintf("  %s %s %s\n",
			nameStyle.Render(tool.Name),
			ui.theme.StyleDim.Render(tool.Description),
			permStyle.Render("["+permIcon+" "+tool.Permission+"]"))
	}
	out += "\n"
	ui.appendToViewport(out)
}

func (ui *BubbleTeaUI) ShowContext(context string) {
	out := "\n" + ui.theme.StyleBold.Render("Conversation Context:") + "\n"
	if context == "" {
		out += "No context stored\n"
	} else {
		out += context + "\n"
	}
	out += "\n"
	ui.appendToViewport(out)
}

// ── Input methods ────────────────────────────────────────────────────────────

func (ui *BubbleTeaUI) ReadLine() (string, error) {
	// Enable input
	if ui.program != nil {
		ui.program.Send(enableInputMsg{})
	}

	// Block until the user submits input or sends EOF
	select {
	case value := <-ui.inputChan:
		return value, nil
	case <-ui.eofChan:
		return "", io.EOF
	}
}

func (ui *BubbleTeaUI) ReadPassword(prompt string) (string, error) {
	// For now, fall back to the simple approach
	ui.appendToViewport(prompt)
	// TODO: implement password input mode in BubbleTea
	return ui.ReadLine()
}

func (ui *BubbleTeaUI) Confirm(prompt string) (bool, error) {
	for {
		ui.appendToViewport(prompt + " (y/n): ")
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

// ── Theme management ─────────────────────────────────────────────────────────

func (ui *BubbleTeaUI) GetTheme() Theme {
	return ui.theme
}

func (ui *BubbleTeaUI) SetTheme(theme Theme) {
	ui.theme = theme
}

func (ui *BubbleTeaUI) SetModel(model string) {
	ui.currentModel = model
	if ui.program != nil {
		ui.program.Send(setModelMsg{model: model})
	}
}

func (ui *BubbleTeaUI) DisableColors() {
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

// ── Completer / History interfaces ───────────────────────────────────────────

func (ui *BubbleTeaUI) SetCompleter(fn func(line string) []Completion) {
	if ui.model != nil {
		ui.model.completer = fn
	}
}

func (ui *BubbleTeaUI) GetHistory(n int) []string {
	if ui.model == nil {
		return nil
	}
	history := ui.model.history
	if n <= 0 || n >= len(history) {
		return history
	}
	return history[len(history)-n:]
}

func (ui *BubbleTeaUI) SearchHistory(query string) []string {
	if ui.model == nil {
		return nil
	}
	var results []string
	for _, h := range ui.model.history {
		if strings.Contains(h, query) {
			results = append(results, h)
		}
	}
	return results
}

func (ui *BubbleTeaUI) ClearHistory() error {
	if ui.model != nil {
		ui.model.history = nil
		ui.model.historyIndex = 0
	}
	if ui.historyFile != "" {
		return os.Remove(ui.historyFile)
	}
	return nil
}

// ── Helpers ──────────────────────────────────────────────────────────────────

func (ui *BubbleTeaUI) appendToViewport(text string) {
	if ui.program != nil {
		ui.program.Send(appendOutputMsg{text: text})
	}
}

// loadHistoryFile reads history entries from a file (one per line).
func loadHistoryFile(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var entries []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			entries = append(entries, line)
		}
	}
	return entries, scanner.Err()
}

// saveHistoryFile writes history entries to a file (one per line).
func saveHistoryFile(path string, entries []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, entry := range entries {
		fmt.Fprintln(f, entry)
	}
	return nil
}
