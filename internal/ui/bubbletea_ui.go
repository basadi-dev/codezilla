package ui

import (
	"bufio"
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// ──────────────────────────────────────────────────────────────────────────────
// Message types for thread-safe communication with the BubbleTea program.
// Any goroutine can send these via tea.Program.Send().
// ──────────────────────────────────────────────────────────────────────────────

type appendOutputMsg struct{ text string }
type flushOutputMsg struct{}
type setSpinnerMsg struct {
	active bool
	label  string
}
type setTokenUsageMsg struct{ usage string }
type enableInputMsg struct{}
type setModelMsg struct{ model string }
type clearViewportMsg struct{}
type clearCopyNoticeMsg struct{}
type appQuitMsg struct{ err error }

// WorkerStatus represents the visible state of a single multi-agent worker.
type WorkerStatus struct {
	WorkerID string
	Number   int           // 1-indexed task number for display
	Role     string
	Model    string        // LLM model this worker uses
	Label    string
	Done     bool
	HasError bool
	Started  time.Time
	Elapsed  time.Duration // frozen duration for completed tasks
}

// WorkerBlockInit carries the data needed to create a live block
// in the viewport for a parallel worker.
type WorkerBlockInit struct {
	WorkerID string
	Number   int
	Role     string
	Model    string
	Label    string
}

// BubbleTea message types for thread-safe worker block updates.
type updateWorkerStatusMsg struct{ status WorkerStatus }
type createWorkerBlockMsg struct{ init WorkerBlockInit }
type updateWorkerBlockMsg struct {
	workerID string
	token    string
}
type updateWorkerToolMsg struct {
	workerID string
	toolName string
}
type updateWorkerSnippetMsg struct {
	workerID string
	snippet  string
}
type collapseWorkerBlockMsg struct{ status WorkerStatus }
type clearWorkerStatusesMsg struct{}

// createPlannerBlockMsg creates a live progress block for the planner phase
// (the /parallel decomposition step). Uses a fixed workerID "__planner__".
type createPlannerBlockMsg struct {
	model string
	label string
}
type updatePlannerBlockMsg struct{ token string }
type collapsePlannerBlockMsg struct{}

// selPoint is a position inside the wrapped content grid — independent of
// viewport scroll offset. line indexes into appModel.wrappedLines; col is a
// visible-column offset (ANSI-aware, counted in display cells).
type selPoint struct {
	line int
	col  int
}

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
	pendingOutput  string
	flushScheduled bool
	// Wrapped form of outputLines, split by "\n". Kept in sync with whatever
	// is set on the viewport so selection math and copy-extraction can operate
	// on exactly what the user sees.
	wrappedLines []string

	// Viewport refresh optimisation — instead of calling the expensive
	// join→wrap→split pipeline on every mutation, callers set viewportDirty
	// and the single doRefreshViewport() runs once at the end of Update().
	viewportDirty    bool // set by mutation handlers; consumed in Update() epilogue
	fullRewrapNeeded bool // forces full re-wrap (resize, collapse, in-place edit)
	wrappedUpTo      int  // number of outputLines already wrapped (for incremental path)
	contentWrapWidth int  // cached wrap width so both paths use the same value

	// Selection state (drag-to-select + copy). Coordinates are in wrappedLines
	// space so scrolling doesn't invalidate them.
	selStart     selPoint
	selEnd       selPoint
	selecting    bool // left button currently held
	hasSelection bool // a finalized or in-progress selection exists
	copyNoticeAt time.Time

	// Spinner / status bar state
	spinnerActive bool
	spinnerLabel  string
	tokenUsage    string
	activeModel   string
	taskStart     time.Time
	stepStart     time.Time

	// Multi-agent parallel worker statuses (stacked status lines)
	workerStatuses []WorkerStatus

	// Live thinking blocks in the viewport (workerID → block)
	liveBlocks map[string]*liveBlock

	// Planner progress block (uses same liveBlock, keyed by "__planner__")
	plannerBlock *liveBlock

	// Input state
	inputEnabled  bool
	inputChan     chan string // signals ReadLine
	eofChan       chan struct{}
	interruptChan chan struct{}

	// Mouse capture state — toggled at runtime via F2
	mouseEnabled bool

	// Cached working directory for the header bar
	cwd string

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

	historyProvider HistoryProvider
}

func newAppModel(theme Theme, prompt string, provider HistoryProvider) appModel {
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

	cwd, _ := os.Getwd()

	return appModel{
		input:         ti,
		spinner:       sp,
		inputChan:     make(chan string, 1),
		eofChan:       make(chan struct{}, 1),
		interruptChan: make(chan struct{}, 1),
		theme:         theme,
		prompt:        prompt,
		inputEnabled:  true,
		mouseEnabled:  true,
		cwd:           cwd,
		history:         nil,
		historyIndex:    0,
		taskStart:       time.Now(),
		stepStart:       time.Now(),
		historyProvider: provider,
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
			// If at the prompt with a live selection, Ctrl+C acts as
			// "copy" rather than EOF — belt-and-suspenders for users who
			// miss the copy-on-release, and it matches the usual shell
			// gesture. During a running task Ctrl+C always interrupts
			// regardless of any stray selection.
			if m.inputEnabled && m.hasSelection && !m.selecting {
				text := m.selectionText()
				if strings.TrimSpace(text) != "" {
					_ = clipboard.WriteAll(text)
					if _, err := os.Stderr.WriteString(osc52Copy(text)); err != nil {
						_ = err
					}
					m.copyNoticeAt = time.Now()
					m.clearSelection()
					return m, tea.Tick(3*time.Second, func(time.Time) tea.Msg {
						return clearCopyNoticeMsg{}
					})
				}
				m.clearSelection()
				return m, nil
			}
			if !m.inputEnabled {
				// Task running — send interrupt to cancel agent processing
				select {
				case m.interruptChan <- struct{}{}:
				default:
				}
				return m, nil
			}
			// At prompt — signal EOF to ReadLine
			select {
			case m.eofChan <- struct{}{}:
			default:
			}
			return m, nil

		case tea.KeyEsc:
			if m.hasSelection {
				m.clearSelection()
				return m, nil
			}
			if !m.inputEnabled {
				// Task running — send interrupt to cancel agent processing
				select {
				case m.interruptChan <- struct{}{}:
				default:
				}
			}
			return m, nil

		case tea.KeyF2:
			m.mouseEnabled = !m.mouseEnabled
			if m.mouseEnabled {
				return m, tea.EnableMouseCellMotion
			}
			return m, tea.DisableMouse

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
					if m.historyProvider != nil {
						_ = m.historyProvider.AddHistory(value)
					}
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

	case tea.MouseMsg:
		// Wheel events drive scrolling — fall through to viewport. Selection
		// is stored in content-line coordinates, so it stays correct as the
		// viewport scrolls underneath it.
		if msg.Button == tea.MouseButtonWheelUp || msg.Button == tea.MouseButtonWheelDown ||
			msg.Button == tea.MouseButtonWheelLeft || msg.Button == tea.MouseButtonWheelRight {
			break
		}

		headerH := 1
		vpY0 := headerH
		vpY1 := headerH + m.viewport.Height - 1
		inViewport := msg.Y >= vpY0 && msg.Y <= vpY1 && msg.X >= 0 && msg.X < m.viewport.Width

		switch msg.Action {
		case tea.MouseActionPress:
			if msg.Button == tea.MouseButtonLeft && inViewport {
				contentLine := (msg.Y - vpY0) + m.viewport.YOffset
				p := selPoint{line: contentLine, col: msg.X}
				m.selecting = true
				m.hasSelection = true
				m.selStart = p
				m.selEnd = p
				return m, nil
			}
			// Press anywhere else (header, input, status) cancels any live
			// selection so the user can click to dismiss.
			if m.hasSelection {
				m.clearSelection()
				return m, nil
			}

		case tea.MouseActionMotion:
			if m.selecting {
				// Clamp drag position into the viewport so selection extends
				// cleanly when the user drags past the edges.
				y := msg.Y
				if y < vpY0 {
					y = vpY0
				}
				if y > vpY1 {
					y = vpY1
				}
				x := msg.X
				if x < 0 {
					x = 0
				}
				if x > m.viewport.Width {
					x = m.viewport.Width
				}
				m.selEnd = selPoint{line: (y - vpY0) + m.viewport.YOffset, col: x}
				return m, nil
			}

		case tea.MouseActionRelease:
			if m.selecting {
				m.selecting = false
				text := m.selectionText()
				if strings.TrimSpace(text) == "" {
					// Empty selection (just a click) — dismiss without notice.
					m.clearSelection()
					return m, nil
				}
				_ = clipboard.WriteAll(text)
				// Also emit OSC 52 so the copy works over SSH when the
				// outer terminal supports it. bubbletea renders via stdout
				// so we write the escape to stderr to avoid interleaving
				// with the frame buffer.
				if _, err := os.Stderr.WriteString(osc52Copy(text)); err != nil {
					_ = err
				}
				m.copyNoticeAt = time.Now()
				m.clearSelection() // Clear selection so UI state doesn't stay stuck
				return m, tea.Tick(3*time.Second, func(time.Time) tea.Msg {
					return clearCopyNoticeMsg{}
				})
			}
		}

	case clearCopyNoticeMsg:
		m.copyNoticeAt = time.Time{}
		return m, nil

	case appendOutputMsg:
		m.pendingOutput += msg.text
		if !m.flushScheduled {
			m.flushScheduled = true
			return m, tea.Tick(30*time.Millisecond, func(time.Time) tea.Msg {
				return flushOutputMsg{}
			})
		}
		return m, nil

	case flushOutputMsg:
		m.flushScheduled = false
		if m.pendingOutput != "" {
			m.appendOutput(m.pendingOutput)
			m.pendingOutput = ""
		}
		return m, nil

	case clearViewportMsg:
		m.outputLines = nil
		m.pendingOutput = ""
		if m.ready {
			m.viewport.SetContent("")
			m.viewport.GotoTop()
		}
		return m, nil

	case setSpinnerMsg:
		if m.spinnerLabel != msg.label || (!m.spinnerActive && msg.active) {
			m.stepStart = time.Now()
		}

		m.spinnerActive = msg.active
		m.spinnerLabel = msg.label
		if msg.active {
			// Do not reset taskStart here, so timer reflects total duration since user input
			cmds = append(cmds, m.spinner.Tick)
		}

	case setTokenUsageMsg:
		m.tokenUsage = msg.usage

	case enableInputMsg:
		m.inputEnabled = true
		m.historyIndex = len(m.history)
		m.input.Reset()
		m = m.resizeViews(false)
		cmds = append(cmds, m.input.Focus())
		m.workerStatuses = nil // clear worker statuses when input re-enables
		return m, tea.Batch(cmds...)

	case setModelMsg:
		m.activeModel = msg.model

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		// If the agent is busy (!m.inputEnabled), we keep the tick loop
		// alive even if the spinner is visually hidden. This acts as a
		// heartbeat to continuously update the elapsed timers in the status bar.
		if m.spinnerActive || !m.inputEnabled {
			if cmd != nil {
				cmds = append(cmds, cmd)
			} else {
				cmds = append(cmds, m.spinner.Tick)
			}
		}

		// Refresh live block headers on each tick so elapsed timers
		// update smoothly (~100ms) even when no tokens are arriving.
		if len(m.liveBlocks) > 0 || m.plannerBlock != nil {
			if m.plannerBlock != nil && m.plannerBlock.headerIdx < len(m.outputLines) {
				m.outputLines[m.plannerBlock.headerIdx] = renderBlockHeader(m.plannerBlock, false, false, 0)
			}
			for _, blk := range m.liveBlocks {
				if blk.headerIdx < len(m.outputLines) {
					m.outputLines[blk.headerIdx] = renderBlockHeader(blk, false, false, 0)
				}
			}
			m.fullRewrapNeeded = true
			m.viewportDirty = true
		}

	case updateWorkerStatusMsg:
		found := false
		for i, ws := range m.workerStatuses {
			if ws.WorkerID == msg.status.WorkerID {
				m.workerStatuses[i] = msg.status
				found = true
				break
			}
		}
		if !found {
			m.workerStatuses = append(m.workerStatuses, msg.status)
		}
		return m, nil

	case createWorkerBlockMsg:
		blk := &liveBlock{
			workerID:    msg.init.WorkerID,
			number:      msg.init.Number,
			role:        msg.init.Role,
			model:       msg.init.Model,
			label:       msg.init.Label,
			headerIdx:   len(m.outputLines),
			activityIdx: len(m.outputLines) + 1,
			started:     time.Now(),
		}
		if m.liveBlocks == nil {
			m.liveBlocks = make(map[string]*liveBlock)
		}
		m.liveBlocks[msg.init.WorkerID] = blk

		// Insert header + activity lines + blank separator
		lineWidth := m.width - 10
		lines := newBlockLines(blk, lineWidth)
		m.outputLines = append(m.outputLines, lines...)
		m.viewportDirty = true
		return m, nil

	case updateWorkerBlockMsg:
		blk, ok := m.liveBlocks[msg.workerID]
		if !ok {
			return m, nil
		}
		appendStreamToken(blk, msg.token)
		m.patchBlockLinesInPlace(blk)
		m.fullRewrapNeeded = true
		m.viewportDirty = true
		return m, nil

	case updateWorkerToolMsg:
		blk, ok := m.liveBlocks[msg.workerID]
		if !ok {
			return m, nil
		}
		// Push "called: <tool>" as a discrete log line in the rolling mini-viewport.
		appendActivityLine(blk, "called: "+msg.toolName)
		m.patchBlockLinesInPlace(blk)
		m.fullRewrapNeeded = true
		m.viewportDirty = true
		return m, nil

	case updateWorkerSnippetMsg:
		// No-op: the unified displayBuf in liveBlock handles all activity
		// (tool calls + stream) so there is no separate snippet field.
		return m, nil

	case collapseWorkerBlockMsg:
		blk, ok := m.liveBlocks[msg.status.WorkerID]
		if !ok {
			return m, nil
		}

		// Replace header with completed status
		if blk.headerIdx < len(m.outputLines) {
			m.outputLines[blk.headerIdx] = renderBlockHeader(blk, true, msg.status.HasError, msg.status.Elapsed)
		}

		// Remove the blockActivityLines activity rows + blank separator.
		removeStart := blk.activityIdx
		removeEnd := blk.activityIdx + blockActivityLines + 1 // +1 for blank sep
		if removeEnd > len(m.outputLines) {
			removeEnd = len(m.outputLines)
		}
		removeCount := removeEnd - removeStart
		if removeCount > 0 {
			m.outputLines = append(m.outputLines[:removeStart], m.outputLines[removeEnd:]...)
			// Shift indices of all subsequent live blocks
			for _, other := range m.liveBlocks {
				if other.workerID == blk.workerID {
					continue
				}
				if other.headerIdx > blk.headerIdx {
					other.headerIdx -= removeCount
					other.activityIdx -= removeCount
				}
			}
			// Shift plannerBlock indices too
			if m.plannerBlock != nil && m.plannerBlock.headerIdx > blk.headerIdx {
				m.plannerBlock.headerIdx -= removeCount
				m.plannerBlock.activityIdx -= removeCount
			}
		}

		// Collapse invalidates line indices — clear any in-progress selection
		// rather than attempting complex coordinate adjustment.
		if m.hasSelection {
			m.clearSelection()
		}
		m.fullRewrapNeeded = true
		m.viewportDirty = true
		return m, nil

	case clearWorkerStatusesMsg:
		m.workerStatuses = nil
		m.liveBlocks = nil
		// Also clear planner block if present
		if m.plannerBlock != nil {
			removeStart := m.plannerBlock.activityIdx
			removeEnd := m.plannerBlock.activityIdx + blockActivityLines + 1
			if removeEnd > len(m.outputLines) {
				removeEnd = len(m.outputLines)
			}
			if removeEnd > removeStart {
				m.outputLines = append(m.outputLines[:removeStart], m.outputLines[removeEnd:]...)
			}
			if m.plannerBlock.headerIdx < len(m.outputLines) {
				m.outputLines = append(m.outputLines[:m.plannerBlock.headerIdx], m.outputLines[m.plannerBlock.headerIdx+1:]...)
			}
			m.plannerBlock = nil
			if m.hasSelection {
				m.clearSelection()
			}
			m.fullRewrapNeeded = true
			m.viewportDirty = true
		}
		return m, nil

	case createPlannerBlockMsg:
		blk := &liveBlock{
			workerID:    "__planner__",
			number:      0,
			role:        "Planner",
			model:       msg.model,
			label:       msg.label,
			headerIdx:   len(m.outputLines),
			activityIdx: len(m.outputLines) + 1,
			started:     time.Now(),
		}
		m.plannerBlock = blk
		lineWidth := m.width - 10
		lines := newBlockLines(blk, lineWidth)
		m.outputLines = append(m.outputLines, lines...)
		m.viewportDirty = true
		return m, nil

	case updatePlannerBlockMsg:
		// No-op: the planner streams raw decomposition text/JSON which is noisy
		// and not meaningful as a live display. The header's spinner + elapsed
		// timer (updated on every tick) gives sufficient progress feedback.
		return m, nil

	case collapsePlannerBlockMsg:
		if m.plannerBlock == nil {
			return m, nil
		}
		blk := m.plannerBlock

		// Replace header with completed status
		if blk.headerIdx < len(m.outputLines) {
			m.outputLines[blk.headerIdx] = renderBlockHeader(blk, true, false, time.Since(blk.started))
		}

		// Remove activity rows + blank separator
		removeStart := blk.activityIdx
		removeEnd := blk.activityIdx + blockActivityLines + 1
		if removeEnd > len(m.outputLines) {
			removeEnd = len(m.outputLines)
		}
		removeCount := removeEnd - removeStart
		if removeCount > 0 {
			// Shift worker block indices that come after the planner
			for _, other := range m.liveBlocks {
				if other.headerIdx > blk.headerIdx {
					other.headerIdx -= removeCount
					other.activityIdx -= removeCount
				}
			}
			m.outputLines = append(m.outputLines[:removeStart], m.outputLines[removeEnd:]...)
		}
		m.plannerBlock = nil

		if m.hasSelection {
			m.clearSelection()
		}
		m.fullRewrapNeeded = true
		m.viewportDirty = true
		return m, nil

	case appQuitMsg:
		// App finished (or error) — quit the tea program
		return m, tea.Quit

	}

	// Delegate to sub-models.
	// IMPORTANT: Never forward tea.MouseMsg to the textarea — the textarea's
	// own mouse handler can inject phantom characters or move the cursor when
	// the user is simply scrolling the viewport. All mouse interaction is
	// handled above (viewport scroll, drag-to-select) so the textarea should
	// never see mouse events.
	if m.inputEnabled {
		if _, isMouse := msg.(tea.MouseMsg); !isMouse {
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(msg)
			cmds = append(cmds, cmd)
			// Resize dynamically if the user added/removed lines
			m = m.resizeViews(false)
		}
	}

	// Viewport always updates (scroll support)
	{
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		cmds = append(cmds, cmd)
	}

	// ── Dirty-flag epilogue ─────────────────────────────────────────────────
	// Consume the viewport dirty flag once, at the very end of Update(),
	// so multiple handlers that ran in a single cycle only produce one
	// re-wrap instead of N.
	if m.viewportDirty {
		m.viewportDirty = false
		m.doRefreshViewport()
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
		m.viewportDirty = true
	}
}

// doRefreshViewport re-renders the viewport from outputLines.
// When possible, only newly appended lines are wrapped (incremental path).
// In-place mutations (block updates, collapses, resizes) set
// fullRewrapNeeded to force the full join→wrap→split pipeline.
func (m *appModel) doRefreshViewport() {
	if !m.ready {
		return
	}

	// Compute the canonical wrap width once.
	w := m.contentWrapWidth
	if w < 1 {
		w = m.width - 3
		if w < 1 {
			w = 1
		}
	}

	// ── Incremental path: only wrap newly appended lines ────────────────
	canIncremental := !m.fullRewrapNeeded &&
		m.wrappedUpTo > 0 &&
		m.wrappedUpTo <= len(m.outputLines) &&
		len(m.outputLines) > m.wrappedUpTo

	if canIncremental {
		// Wrap only the new tail, append to existing wrappedLines.
		newLines := m.outputLines[m.wrappedUpTo:]
		newContent := strings.Join(newLines, "\n")
		newWrapped := lipgloss.NewStyle().Width(w).Render(newContent)
		m.wrappedLines = append(m.wrappedLines, strings.Split(newWrapped, "\n")...)
		m.wrappedUpTo = len(m.outputLines)
		// SetContent must receive the full document so the viewport's internal
		// line count and scroll range are correct.  Re-join wrappedLines (which
		// are already display-width-wrapped) as the canonical full content.
		// This keeps wrappedLines and viewport content in sync for selection.
		m.viewport.SetContent(strings.Join(m.wrappedLines, "\n"))
	} else {
		// ── Full re-wrap (resize, block collapse, in-place edit) ────────
		content := strings.Join(m.outputLines, "\n")
		wrapped := lipgloss.NewStyle().Width(w).Render(content)
		m.wrappedLines = strings.Split(wrapped, "\n")
		m.wrappedUpTo = len(m.outputLines)
		m.fullRewrapNeeded = false
		m.viewport.SetContent(wrapped)
	}

	// Selection-aware auto-scroll: don't yank the viewport to the bottom
	// while the user is dragging or has an active selection, or has
	// manually scrolled up to read earlier output.
	if !m.hasSelection && !m.selecting && m.isAtBottom() {
		m.viewport.GotoBottom()
	}
}

// isAtBottom returns true if the viewport is within 2 lines of the bottom,
// meaning the user has not deliberately scrolled up.
func (m appModel) isAtBottom() bool {
	maxOffset := m.viewport.TotalLineCount() - m.viewport.Height
	if maxOffset < 0 {
		maxOffset = 0
	}
	return m.viewport.YOffset >= maxOffset-2
}

// patchBlockLinesInPlace updates the header and activity rows of a
// live block directly in outputLines without triggering a full re-wrap.
// The caller must still set viewportDirty = true.
func (m *appModel) patchBlockLinesInPlace(blk *liveBlock) {
	lineWidth := m.width - 10
	// Header
	if blk.headerIdx < len(m.outputLines) {
		m.outputLines[blk.headerIdx] = renderBlockHeader(blk, false, false, 0)
	}
	// Activity rows
	activity := renderActivityLines(blk, lineWidth)
	for i, line := range activity {
		idx := blk.activityIdx + i
		if idx < len(m.outputLines) {
			m.outputLines[idx] = line
		}
	}
}

// renderLiveBlockHeader is kept for any callers that haven't been migrated yet.
// Delegates to the package-level renderBlockHeader in agent_block.go.
func (m appModel) renderLiveBlockHeader(blk *liveBlock, done bool, hasError bool, elapsed time.Duration) string {
	return renderBlockHeader(blk, done, hasError, elapsed)
}

// selectionRange returns the normalized [start, end] endpoints such that
// start comes strictly before end in reading order. Returns !ok when there
// is no active selection.
func (m appModel) selectionRange() (start, end selPoint, ok bool) {
	if !m.hasSelection {
		return selPoint{}, selPoint{}, false
	}
	a, b := m.selStart, m.selEnd
	if a.line > b.line || (a.line == b.line && a.col > b.col) {
		a, b = b, a
	}
	return a, b, true
}

// selectionText extracts the current selection as plain text (ANSI codes
// stripped). Returns "" when the selection is empty.
func (m appModel) selectionText() string {
	start, end, ok := m.selectionRange()
	if !ok {
		return ""
	}
	if len(m.wrappedLines) == 0 {
		return ""
	}
	const farRight = 1 << 30
	if start.line == end.line {
		if start.line < 0 || start.line >= len(m.wrappedLines) {
			return ""
		}
		return ansi.Strip(ansi.Cut(m.wrappedLines[start.line], start.col, end.col))
	}
	var b strings.Builder
	for li := start.line; li <= end.line; li++ {
		if li < 0 || li >= len(m.wrappedLines) {
			if li < end.line {
				b.WriteByte('\n')
			}
			continue
		}
		line := m.wrappedLines[li]
		switch {
		case li == start.line:
			b.WriteString(ansi.Strip(ansi.Cut(line, start.col, farRight)))
		case li == end.line:
			b.WriteString(ansi.Strip(ansi.Cut(line, 0, end.col)))
		default:
			b.WriteString(ansi.Strip(line))
		}
		if li < end.line {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// renderSelectedViewport returns the viewport's current frame with the
// active selection painted as a highlighted background. When there is no
// selection, returns viewport.View() unchanged.
func (m appModel) renderSelectedViewport() string {
	v := m.viewport.View()
	start, end, ok := m.selectionRange()
	if !ok {
		return v
	}
	lines := strings.Split(v, "\n")
	yOff := m.viewport.YOffset
	highlight := lipgloss.NewStyle().Background(colSelectionBg).Foreground(colSelectionFg)
	const farRight = 1 << 30

	for i := range lines {
		contentLine := i + yOff
		if contentLine < start.line || contentLine > end.line {
			continue
		}
		l := lines[i]
		width := ansi.StringWidth(l)
		var startCol, endCol int
		switch {
		case contentLine == start.line && contentLine == end.line:
			startCol, endCol = start.col, end.col
		case contentLine == start.line:
			startCol, endCol = start.col, width
		case contentLine == end.line:
			startCol, endCol = 0, end.col
		default:
			startCol, endCol = 0, width
		}
		if endCol > width {
			endCol = width
		}
		if startCol < 0 {
			startCol = 0
		}
		if startCol >= endCol {
			continue
		}
		before := ansi.Cut(l, 0, startCol)
		sel := ansi.Cut(l, startCol, endCol)
		after := ansi.Cut(l, endCol, farRight)
		// Strip existing ANSI styling from the selected portion so the
		// highlight foreground actually takes effect. Without this, lines
		// with heavy styling (block headers, glamour output) keep their
		// original foreground colors, which clash with the selection bg.
		lines[i] = before + highlight.Render(ansi.Strip(sel)) + after
	}
	return strings.Join(lines, "\n")
}

// clearSelection wipes any active selection state.
func (m *appModel) clearSelection() {
	m.selecting = false
	m.hasSelection = false
	m.selStart = selPoint{}
	m.selEnd = selPoint{}
}

// osc52Copy returns the OSC 52 escape sequence that asks the outer terminal
// to set its clipboard to text. Works over SSH when the terminal supports
// OSC 52 (iTerm2, kitty, WezTerm, recent Terminal.app, etc.).
func osc52Copy(text string) string {
	enc := base64.StdEncoding.EncodeToString([]byte(text))
	return "\x1b]52;c;" + enc + "\a"
}

// Palette — adaptive colors used across the shell chrome.
var (
	colAccent      = lipgloss.AdaptiveColor{Light: "#7AA2F7", Dark: "#7AA2F7"}
	colCyan        = lipgloss.AdaptiveColor{Light: "#0DB9D7", Dark: "#7DCFFF"}
	colGreen       = lipgloss.AdaptiveColor{Light: "#4F6832", Dark: "#9ECE6A"}
	colPurple      = lipgloss.AdaptiveColor{Light: "#9D7CD8", Dark: "#BB9AF7"}
	colMuted       = lipgloss.AdaptiveColor{Light: "#565F89", Dark: "#565F89"}
	colBorderDim   = lipgloss.AdaptiveColor{Light: "#292E42", Dark: "#3B4261"}
	colBgContrast  = lipgloss.AdaptiveColor{Light: "#1A1B26", Dark: "#1A1B26"}
	colSelectionBg = lipgloss.AdaptiveColor{Light: "#A7C2E7", Dark: "#364A82"}
	colSelectionFg = lipgloss.AdaptiveColor{Light: "#1A1B26", Dark: "#C0CAF5"}
)

func (m appModel) View() string {
	if !m.ready {
		return "Loading..."
	}

	header := m.buildHeader()

	viewportRow := m.renderViewportWithScrollbar()

	inputView := m.renderInputBox()
	status := m.buildStatusBar()

	out := lipgloss.JoinVertical(lipgloss.Left,
		header,
		viewportRow,
		inputView,
		status,
	)

	// Final safety clamp: ensure no line exceeds terminal width.
	// This prevents any rendering artifact from JoinVertical padding,
	// emoji width mismatches, or ANSI code interactions.
	if m.width > 0 {
		lines := strings.Split(out, "\n")
		for i, l := range lines {
			if ansi.StringWidth(l) > m.width {
				lines[i] = ansi.Truncate(l, m.width, "")
			}
		}
		out = strings.Join(lines, "\n")
	}
	return out
}

func (m appModel) buildHeader() string {
	if m.width <= 0 {
		return ""
	}

	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(colBgContrast).
		Background(colPurple).
		Padding(0, 1)
	title := titleStyle.Render("✨ codezilla")

	model := m.activeModel
	if model == "" {
		model = "—"
	}
	cwd := m.shortCwd()

	sep := lipgloss.NewStyle().Foreground(colMuted).Render("  ·  ")
	modelSeg := lipgloss.NewStyle().Foreground(colCyan).Render(model)
	cwdSeg := lipgloss.NewStyle().Foreground(colGreen).Render(cwd)
	info := modelSeg + sep + cwdSeg

	titleW := lipgloss.Width(title)
	infoW := lipgloss.Width(info)
	space := m.width - titleW - infoW - 1

	// If too narrow, progressively truncate cwd (as plain) and re-render.
	if space < 0 {
		overflow := -space
		cwdRunes := []rune(cwd)
		if overflow+1 < len(cwdRunes) {
			cwd = string(cwdRunes[:len(cwdRunes)-(overflow+1)]) + "…"
		} else {
			cwd = "…"
		}
		cwdSeg = lipgloss.NewStyle().Foreground(colGreen).Render(cwd)
		info = modelSeg + sep + cwdSeg
		infoW = lipgloss.Width(info)
		space = m.width - titleW - infoW - 1
		if space < 0 {
			space = 0
		}
	}

	gap := strings.Repeat(" ", space)
	return title + gap + info + " "
}

func (m appModel) shortCwd() string {
	cwd := m.cwd
	if cwd == "" {
		return "—"
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" && strings.HasPrefix(cwd, home) {
		return "~" + strings.TrimPrefix(cwd, home)
	}
	return cwd
}

func (m appModel) renderInputBox() string {
	// While a task is running we don't accept input — drop the bordered box
	// and show a lightweight colored status line with the spinner + label.
	// Mirrors how Claude Code / Codex / Copilot CLI render "thinking".
	if !m.inputEnabled {
		return m.renderThinkingLine()
	}
	style := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(colAccent).
		Padding(0, 1)
	// Inner width = total - 2 (border) - 2 (padding)
	inner := m.width - 4
	if inner < 1 {
		inner = 1
	}
	style = style.Width(inner)
	return style.Render(m.input.View())
}

// renderThinkingLine produces the borderless status line shown in place of
// the input box while the agent is processing. Single line, no frame, just
// colored indicator + label so it reads like a hint rather than a boxed
// control.
func (m appModel) renderThinkingLine() string {
	dim := lipgloss.NewStyle().Foreground(colMuted)
	accent := lipgloss.NewStyle().Foreground(colCyan)

	var icon string
	if m.spinnerActive {
		icon = m.spinner.View()
	} else {
		icon = accent.Render("◆")
	}
	label := m.spinnerLabel
	if label == "" {
		label = "thinking"
	}
	label = strings.ReplaceAll(label, "\n", " ")
	hint := dim.Render("  esc/^c to interrupt")
	mainLine := " " + icon + " " + accent.Render(label) + hint

	// If multi-agent workers are active, render stacked status lines below
	if len(m.workerStatuses) > 0 {
		lines := []string{mainLine}
		for _, ws := range m.workerStatuses {
			lines = append(lines, m.renderWorkerStatusLine(ws))
		}
		return strings.Join(lines, "\n")
	}

	return mainLine
}

// renderWorkerStatusLine produces a single colored line for a parallel worker
// status shown in the thinking bar beneath the input.
//
//	[-] [Researcher]  Scanning internal/tools/...       (12s)
//	[+] [Developer]   Completed                          (8s)
//	[!] [Reviewer]    Error: context deadline exceeded   (3s)
func (m appModel) renderWorkerStatusLine(ws WorkerStatus) string {
	dim := lipgloss.NewStyle().Foreground(colMuted)
	roleStyle := lipgloss.NewStyle().Foreground(colPurple).Bold(true)
	labelStyle := lipgloss.NewStyle().Foreground(colCyan)
	greenStyle := lipgloss.NewStyle().Foreground(colGreen)
	redFg := lipgloss.AdaptiveColor{Light: "#F7768E", Dark: "#F7768E"}
	redStyle := lipgloss.NewStyle().Foreground(redFg)

	var icon string
	switch {
	case ws.Done && ws.HasError:
		icon = redStyle.Render("[!]")
	case ws.Done:
		icon = greenStyle.Render("[+]")
	default:
		icon = lipgloss.NewStyle().Foreground(colCyan).Render("[" + m.spinner.View() + "]")
	}

	role := roleStyle.Render("[" + ws.Role + "]")

	label := ws.Label
	if label == "" && ws.Done && ws.HasError {
		label = "failed"
	} else if label == "" && ws.Done {
		label = "completed"
	} else if label == "" {
		label = "working"
	}

	elapsed := ""
	if ws.Done && ws.Elapsed > 0 {
		elapsed = dim.Render(fmt.Sprintf("(%s)", ws.Elapsed.Round(time.Second)))
	} else if !ws.Started.IsZero() {
		elapsed = dim.Render(fmt.Sprintf("(%s)", time.Since(ws.Started).Round(time.Second)))
	}

	numStr := ""
	if ws.Number > 0 {
		numStr = dim.Render(fmt.Sprintf("%d.", ws.Number)) + " "
	}

	modelStr := ""
	if ws.Model != "" {
		modelStr = "  " + dim.Render("("+ws.Model+")")
	}

	return "  " + icon + " " + numStr + role + "  " + labelStyle.Render(label) + "  " + elapsed + modelStr
}

// renderViewportWithScrollbar composites the viewport output with a
// single-column scrollbar by appending the scrollbar glyph to each
// viewport line individually. This avoids lipgloss.JoinHorizontal
// which can misalign the scrollbar when viewport lines have varying
// visible widths due to ANSI escape sequences.
func (m appModel) renderViewportWithScrollbar() string {
	vpView := m.renderSelectedViewport()
	vpLines := strings.Split(vpView, "\n")
	h := m.viewport.Height

	// Use background-colored spaces for the scrollbar to avoid any
	// ambiguous-width Unicode character issues with block characters.
	thumbGlyph := lipgloss.NewStyle().Background(colAccent).Render(" ")
	trackGlyph := lipgloss.NewStyle().Foreground(colBorderDim).Render(" ")

	total := m.viewport.TotalLineCount()
	visible := m.viewport.VisibleLineCount()

	scrollGlyphs := make([]string, h)
	if total <= visible || h <= 0 {
		for i := range scrollGlyphs {
			scrollGlyphs[i] = " "
		}
	} else {
		thumbSize := h * visible / total
		if thumbSize < 1 {
			thumbSize = 1
		}
		if thumbSize > h {
			thumbSize = h
		}
		thumbPos := int(float64(h-thumbSize) * m.viewport.ScrollPercent())
		for i := 0; i < h; i++ {
			if i >= thumbPos && i < thumbPos+thumbSize {
				scrollGlyphs[i] = thumbGlyph
			} else {
				scrollGlyphs[i] = trackGlyph
			}
		}
	}

	// Use ANSI CHA (Cursor Horizontal Absolute, \x1b[nG) to position the
	// scrollbar at a fixed terminal column. This completely bypasses content
	// width calculations, eliminating emoji/Unicode measurement mismatches
	// between Go libraries and the terminal. The scrollbar always appears
	// at the same column regardless of how wide the terminal renders content.
	scrollCol := m.width - 1 // 1-indexed; second-to-last column (last = margin)
	cha := fmt.Sprintf("\x1b[%dG", scrollCol)
	result := make([]string, h)
	for i := 0; i < h; i++ {
		var line string
		if i < len(vpLines) {
			line = vpLines[i]
		}
		result[i] = line + cha + scrollGlyphs[i]
	}
	return strings.Join(result, "\n")
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
	// Input textarea content area: total - 2 (rounded border) - 2 (padding)
	inputContentWidth := m.width - 4
	if inputContentWidth < 1 {
		inputContentWidth = 1
	}
	m.input.SetWidth(inputContentWidth)
	m.input.SetHeight(m.getRequiredInputHeight())

	headerHeight := 1
	inputHeight := m.input.Height() + 2 // rounded border top+bottom
	if !m.inputEnabled {
		inputHeight = 1 + len(m.workerStatuses) // thinking line + stacked worker lines
	}
	statusHeight := 1

	vpHeight := m.height - headerHeight - inputHeight - statusHeight
	if vpHeight < 1 {
		vpHeight = 1
	}

	// Viewport leaves two columns on the right: one for the scrollbar
	// glyph and one trailing space buffer so the scrollbar never sits
	// at the terminal's very last column (which causes rendering artifacts
	// with auto-margin / line-wrapping in many terminals).
	vpWidth := m.width - 2
	if vpWidth < 1 {
		vpWidth = 1
	}

	// Compute the canonical wrap width once — used by doRefreshViewport too.
	wrapWidth := vpWidth - 1
	if wrapWidth < 1 {
		wrapWidth = 1
	}
	oldWrapWidth := m.contentWrapWidth
	m.contentWrapWidth = wrapWidth

	if !m.ready {
		// First-time init: do the full wrap inline since doRefreshViewport
		// checks m.ready and we need content before setting ready = true.
		wrapped := lipgloss.NewStyle().Width(wrapWidth).Render(strings.Join(m.outputLines, "\n"))
		m.wrappedLines = strings.Split(wrapped, "\n")
		m.wrappedUpTo = len(m.outputLines)
		m.viewport = viewport.New(vpWidth, vpHeight)
		m.viewport.SetContent(wrapped)
		m.viewport.GotoBottom()
		m.ready = true
	} else {
		m.viewport.Width = vpWidth
		m.viewport.Height = vpHeight

		// If the wrap width changed (terminal resize) or a reflow is forced,
		// schedule a full re-wrap via the dirty flag.
		if forceReflow || oldWrapWidth != wrapWidth {
			m.fullRewrapNeeded = true
			m.viewportDirty = true
		}
	}
	return m
}

// buildStatusBar renders a single fixed-width line with left-aligned activity
// info and right-aligned meta (tokens + keybinding hints). Widths are stable
// across frames so the line never jumps.
func (m appModel) buildStatusBar() string {
	if m.width <= 0 {
		return ""
	}

	dim := lipgloss.NewStyle().Foreground(colMuted)
	accent := lipgloss.NewStyle().Foreground(colCyan)
	key := lipgloss.NewStyle().Foreground(colPurple).Bold(true)
	sepStr := dim.Render(" · ")

	// ── Left: activity state ──
	var left string
	switch {
	case !m.copyNoticeAt.IsZero() && time.Since(m.copyNoticeAt) < 3*time.Second:
		left = lipgloss.NewStyle().Foreground(colGreen).Render("✓ copied to clipboard") + sepStr + dim.Render("drag to select · esc to clear")
	case m.hasSelection:
		start, end, _ := m.selectionRange()
		lineCount := end.line - start.line + 1
		label := fmt.Sprintf("selection · %d line", lineCount)
		if lineCount != 1 {
			label += "s"
		}
		left = lipgloss.NewStyle().Foreground(colPurple).Render("▎ ") + accent.Render(label) + sepStr + dim.Render("release to copy · esc to clear")
	case !m.inputEnabled:
		// The thinking line above already shows spinner + label; here we
		// just surface the elapsed timers so the user can see the task is
		// still progressing.
		step := time.Since(m.stepStart).Round(time.Second)
		total := time.Since(m.taskStart).Round(time.Second)
		left = dim.Render(fmt.Sprintf("step %s · total %s", step, total))
	default:
		left = dim.Render("●") + " " + lipgloss.NewStyle().Foreground(colGreen).Render("ready") + sepStr + dim.Render("enter to send · alt+enter for newline")
	}

	// ── Right: tokens + keybindings ──
	tok := m.tokenUsage
	if tok == "" {
		tok = "0 tok"
	}
	mouseLabel := "mouse on"
	if !m.mouseEnabled {
		mouseLabel = "mouse off"
	}
	right := dim.Render(tok) + sepStr + key.Render("F2") + " " + dim.Render(mouseLabel) + sepStr + key.Render("^C") + " " + dim.Render("interrupt")

	// ── Compose with stable total width ──
	leftW := lipgloss.Width(left)
	rightW := lipgloss.Width(right)
	gap := m.width - leftW - rightW - 2 // 1-char pad on each side

	if gap < 1 {
		// Drop the keybinding hints from the right side when too narrow.
		right = dim.Render(tok) + sepStr + key.Render("F2") + " " + dim.Render(mouseLabel)
		rightW = lipgloss.Width(right)
		gap = m.width - leftW - rightW - 2
	}
	if gap < 1 {
		// Drop the right side entirely.
		right = ""
		rightW = 0
		gap = m.width - leftW - 1
	}
	if gap < 0 {
		gap = 0
	}

	return " " + left + strings.Repeat(" ", gap) + right + " "
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
	provider     HistoryProvider

	// inline, when true, runs on the main terminal buffer instead of the
	// alt-screen. Trade-off: scrollback survives exit, but rendering can
	// feel busier during the session.
	inline bool

	// mouseEnabled is the initial mouse-capture state applied when the tea
	// program starts. F2 flips it at runtime; this field is not kept in
	// sync with that toggle.
	mouseEnabled bool

	// ReadLine synchronization — these channels are in the appModel
	inputChan     chan string
	eofChan       chan struct{}
	interruptChan chan struct{}
}

// TUIRunner is an optional interface satisfied by BubbleTeaUI.
// main.go detects this and uses it to run the tea program on the main goroutine.
type TUIRunner interface {
	// RunTUI starts the BubbleTea program on the calling goroutine and runs
	// appFn concurrently. Blocks until the program exits.
	RunTUI(ctx context.Context, appFn func(context.Context) error) error
}

// NewBubbleTeaUI creates a new BubbleTea-based UI implementation.
//
// inline=true renders on the main terminal buffer instead of the alt-screen.
// disableMouse=true skips mouse capture at launch so native text selection
// keeps working — the F2 keybinding still flips mouse mode at runtime.
//
// CODEZILLA_INLINE=1 and CODEZILLA_NO_MOUSE=1 env vars are equivalent opt-ins
// for scripts when the flags aren't set.
func NewBubbleTeaUI(provider HistoryProvider, inline, disableMouse bool) (UI, error) {
	if !inline {
		if v := strings.ToLower(strings.TrimSpace(os.Getenv("CODEZILLA_INLINE"))); v == "1" || v == "true" || v == "yes" {
			inline = true
		}
	}
	if !disableMouse {
		if v := strings.ToLower(strings.TrimSpace(os.Getenv("CODEZILLA_NO_MOUSE"))); v == "1" || v == "true" || v == "yes" {
			disableMouse = true
		}
	}

	theme := ThemeTokyoNight()
	theme.IconSuccess = "✅"
	theme.IconError = "❌"
	theme.IconWarning = "⚠️"
	theme.IconInfo = "💡"
	theme.IconPrompt = "🤖"

	prompt := "codezilla " + theme.IconPrompt + " "

	model := newAppModel(theme, prompt, provider)
	model.mouseEnabled = !disableMouse

	// Load history from provider
	if provider != nil {
		entries := provider.GetHistory(-1)
		model.history = entries
		model.historyIndex = len(entries)
	}

	ui := &BubbleTeaUI{
		theme:         theme,
		provider:      provider,
		inline:        inline,
		mouseEnabled:  !disableMouse,
		inputChan:     model.inputChan,
		eofChan:       model.eofChan,
		interruptChan: model.interruptChan,
		model:         &model,
	}

	return ui, nil
}

// RunTUI starts the BubbleTea program on the calling goroutine and runs
// appFn concurrently. Blocks until the tea program exits.
func (ui *BubbleTeaUI) RunTUI(ctx context.Context, appFn func(context.Context) error) error {
	var opts []tea.ProgramOption
	if ui.mouseEnabled {
		opts = append(opts, tea.WithMouseCellMotion())
	}
	if !ui.inline {
		opts = append(opts, tea.WithAltScreen())
	}
	ui.program = tea.NewProgram(*ui.model, opts...)

	// Run the application logic in a background goroutine.
	// When it finishes, send appQuitMsg to shut down the tea program.
	go func() {
		err := appFn(ctx)
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

func (ui *BubbleTeaUI) InterruptChan() <-chan struct{} {
	return ui.interruptChan
}

func (ui *BubbleTeaUI) Clear() {
	if ui.program != nil {
		ui.program.Send(clearViewportMsg{})
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

// ── Multi-Agent Worker Status ────────────────────────────────────────────────

// UpdateWorkerStatus pushes a worker's current state to the TUI. Call this
// from the multi-agent orchestrator's bus event listener to show stacked
// per-worker status lines during parallel execution.
func (ui *BubbleTeaUI) UpdateWorkerStatus(status WorkerStatus) {
	if ui.program != nil {
		ui.program.Send(updateWorkerStatusMsg{status: status})
	}
}

// ClearWorkerStatuses removes all worker status lines from the display.
func (ui *BubbleTeaUI) ClearWorkerStatuses() {
	if ui.program != nil {
		ui.program.Send(clearWorkerStatusesMsg{})
	}
}

// CreateWorkerBlock inserts a live thinking block into the viewport for a
// parallel worker. The block includes a header line and placeholder thinking
// lines that will be updated in-place as tokens stream in.
func (ui *BubbleTeaUI) CreateWorkerBlock(init WorkerBlockInit) {
	if ui.program != nil {
		ui.program.Send(createWorkerBlockMsg{init: init})
	}
}

// UpdateWorkerBlock appends a thinking/output token to a specific worker's
// live block in the viewport, updating the visible thinking lines in-place.
func (ui *BubbleTeaUI) UpdateWorkerBlock(workerID, token string) {
	if ui.program != nil {
		ui.program.Send(updateWorkerBlockMsg{workerID: workerID, token: token})
	}
}

// CollapseWorkerBlock replaces a worker's expanded activity block with a
// single completed/failed summary line, removing the activity sub-lines.
func (ui *BubbleTeaUI) CollapseWorkerBlock(status WorkerStatus) {
	if ui.program != nil {
		ui.program.Send(collapseWorkerBlockMsg{status: status})
	}
}

// UpdateWorkerBlockTool updates the tool-use activity row for a running worker block.
// Implements AgentBlockDisplayer.
func (ui *BubbleTeaUI) UpdateWorkerBlockTool(workerID, toolName string) {
	if ui.program != nil {
		ui.program.Send(updateWorkerToolMsg{workerID: workerID, toolName: toolName})
	}
}

// UpdateWorkerBlockSnippet updates the result-snippet activity row for a running worker block.
// Implements AgentBlockDisplayer.
func (ui *BubbleTeaUI) UpdateWorkerBlockSnippet(workerID, snippet string) {
	if ui.program != nil {
		ui.program.Send(updateWorkerSnippetMsg{workerID: workerID, snippet: snippet})
	}
}

// ── Planner progress block ──────────────────────────────────────────────────

// CreatePlannerBlock inserts a live 3-line progress block in the viewport
// for the /parallel planning/decomposition phase.
func (ui *BubbleTeaUI) CreatePlannerBlock(model, label string) {
	if ui.program != nil {
		ui.program.Send(createPlannerBlockMsg{model: model, label: label})
	}
}

// UpdatePlannerBlock streams a token into the planner's progress block.
func (ui *BubbleTeaUI) UpdatePlannerBlock(token string) {
	if ui.program != nil {
		ui.program.Send(updatePlannerBlockMsg{token: token})
	}
}

// CollapsePlannerBlock collapses the planner's progress block into a
// single completed header line.
func (ui *BubbleTeaUI) CollapsePlannerBlock() {
	if ui.program != nil {
		ui.program.Send(collapsePlannerBlockMsg{})
	}
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
	width := 80
	if ui.model != nil && ui.model.width > 0 {
		width = ui.model.width - 4
	}
	md := "```" + language + "\n" + code
	if !strings.HasSuffix(md, "\n") {
		md += "\n"
	}
	md += "```\n"

	renderer, err := glamour.NewTermRenderer(
		glamour.WithStyles(styles.DarkStyleConfig),
		glamour.WithWordWrap(width),
	)
	if err != nil {
		ui.appendToViewport(md)
		return
	}
	rendered, err := renderer.Render(md)
	if err != nil {
		ui.appendToViewport(md)
		return
	}
	ui.appendToViewport(rendered)
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

	var out strings.Builder
	for _, line := range banner {
		out.WriteString(ui.theme.StyleCyan.Render(line))
		out.WriteByte('\n')
	}
	out.WriteByte('\n')
	out.WriteString(ui.theme.StyleYellow.Render("✨ "))
	out.WriteString("AI-Powered Coding Assistant ✨\n")

	// Gradient separator, built in one pass.
	colors := []lipgloss.Style{ui.theme.StylePurple, ui.theme.StyleBlue, ui.theme.StyleCyan, ui.theme.StyleBlue, ui.theme.StylePurple}
	separatorChars := []string{"═", "╪", "╬", "╪", "═"}
	for i := 0; i < 80; i++ {
		colorIndex := (i * len(colors)) / 80
		charIndex := (i * len(separatorChars)) / 80
		if charIndex >= len(separatorChars) {
			charIndex = len(separatorChars) - 1
		}
		out.WriteString(colors[colorIndex].Render(separatorChars[charIndex]))
	}
	out.WriteByte('\n')

	ui.appendToViewport(out.String())
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
		ui.appendToViewport("  ⚡️ " + keyStyle.Render("Fast:") + " " + ui.theme.StyleYellow.Render(fastModel) + "\n")
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
		{"Drag in viewport", "Select text · release to copy to clipboard"},
		{"Esc (with selection)", "Clear selection highlight"},
		{"F2", "Toggle mouse capture (off = native terminal selection)"},
		{"Ctrl+C", "Copy active selection · interrupt task · exit at empty prompt"},
		{"Alt+Enter", "Insert newline in the input box"},
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
	if ui.provider != nil {
		return ui.provider.ClearHistory()
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
