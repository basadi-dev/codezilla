package ui

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/charmbracelet/lipgloss"
	"golang.org/x/term"
)

// FixedInput implements InputReader with simple, reliable single-line input
type FixedInput struct {
	prompt         string
	reader         *bufio.Reader
	provider       HistoryProvider
	history        []string
	historyIndex   int
	mu             sync.Mutex
	rawMode        bool
	fd             int
	currentLines   int                            // Track how many lines the current prompt/input spans
	completer      func(line string) []Completion // optional Tab-completion callback
	menuActive     bool                           // whether the interactive dropdown is visible
	menuIndex      int                            // currently highlighted item in the dropdown
	menuCandidates []Completion                   // cached results for the dropdown
	menuLines      int                            // how many lines were printed below the prompt for the menu
	cursorLine     int                            // tracks vertical offset of terminal cursor from prompt start
	drawnBelow     int                            // total lines drawn below cursor (menu + footer)
	theme          Theme                          // active theme for the UI
	footerCallback func() string                  // dynamic footer text callback
}

// SetTheme updates the active theme dynamically
func (fi *FixedInput) SetTheme(t Theme) {
	fi.mu.Lock()
	fi.theme = t
	fi.mu.Unlock()
}

// SetPrompt updates the prompt string
func (fi *FixedInput) SetPrompt(prompt string) {
	fi.prompt = prompt
}

// SetCompleter sets the Tab-completion callback. The function receives the
// current input text (up to the cursor) and returns a slice of completion
// candidates. Pass nil to disable completion.
func (fi *FixedInput) SetCompleter(fn func(line string) []Completion) {
	fi.mu.Lock()
	defer fi.mu.Unlock()
	fi.completer = fn
}

// SetFooter assigns a dynamic string generator to be printed below the input prompt.
func (fi *FixedInput) SetFooter(fn func() string) {
	fi.mu.Lock()
	defer fi.mu.Unlock()
	fi.footerCallback = fn
}

// promptDisplayWidth calculates the actual display width of the prompt
// accounting for styles and multi-width characters like emoji
func (fi *FixedInput) promptDisplayWidth() int {
	return lipgloss.Width(fi.prompt)
}

// NewFixedInput creates a new input reader with history support but no multi-line bugs
func NewFixedInput(prompt string, provider HistoryProvider) (*FixedInput, error) {
	fd := int(os.Stdin.Fd())

	input := &FixedInput{
		prompt:       prompt,
		reader:       bufio.NewReader(os.Stdin),
		provider:     provider,
		history:      make([]string, 0, 500),
		historyIndex: -1,
		fd:           fd,
		rawMode:      term.IsTerminal(fd),
		currentLines: 1,
	}

	// Load history from provider if it exists
	if provider != nil {
		history := provider.GetHistory(-1)
		input.history = append(input.history, history...)
	}

	return input, nil
}

// ReadLine reads a line of input - simple and reliable
func (fi *FixedInput) ReadLine() (string, error) {
	// If not in a terminal, just do simple reading
	if !fi.rawMode {
		return fi.readSimple()
	}

	// Try to use raw mode for arrow key support
	oldState, err := term.MakeRaw(fi.fd)
	if err != nil {
		// Fall back to simple reading if raw mode fails
		return fi.readSimple()
	}
	defer func() {
		if err := term.Restore(fi.fd, oldState); err != nil {
			// Log error but don't return it as we're in a deferred function
			fmt.Fprintf(os.Stderr, "Failed to restore terminal: %v\n", err)
		}
	}()

	// Buffer for the current line
	var line []rune
	pos := 0

	// Reset state
	fi.menuActive = false
	fi.cursorLine = 0
	fi.drawnBelow = 0
	fi.menuLines = 0
	fi.menuIndex = 0
	fi.menuCandidates = nil

	// Print initial prompt (and footer) before waiting for keys
	fi.currentLines = 1 // Already set in NewFixedInput but just safe
	fi.redrawLine(line, pos)

	// History navigation state
	fi.mu.Lock()
	fi.historyIndex = len(fi.history)
	historySize := len(fi.history)
	fi.mu.Unlock()
	savedLine := ""
	historyNavMode := false

	for {
		// Read one byte
		b := make([]byte, 1)
		_, err := fi.reader.Read(b)
		if err != nil {
			if err == io.EOF {
				fmt.Print("\r\n")
				return "", io.EOF
			}
			return "", err
		}

		switch b[0] {
		case '\r', '\n': // Enter
			if fi.menuActive && fi.menuIndex >= 0 && fi.menuIndex < len(fi.menuCandidates) {
				// Accept menu option AND submit
				line = []rune(fi.menuCandidates[fi.menuIndex].Text)
				pos = len(line)
				fi.menuActive = false
			}

			// Clear menu, footer, and everything drawn below prompt
			fi.clearDrawnContent()

			// Re-print prompt and line cleanly with a final newline
			fmt.Print(fi.prompt)
			fmt.Print(string(line))
			fmt.Print("\r\n")

			result := string(line)
			if result != "" {
				fi.addHistory(result)
			}
			fi.currentLines = 1
			fi.cursorLine = 0
			fi.drawnBelow = 0
			return result, nil

		case 0x03: // Ctrl-C
			fi.clearDrawnContent()
			fmt.Print(fi.prompt)
			fmt.Print(string(line))
			fmt.Print("^C\r\n")
			fi.currentLines = 1
			fi.cursorLine = 0
			fi.drawnBelow = 0
			return "", io.EOF

		case 0x04: // Ctrl-D
			if len(line) == 0 {
				fi.clearDrawnContent()
				fmt.Print(fi.prompt)
				fmt.Print("\r\n")
				fi.currentLines = 1
				fi.cursorLine = 0
				fi.drawnBelow = 0
				return "", io.EOF
			}
			// Delete character at cursor
			if pos < len(line) {
				historyNavMode = false
				line = append(line[:pos], line[pos+1:]...)
				fi.redrawLine(line, pos)
			}

		case 0x01: // Ctrl-A - beginning of line
			historyNavMode = false
			pos = 0
			fi.redrawLine(line, pos)

		case 0x05: // Ctrl-E - end of line
			historyNavMode = false
			pos = len(line)
			fi.redrawLine(line, pos)

		case 0x0B: // Ctrl-K - kill to end of line
			historyNavMode = false
			if pos < len(line) {
				line = line[:pos]
				fi.redrawLine(line, pos)
			}

		case 0x15: // Ctrl-U - kill to beginning of line
			historyNavMode = false
			line = line[pos:]
			pos = 0
			fi.redrawLine(line, pos)

		case 0x17: // Ctrl-W - delete word backward
			historyNavMode = false
			if pos > 0 {
				start := pos
				// Skip spaces
				for start > 0 && line[start-1] == ' ' {
					start--
				}
				// Skip word
				for start > 0 && line[start-1] != ' ' {
					start--
				}
				if start < pos {
					line = append(line[:start], line[pos:]...)
					pos = start
					fi.redrawLine(line, pos)
				}
			}

		case 0x0C: // Ctrl-L - clear screen
			fmt.Print("\033[2J\033[H")
			fi.redrawLine(line, pos)

		case 0x7F, 0x08: // Backspace
			historyNavMode = false
			if pos > 0 {
				line = append(line[:pos-1], line[pos:]...)
				pos--
				fi.redrawLine(line, pos)
			}

		case 0x1B: // ESC - start of escape sequence
			// Read next two bytes for arrow keys
			seq := make([]byte, 2)
			n, _ := fi.reader.Read(seq)
			if n == 2 && seq[0] == '[' {
				switch seq[1] {
				case 'A': // Up arrow
					if fi.menuActive && len(fi.menuCandidates) > 0 && !historyNavMode {
						fi.menuIndex--
						if fi.menuIndex < 0 {
							fi.menuIndex = len(fi.menuCandidates) - 1
						}
						fi.redrawLine(line, pos)
					} else if historySize > 0 && fi.historyIndex > 0 {
						historyNavMode = true
						// previous history
						if fi.historyIndex == historySize {
							savedLine = string(line)
						}
						fi.historyIndex--
						fi.mu.Lock()
						line = []rune(fi.history[fi.historyIndex])
						fi.mu.Unlock()
						pos = len(line)
						fi.redrawLine(line, pos)
					}

				case 'B': // Down arrow
					if fi.menuActive && len(fi.menuCandidates) > 0 && !historyNavMode {
						fi.menuIndex++
						if fi.menuIndex >= len(fi.menuCandidates) {
							fi.menuIndex = 0
						}
						fi.redrawLine(line, pos)
					} else if historySize > 0 && fi.historyIndex < historySize {
						historyNavMode = true
						// next history
						fi.historyIndex++
						if fi.historyIndex == historySize {
							// Restore saved line
							line = []rune(savedLine)
							historyNavMode = false
						} else {
							fi.mu.Lock()
							line = []rune(fi.history[fi.historyIndex])
							fi.mu.Unlock()
						}
						pos = len(line)
						fi.redrawLine(line, pos)
					}

				case 'C': // Right arrow
					historyNavMode = false
					if pos < len(line) {
						pos++
						fi.redrawLine(line, pos)
					} else {
						// At end of line, check if we can accept ghost suggestion
						fi.mu.Lock()
						completerFn := fi.completer
						fi.mu.Unlock()
						if completerFn != nil {
							prefix := string(line)
							candidates := completerFn(prefix)
							if len(candidates) > 0 {
								lcp := longestCommonPrefix(candidates)
								if len(lcp) > len(prefix) && strings.HasPrefix(lcp, prefix) {
									line = []rune(lcp)
									pos = len(line)
									fi.redrawLine(line, pos)
								}
							}
						}
					}

				case 'D': // Left arrow
					historyNavMode = false
					if pos > 0 {
						pos--
						fi.redrawLine(line, pos)
					}

				case 'H': // Home
					historyNavMode = false
					pos = 0
					fi.redrawLine(line, pos)

				case 'F': // End
					historyNavMode = false
					pos = len(line)
					fi.redrawLine(line, pos)

				case '3': // Delete key (sequence is ESC[3~)
					extra := make([]byte, 1)
					if _, err := fi.reader.Read(extra); err != nil {
						// Continue on error
						continue
					}
					if extra[0] == '~' && pos < len(line) {
						historyNavMode = false
						line = append(line[:pos], line[pos+1:]...)
						fi.redrawLine(line, pos)
					}
				}
			}

		case 0x09: // Tab — auto-complete
			historyNavMode = false
			if fi.menuActive && fi.menuIndex >= 0 && fi.menuIndex < len(fi.menuCandidates) {
				// Accept currently highlighted menu option
				line = []rune(fi.menuCandidates[fi.menuIndex].Text)
				pos = len(line)
				fi.menuActive = false
				fi.redrawLine(line, pos)
			} else {
				fi.mu.Lock()
				completerFn := fi.completer
				fi.mu.Unlock()

				if completerFn != nil {
					prefix := string(line)
					candidates := completerFn(prefix)
					if len(candidates) > 0 {
						fi.menuActive = true
						fi.menuCandidates = candidates
						fi.menuIndex = 0
						fi.redrawLine(line, pos)
					} else {
						fmt.Print("\007") // beep if no match
					}
				}
			}

		default:
			// Regular character
			if b[0] >= 32 && b[0] < 127 {
				historyNavMode = false
				// Insert character at position
				line = append(line[:pos], append([]rune{rune(b[0])}, line[pos:]...)...)
				pos++
				fi.redrawLine(line, pos)
			}
		}
	}
}

// redrawLine redraws the current line with proper wrapping support
func (fi *FixedInput) redrawLine(line []rune, pos int) {
	// Get terminal dimensions
	termWidth := 80 // Default
	if width, _, err := term.GetSize(fi.fd); err == nil && width > 0 {
		termWidth = width
	}

	// Calculate prompt width
	promptLen := fi.promptDisplayWidth()

	// Calculate total content including prompt
	totalLen := promptLen + len(line)
	numLines := (totalLen + termWidth - 1) / termWidth // Ceiling division
	if numLines == 0 {
		numLines = 1
	}

	// Move cursor to the beginning of the prompt using our tracked cursor position
	if fi.cursorLine > 0 {
		fmt.Printf("\033[%dA", fi.cursorLine)
	}
	fmt.Print("\r")

	// Explicitly clear all previously drawn lines one by one.
	// This is more reliable than \033[J across terminal emulators,
	// especially in raw mode where OPOST is disabled.
	totalClear := fi.cursorLine + fi.drawnBelow + 1
	for i := 0; i < totalClear; i++ {
		fmt.Print("\033[2K") // Erase entire line
		if i < totalClear-1 {
			fmt.Print("\033[B") // Cursor down (CSI B), avoids \n raw-mode issues
		}
	}
	// Move back to prompt start
	if totalClear > 1 {
		fmt.Printf("\033[%dA", totalClear-1)
	}
	fmt.Print("\r")

	// Print prompt
	fmt.Print(fi.prompt)

	// Print the line with wrapping
	text := string(line)
	fmt.Print(text)

	// Automatically trigger interactive menu if we have a slash command
	if len(text) > 0 && strings.HasPrefix(text, "/") {
		fi.menuActive = true
	} else {
		fi.menuActive = false
	}

	// Show inline ghost suggestion if no menu is active, or menu is active but we still want a hint.
	// We'll hide the inline hint if the menu is active to avoid clutter, since the menu already shows options.
	suggestionPrinted := 0
	if !fi.menuActive && pos == len(line) && pos > 0 {
		fi.mu.Lock()
		completerFn := fi.completer
		fi.mu.Unlock()
		if completerFn != nil {
			candidates := completerFn(text)
			if len(candidates) > 0 {
				ghostText := ""
				lcp := longestCommonPrefix(candidates)
				if len(lcp) > len(text) && strings.HasPrefix(lcp, text) {
					ghostText = lcp[len(text):]
				} else if strings.HasPrefix(candidates[0].Text, text) {
					ghostText = candidates[0].Text[len(text):]
				}

				if ghostText != "" {
					// Print in dim/gray
					fmt.Printf("\033[90m%s\033[0m", ghostText)
					suggestionPrinted = len(ghostText)
				}
			}
		}
	}

	totalPrintedLen := promptLen + len(line) + suggestionPrinted
	numLines = (totalPrintedLen + termWidth - 1) / termWidth // Ceiling division
	if numLines == 0 {
		numLines = 1
	}

	// Update the number of lines we're using
	fi.currentLines = numLines

	endLine := 0
	if totalPrintedLen > 0 {
		endLine = (totalPrintedLen - 1) / termWidth
	}
	currentCursorY := endLine // This is our physical line drawn so far

	// Menu Drawing
	menuDrawnLines := 0
	if fi.menuActive {
		fi.mu.Lock()
		completerFn := fi.completer
		fi.mu.Unlock()

		if completerFn != nil {
			candidates := completerFn(text)
			fi.menuCandidates = candidates
			if len(candidates) > 0 {
				// Bounds check menuIndex (in case candidates shrank)
				if fi.menuIndex < 0 {
					fi.menuIndex = len(candidates) - 1
				} else if fi.menuIndex >= len(candidates) {
					fi.menuIndex = 0
				}

				maxMenuLines := 7
				drawCount := len(candidates)
				if drawCount > maxMenuLines {
					drawCount = maxMenuLines
				}

				windowStart := 0
				if fi.menuIndex >= maxMenuLines {
					windowStart = fi.menuIndex - maxMenuLines + 1
				}

				fmt.Print("\r\n")
				currentCursorY++

				// Determine separator width
				width := termWidth - 1
				if width > 120 {
					width = 120
				}
				if width < 20 {
					width = 20
				}

				cmdStyle := fi.theme.ACTheme.Cmd
				descStyle := fi.theme.ACTheme.Desc
				hiCmdStyle := fi.theme.ACTheme.HiCmd
				hiDescStyle := fi.theme.ACTheme.HiDesc
				hiPrefixStyle := fi.theme.ACTheme.HiPrefix
				separator := fi.theme.ACTheme.Separator.Render(strings.Repeat("─", width))

				fmt.Print(separator + "\r\n")
				currentCursorY++
				menuDrawnLines += 2 // newline + separator

				// padding calculation (first column)
				maxLen := 0
				for _, c := range candidates {
					disp := c.Display
					if disp == "" {
						disp = c.Text
					}
					if w := lipgloss.Width(disp); w > maxLen {
						maxLen = w
					}
				}
				pad := maxLen + 2

				for i := 0; i < drawCount; i++ {
					idx := windowStart + i
					if idx >= len(candidates) {
						break
					}
					c := candidates[idx]

					isHighlighted := (idx == fi.menuIndex)
					prefixStr := "   "
					if isHighlighted {
						prefixStr = hiPrefixStyle.Render(" ❯ ")
					}

					cmdStr := c.Display
					if cmdStr == "" {
						cmdStr = c.Text
					}
					descStr := c.Description
					if isHighlighted {
						cmdStr = hiCmdStyle.Render(cmdStr)
						if descStr != "" {
							descStr = hiDescStyle.Render(descStr)
						}
					} else {
						cmdStr = cmdStyle.Render(cmdStr)
						if descStr != "" {
							descStr = descStyle.Render(descStr)
						}
					}

					rawDisp := c.Display
					if rawDisp == "" {
						rawDisp = c.Text
					}
					visualWidth := lipgloss.Width(rawDisp)
					spacing := ""
					if pad > visualWidth {
						spacing = strings.Repeat(" ", pad-visualWidth)
					}

					fmt.Printf("%s%s%s %s\r\n", prefixStr, cmdStr, spacing, descStr)
					currentCursorY++
					menuDrawnLines++
				}

				bottomSeparator := fi.theme.ACTheme.Separator.Render(strings.Repeat("─", width))
				fmt.Print(bottomSeparator) // Bottom border without trailing newline
			} else {
				fi.menuActive = false // no candidates, deactivate
			}
		}
	}

	// Check and Draw Footer below menu
	fi.mu.Lock()
	footerCb := fi.footerCallback
	fi.mu.Unlock()
	if footerCb != nil {
		footerStr := footerCb()
		if footerStr != "" {
			// Use explicit \r\n for raw-mode compatibility (OPOST disabled)
			fmt.Print("\r\n" + footerStr)
			footerLines := strings.Count(footerStr, "\n") + 1
			currentCursorY += footerLines
			menuDrawnLines += footerLines
		}
	}

	fi.menuLines = menuDrawnLines

	// Calculate target cursor position with wrapping
	absPos := promptLen + pos
	targetCursorLine := absPos / termWidth
	targetCursorCol := absPos % termWidth

	// Move cursor from bottom of drawn content (currentCursorY) back up to the exact target input line
	if currentCursorY > targetCursorLine {
		fmt.Printf("\033[%dA", currentCursorY-targetCursorLine)
	}

	// Move to beginning of line and then to cursor column
	fmt.Print("\r")
	if targetCursorCol > 0 {
		fmt.Printf("\033[%dC", targetCursorCol)
	}

	// Update tracked cursor line offset so we know where we are purely relative to prompt
	fi.cursorLine = targetCursorLine
	// Track total lines drawn below the cursor so Ctrl+C/D can clean up
	fi.drawnBelow = currentCursorY - targetCursorLine
}

// clearDrawnContent moves the cursor back to the start of the prompt and erases
// everything drawn below (menu, footer, ghost text). This must be called before
// any exit path from ReadLine so subsequent output doesn't collide with stale
// footer/menu content.
func (fi *FixedInput) clearDrawnContent() {
	// Move cursor up to prompt start (cursorLine tracks our offset within the text)
	if fi.cursorLine > 0 {
		fmt.Printf("\033[%dA", fi.cursorLine)
	}
	fmt.Print("\r")

	// Explicitly clear all drawn lines (prompt + text lines + menu + footer)
	totalClear := fi.cursorLine + fi.drawnBelow + 1
	for i := 0; i < totalClear; i++ {
		fmt.Print("\033[2K")
		if i < totalClear-1 {
			fmt.Print("\033[B")
		}
	}
	// Move back to prompt start
	if totalClear > 1 {
		fmt.Printf("\033[%dA", totalClear-1)
	}
	fmt.Print("\r")
}

// longestCommonPrefix returns the longest string that is a prefix of all items.
func longestCommonPrefix(items []Completion) string {
	if len(items) == 0 {
		return ""
	}
	prefix := items[0].Text
	for _, s := range items[1:] {
		for !strings.HasPrefix(s.Text, prefix) {
			if len(prefix) == 0 {
				return ""
			}
			prefix = prefix[:len(prefix)-1]
		}
	}
	return prefix
}

// readSimple provides fallback for non-terminal input
func (fi *FixedInput) readSimple() (string, error) {
	fmt.Print(fi.prompt)
	line, err := fi.reader.ReadString('\n')
	if err != nil {
		return "", err
	}
	result := strings.TrimRight(line, "\r\n")
	if result != "" {
		fi.addHistory(result)
	}
	return result, nil
}

// Close cleans up resources
func (fi *FixedInput) Close() error {
	return nil
}

// History management

func (fi *FixedInput) addHistory(line string) {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	// Don't add duplicates
	if len(fi.history) > 0 && fi.history[len(fi.history)-1] == line {
		return
	}

	fi.history = append(fi.history, line)
	fi.historyIndex = len(fi.history)

	// Save asynchronously
	go func() {
		if fi.provider != nil {
			if err := fi.provider.AddHistory(line); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to save history: %v\n", err)
			}
		}
	}()
}

// GetHistory returns the most recent n history entries (newest last).
// If n <= 0, returns all entries.
func (fi *FixedInput) GetHistory(n int) []string {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	if n <= 0 || n >= len(fi.history) {
		result := make([]string, len(fi.history))
		copy(result, fi.history)
		return result
	}

	start := len(fi.history) - n
	result := make([]string, n)
	copy(result, fi.history[start:])
	return result
}

// SearchHistory returns history entries that contain the query substring,
// ordered newest last.
func (fi *FixedInput) SearchHistory(query string) []string {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	var results []string
	for _, entry := range fi.history {
		if strings.Contains(entry, query) {
			results = append(results, entry)
		}
	}
	return results
}

// ClearHistory removes all history entries
func (fi *FixedInput) ClearHistory() error {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	fi.history = fi.history[:0]
	fi.historyIndex = 0

	if fi.provider != nil {
		return fi.provider.ClearHistory()
	}
	return nil
}

// GetDefaultHistoryFilePath Returns an empty string as we use SQLite now. Kept for backwards compatibility
func GetDefaultHistoryFilePath() (string, error) {
	return "", nil
}
