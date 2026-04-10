package ui

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/mattn/go-runewidth"
	"golang.org/x/term"
)

// FixedInput implements InputReader with simple, reliable single-line input
type FixedInput struct {
	prompt       string
	reader       *bufio.Reader
	historyFile  string
	history      []string
	historyIndex int
	mu           sync.Mutex
	rawMode      bool
	fd           int
	currentLines int // Track how many lines the current input spans
}

// SetPrompt updates the prompt string
func (fi *FixedInput) SetPrompt(prompt string) {
	fi.prompt = prompt
}

// stripANSI removes ANSI escape sequences from a string
func stripANSI(str string) string {
	ansiRegex := regexp.MustCompile(`\x1b\[[0-9;]*m`)
	return ansiRegex.ReplaceAllString(str, "")
}

// promptDisplayWidth calculates the actual display width of the prompt
// accounting for ANSI codes and multi-width characters like emoji
func (fi *FixedInput) promptDisplayWidth() int {
	// Remove ANSI escape sequences first
	clean := stripANSI(fi.prompt)
	// Calculate display width using runewidth
	return runewidth.StringWidth(clean)
}

// NewFixedInput creates a new input reader with history support but no multi-line bugs
func NewFixedInput(prompt string, historyFile string) (*FixedInput, error) {
	fd := int(os.Stdin.Fd())

	input := &FixedInput{
		prompt:       prompt,
		reader:       bufio.NewReader(os.Stdin),
		historyFile:  historyFile,
		history:      make([]string, 0, 500),
		historyIndex: -1,
		fd:           fd,
		rawMode:      term.IsTerminal(fd),
		currentLines: 1,
	}

	// Load history from file if it exists
	if historyFile != "" {
		input.loadHistory()
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

	// Print prompt
	fmt.Print(fi.prompt)

	// Buffer for the current line
	var line []rune
	pos := 0

	// History navigation state
	fi.mu.Lock()
	fi.historyIndex = len(fi.history)
	historySize := len(fi.history)
	fi.mu.Unlock()
	savedLine := ""

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
			fmt.Print("\r\n")
			result := string(line)
			if result != "" {
				fi.addHistory(result)
			}
			fi.currentLines = 1 // Reset for next input
			return result, nil

		case 0x03: // Ctrl-C
			fmt.Print("^C\r\n")
			fi.currentLines = 1 // Reset for next input
			return "", io.EOF

		case 0x04: // Ctrl-D
			if len(line) == 0 {
				fmt.Print("\r\n")
				fi.currentLines = 1 // Reset for next input
				return "", io.EOF
			}
			// Delete character at cursor
			if pos < len(line) {
				line = append(line[:pos], line[pos+1:]...)
				fi.redrawLine(line, pos)
			}

		case 0x01: // Ctrl-A - beginning of line
			pos = 0
			fi.redrawLine(line, pos)

		case 0x05: // Ctrl-E - end of line
			pos = len(line)
			fi.redrawLine(line, pos)

		case 0x0B: // Ctrl-K - kill to end of line
			if pos < len(line) {
				line = line[:pos]
				fi.redrawLine(line, pos)
			}

		case 0x15: // Ctrl-U - kill to beginning of line
			line = line[pos:]
			pos = 0
			fi.redrawLine(line, pos)

		case 0x17: // Ctrl-W - delete word backward
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
				case 'A': // Up arrow - previous history
					if historySize > 0 && fi.historyIndex > 0 {
						// Save current line if first time
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

				case 'B': // Down arrow - next history
					if historySize > 0 && fi.historyIndex < historySize {
						fi.historyIndex++
						if fi.historyIndex == historySize {
							// Restore saved line
							line = []rune(savedLine)
						} else {
							fi.mu.Lock()
							line = []rune(fi.history[fi.historyIndex])
							fi.mu.Unlock()
						}
						pos = len(line)
						fi.redrawLine(line, pos)
					}

				case 'C': // Right arrow
					if pos < len(line) {
						pos++
						fi.redrawLine(line, pos)
					}

				case 'D': // Left arrow
					if pos > 0 {
						pos--
						fi.redrawLine(line, pos)
					}

				case 'H': // Home
					pos = 0
					fi.redrawLine(line, pos)

				case 'F': // End
					pos = len(line)
					fi.redrawLine(line, pos)

				case '3': // Delete key (sequence is ESC[3~)
					extra := make([]byte, 1)
					if _, err := fi.reader.Read(extra); err != nil {
						// Continue on error
						continue
					}
					if extra[0] == '~' && pos < len(line) {
						line = append(line[:pos], line[pos+1:]...)
						fi.redrawLine(line, pos)
					}
				}
			}

		default:
			// Regular character
			if b[0] >= 32 && b[0] < 127 {
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
	currentLines := fi.currentLines

	// Move cursor to the beginning of the input area
	if currentLines > 1 {
		// Move up to the first line of input
		fmt.Printf("\033[%dA", currentLines-1)
	}
	fmt.Print("\r")

	// Clear all lines we previously used
	for i := 0; i < currentLines; i++ {
		fmt.Print("\033[K") // Clear line
		if i < currentLines-1 {
			fmt.Print("\n")
		}
	}

	// Move back to start
	if currentLines > 1 {
		fmt.Printf("\033[%dA", currentLines-1)
	}
	fmt.Print("\r")

	// Print prompt
	fmt.Print(fi.prompt)

	// Print the line with wrapping
	text := string(line)
	fmt.Print(text)

	// Update the number of lines we're using
	fi.currentLines = numLines

	// Calculate cursor position with wrapping
	absPos := promptLen + pos
	cursorLine := absPos / termWidth
	cursorCol := absPos % termWidth

	// Move cursor to correct position
	// First, figure out where we are after printing
	totalPrintedLen := promptLen + len(line)
	endLine := 0
	if totalPrintedLen > 0 {
		endLine = (totalPrintedLen - 1) / termWidth
	}

	// Move from end position to cursor position
	if endLine > cursorLine {
		// Move up
		fmt.Printf("\033[%dA", endLine-cursorLine)
	}

	// Move to beginning of line and then to cursor column
	fmt.Print("\r")
	if cursorCol > 0 {
		fmt.Printf("\033[%dC", cursorCol)
	}
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
	if fi.historyFile != "" {
		if err := fi.saveHistory(); err != nil {
			return fmt.Errorf("failed to save history: %w", err)
		}
	}
	return nil
}

// History management

func (fi *FixedInput) loadHistory() {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	file, err := os.Open(fi.historyFile)
	if err != nil {
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			fi.history = append(fi.history, line)
		}
	}
}

func (fi *FixedInput) saveHistory() error {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	// Ensure directory exists
	dir := filepath.Dir(fi.historyFile)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	file, err := os.Create(fi.historyFile)
	if err != nil {
		return err
	}
	defer file.Close()

	// Keep last 500 entries
	start := 0
	if len(fi.history) > 500 {
		start = len(fi.history) - 500
	}

	for i := start; i < len(fi.history); i++ {
		fmt.Fprintln(file, fi.history[i])
	}

	return nil
}

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
		if err := fi.saveHistory(); err != nil {
			// Log error but don't block
			fmt.Fprintf(os.Stderr, "Failed to save history: %v\n", err)
		}
	}()
}

// GetDefaultHistoryFilePath returns the default path for the command history file
func GetDefaultHistoryFilePath() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	configDir := filepath.Join(homeDir, ".config", "codezilla")
	return filepath.Join(configDir, "history"), nil
}
