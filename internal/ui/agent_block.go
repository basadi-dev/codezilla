package ui

// agent_block.go — reusable live agent progress block component.
//
// Each parallel worker gets a self-contained viewport block:
//
//	[*] 1. [Researcher]  analyze auth module         (12s)  (qwen2.5)
//	     called grepSearch
//	     searching for JWT validation in middleware a
//	     nd auth/token.go — found TokenParser struct
//
// The 3 lines below the header act as a mini scrolling viewport: the last
// blockActivityLines display-width chunks of displayBuf are shown.
//
// displayBuf is a single rolling string:
//   - tool-call events are appended as "called: <name>\n" (newline = chunk boundary)
//   - stream tokens are appended raw (no forced newline)
//
// This means the display always reflects the most recent activity regardless
// of whether the LLM is emitting newlines or not, solving the main problem
// with commit-on-newline approaches.
//
// States:
//
//	[*]  running  (ASCII spinner, wall-clock driven)
//	[+]  done OK  (green)
//	[!]  error    (red)

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// ── Block geometry ────────────────────────────────────────────────────────────

// blockActivityLines is the number of rolling log lines shown under the header.
const blockActivityLines = 8

// blockTotalLines is the full height of a block: 1 header + N activity + 1 blank.
const blockTotalLines = blockActivityLines + 2

// displayBufMaxRunes caps the rolling display buffer so memory is bounded.
// ~800 runes covers several LLM paragraphs comfortably.
const displayBufMaxRunes = 800

// ── liveBlock ─────────────────────────────────────────────────────────────────

// liveBlock tracks a live worker block within appModel.outputLines.
type liveBlock struct {
	workerID    string
	number      int    // 1-based task number for display (0 = planner)
	role        string // "Researcher", "Developer", …
	model       string // LLM model name
	label       string // task description (truncated)
	headerIdx   int    // index of the header line in outputLines
	activityIdx int    // index of first activity line (header+1)

	// displayBuf is the single rolling content buffer for this block.
	// Tool calls contribute "called: <name>\n"; stream tokens are appended raw.
	// Capped at displayBufMaxRunes to bound memory.
	displayBuf string

	// Think-marker state machine
	inThink bool

	started time.Time
}

// ── Think-marker stripping ────────────────────────────────────────────────────

// stripThinkMarkers removes <think>/<thinking>/</think>/</thinking> delimiters
// from token fragments while keeping all content between them.
// inThink is passed by pointer so marker state persists across calls.
func stripThinkMarkers(token string, inThink *bool) string {
	var buf strings.Builder
	remain := token
	for len(remain) > 0 {
		if !*inThink {
			if idx := strings.Index(remain, "<thinking>"); idx != -1 {
				buf.WriteString(remain[:idx])
				remain = remain[idx+len("<thinking>"):]
				*inThink = true
			} else if idx := strings.Index(remain, "<think>"); idx != -1 {
				buf.WriteString(remain[:idx])
				remain = remain[idx+len("<think>"):]
				*inThink = true
			} else {
				buf.WriteString(remain)
				break
			}
		}
		if *inThink {
			if idx := strings.Index(remain, "</thinking>"); idx != -1 {
				buf.WriteString(remain[:idx])
				remain = remain[idx+len("</thinking>"):]
				*inThink = false
			} else if idx := strings.Index(remain, "</think>"); idx != -1 {
				buf.WriteString(remain[:idx])
				remain = remain[idx+len("</think>"):]
				*inThink = false
			} else {
				buf.WriteString(remain)
				break
			}
		}
	}
	return buf.String()
}

// ── Spinner chars ─────────────────────────────────────────────────────────────

var spinnerFrames = []string{"-", "\\", "|", "/"}

// spinnerFrame returns the current ASCII spinner character driven by wall clock
// so all running blocks animate in sync without per-block timers.
func spinnerFrame(t time.Time) string {
	elapsed := int(time.Since(t).Milliseconds() / 120)
	return spinnerFrames[elapsed%len(spinnerFrames)]
}

// ── Header rendering ──────────────────────────────────────────────────────────

// renderBlockHeader produces the header line for a live block.
func renderBlockHeader(blk *liveBlock, done, hasError bool, elapsed time.Duration) string {
	dim := lipgloss.NewStyle().Foreground(colMuted)

	colRed := lipgloss.AdaptiveColor{Light: "#F7768E", Dark: "#F7768E"}

	roleStyle := lipgloss.NewStyle().Foreground(colPurple).Bold(true)
	labelStyle := lipgloss.NewStyle().Foreground(colCyan)
	greenStyle := lipgloss.NewStyle().Foreground(colGreen)
	redStyle := lipgloss.NewStyle().Foreground(colRed)

	var stateIcon string
	switch {
	case done && hasError:
		stateIcon = redStyle.Render("[!]")
	case done:
		stateIcon = greenStyle.Render("[+]")
	default:
		frame := spinnerFrame(blk.started)
		stateIcon = lipgloss.NewStyle().Foreground(colCyan).Render("[" + frame + "]")
	}

	numStr := ""
	if blk.number > 0 {
		numStr = dim.Render(fmt.Sprintf("%d.", blk.number)) + " "
	}

	role := roleStyle.Render("[" + blk.role + "]")

	label := blk.label
	if label == "" {
		switch {
		case done && hasError:
			label = "failed"
		case done:
			label = "completed"
		default:
			label = "working"
		}
	}

	var elapsedStr string
	if done && elapsed > 0 {
		elapsedStr = dim.Render(fmt.Sprintf("(%s)", elapsed.Round(time.Second)))
	} else if !blk.started.IsZero() {
		elapsedStr = dim.Render(fmt.Sprintf("(%s)", time.Since(blk.started).Round(time.Second)))
	}

	modelStr := ""
	if blk.model != "" {
		modelStr = "  " + dim.Render("("+blk.model+")")
	}

	return "  " + stateIcon + " " + numStr + role + "  " + labelStyle.Render(label) + "  " + elapsedStr + modelStr
}

// ── Activity lines ────────────────────────────────────────────────────────────

// renderActivityLines renders the rolling display buffer as blockActivityLines
// plain, uniformly-styled lines — no row-type labels or prefixes.
//
// Algorithm:
//  1. Split displayBuf on '\n' to get segments (tool calls are each a segment;
//     stream tokens may span multiple segments if the LLM emits newlines).
//  2. Re-chunk each non-empty segment into runs of maxChars wide.
//  3. Take the last blockActivityLines chunks — this is what the user sees.
func renderActivityLines(blk *liveBlock, lineWidth int) [blockActivityLines]string {
	if lineWidth < 20 {
		lineWidth = 20
	}
	maxChars := lineWidth - 6 // 5-space indent + 1 safety margin
	if maxChars < 10 {
		maxChars = 10
	}

	lineStyle := lipgloss.NewStyle().Foreground(colMuted).Faint(true)
	indent := "     " // 5 spaces — aligns under the header text

	// Split on newlines to honour tool-call boundaries.
	segments := strings.Split(strings.ReplaceAll(blk.displayBuf, "\r", ""), "\n")

	// Re-chunk each segment into display-width slices.
	var chunks []string
	for _, seg := range segments {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}
		runes := []rune(seg)
		for len(runes) > 0 {
			end := maxChars
			if end > len(runes) {
				end = len(runes)
			}
			chunks = append(chunks, string(runes[:end]))
			runes = runes[end:]
		}
	}

	// Show only the last blockActivityLines chunks (newest at bottom).
	if len(chunks) > blockActivityLines {
		chunks = chunks[len(chunks)-blockActivityLines:]
	}

	var result [blockActivityLines]string
	for i := range result {
		if i < len(chunks) {
			result[i] = indent + lineStyle.Render(chunks[i])
		} else {
			result[i] = ""
		}
	}
	return result
}

// ── Block insertion helpers ───────────────────────────────────────────────────

// newBlockLines returns the full set of lines to insert into outputLines when
// a new block is created: header + blockActivityLines blank rows + gap line.
func newBlockLines(blk *liveBlock, lineWidth int) []string {
	header := renderBlockHeader(blk, false, false, 0)
	activity := renderActivityLines(blk, lineWidth)
	lines := make([]string, 0, blockTotalLines)
	lines = append(lines, header)
	lines = append(lines, activity[:]...)
	lines = append(lines, "") // blank separator between blocks
	return lines
}

// ── Buffer maintenance ────────────────────────────────────────────────────────

// capDisplayBuf trims displayBuf to at most displayBufMaxRunes runes, keeping
// the most recent content. Called after every write.
func capDisplayBuf(blk *liveBlock) {
	runes := []rune(blk.displayBuf)
	if len(runes) > displayBufMaxRunes {
		blk.displayBuf = string(runes[len(runes)-displayBufMaxRunes:])
	}
}

// appendActivityLine appends a discrete event (e.g. a tool call) to the
// rolling display buffer as its own line, followed by a newline so it forms
// a natural chunk boundary when rendered.
func appendActivityLine(blk *liveBlock, line string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}
	blk.displayBuf += line + "\n"
	capDisplayBuf(blk)
}

// appendStreamToken accumulates a streaming LLM token into the rolling display
// buffer. Think-markers are stripped. The token is appended raw — no newline
// is added — so the display updates character-by-character regardless of
// whether the LLM ever emits newlines.
func appendStreamToken(blk *liveBlock, token string) {
	clean := stripThinkMarkers(token, &blk.inThink)
	if clean == "" {
		return
	}
	blk.displayBuf += clean
	capDisplayBuf(blk)
}
