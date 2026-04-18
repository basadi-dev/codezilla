package tools

import (
	"fmt"
	"os"
	"strings"

	"codezilla/pkg/style"

	"github.com/charmbracelet/lipgloss"
	"github.com/sergi/go-diff/diffmatchpatch"
	"golang.org/x/term"
)

// GenerateDiff creates a smart diff output between the two content strings
func GenerateDiff(contentA, contentB string, contextLines int) string {
	dmp := diffmatchpatch.New()
	text1, text2, lineArray := dmp.DiffLinesToChars(contentA, contentB)
	diffs := dmp.DiffMain(text1, text2, false)
	diffs = dmp.DiffCharsToLines(diffs, lineArray)

	// If there are no differences, say so
	if len(diffs) == 0 || (len(diffs) == 1 && diffs[0].Type == diffmatchpatch.DiffEqual) {
		text := "No differences found."
		if style.UseColors {
			return lipgloss.NewStyle().Foreground(lipgloss.Color("#FFFFFF")).Render(text)
		}
		return text
	}

	type DiffLine struct {
		Type     diffmatchpatch.Operation
		Text     string
		LineNumA int
		LineNumB int
	}

	var allLines []DiffLine
	currentA := 1
	currentB := 1

	for _, diff := range diffs {
		lines := splitLines(diff.Text)
		for _, l := range lines {
			dl := DiffLine{
				Type: diff.Type,
				Text: l,
			}
			if diff.Type == diffmatchpatch.DiffEqual {
				dl.LineNumA = currentA
				dl.LineNumB = currentB
				currentA++
				currentB++
			} else if diff.Type == diffmatchpatch.DiffDelete {
				dl.LineNumA = currentA
				currentA++
			} else if diff.Type == diffmatchpatch.DiffInsert {
				dl.LineNumB = currentB
				currentB++
			}
			allLines = append(allLines, dl)
		}
	}

	// Mark lines to show (context window)
	showMap := make([]bool, len(allLines))
	for i, dl := range allLines {
		if dl.Type != diffmatchpatch.DiffEqual {
			start := i - contextLines
			if start < 0 {
				start = 0
			}
			end := i + contextLines
			if end >= len(allLines) {
				end = len(allLines) - 1
			}
			for j := start; j <= end; j++ {
				showMap[j] = true
			}
		}
	}

	// Calculate line number padding
	maxLineNum := max(currentA, currentB)
	lineNumWidth := len(fmt.Sprintf("%d", maxLineNum))

	formatLineNum := func(num int) string {
		if num == 0 {
			return strings.Repeat(" ", lineNumWidth)
		}
		return fmt.Sprintf("%"+fmt.Sprintf("%d", lineNumWidth)+"d", num)
	}

	// Pre-define styles
	styleRed := lipgloss.NewStyle()
	styleGreen := lipgloss.NewStyle()
	styleDim := lipgloss.NewStyle()
	styleLineNum := lipgloss.NewStyle()
	styleHeaderLeft := lipgloss.NewStyle()
	styleHeaderRight := lipgloss.NewStyle()

	if style.UseColors {
		styleRed = styleRed.Foreground(lipgloss.Color("#FF5F87"))
		styleGreen = styleGreen.Foreground(lipgloss.Color("#00D787"))
		styleDim = styleDim.Foreground(lipgloss.Color("#626262"))
		styleLineNum = styleLineNum.Foreground(lipgloss.Color("#626262"))
		styleHeaderLeft = styleHeaderLeft.Foreground(lipgloss.Color("#FF5F87")).Bold(true)
		styleHeaderRight = styleHeaderRight.Foreground(lipgloss.Color("#00D787")).Bold(true)
	}

	// Calculate available width for text
	width, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil || width <= 0 {
		width = 120 // fallback width
	}

	// Prefix width: lineNumA + " │ " + lineNumB + " │ " = lineNumWidth*2 + 6
	prefixWidth := lineNumWidth*2 + 6
	maxTextWidth := width - prefixWidth
	if maxTextWidth < 20 {
		maxTextWidth = 20
	}

	var result strings.Builder

	// Header line aligned with content columns
	// Format: "  lnA │ lnB │ text"
	// Header: "  - │ + │"
	headerLnA := styleHeaderLeft.Render(strings.Repeat("-", lineNumWidth))
	headerLnB := styleHeaderRight.Render(strings.Repeat("+", lineNumWidth))
	result.WriteString(fmt.Sprintf("%s │ %s │\n", headerLnA, headerLnB))

	var outputLines []string
	wasShowing := false

	for i, dl := range allLines {
		if !showMap[i] {
			if wasShowing {
				// Separator line matching prefix width
				sep := strings.Repeat("╌", prefixWidth)
				outputLines = append(outputLines, styleDim.Render(sep))
			}
			wasShowing = false
			continue
		}
		wasShowing = true

		// Truncate long lines
		text := truncateRight(dl.Text, maxTextWidth-1)

		if dl.Type == diffmatchpatch.DiffEqual {
			lnA := styleLineNum.Render(formatLineNum(dl.LineNumA))
			lnB := styleLineNum.Render(formatLineNum(dl.LineNumB))
			outputLines = append(outputLines, fmt.Sprintf("%s │ %s │ %s", lnA, lnB, styleDim.Render(" "+text)))
		} else if dl.Type == diffmatchpatch.DiffDelete {
			lnA := styleLineNum.Render(formatLineNum(dl.LineNumA))
			lnB := styleLineNum.Render(formatLineNum(0))
			styledText := styleRed.Render("-" + text)
			outputLines = append(outputLines, fmt.Sprintf("%s │ %s │ %s", lnA, lnB, styledText))
		} else if dl.Type == diffmatchpatch.DiffInsert {
			lnA := styleLineNum.Render(formatLineNum(0))
			lnB := styleLineNum.Render(formatLineNum(dl.LineNumB))
			styledText := styleGreen.Render("+" + text)
			outputLines = append(outputLines, fmt.Sprintf("%s │ %s │ %s", lnA, lnB, styledText))
		}
	}

	result.WriteString(strings.Join(outputLines, "\n"))

	return result.String()
}

func splitLines(s string) []string {
	var res []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			res = append(res, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		res = append(res, s[start:])
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func truncateRight(s string, max int) string {
	if max <= 0 {
		return s
	}
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	if max > 3 {
		return string(runes[:max-3]) + "..."
	}
	return string(runes[:max])
}