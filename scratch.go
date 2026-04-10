package main

import (
	"fmt"
	"strings"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
)

func addTableSpacing(md string) string {
    lines := strings.Split(md, "\n")
    var out []string
    
    // Helper to check if string looks like a markdown table separator
    isSeparator := func(s string) bool {
        s = strings.TrimSpace(s)
        if !strings.HasPrefix(s, "|") || !strings.HasSuffix(s, "|") {
            return false
        }
        // removing everything but dashes and colons and pipes
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
            // Is this a separator?
            if isSeparator(line) {
                continue
            }
            
            // Is the next line a separator?
            if i+1 < len(lines) && isSeparator(lines[i+1]) {
                continue
            }
            
            // It's a data row! Append an empty row.
            trimmed := strings.TrimSpace(line)
            pipes := strings.Count(trimmed, "|")
            if pipes >= 2 {
                emptyCols := strings.Repeat("   |", pipes-1)
                out = append(out, "|" + emptyCols)
            }
        }
    }
    return strings.Join(out, "\n")
}

func main() {
    md := `Some text.

| What it is | A modular, command-line AI coding assistant |
|---|---|
| Core idea | Let a local LLM chat with you in the terminal. |
| LLM infra | Uses the Ollama API. |

Some more text.
`
    processed := addTableSpacing(md)
	renderer, _ := glamour.NewTermRenderer(
		glamour.WithStyles(styles.DarkStyleConfig),
		glamour.WithWordWrap(80),
	)
	rendered, _ := renderer.Render(processed)
	fmt.Println(rendered)
}
