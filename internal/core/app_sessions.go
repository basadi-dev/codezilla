package core

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"codezilla/internal/session"
	"codezilla/internal/ui"
	"github.com/charmbracelet/lipgloss"
)

func (app *App) handleSessionCommand(ctx context.Context, parts []string) {
	if len(parts) < 2 {
		app.listSessions()
		return
	}
	sub := strings.ToLower(parts[1])
	if sub == "ls" || sub == "list" {
		app.listSessions()
	} else if sub == "play" || sub == "replay" {
		if len(parts) < 3 {
			app.ui.Warning("Usage: /session play <filename>")
			return
		}
		app.playSessionCtx(ctx, parts[2])
	} else {
		app.ui.Warning("Usage: /session [ls | play <filename>]")
	}
}

type sessionSummary struct {
	Filename string
	Time     time.Time
	Snippet  string
}

func (app *App) listSessions() {
	dir := app.config.SessionEventsDir
	if dir == "" {
		app.ui.Error("SessionEventsDir is not configured.")
		return
	}
	files, err := os.ReadDir(dir)
	if err != nil {
		app.ui.Error("Failed to read sessions: %v", err)
		return
	}

	var summaries []sessionSummary
	for _, f := range files {
		if f.IsDir() || !strings.HasSuffix(f.Name(), ".jsonl") {
			continue
		}
		info, err := f.Info()
		if err != nil {
			continue
		}

		path := filepath.Join(dir, f.Name())
		snippet := "<No UI input recorded>"

		// Extract first UI Input
		if sf, err := os.Open(path); err == nil {
			scanner := bufio.NewScanner(sf)
			for scanner.Scan() {
				var evt session.Event
				if json.Unmarshal(scanner.Bytes(), &evt) == nil && evt.Type == session.EventUIInput {
					if text, ok := evt.Data["text"].(string); ok {
						if len(text) > 50 {
							snippet = text[:47] + "..."
						} else {
							snippet = text
						}
						break
					}
				}
			}
			sf.Close()
		}

		summaries = append(summaries, sessionSummary{
			Filename: f.Name(),
			Time:     info.ModTime(),
			Snippet:  strings.ReplaceAll(snippet, "\n", " "),
		})
	}

	sort.Slice(summaries, func(i, j int) bool {
		return summaries[i].Time.After(summaries[j].Time)
	})

	app.ui.Println("\n%s", lipgloss.NewStyle().Bold(true).Render("Available Sessions:"))
	for i, s := range summaries {
		if i >= 15 {
			break
		}
		app.ui.Println("  %s  %-22s  %s", lipgloss.NewStyle().Foreground(lipgloss.Color("3")).Render("ID"), s.Filename, lipgloss.NewStyle().Faint(true).Render(s.Snippet))
	}
	app.ui.Print("\n")
}

func (app *App) playSessionCtx(ctx context.Context, sessionID string) {
	targetPath := sessionID
	if !strings.Contains(sessionID, string(os.PathSeparator)) {
		if !strings.HasSuffix(sessionID, ".jsonl") {
			sessionID += ".jsonl"
		}
		targetPath = filepath.Join(app.config.SessionEventsDir, sessionID)
	}

	f, err := os.Open(targetPath)
	if err != nil {
		app.ui.Error("Cannot open session: %v", err)
		return
	}
	defer f.Close()

	app.ui.Info("Starting playback of %s (Press Ctrl+C to abort)...", sessionID)
	app.ui.Print("\n---\n\n")

	scanner := bufio.NewScanner(f)
	var lastTs int64

	for scanner.Scan() {
		// Check for abort
		select {
		case <-ctx.Done():
			app.ui.Warning("\nPlayback aborted.")
			return
		default:
		}

		var evt session.Event
		if err := json.Unmarshal(scanner.Bytes(), &evt); err != nil {
			continue
		}

		if lastTs > 0 {
			delta := evt.TimestampNano - lastTs
			if delta > 0 && delta < int64(3*time.Second) { // max cap
				time.Sleep(time.Duration(delta))
			}
		}
		lastTs = evt.TimestampNano

		switch evt.Type {
		case session.EventUIInput:
			text, _ := evt.Data["text"].(string)
			app.ui.Println("\n\n%s\n%s", app.ui.GetTheme().StyleBlue.Render("👤 You:"), text)
		case session.EventToken:
			app.ui.Print("%s", evt.Data["token"])
		case session.EventToolStart:
			app.ui.Print("\n  %s %s", lipgloss.NewStyle().Foreground(lipgloss.Color("3")).Render("🔧 Invoking:"), evt.Data["tool"])
		case session.EventToolResult:
			if errStr, _ := evt.Data["error"].(string); errStr != "" {
				app.ui.Print("\n  %s %s failed -> %s\n", lipgloss.NewStyle().Foreground(lipgloss.Color("1")).Render("❌"), evt.Data["tool"], errStr)
			} else {
				app.ui.Print("\n  %s %s inside %s\n", lipgloss.NewStyle().Foreground(lipgloss.Color("2")).Render("✅"), evt.Data["tool"], evt.Data["duration"])
			}
		}
	}
	app.ui.Success("\n\n--- Playback finished.")
}

// sessionCompletions returns a list of completion candidates for active session files
func (app *App) sessionCompletions() []ui.Completion {
	dir := app.config.SessionEventsDir
	if dir == "" {
		return nil
	}
	
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}

	var opts []ui.Completion
	for _, f := range files {
		if f.IsDir() || !strings.HasSuffix(f.Name(), ".jsonl") {
			continue
		}
		opts = append(opts, ui.Completion{
			Text: f.Name(),
		})
	}
	
	// Better to sort them descending (newest first) but since file string is date we can just sort descending
	sort.Slice(opts, func(i, j int) bool {
		return opts[i].Text > opts[j].Text
	})
	
	return opts
}
