package core

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
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
	} else if sub == "resume" {
		if len(parts) < 3 {
			app.ui.Warning("Usage: /session resume <filename>")
			return
		}
		app.handleSessionResume(ctx, parts[2])
	} else {
		app.ui.Warning("Usage: /session [ls | play <filename> | resume <filename>]")
	}
}

type sessionSummary struct {
	Filename string
	Time     time.Time
	Duration time.Duration
	Snippet  string
}

func formatDuration(d time.Duration) string {
	if d < time.Second {
		return "<1s"
	}
	d = d.Round(time.Second)
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second

	if h > 0 {
		return fmt.Sprintf("%dh %dm", h, m)
	}
	if m > 0 {
		return fmt.Sprintf("%dm %ds", m, s)
	}
	return fmt.Sprintf("%ds", s)
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
		var firstTs, lastTs int64

		if sf, err := os.Open(path); err == nil {
			scanner := bufio.NewScanner(sf)
			for scanner.Scan() {
				var evt session.Event
				if json.Unmarshal(scanner.Bytes(), &evt) == nil {
					if firstTs == 0 {
						firstTs = evt.TimestampNano
					}
					lastTs = evt.TimestampNano

					if evt.Type == session.EventUIInput && snippet == "<No UI input recorded>" {
						if text, ok := evt.Data["text"].(string); ok {
							if len(text) > 50 {
								snippet = text[:47] + "..."
							} else {
								snippet = text
							}
						}
					}
				}
			}
			sf.Close()
		}

		duration := time.Duration(0)
		if firstTs > 0 && lastTs > firstTs {
			duration = time.Duration(lastTs - firstTs)
		}

		sessionTime := info.ModTime()
		if firstTs > 0 {
			sessionTime = time.Unix(0, firstTs)
		}

		summaries = append(summaries, sessionSummary{
			Filename: f.Name(),
			Time:     sessionTime,
			Duration: duration,
			Snippet:  strings.ReplaceAll(snippet, "\n", " "),
		})
	}

	sort.Slice(summaries, func(i, j int) bool {
		return summaries[i].Time.After(summaries[j].Time)
	})

	var top []sessionSummary
	for i, s := range summaries {
		if i >= 15 {
			break
		}
		top = append(top, s)
	}

	for i := 0; i < len(top)/2; i++ {
		j := len(top) - i - 1
		top[i], top[j] = top[j], top[i]
	}

	app.ui.Println("\n%s", lipgloss.NewStyle().Bold(true).Render("Available Sessions:"))
	for _, s := range top {
		durationStr := lipgloss.NewStyle().Foreground(lipgloss.Color("4")).Render(formatDuration(s.Duration))
		app.ui.Println("  %s  %-22s  %-6s  %s",
			lipgloss.NewStyle().Foreground(lipgloss.Color("3")).Render("ID"),
			s.Filename,
			durationStr,
			lipgloss.NewStyle().Faint(true).Render(s.Snippet))
	}
	app.ui.Print("\n")
}

func (app *App) handleSessionResume(ctx context.Context, sessionID string) {
	targetPath := sessionID
	if !strings.Contains(sessionID, string(os.PathSeparator)) {
		if !strings.HasSuffix(sessionID, ".jsonl") {
			sessionID += ".jsonl"
		}
		targetPath = filepath.Join(app.config.SessionEventsDir, sessionID)
	}

	app.ui.Info("\nResuming session: %s...", sessionID)

	// Close old recorder
	if app.sessionRecord != nil {
		app.sessionRecord.Close()
	}

	// Wipe context
	app.agent.ClearContext()

	// Load previous events
	events, err := session.LoadEvents(targetPath)
	if err == nil {
		var currentAssistantResponse strings.Builder
		for _, evt := range events {
			if evt.Type == session.EventUIInput {
				if currentAssistantResponse.Len() > 0 {
					app.agent.AddAssistantMessage(currentAssistantResponse.String())
					currentAssistantResponse.Reset()
				}
				if text, ok := evt.Data["text"].(string); ok {
					app.agent.AddUserMessage(text)
				}
			} else if evt.Type == session.EventToken {
				if token, ok := evt.Data["token"].(string); ok {
					currentAssistantResponse.WriteString(token)
				}
			} else if evt.Type == session.EventToolStart {
				if currentAssistantResponse.Len() > 0 {
					app.agent.AddAssistantMessage(currentAssistantResponse.String())
					currentAssistantResponse.Reset()
				}
			}
		}
		if currentAssistantResponse.Len() > 0 {
			app.agent.AddAssistantMessage(currentAssistantResponse.String())
		}
		app.ui.Success("Session context restored.")

		msgs := app.agent.GetMessages()
		for _, msg := range msgs {
			if msg.Role == "user" {
				app.ui.Print("\n%s\n", app.ui.GetTheme().StyleBlue.Render("👤 You:"))
				app.ui.Print("%s\n", msg.Content)
			} else if msg.Role == "assistant" {
				content := msg.Content
				if content != "" {
					app.ui.ShowResponse(content)
				}
			}
		}
		if len(msgs) > 0 {
			app.ui.Print("\n")
		}
	} else {
		app.ui.Warning("Failed to load session history: %v", err)
	}

	// Create new recorder appending to the target path
	recorder, err := session.NewRecorder(targetPath, app.logger)
	if err != nil {
		app.ui.Error("Failed to initialize session recorder: %v", err)
	} else {
		app.sessionRecord = recorder
		// Ensure agent points to new recorder
		if agentInstance, ok := app.agent.(interface{ SetSessionRecorder(*session.Recorder) }); ok {
			agentInstance.SetSessionRecorder(recorder)
		}
	}
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
