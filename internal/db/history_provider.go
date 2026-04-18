package db

import (
	"context"
	"database/sql"

	"codezilla/internal/db/generated"
)

// SQLHistoryProvider implements ui.HistoryProvider using SQLite
type SQLHistoryProvider struct {
	db      *DB
	queries *generated.Queries
}

// NewSQLHistoryProvider creates a new SQLHistoryProvider
func NewSQLHistoryProvider(db *DB) *SQLHistoryProvider {
	return &SQLHistoryProvider{
		db:      db,
		queries: generated.New(db.Conn()),
	}
}

// AddHistory adds a new prompt entry to the database
func (p *SQLHistoryProvider) AddHistory(prompt string) error {
	return p.queries.AddHistory(context.Background(), generated.AddHistoryParams{
		Prompt:    prompt,
		SessionID: sql.NullString{}, // Global history by default
	})
}

// GetHistory returns the most recent n history entries 
func (p *SQLHistoryProvider) GetHistory(n int) []string {
	limit := int64(n)
	if limit <= 0 {
		limit = -1 // SQLite limit -1 means practically all rows
	}
	prompts, err := p.queries.GetRecentHistory(context.Background(), limit)
	if err != nil {
		return nil
	}
	// Return in oldest-first order so up arrow goes back in time smoothly
	for i, j := 0, len(prompts)-1; i < j; i, j = i+1, j-1 {
		prompts[i], prompts[j] = prompts[j], prompts[i]
	}
	return prompts
}

// SearchHistory returns history entries that contain the query substring.
func (p *SQLHistoryProvider) SearchHistory(query string) []string {
	prompts, err := p.queries.SearchHistory(context.Background(), sql.NullString{String: query, Valid: true})
	if err != nil {
		return nil
	}
	return prompts
}

// ClearHistory removes all history entries
func (p *SQLHistoryProvider) ClearHistory() error {
	return p.queries.ClearHistory(context.Background())
}
