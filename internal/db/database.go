package db

import (
	"context"
	"database/sql"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

//go:embed sql/schema.sql
var schemaSQL string

// DB wraps the sql.DB connection and provides database operations
type DB struct {
	db *sql.DB
	mu sync.RWMutex
}

// Config holds database configuration
type Config struct {
	// Path to the database file (default: ~/.codezilla/codezilla.db)
	Path string
	// Enable query logging
	LogQueries bool
	// Connection timeout
	Timeout time.Duration
	// Enable foreign keys
	ForeignKeys bool
}

// DefaultConfig returns a default database configuration
func DefaultConfig() *Config {
	// Get home directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		homeDir = "."
	}

	dbPath := filepath.Join(homeDir, ".codezilla", "codezilla.db")

	return &Config{
		Path:        dbPath,
		LogQueries:  false,
		Timeout:     30 * time.Second,
		ForeignKeys: true,
	}
}

// New creates a new database connection
func New(cfg *Config) (*DB, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	// Ensure directory exists
	dir := filepath.Dir(cfg.Path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create database directory: %w", err)
	}

	// Open database connection
	db, err := sql.Open("sqlite", cfg.Path)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Set connection pool settings
	db.SetMaxOpenConns(1) // SQLite works best with single writer
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(time.Hour)

	// Set timeout
	if cfg.Timeout > 0 {
		db.SetConnMaxLifetime(cfg.Timeout)
	}

	// Enable foreign keys if requested
	if cfg.ForeignKeys {
		if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
			db.Close()
			return nil, fmt.Errorf("failed to enable foreign keys: %w", err)
		}
	}

	// Enable WAL mode for better concurrency
	if _, err := db.Exec("PRAGMA journal_mode = WAL"); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to enable WAL mode: %w", err)
	}

	database := &DB{
		db: db,
	}

	return database, nil
}

// Close closes the database connection
func (d *DB) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.db != nil {
		return d.db.Close()
	}
	return nil
}

// Conn returns the underlying sql.DB connection
func (d *DB) Conn() *sql.DB {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.db
}

// Ping checks if the database connection is alive
func (d *DB) Ping() error {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.db.Ping()
}

// BeginTx starts a new transaction
func (d *DB) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.db.BeginTx(ctx, opts)
}

// WithTx executes a function within a transaction
func (d *DB) WithTx(ctx context.Context, fn func(tx *sql.Tx) error) error {
	tx, err := d.BeginTx(ctx, nil)
	if err != nil {
		return err
	}

	defer func() {
		if p := recover(); p != nil {
			tx.Rollback()
			panic(p)
		}
	}()

	if err := fn(tx); err != nil {
		if rbErr := tx.Rollback(); rbErr != nil {
			return fmt.Errorf("rollback error: %v, original error: %w", rbErr, err)
		}
		return err
	}

	return tx.Commit()
}

// Initialize runs database migrations and setup.
// The schema is embedded at compile time, so this works regardless of
// the working directory the binary is launched from.
func (d *DB) Initialize(ctx context.Context) error {
	if _, err := d.db.ExecContext(ctx, schemaSQL); err != nil {
		return fmt.Errorf("failed to execute schema: %w", err)
	}
	return nil
}

// Health checks database health
func (d *DB) Health(ctx context.Context) error {
	if err := d.Ping(); err != nil {
		return fmt.Errorf("database ping failed: %w", err)
	}

	// Check foreign keys
	var fkEnabled int
	err := d.db.QueryRowContext(ctx, "PRAGMA foreign_keys").Scan(&fkEnabled)
	if err != nil {
		return fmt.Errorf("failed to check foreign keys: %w", err)
	}
	if fkEnabled != 1 {
		return fmt.Errorf("foreign keys not enabled")
	}

	return nil
}
