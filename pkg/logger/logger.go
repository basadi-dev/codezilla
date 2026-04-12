package logger

import (
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
)

// Logger wraps slog.Logger with additional functionality
type Logger struct {
	slogger *slog.Logger
	level   slog.Level
	silent  bool
	writer  io.Writer
	file    *os.File
}

// Config contains logger configuration
type Config struct {
	LogFile  string
	LogLevel string
	Silent   bool
}

// LogLevel represents logging levels
type LogLevel string

const (
	// Log levels
	LevelDebug LogLevel = "debug"
	LevelInfo  LogLevel = "info"
	LevelWarn  LogLevel = "warn"
	LevelError LogLevel = "error"
)

// New creates a new Logger instance
func New(config Config) (*Logger, error) {
	level := slog.LevelInfo
	switch LogLevel(config.LogLevel) {
	case LevelDebug:
		level = slog.LevelDebug
	case LevelInfo:
		level = slog.LevelInfo
	case LevelWarn:
		level = slog.LevelWarn
	case LevelError:
		level = slog.LevelError
	}

	// Create a writer for logs
	var writer io.Writer
	var file *os.File

	if config.LogFile != "" {
		// Ensure directory exists
		dir := filepath.Dir(config.LogFile)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, err
		}

		var err error
		file, err = os.OpenFile(config.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return nil, err
		}

		// Always write to file only
		writer = file
	} else {
		// If no log file specified and not silent, use stdout
		if !config.Silent {
			writer = os.Stdout
		} else {
			// Use a discard writer if silent and no log file
			writer = io.Discard
		}
	}

	// Create Text handler instead of JSON for human readability
	handler := slog.NewTextHandler(writer, &slog.HandlerOptions{
		Level: level,
	})

	return &Logger{
		slogger: slog.New(handler),
		level:   level,
		silent:  config.Silent,
		writer:  writer,
		file:    file,
	}, nil
}

// Close closes the logger file if it exists
func (l *Logger) Close() error {
	if l.file != nil {
		return l.file.Close()
	}
	return nil
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, args ...any) {
	l.slogger.Debug(msg, args...)
}

// DumpJSON explicitly writes a nicely indented JSON block bypassing structural single-line quoting
func (l *Logger) DumpJSON(msg string, data any) {
	if l.level > slog.LevelDebug {
		return
	}
	l.slogger.Debug(msg)
	b, err := json.MarshalIndent(data, "", "  ")
	if err == nil {
		if l.file != nil {
			_, _ = l.file.Write(append(b, '\n'))
		} else if l.writer != nil && !l.silent {
			_, _ = l.writer.Write(append(b, '\n'))
		}
	}
}

// Info logs an info message
func (l *Logger) Info(msg string, args ...any) {
	l.slogger.Info(msg, args...)
}

// Warn logs a warning message
func (l *Logger) Warn(msg string, args ...any) {
	l.slogger.Warn(msg, args...)
}

// Error logs an error message
func (l *Logger) Error(msg string, args ...any) {
	l.slogger.Error(msg, args...)
}

// With returns a new Logger with the given attributes added to each log entry
func (l *Logger) With(args ...any) *Logger {
	return &Logger{
		slogger: l.slogger.With(args...),
		level:   l.level,
		silent:  l.silent,
		writer:  l.writer,
		file:    l.file,
	}
}

// WithGroup returns a new Logger with the given group added to each log entry
func (l *Logger) WithGroup(name string) *Logger {
	return &Logger{
		slogger: l.slogger.WithGroup(name),
		level:   l.level,
		silent:  l.silent,
		writer:  l.writer,
		file:    l.file,
	}
}

// DefaultLogger creates a basic logger that writes to stdout
func DefaultLogger() *Logger {
	handler := slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})

	return &Logger{
		slogger: slog.New(handler),
		level:   slog.LevelInfo,
		writer:  os.Stdout,
	}
}
