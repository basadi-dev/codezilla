package tools

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// ─── Per-file write lock ──────────────────────────────────────────────────────

// fileLocks provides per-file mutual exclusion for read-modify-write operations.
//
// Problem: The agent orchestrator executes multiple tool calls concurrently
// (goroutines). If two fileEdit or multiReplace calls target the same file in
// the same turn, both goroutines race through: read → compute replacement →
// write. The second write silently overwrites the first edit.
//
// Solution: Before any read-modify-write sequence, callers acquire the mutex
// for their target file path using acquireFileLock. The lock is released via
// the returned unlock function.
var fileLocks sync.Map

// acquireFileLock returns a per-file mutex locked for the given absolute path,
// and an unlock function that must be deferred by the caller.
//
//	unlock := acquireFileLock(path)
//	defer unlock()
func acquireFileLock(absPath string) func() {
	val, _ := fileLocks.LoadOrStore(absPath, &sync.Mutex{})
	mu := val.(*sync.Mutex)
	mu.Lock()
	return mu.Unlock
}

// ─── Read-before-write tracker ───────────────────────────────────────────────

// fileReadTimes maps absolute file paths to the time they were last read in
// this process. A file must be read before it can be edited — this enforces
// that the LLM always has fresh, accurate content in context before it tries
// to generate a target_content for fileEdit or multiReplace.
var fileReadTimes sync.Map // map[absPath string] → time.Time

// RecordFileRead marks absPath as having been read right now. Call this from
// every tool that reads a file (fileManage read, fileRead, etc.).
func RecordFileRead(absPath string) {
	fileReadTimes.Store(absPath, time.Now())
}

// EnforceReadBeforeWrite returns a non-nil error if the LLM is trying to edit
// a file it has not read in the current session, or if the file has been
// modified on disk since the last read.
//
// Why both checks?
//   - "Never read" → the LLM is likely hallucinating the content.
//   - "Modified since read" → an external tool (or a previous fileEdit in the
//     same turn) changed the file; the LLM's in-context copy is stale.
func EnforceReadBeforeWrite(absPath, toolName string) error {
	val, ok := fileReadTimes.Load(absPath)
	if !ok {
		return fmt.Errorf(
			"[%s] file %q has not been read yet in this session. "+
				"Read it first with fileManage (action:read) so that "+
				"target_content matches the current file contents exactly.",
			toolName, absPath)
	}

	lastRead := val.(time.Time)

	info, err := os.Stat(absPath)
	if err != nil {
		// File may not exist yet (new file) — that's fine for writes.
		return nil
	}

	// Truncate to second precision: some filesystems only have 1-second mtime resolution.
	modTime := info.ModTime().Truncate(time.Second)
	readTime := lastRead.Truncate(time.Second)

	if modTime.After(readTime) {
		return fmt.Errorf(
			"[%s] file %q was modified on disk (mtime %s) after it was last read (%s). "+
				"Re-read the file to get the current contents before editing.",
			toolName, absPath,
			modTime.Format(time.RFC3339),
			readTime.Format(time.RFC3339))
	}

	return nil
}

// ─── Version history / undo stack ────────────────────────────────────────────

const maxUndoDepth = 10 // maximum saved versions per file

// fileBackups maps absolute file paths to a slice of previous file contents
// (index 0 = oldest, last index = most recent backup).
var fileBackups sync.Map // map[absPath string] → []string

// PushBackup saves the current content of absPath before it is overwritten.
// Call this immediately before any os.WriteFile on a tracked file.
// If the stack already holds maxUndoDepth entries the oldest is dropped.
func PushBackup(absPath, content string) {
	backupStore.push(absPath, content)
}

// PopBackup retrieves and removes the most-recent backup for absPath.
// Returns the content and true on success, or ("", false) if no backup exists.
func PopBackup(absPath string) (string, bool) {
	return backupStore.pop(absPath)
}

// UndoDepth returns the number of stored backups for absPath.
func UndoDepth(absPath string) int {
	return backupStore.depth(absPath)
}

// backupStore is the concrete storage for the undo history. Using a dedicated
// type (rather than sync.Map directly) lets us protect each per-file slice
// with its own mutex instead of a coarse global lock.
var backupStore = &undoRegistry{stacks: make(map[string]*fileStack)}

type fileStack struct {
	mu      sync.Mutex
	entries []string
}

type undoRegistry struct {
	mu     sync.RWMutex
	stacks map[string]*fileStack
}

func (r *undoRegistry) get(absPath string) *fileStack {
	r.mu.RLock()
	s, ok := r.stacks[absPath]
	r.mu.RUnlock()
	if ok {
		return s
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if s, ok = r.stacks[absPath]; ok {
		return s
	}
	s = &fileStack{}
	r.stacks[absPath] = s
	return s
}

func (r *undoRegistry) push(absPath, content string) {
	s := r.get(absPath)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = append(s.entries, content)
	if len(s.entries) > maxUndoDepth {
		s.entries = s.entries[len(s.entries)-maxUndoDepth:]
	}
}

func (r *undoRegistry) pop(absPath string) (string, bool) {
	s := r.get(absPath)
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.entries) == 0 {
		return "", false
	}
	last := s.entries[len(s.entries)-1]
	s.entries = s.entries[:len(s.entries)-1]
	return last, true
}

func (r *undoRegistry) depth(absPath string) int {
	s := r.get(absPath)
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.entries)
}
