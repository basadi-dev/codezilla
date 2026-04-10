package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// TodoPersistence handles saving and loading todo state
type TodoPersistence struct {
	dataDir  string
	fileName string
}

// NewTodoPersistence creates a new persistence manager
func NewTodoPersistence(dataDir string) *TodoPersistence {
	return &TodoPersistence{
		dataDir:  dataDir,
		fileName: "todo_state.json",
	}
}

// TodoState represents the persistent state
type TodoState struct {
	Plans         map[string]*TodoPlan `json:"plans"`
	CurrentPlanID string               `json:"current_plan_id"`
}

// Save saves the current todo state to disk
func (tp *TodoPersistence) Save(manager *TodoManager) error {
	if manager == nil {
		return fmt.Errorf("manager is nil")
	}

	// Ensure directory exists
	if err := os.MkdirAll(tp.dataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %w", err)
	}

	manager.mu.RLock()
	state := TodoState{
		Plans:         manager.plans,
		CurrentPlanID: manager.currentPlanID,
	}
	manager.mu.RUnlock()

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal todo state: %w", err)
	}

	filePath := filepath.Join(tp.dataDir, tp.fileName)
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write todo state: %w", err)
	}

	return nil
}

// Load loads the todo state from disk
func (tp *TodoPersistence) Load(manager *TodoManager) error {
	if manager == nil {
		return fmt.Errorf("manager is nil")
	}

	filePath := filepath.Join(tp.dataDir, tp.fileName)
	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			// No saved state, that's okay
			return nil
		}
		return fmt.Errorf("failed to read todo state: %w", err)
	}

	var state TodoState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("failed to unmarshal todo state: %w", err)
	}

	manager.mu.Lock()
	manager.plans = state.Plans
	manager.currentPlanID = state.CurrentPlanID
	manager.mu.Unlock()

	return nil
}

// AutoSave returns a function that automatically saves after operations
func (tp *TodoPersistence) AutoSave(manager *TodoManager) func() {
	return func() {
		if err := tp.Save(manager); err != nil {
			// Log error but don't fail the operation
			fmt.Fprintf(os.Stderr, "Warning: failed to save todo state: %v\n", err)
		}
	}
}

// DefaultTodoPersistencePath returns the default path for todo state persistence.
func DefaultTodoPersistencePath() string {
	dataDir := ".codezilla"
	if home, err := os.UserHomeDir(); err == nil {
		dataDir = filepath.Join(home, ".codezilla", "todos")
	}
	return dataDir
}
