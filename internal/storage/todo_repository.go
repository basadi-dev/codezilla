package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"codezilla/internal/tools"
)

// TodoRepository defines the interface for todo persistence
type TodoRepository interface {
	Save(manager *tools.TodoManagerV2) error
	Load(manager *tools.TodoManagerV2) error
}

// FileRepository implements TodoRepository using file storage
type FileRepository struct {
	dataDir  string
	filename string
}

// NewFileRepository creates a new file-based repository
func NewFileRepository(dataDir string) TodoRepository {
	return &FileRepository{
		dataDir:  dataDir,
		filename: "todo_state.json",
	}
}

// TodoState represents the persistent state
type TodoState struct {
	Plans         map[string]*tools.TodoPlan `json:"plans"`
	CurrentPlanID string                     `json:"current_plan_id"`
}

// Save saves the current todo state to disk
func (r *FileRepository) Save(manager *tools.TodoManagerV2) error {
	// Ensure data directory exists
	if err := os.MkdirAll(r.dataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %w", err)
	}

	state := TodoState{
		Plans: make(map[string]*tools.TodoPlan),
	}

	// Copy plans from manager
	for _, plan := range manager.GetAllPlans() {
		state.Plans[plan.ID] = plan
	}

	if currentPlan, ok := manager.GetCurrentPlan(); ok {
		state.CurrentPlanID = currentPlan.ID
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal todo state: %w", err)
	}

	filePath := filepath.Join(r.dataDir, r.filename)
	if err := os.WriteFile(filePath, data, 0600); err != nil {
		return fmt.Errorf("failed to write todo state: %w", err)
	}

	return nil
}

// Load loads the todo state from disk
func (r *FileRepository) Load(manager *tools.TodoManagerV2) error {
	filePath := filepath.Join(r.dataDir, r.filename)

	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		// File doesn't exist yet, that's okay
		return nil
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read todo state: %w", err)
	}

	var state TodoState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("failed to unmarshal todo state: %w", err)
	}

	// Load plans into manager
	// This is a bit tricky since we need to modify the manager's internal state
	// For now, we'll add plans one by one
	for _, plan := range state.Plans {
		_ = manager.AddPlan(plan)
	}

	return nil
}
