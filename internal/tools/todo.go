package tools

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// TodoItem represents a single todo task
type TodoItem struct {
	ID           string     `json:"id"`
	Content      string     `json:"content"`
	Status       string     `json:"status"`   // pending, in_progress, completed, cancelled
	Priority     string     `json:"priority"` // high, medium, low
	CreatedAt    time.Time  `json:"created_at"`
	UpdatedAt    time.Time  `json:"updated_at"`
	CompletedAt  *time.Time `json:"completed_at,omitempty"`
	Dependencies []string   `json:"dependencies,omitempty"` // IDs of tasks that must be completed first
}

// TodoPlan represents a collection of todo items with planning metadata
type TodoPlan struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Items       []TodoItem `json:"items"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
}

// TodoManager manages todo lists and plans
type TodoManager struct {
	mu            sync.RWMutex
	plans         map[string]*TodoPlan
	currentPlanID string
}

// NewTodoManager creates a new todo manager
func NewTodoManager() *TodoManager {
	return &TodoManager{
		plans: make(map[string]*TodoPlan),
	}
}

// CurrentPlan returns a snapshot of the current plan (name + items) for display.
// Returns nil if no plan is set.
func (m *TodoManager) CurrentPlan() *TodoPlan {
	m.mu.RLock()
	defer m.mu.RUnlock()
	plan, ok := m.plans[m.currentPlanID]
	if !ok {
		return nil
	}
	// Return a shallow copy so the caller doesn't need to hold the lock
	items := make([]TodoItem, len(plan.Items))
	copy(items, plan.Items)
	return &TodoPlan{
		ID:          plan.ID,
		Name:        plan.Name,
		Description: plan.Description,
		Items:       items,
	}
}

// TodoCreateTool creates new todo plans
type TodoCreateTool struct{ mgr *TodoManager }

func (t TodoCreateTool) Name() string {
	return "todoCreate"
}

func (t TodoCreateTool) Description() string {
	return "Create a new todo plan with tasks"
}

func (t TodoCreateTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"name":        {Type: "string", Description: "Name of the todo plan"},
			"description": {Type: "string", Description: "Description of what this plan aims to achieve"},
			"items": {
				Type:        "array",
				Description: "List of todo items",
				Items: &JSONSchema{
					Type: "object",
					Properties: map[string]JSONSchema{
						"content":  {Type: "string", Description: "Task description"},
						"priority": {Type: "string", Enum: []interface{}{"high", "medium", "low"}, Default: "medium"},
						"dependencies": {
							Type:        "array",
							Items:       &JSONSchema{Type: "string"},
							Description: "IDs of tasks that must be completed first",
						},
					},
					Required: []string{"content"},
				},
			},
		},
		Required: []string{"name", "items"},
	}
}

func (t TodoCreateTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	name, _ := params["name"].(string)
	description, _ := params["description"].(string)

	plan := &TodoPlan{
		ID:          fmt.Sprintf("plan_%d", time.Now().UnixNano()),
		Name:        name,
		Description: description,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Items:       []TodoItem{},
	}

	if items, ok := params["items"].([]interface{}); ok {
		for _, item := range items {
			contentStr := "Untitled Task"
			priorityStr := "medium"
			var deps []string

			if strItem, ok := item.(string); ok {
				contentStr = strItem
			} else if itemMap, ok := item.(map[string]interface{}); ok {
				if c, ok := itemMap["content"].(string); ok && c != "" {
					contentStr = c
				}
				if contentStr == "Untitled Task" {
					if c, ok := itemMap["name"].(string); ok && c != "" {
						contentStr = c
					}
				}
				if contentStr == "Untitled Task" {
					if c, ok := itemMap["title"].(string); ok && c != "" {
						contentStr = c
					}
				}
				if contentStr == "Untitled Task" {
					if c, ok := itemMap["item"].(string); ok && c != "" {
						contentStr = c
					}
				}
				if contentStr == "Untitled Task" {
					if c, ok := itemMap["task"].(string); ok && c != "" {
						contentStr = c
					}
				}
				if contentStr == "Untitled Task" {
					if c, ok := itemMap["description"].(string); ok && c != "" {
						contentStr = c
					}
				}

				if p, ok := itemMap["priority"].(string); ok {
					priorityStr = p
				}

				if dInfo, ok := itemMap["dependencies"].([]interface{}); ok {
					for _, dep := range dInfo {
						if depStr, ok := dep.(string); ok {
							deps = append(deps, depStr)
						}
					}
				}
			}

			todoItem := TodoItem{
				ID:           fmt.Sprintf("t%d", len(plan.Items)+1),
				Content:      contentStr,
				Status:       "pending",
				Priority:     priorityStr,
				Dependencies: deps,
				CreatedAt:    time.Time{}, // Time assignment follows
			}
			todoItem.CreatedAt = time.Now()
			todoItem.UpdatedAt = time.Now()

			plan.Items = append(plan.Items, todoItem)
		}
	}

	globalTodoManager := t.mgr
	globalTodoManager.mu.Lock()
	globalTodoManager.plans[plan.ID] = plan
	if setCur, ok := params["set_current"].(bool); ok && setCur {
		globalTodoManager.currentPlanID = plan.ID
	}

	globalTodoManager.mu.Unlock()

	result := fmt.Sprintf("Created todo plan: %s (ID: %s)\n", plan.Name, plan.ID)
	for _, item := range plan.Items {
		result += fmt.Sprintf("- [%s] %s\n", item.ID, item.Content)
	}
	return result, nil
}

// TodoManageTool combines update, list, analyze, and set_current operations.
type TodoManageTool struct{ mgr *TodoManager }

func (t TodoManageTool) Name() string { return "todoManage" }
func (t TodoManageTool) Description() string {
	return "Manage todo plans: update status, list items, analyze, or set active plan"
}

func (t TodoManageTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"action":        {Type: "string", Description: "Action to perform", Enum: []interface{}{"update", "list", "analyze", "set_current"}},
			"plan_id":       {Type: "string", Description: "Plan ID (optional, uses current plan)"},
			"task_id":       {Type: "string", Description: "Task ID to update (for action=update)"},
			"status":        {Type: "string", Description: "New status (for action=update)", Enum: []interface{}{"pending", "in_progress", "completed", "cancelled"}},
			"task_name":     {Type: "string", Description: "Task name for display (for action=update)"},
			"content":       {Type: "string", Description: "Updated task content (optional, for action=update)"},
			"status_filter": {Type: "string", Description: "Filter by status (for action=list)", Enum: []interface{}{"all", "pending", "in_progress", "completed", "cancelled"}},
		},
		Required: []string{"action"},
	}
}

func (t TodoManageTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	action, _ := params["action"].(string)
	switch action {
	case "update":
		return t.execUpdate(params)
	case "list":
		return t.execList(params)
	case "analyze":
		return t.execAnalyze(params)
	case "set_current":
		return t.execSetCurrent(params)
	default:
		return nil, fmt.Errorf("unknown action: %s (valid: update, list, analyze, set_current)", action)
	}
}

func (t TodoManageTool) execUpdate(params map[string]interface{}) (interface{}, error) {
	taskID, _ := params["task_id"].(string)
	status, _ := params["status"].(string)

	t.mgr.mu.Lock()
	defer t.mgr.mu.Unlock()

	planID := t.mgr.currentPlanID
	if pid, ok := params["plan_id"].(string); ok {
		planID = pid
	}

	plan, exists := t.mgr.plans[planID]
	if !exists {
		return "", fmt.Errorf("plan not found: %s", planID)
	}

	for i := range plan.Items {
		if plan.Items[i].ID == taskID {
			plan.Items[i].Status = status
			plan.Items[i].UpdatedAt = time.Now()

			if status == "completed" {
				now := time.Now()
				plan.Items[i].CompletedAt = &now
			}

			if content, ok := params["content"].(string); ok {
				plan.Items[i].Content = content
			}

			plan.UpdatedAt = time.Now()

			statusIcon := "○"
			switch status {
			case "in_progress":
				statusIcon = "◐"
			case "completed":
				statusIcon = "●"
			case "cancelled":
				statusIcon = "⊘"
			}
			return fmt.Sprintf("%s %s: %s", statusIcon, status, plan.Items[i].Content), nil
		}
	}

	return "", fmt.Errorf("task not found: %s", taskID)
}

func (t TodoManageTool) execList(params map[string]interface{}) (interface{}, error) {
	t.mgr.mu.RLock()
	defer t.mgr.mu.RUnlock()

	statusFilter := "all"
	if filter, ok := params["status_filter"].(string); ok {
		statusFilter = filter
	}

	var output string

	if planID, ok := params["plan_id"].(string); ok {
		plan, exists := t.mgr.plans[planID]
		if !exists {
			return "", fmt.Errorf("plan not found: %s", planID)
		}
		output = formatPlan(plan, statusFilter)
	} else {
		if len(t.mgr.plans) == 0 {
			return "No todo plans created yet.", nil
		}

		output = "# Todo Plans\n\n"
		for _, plan := range t.mgr.plans {
			output += formatPlan(plan, statusFilter) + "\n---\n\n"
		}
	}

	return output, nil
}

func formatPlan(plan *TodoPlan, statusFilter string) string {
	output := fmt.Sprintf("## %s\n", plan.Name)
	if plan.Description != "" {
		output += fmt.Sprintf("*%s*\n\n", plan.Description)
	}
	output += fmt.Sprintf("ID: %s | Created: %s\n\n", plan.ID, plan.CreatedAt.Format("2006-01-02 15:04"))

	statusIcons := map[string]string{
		"pending":     "○",
		"in_progress": "◐",
		"completed":   "●",
		"cancelled":   "⊘",
	}

	completed := 0
	for _, item := range plan.Items {
		if item.Status == "completed" {
			completed++
		}

		if statusFilter != "all" && item.Status != statusFilter {
			continue
		}

		icon := statusIcons[item.Status]

		priorityIcon := ""
		switch item.Priority {
		case "high":
			priorityIcon = "🔴"
		case "medium":
			priorityIcon = "🟡"
		case "low":
			priorityIcon = "🟢"
		}

		output += fmt.Sprintf("%s %s %s [ID: %s]\n", icon, priorityIcon, item.Content, item.ID)

		if len(item.Dependencies) > 0 {
			output += fmt.Sprintf("  Dependencies: %v\n", item.Dependencies)
		}
	}
	output += "\n"

	// Show progress
	total := len(plan.Items)
	if total > 0 {
		progress := float64(completed) / float64(total) * 100
		output += fmt.Sprintf("**Progress: %d/%d (%.0f%%)**\n", completed, total, progress)
	}

	return output
}

func (t TodoManageTool) execAnalyze(params map[string]interface{}) (interface{}, error) {
	t.mgr.mu.RLock()
	defer t.mgr.mu.RUnlock()

	planID := t.mgr.currentPlanID
	if pid, ok := params["plan_id"].(string); ok {
		planID = pid
	}

	plan, exists := t.mgr.plans[planID]
	if !exists {
		return "", fmt.Errorf("plan not found: %s", planID)
	}

	// Build dependency map
	depMap := make(map[string][]string)
	taskMap := make(map[string]*TodoItem)
	for i := range plan.Items {
		item := &plan.Items[i]
		taskMap[item.ID] = item
		for _, dep := range item.Dependencies {
			depMap[dep] = append(depMap[dep], item.ID)
		}
	}

	// Find actionable tasks (no incomplete dependencies)
	var actionable []TodoItem
	var blocked []TodoItem
	var inProgress []TodoItem

	for _, item := range plan.Items {
		if item.Status == "completed" || item.Status == "cancelled" {
			continue
		}

		if item.Status == "in_progress" {
			inProgress = append(inProgress, item)
			continue
		}

		// Check if all dependencies are complete
		canStart := true
		for _, depID := range item.Dependencies {
			if dep, exists := taskMap[depID]; exists {
				if dep.Status != "completed" {
					canStart = false
					break
				}
			}
		}

		if canStart {
			actionable = append(actionable, item)
		} else {
			blocked = append(blocked, item)
		}
	}

	// Sort actionable by priority
	priorityOrder := map[string]int{"high": 0, "medium": 1, "low": 2}
	for i := 0; i < len(actionable)-1; i++ {
		for j := i + 1; j < len(actionable); j++ {
			if priorityOrder[actionable[i].Priority] > priorityOrder[actionable[j].Priority] {
				actionable[i], actionable[j] = actionable[j], actionable[i]
			}
		}
	}

	// Generate analysis
	output := fmt.Sprintf("# Todo Plan Analysis: %s\n\n", plan.Name)

	if len(inProgress) > 0 {
		output += "## 🔄 Currently In Progress\n"
		for _, item := range inProgress {
			output += fmt.Sprintf("- %s (ID: %s)\n", item.Content, item.ID)
		}
		output += "\n"
	}

	if len(actionable) > 0 {
		output += "## ✅ Ready to Start\nThese tasks have no blocking dependencies:\n\n"
		for _, item := range actionable {
			priorityIcon := map[string]string{"high": "🔴", "medium": "🟡", "low": "🟢"}[item.Priority]
			output += fmt.Sprintf("- %s %s (ID: %s)\n", priorityIcon, item.Content, item.ID)

			if deps := depMap[item.ID]; len(deps) > 0 {
				output += "  Completing this will unlock:\n"
				for _, depID := range deps {
					if dep := taskMap[depID]; dep != nil {
						output += fmt.Sprintf("    - %s\n", dep.Content)
					}
				}
			}
		}
		output += "\n"
	}

	if len(blocked) > 0 {
		output += "## 🚫 Blocked Tasks\n"
		for _, item := range blocked {
			output += fmt.Sprintf("- %s (ID: %s)\n", item.Content, item.ID)
			output += "  Waiting for:\n"
			for _, depID := range item.Dependencies {
				if dep := taskMap[depID]; dep != nil && dep.Status != "completed" {
					output += fmt.Sprintf("    - %s (Status: %s)\n", dep.Content, dep.Status)
				}
			}
		}
		output += "\n"
	}

	// Recommendations
	output += "## 📋 Recommendations\n\n"
	if len(inProgress) > 0 {
		output += "1. Focus on completing the in-progress tasks first\n"
	}
	if len(actionable) > 0 {
		if actionable[0].Priority == "high" {
			output += fmt.Sprintf("2. Start with high-priority task: %s (ID: %s)\n",
				actionable[0].Content, actionable[0].ID)
		} else {
			output += fmt.Sprintf("2. Next recommended task: %s (ID: %s)\n",
				actionable[0].Content, actionable[0].ID)
		}
	}
	if len(blocked) > 0 {
		output += fmt.Sprintf("3. %d tasks are blocked by dependencies\n", len(blocked))
	}

	return output, nil
}

func (t TodoManageTool) execSetCurrent(params map[string]interface{}) (interface{}, error) {
	planID, _ := params["plan_id"].(string)
	t.mgr.mu.Lock()
	defer t.mgr.mu.Unlock()
	if _, ok := t.mgr.plans[planID]; !ok {
		return "", fmt.Errorf("plan not found: %s", planID)
	}
	t.mgr.currentPlanID = planID
	return fmt.Sprintf("Current plan set to %s", planID), nil
}

// GetTodoTools returns all todo management tools using the provided manager.
func GetTodoTools(mgr *TodoManager) []Tool {
	return []Tool{
		TodoCreateTool{mgr: mgr},
		TodoManageTool{mgr: mgr},
	}
}
