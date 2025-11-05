package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// TodoRepository defines the interface for todo persistence
type TodoRepository interface {
	Save(manager *TodoManagerV2) error
	Load(manager *TodoManagerV2) error
}

// TodoManagerV2 manages todo lists and plans (no global state)
type TodoManagerV2 struct {
	plans         map[string]*TodoPlan
	currentPlanID string
	persistence   TodoRepository
}

// NewTodoManagerV2 creates a new todo manager with dependency injection
func NewTodoManagerV2(persistence TodoRepository) *TodoManagerV2 {
	mgr := &TodoManagerV2{
		plans:       make(map[string]*TodoPlan),
		persistence: persistence,
	}

	// Load existing state if available
	if persistence != nil {
		_ = persistence.Load(mgr)
	}

	return mgr
}

// AddPlan adds a plan to the manager
func (tm *TodoManagerV2) AddPlan(plan *TodoPlan) error {
	tm.plans[plan.ID] = plan
	tm.currentPlanID = plan.ID

	// Auto-save if persistence is available
	if tm.persistence != nil {
		return tm.persistence.Save(tm)
	}

	return nil
}

// GetPlan retrieves a plan by ID
func (tm *TodoManagerV2) GetPlan(id string) (*TodoPlan, bool) {
	plan, ok := tm.plans[id]
	return plan, ok
}

// GetCurrentPlan returns the current active plan
func (tm *TodoManagerV2) GetCurrentPlan() (*TodoPlan, bool) {
	if tm.currentPlanID == "" {
		return nil, false
	}
	return tm.GetPlan(tm.currentPlanID)
}

// UpdatePlan updates a plan
func (tm *TodoManagerV2) UpdatePlan(plan *TodoPlan) error {
	tm.plans[plan.ID] = plan

	// Auto-save
	if tm.persistence != nil {
		return tm.persistence.Save(tm)
	}

	return nil
}

// GetAllPlans returns all plans
func (tm *TodoManagerV2) GetAllPlans() []*TodoPlan {
	plans := make([]*TodoPlan, 0, len(tm.plans))
	for _, plan := range tm.plans {
		plans = append(plans, plan)
	}
	return plans
}

// TodoCreateToolV2 creates new todo plans with dependency injection
type TodoCreateToolV2 struct {
	manager *TodoManagerV2
}

// NewTodoCreateToolV2 creates a new todo create tool
func NewTodoCreateToolV2(manager *TodoManagerV2) Tool {
	return &TodoCreateToolV2{
		manager: manager,
	}
}

func (t *TodoCreateToolV2) Name() string {
	return "todo_create"
}

func (t *TodoCreateToolV2) Description() string {
	return "Create a new todo plan with tasks"
}

func (t *TodoCreateToolV2) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"name":        {Type: "string", Description: "Name of the todo plan"},
			"description": {Type: "string", Description: "Description of the plan"},
			"items": {
				Type:        "array",
				Description: "List of todo items",
				Items: &JSONSchema{
					Type: "object",
					Properties: map[string]JSONSchema{
						"content":      {Type: "string", Description: "Content/title of the todo item"},
						"priority":     {Type: "string", Description: "Priority: 'high', 'medium', 'low'"},
						"dependencies": {Type: "array", Items: &JSONSchema{Type: "string"}, Description: "List of task IDs this depends on"},
					},
					Required: []string{"content"},
				},
			},
		},
		Required: []string{"name", "items"},
	}
}

func (t *TodoCreateToolV2) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
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
		for i, item := range items {
			if itemMap, ok := item.(map[string]interface{}); ok {
				todoItem := TodoItem{
					ID:        fmt.Sprintf("task_%d_%d", time.Now().UnixNano(), i),
					Content:   itemMap["content"].(string),
					Status:    "pending",
					CreatedAt: time.Now(),
					UpdatedAt: time.Now(),
					Priority:  "medium",
				}

				if priority, ok := itemMap["priority"].(string); ok {
					todoItem.Priority = priority
				}

				if deps, ok := itemMap["dependencies"].([]interface{}); ok {
					for _, dep := range deps {
						if depStr, ok := dep.(string); ok {
							todoItem.Dependencies = append(todoItem.Dependencies, depStr)
						}
					}
				}

				plan.Items = append(plan.Items, todoItem)
			}
		}
	}

	if err := t.manager.AddPlan(plan); err != nil {
		return nil, fmt.Errorf("failed to save plan: %w", err)
	}

	result, _ := json.MarshalIndent(plan, "", "  ")
	return fmt.Sprintf("Created todo plan:\n%s", string(result)), nil
}

// TodoUpdateToolV2 updates todo item status with dependency injection
type TodoUpdateToolV2 struct {
	manager *TodoManagerV2
}

// NewTodoUpdateToolV2 creates a new todo update tool
func NewTodoUpdateToolV2(manager *TodoManagerV2) Tool {
	return &TodoUpdateToolV2{
		manager: manager,
	}
}

func (t *TodoUpdateToolV2) Name() string {
	return "todo_update"
}

func (t *TodoUpdateToolV2) Description() string {
	return "Update the status of todo items"
}

func (t *TodoUpdateToolV2) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"item_id": {Type: "string", Description: "ID of the todo item to update"},
			"status":  {Type: "string", Description: "New status: 'pending', 'in_progress', 'completed'"},
		},
		Required: []string{"item_id", "status"},
	}
}

func (t *TodoUpdateToolV2) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	itemID, _ := params["item_id"].(string)
	status, _ := params["status"].(string)

	plan, ok := t.manager.GetCurrentPlan()
	if !ok {
		return nil, fmt.Errorf("no active todo plan")
	}

	found := false
	for i := range plan.Items {
		if plan.Items[i].ID == itemID {
			plan.Items[i].Status = status
			plan.Items[i].UpdatedAt = time.Now()
			found = true
			break
		}
	}

	if !found {
		return nil, fmt.Errorf("todo item '%s' not found", itemID)
	}

	plan.UpdatedAt = time.Now()
	if err := t.manager.UpdatePlan(plan); err != nil {
		return nil, fmt.Errorf("failed to update plan: %w", err)
	}

	return fmt.Sprintf("Updated todo item '%s' to status '%s'", itemID, status), nil
}

// TodoListToolV2 lists current todo plans and items with dependency injection
type TodoListToolV2 struct {
	manager *TodoManagerV2
}

// NewTodoListToolV2 creates a new todo list tool
func NewTodoListToolV2(manager *TodoManagerV2) Tool {
	return &TodoListToolV2{
		manager: manager,
	}
}

func (t *TodoListToolV2) Name() string {
	return "todo_list"
}

func (t *TodoListToolV2) Description() string {
	return "List todo plans and their items"
}

func (t *TodoListToolV2) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"plan_id": {Type: "string", Description: "Optional: Specific plan ID to list"},
			"status":  {Type: "string", Description: "Optional: Filter by status"},
		},
	}
}

func (t *TodoListToolV2) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	planID, hasPlanID := params["plan_id"].(string)
	statusFilter, _ := params["status"].(string)

	var output string

	if hasPlanID {
		plan, ok := t.manager.GetPlan(planID)
		if !ok {
			return nil, fmt.Errorf("plan '%s' not found", planID)
		}
		output = formatPlan(plan, statusFilter)
	} else {
		plans := t.manager.GetAllPlans()
		if len(plans) == 0 {
			return "No todo plans created yet.", nil
		}

		output = "# Todo Plans\n\n"
		for _, plan := range plans {
			output += formatPlan(plan, statusFilter) + "\n\n"
		}
	}

	return output, nil
}

// TodoAnalyzeToolV2 analyzes the current plan and suggests next actions with dependency injection
type TodoAnalyzeToolV2 struct {
	manager *TodoManagerV2
}

// NewTodoAnalyzeToolV2 creates a new todo analyze tool
func NewTodoAnalyzeToolV2(manager *TodoManagerV2) Tool {
	return &TodoAnalyzeToolV2{
		manager: manager,
	}
}

func (t *TodoAnalyzeToolV2) Name() string {
	return "todo_analyze"
}

func (t *TodoAnalyzeToolV2) Description() string {
	return "Analyze todo plan and suggest next actions based on dependencies and priorities"
}

func (t *TodoAnalyzeToolV2) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type:       "object",
		Properties: map[string]JSONSchema{},
	}
}

func (t *TodoAnalyzeToolV2) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	plan, ok := t.manager.GetCurrentPlan()
	if !ok {
		return nil, fmt.Errorf("no active todo plan")
	}

	// Analyze plan and suggest next actions
	var completed, pending, inProgress int
	taskMap := make(map[string]*TodoItem)

	for i := range plan.Items {
		item := &plan.Items[i]
		taskMap[item.ID] = item

		switch item.Status {
		case "completed":
			completed++
		case "in_progress":
			inProgress++
		case "pending":
			pending++
		}
	}

	// Find actionable items (no blocking dependencies)
	var actionable []TodoItem
	var blocked []TodoItem

	for _, item := range plan.Items {
		if item.Status != "pending" {
			continue
		}

		isBlocked := false
		for _, depID := range item.Dependencies {
			if dep, ok := taskMap[depID]; ok {
				if dep.Status != "completed" {
					isBlocked = true
					break
				}
			}
		}

		if isBlocked {
			blocked = append(blocked, item)
		} else {
			actionable = append(actionable, item)
		}
	}

	output := fmt.Sprintf("# Todo Plan Analysis: %s\n\n", plan.Name)
	output += fmt.Sprintf("**Progress**: %d completed, %d in progress, %d pending\n\n", completed, inProgress, pending)

	if len(actionable) > 0 {
		output += "## Actionable Tasks (No Blockers)\n\n"
		for _, item := range actionable {
			output += fmt.Sprintf("- [%s] %s (Priority: %s)\n", item.ID, item.Content, item.Priority)
		}
		output += "\n"
	}

	if len(blocked) > 0 {
		output += "## Blocked Tasks\n\n"
		for _, item := range blocked {
			output += fmt.Sprintf("- [%s] %s (Waiting on: %v)\n", item.ID, item.Content, item.Dependencies)
		}
	}

	return output, nil
}

// GetTodoToolsV2 returns all todo management tools with dependency injection
func GetTodoToolsV2(manager *TodoManagerV2) []Tool {
	return []Tool{
		NewTodoCreateToolV2(manager),
		NewTodoUpdateToolV2(manager),
		NewTodoListToolV2(manager),
		NewTodoAnalyzeToolV2(manager),
	}
}
