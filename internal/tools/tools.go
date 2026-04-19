package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

// JSONSchema represents a simplified JSON Schema structure
type JSONSchema struct {
	Type                 string                `json:"type"`
	Description          string                `json:"description,omitempty"`
	Properties           map[string]JSONSchema `json:"properties,omitempty"`
	Required             []string              `json:"required,omitempty"`
	Items                *JSONSchema           `json:"items,omitempty"`
	Enum                 []interface{}         `json:"enum,omitempty"`
	Format               string                `json:"format,omitempty"`
	Minimum              *float64              `json:"minimum,omitempty"`
	Maximum              *float64              `json:"maximum,omitempty"`
	Default              interface{}           `json:"default,omitempty"`
	AdditionalProperties interface{}           `json:"additionalProperties,omitempty"`
}

// ToolSpec defines the specification of a tool for the LLM
type ToolSpec struct {
	Name            string     `json:"name"`
	Description     string     `json:"description"`
	ParameterSchema JSONSchema `json:"parameters"`
}

// Tool defines the interface that all tools must implement
type Tool interface {
	// Name returns the unique name of the tool
	Name() string

	// Description returns a user-friendly description of what the tool does
	Description() string

	// ParameterSchema returns a JSON schema describing the parameters this tool accepts
	ParameterSchema() JSONSchema

	// Execute runs the tool with the given parameters and returns a result
	Execute(ctx context.Context, params map[string]interface{}) (interface{}, error)
}

// ErrInvalidToolParams is returned when invalid parameters are provided to a tool
type ErrInvalidToolParams struct {
	ToolName string
	Message  string
}

func (e ErrInvalidToolParams) Error() string {
	return fmt.Sprintf("invalid parameters for tool '%s': %s", e.ToolName, e.Message)
}

// ErrToolExecution is returned when there's an error executing a tool
type ErrToolExecution struct {
	ToolName string
	Message  string
	Err      error
}

func (e ErrToolExecution) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("error executing tool '%s': %s: %v", e.ToolName, e.Message, e.Err)
	}
	return fmt.Sprintf("error executing tool '%s': %s", e.ToolName, e.Message)
}

// ToolRegistry manages the registration and retrieval of tools
type ToolRegistry interface {
	// RegisterTool adds a tool to the registry
	RegisterTool(tool Tool)

	// GetTool retrieves a tool by name
	GetTool(name string) (Tool, bool)

	// ListTools returns a list of all registered tools
	ListTools() []Tool

	// GetToolSpecs returns specifications for all registered tools
	GetToolSpecs() []ToolSpec

	// FilterTools removes any tool whose name causes the predicate to return false
	FilterTools(predicate func(string) bool)
}

// toolRegistry is the default implementation of ToolRegistry
type toolRegistry struct {
	tools map[string]Tool
	mu    sync.RWMutex
}

// NewToolRegistry creates a new tool registry
func NewToolRegistry() ToolRegistry {
	return &toolRegistry{
		tools: make(map[string]Tool),
	}
}

// RegisterTool adds a tool to the registry
func (r *toolRegistry) RegisterTool(tool Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[tool.Name()] = tool
}

// GetTool retrieves a tool by name
func (r *toolRegistry) GetTool(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tool, ok := r.tools[name]
	return tool, ok
}

// ListTools returns a list of all registered tools
func (r *toolRegistry) ListTools() []Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tools := make([]Tool, 0, len(r.tools))
	for _, tool := range r.tools {
		tools = append(tools, tool)
	}
	return tools
}

// GetToolSpecs returns specifications for all registered tools
func (r *toolRegistry) GetToolSpecs() []ToolSpec {
	r.mu.RLock()
	defer r.mu.RUnlock()

	specs := make([]ToolSpec, 0, len(r.tools))
	for _, tool := range r.tools {
		specs = append(specs, ToolSpec{
			Name:            tool.Name(),
			Description:     tool.Description(),
			ParameterSchema: tool.ParameterSchema(),
		})
	}
	return specs
}

// FilterTools removes any tool whose name causes the predicate to return false
func (r *toolRegistry) FilterTools(predicate func(string) bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name := range r.tools {
		if !predicate(name) {
			delete(r.tools, name)
		}
	}
}

// ValidateToolParams validates the parameters against the schema
func ValidateToolParams(tool Tool, params map[string]interface{}) error {
	schema := tool.ParameterSchema()

	// Basic validation: Check required parameters
	if schema.Properties != nil && schema.Required != nil {
		for _, requiredParam := range schema.Required {
			if _, exists := params[requiredParam]; !exists {
				return &ErrInvalidToolParams{
					ToolName: tool.Name(),
					Message:  fmt.Sprintf("missing required parameter: %s", requiredParam),
				}
			}
		}
	}

	// Additional validation could be implemented here
	// For a more comprehensive validation, consider using a full JSON Schema validator

	return nil
}

// FormatToolSpecsForLLM formats tool specifications in a way suitable for inclusion in LLM prompts
func FormatToolSpecsForLLM(specs []ToolSpec) (string, error) {
	formattedSpecs := make([]map[string]interface{}, len(specs))

	for i, spec := range specs {
		formattedSpecs[i] = map[string]interface{}{
			"name":        spec.Name,
			"description": spec.Description,
			"parameters":  spec.ParameterSchema,
		}
	}

	bytes, err := json.MarshalIndent(formattedSpecs, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal tool specs: %w", err)
	}

	return string(bytes), nil
}
