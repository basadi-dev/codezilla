package tools

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// PermissionLevel defines how permissions are handled for tools
type PermissionLevel int

const (
	// AlwaysAsk means permission will be requested for every tool execution
	AlwaysAsk PermissionLevel = iota
	// AskOnce means permission will be requested only the first time a tool is used
	AskOnce
	// NeverAsk means permission is automatically granted
	NeverAsk
)

var (
	// ErrPermissionDenied is returned when tool execution permission is denied
	ErrPermissionDenied = errors.New("permission denied for tool execution")
)

// ToolContext represents a specific tool execution with its parameters
type ToolContext struct {
	ToolName string
	Params   map[string]interface{}
	Time     time.Time
}

// PermissionRequest contains information about a permission request
type PermissionRequest struct {
	ToolContext ToolContext
	Description string
	Tool        Tool
}

// PermissionResponse represents the user's response to a permission request
type PermissionResponse struct {
	Granted    bool
	RememberMe bool // Whether to remember this choice for future uses
}

// PermissionCallback is a function that handles permission requests
// It should present the request to the user and return their response
type PermissionCallback func(ctx context.Context, request PermissionRequest) (PermissionResponse, error)

// ToolPermissionManager manages permissions for tool executions
type ToolPermissionManager interface {
	// RequestPermission requests permission to execute a tool
	// If permission is granted, it returns true. Otherwise, it returns false
	// If an error occurs during the request, it returns an error
	RequestPermission(ctx context.Context, toolName string, params map[string]interface{}, tool Tool) (bool, error)

	// GetDefaultPermissionLevel returns the default permission level for a tool
	GetDefaultPermissionLevel(toolName string) PermissionLevel

	// SetDefaultPermissionLevel sets the default permission level for a tool
	SetDefaultPermissionLevel(toolName string, level PermissionLevel)

	// GetPermissionCallback returns the callback function used to request permissions
	GetPermissionCallback() PermissionCallback

	// SetPermissionCallback sets the callback function used to request permissions
	SetPermissionCallback(callback PermissionCallback)
}

// ToolPermission stores permission settings for a specific tool
type ToolPermission struct {
	Level           PermissionLevel
	ApprovedActions map[string]bool // Maps serialized parameters to approval status
}

// defaultPermissionManager is the default implementation of ToolPermissionManager
type defaultPermissionManager struct {
	permissions      map[string]ToolPermission
	permissionsMutex sync.RWMutex
	callback         PermissionCallback
}

// NewPermissionManager creates a new tool permission manager
func NewPermissionManager(callback PermissionCallback) ToolPermissionManager {
	if callback == nil {
		// Use a default callback that always denies permissions
		callback = func(ctx context.Context, request PermissionRequest) (PermissionResponse, error) {
			return PermissionResponse{Granted: false}, nil
		}
	}

	return &defaultPermissionManager{
		permissions: make(map[string]ToolPermission),
		callback:    callback,
	}
}

// RequestPermission requests permission to execute a tool
func (m *defaultPermissionManager) RequestPermission(ctx context.Context, toolName string, params map[string]interface{}, tool Tool) (bool, error) {
	m.permissionsMutex.RLock()
	perm, exists := m.permissions[toolName]
	m.permissionsMutex.RUnlock()

	if !exists {
		// Create default permission for this tool
		perm = ToolPermission{
			Level:           getInitialPermissionLevel(toolName),
			ApprovedActions: make(map[string]bool),
		}
	}

	// If we never ask for permission, immediately return granted
	if perm.Level == NeverAsk {
		return true, nil
	}

	// For AskOnce level, check if we've seen this action before
	if perm.Level == AskOnce {
		actionKey := serializeParams(params)
		if approved, found := perm.ApprovedActions[actionKey]; found {
			return approved, nil
		}
	}

	// We need to ask for permission
	// Copy params to avoid any potential race conditions
	paramsCopy := make(map[string]interface{})
	for k, v := range params {
		paramsCopy[k] = v
	}

	request := PermissionRequest{
		ToolContext: ToolContext{
			ToolName: toolName,
			Params:   paramsCopy, // Use our copy
			Time:     time.Now(),
		},
		Description: generateDescription(tool, paramsCopy),
		Tool:        tool,
	}

	response, err := m.callback(ctx, request)
	if err != nil {
		return false, err
	}

	// Remember this decision if requested
	if response.RememberMe {
		m.permissionsMutex.Lock()
		defer m.permissionsMutex.Unlock()

		// Get the latest permissions (they might have changed since we checked)
		perm, exists = m.permissions[toolName]
		if !exists {
			perm = ToolPermission{
				Level:           getInitialPermissionLevel(toolName),
				ApprovedActions: make(map[string]bool),
			}
		}

		actionKey := serializeParams(params)
		perm.ApprovedActions[actionKey] = response.Granted
		m.permissions[toolName] = perm
	}

	return response.Granted, nil
}

// GetDefaultPermissionLevel returns the default permission level for a tool
func (m *defaultPermissionManager) GetDefaultPermissionLevel(toolName string) PermissionLevel {
	m.permissionsMutex.RLock()
	defer m.permissionsMutex.RUnlock()

	perm, exists := m.permissions[toolName]
	if !exists {
		return getInitialPermissionLevel(toolName)
	}
	return perm.Level
}

// SetDefaultPermissionLevel sets the default permission level for a tool
func (m *defaultPermissionManager) SetDefaultPermissionLevel(toolName string, level PermissionLevel) {
	m.permissionsMutex.Lock()
	defer m.permissionsMutex.Unlock()

	perm, exists := m.permissions[toolName]
	if !exists {
		perm = ToolPermission{
			Level:           level,
			ApprovedActions: make(map[string]bool),
		}
	} else {
		perm.Level = level
	}
	m.permissions[toolName] = perm
}

// GetPermissionCallback returns the callback function used to request permissions
func (m *defaultPermissionManager) GetPermissionCallback() PermissionCallback {
	return m.callback
}

// SetPermissionCallback sets the callback function used to request permissions
func (m *defaultPermissionManager) SetPermissionCallback(callback PermissionCallback) {
	if callback != nil {
		m.callback = callback
	}
}

// Helper function to serialize parameters for storage
func serializeParams(params map[string]interface{}) string {
	// Simple string concatenation for demo purposes
	// In a real implementation, this would be more robust
	result := ""
	for k, v := range params {
		result += fmt.Sprintf("%s=%v;", k, v)
	}
	return result
}

// Helper function to generate a human-readable description of the tool execution
func generateDescription(tool Tool, params map[string]interface{}) string {
	switch tool.Name() {
	case "execute":
		if cmd, ok := params["command"].(string); ok {
			return fmt.Sprintf("Execute shell command: %s", cmd)
		}
		return "Execute shell command"
	case "fileRead":
		if path, ok := params["file_path"].(string); ok {
			return fmt.Sprintf("Read file: %s", path)
		}
		return "Read file"
	case "fileWrite":
		if path, ok := params["file_path"].(string); ok {
			append := false
			if appendVal, ok := params["append"].(bool); ok {
				append = appendVal
			}
			if append {
				return fmt.Sprintf("Append to file: %s", path)
			}
			return fmt.Sprintf("Write to file: %s", path)
		}
		return "Write to file"
	default:
		return fmt.Sprintf("Execute tool: %s", tool.Name())
	}
}

// Helper function to determine the initial permission level for a tool
func getInitialPermissionLevel(toolName string) PermissionLevel {
	// Set sensible defaults
	switch toolName {
	case "execute":
		// Commands are potentially dangerous, but default to never ask per user request
		return NeverAsk
	case "fileWrite":
		// Writing files is potentially dangerous, but default to never ask per user request
		return NeverAsk
	case "fileRead":
		// Reading files is safe, never ask
		return NeverAsk
	case "listFiles":
		// Listing files is safe, never ask
		return NeverAsk
	default:
		// For unknown tools, default to never ask per user request
		return NeverAsk
	}
}
