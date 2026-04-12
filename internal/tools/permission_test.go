package tools

import (
	"context"
	"testing"
)

func TestPermissionManager(t *testing.T) {
	tests := []struct {
		name           string
		toolName       string
		defaultPerm    PermissionLevel
		userResponse   PermissionResponse
		expectCallback bool
		expectGranted  bool
	}{
		{
			name:           "Never ask permission",
			toolName:       "fileManage",
			defaultPerm:    NeverAsk,
			userResponse:   PermissionResponse{},
			expectCallback: false,
			expectGranted:  true,
		},
		{
			name:           "Always ask permission - granted",
			toolName:       "fileManage",
			defaultPerm:    AlwaysAsk,
			userResponse:   PermissionResponse{Granted: true, RememberMe: false},
			expectCallback: true,
			expectGranted:  true,
		},
		{
			name:           "Always ask permission - denied",
			toolName:       "execute",
			defaultPerm:    AlwaysAsk,
			userResponse:   PermissionResponse{Granted: false, RememberMe: false},
			expectCallback: true,
			expectGranted:  false,
		},
		{
			name:           "Remember permission",
			toolName:       "fileManage",
			defaultPerm:    AskOnce,
			userResponse:   PermissionResponse{Granted: true, RememberMe: true},
			expectCallback: true,
			expectGranted:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			callbackCalled := false
			pm := NewPermissionManager(func(ctx context.Context, req PermissionRequest) (PermissionResponse, error) {
				callbackCalled = true

				// Verify request structure
				if req.ToolContext.ToolName != tt.toolName {
					t.Errorf("Expected tool name %s, got %s", tt.toolName, req.ToolContext.ToolName)
				}

				return tt.userResponse, nil
			})

			// Set default permission
			pm.SetDefaultPermissionLevel(tt.toolName, tt.defaultPerm)

			// Create context
			ctx := context.Background()

			// Create a mock tool
			mockTool := &FileManageTool{}

			// Request permission
			granted, err := pm.RequestPermission(ctx, tt.toolName, map[string]interface{}{}, mockTool)
			if err != nil {
				t.Fatalf("RequestPermission failed: %v", err)
			}

			// Verify callback was called as expected
			if callbackCalled != tt.expectCallback {
				t.Errorf("Callback called = %v, expected %v", callbackCalled, tt.expectCallback)
			}

			// Verify response
			if granted != tt.expectGranted {
				t.Errorf("Expected granted = %v, got %v", tt.expectGranted, granted)
			}

			// Test remember functionality
			if tt.userResponse.RememberMe && granted {
				// Request permission again - should not call callback
				callbackCalled = false
				granted2, err := pm.RequestPermission(ctx, tt.toolName, map[string]interface{}{}, mockTool)
				if err != nil {
					t.Fatalf("Second RequestPermission failed: %v", err)
				}

				if callbackCalled {
					t.Error("Callback should not be called when permission is remembered")
				}

				if granted2 != granted {
					t.Errorf("Remembered permission = %v, expected %v", granted2, granted)
				}
			}
		})
	}
}

func TestPermissionRequestFormatting(t *testing.T) {
	// Test that permission requests are properly formatted
	callbackCalled := false
	var capturedRequest PermissionRequest

	pm := NewPermissionManager(func(ctx context.Context, req PermissionRequest) (PermissionResponse, error) {
		callbackCalled = true
		capturedRequest = req
		return PermissionResponse{Granted: true}, nil
	})

	pm.SetDefaultPermissionLevel("test", AlwaysAsk)

	ctx := context.Background()
	params := map[string]interface{}{
		"file_path": "/etc/passwd",
		"content":   "sensitive data",
	}

	mockTool := &FileManageTool{}

	_, err := pm.RequestPermission(ctx, "test", params, mockTool)
	if err != nil {
		t.Fatalf("RequestPermission failed: %v", err)
	}

	if !callbackCalled {
		t.Fatal("Callback was not called")
	}

	// Verify request was properly populated
	if capturedRequest.ToolContext.ToolName != "test" {
		t.Errorf("Expected tool name 'test', got %s", capturedRequest.ToolContext.ToolName)
	}

	if len(capturedRequest.ToolContext.Params) != len(params) {
		t.Errorf("Expected %d params, got %d", len(params), len(capturedRequest.ToolContext.Params))
	}

	if capturedRequest.Tool == nil {
		t.Error("Tool should not be nil in request")
	}
}
