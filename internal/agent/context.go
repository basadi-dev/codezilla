package agent

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"codezilla/pkg/logger"
	anyllm "github.com/mozilla-ai/any-llm-go"
)

// Role defines the role of a message in a conversation
type Role string

const (
	// Message roles
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message represents a message in a conversation
type Message struct {
	Role            Role              `json:"role"`
	Content         string            `json:"content"`
	ToolCall        *ToolCall         `json:"tool_call,omitempty"`
	ToolResult      *ToolResult       `json:"tool_result,omitempty"`
	NativeToolCalls []anyllm.ToolCall `json:"native_tool_calls,omitempty"`
	Timestamp       time.Time         `json:"timestamp"`
}

// ToolCall represents a call to a tool
type ToolCall struct {
	ToolName string                 `json:"tool_name"`
	Params   map[string]interface{} `json:"params"`
}

// ToolResult represents the result of a tool call
type ToolResult struct {
	ToolCallID string      `json:"tool_call_id,omitempty"`
	Result     interface{} `json:"result"`
	Error      string      `json:"error,omitempty"`
}

// Context manages the conversation context for an agent
type Context struct {
	mu             sync.RWMutex
	Messages       []Message
	MaxTokens      int
	CurrentTokens  int
	TruncateOldest bool
	logger         *logger.Logger
}

// NewContext creates a new conversation context
func NewContext(maxTokens int, log *logger.Logger) *Context {
	if maxTokens <= 0 {
		maxTokens = 4000 // Default token limit
	}
	if log == nil {
		log = logger.DefaultLogger()
	}

	return &Context{
		Messages:       []Message{},
		MaxTokens:      maxTokens,
		CurrentTokens:  0,
		TruncateOldest: true,
		logger:         log,
	}
}

// ClearContext clears all non-system messages from the context
func (c *Context) ClearContext() {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Keep only system messages
	var systemMessages []Message
	var systemTokenCount int

	// Collect system messages and their token counts
	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			systemMessages = append(systemMessages, msg)
			systemTokenCount += estimateTokens(msg.Content)
			// Add any additional tokens for tool calls/results if present
			if msg.ToolCall != nil {
				systemTokenCount += estimateToolCallTokens(msg.ToolCall)
			}
			if msg.ToolResult != nil {
				systemTokenCount += estimateToolResultTokens(msg.ToolResult)
			}
		}
	}

	// Reset context to only system messages
	c.Messages = systemMessages
	c.CurrentTokens = systemTokenCount
}

// ClearLastUserMessage removes the most recently added user message from context.
// This is used when retrying a failed request to avoid duplicating the message.
func (c *Context) ClearLastUserMessage() {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := len(c.Messages) - 1; i >= 0; i-- {
		if c.Messages[i].Role == RoleUser {
			tokens := estimateTokens(c.Messages[i].Content)
			c.Messages = append(c.Messages[:i], c.Messages[i+1:]...)
			c.CurrentTokens -= tokens
			return
		}
	}
}

// AddSystemMessage adds a system message to the context
func (c *Context) AddSystemMessage(content string) {
	c.AddMessage(Message{
		Role:      RoleSystem,
		Content:   content,
		Timestamp: time.Now(),
	})
}

// AddUserMessage adds a user message to the context
func (c *Context) AddUserMessage(content string) {
	c.AddMessage(Message{
		Role:      RoleUser,
		Content:   content,
		Timestamp: time.Now(),
	})
}

// AddAssistantMessage adds an assistant message to the context
func (c *Context) AddAssistantMessage(content string) {
	c.AddMessage(Message{
		Role:      RoleAssistant,
		Content:   content,
		Timestamp: time.Now(),
	})
}

// AddToolCallMessage adds a tool call message to the context (Text-pattern fallback)
func (c *Context) AddToolCallMessage(toolName string, params map[string]interface{}) {
	c.AddMessage(Message{
		Role:    RoleAssistant,
		Content: fmt.Sprintf("I'm using the %s tool.", toolName),
		ToolCall: &ToolCall{
			ToolName: toolName,
			Params:   params,
		},
		Timestamp: time.Now(),
	})
}

// AddToolResultMessage adds a tool result message to the context (Text-pattern fallback)
func (c *Context) AddToolResultMessage(result interface{}, err error) {
	var errStr string
	if err != nil {
		errStr = err.Error()
	}

	c.AddMessage(Message{
		Role:    RoleTool,
		Content: "Tool execution result",
		ToolResult: &ToolResult{
			Result: result,
			Error:  errStr,
		},
		Timestamp: time.Now(),
	})
}

// AddNativeToolCallsMessage adds an assistant message containing native ToolCalls.
func (c *Context) AddNativeToolCallsMessage(content string, nativeCalls []anyllm.ToolCall) {
	c.AddMessage(Message{
		Role:            RoleAssistant,
		Content:         content,
		NativeToolCalls: nativeCalls,
		Timestamp:       time.Now(),
	})
}

// AddNativeToolResultMessage adds a native tool result to the context.
func (c *Context) AddNativeToolResultMessage(toolCallID string, result interface{}, err error) {
	var errStr string
	if err != nil {
		errStr = err.Error()
	}

	c.AddMessage(Message{
		Role:    RoleTool,
		Content: "Tool execution result",
		ToolResult: &ToolResult{
			ToolCallID: toolCallID,
			Result:     result,
			Error:      errStr,
		},
		Timestamp: time.Now(),
	})
}

// AddMessage adds a message to the context
func (c *Context) AddMessage(msg Message) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Estimate token count (very rough)
	tokens := estimateTokens(msg.Content)
	if msg.ToolCall != nil {
		tokens += estimateToolCallTokens(msg.ToolCall)
	}
	if len(msg.NativeToolCalls) > 0 {
		for _, tc := range msg.NativeToolCalls {
			tokens += 20 + len(tc.Function.Name) + len(tc.Function.Arguments)
		}
	}
	if msg.ToolResult != nil {
		tokens += estimateToolResultTokens(msg.ToolResult)
	}

	c.logger.Debug("Adding message to context", "role", string(msg.Role), "tokens", tokens, "currentTokens", c.CurrentTokens)
	c.Messages = append(c.Messages, msg)
	c.CurrentTokens += tokens

	// Truncate if needed
	if c.TruncateOldest {
		c.TruncateIfNeeded()
	}
}

// TruncateIfNeeded removes oldest non-system messages if context exceeds max tokens
func (c *Context) TruncateIfNeeded() {
	c.logger.Debug("Checking if context needs truncation", "currentTokens", c.CurrentTokens, "maxTokens", c.MaxTokens)
	if c.CurrentTokens <= c.MaxTokens {
		return
	}

	// Keep system messages and remove oldest non-system messages first
	var newMessages []Message
	var newTokenCount int

	// Always keep system messages
	systemMessages := make([]Message, 0)
	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			systemMessages = append(systemMessages, msg)
			newTokenCount += estimateTokens(msg.Content)
		}
	}

	// Start with system messages
	newMessages = append(newMessages, systemMessages...)

	// Add most recent messages until we're under the token limit
	// Start from the end (most recent) and work backwards
	for i := len(c.Messages) - 1; i >= 0; i-- {
		msg := c.Messages[i]
		if msg.Role == RoleSystem {
			// Already added
			continue
		}

		msgTokens := estimateTokens(msg.Content)
		if msg.ToolCall != nil {
			msgTokens += estimateToolCallTokens(msg.ToolCall)
		}
		if len(msg.NativeToolCalls) > 0 {
			for _, tc := range msg.NativeToolCalls {
				msgTokens += 20 + len(tc.Function.Name) + len(tc.Function.Arguments)
			}
		}
		if msg.ToolResult != nil {
			msgTokens += estimateToolResultTokens(msg.ToolResult)
		}

		if newTokenCount+msgTokens > c.MaxTokens {
			// Would exceed limit, skip this message
			continue
		}

		// Add this message (will be in reverse order for now)
		newMessages = append(newMessages, msg)
		newTokenCount += msgTokens
	}

	// If we have both system and non-system messages, we need to reverse the non-system part
	if len(systemMessages) > 0 && len(newMessages) > len(systemMessages) {
		// Reverse the order of non-system messages to restore chronological order
		reversed := make([]Message, 0, len(newMessages))

		// First add system messages
		reversed = append(reversed, systemMessages...)

		// Then add non-system messages in reverse order (chronological)
		for i := len(newMessages) - 1; i >= len(systemMessages); i-- {
			reversed = append(reversed, newMessages[i])
		}

		newMessages = reversed
	}

	c.Messages = newMessages
	c.CurrentTokens = newTokenCount
}

// GetFormattedMessages returns messages formatted for the LLM
func (c *Context) GetFormattedMessages() []anyllm.Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	c.logger.Debug("Formatting messages for LLM", "messageCount", len(c.Messages))
	formatted := make([]anyllm.Message, 0, len(c.Messages))

	for _, msg := range c.Messages {
		var role string
		switch msg.Role {
		case RoleSystem:
			role = "system"
		case RoleUser:
			role = "user"
		case RoleAssistant:
			role = "assistant"
		case RoleTool:
			// For native tool results, ANY-LLM natively expects RoleTool
			if msg.ToolResult != nil && msg.ToolResult.ToolCallID != "" {
				role = "tool"
			} else {
				// Otherwise map back to user as fallback for text pattern logic
				role = "user"
			}
		default:
			role = "user"
		}

		formattedMsg := anyllm.Message{
			Role: role,
		}

		// Handle different message types
		if len(msg.NativeToolCalls) > 0 {
			formattedMsg.Content = msg.Content
			formattedMsg.ToolCalls = msg.NativeToolCalls
		} else if msg.ToolCall != nil {
			// Basic generic mapping for text-pattern tool calls
			formattedMsg.Content = msg.Content
		} else if msg.ToolResult != nil {
			if msg.ToolResult.ToolCallID != "" {
				// Native API format
				formattedMsg.ToolCallID = msg.ToolResult.ToolCallID
				if msg.ToolResult.Error != "" {
					formattedMsg.Content = fmt.Sprintf("Error: %s", msg.ToolResult.Error)
				} else {
					formattedMsg.Content = formatToolResult(msg.ToolResult.Result)
				}
			} else {
				// Fallback Text XML format
				if msg.ToolResult.Error != "" {
					formattedMsg.Content = fmt.Sprintf("Tool Result:\n<tool_result>\n  <error>%s</error>\n</tool_result>", escapeXML(msg.ToolResult.Error))
				} else {
					content := formatToolResult(msg.ToolResult.Result)
					formattedMsg.Content = "Tool Result:\n" + content
				}
			}
		} else {
			// Regular message
			formattedMsg.Content = msg.Content
		}

		formatted = append(formatted, formattedMsg)
	}

	return formatted
}

// GetMessages returns a copy of all messages
func (c *Context) GetMessages() []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]Message{}, c.Messages...)
}

// formatToolResult formats a tool result for display in the conversation using XML
func formatToolResult(result interface{}) string {
	switch v := result.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	case map[string]interface{}:
		// Format map as XML
		var builder strings.Builder
		builder.WriteString("<tool_result>\n")

		// Sort the keys for consistent output
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		// Add each field as an XML element
		for _, k := range keys {
			val := v[k]
			builder.WriteString(fmt.Sprintf("  <%s>%v</%s>\n", k, formatXMLValue(val), k))
		}

		builder.WriteString("</tool_result>")
		return builder.String()
	default:
		// For simple values, just return the string representation
		return fmt.Sprintf("%v", v)
	}
}

// formatXMLValue formats a value for inclusion in XML
func formatXMLValue(value interface{}) string {
	switch v := value.(type) {
	case string:
		// Escape XML special characters
		return escapeXML(v)
	case []interface{}:
		// Format arrays as nested elements
		var builder strings.Builder
		builder.WriteString("\n")
		for i, item := range v {
			builder.WriteString(fmt.Sprintf("    <item index=\"%d\">%v</item>\n", i, formatXMLValue(item)))
		}
		builder.WriteString("  ")
		return builder.String()
	case map[string]interface{}:
		// Format nested maps as nested XML
		var builder strings.Builder
		builder.WriteString("\n")

		// Sort the keys for consistent output
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		for _, k := range keys {
			builder.WriteString(fmt.Sprintf("    <%s>%v</%s>\n", k, formatXMLValue(v[k]), k))
		}
		builder.WriteString("  ")
		return builder.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

// escapeXML escapes XML special characters
func escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	s = strings.ReplaceAll(s, "'", "&apos;")
	return s
}

// estimateTokens provides a very rough estimate of token count for a string
// This is not accurate but serves as a simple heuristic
func estimateTokens(s string) int {
	// Roughly 4 characters per token as a heuristic
	return len(s) / 4
}

// estimateValueTokens estimates tokens for various value types
func estimateValueTokens(v interface{}) int {
	switch val := v.(type) {
	case string:
		return len(val) / 4
	case map[string]interface{}:
		count := 10 // Overhead for map structure
		for k, vv := range val {
			count += len(k)
			count += estimateValueTokens(vv)
		}
		return count
	case []interface{}:
		count := 5 // Overhead for array structure
		for _, item := range val {
			count += estimateValueTokens(item)
		}
		return count
	default:
		// Numbers, booleans, etc.
		return 5
	}
}

// estimateToolCallTokens estimates the token count for a tool call
func estimateToolCallTokens(tc *ToolCall) int {
	tokens := 20 // Base overhead for tool call
	tokens += len(tc.ToolName)
	for k, v := range tc.Params {
		tokens += len(k)
		tokens += estimateValueTokens(v)
	}
	return tokens
}

// estimateToolResultTokens estimates the token count for a tool result
func estimateToolResultTokens(tr *ToolResult) int {
	tokens := 20 // Base overhead for tool result
	tokens += len(tr.Error)
	tokens += estimateValueTokens(tr.Result)
	return tokens
}
