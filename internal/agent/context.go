package agent

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"codezilla/pkg/logger"
	anyllm "github.com/mozilla-ai/any-llm-go"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Message struct {
	Role       Role              `json:"role"`
	Content    string            `json:"content"`
	ToolCalls  []anyllm.ToolCall `json:"tool_calls,omitempty"`
	ToolResult *ToolResult       `json:"tool_result,omitempty"`
	Timestamp  time.Time         `json:"timestamp"`
}

type ToolResult struct {
	ToolCallID string      `json:"tool_call_id,omitempty"`
	Result     interface{} `json:"result"`
	Error      string      `json:"error,omitempty"`
}

type Context struct {
	mu                sync.RWMutex
	Messages          []Message
	RollingSummary    string // compressed summary of evicted messages
	SlidingWindowSize int    // number of recent non-system messages to keep verbatim (0 = disabled)
	MaxTokens         int
	CurrentTokens     int
	TruncateOldest    bool
	logger            *logger.Logger
}

func NewContext(maxTokens int, log *logger.Logger) *Context {
	if maxTokens <= 0 {
		maxTokens = 4000
	}
	if log == nil {
		log = logger.DefaultLogger()
	}

	return &Context{
		Messages:          []Message{},
		MaxTokens:         maxTokens,
		CurrentTokens:     0,
		TruncateOldest:    true,
		SlidingWindowSize: 20, // default: keep last 20 non-system messages verbatim
		logger:            log,
	}
}

func (c *Context) ClearContext() {
	c.mu.Lock()
	defer c.mu.Unlock()

	var systemMessages []Message
	var systemTokenCount int

	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			systemMessages = append(systemMessages, msg)
			systemTokenCount += estimateTokens(msg.Content)
			if len(msg.ToolCalls) > 0 {
				systemTokenCount += estimateToolCallsTokens(msg.ToolCalls)
			}
			if msg.ToolResult != nil {
				systemTokenCount += estimateToolResultTokens(msg.ToolResult)
			}
		}
	}

	c.Messages = systemMessages
	c.CurrentTokens = systemTokenCount
}

// AggressiveTrim drops the oldest ~50% of non-system messages. This is the
// nuclear option used when the LLM rejects prompt due to context overflow.
// Returns the number of messages removed and the evicted messages (for summarisation).
func (c *Context) AggressiveTrim() (int, []Message) {
	c.mu.Lock()
	defer c.mu.Unlock()

	var systemMsgs []Message
	var otherMsgs []Message
	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			systemMsgs = append(systemMsgs, msg)
		} else {
			otherMsgs = append(otherMsgs, msg)
		}
	}

	if len(otherMsgs) <= 2 {
		// Nothing meaningful to trim — keep at least the latest user+assistant pair
		return 0, nil
	}

	// Keep the newest 50% (rounded up)
	keepCount := (len(otherMsgs) + 1) / 2
	removed := len(otherMsgs) - keepCount
	evicted := make([]Message, removed)
	copy(evicted, otherMsgs[:removed])
	kept := otherMsgs[len(otherMsgs)-keepCount:]

	// Rebuild messages and recount tokens
	newMsgs := make([]Message, 0, len(systemMsgs)+keepCount)
	newMsgs = append(newMsgs, systemMsgs...)
	newMsgs = append(newMsgs, kept...)

	newTokens := 0
	for _, msg := range newMsgs {
		newTokens += estimateTokens(msg.Content)
		if len(msg.ToolCalls) > 0 {
			newTokens += estimateToolCallsTokens(msg.ToolCalls)
		}
		if msg.ToolResult != nil {
			newTokens += estimateToolResultTokens(msg.ToolResult)
		}
	}

	// Keep summary tokens accounted for
	if c.RollingSummary != "" {
		newTokens += estimateTokens(c.RollingSummary)
	}

	c.Messages = newMsgs
	c.CurrentTokens = newTokens
	c.logger.Info("AggressiveTrim: dropped oldest messages",
		"removed", removed,
		"remaining", len(newMsgs),
		"~tokens", newTokens)

	return removed, evicted
}

// ContextStats returns the current message count and estimated token count.
func (c *Context) ContextStats() (msgCount int, estimatedTokens int) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.Messages), c.CurrentTokens
}

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

func (c *Context) AddSystemMessage(content string) {
	c.AddMessage(Message{
		Role:      RoleSystem,
		Content:   content,
		Timestamp: time.Now(),
	})
}

// ReplaceSystemMessage replaces the first system message in context with new content.
// If no system message exists, adds one. Used to inject skill instructions dynamically.
func (c *Context) ReplaceSystemMessage(content string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, msg := range c.Messages {
		if msg.Role == RoleSystem {
			oldTokens := estimateTokens(c.Messages[i].Content)
			newTokens := estimateTokens(content)
			c.Messages[i].Content = content
			c.Messages[i].Timestamp = time.Now()
			c.CurrentTokens = c.CurrentTokens - oldTokens + newTokens
			return
		}
	}
	// No system message found — add one
	c.Messages = append([]Message{{
		Role:      RoleSystem,
		Content:   content,
		Timestamp: time.Now(),
	}}, c.Messages...)
	c.CurrentTokens += estimateTokens(content)
}

func (c *Context) AddUserMessage(content string) {
	c.AddMessage(Message{
		Role:      RoleUser,
		Content:   content,
		Timestamp: time.Now(),
	})
}

func (c *Context) AddAssistantMessage(content string) {
	c.AddMessage(Message{
		Role:      RoleAssistant,
		Content:   content,
		Timestamp: time.Now(),
	})
}

func (c *Context) AddToolCallsMessage(content string, calls []anyllm.ToolCall) {
	c.AddMessage(Message{
		Role:      RoleAssistant,
		Content:   content,
		ToolCalls: calls,
		Timestamp: time.Now(),
	})
}

// maxToolResultChars caps individual tool results to prevent context explosion.
// Tool results like `ls -R` or large file reads can produce enormous output.
const maxToolResultChars = 8000

func (c *Context) AddToolResultMessage(toolCallID string, result interface{}, err error) {
	var errStr string
	if err != nil {
		errStr = err.Error()
	}

	// Cap tool result size at insertion time to prevent context explosion
	result = capToolResult(result, maxToolResultChars)

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

// capToolResult truncates tool results that exceed maxChars.
func capToolResult(result interface{}, maxChars int) interface{} {
	switch v := result.(type) {
	case string:
		if len(v) > maxChars {
			return v[:maxChars] + fmt.Sprintf("\n\n[TRUNCATED: output was %d chars, capped at %d to preserve context]", len(v), maxChars)
		}
		return v
	case map[string]interface{}:
		// Serialize to check total size
		data, err := json.Marshal(v)
		if err == nil && len(data) > maxChars {
			return string(data[:maxChars]) + fmt.Sprintf("\n\n[TRUNCATED: output was %d chars, capped at %d to preserve context]", len(data), maxChars)
		}
		return v
	default:
		s := fmt.Sprintf("%v", result)
		if len(s) > maxChars {
			return s[:maxChars] + fmt.Sprintf("\n\n[TRUNCATED: output was %d chars, capped at %d to preserve context]", len(s), maxChars)
		}
		return result
	}
}

func (c *Context) AddMessage(msg Message) {
	c.mu.Lock()
	defer c.mu.Unlock()

	tokens := estimateTokens(msg.Content)
	if len(msg.ToolCalls) > 0 {
		tokens += estimateToolCallsTokens(msg.ToolCalls)
	}
	if msg.ToolResult != nil {
		tokens += estimateToolResultTokens(msg.ToolResult)
	}

	c.logger.Debug("Adding message to context", "role", string(msg.Role), "tokens", tokens, "currentTokens", c.CurrentTokens)
	c.Messages = append(c.Messages, msg)
	c.CurrentTokens += tokens

	if c.TruncateOldest {
		c.TruncateIfNeeded()
	}
}

// PreFlightTrim proactively trims the context to fit within the given token budget.
// Call this before each LLM request, passing (modelContextWindow - toolSchemaTokens)
// as the budget. Returns the number of messages removed and evicted messages (for summarisation).
func (c *Context) PreFlightTrim(budgetTokens int) (int, []Message) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if budgetTokens <= 0 || c.CurrentTokens <= budgetTokens {
		return 0, nil
	}

	removed := 0

	// Phase 1: Compress large tool results (> 500 chars → summary)
	for i := range c.Messages {
		if c.CurrentTokens <= budgetTokens {
			break
		}
		if c.Messages[i].Role == RoleTool && c.Messages[i].ToolResult != nil {
			resultStr := fmt.Sprintf("%v", c.Messages[i].ToolResult.Result)
			if len(resultStr) > 500 {
				oldTokens := estimateToolResultTokens(c.Messages[i].ToolResult)
				c.Messages[i].ToolResult.Result = fmt.Sprintf(
					"[COMPRESSED: tool result was %d chars — trimmed to save context]",
					len(resultStr),
				)
				newTokens := estimateToolResultTokens(c.Messages[i].ToolResult)
				c.CurrentTokens -= (oldTokens - newTokens)
			}
		}
	}

	if c.CurrentTokens <= budgetTokens {
		c.logger.Info("PreFlightTrim: compressed tool results to fit budget",
			"~tokens", c.CurrentTokens, "budget", budgetTokens)
		return removed, nil
	}

	// Phase 2: Drop oldest non-system messages until we fit, collecting evicted messages
	newMsgs := make([]Message, 0, len(c.Messages))
	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			newMsgs = append(newMsgs, msg)
		}
	}

	// Walk from newest to oldest, keeping messages until budget is exceeded
	var keptNonSystem []Message
	var evicted []Message
	keptTokens := 0
	for _, msg := range newMsgs {
		keptTokens += estimateTokens(msg.Content)
	}
	// Account for rolling summary tokens
	if c.RollingSummary != "" {
		keptTokens += estimateTokens(c.RollingSummary)
	}

	for i := len(c.Messages) - 1; i >= 0; i-- {
		msg := c.Messages[i]
		if msg.Role == RoleSystem {
			continue
		}
		msgToks := estimateTokens(msg.Content)
		if len(msg.ToolCalls) > 0 {
			msgToks += estimateToolCallsTokens(msg.ToolCalls)
		}
		if msg.ToolResult != nil {
			msgToks += estimateToolResultTokens(msg.ToolResult)
		}

		if keptTokens+msgToks <= budgetTokens {
			keptNonSystem = append(keptNonSystem, msg)
			keptTokens += msgToks
		} else {
			evicted = append(evicted, msg)
			removed++
		}
	}

	// Reverse keptNonSystem to restore chronological order
	for i, j := 0, len(keptNonSystem)-1; i < j; i, j = i+1, j-1 {
		keptNonSystem[i], keptNonSystem[j] = keptNonSystem[j], keptNonSystem[i]
	}

	// Reverse evicted to chronological order (oldest first)
	for i, j := 0, len(evicted)-1; i < j; i, j = i+1, j-1 {
		evicted[i], evicted[j] = evicted[j], evicted[i]
	}

	newMsgs = append(newMsgs, keptNonSystem...)
	c.Messages = newMsgs
	c.CurrentTokens = keptTokens

	if removed > 0 {
		c.logger.Info("PreFlightTrim: dropped old messages to fit budget",
			"removed", removed,
			"remaining", len(newMsgs),
			"~tokens", keptTokens,
			"budget", budgetTokens)
	}

	return removed, evicted
}

// SlidingWindowEvict removes the oldest non-system messages that fall outside
// the sliding window. Returns the evicted messages for summarisation.
func (c *Context) SlidingWindowEvict() []Message {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.SlidingWindowSize <= 0 {
		return nil // disabled
	}

	// Count non-system messages
	var nonSystemCount int
	for _, msg := range c.Messages {
		if msg.Role != RoleSystem {
			nonSystemCount++
		}
	}

	if nonSystemCount <= c.SlidingWindowSize {
		return nil // everything fits in the window
	}

	toEvict := nonSystemCount - c.SlidingWindowSize
	var evicted []Message
	var kept []Message
	evictedCount := 0

	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			kept = append(kept, msg)
			continue
		}
		if evictedCount < toEvict {
			evicted = append(evicted, msg)
			evictedCount++
		} else {
			kept = append(kept, msg)
		}
	}

	// Recount tokens for kept messages
	newTokens := 0
	for _, msg := range kept {
		newTokens += estimateTokens(msg.Content)
		if len(msg.ToolCalls) > 0 {
			newTokens += estimateToolCallsTokens(msg.ToolCalls)
		}
		if msg.ToolResult != nil {
			newTokens += estimateToolResultTokens(msg.ToolResult)
		}
	}
	if c.RollingSummary != "" {
		newTokens += estimateTokens(c.RollingSummary)
	}

	c.Messages = kept
	c.CurrentTokens = newTokens

	if len(evicted) > 0 {
		c.logger.Info("SlidingWindowEvict: evicted messages from window",
			"evicted", len(evicted),
			"remaining", len(kept),
			"window_size", c.SlidingWindowSize,
			"~tokens", newTokens)
	}

	return evicted
}

// SetRollingSummary stores a compressed summary of evicted messages.
// Updates the token count to account for the summary.
func (c *Context) SetRollingSummary(summary string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Remove old summary tokens
	if c.RollingSummary != "" {
		c.CurrentTokens -= estimateTokens(c.RollingSummary)
	}

	c.RollingSummary = summary

	// Add new summary tokens
	if summary != "" {
		c.CurrentTokens += estimateTokens(summary)
	}

	c.logger.Info("SetRollingSummary: updated rolling summary",
		"summary_len", len(summary),
		"~tokens", c.CurrentTokens)
}

// EstimateToolSchemaTokens estimates the token overhead of tool schemas.
// This accounts for the JSON schema definitions that are sent alongside messages.
func EstimateToolSchemaTokens(toolCount int) int {
	// Each tool schema averages ~150-300 tokens depending on parameter complexity.
	// Use 250 as a conservative middle estimate.
	return toolCount * 250
}

func (c *Context) TruncateIfNeeded() {
	if c.CurrentTokens <= c.MaxTokens {
		return
	}

	var newMessages []Message
	var newTokenCount int
	systemMessages := make([]Message, 0)

	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			systemMessages = append(systemMessages, msg)
			newTokenCount += estimateTokens(msg.Content)
		}
	}

	newMessages = append(newMessages, systemMessages...)

	for i := len(c.Messages) - 1; i >= 0; i-- {
		msg := c.Messages[i]
		if msg.Role == RoleSystem {
			continue
		}

		msgTokens := estimateTokens(msg.Content)
		if len(msg.ToolCalls) > 0 {
			msgTokens += estimateToolCallsTokens(msg.ToolCalls)
		}
		if msg.ToolResult != nil {
			msgTokens += estimateToolResultTokens(msg.ToolResult)
		}

		if newTokenCount+msgTokens > c.MaxTokens {
			// Try to compress large, old messages rather than dropping entirely
			compressed := false
			if msg.Role == RoleTool && msg.ToolResult != nil && msgTokens > 200 {
				oldContent := ""
				if msg.ToolResult.Result != nil {
					oldContent = fmt.Sprintf("%v", msg.ToolResult.Result)
				}
				if len(oldContent) > 200 {
					msg.ToolResult.Result = fmt.Sprintf("[TRUNCATED context limit: Tool originally returned %d chars]", len(oldContent))
					msgTokens = estimateToolResultTokens(msg.ToolResult)
					compressed = true
				}
			} else if msg.Role == RoleAssistant && len(msg.Content) > 400 {
				msg.Content = fmt.Sprintf("[TRUNCATED context limit] %s...", msg.Content[:200])
				msgTokens = estimateTokens(msg.Content)
				if len(msg.ToolCalls) > 0 {
					msgTokens += estimateToolCallsTokens(msg.ToolCalls)
				}
				compressed = true
			}

			// If compression didn't save enough space, drop it.
			if !compressed || newTokenCount+msgTokens > c.MaxTokens {
				continue
			}
		}

		newMessages = append(newMessages, msg)
		newTokenCount += msgTokens
	}

	if len(systemMessages) > 0 && len(newMessages) > len(systemMessages) {
		reversed := make([]Message, 0, len(newMessages))
		reversed = append(reversed, systemMessages...)
		for i := len(newMessages) - 1; i >= len(systemMessages); i-- {
			reversed = append(reversed, newMessages[i])
		}
		newMessages = reversed
	}

	c.Messages = newMessages
	c.CurrentTokens = newTokenCount
}

func (c *Context) GetFormattedMessages() []anyllm.Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	formatted := make([]anyllm.Message, 0, len(c.Messages)+1) // +1 for possible summary

	// Track where system messages end so we can inject summary after them
	summaryInjected := false

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
			role = "tool"
		default:
			role = "user"
		}

		// Inject rolling summary before the first non-system message
		if !summaryInjected && msg.Role != RoleSystem && c.RollingSummary != "" {
			formatted = append(formatted, anyllm.Message{
				Role:    "system",
				Content: "[Conversation History Summary]\n" + c.RollingSummary,
			})
			summaryInjected = true
		}

		formattedMsg := anyllm.Message{
			Role: role,
		}

		if len(msg.ToolCalls) > 0 {
			formattedMsg.Content = msg.Content
			formattedMsg.ToolCalls = msg.ToolCalls
		} else if msg.ToolResult != nil {
			if msg.ToolResult.ToolCallID != "" {
				formattedMsg.ToolCallID = msg.ToolResult.ToolCallID
				if msg.ToolResult.Error != "" {
					formattedMsg.Content = fmt.Sprintf("Error: %s", msg.ToolResult.Error)
				} else {
					formattedMsg.Content = formatToolResult(msg.ToolResult.Result)
				}
			} else {
				if msg.ToolResult.Error != "" {
					formattedMsg.Content = fmt.Sprintf("Tool Result:\n<tool_result>\n  <error>%s</error>\n</tool_result>", escapeXML(msg.ToolResult.Error))
				} else {
					content := formatToolResult(msg.ToolResult.Result)
					formattedMsg.Content = "Tool Result:\n" + content
				}
			}
		} else {
			formattedMsg.Content = msg.Content
		}

		formatted = append(formatted, formattedMsg)
	}

	return formatted
}

func (c *Context) GetMessages() []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]Message{}, c.Messages...)
}

func formatToolResult(result interface{}) string {
	switch v := result.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	case map[string]interface{}:
		var builder strings.Builder
		builder.WriteString("<tool_result>\n")
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			val := v[k]
			builder.WriteString(fmt.Sprintf("  <%s>%v</%s>\n", k, formatXMLValue(val), k))
		}
		builder.WriteString("</tool_result>")
		return builder.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

func formatXMLValue(value interface{}) string {
	switch v := value.(type) {
	case string:
		return escapeXML(v)
	case []interface{}:
		var builder strings.Builder
		builder.WriteString("\n")
		for i, item := range v {
			builder.WriteString(fmt.Sprintf("    <item index=\"%d\">%v</item>\n", i, formatXMLValue(item)))
		}
		builder.WriteString("  ")
		return builder.String()
	case map[string]interface{}:
		var builder strings.Builder
		builder.WriteString("\n")
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			builder.WriteString(fmt.Sprintf("  <%s>%v</%s>\n", k, formatXMLValue(v[k]), k))
		}
		builder.WriteString("  ")
		return builder.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

func escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	s = strings.ReplaceAll(s, "'", "&apos;")
	return s
}

func estimateTokens(s string) int {
	return len(s) / 4
}

func estimateValueTokens(v interface{}) int {
	switch val := v.(type) {
	case string:
		return len(val) / 4
	case map[string]interface{}:
		count := 10
		for k, vv := range val {
			count += len(k)
			count += estimateValueTokens(vv)
		}
		return count
	case []interface{}:
		count := 5
		for _, item := range val {
			count += estimateValueTokens(item)
		}
		return count
	default:
		return 5
	}
}

func estimateToolCallsTokens(calls []anyllm.ToolCall) int {
	tokens := 0
	for _, tc := range calls {
		tokens += 20 + len(tc.Function.Name)
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err == nil {
			for k, v := range args {
				tokens += len(k)
				tokens += estimateValueTokens(v)
			}
		} else {
			tokens += len(tc.Function.Arguments) / 4
		}
	}
	return tokens
}

func estimateToolResultTokens(tr *ToolResult) int {
	tokens := 20
	tokens += len(tr.Error)
	tokens += estimateValueTokens(tr.Result)
	return tokens
}
