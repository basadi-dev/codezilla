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
	Role         Role              `json:"role"`
	Content      string            `json:"content"`
	ToolCalls    []anyllm.ToolCall `json:"tool_calls,omitempty"`
	ToolResult   *ToolResult       `json:"tool_result,omitempty"`
	Timestamp    time.Time         `json:"timestamp"`
	// ThinkContent stores the raw <think>...</think> block produced by the model.
	// It is preserved for logging and future analysis but is never sent to the LLM.
	ThinkContent string            `json:"think_content,omitempty"`
}

type ToolResult struct {
	ToolCallID string      `json:"tool_call_id,omitempty"`
	Name       string      `json:"name,omitempty"`
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

func (c *Context) SetMaxTokens(maxTokens int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.MaxTokens = maxTokens
}

// Clone creates a deep copy of the context suitable for initializing a parallel 
// agent without sharing mutable state.
func (c *Context) Clone() *Context {
	c.mu.RLock()
	defer c.mu.RUnlock()

	newCtx := &Context{
		Messages:          make([]Message, len(c.Messages)),
		RollingSummary:    c.RollingSummary,
		SlidingWindowSize: c.SlidingWindowSize,
		MaxTokens:         c.MaxTokens,
		CurrentTokens:     c.CurrentTokens,
		TruncateOldest:    c.TruncateOldest,
		logger:            c.logger,
	}
	
	// Deep copy messages mapping
	for i, msg := range c.Messages {
		newMsg := Message{
			Role:         msg.Role,
			Content:      msg.Content,
			Timestamp:    msg.Timestamp,
			ThinkContent: msg.ThinkContent,
		}
		if len(msg.ToolCalls) > 0 {
			newMsg.ToolCalls = make([]anyllm.ToolCall, len(msg.ToolCalls))
			copy(newMsg.ToolCalls, msg.ToolCalls)
		}
		if msg.ToolResult != nil {
			newMsg.ToolResult = &ToolResult{
				ToolCallID: msg.ToolResult.ToolCallID,
				Result:     msg.ToolResult.Result, // Keep result as is (interface limit)
				Error:      msg.ToolResult.Error,
			}
		}
		newCtx.Messages[i] = newMsg
	}

	return newCtx
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


type messageGroup struct {
	Messages []Message
	Tokens   int
	HasUser  bool
}

func groupMessages(msgs []Message) []messageGroup {
	var groups []messageGroup
	for _, msg := range msgs {
		toks := estimateTokens(msg.Content)
		if len(msg.ToolCalls) > 0 {
			toks += estimateToolCallsTokens(msg.ToolCalls)
		}
		if msg.ToolResult != nil {
			toks += estimateToolResultTokens(msg.ToolResult)
		}

		isToolResult := msg.Role == RoleTool
		if isToolResult && len(groups) > 0 {
			lastGroupIdx := len(groups) - 1
			// Append to the last group if it starts with an Assistant message containing ToolCalls
			if len(groups[lastGroupIdx].Messages) > 0 && groups[lastGroupIdx].Messages[0].Role == RoleAssistant && len(groups[lastGroupIdx].Messages[0].ToolCalls) > 0 {
				groups[lastGroupIdx].Messages = append(groups[lastGroupIdx].Messages, msg)
				groups[lastGroupIdx].Tokens += toks
				continue
			}
		}

		groups = append(groups, messageGroup{
			Messages: []Message{msg},
			Tokens:   toks,
			HasUser:  msg.Role == RoleUser,
		})
	}
	return groups
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

	groups := groupMessages(otherMsgs)

	if len(groups) <= 2 {
		// Nothing meaningful to trim — keep at least the latest user+assistant pair
		return 0, nil
	}

	// Keep the newest 50% (rounded up)
	keepCount := (len(groups) + 1) / 2
	removedGroupsCount := len(groups) - keepCount
	
	var kept []Message
	var evicted []Message
	for i, g := range groups {
		if i < removedGroupsCount {
			evicted = append(evicted, g.Messages...)
		} else {
			kept = append(kept, g.Messages...)
		}
	}

	// Rebuild messages and recount tokens
	newMsgs := make([]Message, 0, len(systemMsgs)+len(kept))
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
		"removed", len(evicted),
		"remaining", len(newMsgs),
		"~tokens", newTokens)

	return len(evicted), evicted
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

// AddAssistantMessageWithThink stores the assistant's response together with
// the raw <think> block it produced. ThinkContent is persisted for logging and
// future analysis but is never forwarded to the LLM.
func (c *Context) AddAssistantMessageWithThink(content, thinkContent string) {
	c.AddMessage(Message{
		Role:         RoleAssistant,
		Content:      content,
		ThinkContent: thinkContent,
		Timestamp:    time.Now(),
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

// AddToolCallsMessageWithThink stores the tool-calling message together with
// the raw <think> block that preceded it. ThinkContent is persisted for logging
// and future analysis but is never forwarded to the LLM.
func (c *Context) AddToolCallsMessageWithThink(content, thinkContent string, calls []anyllm.ToolCall) {
	c.AddMessage(Message{
		Role:         RoleAssistant,
		Content:      content,
		ToolCalls:    calls,
		ThinkContent: thinkContent,
		Timestamp:    time.Now(),
	})
}

// maxToolResultChars caps individual tool results to prevent context explosion.
// Tool results like `ls -R` or large file reads can produce enormous output.
//
// 24 000 chars ≈ ~6 000 tokens, roughly 600 lines of Go/Python code.
// This is large enough to hold a full medium-sized source file in context,
// which is critical for the LLM to generate correct target_content for fileEdit.
// File reads that exceed this limit display an actionable truncation hint.
const maxToolResultChars = 24000

func (c *Context) AddToolResultMessage(toolCallID, name string, result interface{}, err error) {
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
			Name:       name,
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
			return v[:maxChars] + fmt.Sprintf(
				"\n\n[TRUNCATED: output was %d chars, showing first %d. "+
					"If this was a file read, use line_start/line_end parameters to read specific line ranges.]",
				len(v), maxChars)
		}
		return v
	case map[string]interface{}:
		// Serialize to check total size. For structured results (file diffs, tool
		// success maps) we prefer to keep them intact; only stringify-truncate
		// when they blow past the cap.
		data, err := json.Marshal(v)
		if err == nil && len(data) > maxChars {
			return string(data[:maxChars]) + fmt.Sprintf(
				"\n\n[TRUNCATED: output was %d chars, showing first %d. "+
					"If this was a file read, use line_start/line_end parameters to read specific line ranges.]",
				len(data), maxChars)
		}
		return v
	default:
		s := fmt.Sprintf("%v", result)
		if len(s) > maxChars {
			return s[:maxChars] + fmt.Sprintf(
				"\n\n[TRUNCATED: output was %d chars, showing first %d. "+
					"If this was a file read, use line_start/line_end parameters to read specific line ranges.]",
				len(s), maxChars)
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

	// Phase 2: Calculate how many tokens we need to evict to fit the budget.
	// We drop the oldest non-system groups until we reach the budget, and
	// continue dropping until the new first message is a RoleUser, preserving
	// strict alternation and tool-call/result pairs.
	var nonSystemMsgs []Message
	var systemMsgs []Message
	totalTokens := 0

	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			systemMsgs = append(systemMsgs, msg)
			totalTokens += estimateTokens(msg.Content)
		} else {
			nonSystemMsgs = append(nonSystemMsgs, msg)
			toks := estimateTokens(msg.Content)
			if len(msg.ToolCalls) > 0 {
				toks += estimateToolCallsTokens(msg.ToolCalls)
			}
			if msg.ToolResult != nil {
				toks += estimateToolResultTokens(msg.ToolResult)
			}
			totalTokens += toks
		}
	}

	if c.RollingSummary != "" {
		totalTokens += estimateTokens(c.RollingSummary)
	}

	groups := groupMessages(nonSystemMsgs)
	var keptGroups []messageGroup
	var evicted []Message

	// Find the last group that contains a RoleUser message to protect it.
	lastUserGroupIdx := -1
	for i, g := range groups {
		if g.HasUser {
			lastUserGroupIdx = i
		}
	}

	if totalTokens <= budgetTokens {
		keptGroups = groups
	} else {
		tokensToEvict := totalTokens - budgetTokens
		evictedTokens := 0

		for i, g := range groups {
			if i == lastUserGroupIdx {
				keptGroups = append(keptGroups, g)
				continue
			}

			// We drop this group if we haven't met the eviction quota yet,
			// or if we have met the quota but haven't found a User group
			// to start the new sequence with.
			if evictedTokens < tokensToEvict || (len(keptGroups) == 0 && !g.HasUser) {
				evictedTokens += g.Tokens
				evicted = append(evicted, g.Messages...)
				removed += len(g.Messages)
			} else {
				keptGroups = append(keptGroups, g)
			}
		}
	}

	var keptNonSystem []Message
	for _, g := range keptGroups {
		keptNonSystem = append(keptNonSystem, g.Messages...)
	}

	// If protecting the last user group caused it to appear out of order, sort intact messages.
	sortMessages(keptNonSystem, nonSystemMsgs)

	newMsgs := append(systemMsgs, keptNonSystem...)
	c.Messages = newMsgs

	// Recalculate token count for safety
	c.CurrentTokens = 0
	for _, msg := range newMsgs {
		c.CurrentTokens += estimateTokens(msg.Content)
		if len(msg.ToolCalls) > 0 {
			c.CurrentTokens += estimateToolCallsTokens(msg.ToolCalls)
		}
		if msg.ToolResult != nil {
			c.CurrentTokens += estimateToolResultTokens(msg.ToolResult)
		}
	}
	if c.RollingSummary != "" {
		c.CurrentTokens += estimateTokens(c.RollingSummary)
	}

	if removed > 0 {
		c.logger.Info("PreFlightTrim: dropped old messages to fit budget",
			"removed", removed,
			"remaining", len(newMsgs),
			"~tokens", c.CurrentTokens,
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

	var nonSystemMsgs []Message
	var systemMsgs []Message
	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			systemMsgs = append(systemMsgs, msg)
		} else {
			nonSystemMsgs = append(nonSystemMsgs, msg)
		}
	}

	groups := groupMessages(nonSystemMsgs)

	if len(groups) <= c.SlidingWindowSize {
		return nil // everything fits in the window
	}

	toEvict := len(groups) - c.SlidingWindowSize
	var evicted []Message
	var keptGroups []messageGroup
	evictedCount := 0
	foundFirstUser := false

	// Find the last user group index to protect it.
	lastUserGroupIdx := -1
	for i, g := range groups {
		if g.HasUser {
			lastUserGroupIdx = i
		}
	}

	for i, g := range groups {
		if evictedCount < toEvict && i != lastUserGroupIdx {
			evicted = append(evicted, g.Messages...)
			evictedCount++
		} else if !foundFirstUser {
			// Strict LLMs require alternating sequences starting with a user message.
			if !g.HasUser && i != lastUserGroupIdx {
				evicted = append(evicted, g.Messages...)
				evictedCount++
			} else {
				foundFirstUser = true
				keptGroups = append(keptGroups, g)
			}
		} else {
			keptGroups = append(keptGroups, g)
		}
	}

	var keptNonSystem []Message
	for _, g := range keptGroups {
		keptNonSystem = append(keptNonSystem, g.Messages...)
	}

	// Recount tokens for kept messages
	var kept []Message
	kept = append(kept, systemMsgs...)
	kept = append(kept, keptNonSystem...)
	
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
	systemMessages := make([]Message, 0)
	var nonSystemMsgs []Message
	totalTokens := 0

	for _, msg := range c.Messages {
		if msg.Role == RoleSystem {
			systemMessages = append(systemMessages, msg)
			totalTokens += estimateTokens(msg.Content)
		} else {
			nonSystemMsgs = append(nonSystemMsgs, msg)
		}
	}

	for _, msg := range nonSystemMsgs {
		toks := estimateTokens(msg.Content)
		if len(msg.ToolCalls) > 0 {
			toks += estimateToolCallsTokens(msg.ToolCalls)
		}
		if msg.ToolResult != nil {
			toks += estimateToolResultTokens(msg.ToolResult)
		}
		totalTokens += toks
	}

	var keptGroups []messageGroup
	groups := groupMessages(nonSystemMsgs)

	if totalTokens <= c.MaxTokens {
		keptGroups = groups
	} else {
		tokensToEvict := totalTokens - c.MaxTokens
		evictedTokens := 0

		for _, g := range groups {
			if evictedTokens < tokensToEvict || (len(keptGroups) == 0 && !g.HasUser) {
				// Try to compress large, old messages within the group before giving up
				compressed := false
				for i := range g.Messages {
					msg := &g.Messages[i]
					msgToks := estimateTokens(msg.Content)
					if len(msg.ToolCalls) > 0 {
						msgToks += estimateToolCallsTokens(msg.ToolCalls)
					}
					if msg.ToolResult != nil {
						msgToks += estimateToolResultTokens(msg.ToolResult)
					}
					
					oldToks := msgToks
					if msg.Role == RoleTool && msg.ToolResult != nil && msgToks > 200 {
						oldContent := ""
						if msg.ToolResult.Result != nil {
							oldContent = fmt.Sprintf("%v", msg.ToolResult.Result)
						}
						if len(oldContent) > 200 {
							msg.ToolResult.Result = fmt.Sprintf("[TRUNCATED context limit: Tool originally returned %d chars]", len(oldContent))
							msgToks = estimateToolResultTokens(msg.ToolResult)
							compressed = true
						}
					} else if msg.Role == RoleAssistant && len(msg.Content) > 400 {
						msg.Content = fmt.Sprintf("[TRUNCATED context limit] %s...", msg.Content[:200])
						msgToks = estimateTokens(msg.Content)
						if len(msg.ToolCalls) > 0 {
							msgToks += estimateToolCallsTokens(msg.ToolCalls)
						}
						compressed = true
					}

					if compressed {
						saved := oldToks - msgToks
						tokensToEvict -= saved // we don't have to evict as much anymore
						g.Tokens -= saved
					}
				}

				// Check again if we still MUST drop the group to meet the quota
				if evictedTokens < tokensToEvict || (len(keptGroups) == 0 && !g.HasUser) {
					evictedTokens += g.Tokens
					continue // Dropped the entire group
				}
			}

			keptGroups = append(keptGroups, g)
		}
	}

	var keptNonSystem []Message
	for _, g := range keptGroups {
		keptNonSystem = append(keptNonSystem, g.Messages...)
	}

	newMessages = append(systemMessages, keptNonSystem...)
	c.Messages = newMessages

	// Re-sum safely
	c.CurrentTokens = 0
	for _, msg := range newMessages {
		c.CurrentTokens += estimateTokens(msg.Content)
		if len(msg.ToolCalls) > 0 {
			c.CurrentTokens += estimateToolCallsTokens(msg.ToolCalls)
		}
		if msg.ToolResult != nil {
			c.CurrentTokens += estimateToolResultTokens(msg.ToolResult)
		}
	}
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
				formattedMsg.Name = msg.ToolResult.Name
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

func estimateToolResultTokens(result *ToolResult) int {
	if result == nil {
		return 0
	}
	s := fmt.Sprintf("%v", result.Result)
	if result.Error != "" {
		s += result.Error
	}
	return estimateTokens(s)
}

// sortMessages sorts a slice of messages based on their index in a reference slice.
// This is used to restore chronological order if eviction protection caused messages
// to be appended out of order.
func sortMessages(target []Message, reference []Message) {
	// Build a map of message pointer (using timestamp and content as a proxy for identity)
	// to its original index in the reference slice.
	refIndex := make(map[string]int)
	for i, msg := range reference {
		// Create a synthetic key
		key := fmt.Sprintf("%d-%s-%s", msg.Timestamp.UnixNano(), msg.Role, msg.Content)
		refIndex[key] = i
	}

	for i := 0; i < len(target); i++ {
		for j := i + 1; j < len(target); j++ {
			keyI := fmt.Sprintf("%d-%s-%s", target[i].Timestamp.UnixNano(), target[i].Role, target[i].Content)
			keyJ := fmt.Sprintf("%d-%s-%s", target[j].Timestamp.UnixNano(), target[j].Role, target[j].Content)
			
			idxI, okI := refIndex[keyI]
			idxJ, okJ := refIndex[keyJ]
			
			if okI && okJ && idxI > idxJ {
				// Swap
				target[i], target[j] = target[j], target[i]
			}
		}
	}
}
