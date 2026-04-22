package agent

import (
	"context"
	"fmt"
	"strings"
	"time"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// preFlightContextTrim proactively trims context before an LLM call.
// It queries the model's actual context window from the provider (Ollama /api/show)
// and accounts for tool schema overhead. Falls back to MaxTokens if unknown.
// Also runs sliding window eviction and summarisation.
func (o *AgentOrchestrator) preFlightContextTrim(ctx context.Context, toolCount int) {
	// Step 1: Sliding window eviction (proactive, before budget check)
	if evicted := o.agent.context.SlidingWindowEvict(); len(evicted) > 0 {
		o.summarizeAndStore(ctx, evicted)
	}

	budget := o.agent.config.MaxTokens
	if budget <= 0 {
		return // no budget configured; skip
	}

	// Try to discover the real model context window
	if o.agent.llmClient != nil {
		queryCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		realCtx := o.agent.llmClient.GetModelContextLength(queryCtx, o.agent.config.Provider, o.effectiveModel())
		cancel()

		if realCtx > 0 {
			o.logger.Debug("Discovered model context window",
				"model", o.effectiveModel(),
				"context_window", realCtx,
				"config_max_tokens", budget)
			// Use the smaller of config budget and real context window
			if realCtx < budget {
				budget = realCtx
			}
		}
	}

	// Update the Agent's context limit to dynamically scale UI feedback
	o.agent.context.SetMaxTokens(budget)

	// Reserve space for tool schemas + response
	schemaOverhead := EstimateToolSchemaTokens(toolCount)
	responseReserve := 1024 // leave room for the LLM to generate a response
	effectiveBudget := budget - schemaOverhead - responseReserve

	if effectiveBudget <= 0 {
		o.logger.Warn("PreFlight: tool schema overhead exceeds budget",
			"budget", budget,
			"schema_overhead", schemaOverhead)
		return
	}

	_, currentTokens := o.agent.context.ContextStats()
	if currentTokens > effectiveBudget {
		o.logger.Info("PreFlight: context exceeds budget, trimming proactively",
			"current_tokens", currentTokens,
			"effective_budget", effectiveBudget,
			"schema_overhead", schemaOverhead)
		_, evicted := o.agent.context.PreFlightTrim(effectiveBudget)
		if len(evicted) > 0 {
			o.summarizeAndStore(ctx, evicted)
		}
	}
}

// summarizeAndStore summarises evicted messages and stores the rolling summary.
// Falls back to keeping the existing summary if summarisation fails.
func (o *AgentOrchestrator) summarizeAndStore(ctx context.Context, evicted []Message) {
	if len(evicted) == 0 {
		return
	}

	existingSummary := o.agent.context.RollingSummary
	newSummary := o.summarizeEvictedMessages(ctx, existingSummary, evicted)
	if newSummary != "" {
		o.agent.context.SetRollingSummary(newSummary)
	}
	// If summarisation failed, existing summary is preserved (no-op)
}

// summarizeEvictedMessages uses the SummariserModel to incrementally update
// the rolling conversation summary with newly evicted messages.
func (o *AgentOrchestrator) summarizeEvictedMessages(ctx context.Context, existingSummary string, evicted []Message) string {
	summariserModel := o.agent.GetModelForTier(TierFast)
	if summariserModel == "" {
		o.logger.Debug("No fast model configured, skipping eviction summary")
		return "" // no summariser = fall back to hard-drop
	}

	// Format evicted messages for the summariser
	var evictedText strings.Builder
	for _, msg := range evicted {
		evictedText.WriteString(fmt.Sprintf("[%s]: ", msg.Role))
		if msg.Content != "" {
			// Cap content to avoid sending huge messages to summariser
			content := msg.Content
			if len(content) > 500 {
				content = content[:500] + "..."
			}
			evictedText.WriteString(content)
		}
		if msg.ToolResult != nil {
			result := fmt.Sprintf("%v", msg.ToolResult.Result)
			if len(result) > 200 {
				result = result[:200] + "..."
			}
			evictedText.WriteString(fmt.Sprintf(" [tool_result: %s]", result))
		}
		if len(msg.ToolCalls) > 0 {
			var toolNames []string
			for _, tc := range msg.ToolCalls {
				toolNames = append(toolNames, tc.Function.Name)
			}
			evictedText.WriteString(fmt.Sprintf(" [called: %s]", strings.Join(toolNames, ", ")))
		}
		evictedText.WriteString("\n")
	}

	var userContent string
	if existingSummary != "" {
		userContent = fmt.Sprintf("EXISTING SUMMARY:\n%s\n\nNEWLY EVICTED MESSAGES:\n%s", existingSummary, evictedText.String())
	} else {
		userContent = fmt.Sprintf("EVICTED MESSAGES:\n%s", evictedText.String())
	}

	msgs := []anyllm.Message{
		{Role: "system", Content: `You are an internal context compactor. Given evicted conversation messages (and optionally an existing summary), produce an updated summary that captures:
- Key decisions made
- Files read or modified (with paths)
- Important findings, conclusions, and error resolutions
- User preferences expressed
- Current task state and progress
- Tool results that matter for future turns

Be extremely concise (max 300 tokens). Use bullet points. Never include raw file contents or full tool outputs. 
IMPORTANT: You MUST end your summary with a single line starting with "Current goal: " that explicitly states what the user originally asked for and what the agent is currently trying to accomplish.`},
		{Role: "user", Content: userContent},
	}

	o.logger.Debug("Summarising evicted messages",
		"model", summariserModel,
		"evicted_count", len(evicted),
		"existing_summary_len", len(existingSummary))

	if o.agent.config.OnContextSummarizing != nil {
		o.agent.config.OnContextSummarizing()
	}

	comp, err := o.agent.llmClient.Complete(ctx, o.agent.config.Provider, summariserModel, msgs, 0.2, "", nil)
	if err != nil {
		o.logger.Warn("Eviction summary failed, keeping existing summary", "error", err)
		return existingSummary
	}

	if len(comp.Choices) > 0 {
		result := strings.TrimSpace(comp.Choices[0].Message.ContentString())
		o.logger.Info("Eviction summary complete",
			"new_summary_len", len(result),
			"evicted_messages", len(evicted))
		return result
	}

	return existingSummary
}

// summarizeThink creates a condensed version of a large <think> block using the Summariser model.
func (o *AgentOrchestrator) summarizeThink(ctx context.Context, thinkText string) string {
	summariserModel := o.agent.GetModelForTier(TierFast)
	if summariserModel == "" {
		summariserModel = o.agent.GetModelForTier(TierDefault) // fallback
	}

	msgs := []anyllm.Message{
		{Role: "system", Content: "You are an internal summariser. You receive the raw internal thinking of an AI agent. Your job is to concisely summarize the core logical steps and decisions it made. Be extremely brief (2-4 sentences max), focusing on the reasoning that matters for the next steps."},
		{Role: "user", Content: thinkText},
	}

	o.logger.Debug("Summarizing deep thought", "model", summariserModel, "chars", len(thinkText))

	// Create a fast, no-tool completion call. Low temperature for factuality.
	comp, err := o.agent.llmClient.Complete(ctx, o.agent.config.Provider, summariserModel, msgs, 0.3, "", nil)
	if err != nil {
		o.logger.Warn("Think summarize failed", "error", err)
		return ""
	}

	if len(comp.Choices) > 0 {
		return strings.TrimSpace(comp.Choices[0].Message.ContentString())
	}
	return ""
}

// stripThinkBlocks removes all <think>...</think> blocks from a response.
// Think blocks are internal reasoning meant for display only — they should
// not be stored in conversation context where they confuse subsequent turns.
func stripThinkBlocks(s string) string {
	stripped, _ := extractAndStripThinkBlocks(s)
	return stripped
}

// extractAndStripThinkBlocks extracts the content of all <think>...</think> blocks
// and returns the stripped response alongside the joined think content.
// The think content is suitable for storing on a Message for logging and future
// analysis; the stripped response is what gets sent to the LLM.
func extractAndStripThinkBlocks(s string) (stripped, thinkContent string) {
	var thinks []string
	for {
		start := strings.Index(s, "<think>")
		end := strings.Index(s, "</think>")
		if start == -1 || end == -1 || end <= start {
			break
		}
		thinks = append(thinks, strings.TrimSpace(s[start+7:end]))
		s = s[:start] + s[end+len("</think>"):]
	}
	return strings.TrimSpace(s), strings.Join(thinks, "\n---\n")
}
