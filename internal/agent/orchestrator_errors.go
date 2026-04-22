package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"codezilla/internal/core/llm"
)

// errorRecoveryResult holds the outcome of error recovery.
type errorRecoveryResult struct {
	NextState     OrchestratorState
	FinalResponse string // non-empty only when recovery decides to complete with a message
	FatalError    error  // non-nil only when the error is unrecoverable
}

// handleErrorRecovery processes an LLM error and determines the next action.
// It handles context-length errors, tool-call mismatches, and transient provider errors.
// It fires the OnLLMError callback so the UI can display what's happening.
func (o *AgentOrchestrator) handleErrorRecovery(ctx context.Context, llmErr error, stream bool) errorRecoveryResult {
	model := o.effectiveModel()

	// --- Context-length error: trim and retry ---
	if errors.Is(llmErr, llm.ErrContextLengthExceeded) && o.contextLengthRetries < maxContextLengthRetries {
		o.contextLengthRetries++
		o.notifyLLMError(model, llmErr, true)

		msgsBefore, toksBefore := o.agent.context.ContextStats()
		removed, evicted := o.agent.context.AggressiveTrim()
		msgsAfter, toksAfter := o.agent.context.ContextStats()

		// Summarise evicted messages before retrying
		if len(evicted) > 0 {
			o.summarizeAndStore(ctx, evicted)
		}

		o.logger.Warn("Context length exceeded — auto-trimming and retrying",
			"retry", o.contextLengthRetries,
			"max_retries", maxContextLengthRetries,
			"msgs_before", msgsBefore,
			"msgs_after", msgsAfter,
			"~toks_before", toksBefore,
			"~toks_after", toksAfter,
			"removed", removed)

		if removed == 0 {
			// Nothing left to trim — give up
			o.logger.Error("Cannot trim further, context is minimal")
			o.notifyLLMError(model, llmErr, false)
			return errorRecoveryResult{
				NextState:     StateComplete,
				FinalResponse: "I'm sorry, the prompt is too long for this model's context window and I cannot trim further. Try using a model with a larger context window, or start a new conversation.",
			}
		}

		return errorRecoveryResult{NextState: o.promptingState(stream)}
	}

	// --- Tool-call mismatch: disable tools and retry ---
	errStr := llmErr.Error()
	if !o.toolsDisabled && strings.Contains(errStr, "function calls and responses") {
		o.toolsDisabled = true
		o.notifyLLMError(model, fmt.Errorf("tool-call mismatch (disabling tools): %w", llmErr), true)
		o.logger.Warn("Tool-call mismatch 400 error — disabling tools and retrying as plain-text agent",
			"model", model)
		return errorRecoveryResult{NextState: o.promptingState(stream)}
	}

	// --- Transient provider error: exponential backoff retry ---
	if llm.IsTransientError(llmErr) && o.transientRetries < maxTransientRetries {
		o.transientRetries++
		delay := time.Duration(1<<o.transientRetries) * time.Second
		o.notifyLLMError(model, fmt.Errorf("%v (retrying in %v, attempt %d/%d)", llmErr, delay, o.transientRetries, maxTransientRetries), true)
		o.logger.Warn(fmt.Sprintf("Provider error detected: %v. Retrying in %v (Attempt %d/%d)...",
			llmErr, delay, o.transientRetries, maxTransientRetries))
		time.Sleep(delay)
		return errorRecoveryResult{NextState: o.promptingState(stream)}
	}

	// --- Unrecoverable ---
	o.logger.Error("LLM Error Recovery: unrecoverable", "error", llmErr)
	if errors.Is(llmErr, llm.ErrContextLengthExceeded) {
		fatalErr := fmt.Errorf("[%s] the prompt is too long for this model's context window. Try `/clear` to reset the conversation, or switch to a model with a larger context window: %w", model, llmErr)
		o.notifyLLMError(model, fatalErr, false)
		return errorRecoveryResult{FatalError: fatalErr}
	}
	fatalErr := fmt.Errorf("[%s] %w", model, llmErr)
	o.notifyLLMError(model, fatalErr, false)
	return errorRecoveryResult{FatalError: fatalErr}
}

// notifyLLMError fires the OnLLMError callback if configured.
func (o *AgentOrchestrator) notifyLLMError(model string, err error, willRetry bool) {
	if o.agent.config.OnLLMError != nil {
		o.agent.config.OnLLMError(model, err, willRetry)
	}
}

// promptingState returns the appropriate initial state based on whether streaming is enabled.
func (o *AgentOrchestrator) promptingState(stream bool) OrchestratorState {
	if stream {
		return StateStreaming
	}
	return StatePrompting
}
