package agent

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"codezilla/internal/core/llm"
	"codezilla/internal/session"
	"codezilla/pkg/logger"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// loopDetector tracks recent tool calls and detects infinite-loop patterns.
// A loop is detected when the same (toolName, argsHash) pair appears consecutively
// more than maxRepeat times within the sliding history window.
type loopDetector struct {
	history   []string // ring of "toolName:argsHash" keys
	window    int      // max history length
	maxRepeat int      // consecutive identical calls before kill
}

func newLoopDetector(window, maxRepeat int) *loopDetector {
	if window <= 0 {
		window = 10
	}
	if maxRepeat <= 0 {
		maxRepeat = 3
	}
	return &loopDetector{window: window, maxRepeat: maxRepeat}
}

// record adds a tool call key and returns the loop-detected tool name if a loop
// is found, or empty string if all is well.
func (d *loopDetector) record(toolName, argsJSON string) string {
	// Canonicalize JSON to ignore whitespace or key ordering differences
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &m); err == nil {
		if canonical, err := json.Marshal(m); err == nil {
			argsJSON = string(canonical)
		}
	}

	h := md5.Sum([]byte(argsJSON)) //nolint:gosec // not crypto use
	key := fmt.Sprintf("%s:%x", toolName, h)

	d.history = append(d.history, key)
	if len(d.history) > d.window {
		d.history = d.history[len(d.history)-d.window:]
	}

	// Count consecutive identical tail entries.
	count := 0
	for i := len(d.history) - 1; i >= 0; i-- {
		if d.history[i] == key {
			count++
		} else {
			break
		}
	}
	if count >= d.maxRepeat {
		return toolName
	}
	return ""
}

// OrchestratorState represents the internal state of the Agent
type OrchestratorState int

const (
	StatePrompting OrchestratorState = iota
	StateStreaming
	StateParsing
	StateExecutingTools
	StateErrorRecovery
	StateComplete
)

func (s OrchestratorState) String() string {
	switch s {
	case StatePrompting:
		return "StatePrompting"
	case StateStreaming:
		return "StateStreaming"
	case StateParsing:
		return "StateParsing"
	case StateExecutingTools:
		return "StateExecutingTools"
	case StateErrorRecovery:
		return "StateErrorRecovery"
	case StateComplete:
		return "StateComplete"
	default:
		return "UnknownState"
	}
}

type AgentOrchestrator struct {
	agent                *agent
	logger               *logger.Logger
	tracker              *StreamProcessor
	loops                *loopDetector
	tokens               *TokenTracker
	contextLengthRetries int         // tracks retries for context-length errors within a single Run()
	transientRetries     int         // tracks retries for model provider transient errors
	verifyRetries        int         // tracks retries for post-edit verification failures
	routedModel          string      // per-Run model override set by the router (empty = use config.Model)
	routedTier           RequestTier // per-Run tier set by the router (default = TierDefault)
	prevToolCount        int         // tool count from previous Run (used by router)
	// toolsDisabled is set to true after a 400 "function calls and responses" error.
	// Some cloud-proxied models (e.g. ministral via Ollama cloud gateway) reject
	// requests that include tool schemas. Retrying without tools lets them
	// function as plain text agents.
	toolsDisabled bool
}

const (
	maxContextLengthRetries = 3
	maxTransientRetries     = 3
)

func NewAgentOrchestrator(a *agent) *AgentOrchestrator {
	return &AgentOrchestrator{
		agent:   a,
		logger:  a.logger,
		tracker: NewStreamProcessor(a.logger),
		loops:   newLoopDetector(a.config.LoopDetectWindow, a.config.LoopDetectMaxRepeat),
		tokens:  NewTokenTracker(),
	}
}

// effectiveModel returns the model to use for LLM calls in this Run.
// If the router selected a model, that takes priority; otherwise falls back
// to the agent's configured default model.
func (o *AgentOrchestrator) effectiveModel() string {
	if o.routedModel != "" {
		return o.routedModel
	}
	return o.agent.config.Model
}

// toolsForTier returns the appropriate tool set for the current model tier.
//   - Fast tier: no tools (pure text completion for simple Q&A)
//   - Default tier: core tools only (file ops, grep, execute, repo map)
//   - Heavy tier: all tools (full capability)
//
// This dramatically reduces token overhead for lightweight models that choke
// on large tool schemas.
func (o *AgentOrchestrator) toolsForTier(allTools []anyllm.Tool) []anyllm.Tool {
	if o.toolsDisabled {
		return nil
	}

	switch o.routedTier {
	case TierFast:
		return nil // no tools needed for greetings/simple Q&A
	case TierDefault:
		return filterCoreTools(allTools)
	default:
		return allTools
	}
}

// filterCoreTools returns only the essential tools for standard coding tasks.
// Everything else (web search, project analyzer, sub-agent, todo) is reserved
// for heavy-tier requests where the model is powerful enough to benefit.
func filterCoreTools(tools []anyllm.Tool) []anyllm.Tool {
	var core []anyllm.Tool
	for _, t := range tools {
		if CoreToolNames[t.Function.Name] {
			core = append(core, t)
		}
	}
	return core
}

// Run executes the core processing loop via state machine transitions
func (o *AgentOrchestrator) Run(ctx context.Context, initialMessage string, onToken func(string), stream bool) (string, error) {
	state := StatePrompting
	var finalResponse string
	var currentLLMError error
	var toolsToExecute []anyllm.ToolCall
	var toolFormatRetries int // tracks auto-correction retries for leaked tool calls

	o.agent.AddUserMessage(initialMessage)

	// ---- Model routing: classify the request and pick the best model tier ----
	if o.agent.router != nil && o.agent.router.Enabled {
		tier, reason := o.agent.router.Classify(initialMessage, o.prevToolCount)
		o.routedTier = tier
		routedModel := o.agent.router.ModelForTier(tier)
		if routedModel != o.agent.config.Model {
			o.routedModel = routedModel
			o.logger.Info("Model router",
				"tier", tier.String(),
				"model", routedModel,
				"reason", reason)
		} else {
			o.logger.Debug("Model router: staying on default",
				"tier", tier.String(),
				"reason", reason)
		}
		// Notify UI callback
		if o.agent.config.OnModelRouted != nil {
			o.agent.config.OnModelRouted(routedModel, reason)
		}
		// Record routing decision in session
		if o.agent.config.SessionRecorder != nil {
			o.agent.config.SessionRecorder.Record(session.EventStateChange, map[string]interface{}{
				"routing_tier":   tier.String(),
				"routing_model":  routedModel,
				"routing_reason": reason,
			})
		}
	}

	if o.agent.config.AutoPlan && o.agent.shouldCreateTodoPlan(initialMessage) {
		o.logger.Debug("Creating automatic todo plan for complex task")
		planResponse, err := o.agent.createAutomaticTodoPlan(ctx, initialMessage)
		if err != nil {
			o.logger.Error("Failed to create automatic todo plan", "error", err)
		} else if planResponse != "" {
			o.agent.AddAssistantMessage("[System Background Action: Auto-Plan Created]\n" + planResponse + "\n\nI have successfully generated the plan using the tool. Since the user has already seen the visual checklist, I will NOT output the plan again. I will simply acknowledge the plan is ready and ask what to tackle first.")
		}
	}

	// 0 means unlimited. Use configured limit if set > 0.
	maxIter := o.agent.config.MaxIterations
	iter := 0

	for {
		if maxIter > 0 && iter >= maxIter {
			return "Reached maximum tool execution iterations.", nil
		}

		if o.agent.config.SessionRecorder != nil {
			o.agent.config.SessionRecorder.Record(session.EventStateChange, map[string]interface{}{
				"state": state.String(),
				"iter":  iter,
			})
		}

		switch state {
		case StatePrompting:
			if stream {
				state = StateStreaming
				continue
			}
			iter++
			// blocking complete
			llmTools := o.toolsForTier(o.agent.buildLLMTools())
			// Proactively trim context before sending to LLM
			o.preFlightContextTrim(ctx, len(llmTools))

			if o.agent.config.OnLLMCall != nil {
				msgsForCount := o.agent.buildChatMessages()
				totalChars := 0
				for _, m := range msgsForCount {
					if s, ok := m.Content.(string); ok {
						totalChars += len(s)
					}
				}
				o.agent.config.OnLLMCall(iter, len(msgsForCount), totalChars/4)
			}

			var toolNames []string
			for _, t := range llmTools {
				toolNames = append(toolNames, t.Function.Name)
			}

			msgsForLog := o.agent.buildChatMessages()
			totalChars := 0
			for _, m := range msgsForLog {
				if s, ok := m.Content.(string); ok {
					totalChars += len(s)
				}
			}
			approxToks := totalChars / 4

			o.logger.Info("LLM request (complete)",
				"iter", iter,
				"model", o.effectiveModel(),
				"messages", len(msgsForLog),
				"~tokens", approxToks,
				"tools", len(toolNames))

			o.logger.DumpPretty("LLM Request Detail (Prompting)", map[string]any{
				"provider": o.agent.config.Provider,
				"model":    o.effectiveModel(),
				"tools":    toolNames,
			})

			llmStart := time.Now()
			completion, err := o.agent.generateCompletion(ctx, "", llmTools, o.routedTier)
			llmDur := time.Since(llmStart)

			if err != nil {
				o.logger.Error("LLM complete failed",
					"iter", iter,
					"duration", llmDur.Round(time.Millisecond),
					"error", err)
				currentLLMError = err
				state = StateErrorRecovery
				continue
			}

			o.logger.Info("LLM response (complete)",
				"iter", iter,
				"duration", llmDur.Round(time.Millisecond))

			// Capture token usage from the completion response
			if completion.Usage != nil {
				o.tokens.Record(o.effectiveModel(), completion.Usage)
				o.logger.Info("LLM token usage",
					"prompt", completion.Usage.PromptTokens,
					"completion", completion.Usage.CompletionTokens,
					"total", completion.Usage.TotalTokens)
				if o.agent.config.OnLLMUsage != nil {
					o.agent.config.OnLLMUsage(o.tokens.LastTurn(), o.tokens.SessionTotal(), o.tokens.TurnModelBreakdown())
				}
			}

			if len(completion.Choices) > 0 {
				msg := completion.Choices[0].Message

				text := strings.TrimSpace(msg.ContentString())
				if text == "" && msg.Reasoning != nil {
					text = strings.TrimSpace(msg.Reasoning.Content)
				}

				// Layer 1: sanitise leaked special tokens before any further processing
				text = SanitiseSpecialTokens(text)
				finalResponse = text

				if onToken != nil && text != "" {
					onToken(text)
				}

				o.logger.DumpPretty("Received LLM Response (Prompting)", map[string]any{
					"content":     finalResponse,
					"tools_count": len(msg.ToolCalls),
				})

				if len(msg.ToolCalls) > 0 {
					toolsToExecute = msg.ToolCalls
					state = StateExecutingTools
				} else {
					state = StateParsing
				}
			} else {
				currentLLMError = fmt.Errorf("empty response")
				state = StateErrorRecovery
			}

		case StateStreaming:
			iter++
			llmTools := o.toolsForTier(o.agent.buildLLMTools())

			// Proactively trim context before sending to LLM
			o.preFlightContextTrim(ctx, len(llmTools))

			msgs := o.agent.buildChatMessages()
			sysPrompt := o.agent.buildSystemPromptForTier(o.routedTier)
			if len(msgs) > 0 && msgs[0].Role != "system" {
				msgs = append([]anyllm.Message{{Role: "system", Content: sysPrompt}}, msgs...)
			}

			var toolNames []string
			for _, t := range llmTools {
				toolNames = append(toolNames, t.Function.Name)
			}

			// Compute context size for logging and session recording
			ctxChars := 0
			for _, m := range msgs {
				if s, ok := m.Content.(string); ok {
					ctxChars += len(s)
				}
			}

			if o.agent.config.OnLLMCall != nil {
				o.agent.config.OnLLMCall(iter, len(msgs), ctxChars/4)
			}

			if o.agent.config.SessionRecorder != nil {
				o.agent.config.SessionRecorder.Record(session.EventLLMRequest, map[string]interface{}{
					"provider": o.agent.config.Provider,
					"model":    o.effectiveModel(),
					"messages": len(msgs),
					"~tokens":  ctxChars / 4,
				})
			}

			o.logger.Info("LLM request (stream)",
				"iter", iter,
				"model", o.effectiveModel(),
				"messages", len(msgs),
				"~tokens", ctxChars/4,
				"tools", len(toolNames))

			o.logger.DumpPretty("LLM Request Detail (Streaming)", map[string]any{
				"provider": o.agent.config.Provider,
				"model":    o.effectiveModel(),
				"tools":    toolNames,
			})

			llmStart := time.Now()
			streamCh, errCh, err := o.agent.llmClient.Stream(ctx, o.agent.config.Provider, o.effectiveModel(), msgs, o.agent.config.Temperature, o.agent.config.ReasoningEffort, llmTools)

			if err != nil {
				o.logger.Error("Stream init failed",
					"iter", iter,
					"duration", time.Since(llmStart).Round(time.Millisecond),
					"error", err)
				stream = false
				state = StatePrompting
				continue
			}

			fullResp, nativeTools, streamUsage, streamErr := o.tracker.ProcessChannel(ctx, streamCh, errCh, onToken, func(toolName string) {
				if o.agent.config.OnToolPreparing != nil {
					o.agent.config.OnToolPreparing(toolName)
				}
			})
			if o.agent.config.OnLLMStreamEnd != nil {
				o.agent.config.OnLLMStreamEnd()
			}
			llmDur := time.Since(llmStart)

			if streamErr != nil {
				// Wrap stream errors for context-length detection
				if llm.IsContextLengthError(streamErr) {
					streamErr = fmt.Errorf("%w: %v", llm.ErrContextLengthExceeded, streamErr)
				}
				o.logger.Error("LLM stream failed",
					"iter", iter,
					"duration", llmDur.Round(time.Millisecond),
					"error", streamErr)
				currentLLMError = streamErr
				state = StateErrorRecovery
				continue
			}

			o.logger.Info("LLM response (stream)",
				"iter", iter,
				"duration", llmDur.Round(time.Millisecond),
				"resp_len", len(fullResp),
				"native_tools", len(nativeTools))

			// Capture token usage from stream
			if streamUsage != nil {
				o.tokens.Record(o.effectiveModel(), streamUsage)
				o.logger.Info("LLM token usage (stream)",
					"prompt", streamUsage.PromptTokens,
					"completion", streamUsage.CompletionTokens,
					"total", streamUsage.TotalTokens)
				if o.agent.config.OnLLMUsage != nil {
					o.agent.config.OnLLMUsage(o.tokens.LastTurn(), o.tokens.SessionTotal(), o.tokens.TurnModelBreakdown())
				}
			}

			fullResp = strings.TrimPrefix(strings.TrimSpace(fullResp), "Assistant:")
			// Layer 1: sanitise leaked special tokens before any further processing
			finalResponse = SanitiseSpecialTokens(strings.TrimSpace(fullResp))

			o.logger.DumpPretty("Received LLM Response (Streaming)", map[string]any{
				"content":            finalResponse,
				"native_tools_count": len(nativeTools),
			})

			if len(nativeTools) > 0 {
				toolsToExecute = nativeTools
				state = StateExecutingTools
			} else if finalResponse == "" {
				currentLLMError = fmt.Errorf("empty stream content")
				state = StateErrorRecovery
			} else {
				state = StateParsing
			}

		case StateParsing:
			strippedContent, _ := extractAndStripThinkBlocks(finalResponse)
			cleanedText, parsedTools := ParseLLMResponse(finalResponse, o.logger)
			if len(parsedTools) > 0 {
				toolsToExecute = parsedTools
				finalResponse = cleanedText
				toolFormatRetries = 0 // reset on success
				state = StateExecutingTools
			} else if looksLikeLeakedToolCall(finalResponse) && toolFormatRetries < 2 {
				// Layer 4: auto-correction — the model tried to call a tool but
				// the format was unrecoverable. Inject a correction and re-prompt.
				toolFormatRetries++
				o.logger.Warn("Detected unrecoverable leaked tool call, injecting correction",
					"retry", toolFormatRetries,
					"response_preview", finalResponse[:min(len(finalResponse), 120)])
				correctionMsg := fmt.Sprintf(
					"[SYSTEM CORRECTION] Your previous response contained a malformed tool call "+
						"that could not be executed. You must use the native tool calling mechanism — "+
						"never output tool calls as raw text or JSON in your message. "+
						"The malformed text was: %q. Please retry using the proper tool format.",
					finalResponse[:min(len(finalResponse), 200)],
				)
				// Stash the botched attempt
				stored, think := extractAndStripThinkBlocks(finalResponse)
				if stored != "" || think != "" {
					o.agent.context.AddAssistantMessageWithThink(stored, think)
				}
				// Push as a User message to safely bounce back a warning to the LLM
				o.agent.context.AddUserMessage(correctionMsg)
				finalResponse = ""
				state = StatePrompting
			} else if strings.TrimSpace(strippedContent) == "" && toolFormatRetries < 2 {
				// The model exhausted its output solely on internal thoughts, missing its tool call entirely
				// or abruptly cutting off. Retain its reasoning so it doesn't lose context, and forcefully re-prompt.
				toolFormatRetries++
				o.logger.Warn("Model output only thoughts and no visible content or tools. Reprompting.", "retry", toolFormatRetries)
				
				// Stash the incomplete thoughts onto the context so it remembers its work so far
				o.agent.context.AddAssistantMessageWithThink("", finalResponse)
				
				correctionMsg := "[SYSTEM PROMPT] Your previous response consisted entirely of internal thoughts with no actual output or tool call. If you reached a decision, please execute your tool call now, or provide a text response."
				
				// Push as a User message to safely bounce back a warning to the LLM
				o.agent.context.AddUserMessage(correctionMsg)
				finalResponse = ""
				state = StatePrompting
			} else {
				state = StateComplete
			}

		case StateExecutingTools:
			if len(toolsToExecute) == 0 {
				state = StatePrompting
				continue
			}

			// Ensure all tool calls have a valid ID. Some providers (e.g. Ollama via any-llm-go)
			// may natively emit tool calls without IDs depending on the model tier,
			// causing "400 Bad Request: Not the same number of function calls and responses"
			// if the subsequent ToolResult role messages appear orphaned.
			for i := range toolsToExecute {
				if toolsToExecute[i].ID == "" {
					toolsToExecute[i].ID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), i)
				}
			}

			// Extract and strip <think> blocks — keep content for logging,
			// but never include it in what's sent to the LLM.
			msgContent, thinkContent := extractAndStripThinkBlocks(finalResponse)
			o.agent.context.AddToolCallsMessageWithThink(msgContent, thinkContent, toolsToExecute)

			type ToolResult struct {
				ID     string
				Result interface{}
				Err    error
				Dur    string
				Name   string
			}
			results := make([]ToolResult, len(toolsToExecute))
			var wg sync.WaitGroup
			var loopErr string

			// Synchronous loop-detection check to prevent spinning up threads if trapped
			for i, tc := range toolsToExecute {
				results[i].ID = tc.ID
				results[i].Name = tc.Function.Name
				if looped := o.loops.record(tc.Function.Name, tc.Function.Arguments); looped != "" {
					loopErr = fmt.Sprintf("Detected infinite loop: tool '%s' called with identical arguments %d times consecutively. Stopping.", looped, o.agent.config.LoopDetectMaxRepeat)
					break
				}
			}

			if loopErr != "" {
				o.logger.Warn(loopErr)
				return loopErr, nil
			}

			// Parallel Tool Execution
			for i, tc := range toolsToExecute {
				wg.Add(1)
				go func(idx int, call anyllm.ToolCall) {
					defer wg.Done()
					var params map[string]interface{}
					_ = json.Unmarshal([]byte(call.Function.Arguments), &params)

					if o.agent.config.OnToolExecution != nil {
						o.agent.config.OnToolExecution(call.Function.Name, params)
					}

					start := time.Now()
					res, err := o.agent.ExecuteTool(ctx, call.Function.Name, params)
					dur := time.Since(start).Round(time.Millisecond).String()

					results[idx].Result = res
					results[idx].Err = err
					results[idx].Dur = dur
				}(i, tc)
			}
			wg.Wait()

			// Synchronous recording to maintain context order
			didModifyFiles := false
			for _, r := range results {
				if o.agent.config.SessionRecorder != nil {
					errStr := ""
					if r.Err != nil {
						errStr = r.Err.Error()
					}
					o.agent.config.SessionRecorder.Record(session.EventToolResult, map[string]interface{}{
						"tool":     r.Name,
						"duration": r.Dur,
						"error":    errStr,
					})
				}
				if r.Err != nil {
					o.agent.context.AddToolResultMessage(r.ID, r.Name, nil, r.Err)
				} else {
					o.agent.context.AddToolResultMessage(r.ID, r.Name, r.Result, nil)
				}
				
				// Re-parse params to check if this was a file modification
				var params map[string]interface{}
				for _, tc := range toolsToExecute {
					if tc.ID == r.ID {
						_ = json.Unmarshal([]byte(tc.Function.Arguments), &params)
						break
					}
				}
				if isFileModifyingTool(r.Name, params) && r.Err == nil {
					didModifyFiles = true
				}
			}

			// Run post-edit verification if enabled
			if o.agent.config.AutoVerify && didModifyFiles {
				maxRetries := o.agent.config.MaxVerifyRetries
				if maxRetries <= 0 {
					maxRetries = maxVerifyRetries
				}

				if o.verifyRetries < maxRetries {
					workDir := o.agent.config.WorkingDirectory
					if workDir == "" {
						workDir = "."
					}

					cmds := ResolveVerifyCommands(workDir, o.agent.config.VerifyProfiles)

					if len(cmds) > 0 {
						vr := RunVerification(ctx, workDir, cmds)
						if !vr.Passed {
							o.verifyRetries++
							if o.agent.config.OnVerifyFailed != nil {
								o.agent.config.OnVerifyFailed(vr.Errors, o.verifyRetries)
							}

							// Inject error feedback to force self-correction
							errMsg := fmt.Sprintf("System Verification failed after your edits:\n\n%s\n\nPlease fix the errors above.", strings.Join(vr.Errors, "\n\n"))
							o.agent.context.AddSystemMessage(errMsg)
							
							// Escalate reasoning effort for the retry
							newEffort := reasoningEffortForRetry(o.agent.config.ReasoningEffort, o.verifyRetries)
							if newEffort != o.agent.config.ReasoningEffort {
								o.agent.SetReasoningEffort(newEffort)
								o.logger.Info("Escalating reasoning effort for verify retry", "new_effort", newEffort)
							}
							
						} else {
							// Verification passed, reset counter
							o.verifyRetries = 0
							if o.agent.config.OnVerifyPassed != nil {
								o.agent.config.OnVerifyPassed()
							}
						}
					}
				} else {
					o.logger.Warn("Max verify retries reached, proceeding with errors")
					o.verifyRetries = 0 // reset for the next turn
				}
			}

			toolsToExecute = nil
			finalResponse = ""
			state = StatePrompting

		case StateErrorRecovery:
			result := o.handleErrorRecovery(ctx, currentLLMError, stream)
			if result.FatalError != nil {
				return "", result.FatalError
			}
			if result.FinalResponse != "" {
				finalResponse = result.FinalResponse
			}
			currentLLMError = nil
			state = result.NextState

		case StateComplete:
			if finalResponse != "" {
				// Extract and strip <think> blocks — keep content for logging,
				// but never include it in what's sent to the LLM.
				stored, thinkContent := extractAndStripThinkBlocks(finalResponse)
				if stored != "" {
					o.agent.context.AddAssistantMessageWithThink(stored, thinkContent)
				}
			}
			return finalResponse, nil
		}
	}
}
// Context management methods (preFlightContextTrim, summarizeAndStore,
// summarizeEvictedMessages, summarizeThink, stripThinkBlocks,
// extractAndStripThinkBlocks) are defined in orchestrator_context.go.
//
// Error recovery (handleErrorRecovery, promptingState) is defined
// in orchestrator_errors.go.
