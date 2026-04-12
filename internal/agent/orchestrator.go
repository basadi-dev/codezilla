package agent

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

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
	agent   *agent
	logger  *logger.Logger
	tracker *StreamProcessor
	loops   *loopDetector
}

func NewAgentOrchestrator(a *agent) *AgentOrchestrator {
	return &AgentOrchestrator{
		agent:   a,
		logger:  a.logger,
		tracker: NewStreamProcessor(a.logger),
		loops:   newLoopDetector(a.config.LoopDetectWindow, a.config.LoopDetectMaxRepeat),
	}
}

// Run executes the core processing loop via state machine transitions
func (o *AgentOrchestrator) Run(ctx context.Context, initialMessage string, onToken func(string), stream bool) (string, error) {
	state := StatePrompting
	var finalResponse string
	var currentLLMError error
	var toolsToExecute []anyllm.ToolCall

	o.agent.AddUserMessage(initialMessage)

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
			llmTools := o.agent.buildLLMTools()

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
				"model", o.agent.config.Model,
				"messages", len(msgsForLog),
				"~tokens", approxToks,
				"tools", len(toolNames))

			o.logger.DumpPretty("LLM Request Detail (Prompting)", map[string]any{
				"provider": o.agent.config.Provider,
				"model":    o.agent.config.Model,
				"tools":    toolNames,
			})

			llmStart := time.Now()
			completion, err := o.agent.generateCompletion(ctx, "", llmTools)
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

			if len(completion.Choices) > 0 {
				msg := completion.Choices[0].Message

				text := strings.TrimSpace(msg.ContentString())
				if text == "" && msg.Reasoning != nil {
					text = strings.TrimSpace(msg.Reasoning.Content)
				}

				finalResponse = text

				// Think compression logic
				if threshold := o.agent.config.ThinkCompressThreshold; threshold > 0 {
					startIdx := strings.Index(finalResponse, "<think>")
					endIdx := strings.Index(finalResponse, "</think>")
					if startIdx != -1 && endIdx != -1 && endIdx > startIdx {
						thinkContent := finalResponse[startIdx+7 : endIdx]
						if len(thinkContent) > threshold {
							summary := o.summarizeThink(ctx, thinkContent)
							if summary != "" {
								finalResponse = finalResponse[:startIdx+7] + "\n[Thought Process Summarized]\n" + summary + "\n" + finalResponse[endIdx:]
								o.logger.Info("Compressed <think> block", "original_len", len(thinkContent), "summary_len", len(summary))
							}
						}
					}
				}

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
			llmTools := o.agent.buildLLMTools()
			msgs := o.agent.buildChatMessages()
			sysPrompt := o.agent.buildSystemPrompt()
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
					"model":    o.agent.config.Model,
					"messages": len(msgs),
					"~tokens":  ctxChars / 4,
				})
			}

			o.logger.Info("LLM request (stream)",
				"iter", iter,
				"model", o.agent.config.Model,
				"messages", len(msgs),
				"~tokens", ctxChars/4,
				"tools", len(toolNames))

			o.logger.DumpPretty("LLM Request Detail (Streaming)", map[string]any{
				"provider": o.agent.config.Provider,
				"model":    o.agent.config.Model,
				"tools":    toolNames,
			})

			llmStart := time.Now()
			streamCh, errCh, err := o.agent.llmClient.Stream(ctx, o.agent.config.Provider, o.agent.config.Model, msgs, o.agent.config.Temperature, llmTools)

			if err != nil {
				o.logger.Error("Stream init failed",
					"iter", iter,
					"duration", time.Since(llmStart).Round(time.Millisecond),
					"error", err)
				stream = false
				state = StatePrompting
				continue
			}

			fullResp, nativeTools, streamErr := o.tracker.ProcessChannel(ctx, streamCh, errCh, onToken, func(toolName string) {
				if o.agent.config.OnToolPreparing != nil {
					o.agent.config.OnToolPreparing(toolName)
				}
			})
			if o.agent.config.OnLLMStreamEnd != nil {
				o.agent.config.OnLLMStreamEnd()
			}
			llmDur := time.Since(llmStart)

			if streamErr != nil {
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

			fullResp = strings.TrimPrefix(strings.TrimSpace(fullResp), "Assistant:")
			finalResponse = strings.TrimSpace(fullResp)

			// Think compression logic
			if threshold := o.agent.config.ThinkCompressThreshold; threshold > 0 {
				startIdx := strings.Index(finalResponse, "<think>")
				endIdx := strings.Index(finalResponse, "</think>")
				if startIdx != -1 && endIdx != -1 && endIdx > startIdx {
					thinkContent := finalResponse[startIdx+7 : endIdx]
					if len(thinkContent) > threshold {
						// Summarize the think block to save memory context
						summary := o.summarizeThink(ctx, thinkContent)
						if summary != "" {
							// Replace the large block with the summary
							finalResponse = finalResponse[:startIdx+7] + "\n[Thought Process Summarized]\n" + summary + "\n" + finalResponse[endIdx:]
							o.logger.Info("Compressed <think> block", "original_len", len(thinkContent), "summary_len", len(summary))
						}
					}
				}
			}

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
			cleanedText, parsedTools := ParseLLMResponse(finalResponse, o.logger)
			if len(parsedTools) > 0 {
				toolsToExecute = parsedTools
				finalResponse = cleanedText
				state = StateExecutingTools
			} else {
				state = StateComplete
			}

		case StateExecutingTools:
			if len(toolsToExecute) == 0 {
				state = StatePrompting
				continue
			}

			thinkContent := ""
			if finalResponse != "" {
				thinkContent = finalResponse
			} else {
				thinkContent = fmt.Sprintf("I am calling %d tools natively...", len(toolsToExecute))
			}
			o.agent.context.AddToolCallsMessage(thinkContent, toolsToExecute)

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
					o.agent.context.AddToolResultMessage(r.ID, nil, r.Err)
				} else {
					o.agent.context.AddToolResultMessage(r.ID, r.Result, nil)
				}
			}

			toolsToExecute = nil
			finalResponse = ""
			state = StatePrompting

		case StateErrorRecovery:
			o.logger.Error("LLM Error Recovery",
				"iter", iter,
				"error", currentLLMError)
			finalResponse = "I'm sorry, an error occurred communicating with the model: " + currentLLMError.Error()
			state = StateComplete

		case StateComplete:
			if finalResponse != "" {
				o.agent.AddAssistantMessage(finalResponse)
			}
			return finalResponse, nil
		}
	}
}

// summarizeThink creates a condensed version of a large <think> block using the Summariser model.
func (o *AgentOrchestrator) summarizeThink(ctx context.Context, thinkText string) string {
	summariserModel := o.agent.config.SummariserModel
	if summariserModel == "" {
		summariserModel = o.agent.config.Model // fallback
	}

	msgs := []anyllm.Message{
		{Role: "system", Content: "You are an internal summariser. You receive the raw internal thinking of an AI agent. Your job is to concisely summarize the core logical steps and decisions it made. Be extremely brief (2-4 sentences max), focusing on the reasoning that matters for the next steps."},
		{Role: "user", Content: thinkText},
	}

	o.logger.Debug("Summarizing deep thought", "model", summariserModel, "chars", len(thinkText))

	// Create a fast, no-tool completion call. Low temperature for factuality.
	comp, err := o.agent.llmClient.Complete(ctx, o.agent.config.Provider, summariserModel, msgs, 0.3, nil)
	if err != nil {
		o.logger.Warn("Think summarize failed", "error", err)
		return ""
	}

	if len(comp.Choices) > 0 {
		return strings.TrimSpace(comp.Choices[0].Message.ContentString())
	}
	return ""
}
