package agent

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"strings"

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

		switch state {
		case StatePrompting:
			if stream {
				state = StateStreaming
				continue
			}
			iter++
			// blocking complete
				if o.agent.config.OnLLMCall != nil {
					o.agent.config.OnLLMCall(iter)
				}
				llmTools := o.agent.buildLLMTools()
				completion, err := o.agent.generateCompletion(ctx, "", llmTools)
				if err != nil {
					currentLLMError = err
					state = StateErrorRecovery
					continue
				}

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
			if o.agent.config.OnLLMCall != nil {
				o.agent.config.OnLLMCall(iter)
			}
			llmTools := o.agent.buildLLMTools()
			msgs := o.agent.buildChatMessages()
			sysPrompt := o.agent.buildSystemPrompt()
			if len(msgs) > 0 && msgs[0].Role != "system" {
				msgs = append([]anyllm.Message{{Role: "system", Content: sysPrompt}}, msgs...)
			}

			streamCh, errCh, err := o.agent.llmClient.Stream(ctx, o.agent.config.Provider, o.agent.config.Model, msgs, o.agent.config.Temperature, llmTools)

			if err != nil {
				o.logger.Warn("Streaming failed, falling back to complete", "error", err)
				stream = false
				state = StatePrompting
				continue
			}

			fullResp, nativeTools, streamErr := o.tracker.ProcessChannel(ctx, streamCh, errCh, onToken)

			if streamErr != nil {
				currentLLMError = streamErr
				state = StateErrorRecovery
				continue
			}

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

			for _, tc := range toolsToExecute {
				var params map[string]interface{}
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &params)

				// Loop detection: kill if same tool+args repeated too many consecutive times.
				if looped := o.loops.record(tc.Function.Name, tc.Function.Arguments); looped != "" {
					msg := fmt.Sprintf("Detected infinite loop: tool '%s' called with identical arguments %d times consecutively. Stopping.", looped, o.agent.config.LoopDetectMaxRepeat)
					o.logger.Warn(msg)
					return msg, nil
				}

				if o.agent.config.OnToolExecution != nil {
					o.agent.config.OnToolExecution(tc.Function.Name, params)
				}

				result, err := o.agent.ExecuteTool(ctx, tc.Function.Name, params)
				if err != nil {
					o.agent.context.AddToolResultMessage(tc.ID, nil, err)
				} else {
					o.agent.context.AddToolResultMessage(tc.ID, result, nil)
				}
			}

			toolsToExecute = nil
			finalResponse = ""
			state = StatePrompting

		case StateErrorRecovery:
			o.logger.Warn("LLM Error Recovery", "error", currentLLMError)
			// Simple backoff or inject error and retry once
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
