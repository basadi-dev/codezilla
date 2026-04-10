package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"codezilla/pkg/logger"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

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
}

func NewAgentOrchestrator(a *agent) *AgentOrchestrator {
	return &AgentOrchestrator{
		agent:   a,
		logger:  a.logger,
		tracker: NewStreamProcessor(a.logger),
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
			o.agent.AddAssistantMessage(planResponse)
		}
	}

	maxIter := 10
	iter := 0

	for {
		if iter >= maxIter {
			return "Reached maximum tool execution iterations.", nil
		}

		switch state {
		case StatePrompting:
			iter++
			if stream {
				state = StateStreaming
			} else {
				// blocking complete
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
			}

		case StateStreaming:
			iter++
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
