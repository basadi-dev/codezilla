package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// shouldCreateTodoPlan analyzes if a message warrants automatic todo planning.
// The threshold is intentionally high to avoid false positives for simple queries.
func (a *agent) shouldCreateTodoPlan(message string) bool {
	if !a.config.AutoPlan {
		return false
	}

	// Only trigger on clearly multi-step requests. Use a tighter, more specific set.
	complexIndicators := []string{
		"implement", "refactor", "integrate", "migrate",
		"set up", "configure", "step by step", "task list",
		"and then", "followed by",
	}

	messageLower := strings.ToLower(message)
	indicatorCount := 0

	for _, indicator := range complexIndicators {
		if strings.Contains(messageLower, indicator) {
			indicatorCount++
		}
	}

	// Numbered lists are a strong signal
	if regexp.MustCompile(`\d+\.`).MatchString(message) {
		indicatorCount += 2
	}

	// Long messages that explicitly ask for a plan
	if len(message) > 300 && strings.Contains(messageLower, "plan") {
		indicatorCount++
	}

	// Require 3 or more indicators (was 2)
	return indicatorCount >= 3
}

// createAutomaticTodoPlan creates a todo plan based on the user's message
func (a *agent) createAutomaticTodoPlan(ctx context.Context, message string) (string, error) {
	planPrompt := fmt.Sprintf(`Based on this request, create a todo plan:

"%s"

Use the todoCreate tool with these exact parameters:
- name: A descriptive name for the plan
- description: What this plan aims to achieve
- items: An array of task objects, each with:
  - content: The task description
  - priority: "high", "medium", or "low"
  - dependencies: Array of task IDs that must complete first (optional)

Example format:
<tool>
  <name>todoCreate</name>
  <params>
    <name>Feature Implementation Plan</name>
    <description>Plan for implementing the new user feature</description>
    <items>
      <content>Design the feature</content>
      <priority>high</priority>
    </items>
  </params>
</tool>`, message)

	a.context.AddSystemMessage(planPrompt)

	llmTools := a.buildLLMTools()
	
	targetModel := a.config.PlannerModel
	if targetModel == "" {
		targetModel = a.config.Model
	}

	completion, err := a.generateCompletion(ctx, targetModel, llmTools)
	if err != nil {
		return "", err
	}

	response := ""
	if len(completion.Choices) > 0 {
		response = completion.Choices[0].Message.ContentString()
	}

	_, toolCalls := ParseLLMResponse(response, a.logger)
	for _, tc := range toolCalls {
		if tc.Function.Name == "todoCreate" {
			var params map[string]interface{}
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &params)
			result, err := a.ExecuteTool(ctx, "todoCreate", params)
			if err != nil {
				return "", err
			}
			return fmt.Sprintf("%v", result), nil
		}
	}

	return response, nil
}

// checkAndSuggestNextTodoSteps analyzes current todo progress and suggests next steps
func (a *agent) checkAndSuggestNextTodoSteps(ctx context.Context) {
	result, err := a.ExecuteTool(ctx, "todoAnalyze", map[string]interface{}{})
	if err != nil {
		a.logger.Error("Failed to analyze todo progress", "error", err)
		return
	}

	if result != nil {
		a.context.AddSystemMessage(fmt.Sprintf("Todo Progress Update:\n%v\n\nConsider working on the recommended next task.", result))
	}
}
