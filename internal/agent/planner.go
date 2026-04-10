package agent

import (
	"context"
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
	// Create a special prompt to analyze and create a plan
	planPrompt := fmt.Sprintf(`Based on this request, create a todo plan:

"%s"

Use the todo_create tool with these exact parameters:
- name: A descriptive name for the plan
- description: What this plan aims to achieve
- items: An array of task objects, each with:
  - content: The task description
  - priority: "high", "medium", or "low"
  - dependencies: Array of task IDs that must complete first (optional)

Example format:
<tool>
  <name>todo_create</name>
  <params>
    <name>Feature Implementation Plan</name>
    <description>Plan for implementing the new user feature</description>
    <items>
      <content>Design the feature</content>
      <priority>high</priority>
    </items>
    <items>
      <content>Implement backend</content>
      <priority>high</priority>
      <dependencies>task_1</dependencies>
    </items>
  </params>
</tool>`, message)

	// Temporarily add this as a system message
	a.context.AddSystemMessage(planPrompt)

	// Generate response which should create a todo plan
	response, err := a.generateResponse(ctx)
	if err != nil {
		return "", err
	}

	// Process any tool calls in the response
	toolCall, _, hasTool := a.extractToolCall(response)
	if hasTool && toolCall.ToolName == "todo_create" {
		result, err := a.ExecuteTool(ctx, toolCall.ToolName, toolCall.Params)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("%v", result), nil
	}

	return response, nil
}

// checkAndSuggestNextTodoSteps analyzes current todo progress and suggests next steps
func (a *agent) checkAndSuggestNextTodoSteps(ctx context.Context) {
	result, err := a.ExecuteTool(ctx, "todo_analyze", map[string]interface{}{})
	if err != nil {
		a.logger.Error("Failed to analyze todo progress", "error", err)
		return
	}

	if result != nil {
		a.context.AddSystemMessage(fmt.Sprintf("Todo Progress Update:\n%v\n\nConsider working on the recommended next task.", result))
	}
}
