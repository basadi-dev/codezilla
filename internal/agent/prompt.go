package agent

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"codezilla/internal/tools"
	anyllm "github.com/mozilla-ai/any-llm-go"
)

// PromptTemplate contains templates for different prompt components
type PromptTemplate struct {
	SystemTemplate    string
	UserTemplate      string
	AssistantTemplate string
	ToolTemplate      string
}

// DefaultPromptTemplate returns the default prompt template
func DefaultPromptTemplate() *PromptTemplate {
	return &PromptTemplate{
		SystemTemplate: `You are a helpful AI assistant with access to a set of tools. When you need to use a tool, you MUST format your response using XML format like this:
<tool>
  <name>toolName</name>
  <params>
    <param1>value1</param1>
    <param2>value2</param2>
  </params>
</tool>

IMPORTANT: Always use the XML format shown above, NEVER use JSON format inside the tool tags. XML is required for proper tool execution.

Wait for the tool response before continuing the conversation. The available tools are:

{{tools}}

## Planning and Task Management
When given complex tasks:
1. Use todoCreate to break down complex requests into manageable steps
   - Provide a descriptive "name" for the plan
   - Add a "description" of what the plan achieves
   - Create an "items" array with task objects containing "content", "priority", and optional "dependencies"
2. Use todoList to view your current tasks and progress
3. Use todoUpdate to mark tasks as in_progress when starting them and completed when done
   - Provide "task_id" and "status" (pending/in_progress/completed/cancelled)
4. Use todoAnalyze to get recommendations on what to work on next
5. Always update task status as you progress through the plan

Example todoCreate usage:
<tool>
  <name>todoCreate</name>
  <params>
    <name>New Feature Implementation</name>
    <description>Implement user authentication feature</description>
    <items>
      <content>Design authentication flow</content>
      <priority>high</priority>
    </items>
    <items>
      <content>Implement backend API</content>
      <priority>high</priority>
    </items>
  </params>
</tool>

## Code Exploration — Tool Priority
When a user asks you to find, locate, or understand code, use tools in this order:
1. **grepSearch** first — runs in <50ms, handles exact symbol/function/pattern lookups perfectly
2. **repoMapGenerator** — runs in <500ms, gives a structural outline (functions, types, classes) without reading full files
3. **listFiles** — use to understand directory layout before diving into files
4. **fileRead** (via fileManage) — read specific files once you know which ones matter
5. **projectScanAnalyzer** — use ONLY when the above tools cannot answer the question; it performs deep per-file LLM analysis on each candidate file and is significantly slower. Always pass specificDirs to limit scope; never scan the whole repo blindly.

Remember:
1. OUTPUT RULE — CRITICAL: Your response is the FINAL message shown to the user. NEVER include internal reasoning, deliberation, or meta-commentary. Banned phrases include: "The user wants...", "We need to...", "I think...", "Let me consider...", "Perhaps they want...", "So we should...", "They didn't ask..." — any such text must be deleted before responding. Write ONLY your answer.
2. PLAN DISPLAY RULE: When you call todoCreate, the task list is ALREADY rendered in the terminal UI automatically. Do NOT repeat, re-list, or summarise the tasks in your text response. Just reply with one short sentence confirming the plan was created and ask what to tackle first (if appropriate).
3. Use tools when needed to gather information or perform actions
4. Don't make up information - use tools to get accurate data
5. Keep responses concise and direct — no preambles, no meta-commentary, no redundant tables
6. ALWAYS use XML format for tool calls, not JSON
7. Create todo plans for complex multi-step tasks
8. Update todo status as you work through tasks

## Code Editing and Compilation
When writing or modifying code, follow these best practices:
1. **Targeted Edits**: Use the **fileEdit** tool (or **fileManage** with action: 'edit') to replace specific blocks of text in existing files. Do not rewrite entire files unless necessary.
2. **Exact Matching**: When using fileEdit, the **target_content** MUST EXACTLY MATCH the file's current text, including all spaces, tabs, and newlines.
3. **Compile and Format**: After making code changes, ALWAYS use the **execute** tool to format the code (e.g., 'go fmt ./...', 'npm run format') and verify compilation/tests (e.g., 'go build ./...', 'go test ./...'). Do not wait for the user to ask for verification.
4. **Shell Execution**: Use the **execute** tool to run basic shell commands when needed. Assume standard tools are available. Remember that the execute tool DOES NOT support shell operators like '&&', '||', or pipes '|'. Execute commands separately.`,

		UserTemplate: `{{content}}`,

		AssistantTemplate: `{{content}}`,

		ToolTemplate: `Tool result: {{result}}`,
	}
}

// FormatSystemPrompt formats the system prompt with tool specifications
func FormatSystemPrompt(template string, toolSpecs []tools.ToolSpec) string {
	// Convert tool specs to a readable format
	toolsDescription := formatToolSpecsForPrompt(toolSpecs)

	// Replace the {{tools}} placeholder with the tool descriptions
	return strings.Replace(template, "{{tools}}", toolsDescription, 1)
}

// formatToolSpecsForPrompt formats tool specifications in a readable way for the prompt
func formatToolSpecsForPrompt(specs []tools.ToolSpec) string {
	var builder strings.Builder

	for _, spec := range specs {
		builder.WriteString(fmt.Sprintf("## %s\n", spec.Name))
		builder.WriteString(fmt.Sprintf("Description: %s\n", spec.Description))

		// Format parameters
		builder.WriteString("Parameters:\n")

		if spec.ParameterSchema.Properties != nil {
			formatSchemaProperties(&builder, spec.ParameterSchema.Properties, spec.ParameterSchema.Required, "- ", 0)
		}

		builder.WriteString("\n")
	}

	return builder.String()
}

// formatSchemaProperties recursively formats schema properties
func formatSchemaProperties(builder *strings.Builder, properties map[string]tools.JSONSchema, required []string, indent string, depth int) {
	for paramName, paramSchema := range properties {
		isRequired := contains(required, paramName)
		requiredStr := ""
		if isRequired {
			requiredStr = " (required)"
		}

		// Write parameter info
		builder.WriteString(fmt.Sprintf("%s%s: %s%s [%s]",
			strings.Repeat("  ", depth)+indent,
			paramName,
			paramSchema.Description,
			requiredStr,
			paramSchema.Type))

		// Add enum values if present
		if len(paramSchema.Enum) > 0 {
			builder.WriteString(" (options: ")
			for i, v := range paramSchema.Enum {
				if i > 0 {
					builder.WriteString(", ")
				}
				builder.WriteString(fmt.Sprintf("%v", v))
			}
			builder.WriteString(")")
		}

		// Add default value if present
		if paramSchema.Default != nil {
			builder.WriteString(fmt.Sprintf(" (default: %v)", paramSchema.Default))
		}

		builder.WriteString("\n")

		// Handle nested properties for objects
		if paramSchema.Type == "object" && paramSchema.Properties != nil {
			builder.WriteString(fmt.Sprintf("%s  Properties:\n", strings.Repeat("  ", depth+1)))
			formatSchemaProperties(builder, paramSchema.Properties, paramSchema.Required, "- ", depth+2)
		}

		// Handle array items
		if paramSchema.Type == "array" && paramSchema.Items != nil {
			builder.WriteString(fmt.Sprintf("%s  Array items:\n", strings.Repeat("  ", depth+1)))
			if paramSchema.Items.Type == "object" && paramSchema.Items.Properties != nil {
				formatSchemaProperties(builder, paramSchema.Items.Properties, paramSchema.Items.Required, "- ", depth+2)
			} else {
				builder.WriteString(fmt.Sprintf("%s  - Type: %s\n", strings.Repeat("  ", depth+1), paramSchema.Items.Type))
				if paramSchema.Items.Description != "" {
					builder.WriteString(fmt.Sprintf("%s    Description: %s\n", strings.Repeat("  ", depth+1), paramSchema.Items.Description))
				}
			}
		}
	}
}

// FormatToolCallPrompt formats a tool call message for the LLM
func FormatToolCallPrompt(toolCall *anyllm.ToolCall) string {
	var builder strings.Builder
	builder.WriteString("<tool>\n")
	builder.WriteString(fmt.Sprintf("  <name>%s</name>\n", escapeXML(toolCall.Function.Name)))
	builder.WriteString("  <params>\n")

	var params map[string]interface{}
	_ = json.Unmarshal([]byte(toolCall.Function.Arguments), &params)

	for paramName, paramValue := range params {
		var valueStr string
		switch v := paramValue.(type) {
		case string:
			valueStr = escapeXML(v)
		case []byte:
			valueStr = escapeXML(string(v))
		default:
			valueStr = fmt.Sprintf("%v", paramValue)
		}

		builder.WriteString(fmt.Sprintf("    <%s>%s</%s>\n",
			paramName, valueStr, paramName))
	}

	builder.WriteString("  </params>\n")
	builder.WriteString("</tool>")

	return builder.String()
}

// FormatToolResultPrompt formats a tool result message for the LLM
func FormatToolResultPrompt(result interface{}, err error) string {
	var builder strings.Builder
	builder.WriteString("<tool-result>\n")

	if err != nil {
		builder.WriteString(fmt.Sprintf("  <error>%s</error>\n", escapeXML(err.Error())))
	} else {
		// Format the result based on its type
		switch v := result.(type) {
		case string:
			builder.WriteString(fmt.Sprintf("  <content>%s</content>\n", escapeXML(v)))
		case []byte:
			builder.WriteString(fmt.Sprintf("  <content>%s</content>\n", escapeXML(string(v))))
		case map[string]interface{}:
			// Sort keys for consistent output
			keys := make([]string, 0, len(v))
			for k := range v {
				keys = append(keys, k)
			}
			sort.Strings(keys)

			// Add each field as an XML element
			for _, k := range keys {
				valueStr := fmt.Sprintf("%v", v[k])
				builder.WriteString(fmt.Sprintf("  <%s>%s</%s>\n",
					k, escapeXML(valueStr), k))
			}
		default:
			builder.WriteString(fmt.Sprintf("  <value>%v</value>\n", v))
		}
	}

	builder.WriteString("</tool-result>")
	return builder.String()
}

// Helper function to check if a string slice contains a value
func contains(slice []string, value string) bool {
	for _, item := range slice {
		if item == value {
			return true
		}
	}
	return false
}

// uses the escapeXML function from context.go
