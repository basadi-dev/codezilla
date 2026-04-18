package agent

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"math/rand"
	"regexp"
	"strconv"
	"strings"
	"time"

	"codezilla/pkg/logger"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// XMLToolCall represents the XML structure for tool calls
type XMLToolCall struct {
	Name   string    `xml:"name"`
	Params XMLParams `xml:"params"`
}

type XMLParams struct {
	XMLData []byte `xml:",innerxml"`
}

// specialTokenPatterns matches leaked model-specific formatting tokens that
// should never appear in user-visible output. These are stripped before any
// further processing.
var specialTokenPatterns = []*regexp.Regexp{
	// Chat template tokens (Llama, Qwen, Mistral variants)
	regexp.MustCompile(`<\|?im_start\|?>[a-zA-Z]*\n?`),
	regexp.MustCompile(`<\|?im_end\|?>\n?`),
	regexp.MustCompile(`<\|message\|>`),
	regexp.MustCompile(`<\|?tool_call\|?>`),
	regexp.MustCompile(`<\|?tool_call_end\|?>`),
	regexp.MustCompile(`<\|end_of_turn\|>`),
	regexp.MustCompile(`<\|eot_id\|>`),
	regexp.MustCompile(`<\|start_header_id\|>[a-z]*<\|end_header_id\|>\n?`),
	regexp.MustCompile(`<\|python_tag\|>`),
	// Closing analysis/reasoning tags that leaked out
	regexp.MustCompile(`</?(analysis|reasoning|thought)[^>]*>`),
	// Function call markers (various model families)
	regexp.MustCompile(`</?function_calls>\n?`),
	regexp.MustCompile(`</?invoke>\n?`),
	// GPT-OSS/custom model tokens
	regexp.MustCompile(`to=functions\.[a-zA-Z0-9_]+\s*`),
}

// stripSpecialTokens strips tokens but preserves leading/trailing whitespace (used for live streaming UI)
func stripSpecialTokens(text string) string {
	for _, pattern := range specialTokenPatterns {
		text = pattern.ReplaceAllString(text, "")
	}
	return text
}

// SanitiseSpecialTokens strips leaked model-internal formatting tokens from
// text before it is shown to the user or processed further.
func SanitiseSpecialTokens(text string) string {
	return strings.TrimSpace(stripSpecialTokens(text))
}

// looksLikeLeakedToolCall checks if text contains suspicious patterns that
// suggest the model tried to call a tool but leaked the syntax as plain text.
// Returns true only when there is strong evidence of a leaked call (not just
// explanatory JSON shown to the user).
func looksLikeLeakedToolCall(text string) bool {
	// Must contain something JSON-like
	if !strings.Contains(text, "{") {
		return false
	}

	// Strong indicators: model-specific special tokens in the text
	for _, p := range specialTokenPatterns {
		if p.MatchString(text) {
			return true
		}
	}

	// Strong indicator: text is almost entirely a JSON object with tool-like keys
	// and has very little surrounding prose
	trimmed := strings.TrimSpace(text)

	// Strip any leading/trailing tags before checking for bare JSON
	tagStripped := regexp.MustCompile(`^[^{]*`).ReplaceAllString(trimmed, "")
	if strings.HasPrefix(tagStripped, "{") {
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(tagStripped), &obj); err == nil {
			// JSON decoded cleanly — check if it has known tool parameter keys
			toolParamKeys := []string{"action", "path", "file_path", "command", "query", "url", "task", "content"}
			matchCount := 0
			for _, k := range toolParamKeys {
				if _, ok := obj[k]; ok {
					matchCount++
				}
			}
			// Also check for explicit tool/name/function keys
			_, hasName := obj["name"]
			_, hasTool := obj["tool"]
			_, hasFunction := obj["function"]
			if hasName || hasTool || hasFunction || matchCount >= 2 {
				// Make sure the surrounding text is minimal (not embedded in prose)
				nonJSONLen := len(trimmed) - len(tagStripped)
				return nonJSONLen < 80 // less than 80 chars of prose around the JSON
			}
		}
	}

	// Known leak patterns from real models in the wild
	leakPatterns := []*regexp.Regexp{
		regexp.MustCompile(`to=functions\.[a-zA-Z0-9_]+`),
		regexp.MustCompile(`<tool_call>\s*\{`),
		regexp.MustCompile(`<\|tool_call\|>`),
		regexp.MustCompile(`"action"\s*:\s*"(read|write|list|execute|search)"`),
	}
	for _, p := range leakPatterns {
		if p.MatchString(text) {
			return true
		}
	}

	return false
}

// ParseLLMResponse scans the raw text from the LLM. If it detects embedded
// tool calls (like markdown JSON/Bash blocks, XML blocks, or leaked model
// special-token syntax), it strips them from the text and converts them into
// native anyllm.ToolCall structs.
// It returns the cleaned text and the array of tool calls.
func ParseLLMResponse(response string, logger *logger.Logger) (string, []anyllm.ToolCall) {
	// Layer 1: sanitise special tokens before any further processing
	response = SanitiseSpecialTokens(response)

	var toolCalls []anyllm.ToolCall
	currentText := response

	// Keep extracting tool calls until none are found
	for {
		toolCall, remainingText, found := extractToolCall(currentText, logger)
		if !found {
			break
		}
		if toolCall != nil {
			toolCalls = append(toolCalls, *toolCall)
		}
		currentText = remainingText
	}

	return strings.TrimSpace(currentText), toolCalls
}

// extractToolCall attempts to extract a single tool call from the response.
// It tries multiple patterns in order of specificity.
func extractToolCall(response string, log *logger.Logger) (*anyllm.ToolCall, string, bool) {
	log.Debug("Checking for tool calls in response", "responseLength", len(response))

	// Pattern registry: ordered from most-specific to least-specific
	// to avoid false positives.
	type pattern struct {
		name  string
		regex *regexp.Regexp
	}

	patterns := []pattern{
		// Explicit tool_call wrapper tags (Llama 3.x, Qwen)
		{name: "tool_call_tag", regex: regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)},
		// Anthropic-style <invoke> XML blocks
		{name: "invoke_xml", regex: regexp.MustCompile(`(?s)<invoke>(.*?)</invoke>`)},
		// Standard markdown JSON block
		{name: "json_block", regex: regexp.MustCompile("(?s)```json\\s*\\n(.*?)\\n?```")},
		// Generic <tool> XML block
		{name: "tool_xml", regex: regexp.MustCompile(`(?s)<tool>[\s\n]*(.*?)[\s\n]*</tool>`)},
		// Bare JSON object at start of response (leaked raw call)
		{name: "bare_json", regex: regexp.MustCompile(`(?s)^\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\})`)},
		// to=functions.X{...} leak pattern (GPT-OSS, some Ollama models)
		{name: "functions_leak", regex: regexp.MustCompile(`(?s)to=functions\.([a-zA-Z0-9_]+)\s*(\{.*?\})`)},
	}

	type match struct {
		start      int
		end        int
		patternIdx int
		submatches []string
	}

	var earliest *match

	for i, p := range patterns {
		loc := p.regex.FindStringSubmatchIndex(response)
		if len(loc) < 2 {
			continue
		}
		if earliest == nil || loc[0] < earliest.start {
			earliest = &match{
				start:      loc[0],
				end:        loc[1],
				patternIdx: i,
				submatches: p.regex.FindStringSubmatch(response),
			}
		}
	}

	if earliest == nil {
		return nil, response, false
	}

	remainingText := strings.TrimSpace(response[:earliest.start] + response[earliest.end:])
	id := fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), rand.Intn(1000)) //nolint:gosec // weak rng is fine for tool call IDs

	patternName := patterns[earliest.patternIdx].name
	log.Debug("Tool call pattern matched", "pattern", patternName, "start", earliest.start)

	switch patternName {
	case "tool_call_tag":
		// <tool_call>{"name":"X","arguments":{...}}</tool_call>
		if len(earliest.submatches) >= 2 {
			tc := tryParseToolCallJSON(earliest.submatches[1], id, log)
			if tc != nil {
				return tc, remainingText, true
			}
		}

	case "json_block":
		// ```json\n{...}\n```
		if len(earliest.submatches) >= 2 {
			tc := tryParseToolCallJSON(earliest.submatches[1], id, log)
			if tc != nil {
				return tc, remainingText, true
			}
		}

	case "bare_json":
		// Raw JSON object at start of text, only if it looks like a tool call
		if len(earliest.submatches) >= 2 {
			jsonStr := strings.TrimSpace(earliest.submatches[1])
			tc := tryParseToolCallJSON(jsonStr, id, log)
			if tc != nil {
				return tc, remainingText, true
			}
		}

	case "functions_leak":
		// to=functions.fileManage{"action":"read",...}
		if len(earliest.submatches) >= 3 {
			toolName := earliest.submatches[1]
			argsStr := strings.TrimSpace(earliest.submatches[2])
			var params map[string]interface{}
			if err := json.Unmarshal([]byte(argsStr), &params); err == nil {
				argsBytes, _ := json.Marshal(params)
				return &anyllm.ToolCall{
					ID:   id,
					Type: "function",
					Function: providers.FunctionCall{
						Name:      toolName,
						Arguments: string(argsBytes),
					},
				}, remainingText, true
			}
		}

	case "invoke_xml":
		// <invoke><tool_name>X</tool_name><parameters>...</parameters></invoke>
		if len(earliest.submatches) >= 2 {
			inner := "<invoke>" + earliest.submatches[1] + "</invoke>"
			var invokeXML struct {
				ToolName   string `xml:"tool_name"`
				Parameters struct {
					Inner []byte `xml:",innerxml"`
				} `xml:"parameters"`
			}
			if err := xml.Unmarshal([]byte(inner), &invokeXML); err == nil && invokeXML.ToolName != "" {
				params, _ := extractXMLParams(string(invokeXML.Parameters.Inner), log)
				argsBytes, _ := json.Marshal(params)
				return &anyllm.ToolCall{
					ID:   id,
					Type: "function",
					Function: providers.FunctionCall{
						Name:      invokeXML.ToolName,
						Arguments: string(argsBytes),
					},
				}, remainingText, true
			}
		}

	case "tool_xml":
		// <tool>...</tool>
		toolXML := earliest.submatches[0]
		toolXML = strings.TrimPrefix(toolXML, "```xml")
		toolXML = strings.TrimSuffix(toolXML, "```")
		toolXML = strings.TrimSpace(toolXML)

		if !strings.HasPrefix(toolXML, "<") {
			toolXML = "<tool>" + toolXML + "</tool>"
		}

		var xmlToolCall XMLToolCall
		if err := xml.Unmarshal([]byte(toolXML), &xmlToolCall); err == nil && xmlToolCall.Name != "" {
			params, _ := extractXMLParams(string(xmlToolCall.Params.XMLData), log)
			argsBytes, _ := json.Marshal(params)
			return &anyllm.ToolCall{
				ID:   id,
				Type: "function",
				Function: providers.FunctionCall{
					Name:      xmlToolCall.Name,
					Arguments: string(argsBytes),
				},
			}, remainingText, true
		}
	}

	return nil, remainingText, true // Match parsed badly but was found — avoid infinite loop
}

// tryParseToolCallJSON attempts to parse a JSON string as a tool call.
// It handles multiple JSON schemas used by different model families.
func tryParseToolCallJSON(jsonStr string, id string, log *logger.Logger) *anyllm.ToolCall {
	jsonStr = strings.TrimSpace(jsonStr)

	// Schema 1: {"name":"toolName","arguments":{...}} — OpenAI native format
	var schema1 struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(jsonStr), &schema1); err == nil && schema1.Name != "" && schema1.Arguments != nil {
		argsBytes, _ := json.Marshal(schema1.Arguments)
		return &anyllm.ToolCall{
			ID:   id,
			Type: "function",
			Function: providers.FunctionCall{
				Name:      schema1.Name,
				Arguments: string(argsBytes),
			},
		}
	}

	// Schema 2: {"tool":"toolName","params":{...}} or {"name":"toolName","params":{...}}
	var schema2 struct {
		Tool   string                 `json:"tool"`
		Name   string                 `json:"name"`
		Params map[string]interface{} `json:"params"`
	}
	if err := json.Unmarshal([]byte(jsonStr), &schema2); err == nil {
		name := schema2.Tool
		if name == "" {
			name = schema2.Name
		}
		if name != "" && schema2.Params != nil {
			argsBytes, _ := json.Marshal(schema2.Params)
			return &anyllm.ToolCall{
				ID:   id,
				Type: "function",
				Function: providers.FunctionCall{
					Name:      name,
					Arguments: string(argsBytes),
				},
			}
		}
	}

	// Schema 3: {"action":"read","path":"..."} — raw params with action as tool discriminator
	// This is the leaked format seen from gpt-oss and some Ollama models
	var rawParams map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &rawParams); err == nil {
		// Try to infer the tool name from known discriminator keys
		toolName := inferToolNameFromParams(rawParams)
		if toolName != "" {
			argsBytes, _ := json.Marshal(rawParams)
			log.Info("Tool call recovered from raw params JSON", "inferred_tool", toolName)
			return &anyllm.ToolCall{
				ID:   id,
				Type: "function",
				Function: providers.FunctionCall{
					Name:      toolName,
					Arguments: string(argsBytes),
				},
			}
		}
	}

	return nil
}

// inferToolNameFromParams tries to determine the tool name from raw parameter
// objects where the tool name wasn't explicitly included. This handles the
// common case where local models leak {"action":"read","path":"..."} without
// wrapping it in a proper tool call structure.
func inferToolNameFromParams(params map[string]interface{}) string {
	action, _ := params["action"].(string)
	_, hasPath := params["path"]
	_, hasFilePath := params["file_path"]
	_, hasCommand := params["command"]
	_, hasQuery := params["query"]
	_, hasURL := params["url"]
	_, hasTask := params["task"]
	_, hasContent := params["content"]
	_, hasDir := params["dir"]
	_, hasDirectory := params["directory"]
	_, hasTargetContent := params["target_content"]
	_, hasReplacementContent := params["replacement_content"]
	_, hasReplacements := params["replacements"]

	// fileManage: has "action" + "path"
	if action != "" && (hasPath || hasFilePath || hasDir || hasDirectory) {
		switch action {
		case "read", "write", "delete", "copy", "move", "list", "mkdir":
			return "fileManage"
		}
	}

	// execute: has "command"
	if hasCommand {
		return "execute"
	}

	// webSearch: has "query" but not file-related keys
	if hasQuery && !hasPath && !hasFilePath {
		return "webSearch"
	}

	// fetchURL: has "url"
	if hasURL {
		return "fetchURL"
	}

	// subAgent: has "task"
	if hasTask {
		return "subAgent"
	}

	// IMPORTANT: fileEdit/multiReplace must be checked BEFORE fileWrite.
	// Both have file_path, but their distinguishing keys are target_content
	// and replacement_content. Without this check, edit calls were previously
	// silently routed to fileWrite, causing destructive full-file overwrites.
	if hasFilePath && hasTargetContent && hasReplacementContent {
		return "fileEdit"
	}
	// multiReplace: has "file_path" and "replacements" array
	if hasFilePath && hasReplacements {
		return "multiReplace"
	}

	// fileWrite: has "file_path" and "content" (full overwrite, intentional)
	if hasFilePath && hasContent {
		return "fileWrite"
	}
	if hasFilePath {
		return "fileRead"
	}

	return ""
}

// extractXMLParams extracts tool parameters from the raw inner XML of a <params> block.
//
// Design note: We intentionally do NOT use encoding/xml.Decoder here. The standard
// XML decoder requires well-formed XML, but code passed as parameter values often
// contains unescaped angle brackets (Go generics, HTML templates, comparison operators).
// A robust raw-string extraction strategy is used instead:
//
//  1. Discover top-level parameter tag names via regex (tag names only, not content).
//  2. For each parameter, find the FIRST opening tag position and the LAST closing tag
//     position. Using LastIndex for the close tag means content that contains
//     a partial closing tag (e.g. a comment like "// </foo>") won't truncate the value.
//  3. Extract the raw string between those positions and unescape XML entities.
//
// This correctly handles multi-line code blocks, Go generics, HTML inside code, etc.
func extractXMLParams(paramsXML string, log *logger.Logger) (map[string]interface{}, error) {
	params := make(map[string]interface{})
	namePattern := regexp.MustCompile(`<([a-zA-Z0-9_-]+)[^>]*>`)
	potentialNames := namePattern.FindAllStringSubmatch(paramsXML, -1)
	if len(potentialNames) == 0 {
		return params, nil
	}

	seen := make(map[string]bool)
	for _, nameMatch := range potentialNames {
		if len(nameMatch) < 2 {
			continue
		}
		paramName := nameMatch[1]
		if seen[paramName] || paramName == "params" {
			continue
		}
		seen[paramName] = true

		// Find where the opening tag ends (content starts)
		openTagPattern := regexp.MustCompile(fmt.Sprintf(`<%s[^>]*>`, regexp.QuoteMeta(paramName)))
		openMatch := openTagPattern.FindStringIndex(paramsXML)
		if openMatch == nil {
			continue
		}
		contentStart := openMatch[1] // byte position after the '>' of the opening tag

		// Use LastIndex for the closing tag so that code containing a partial close
		// tag (e.g. inside a comment) doesn't prematurely terminate extraction.
		closeTag := "</" + paramName + ">"
		remaining := paramsXML[contentStart:]
		closeIdx := strings.LastIndex(remaining, closeTag)
		if closeIdx == -1 {
			continue
		}

		paramValue := remaining[:closeIdx]

		// Unescape standard XML entities. Code content that contains literal angle
		// brackets should have been entity-encoded by the model (e.g. &lt; for <).
		paramValue = unescapeXMLEntities(paramValue)

		// For non-code scalar values, also trim surrounding whitespace.
		// We do NOT trim for long values (code blocks) to preserve indentation.
		trimmed := strings.TrimSpace(paramValue)

		switch {
		case trimmed == "true" || trimmed == "false":
			params[paramName] = trimmed == "true"
		case regexp.MustCompile(`^-?\d+$`).MatchString(trimmed):
			intVal, err := strconv.Atoi(trimmed)
			if err == nil {
				params[paramName] = intVal
			} else {
				params[paramName] = paramValue
			}
		case regexp.MustCompile(`^-?\d+\.\d+$`).MatchString(trimmed):
			floatVal, err := strconv.ParseFloat(trimmed, 64)
			if err == nil {
				params[paramName] = floatVal
			} else {
				params[paramName] = paramValue
			}
		default:
			// Preserve the raw (entity-unescaped) value. For code blocks this retains
			// all original whitespace and indentation.
			params[paramName] = paramValue
		}
	}
	return params, nil
}

// unescapeXMLEntities replaces the five standard XML character entities with
// their literal equivalents. Called after raw-string extraction from XML.
func unescapeXMLEntities(s string) string {
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&apos;", "'")
	return s
}
