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

// ParseLLMResponse scans the raw text from the LLM. If it detects embedded
// tool calls (like markdown JSON/Bash blocks or XML blocks), it strips them
// from the text and converts them into native anyllm.ToolCall structs.
// It returns the cleaned text and the array of tool calls.
func ParseLLMResponse(response string, logger *logger.Logger) (string, []anyllm.ToolCall) {
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

// extractToolCall extracts a single tool call from the response
func extractToolCall(response string, log *logger.Logger) (*anyllm.ToolCall, string, bool) {
	log.Debug("Checking for tool calls in response", "responseLength", len(response))

	jsonPattern := regexp.MustCompile("(?s)```json\\s*\\n(.*?)\\n?```")
	bashPattern := regexp.MustCompile("(?s)```(bash|sh|shell|terminal|console)\\s*\\n(.*?)\\n?```")
	xmlPattern := regexp.MustCompile(`(?s)<tool>[\s\n]*(.*?)[\s\n]*</tool>`)

	type match struct {
		start      int
		end        int
		matchType  string
		submatches []string
	}

	var earliestMatch *match

	if loc := jsonPattern.FindStringSubmatchIndex(response); loc != nil && len(loc) >= 4 {
		earliestMatch = &match{start: loc[0], end: loc[1], matchType: "json", submatches: jsonPattern.FindStringSubmatch(response)}
	}
	if loc := bashPattern.FindStringSubmatchIndex(response); loc != nil && len(loc) >= 6 {
		if earliestMatch == nil || loc[0] < earliestMatch.start {
			earliestMatch = &match{start: loc[0], end: loc[1], matchType: "bash", submatches: bashPattern.FindStringSubmatch(response)}
		}
	}
	if loc := xmlPattern.FindStringSubmatchIndex(response); loc != nil && len(loc) >= 2 {
		if earliestMatch == nil || loc[0] < earliestMatch.start {
			earliestMatch = &match{start: loc[0], end: loc[1], matchType: "xml", submatches: xmlPattern.FindStringSubmatch(response)}
		}
	}

	if earliestMatch == nil {
		return nil, response, false
	}

	remainingText := response[:earliestMatch.start] + response[earliestMatch.end:]
	remainingText = strings.TrimSpace(remainingText)
	id := fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), rand.Intn(1000))

	switch earliestMatch.matchType {
	case "json":
		if len(earliestMatch.submatches) >= 2 {
			jsonContent := strings.TrimSpace(earliestMatch.submatches[1])
			var jsonObj struct {
				Tool   string                 `json:"tool"`
				Name   string                 `json:"name"`
				Params map[string]interface{} `json:"params"`
			}
			if err := json.Unmarshal([]byte(jsonContent), &jsonObj); err == nil {
				name := jsonObj.Tool
				if name == "" {
					name = jsonObj.Name
				}
				if name != "" && jsonObj.Params != nil {
					argsBytes, _ := json.Marshal(jsonObj.Params)
					return &anyllm.ToolCall{
						ID:   id,
						Type: "function",
						Function: providers.FunctionCall{
							Name:      name,
							Arguments: string(argsBytes),
						},
					}, remainingText, true
				}
			}
		}
	case "bash":
		if len(earliestMatch.submatches) >= 3 {
			command := strings.TrimSpace(earliestMatch.submatches[2])
			argsBytes, _ := json.Marshal(map[string]interface{}{"command": command})
			return &anyllm.ToolCall{
				ID:   id,
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "execute",
					Arguments: string(argsBytes),
				},
			}, remainingText, true
		}
	case "xml":
		toolXML := earliestMatch.submatches[0]
		toolXML = strings.TrimPrefix(toolXML, "```xml")
		toolXML = strings.TrimSuffix(toolXML, "```")
		toolXML = strings.TrimSpace(toolXML)

		if !strings.HasPrefix(toolXML, "<") {
			toolXML = "<tool>" + toolXML + "</tool>"
		}

		var xmlToolCall XMLToolCall
		err := xml.Unmarshal([]byte(toolXML), &xmlToolCall)
		if err == nil && xmlToolCall.Name != "" {
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

	return nil, remainingText, true // Match parsed badly, but it was found. Avoid infinite loop
}

func extractXMLParams(paramsXML string, log *logger.Logger) (map[string]interface{}, error) {
	params := make(map[string]interface{})
	namePattern := regexp.MustCompile(`<([a-zA-Z0-9_-]+)[^>]*>`)
	potentialNames := namePattern.FindAllStringSubmatch(paramsXML, -1)
	if len(potentialNames) == 0 {
		return params, nil
	}

	for _, nameMatch := range potentialNames {
		if len(nameMatch) < 2 {
			continue
		}
		paramName := nameMatch[1]
		if _, exists := params[paramName]; exists || paramName == "params" {
			continue
		}
		openTag := regexp.MustCompile(fmt.Sprintf(`<%s[^>]*>`, regexp.QuoteMeta(paramName)))
		closeTag := regexp.MustCompile(fmt.Sprintf(`</%s>`, regexp.QuoteMeta(paramName)))

		openMatches := openTag.FindAllStringIndex(paramsXML, -1)
		closeMatches := closeTag.FindAllStringIndex(paramsXML, -1)

		if len(openMatches) == 0 || len(closeMatches) == 0 {
			continue
		}

		openPos := openMatches[0][1]
		closePos := closeMatches[0][0]

		if openPos >= closePos || openPos >= len(paramsXML) || closePos > len(paramsXML) {
			continue
		}

		paramValue := strings.TrimSpace(paramsXML[openPos:closePos])
		switch {
		case paramValue == "true" || paramValue == "false":
			params[paramName] = paramValue == "true"
		case regexp.MustCompile(`^-?\d+$`).MatchString(paramValue):
			intVal, err := strconv.Atoi(paramValue)
			if err == nil {
				params[paramName] = intVal
			} else {
				params[paramName] = paramValue
			}
		case regexp.MustCompile(`^-?\d+\.\d+$`).MatchString(paramValue):
			floatVal, err := strconv.ParseFloat(paramValue, 64)
			if err == nil {
				params[paramName] = floatVal
			} else {
				params[paramName] = paramValue
			}
		default:
			params[paramName] = strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(paramValue, "&amp;", "&"), "&lt;", "<"), "&gt;", ">"), "&quot;", "\""), "&apos;", "'")
		}
	}
	return params, nil
}
