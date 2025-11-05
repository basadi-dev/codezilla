package agent

import (
	"encoding/json"
	"encoding/xml"
	"regexp"
	"strings"
)

// ToolExtractor extracts tool calls from LLM responses
type ToolExtractor struct {
	parsers []ToolCallParser
}

// ToolCallParser interface for different tool call formats
type ToolCallParser interface {
	CanParse(content string) bool
	Extract(content string) ([]ExtractedToolCall, string, error)
}

// ExtractedToolCall represents a tool call with its position
type ExtractedToolCall struct {
	ToolCall      ToolCall
	StartPos      int
	EndPos        int
	RemainingText string
}

// NewToolExtractor creates a new tool extractor with default parsers
func NewToolExtractor() *ToolExtractor {
	return &ToolExtractor{
		parsers: []ToolCallParser{
			&XMLToolCallParser{},
			&JSONToolCallParser{},
			&BashToolCallParser{},
		},
	}
}

// ExtractAll extracts all tool calls from content
func (te *ToolExtractor) ExtractAll(content string) []ToolCall {
	var allToolCalls []ToolCall

	for _, parser := range te.parsers {
		if parser.CanParse(content) {
			extracted, _, err := parser.Extract(content)
			if err == nil && len(extracted) > 0 {
				for _, e := range extracted {
					allToolCalls = append(allToolCalls, e.ToolCall)
				}
				// Found tool calls with this parser, return
				return allToolCalls
			}
		}
	}

	return allToolCalls
}

// XMLToolCallParser parses XML format tool calls
type XMLToolCallParser struct{}

func (p *XMLToolCallParser) CanParse(content string) bool {
	return strings.Contains(content, "<tool>") || strings.Contains(content, "<name>")
}

func (p *XMLToolCallParser) Extract(content string) ([]ExtractedToolCall, string, error) {
	var extracted []ExtractedToolCall

	// Find all <tool>...</tool> blocks
	toolRegex := regexp.MustCompile(`(?s)<tool>(.*?)</tool>`)
	matches := toolRegex.FindAllStringSubmatchIndex(content, -1)

	if len(matches) == 0 {
		return nil, content, nil
	}

	for _, match := range matches {
		startPos := match[0]
		endPos := match[1]
		toolXML := content[match[2]:match[3]]

		var toolData struct {
			Name   string                 `xml:"name"`
			Params map[string]interface{} `xml:"params"`
		}

		// Try to parse as XML
		if err := xml.Unmarshal([]byte("<tool>"+toolXML+"</tool>"), &toolData); err == nil {
			extracted = append(extracted, ExtractedToolCall{
				ToolCall: ToolCall{
					ToolName: toolData.Name,
					Params:   toolData.Params,
				},
				StartPos: startPos,
				EndPos:   endPos,
			})
		} else {
			// Try manual parsing if XML unmarshal fails
			nameRegex := regexp.MustCompile(`<name>(.*?)</name>`)
			nameMatch := nameRegex.FindStringSubmatch(toolXML)
			if len(nameMatch) > 1 {
				toolName := strings.TrimSpace(nameMatch[1])
				params := p.extractParams(toolXML)

				extracted = append(extracted, ExtractedToolCall{
					ToolCall: ToolCall{
						ToolName: toolName,
						Params:   params,
					},
					StartPos: startPos,
					EndPos:   endPos,
				})
			}
		}
	}

	// Calculate remaining text (everything after last tool call)
	remainingText := content
	if len(extracted) > 0 {
		lastEndPos := extracted[len(extracted)-1].EndPos
		if lastEndPos < len(content) {
			remainingText = content[lastEndPos:]
		} else {
			remainingText = ""
		}
	}

	return extracted, remainingText, nil
}

func (p *XMLToolCallParser) extractParams(xmlContent string) map[string]interface{} {
	params := make(map[string]interface{})

	// Find <params>...</params> block
	paramsRegex := regexp.MustCompile(`(?s)<params>(.*?)</params>`)
	paramsMatch := paramsRegex.FindStringSubmatch(xmlContent)

	if len(paramsMatch) > 1 {
		paramsXML := paramsMatch[1]

		// Extract individual parameter tags
		paramRegex := regexp.MustCompile(`<(\w+)>(.*?)</\1>`)
		paramMatches := paramRegex.FindAllStringSubmatch(paramsXML, -1)

		for _, match := range paramMatches {
			if len(match) > 2 {
				key := match[1]
				value := strings.TrimSpace(match[2])
				params[key] = value
			}
		}
	}

	return params
}

// JSONToolCallParser parses JSON format tool calls
type JSONToolCallParser struct{}

func (p *JSONToolCallParser) CanParse(content string) bool {
	return strings.Contains(content, `"tool"`) || strings.Contains(content, `"name"`)
}

func (p *JSONToolCallParser) Extract(content string) ([]ExtractedToolCall, string, error) {
	var extracted []ExtractedToolCall

	// Find JSON objects that look like tool calls
	jsonRegex := regexp.MustCompile(`(?s)\{[^}]*"tool"[^}]*\}`)
	matches := jsonRegex.FindAllStringIndex(content, -1)

	for _, match := range matches {
		startPos := match[0]
		endPos := match[1]
		jsonStr := content[startPos:endPos]

		var toolData struct {
			Tool   string                 `json:"tool"`
			Params map[string]interface{} `json:"params"`
		}

		if err := json.Unmarshal([]byte(jsonStr), &toolData); err == nil {
			extracted = append(extracted, ExtractedToolCall{
				ToolCall: ToolCall{
					ToolName: toolData.Tool,
					Params:   toolData.Params,
				},
				StartPos: startPos,
				EndPos:   endPos,
			})
		}
	}

	remainingText := content
	if len(extracted) > 0 {
		lastEndPos := extracted[len(extracted)-1].EndPos
		if lastEndPos < len(content) {
			remainingText = content[lastEndPos:]
		} else {
			remainingText = ""
		}
	}

	return extracted, remainingText, nil
}

// BashToolCallParser parses bash code block format
type BashToolCallParser struct{}

func (p *BashToolCallParser) CanParse(content string) bool {
	return strings.Contains(content, "```bash") || strings.Contains(content, "```sh")
}

func (p *BashToolCallParser) Extract(content string) ([]ExtractedToolCall, string, error) {
	var extracted []ExtractedToolCall

	// Find bash code blocks
	bashRegex := regexp.MustCompile("(?s)```(?:bash|sh)\\s*\n(.*?)```")
	matches := bashRegex.FindAllStringSubmatchIndex(content, -1)

	for _, match := range matches {
		startPos := match[0]
		endPos := match[1]
		command := strings.TrimSpace(content[match[2]:match[3]])

		extracted = append(extracted, ExtractedToolCall{
			ToolCall: ToolCall{
				ToolName: "execute",
				Params: map[string]interface{}{
					"command": command,
				},
			},
			StartPos: startPos,
			EndPos:   endPos,
		})
	}

	remainingText := content
	if len(extracted) > 0 {
		lastEndPos := extracted[len(extracted)-1].EndPos
		if lastEndPos < len(content) {
			remainingText = content[lastEndPos:]
		} else {
			remainingText = ""
		}
	}

	return extracted, remainingText, nil
}
