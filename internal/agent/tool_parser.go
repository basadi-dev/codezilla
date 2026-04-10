package agent

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"codezilla/pkg/logger"
)

// XMLToolCall represents the XML structure for tool calls
type XMLToolCall struct {
	Name   string    `xml:"name"` // Accept standard name tag
	Params XMLParams `xml:"params"`
}

// XMLParams represents the dynamic parameters in XML
type XMLParams struct {
	XMLData []byte `xml:",innerxml"`
}

// toolCallWithRemaining holds a tool call and the remaining text after extraction
type toolCallWithRemaining struct {
	toolCall      *ToolCall
	remainingText string
}

// extractAllToolCalls extracts all tool calls from the response
func (a *agent) extractAllToolCalls(response string) []toolCallWithRemaining {
	var toolCalls []toolCallWithRemaining
	currentText := response

	// Keep extracting tool calls until none are found
	for {
		toolCall, remainingText, found := a.extractToolCall(currentText)
		if !found {
			break
		}
		toolCalls = append(toolCalls, toolCallWithRemaining{
			toolCall:      toolCall,
			remainingText: remainingText,
		})
		currentText = remainingText
	}

	return toolCalls
}

// extractToolCall extracts a single tool call from the response
func (a *agent) extractToolCall(response string) (*ToolCall, string, bool) {
	a.logger.Debug("Checking for tool calls in response", "responseLength", len(response))

	// Define all patterns
	jsonPattern := regexp.MustCompile("(?s)```json\\s*\\n(.*?)\\n?```")
	bashPattern := regexp.MustCompile("(?s)```(bash|sh|shell|terminal|console)\\s*\\n(.*?)\\n?```")
	xmlPattern := regexp.MustCompile(`(?s)<tool>[\s\n]*(.*?)[\s\n]*</tool>`)

	// Find the earliest match among all patterns
	type match struct {
		start      int
		end        int
		matchType  string
		submatches []string
	}

	var earliestMatch *match

	// Check JSON pattern
	if loc := jsonPattern.FindStringSubmatchIndex(response); loc != nil && len(loc) >= 4 {
		earliestMatch = &match{
			start:      loc[0],
			end:        loc[1],
			matchType:  "json",
			submatches: jsonPattern.FindStringSubmatch(response),
		}
	}

	// Check bash pattern
	if loc := bashPattern.FindStringSubmatchIndex(response); loc != nil && len(loc) >= 6 {
		if earliestMatch == nil || loc[0] < earliestMatch.start {
			earliestMatch = &match{
				start:      loc[0],
				end:        loc[1],
				matchType:  "bash",
				submatches: bashPattern.FindStringSubmatch(response),
			}
		}
	}

	// Check XML pattern
	if loc := xmlPattern.FindStringSubmatchIndex(response); loc != nil && len(loc) >= 2 {
		if earliestMatch == nil || loc[0] < earliestMatch.start {
			earliestMatch = &match{
				start:      loc[0],
				end:        loc[1],
				matchType:  "xml",
				submatches: xmlPattern.FindStringSubmatch(response),
			}
		}
	}

	// If no match found, return
	if earliestMatch == nil {
		a.logger.Debug("No tool call patterns found in response")
		return nil, response, false
	}

	// Process the earliest match based on its type
	var result *ToolCall
	remainingText := response[:earliestMatch.start] + response[earliestMatch.end:]
	remainingText = strings.TrimSpace(remainingText)

	switch earliestMatch.matchType {
	case "json":
		if len(earliestMatch.submatches) >= 2 {
			jsonContent := strings.TrimSpace(earliestMatch.submatches[1])
			a.logger.Debug("Found JSON code block", "content", jsonContent)

			// Try to parse as JSON tool call
			var jsonToolCall struct {
				Tool   string                 `json:"tool"`
				Name   string                 `json:"name"` // Alternative field name
				Params map[string]interface{} `json:"params"`
			}

			if err := json.Unmarshal([]byte(jsonContent), &jsonToolCall); err == nil {
				toolName := jsonToolCall.Tool
				if toolName == "" {
					toolName = jsonToolCall.Name
				}

				if toolName != "" && jsonToolCall.Params != nil {
					a.logger.Debug("Successfully parsed JSON tool call", "toolName", toolName)

					result = &ToolCall{
						ToolName: toolName,
						Params:   jsonToolCall.Params,
					}

					return result, remainingText, true
				}
			}
		}

	case "bash":
		if len(earliestMatch.submatches) >= 3 {
			command := strings.TrimSpace(earliestMatch.submatches[2])
			a.logger.Debug("Found bash code block", "language", earliestMatch.submatches[1], "command", command)

			// Create tool call for bash execution
			result = &ToolCall{
				ToolName: "execute",
				Params: map[string]interface{}{
					"command": command,
				},
			}

			return result, remainingText, true
		}

	case "xml":
		// Extract and process the XML tool call
		toolXML := earliestMatch.submatches[0]
		a.logger.Debug("Found potential XML tool call", "xml", toolXML)

		// Continue with XML processing below
	}

	// If we get here and it's not XML, something went wrong
	if earliestMatch.matchType != "xml" {
		return nil, response, false
	}

	// For XML processing, we already have the match from earliestMatch
	var matches []string
	var pattern *regexp.Regexp

	if earliestMatch.matchType == "xml" {
		matches = earliestMatch.submatches
		pattern = xmlPattern
	} else {
		// Should not reach here
		return nil, response, false
	}

	if len(matches) < 2 {
		// Try alternative pattern with backticks that might be used by LLMs
		altPattern := regexp.MustCompile("(?s)```xml[\\s\\n]*(.*?)[\\s\\n]*```")
		matches = altPattern.FindStringSubmatch(response)

		if len(matches) < 2 {
			// If still no matches, look for any <n> tag for backward compatibility
			directToolPattern := regexp.MustCompile(`(?s)<n>\s*(.*?)\s*</n>`)
			matches = directToolPattern.FindStringSubmatch(response)

			if len(matches) >= 2 {
				// Found a direct tool call, wrap it in a tool tag structure
				toolName := matches[1]
				a.logger.Debug("Found direct <n> tag", "toolName", toolName)

				// Find the parameters section
				paramsMatch := regexp.MustCompile(`(?s)<params>(.*?)</params>`).FindStringSubmatch(response)
				if len(paramsMatch) >= 2 {
					// Reconstruct into proper tool XML format
					toolXML := fmt.Sprintf("<n>%s</n>\n<params>%s</params>",
						toolName, paramsMatch[1])
					matches[1] = toolXML
					a.logger.Debug("Reconstructed tool call from direct tag", "toolXML", toolXML)
				} else {
					a.logger.Debug("No params section found for direct tool tag")
					return nil, response, false
				}
			} else {
				a.logger.Debug("No tool call patterns found in response")
				return nil, response, false
			}
		}
	}

	// Extract tool XML content
	toolXML := matches[0]

	// Clean up the XML - remove any leading/trailing backticks or formatting
	toolXML = strings.TrimPrefix(toolXML, "```xml")
	toolXML = strings.TrimSuffix(toolXML, "```")
	toolXML = strings.TrimSpace(toolXML)

	// Ensure XML has root element
	if !strings.HasPrefix(toolXML, "<") {
		toolXML = "<tool>" + toolXML + "</tool>"
	}

	// Try standard XML parsing first
	var xmlToolCall XMLToolCall
	err := xml.Unmarshal([]byte(toolXML), &xmlToolCall)

	a.logger.Debug("Attempting to parse XML", "error", err, "extractedName", xmlToolCall.Name)

	if err == nil && xmlToolCall.Name != "" {
		// Successfully parsed XML
		a.logger.Debug("Successfully parsed tool call with standard XML parser", "toolName", xmlToolCall.Name)

		// Parse the parameters from inner XML
		params, err := parseXMLParams(xmlToolCall.Params.XMLData, a.logger)
		if err != nil {
			a.logger.Error("Failed to parse parameters", "error", err)
			return nil, response, false
		}

		// Create the ToolCall object
		result := &ToolCall{
			ToolName: xmlToolCall.Name,
			Params:   params,
		}

		a.logger.Debug("Successfully extracted tool call",
			"toolName", result.ToolName,
			"paramsCount", len(result.Params))

		return result, remainingText, true
	}

	// If standard parsing failed, try fallback methods
	a.logger.Debug("Standard XML parsing failed, trying fallback methods", "error", err)

	// Try legacy approach
	toolName := extractXMLElement(toolXML, "name")
	if toolName == "" {
		a.logger.Debug("Tool name extracted as empty, this should not happen with improved extraction", "xml", toolXML)

		// Try to handle legacy JSON format for backward compatibility
		if strings.Contains(toolXML, "\"name\"") {
			a.logger.Debug("Detected legacy JSON format, trying to parse as JSON")
			return extractLegacyJSONToolCall(a, toolXML, response, pattern)
		}

		return nil, response, false
	}

	// Extract params section
	paramsSection := extractXMLElement(toolXML, "params")
	if paramsSection == "" {
		a.logger.Error("Tool call missing params section", "xml", toolXML)
		return nil, response, false
	}

	// Parse parameters from params section
	params := extractXMLParams(paramsSection, a.logger)
	if params == nil {
		a.logger.Error("Failed to extract parameters from tool call", "xml", toolXML)
		return nil, response, false
	}

	// Create the ToolCall object
	result = &ToolCall{
		ToolName: toolName,
		Params:   params,
	}

	a.logger.Debug("Successfully extracted tool call using fallback method",
		"toolName", result.ToolName,
		"paramsCount", len(result.Params))

	return result, remainingText, true
}

// extractXMLElement extracts a specific element from an XML string
func extractXMLElement(xmlStr string, elementName string) string {
	// First try the requested element name
	result := tryExtractXMLElement(xmlStr, elementName)

	// If we're looking for "name" and didn't find it, try "n" as an alternative
	if result == "" && elementName == "name" {
		result = tryExtractXMLElement(xmlStr, "n")
	}

	return result
}

// tryExtractXMLElement attempts to extract a specific XML element by name
func tryExtractXMLElement(xmlStr string, elementName string) string {
	// Create patterns for opening and closing tags
	openTag := regexp.MustCompile(fmt.Sprintf(`<%s[^>]*>`, regexp.QuoteMeta(elementName)))
	closeTag := regexp.MustCompile(fmt.Sprintf(`</%s>`, regexp.QuoteMeta(elementName)))

	// First check if it's a self-closing tag
	selfClosingPattern := regexp.MustCompile(fmt.Sprintf(`<%s[^>]*/?>`, regexp.QuoteMeta(elementName)))
	if selfClosingPattern.MatchString(xmlStr) {
		return "" // Self-closing tag has no content
	}

	// Find positions of opening and closing tags
	openMatches := openTag.FindStringIndex(xmlStr)
	closeMatches := closeTag.FindStringIndex(xmlStr)

	// Check if we found both tags
	if openMatches == nil || closeMatches == nil {
		return "" // Element not found
	}

	// Extract content between tags
	openEnd := openMatches[1]     // End position of opening tag
	closeStart := closeMatches[0] // Start position of closing tag

	// Validate positions
	if openEnd >= closeStart || openEnd >= len(xmlStr) || closeStart > len(xmlStr) {
		return "" // Invalid positions
	}

	// Return the content between opening and closing tags, trimmed
	return strings.TrimSpace(xmlStr[openEnd:closeStart])
}

// extractXMLParams is the fallback method for parsing parameters when standard XML parsing fails
func extractXMLParams(paramsXML string, logger *logger.Logger) map[string]interface{} {
	params := make(map[string]interface{})

	// Simple name pattern for XML element names
	namePattern := regexp.MustCompile(`<([a-zA-Z0-9_-]+)[^>]*>`)

	// Parse parameters using a more robust approach without backreferences
	// Get all potential parameter names first
	potentialNames := namePattern.FindAllStringSubmatch(paramsXML, -1)
	if len(potentialNames) == 0 {
		return nil
	}

	// Process each potential parameter
	for _, nameMatch := range potentialNames {
		if len(nameMatch) < 2 {
			continue
		}

		paramName := nameMatch[1]

		// Skip if this is not a direct child element of params (could be nested)
		// or if we already processed this parameter
		if _, exists := params[paramName]; exists || paramName == "params" {
			continue
		}

		// Create patterns specific to this parameter name
		openTag := regexp.MustCompile(fmt.Sprintf(`<%s[^>]*>`, regexp.QuoteMeta(paramName)))
		closeTag := regexp.MustCompile(fmt.Sprintf(`</%s>`, regexp.QuoteMeta(paramName)))

		// Find the positions of opening and closing tags
		openMatches := openTag.FindAllStringIndex(paramsXML, -1)
		closeMatches := closeTag.FindAllStringIndex(paramsXML, -1)

		// Skip if we can't find a matching pair
		if len(openMatches) == 0 || len(closeMatches) == 0 {
			continue
		}

		// Take the first occurrence for simplicity
		openPos := openMatches[0][1]   // End position of opening tag
		closePos := closeMatches[0][0] // Start position of closing tag

		// Check if we have valid positions for extraction
		if openPos >= closePos || openPos >= len(paramsXML) || closePos > len(paramsXML) {
			continue
		}

		// Extract the parameter value
		paramValue := strings.TrimSpace(paramsXML[openPos:closePos])

		// Try to convert to appropriate types (boolean, number, etc.)
		switch {
		case paramValue == "true" || paramValue == "false":
			// Boolean
			params[paramName] = paramValue == "true"
			logger.Debug("Parsed XML parameter as boolean", "name", paramName, "value", params[paramName])

		case regexp.MustCompile(`^-?\d+$`).MatchString(paramValue):
			// Integer
			intVal, err := strconv.Atoi(paramValue)
			if err == nil {
				params[paramName] = intVal
				logger.Debug("Parsed XML parameter as integer", "name", paramName, "value", params[paramName])
			} else {
				params[paramName] = paramValue
				logger.Debug("Failed to parse numeric value, using as string", "name", paramName, "value", paramValue)
			}

		case regexp.MustCompile(`^-?\d+\.\d+$`).MatchString(paramValue):
			// Float
			floatVal, err := strconv.ParseFloat(paramValue, 64)
			if err == nil {
				params[paramName] = floatVal
				logger.Debug("Parsed XML parameter as float", "name", paramName, "value", params[paramName])
			} else {
				params[paramName] = paramValue
				logger.Debug("Failed to parse float value, using as string", "name", paramName, "value", paramValue)
			}

		default:
			// String
			params[paramName] = paramValue
			logger.Debug("Parsed XML parameter as string", "name", paramName, "value", paramValue)
		}
	}

	return params
}

// agentFormatXMLValue formats a value for inclusion in XML
func agentFormatXMLValue(value interface{}) string {
	switch v := value.(type) {
	case string:
		return agentEscapeXML(v)
	case []interface{}:
		var builder strings.Builder
		builder.WriteString("\n")
		for i, item := range v {
			builder.WriteString(fmt.Sprintf("    <item index=\"%d\">%v</item>\n", i, agentFormatXMLValue(item)))
		}
		builder.WriteString("  ")
		return builder.String()
	case map[string]interface{}:
		var builder strings.Builder
		builder.WriteString("\n")
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			builder.WriteString(fmt.Sprintf("    <%s>%v</%s>\n", k, agentFormatXMLValue(v[k]), k))
		}
		builder.WriteString("  ")
		return builder.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

// extractLegacyJSONToolCall handles legacy JSON format for backward compatibility
func extractLegacyJSONToolCall(a *agent, toolJSON string, response string, pattern *regexp.Regexp) (*ToolCall, string, bool) {
	a.logger.Debug("Attempting to parse legacy JSON tool call", "json", toolJSON)

	var toolCall struct {
		Name   string                 `json:"name"`
		Params map[string]interface{} `json:"params"`
	}

	err := json.Unmarshal([]byte(toolJSON), &toolCall)
	if err != nil {
		// Try with preprocessing
		toolJSON = strings.ReplaceAll(toolJSON, "\n", "")
		toolJSON = strings.ReplaceAll(toolJSON, "\r", "")

		err = json.Unmarshal([]byte(toolJSON), &toolCall)
		if err != nil {
			a.logger.Error("Failed to parse legacy JSON after cleaning", "error", err)
			return nil, response, false
		}
	}

	if toolCall.Name == "" {
		a.logger.Error("Legacy JSON tool call missing name field", "json", toolJSON)
		return nil, response, false
	}

	if toolCall.Params == nil {
		a.logger.Error("Legacy JSON tool call missing params field", "json", toolJSON)
		return nil, response, false
	}

	result := &ToolCall{
		ToolName: toolCall.Name,
		Params:   toolCall.Params,
	}

	a.logger.Debug("Successfully extracted legacy JSON tool call",
		"toolName", result.ToolName,
		"paramsCount", len(result.Params))

	remainingText := pattern.ReplaceAllString(response, "")
	remainingText = strings.TrimSpace(remainingText)

	return result, remainingText, true
}

// agentEscapeXML escapes XML special characters
func agentEscapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	s = strings.ReplaceAll(s, "'", "&apos;")
	return s
}
