package skills

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// Skill represents a loaded skill with its metadata and instructions.
type Skill struct {
	// Name is the unique identifier (from front-matter or filename stem).
	Name string
	// Trigger is an optional slash command that auto-activates this skill (e.g. "/caveman").
	Trigger string
	// Description is a short human-readable summary of what the skill does.
	Description string
	// AlwaysOn means the skill should be auto-activated at startup.
	AlwaysOn bool
	// Instructions is the markdown body injected into the system prompt.
	Instructions string
	// FilePath is the source file this skill was loaded from.
	FilePath string
}

// LoadSkill parses a skill markdown file with YAML-like front-matter.
//
// File format:
//
//	---
//	name: caveman
//	trigger: /caveman
//	description: Ultra-compressed comms mode
//	always_on: false
//	---
//	Instructions body here...
func LoadSkill(path string) (*Skill, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open skill file: %w", err)
	}
	defer f.Close()

	skill := &Skill{FilePath: path}
	scanner := bufio.NewScanner(f)

	// Expect opening "---"
	if !scanner.Scan() {
		return nil, fmt.Errorf("skill file %q is empty", path)
	}
	firstLine := strings.TrimSpace(scanner.Text())
	if firstLine != "---" {
		// No front-matter; treat entire file as instructions, derive name from filename.
		skill.Name = stemFromPath(path)
		var bodyLines []string
		bodyLines = append(bodyLines, firstLine)
		for scanner.Scan() {
			bodyLines = append(bodyLines, scanner.Text())
		}
		skill.Instructions = strings.TrimSpace(strings.Join(bodyLines, "\n"))
		return skill, nil
	}

	// Parse front-matter until closing "---"
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "---" {
			break
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		switch key {
		case "name":
			skill.Name = value
		case "trigger":
			skill.Trigger = value
		case "description":
			skill.Description = value
		case "always_on":
			skill.AlwaysOn = value == "true"
		}
	}

	if skill.Name == "" {
		skill.Name = stemFromPath(path)
	}

	// Rest of the file is the instruction body.
	var bodyLines []string
	for scanner.Scan() {
		bodyLines = append(bodyLines, scanner.Text())
	}
	skill.Instructions = strings.TrimSpace(strings.Join(bodyLines, "\n"))

	return skill, nil
}

// stemFromPath returns the filename without directory and extension.
func stemFromPath(path string) string {
	// Strip directory
	name := path
	if idx := strings.LastIndexAny(path, "/\\"); idx >= 0 {
		name = path[idx+1:]
	}
	// Strip extension
	if idx := strings.LastIndex(name, "."); idx >= 0 {
		name = name[:idx]
	}
	return name
}
