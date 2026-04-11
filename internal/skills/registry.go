package skills

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Registry holds all discovered skills and tracks which are active.
type Registry struct {
	mu     sync.RWMutex
	skills map[string]*Skill // keyed by name
	active map[string]bool   // set of active skill names
}

// NewRegistry creates an empty skill registry.
func NewRegistry() *Registry {
	return &Registry{
		skills: make(map[string]*Skill),
		active: make(map[string]bool),
	}
}

// LoadFromDir scans a directory for *.md files and loads each as a skill.
// Non-fatal errors (e.g. a single bad file) are collected and returned together.
func (r *Registry) LoadFromDir(dir string) error {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		// Skills directory optional — not an error.
		return nil
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("read skills dir %q: %w", dir, err)
	}

	var errs []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(entry.Name(), ".md") {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		skill, err := LoadSkill(path)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", entry.Name(), err))
			continue
		}
		r.mu.Lock()
		r.skills[skill.Name] = skill
		r.mu.Unlock()
	}

	if len(errs) > 0 {
		return fmt.Errorf("skill load errors: %s", strings.Join(errs, "; "))
	}
	return nil
}

// Activate marks a skill as active by name. Returns error if not found.
func (r *Registry) Activate(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.skills[name]; !ok {
		return fmt.Errorf("skill %q not found", name)
	}
	r.active[name] = true
	return nil
}

// Deactivate removes a skill from the active set.
func (r *Registry) Deactivate(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.active, name)
}

// DeactivateAll clears all active skills.
func (r *Registry) DeactivateAll() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.active = make(map[string]bool)
}

// IsActive returns whether the given skill is currently active.
func (r *Registry) IsActive(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.active[name]
}

// ActiveInstructions returns the concatenated instructions of all active skills,
// ready to be appended to the system prompt.
func (r *Registry) ActiveInstructions() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if len(r.active) == 0 {
		return ""
	}
	var parts []string
	for name := range r.active {
		skill, ok := r.skills[name]
		if !ok || skill.Instructions == "" {
			continue
		}
		parts = append(parts, fmt.Sprintf("## Skill: %s\n%s", skill.Name, skill.Instructions))
	}
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

// FindByTrigger returns the skill whose trigger matches the given slash command, or nil.
func (r *Registry) FindByTrigger(trigger string) *Skill {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, skill := range r.skills {
		if skill.Trigger == trigger {
			return skill
		}
	}
	return nil
}

// Get returns a skill by name, or nil if not found.
func (r *Registry) Get(name string) *Skill {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.skills[name]
}

// List returns all known skills (not just active ones).
func (r *Registry) List() []*Skill {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Skill, 0, len(r.skills))
	for _, s := range r.skills {
		out = append(out, s)
	}
	return out
}

// AutoActivate activates every skill whose AlwaysOn flag is true,
// plus any skills whose names are in the provided list.
func (r *Registry) AutoActivate(names []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	// AlwaysOn skills
	for _, s := range r.skills {
		if s.AlwaysOn {
			r.active[s.Name] = true
		}
	}
	// Config-specified skills
	for _, name := range names {
		if _, ok := r.skills[name]; ok {
			r.active[name] = true
		}
	}
}
