package agent

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ProjectType represents the detected project build system.
type ProjectType string

const (
	ProjectGo      ProjectType = "go"
	ProjectNode    ProjectType = "node"
	ProjectPython  ProjectType = "python"
	ProjectRust    ProjectType = "rust"
	ProjectC       ProjectType = "c"
	ProjectUnknown ProjectType = "unknown"
)

const (
	// maxVerifyRetries is the number of times the agent will retry after a failed
	// verification before giving up and returning the response as-is.
	maxVerifyRetries = 2

	// verifyTimeout is the maximum time allowed for all verification commands combined.
	verifyTimeout = 30 * time.Second
)

// VerifyResult contains the outcome of running verification commands.
type VerifyResult struct {
	Passed   bool
	Errors   []string // one entry per failing command
	Commands []string // the commands that were executed
	Duration time.Duration
}

// fileModifyingTools is the set of tool names that write or delete files.
// fileManage is inspected action-by-action in isFileModifyingTool below.
var fileModifyingTools = map[string]bool{
	"multiReplace": true,
}

// isFileModifyingTool returns true if the tool call will write or delete files.
func isFileModifyingTool(name string, params map[string]interface{}) bool {
	if name == "fileManage" {
		action, _ := params["action"].(string)
		// write is guarded for existing files; delete is always destructive.
		// read/list/mkdir/move do not touch file content.
		return action == "write" || action == "delete"
	}
	return fileModifyingTools[name]
}

// builtinVerifyMarkers maps marker filenames to built-in default verification
// commands for that project type. Commands are ordered most-specific → least-specific
// so a project with multiple markers (e.g. Cargo.toml + Makefile) hits the right one.
//
// Design rules for every entry:
//   - Fast: compile/type-check only — never run the full test suite.
//   - Non-destructive: no writes, no installs, no side-effects on a clean checkout.
//   - Fail-safe: prefer commands that exit 0 on "not applicable" (|| true guards).
//   - Informative: output must be meaningful when it fails so the LLM can self-correct.
//
// User-supplied verify_profiles in codezilla.yaml take precedence over these.
var builtinVerifyMarkers = []struct {
	marker   string
	commands []string
}{
	// ── Go ────────────────────────────────────────────────────────────────────
	{"go.mod", []string{
		"go build ./...",
		"go vet ./...",
	}},

	// ── Rust ──────────────────────────────────────────────────────────────────
	{"Cargo.toml", []string{
		"cargo check --all-targets 2>&1",
	}},

	// ── JavaScript / TypeScript (Node) ────────────────────────────────────────
	// tsc is preferred; fall through to eslint if tsconfig absent.
	{"tsconfig.json", []string{
		"npx --no-install tsc --noEmit",
	}},
	{"package.json", []string{
		// Type-check if tsconfig present, otherwise do a quick eslint dry-run.
		"[ -f tsconfig.json ] && npx --no-install tsc --noEmit || npx --no-install eslint . --max-warnings=0 2>/dev/null || true",
	}},

	// ── Python ────────────────────────────────────────────────────────────────
	// pyproject.toml may use ruff or mypy; fall back to py_compile.
	{"pyproject.toml", []string{
		"ruff check . 2>/dev/null || python -m py_compile **/*.py 2>/dev/null || true",
	}},
	{"setup.py", []string{
		"python -m py_compile setup.py && find . -name '*.py' -not -path './.git/*' | head -50 | xargs python -m py_compile 2>&1",
	}},
	{"requirements.txt", []string{
		"find . -name '*.py' -not -path './.git/*' | head -50 | xargs python -m py_compile 2>&1",
	}},

	// ── Java (Maven) ──────────────────────────────────────────────────────────
	{"pom.xml", []string{
		"mvn compile -q 2>&1 | tail -20",
	}},

	// ── Java / Kotlin (Gradle) ────────────────────────────────────────────────
	// Kotlin-DSL build file takes priority over Groovy.
	{"build.gradle.kts", []string{
		"./gradlew compileKotlin compileJava 2>&1 | tail -30",
	}},
	{"build.gradle", []string{
		"./gradlew compileJava 2>&1 | tail -30",
	}},

	// ── C# / .NET ─────────────────────────────────────────────────────────────
	// Match any .sln or .csproj at root level.
	{"*.sln", []string{
		"dotnet build --no-restore -v minimal 2>&1 | tail -30",
	}},
	{"*.csproj", []string{
		"dotnet build --no-restore -v minimal 2>&1 | tail -30",
	}},

	// ── Swift / Swift Package Manager ─────────────────────────────────────────
	{"Package.swift", []string{
		"swift build 2>&1 | tail -30",
	}},

	// ── Dart / Flutter ────────────────────────────────────────────────────────
	{"pubspec.yaml", []string{
		"dart analyze --fatal-infos 2>&1 | tail -30",
	}},

	// ── Ruby ──────────────────────────────────────────────────────────────────
	{"Gemfile", []string{
		"bundle exec rubocop --no-color --format progress 2>/dev/null || ruby -c **/*.rb 2>&1 | head -30",
	}},

	// ── PHP ───────────────────────────────────────────────────────────────────
	{"composer.json", []string{
		"find . -name '*.php' -not -path './vendor/*' | head -50 | xargs php -l 2>&1 | grep -v 'No syntax errors' | head -20",
	}},

	// ── Elixir / Mix ──────────────────────────────────────────────────────────
	{"mix.exs", []string{
		"mix compile --warnings-as-errors 2>&1 | tail -20",
	}},

	// ── Haskell (Cabal / Stack) ───────────────────────────────────────────────
	{"stack.yaml", []string{
		"stack build --fast --no-run-tests 2>&1 | tail -30",
	}},
	{"*.cabal", []string{
		"cabal build all 2>&1 | tail -30",
	}},

	// ── Scala ─────────────────────────────────────────────────────────────────
	{"build.sbt", []string{
		"sbt compile 2>&1 | tail -30",
	}},

	// ── Zig ───────────────────────────────────────────────────────────────────
	{"build.zig", []string{
		"zig build 2>&1 | tail -30",
	}},

	// ── C / C++ (CMake) ───────────────────────────────────────────────────────
	// cmake --build requires a configured build dir; fall back to syntax pass.
	{"CMakeLists.txt", []string{
		"cmake -B .codezilla_build -S . -DCMAKE_BUILD_TYPE=Debug -Wno-dev 2>&1 | tail -20 && cmake --build .codezilla_build --target all 2>&1 | tail -30",
	}},

	// ── Generic Makefile (last resort; dry-run only) ──────────────────────────
	{"Makefile", []string{
		"make -n 2>&1 | head -20",
	}},
}


// ResolveVerifyCommands returns the verification commands to run for the given
// working directory. It consults user-supplied profiles first, then built-in
// defaults. A profile with an empty slice means "skip verification".
//
// profiles is the map from codezilla.yaml verify_profiles.
// Marker names may contain glob wildcards (e.g. "*.sln") — filepath.Glob is used.
func ResolveVerifyCommands(workDir string, profiles map[string][]string) []string {
	matchesMarker := func(marker string) bool {
		if strings.ContainsAny(marker, "*?[") {
			// Glob match
			matches, err := filepath.Glob(filepath.Join(workDir, marker))
			return err == nil && len(matches) > 0
		}
		_, err := os.Stat(filepath.Join(workDir, marker))
		return err == nil
	}

	// 1. Check user-supplied profiles first (marker filename → commands).
	//    Profiles override built-ins so users can customise or disable any language.
	for marker, cmds := range profiles {
		if matchesMarker(marker) {
			// Explicit empty slice means "skip verification for this project"
			return cmds
		}
	}

	// 2. Fall back to built-in defaults (ordered slice, first match wins).
	for _, entry := range builtinVerifyMarkers {
		if matchesMarker(entry.marker) {
			return entry.commands
		}
	}

	return nil // unknown project type — no verification
}


// DetectProjectType examines the working directory for build system indicators.
// Kept for backward-compatibility with tests and logging; use ResolveVerifyCommands
// for the actual command list.
func DetectProjectType(workDir string) ProjectType {
	indicators := []struct {
		file    string
		project ProjectType
	}{
		{"go.mod", ProjectGo},
		{"Cargo.toml", ProjectRust},
		{"package.json", ProjectNode},
		{"pyproject.toml", ProjectPython},
		{"setup.py", ProjectPython},
		{"requirements.txt", ProjectPython},
		{"CMakeLists.txt", ProjectC},
	}

	for _, ind := range indicators {
		if _, err := os.Stat(filepath.Join(workDir, ind.file)); err == nil {
			return ind.project
		}
	}
	return ProjectUnknown
}

// DefaultVerifyCommands returns the built-in commands for a project type.
// Prefer ResolveVerifyCommands when profiles config is available.
func DefaultVerifyCommands(pt ProjectType) []string {
	switch pt {
	case ProjectGo:
		return []string{"go build ./...", "go vet ./..."}
	case ProjectNode:
		return []string{"npx --no-install tsc --noEmit 2>/dev/null || true"}
	case ProjectRust:
		return []string{"cargo check"}
	case ProjectPython:
		return []string{"python -m py_compile"}
	case ProjectC:
		return []string{"cmake --build . --target all 2>&1 | head -50"}
	default:
		return nil
	}
}

// RunVerification executes the given commands sequentially in workDir and
// returns a VerifyResult. It runs all commands even after failures to collect
// the full error set.
func RunVerification(ctx context.Context, workDir string, commands []string) *VerifyResult {
	if len(commands) == 0 {
		return &VerifyResult{Passed: true}
	}

	start := time.Now()
	result := &VerifyResult{
		Commands: commands,
	}

	ctx, cancel := context.WithTimeout(ctx, verifyTimeout)
	defer cancel()

	for _, cmdStr := range commands {
		cmd := exec.CommandContext(ctx, "sh", "-c", cmdStr)
		cmd.Dir = workDir
		cmd.Env = append(os.Environ(), "PAGER=cat")

		output, err := cmd.CombinedOutput()
		if err != nil {
			// Truncate very long error outputs to avoid bloating the context
			errOutput := strings.TrimSpace(string(output))
			if len(errOutput) > 2000 {
				errOutput = errOutput[:2000] + "\n[... truncated ...]"
			}

			errMsg := fmt.Sprintf("Command `%s` failed:\n%s", cmdStr, errOutput)
			if errOutput == "" {
				errMsg = fmt.Sprintf("Command `%s` failed: %v", cmdStr, err)
			}
			result.Errors = append(result.Errors, errMsg)
			// Don't stop — run all commands to catch all errors
		}
	}

	result.Duration = time.Since(start)
	result.Passed = len(result.Errors) == 0
	return result
}

// reasoningEffortForRetry returns the appropriate reasoning effort level
// for a given verify retry attempt. Each retry escalates the thinking budget.
func reasoningEffortForRetry(currentEffort string, retryNum int) string {
	levels := []string{"low", "medium", "high"}

	// Find current index
	currentIdx := -1
	for i, l := range levels {
		if l == currentEffort {
			currentIdx = i
			break
		}
	}

	// Escalate: retry 1 → medium, retry 2 → high
	targetIdx := retryNum // 1-indexed retry → 1=medium, 2=high
	if currentIdx >= targetIdx {
		// Already at or above the target level
		return currentEffort
	}
	if targetIdx >= len(levels) {
		return levels[len(levels)-1]
	}
	return levels[targetIdx]
}

