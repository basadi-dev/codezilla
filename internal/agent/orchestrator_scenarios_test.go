package agent

import (
	"encoding/json"
	"testing"

	"codezilla/pkg/logger"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// ──────────────────────────────────────────────────────────────────────────────
// Loop Detector Tests — E1 scenario
// ──────────────────────────────────────────────────────────────────────────────

func TestLoopDetector_NoLoop(t *testing.T) {
	d := newLoopDetector(5, 3)

	// Different calls should not trigger
	if r := d.record("fileRead", `{"path":"/a"}`); r != "" {
		t.Errorf("unexpected loop detected: %s", r)
	}
	if r := d.record("fileRead", `{"path":"/b"}`); r != "" {
		t.Errorf("unexpected loop detected: %s", r)
	}
	if r := d.record("grepSearch", `{"query":"test"}`); r != "" {
		t.Errorf("unexpected loop detected: %s", r)
	}
}

func TestLoopDetector_DetectsConsecutiveLoop(t *testing.T) {
	d := newLoopDetector(10, 3)

	args := `{"path":"/same/file"}`
	d.record("fileRead", args)
	d.record("fileRead", args)
	result := d.record("fileRead", args) // 3rd consecutive = loop

	if result != "fileRead" {
		t.Errorf("expected loop detection for fileRead, got %q", result)
	}
}

func TestLoopDetector_BreakResets(t *testing.T) {
	d := newLoopDetector(10, 3)

	args := `{"path":"/same"}`
	d.record("fileRead", args)
	d.record("fileRead", args)
	d.record("grepSearch", `{"query":"break"}`) // breaks the streak
	result := d.record("fileRead", args)         // not consecutive anymore

	if result != "" {
		t.Errorf("expected no loop after break, got %q", result)
	}
}

func TestLoopDetector_JSONCanonicalisation(t *testing.T) {
	d := newLoopDetector(10, 3)

	// Same args, different whitespace/key order → should still be detected
	d.record("fileRead", `{"path": "/test",  "mode": "read"}`)
	d.record("fileRead", `{"mode":"read","path":"/test"}`)
	result := d.record("fileRead", `{ "path":"/test", "mode": "read" }`)

	if result != "fileRead" {
		t.Errorf("expected loop detection with canonical JSON, got %q", result)
	}
}

func TestLoopDetector_WindowOverflow(t *testing.T) {
	d := newLoopDetector(3, 2) // tiny window

	d.record("a", `{}`)
	d.record("b", `{}`)
	d.record("c", `{}`)  // window is now [a, b, c]
	d.record("a", `{}`)  // window becomes [b, c, a] — a is not consecutive
	result := d.record("a", `{}`) // window becomes [c, a, a] — consecutive!

	if result != "a" {
		t.Errorf("expected loop in window, got %q", result)
	}
}

func TestLoopDetector_DefaultValues(t *testing.T) {
	d := newLoopDetector(0, 0) // should use defaults

	if d.window != 10 {
		t.Errorf("default window = %d, want 10", d.window)
	}
	if d.maxRepeat != 3 {
		t.Errorf("default maxRepeat = %d, want 3", d.maxRepeat)
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Middleware / Tool Parsing Tests — A5 scenario (expanded)
// ──────────────────────────────────────────────────────────────────────────────

func TestSanitiseSpecialTokens_AllPatterns(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"im_start", "<|im_start|>assistant\nHello", "Hello"},
		{"im_end", "Hello<|im_end|>\n", "Hello"},
		{"tool_call", "<|tool_call|>{}", "{}"},
		{"eot_id", "Hello<|eot_id|>", "Hello"},
		{"header_id", "<|start_header_id|>assistant<|end_header_id|>\nHi", "Hi"},
		{"python_tag", "<|python_tag|>code", "code"},
		{"function_calls", "<function_calls>\n<invoke>\nfoo\n</invoke>\n</function_calls>\n", "foo"},
		{"analysis_tag", "<analysis>thinking</analysis>Result", "thinkingResult"},
		{"clean_text", "No special tokens here", "No special tokens here"},
		{"multiple", "<|im_start|>assistant\n<|eot_id|>Hello<|im_end|>", "Hello"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SanitiseSpecialTokens(tt.input)
			if got != tt.want {
				t.Errorf("SanitiseSpecialTokens(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestLooksLikeLeakedToolCall(t *testing.T) {
	tests := []struct {
		name string
		text string
		want bool
	}{
		{"special_token_leak", `<|tool_call|>{"name":"fileRead","arguments":{"path":"/test"}}`, true},
		{"functions_leak", `to=functions.fileRead{"path":"/test"}`, true},
		{"tool_call_tag", `<tool_call> {"name":"fileRead"} </tool_call>`, true}, // <tool_call> IS detected as leaked
		{"bare_json_tool_params", `{"name":"fileRead","arguments":{"path":"/test"}}`, true},
		{"regular_prose_with_json", `Here's how to create a config:\n\n{"key":"value","another":"setting"}\n\nThis configures the system with two settings that control behavior.`, false},
		{"regular_text", "Just a normal response with no tool calls.", false},
		{"action_pattern", `{"action":"read","path":"/etc/hosts"}`, true},
		{"explanation_json", "Here's an example:\n\n```json\n{\"name\": \"test\"}\n```\n\nThis shows the format.", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := looksLikeLeakedToolCall(tt.text)
			if got != tt.want {
				t.Errorf("looksLikeLeakedToolCall(%q) = %v, want %v", tt.text[:min(len(tt.text), 60)], got, tt.want)
			}
		})
	}
}

func TestParseLLMResponse_ToolCallTag(t *testing.T) {
	log, _ := logger.New(logger.Config{Silent: true})

	// <tool_call>{...}</tool_call> format — note: current implementation
	// does NOT parse this format. This test documents that limitation.
	response := `Let me check that.
<tool_call>
{"name":"fileRead","arguments":{"file_path":"/etc/hosts"}}
</tool_call>`

	_, tools := ParseLLMResponse(response, log)

	// Current behavior: <tool_call> tag format is NOT parsed
	// (it IS detected as a leaked tool call by looksLikeLeakedToolCall,
	//  but ParseLLMResponse doesn't handle this specific XML variant)
	if len(tools) != 0 {
		t.Logf("NOTE: <tool_call> tag format IS now parsed — %d tools found", len(tools))
	}
}

func TestParseLLMResponse_FunctionsLeak(t *testing.T) {
	log, _ := logger.New(logger.Config{Silent: true})

	// to=functions.X{...} format (GPT-OSS)
	response := `to=functions.fileManage{"action":"read","path":"/etc/hosts"}`

	_, tools := ParseLLMResponse(response, log)

	if len(tools) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tools))
	}
	if tools[0].Function.Name != "fileManage" {
		t.Errorf("tool name = %q, want fileManage", tools[0].Function.Name)
	}
}

func TestParseLLMResponse_InvokeXML(t *testing.T) {
	log, _ := logger.New(logger.Config{Silent: true})

	// <invoke> XML format — note: current implementation does NOT parse this.
	// This test documents the limitation.
	response := `<invoke>
<tool_name>grepSearch</tool_name>
<parameters>
<query>TODO</query>
<path>/src</path>
</parameters>
</invoke>`

	_, tools := ParseLLMResponse(response, log)

	// Current behavior: <invoke> format is NOT parsed by ParseLLMResponse
	if len(tools) != 0 {
		t.Logf("NOTE: <invoke> XML format IS now parsed — %d tools found", len(tools))
	}
}

func TestParseLLMResponse_NoToolCall(t *testing.T) {
	log, _ := logger.New(logger.Config{Silent: true})

	response := "This is a perfectly normal response that doesn't contain any tool calls whatsoever."
	text, tools := ParseLLMResponse(response, log)

	if len(tools) != 0 {
		t.Errorf("expected 0 tool calls, got %d", len(tools))
	}
	if text != response {
		t.Errorf("text should be unchanged, got %q", text)
	}
}

func TestInferToolNameFromParams(t *testing.T) {
	tests := []struct {
		name   string
		params map[string]interface{}
		want   string
	}{
		{"execute", map[string]interface{}{"command": "ls -la"}, "execute"},
		{"webSearch", map[string]interface{}{"query": "golang testing"}, "webSearch"},
		{"fetchURL", map[string]interface{}{"url": "https://example.com"}, "fetchURL"},
		{"subAgent", map[string]interface{}{"task": "analyze code"}, "subAgent"},
		{"fileRead", map[string]interface{}{"file_path": "/test.go"}, "fileRead"},
		{"fileWrite", map[string]interface{}{"file_path": "/test.go", "content": "package main"}, "fileWrite"},
		{"fileEdit", map[string]interface{}{"file_path": "/test.go", "target_content": "old", "replacement_content": "new"}, "fileEdit"},
		{"multiReplace", map[string]interface{}{"file_path": "/test.go", "replacements": []interface{}{}}, "multiReplace"},
		{"fileManage_read", map[string]interface{}{"action": "read", "path": "/test"}, "fileManage"},
		{"fileManage_write", map[string]interface{}{"action": "write", "path": "/test"}, "fileManage"},
		{"unknown", map[string]interface{}{"random": "data"}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := inferToolNameFromParams(tt.params)
			if got != tt.want {
				t.Errorf("inferToolNameFromParams = %q, want %q", got, tt.want)
			}
		})
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Token Tracker Tests
// ──────────────────────────────────────────────────────────────────────────────

func TestTokenTracker_RecordAndSession(t *testing.T) {
	tracker := NewTokenTracker()

	tracker.Record("model-a", MakeUsage(100, 50, 150))
	tracker.Record("model-a", MakeUsage(200, 100, 300))
	tracker.Record("model-b", MakeUsage(50, 25, 75))

	session := tracker.SessionTotal()
	if session.TotalTokens != 525 {
		t.Errorf("session total = %d, want 525", session.TotalTokens)
	}
	if session.PromptTokens != 350 {
		t.Errorf("session prompt = %d, want 350", session.PromptTokens)
	}

	breakdown := tracker.TurnModelBreakdown()
	if len(breakdown) != 2 {
		t.Fatalf("expected 2 models in breakdown, got %d", len(breakdown))
	}
	if breakdown["model-a"].TotalTokens != 450 {
		t.Errorf("model-a total = %d, want 450", breakdown["model-a"].TotalTokens)
	}
	if breakdown["model-b"].TotalTokens != 75 {
		t.Errorf("model-b total = %d, want 75", breakdown["model-b"].TotalTokens)
	}
}

func TestTokenTracker_ResetTurn(t *testing.T) {
	tracker := NewTokenTracker()

	tracker.Record("model-a", MakeUsage(100, 50, 150))
	tracker.ResetTurn()
	tracker.Record("model-b", MakeUsage(200, 100, 300))

	breakdown := tracker.TurnModelBreakdown()
	if len(breakdown) != 1 {
		t.Fatalf("expected 1 model after reset, got %d", len(breakdown))
	}
	if _, ok := breakdown["model-a"]; ok {
		t.Error("model-a should not be in turn breakdown after reset")
	}

	// Session should still have both
	session := tracker.SessionTotal()
	if session.TotalTokens != 450 {
		t.Errorf("session total = %d, want 450", session.TotalTokens)
	}
}

func TestTokenTracker_NilUsageIgnored(t *testing.T) {
	tracker := NewTokenTracker()
	tracker.Record("model", nil)
	tracker.Record("model", &anyllm.Usage{}) // zero-valued

	if tracker.TurnCount() != 0 {
		t.Errorf("expected 0 turns for nil/zero usage, got %d", tracker.TurnCount())
	}
}

func TestCompactNumber(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{0, "0"},
		{850, "850"},
		{1000, "1k"},
		{1500, "1.5k"},
		{19455, "19.5k"},
		{100000, "100k"},
		{1234567, "1.2m"},
		{-500, "-500"},
	}

	for _, tt := range tests {
		got := CompactNumber(tt.n)
		if got != tt.want {
			t.Errorf("CompactNumber(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}

func TestFormatNumber(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{0, "0"},
		{999, "999"},
		{1234, "1,234"},
		{19455, "19,455"},
		{1234567, "1,234,567"},
	}

	for _, tt := range tests {
		got := FormatNumber(tt.n)
		if got != tt.want {
			t.Errorf("FormatNumber(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Model Router Tests — C3 scenario
// ──────────────────────────────────────────────────────────────────────────────

func TestModelRouter_ClassifyGreeting(t *testing.T) {
	r := NewModelRouter(true, "fast-model", "default-model", "heavy-model")

	tier, reason := r.Classify("hello", 0)
	if tier != TierFast {
		t.Errorf("greeting should route to Fast, got %s", tier.String())
	}
	if reason == "" {
		t.Error("reason should not be empty")
	}
}

func TestModelRouter_ClassifyComplexTask(t *testing.T) {
	r := NewModelRouter(true, "fast-model", "default-model", "heavy-model")

	tier, _ := r.Classify("implement a new REST API with authentication, rate limiting, and database migrations. 1. Design the schema 2. Write the handlers 3. Add tests", 0)
	if tier != TierHeavy {
		t.Errorf("complex task should route to Heavy, got %s", tier.String())
	}
}

func TestModelRouter_ClassifyStandardCoding(t *testing.T) {
	r := NewModelRouter(true, "fast-model", "default-model", "heavy-model")

	tier, _ := r.Classify("what does the processInput function do?", 0)
	if tier == TierHeavy {
		t.Errorf("standard question should not route to Heavy, got %s", tier.String())
	}
}

func TestModelRouter_Disabled(t *testing.T) {
	r := NewModelRouter(false, "fast", "default", "heavy")

	tier, reason := r.Classify("hello", 0)
	if tier != TierDefault {
		t.Errorf("disabled router should return Default, got %s", tier.String())
	}
	if reason != "routing disabled" {
		t.Errorf("reason = %q, want 'routing disabled'", reason)
	}
}

func TestModelRouter_ModelForTier_Extended(t *testing.T) {
	r := NewModelRouter(true, "fast", "default", "heavy")

	if r.ModelForTier(TierFast) != "fast" {
		t.Error("TierFast should return fast model")
	}
	if r.ModelForTier(TierDefault) != "default" {
		t.Error("TierDefault should return default model")
	}
	if r.ModelForTier(TierHeavy) != "heavy" {
		t.Error("TierHeavy should return heavy model")
	}
}

func TestModelRouter_FallbackWhenEmpty(t *testing.T) {
	r := NewModelRouter(true, "", "default", "")

	if r.ModelForTier(TierFast) != "default" {
		t.Error("empty fast model should fallback to default")
	}
	if r.ModelForTier(TierHeavy) != "default" {
		t.Error("empty heavy model should fallback to default")
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Orchestrator State Machine Tests
// ──────────────────────────────────────────────────────────────────────────────

func TestOrchestratorState_String(t *testing.T) {
	states := []struct {
		state OrchestratorState
		want  string
	}{
		{StatePrompting, "StatePrompting"},
		{StateStreaming, "StateStreaming"},
		{StateParsing, "StateParsing"},
		{StateExecutingTools, "StateExecutingTools"},
		{StateErrorRecovery, "StateErrorRecovery"},
		{StateComplete, "StateComplete"},
		{OrchestratorState(99), "UnknownState"},
	}

	for _, tt := range states {
		if got := tt.state.String(); got != tt.want {
			t.Errorf("state %d String() = %q, want %q", tt.state, got, tt.want)
		}
	}
}

func TestRequestTier_String_Extended(t *testing.T) {
	tiers := []struct {
		tier RequestTier
		want string
	}{
		{TierFast, "fast"},
		{TierDefault, "default"},
		{TierHeavy, "heavy"},
		{RequestTier(99), "unknown"},
	}

	for _, tt := range tiers {
		if got := tt.tier.String(); got != tt.want {
			t.Errorf("tier %d String() = %q, want %q", tt.tier, got, tt.want)
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Verify Module Tests — D1/D2 scenario
// ──────────────────────────────────────────────────────────────────────────────

func TestIsFileModifyingTool_Extended(t *testing.T) {
	tests := []struct {
		name   string
		tool   string
		params map[string]interface{}
		want   bool
	}{
		{"multiReplace", "multiReplace", nil, true},
		{"fileManage_write", "fileManage", map[string]interface{}{"action": "write"}, true},
		{"fileManage_delete", "fileManage", map[string]interface{}{"action": "delete"}, true},
		{"fileManage_read", "fileManage", map[string]interface{}{"action": "read"}, false},
		{"fileManage_list", "fileManage", map[string]interface{}{"action": "list"}, false},
		{"fileManage_mkdir", "fileManage", map[string]interface{}{"action": "mkdir"}, false},
		{"fileRead", "fileRead", nil, false},
		{"grepSearch", "grepSearch", nil, false},
		{"execute", "execute", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isFileModifyingTool(tt.tool, tt.params)
			if got != tt.want {
				t.Errorf("isFileModifyingTool(%q, %v) = %v, want %v", tt.tool, tt.params, got, tt.want)
			}
		})
	}
}

func TestReasoningEffortForRetry_Extended(t *testing.T) {
	tests := []struct {
		current  string
		retry    int
		expected string
	}{
		{"low", 1, "medium"},
		{"low", 2, "high"},
		{"medium", 1, "medium"},
		{"medium", 2, "high"},
		{"high", 1, "high"},
		{"high", 2, "high"},
		{"", 1, "medium"},
		{"", 3, "high"},
	}

	for _, tt := range tests {
		got := reasoningEffortForRetry(tt.current, tt.retry)
		if got != tt.expected {
			t.Errorf("reasoningEffortForRetry(%q, %d) = %q, want %q",
				tt.current, tt.retry, got, tt.expected)
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Test Harness Smoke Test
// ──────────────────────────────────────────────────────────────────────────────

func TestTestHarness_Construction(t *testing.T) {
	h := NewTestHarness([]MockLLMResponse{
		{Content: "Hello!"},
	})

	if h.Agent == nil {
		t.Fatal("Agent should not be nil")
	}
	if h.Provider == nil {
		t.Fatal("Provider should not be nil")
	}
	if h.Callbacks == nil {
		t.Fatal("Callbacks should not be nil")
	}
	if h.Config.Model != "test-model" {
		t.Errorf("Model = %q, want test-model", h.Config.Model)
	}
}

func TestTestHarness_WithOptions(t *testing.T) {
	h := NewTestHarness([]MockLLMResponse{},
		WithMaxIterations(5),
		WithLoopDetection(3, 2),
		WithAutoRoute("fast", "default", "heavy"),
	)

	if h.Config.MaxIterations != 5 {
		t.Errorf("MaxIterations = %d, want 5", h.Config.MaxIterations)
	}
	if h.Config.LoopDetectWindow != 3 {
		t.Errorf("LoopDetectWindow = %d, want 3", h.Config.LoopDetectWindow)
	}
	if h.Config.FastModel != "fast" {
		t.Errorf("FastModel = %q, want fast", h.Config.FastModel)
	}
}

func TestMakeToolCall_Format(t *testing.T) {
	tc := MakeToolCall("call_1", "fileRead", map[string]interface{}{
		"file_path": "/test.go",
	})

	if tc.ID != "call_1" {
		t.Errorf("ID = %q, want call_1", tc.ID)
	}
	if tc.Function.Name != "fileRead" {
		t.Errorf("Name = %q, want fileRead", tc.Function.Name)
	}

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatalf("failed to parse arguments: %v", err)
	}
	if args["file_path"] != "/test.go" {
		t.Errorf("file_path = %v, want /test.go", args["file_path"])
	}
}

func TestMockLLMProvider_FIFO(t *testing.T) {
	provider := &mockLLMProvider{
		responses: []MockLLMResponse{
			{Content: "first"},
			{Content: "second"},
		},
	}

	r1, _ := provider.next()
	r2, _ := provider.next()
	_, err := provider.next()

	if r1.Content != "first" {
		t.Errorf("first response = %q, want first", r1.Content)
	}
	if r2.Content != "second" {
		t.Errorf("second response = %q, want second", r2.Content)
	}
	if err == nil {
		t.Error("expected error when exhausting responses")
	}
}

func TestMockLLMProvider_CallTracking(t *testing.T) {
	provider := &mockLLMProvider{
		responses: []MockLLMResponse{{Content: "response"}},
	}

	provider.recordCall("model-a", nil, nil)

	if provider.CallCount() != 1 {
		t.Errorf("CallCount = %d, want 1", provider.CallCount())
	}
	calls := provider.Calls()
	if calls[0].Model != "model-a" {
		t.Errorf("Model = %q, want model-a", calls[0].Model)
	}
}
