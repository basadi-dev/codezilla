# Codezilla Architecture

## High-Level Overview

Codezilla is a terminal-based AI coding assistant built in Rust. It runs an
**agentic loop** — the LLM reasons, calls tools, observes results, and repeats
until the task is done — all rendered in a rich TUI with syntax highlighting,
diff colours, collapsible entries, and approval gates.

It can also operate headlessly via an **exec surface** (`codezilla exec`) or
expose its full API over a **JSON-RPC stdio server** for IDE integrations.

```mermaid
graph TB
    subgraph TUI["TUI Layer"]
        Composer["Composer<br/>(user input)"]
        Transcript["Transcript View<br/>(rendered entries,<br/>collapsible)"]
        ApprovalPanel["Approval Panel<br/>(tool gating)"]
        StatusBar["Status Bar<br/>(tokens, ctx %, state)"]
        ActivityPanel["Activity Panel<br/>(in-flight tools,<br/>sub-agent tree)"]
    end

    subgraph Surfaces["Surface Layer"]
        Interactive["InteractiveSurface<br/>(TUI)"]
        Exec["ExecSurface<br/>(headless / CI)"]
        AppSrv["AppServer<br/>(JSON-RPC stdio)"]
        ExecSrv["ExecServer<br/>(process-only)"]
    end

    subgraph Runtime["ConversationRuntime"]
        ThreadMgr["Thread Manager<br/>(start / resume / fork /<br/>compact / rollback)"]
        TurnMgr["Turn Executor<br/>(agent loop)"]
        EventBus["Event Bus<br/>(pub/sub)"]
        Builder["RuntimeBuilder<br/>(DI wiring)"]
    end

    subgraph Agent["Agent Core"]
        ModelGateway["Model Gateway<br/>(LLM streaming)"]
        ToolOrchestrator["Tool Orchestrator<br/>(dispatch + batching)"]
        ApprovalMgr["Approval Manager<br/>(policy + auto-review)"]
        PermissionMgr["Permission Manager<br/>(sandbox profiles)"]
        SandboxMgr["Sandbox Manager<br/>(command execution)"]
        CheckpointStore["Checkpoint Store<br/>(undo snapshots)"]
        Supervisor["Agent Supervisor<br/>(sub-agent lifecycle)"]
        ExtMgr["Extension Manager<br/>(skills, plugins,<br/>connectors)"]
    end

    subgraph Intel["Codebase Intelligence"]
        RepoMap["RepoMap<br/>(structural summary)"]
        IntelCache["IntelCache<br/>(SHA2-keyed symbols)"]
        SymbolExtractor["Symbol Extractor<br/>(regex, no tree-sitter)"]
        Walker["Walker<br/>(gitignore-aware)"]
    end

    subgraph Tools["Built-in Tools"]
        BashTool["bash_exec"]
        ShellTool["shell_exec"]
        FileTool["read_file / write_file<br/>patch_file / copy_path<br/>create_directory / remove_path"]
        ListDirTool["list_dir"]
        SearchTool["grep_search"]
        WebTool["web_fetch"]
        ImageTool["image_metadata"]
        SpawnAgent["spawn_agent"]
        UserInput["request_user_input"]
    end

    subgraph MCP["MCP (Model Context Protocol)"]
        McpRegistry["McpRegistry<br/>(unified ToolProvider)"]
        McpStdio["StdioMcpClient<br/>(per-server)"]
    end

    subgraph Persistence["Persistence"]
        PersistMgr["Persistence Manager<br/>(SQLite + filesystem)"]
    end

    subgraph Config["Configuration"]
        ConfigMgr["Config Manager<br/>(effective config)"]
    end

    subgraph Bench["Benchmark Infrastructure"]
        BenchRunner["Bench Runner<br/>(parallel task exec)"]
        BenchTask["Task Loader<br/>(YAML specs)"]
    end

    Composer --> Interactive
    Interactive --> ThreadMgr
    Exec --> ThreadMgr
    AppSrv --> ThreadMgr
    ThreadMgr -->|start_turn| TurnMgr
    TurnMgr -->|stream request| ModelGateway
    TurnMgr -->|tool calls| ToolOrchestrator
    ToolOrchestrator -->|dispatch| Tools
    ToolOrchestrator -->|dispatch| McpRegistry
    McpRegistry -->|stdio| McpStdio
    ToolOrchestrator -->|approval check| ApprovalMgr
    ApprovalMgr -->|pending| ApprovalPanel
    ApprovalPanel -->|approve / deny| ApprovalMgr
    ApprovalMgr -->|approved| ToolOrchestrator
    Tools -->|sandbox request| PermissionMgr
    PermissionMgr -->|sandbox config| SandboxMgr
    FileTool -->|snapshot before write| CheckpointStore
    FileTool -->|invalidate symbols| IntelCache
    SpawnAgent -->|delegate| Supervisor
    Supervisor -->|ephemeral thread + turn| ThreadMgr
    TurnMgr -->|persist items| PersistMgr
    TurnMgr -->|publish events| EventBus
    EventBus -->|render updates| Transcript
    EventBus -->|activity tracking| ActivityPanel
    ConfigMgr -->|effective config| TurnMgr
    TurnMgr -->|inject repo map| RepoMap
    RepoMap --> Walker
    RepoMap --> SymbolExtractor
    RepoMap --> IntelCache
    ModelGateway -->|LLM provider API| LLM["LLM Provider<br/>(Anthropic / OpenAI /<br/>Gemini / Ollama)"]
    BenchRunner -->|codezilla exec| Exec
```

## The Agentic Turn Loop

The core of Codezilla is the **TurnExecutor** agent loop. Each user message
starts a turn; the turn keeps running until the model produces a final
assistant message with no tool calls.

```mermaid
flowchart TD
    Start([User sends message]) --> BuildCtx[Build system prompt<br/>+ repo map + intent]
    BuildCtx --> CallLLM[Call LLM via Model Gateway]
    CallLLM --> ParseResp{Parse response}

    ParseResp -->|Text only| RetryCheck{Retry needed?<br/>unexecuted intent /<br/>agentic request}
    RetryCheck -->|No| EmitText[Emit assistant message]
    RetryCheck -->|Yes| CallLLM
    EmitText --> Done([Turn complete])

    ParseResp -->|Tool calls| Validate[Validate args +<br/>promote shell→bash<br/>if needed]
    Validate --> Batch[Partition into<br/>parallel batches]
    Batch --> CheckApproval{Requires<br/>approval?}

    CheckApproval -->|No| ExecTool[Execute tool via<br/>ToolOrchestrator]
    CheckApproval -->|Yes| WaitApproval[Wait for user<br/>approval / auto-review]
    WaitApproval -->|Approved| ExecTool
    WaitApproval -->|Denied| DenyResult[Return denial result]

    ExecTool --> ToolResult[Collect tool result]
    DenyResult --> ToolResult
    ToolResult --> AppendItems[Append items to<br/>conversation + persist]
    AppendItems --> DedupCheck{Duplicate read<br/>detection}
    DedupCheck -->|Duplicate| SkipRead[Short-circuit with<br/>cached result]
    DedupCheck -->|Unique| GuardCheck{Loop guards<br/>pass?}
    SkipRead --> GuardCheck
    GuardCheck -->|Yes| CallLLM
    GuardCheck -->|No — stuck| FailTurn([Turn failed])

    ParseResp -->|Empty| EmptyGuard{Empty response<br/>limit reached?}
    EmptyGuard -->|No| CallLLM
    EmptyGuard -->|Yes| FailTurn

    style Start fill:#1a3c2a,stroke:#64c8a3,color:#fff
    style Done fill:#1a3c2a,stroke:#64c8a3,color:#fff
    style FailTurn fill:#3c1a1a,stroke:#ff6464,color:#fff
    style CallLLM fill:#1a2a3c,stroke:#8cc8ff,color:#fff
    style ExecTool fill:#2a1a3c,stroke:#dc8cff,color:#fff
```

### Loop Guards

The executor has several guards to prevent infinite or degenerate loops:

| Guard | Trigger | Action |
|-------|---------|--------|
| Consecutive failures | Every tool in a round returns `ok: false` | Nudge → fail after threshold |
| Absolute backstop | Too many iterations (≈100) | Fail the turn |
| Read-only saturation | 4+ rounds of only read tools | Nudge to act |
| Empty response | Model returns neither text nor tool calls | Retry once, then fail |
| Cumulative nudges | Too many nudges of any kind | Fail fast |
| Repetition detection | Model repeats the same read pattern | Nudge to break out |
| Cross-round dedup | Same tool call signature across rounds | Short-circuit with prior result |

### Turn Intent Classification

Before the first LLM call, the executor classifies the user's request into
one of `Edit`, `Debug`, `Review`, `Answer`, `Inventory`, or `Unknown`. This
drives decisions like whether a text-only response should trigger a retry
nudge and how verbose the repo map injection should be.

## Tool Dispatch Pipeline

When the LLM requests tool calls, they go through a multi-stage pipeline:

```mermaid
flowchart LR
    ToolCalls["Tool calls<br/>from LLM"] --> Validate["Validate args<br/>+ promote shell→bash"]
    Validate --> Batch["Partition into<br/>parallel batches<br/>(write-set conflict)"]
    Batch --> Orchestrator["ToolOrchestrator<br/>(name → provider)"]
    Orchestrator --> Approval{Approval<br/>required?}
    Approval -->|Yes| Policy["Approval policy<br/>check"]
    Policy --> AutoReview["Auto-review<br/>(category-based)"]
    AutoReview -->|Auto-approved| Sandbox
    AutoReview -->|Needs user| UserApproval["User approval<br/>in TUI"]
    UserApproval -->|Approved| Sandbox
    UserApproval -->|Denied| Denied["Return denial<br/>result"]
    Approval -->|No| Sandbox["Permission Manager<br/>→ Sandbox config"]
    Sandbox --> Execute["SandboxManager<br/>executes"]
    Execute --> Checkpoint{"File write?"}
    Checkpoint -->|Yes| Snap["CheckpointStore<br/>pre-state snapshot"]
    Snap --> Result["ToolResult"]
    Checkpoint -->|No| Result

    style ToolCalls fill:#2a1a3c,stroke:#dc8cff,color:#fff
    style Result fill:#1a3c2a,stroke:#64c8a3,color:#fff
    style Denied fill:#3c1a1a,stroke:#ff6464,color:#fff
```

### Parallel Batching

Tool calls are partitioned into sequential batches using write-set analysis.
Consecutive calls whose write sets don't conflict are grouped into a single
batch and executed with `join_all`. Any call with an unknown or conflicting
write set forces a serialisation barrier. Read-only tools (`read_file`,
`list_dir`, `grep_search`) are always parallel-safe.

## Sub-Agent Supervision

The `AgentSupervisor` manages the lifecycle of child agents spawned via the
`spawn_agent` tool:

```mermaid
flowchart TD
    Parent["Parent turn calls<br/>spawn_agent"] --> Guard{Depth / slot<br/>guard}
    Guard -->|Blocked| Error["Return error<br/>result"]
    Guard -->|OK| Redirect{Directory<br/>inventory?}
    Redirect -->|Yes| ListDir["Redirect to<br/>list_dir"]
    Redirect -->|No| Spawn["Start ephemeral<br/>thread + turn"]
    Spawn --> Announce["Publish<br/>ChildAgentSpawned<br/>event"]
    Announce --> Wait["Await terminal<br/>status via EventBus"]
    Wait --> Timeout{Timed out?}
    Timeout -->|No| Collect["Read last<br/>agent message"]
    Timeout -->|Yes| Cancel["Interrupt +<br/>grace period"]
    Cancel --> Collect
    Collect --> Strip["Strip <think><br/>sections"]
    Strip --> Cleanup{Passed?}
    Cleanup -->|Yes| Delete["Delete ephemeral<br/>thread"]
    Cleanup -->|No| Keep["Preserve for<br/>inspection"]
    Delete --> Return["Return ToolResult"]
    Keep --> Return

    style Parent fill:#1a2a3c,stroke:#8cc8ff,color:#fff
    style Return fill:#1a3c2a,stroke:#64c8a3,color:#fff
    style Error fill:#3c1a1a,stroke:#ff6464,color:#fff
```

Key constraints:
- **Concurrency**: Controlled by a `Semaphore` (`max_concurrent_child_agents`)
- **Depth**: `max_spawn_depth` prevents unbounded recursive spawning
- **Directory inventory redirect**: Prompts that look like "list all files recursively"
  are short-circuited to `list_dir` instead of spawning a model sub-agent

## Codebase Intelligence

The `intel` module provides a structural repository summary injected into the
system prompt at turn start, reducing blind `read_file` / `list_dir` calls:

```mermaid
flowchart LR
    TurnStart["Turn starts"] --> Walk["Walker<br/>(ignore crate,<br/>gitignore-aware)"]
    Walk --> Files["FileSummary[]<br/>(path, lang, size,<br/>binary?)"]
    Files --> Extract["Symbol Extractor<br/>(regex per lang)"]
    Extract --> Cache{"IntelCache<br/>hit?"}
    Cache -->|Hit| Reuse["Use cached symbols"]
    Cache -->|Miss| Parse["Parse + store<br/>(SHA2-keyed)"]
    Reuse --> Format["format_repo_map()<br/>(token budget trim)"]
    Parse --> Format
    Format --> Inject["Inject into<br/>system prompt"]

    style TurnStart fill:#1a2a3c,stroke:#8cc8ff,color:#fff
    style Inject fill:#1a3c2a,stroke:#64c8a3,color:#fff
```

Design decisions:
- **No tree-sitter** — symbols extracted via compiled `Regex` patterns (good enough for navigation)
- **No external processes** — the `ignore` crate handles `.gitignore` traversal in pure Rust
- **O(1) per-call after first run** — SHA2-keyed in-process cache; file writes invalidate entries

## TUI Rendering Pipeline

The TUI renders conversation entries as styled `Line`s using ratatui:

```mermaid
flowchart TD
    Items["ConversationItem<br/>(from persistence)"] --> EntryFrom["entry_from_item()"]
    EntryFrom --> Entry["TranscriptEntry<br/>(kind + title + body<br/>+ collapsed flag)"]

    Entry --> Collapsed{Collapsed?}
    Collapsed -->|Yes| ChevronOnly["▸ title only<br/>(auto-collapse for<br/>read-only tools)"]
    Collapsed -->|No| KindCheck{Entry kind?}

    KindCheck -->|Assistant / Summary<br/>/ Reasoning| Markdown["Markdown renderer<br/>(pulldown-cmark)"]
    KindCheck -->|ToolResult<br/>read_file| ReadFileHL["read_file highlighter<br/>📄 header + syntax HL"]
    KindCheck -->|ToolResult<br/>write_file / patch_file| DiffHL["Diff highlighter<br/>syntax HL + green/red BG"]
    KindCheck -->|FileChange| DiffHL2["Diff highlighter<br/>(same pipeline)"]
    KindCheck -->|Command| CmdRender["$ prefix + output"]
    KindCheck -->|Other| PlainRender["Plain text + body colour"]

    ChevronOnly --> Gutter
    Markdown --> Gutter["Prepend gutter<br/>│ prefix"]
    ReadFileHL --> Gutter
    DiffHL --> Gutter
    DiffHL2 --> Gutter
    CmdRender --> Gutter
    PlainRender --> Gutter

    Gutter --> Lines["Vec&lt;Line&gt;"]
    Lines --> Selection["Apply drag selection<br/>highlight (if active)"]
    Selection --> Viewport["Render to viewport<br/>with scroll"]

    style Markdown fill:#1a2a3c,stroke:#8cc8ff,color:#fff
    style ReadFileHL fill:#1a3c2a,stroke:#64c8a3,color:#fff
    style DiffHL fill:#1a3c2a,stroke:#64c8a3,color:#fff
```

### Syntax Highlighting & Diff Colours

- **`read_file` results** — detected by `📄` header → language inferred from
  path → `highlight_code_line()` applies keyword/string/comment colours.
- **`write_file` / `patch_file` diffs** — detected by `---`/`+++`/`@@` markers →
  language inferred from diff header → added lines get `BG_DIFF_ADD`
  (green tint `Rgb(20,60,30)`), removed lines get `BG_DIFF_REMOVE` (red tint
  `Rgb(60,20,20)`), each with syntax highlighting on top.

### Collapsible Entries

Read-only tool results (`read_file`, `list_dir`, `grep_search`) are
auto-collapsed by default. Each `TranscriptEntry` carries a `collapsed: bool`
flag; when collapsed, only a `▸ title` chevron line is rendered instead of
the full body, reducing transcript noise.

## Activity Tracking

The `ActivityState` reducer tracks in-flight agent work for the TUI header
and activity panel:

- **Tool tracking**: every in-flight tool call with name, hint, and `started_at`
- **Sub-agent tracking**: `ChildAgentActivity` entries with lifecycle status
  (Running → Completed / Failed / Interrupted / TimedOut)
- **Blocked state**: approval waits override the header display
- **Panel rendering**: multi-tool batches and sub-agents get a dedicated panel
  (capped at 6 rows)

## Thread Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Active: start_thread
    Active --> Active: resume_thread / fork_thread
    Active --> Compacting: compact_thread
    Compacting --> Active: compaction done
    Active --> RolledBack: rollback_thread
    RolledBack --> Active: resume_thread
    Active --> Archived: archive_thread
    Archived --> [*]: delete_thread
```

## Event Flow

Runtime events flow from the agent core to consumers via the event bus:

```mermaid
flowchart LR
    subgraph Publishers
        TE["TurnExecutor"]
        MG["ModelGateway"]
        TO["ToolOrchestrator"]
        SV["AgentSupervisor"]
    end

    subgraph EventBus["Event Bus"]
        sub["Subscription<br/>(filtered by thread_id)"]
    end

    subgraph Consumers
        TUI["TUI App<br/>(transcript + status<br/>+ activity)"]
        SA["Spawned sub-agents"]
        AS["AppServer<br/>(JSON-RPC relay)"]
        ES["ExecSurface<br/>(stdout)"]
    end

    TE -->|TurnCompleted<br/>TurnFailed<br/>Warning| EventBus
    MG -->|TokenUsageUpdate<br/>StreamChunk| EventBus
    TO -->|ItemStarted / Updated<br/>/ Completed| EventBus
    SV -->|ChildAgentSpawned| EventBus
    EventBus --> TUI
    EventBus --> SA
    EventBus --> AS
    EventBus --> ES
```

### Typed Event Payloads

The `event_payload` module provides a typed consumer-side view over the
JSON event payloads. `RuntimeEventPayload` is a tagged enum with variants
for every `RuntimeEventKind`, enabling pattern-matching instead of manual
JSON parsing:

| Variant | Key fields |
|---------|-----------|
| `TurnCompleted` | `turn_id`, `token_usage`, `metrics`, `file_changes` |
| `TurnFailed` | `kind` (error label), `reason` |
| `ItemUpdated` | `item_id`, `delta` (streaming), `mode` |
| `ChildAgentSpawned` | `parent_tool_call_id`, `child_thread_id`, `label` |
| `CompactionStatus` | `status` ("started" / "completed" / "failed") |
| `TokenUsageUpdate` | `input_tokens`, `output_tokens`, `cached_tokens` |

## LLM Provider Layer

```mermaid
flowchart TD
    MG["ModelGateway"] --> Trait["LlmClient trait<br/>(stream_chat)"]
    Trait --> Anthropic["AnthropicProvider<br/>(Claude family)"]
    Trait --> OpenAI["OpenAIProvider<br/>(GPT-4o family)"]
    Trait --> Gemini["GeminiProvider<br/>(Gemini family)"]
    Trait --> Ollama["OllamaProvider<br/>(local models,<br/>multi-modal)"]

    style MG fill:#1a2a3c,stroke:#8cc8ff,color:#fff
```

The `ModelGateway` handles token estimation, context-window budgeting,
and streaming response assembly. Token estimation uses character-based
heuristics with per-field multipliers for JSON serialisation overhead.

## Key Data Types

```mermaid
classDiagram
    class ConversationItem {
        +String item_id
        +ThreadId thread_id
        +TurnId turn_id
        +i64 created_at
        +ItemKind kind
        +JsonValue payload
    }

    class ThreadMetadata {
        +ThreadId thread_id
        +String title
        +ThreadStatus status
        +String cwd
        +String model_id
        +String provider_id
    }

    class TurnMetadata {
        +TurnId turn_id
        +TurnStatus status
        +TokenUsage token_usage
    }

    class ToolCall {
        +ToolCallId tool_call_id
        +String tool_name
        +JsonValue arguments
        +ToolProviderKind provider_kind
    }

    class ToolResult {
        +ToolCallId tool_call_id
        +bool ok
        +JsonValue output
        +Option~String~ error_message
    }

    class EffectiveConfig {
        +AgentConfig agent
        +LlmConfig llm
        +ModelSettings model_settings
        +CodebaseIntelConfig codebase_intel
        +AutoCompactionConfig auto_compaction
        +Vec~McpServerConfig~ mcp_servers
    }

    class TranscriptEntry {
        +EntryKind kind
        +String title
        +Vec~Line~ body
        +bool collapsed
    }

    ThreadMetadata "1" --> "*" TurnMetadata : has turns
    TurnMetadata "1" --> "*" ConversationItem : contains
    ConversationItem --> ToolCall : may reference
    ConversationItem --> ToolResult : may reference
```

## Benchmark Infrastructure

The `bench` module provides a task-based evaluation harness:

```mermaid
flowchart TD
    Config["BenchConfig<br/>(tasks_dir, model,<br/>runs, parallelism)"] --> Load["Load YAML<br/>task specs"]
    Load --> Filter["Apply glob<br/>filter"]
    Filter --> Pool["Thread pool<br/>(parallelism N)"]
    Pool --> Task["Per-task:<br/>1. Copy fixtures<br/>2. Assert pre-flight<br/>3. Git init<br/>4. codezilla exec --json<br/>5. Extract metrics<br/>6. Run validation"]
    Task --> Aggregate["Aggregate runs<br/>(flake detection)"]
    Aggregate --> Report["summary.json +<br/>results.jsonl +<br/>per-task logs/<br/>transcript.md"]

    style Config fill:#1a2a3c,stroke:#8cc8ff,color:#fff
    style Report fill:#1a3c2a,stroke:#64c8a3,color:#fff
```

Features: cost estimation (per-model pricing tables), timeout enforcement,
multi-run flake detection, workspace preservation on failure, human-readable
transcript generation from event streams.

## Module Dependency Map

```mermaid
graph BT
    TUI["tui<br/>(render, app, types,<br/>markdown, activity,<br/>transcript_view,<br/>selection, autocomplete)"]
    Surfaces["surfaces<br/>(Interactive, Exec)"]
    Server["server<br/>(AppServer, ExecServer)"]
    Runtime["runtime<br/>(ConversationRuntime,<br/>RuntimeBuilder)"]
    Agent["agent<br/>(executor, model_gateway,<br/>tools, approval, sandbox,<br/>supervisor, checkpoint)"]
    Persistence["persistence<br/>(PersistenceManager)"]
    Config["config<br/>(ConfigManager)"]
    Domain["domain<br/>(types + enums)"]
    Error["error<br/>(CodError)"]
    Intel["intel<br/>(repo map, cache,<br/>symbols, walker)"]
    LLM["llm<br/>(LlmClient trait,<br/>providers)"]
    EventPayload["event_payload<br/>(typed payloads)"]
    MCP["mcp<br/>(McpRegistry,<br/>StdioMcpClient)"]
    Bench["bench<br/>(runner, task)"]

    TUI --> Runtime
    TUI --> Domain
    TUI --> EventPayload
    Surfaces --> Runtime
    Surfaces --> TUI
    Server --> Runtime
    Server --> EventPayload
    Runtime --> Agent
    Runtime --> Persistence
    Runtime --> Config
    Runtime --> Intel
    Agent --> Domain
    Agent --> Persistence
    Agent --> Intel
    Agent --> LLM
    Agent --> Error
    Agent --> MCP
    Bench --> Domain
    EventPayload --> Domain
    Persistence --> Domain
    Config --> Domain
    LLM --> Domain
    MCP --> Domain
```

## Directory Layout

```
src/
├── main.rs                    # CLI entry point (clap)
├── llm/
│   ├── mod.rs                 # LlmClient trait
│   ├── client.rs              # HTTP client wrapper
│   └── providers/
│       ├── anthropic.rs       # Claude family
│       ├── openai.rs          # GPT-4o family
│       ├── gemini.rs          # Gemini family
│       └── ollama.rs          # Local models (multi-modal)
├── logger/                    # Structured logging setup
└── system/
    ├── mod.rs                 # Public re-exports
    ├── domain.rs              # Core types, enums, constants
    ├── error.rs               # CodError hierarchy
    ├── config.rs              # ConfigManager, EffectiveConfig
    ├── persistence.rs         # SQLite + filesystem persistence
    ├── event_payload.rs       # Typed event payload enum
    ├── surfaces.rs            # InteractiveSurface, ExecSurface
    ├── server.rs              # AppServer, ExecServer (JSON-RPC)
    ├── agent/
    │   ├── mod.rs             # Agent subsystem re-exports
    │   ├── executor.rs        # TurnExecutor (main loop)
    │   ├── executor/
    │   │   ├── context.rs     # TurnContext builder
    │   │   ├── tool_dispatch.rs # Batch dispatch + dedup
    │   │   └── utils.rs       # Guards, intent, validation
    │   ├── model_gateway.rs   # LLM streaming + token budget
    │   ├── tools.rs           # Built-in tool providers
    │   ├── supervisor.rs      # Sub-agent spawn/await/cancel
    │   ├── approval.rs        # Approval policies
    │   ├── permission.rs      # Permission profiles
    │   ├── sandbox.rs         # Command sandboxing
    │   ├── checkpoint.rs      # Pre-write state snapshots
    │   ├── event_bus.rs       # Pub/sub event distribution
    │   └── extensions.rs      # Skills, plugins, connectors
    ├── intel/
    │   ├── mod.rs             # RepoMap entry point
    │   ├── walker.rs          # Gitignore-aware file walker
    │   ├── symbols.rs         # Regex-based symbol extraction
    │   ├── cache.rs           # SHA2-keyed symbol cache
    │   └── format.rs          # Token-budgeted map formatter
    ├── mcp/
    │   ├── mod.rs             # MCP module entry
    │   ├── registry.rs        # Multi-server tool routing
    │   └── stdio.rs           # Stdio MCP client transport
    ├── runtime/
    │   ├── mod.rs             # ConversationRuntime, params
    │   ├── builder.rs         # RuntimeBuilder (DI)
    │   ├── thread.rs          # Thread lifecycle ops
    │   ├── turn.rs            # Turn lifecycle ops
    │   └── discovery.rs       # Model discovery
    ├── bench/
    │   ├── mod.rs             # Bench module entry
    │   ├── runner.rs          # Parallel task runner
    │   └── task.rs            # YAML task spec loader
    └── tui/
        ├── mod.rs             # TUI entry point
        ├── app.rs             # InteractiveApp state machine
        ├── render.rs          # Layout + widget rendering
        ├── types.rs           # TranscriptEntry, EntryKind
        ├── transcript_view.rs # Scrollable transcript widget
        ├── markdown.rs        # Markdown → ratatui Lines
        ├── activity.rs        # ActivityState reducer
        ├── approval.rs        # Approval dialog widget
        ├── input.rs           # Input handling
        ├── selection.rs       # Drag-select + copy
        ├── autocomplete.rs    # Slash-command completion
        ├── composer_history.rs# Input history (↑/↓)
        └── threads.rs         # Thread picker overlay
```