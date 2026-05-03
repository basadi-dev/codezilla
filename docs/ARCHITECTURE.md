# Codezilla Architecture

## High-Level Overview

Codezilla is a terminal-based AI coding assistant built in Rust. It runs an
**agentic loop** — the LLM reasons, calls tools, observes results, and repeats
until the task is done — all rendered in a rich TUI with syntax highlighting,
diff colours, and approval gates.

```mermaid
graph TB
    subgraph TUI["TUI Layer"]
        Composer["Composer<br/>(user input)"]
        Transcript["Transcript View<br/>(rendered entries)"]
        ApprovalPanel["Approval Panel<br/>(tool gating)"]
        StatusBar["Status Bar<br/>(tokens, ctx %, state)"]
    end

    subgraph Runtime["ConversationRuntime"]
        ThreadMgr["Thread Manager<br/>(start / resume / fork / compact)"]
        TurnMgr["Turn Executor<br/>(agent loop)"]
        EventBus["Event Bus<br/>(pub/sub)"]
    end

    subgraph Agent["Agent Core"]
        ModelGateway["Model Gateway<br/>(LLM streaming)"]
        ToolOrchestrator["Tool Orchestrator<br/>(dispatch)"]
        ApprovalMgr["Approval Manager<br/>(policy + auto-review)"]
        PermissionMgr["Permission Manager<br/>(sandbox profiles)"]
        SandboxMgr["Sandbox Manager<br/>(command execution)"]
        CheckpointStore["Checkpoint Store<br/>(undo snapshots)"]
        IntelCache["Intel Cache<br/>(repo map + symbols)"]
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
    end

    subgraph Persistence["Persistence"]
        PersistMgr["Persistence Manager<br/>(SQLite)"]
    end

    subgraph Config["Configuration"]
        ConfigMgr["Config Manager<br/>(effective config)"]
    end

    Composer -->|user message| ThreadMgr
    ThreadMgr -->|start_turn| TurnMgr
    TurnMgr -->|stream request| ModelGateway
    TurnMgr -->|tool calls| ToolOrchestrator
    ToolOrchestrator -->|dispatch| Tools
    ToolOrchestrator -->|approval check| ApprovalMgr
    ApprovalMgr -->|pending| ApprovalPanel
    ApprovalPanel -->|approve / deny| ApprovalMgr
    ApprovalMgr -->|approved| ToolOrchestrator
    Tools -->|sandbox request| PermissionMgr
    PermissionMgr -->|sandbox config| SandboxMgr
    FileTool -->|snapshot before write| CheckpointStore
    FileTool -->|invalidate symbols| IntelCache
    TurnMgr -->|persist items| PersistMgr
    TurnMgr -->|publish events| EventBus
    EventBus -->|render updates| Transcript
    ConfigMgr -->|effective config| TurnMgr
    ModelGateway -->|LLM provider API| LLM["LLM Provider<br/>(OpenAI / Ollama / …)"]
```

## The Agentic Turn Loop

The core of Codezilla is the **TurnExecutor** agent loop. Each user message
starts a turn; the turn keeps running until the model produces a final
assistant message with no tool calls.

```mermaid
flowchart TD
    Start([User sends message]) --> BuildCtx[Build system prompt<br/>+ repo map]
    BuildCtx --> CallLLM[Call LLM via Model Gateway]
    CallLLM --> ParseResp{Parse response}

    ParseResp -->|Text only| EmitText[Emit assistant message]
    EmitText --> Done([Turn complete])

    ParseResp -->|Tool calls| CheckApproval{Requires<br/>approval?}

    CheckApproval -->|No| ExecTool[Execute tool via<br/>ToolOrchestrator]
    CheckApproval -->|Yes| WaitApproval[Wait for user<br/>approval / auto-review]
    WaitApproval -->|Approved| ExecTool
    WaitApproval -->|Denied| DenyResult[Return denial result]

    ExecTool --> ToolResult[Collect tool result]
    DenyResult --> ToolResult
    ToolResult --> AppendItems[Append items to<br/>conversation + persist]
    AppendItems --> GuardCheck{Loop guards<br/>pass?}

    GuardCheck -->|Yes| CallLLM
    GuardCheck -->|No — stuck| FailTurn([Turn failed])

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

## Tool Dispatch Pipeline

When the LLM requests a tool call, it goes through a multi-stage pipeline
before execution:

```mermaid
flowchart LR
    ToolCall["Tool call<br/>from LLM"] --> Orchestrator["ToolOrchestrator<br/>(name → provider lookup)"]
    Orchestrator --> Approval{Approval<br/>required?}
    Approval -->|Yes| Policy["Approval policy<br/>check"]
    Policy --> AutoReview["Auto-review<br/>(category-based)"]
    AutoReview -->|Auto-approved| Sandbox
    AutoReview -->|Needs user| UserApproval["User approval<br/>in TUI"]
    UserApproval -->|Approved| Sandbox
    UserApproval -->|Denied| Denied["Return denial<br/>result"]
    Approval -->|No| Sandbox["Permission Manager<br/>→ Sandbox config"]
    Sandbox --> Execute["SandboxManager<br/>executes"]
    Execute --> Result["ToolResult"]

    style ToolCall fill:#2a1a3c,stroke:#dc8cff,color:#fff
    style Result fill:#1a3c2a,stroke:#64c8a3,color:#fff
    style Denied fill:#3c1a1a,stroke:#ff6464,color:#fff
```

## TUI Rendering Pipeline

The TUI renders conversation entries as styled `Line`s using ratatui:

```mermaid
flowchart TD
    Items["ConversationItem<br/>(from persistence)"] --> EntryFrom["entry_from_item()"]
    EntryFrom --> Entry["TranscriptEntry<br/>(kind + title + body)"]

    Entry --> KindCheck{Entry kind?}

    KindCheck -->|Assistant / Summary<br/>/ Reasoning| Markdown["Markdown renderer<br/>(pulldown-cmark)"]
    KindCheck -->|ToolResult<br/>read_file| ReadFileHL["read_file highlighter<br/>📄 header + syntax HL"]
    KindCheck -->|ToolResult<br/>write_file / patch_file| DiffHL["Diff highlighter<br/>syntax HL + green/red BG"]
    KindCheck -->|FileChange| DiffHL2["Diff highlighter<br/>(same pipeline)"]
    KindCheck -->|Command| CmdRender["$ prefix + output"]
    KindCheck -->|Other| PlainRender["Plain text + body colour"]

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

Runtime events flow from the agent core to the TUI via the event bus:

```mermaid
flowchart LR
    subgraph Publishers
        TE["TurnExecutor"]
        MG["ModelGateway"]
        TO["ToolOrchestrator"]
    end

    subgraph EventBus["Event Bus"]
        sub["Subscription<br/>(filtered by thread_id)"]
    end

    subgraph Consumers
        TUI["TUI App<br/>(transcript + status)"]
        SA["Spawned sub-agents"]
    end

    TE -->|TurnCompleted<br/>TurnFailed<br/>Warning| EventBus
    MG -->|TokenUsageUpdate<br/>StreamChunk| EventBus
    TO -->|ItemUpdate| EventBus
    EventBus --> TUI
    EventBus --> SA
```

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
    }

    ThreadMetadata "1" --> "*" TurnMetadata : has turns
    TurnMetadata "1" --> "*" ConversationItem : contains
    ConversationItem --> ToolCall : may reference
    ConversationItem --> ToolResult : may reference
```

## Module Dependency Map

```mermaid
graph BT
    TUI["tui<br/>(render, app, types, markdown)"]
    Runtime["runtime<br/>(ConversationRuntime)"]
    Agent["agent<br/>(executor, model_gateway, tools, approval, sandbox)"]
    Persistence["persistence<br/>(PersistenceManager)"]
    Config["config<br/>(ConfigManager)"]
    Domain["domain<br/>(types + enums)"]
    Error["error<br/>(CodError)"]
    Intel["intel<br/>(repo map, cache)"]
    LLM["llm<br/>(LlmClient trait)"]

    TUI --> Runtime
    TUI --> Domain
    Runtime --> Agent
    Runtime --> Persistence
    Runtime --> Config
    Agent --> Domain
    Agent --> Persistence
    Agent --> Intel
    Agent --> LLM
    Agent --> Error
    Persistence --> Domain
    Config --> Domain
    LLM --> Domain
```