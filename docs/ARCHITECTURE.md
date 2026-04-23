# Codezilla Architecture

> **Codezilla v2.0** — AI-powered coding assistant written in Rust.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Module Graph](#module-graph)
4. [Startup & CLI Flow](#startup--cli-flow)
5. [Request Lifecycle](#request-lifecycle)
6. [System Layer](#system-layer)
   - [Config Resolution](#config-resolution)
   - [ConversationRuntime](#conversationruntime)
   - [Persistence (SQLite)](#persistence-sqlite)
   - [Tool System](#tool-system)
   - [Approval Pipeline](#approval-pipeline)
   - [Event Bus](#event-bus)
7. [LLM Layer](#llm-layer)
8. [Surface Layer](#surface-layer)
   - [InteractiveSurface (TUI)](#interactivesurface-tui)
   - [ExecSurface](#execsurface)
   - [AppServer (JSON-RPC)](#appserver-json-rpc)
   - [ExecServer (JSON-RPC)](#execserver-json-rpc)
9. [Data Model](#data-model)
10. [Key Flows](#key-flows)
    - [Interactive Turn](#interactive-turn-flow)
    - [Tool Call Round-trip](#tool-call-round-trip)
    - [Approval Flow](#approval-flow)

---

## Overview

Codezilla is a single Rust binary (`src/main.rs`) that boots a Tokio multi-threaded runtime and dispatches to one of several *surfaces* based on the CLI sub-command.  All long-running conversational logic lives in `ConversationRuntime`, which sits between every surface and the underlying LLM/persistence layers.

```mermaid
graph TD
    binary["codezilla binary"]
    cli["CLI (clap)"]
    cm["ConfigManager"]
    rt["ConversationRuntime"]
    is["InteractiveSurface<br/>(TUI)"]
    es["ExecSurface<br/>(stdout)"]
    as_["AppServer<br/>(JSON-RPC)"]
    xs["ExecServer<br/>(JSON-RPC)"]

    binary --> cli --> cm --> rt
    rt --> is
    rt --> es
    rt --> as_
    rt --> xs
```

---

## Project Structure

```
codezilla/
├── src/
│   ├── main.rs               # Entry point, CLI, surfaces dispatch
│   ├── config/
│   │   └── mod.rs            # Legacy LLM config (YAML → Config struct)
│   ├── llm/
│   │   ├── mod.rs            # Core types: Message, LlmClient trait, StreamChunk
│   │   ├── client.rs         # UnifiedClient — dispatches to provider impls
│   │   └── providers/
│   │       ├── mod.rs
│   │       ├── ollama.rs     # Ollama OpenAI-compat provider
│   │       ├── openai.rs     # OpenAI / openai-compat provider
│   │       ├── anthropic.rs  # Anthropic provider
│   │       └── gemini.rs     # Google Gemini provider
│   ├── logger/
│   │   └── mod.rs            # tracing subscriber init (file-only, JSON)
│   └── system/
│       ├── mod.rs            # Re-exports from sub-modules
│       ├── config.rs         # EffectiveConfig, ConfigManager, AuthManager
│       ├── domain.rs         # All domain types (enums, structs, type aliases)
│       ├── persistence.rs    # SQLite via rusqlite (threads/turns/items)
│       ├── runtime.rs        # ConversationRuntime + all manager types
│       ├── surfaces.rs       # InteractiveSurface, ExecSurface
│       ├── server.rs         # AppServer, ExecServer (JSON-RPC over stdio)
│       └── interactive_tui.rs# ratatui TUI — full interactive terminal UI
├── Cargo.toml
├── config.yaml               # Default runtime config
├── skills/                   # Markdown skill definitions
└── docs/
    └── ARCHITECTURE.md       # This file
```

---

## Module Graph

```mermaid
graph TD
    main["main.rs<br/>(entry point)"]

    subgraph config_mod["config/"]
        config_mod_rs["mod.rs<br/>Config struct<br/>load_config()"]
    end

    subgraph logger_mod["logger/"]
        logger_mod_rs["mod.rs<br/>init()"]
    end

    subgraph llm_mod["llm/"]
        llm_mod_rs["mod.rs<br/>LlmClient trait<br/>Message, StreamChunk"]
        llm_client["client.rs<br/>UnifiedClient"]
        subgraph providers_mod["providers/"]
            ollama["ollama.rs"]
            openai["openai.rs"]
            anthropic["anthropic.rs"]
            gemini["gemini.rs"]
        end
    end

    subgraph system_mod["system/"]
        sys_config["config.rs<br/>EffectiveConfig<br/>ConfigManager<br/>AuthManager"]
        sys_domain["domain.rs<br/>All domain types"]
        sys_persistence["persistence.rs<br/>PersistenceManager<br/>SQLite"]
        sys_runtime["runtime.rs<br/>ConversationRuntime<br/>ToolProviders<br/>EventBus<br/>ApprovalManager"]
        sys_surfaces["surfaces.rs<br/>InteractiveSurface<br/>ExecSurface"]
        sys_server["server.rs<br/>AppServer<br/>ExecServer"]
        sys_tui["interactive_tui.rs<br/>run_interactive_tui()"]
    end

    main --> config_mod
    main --> logger_mod
    main --> sys_config
    main --> sys_runtime
    main --> sys_surfaces
    main --> sys_server

    sys_config --> config_mod
    sys_config --> sys_domain
    sys_runtime --> llm_client
    sys_runtime --> sys_persistence
    sys_runtime --> sys_domain
    sys_runtime --> sys_config
    sys_surfaces --> sys_runtime
    sys_surfaces --> sys_tui
    sys_server --> sys_runtime
    sys_server --> sys_domain
    sys_tui --> sys_runtime
    sys_tui --> sys_domain

    llm_client --> llm_mod_rs
    llm_client --> ollama
    llm_client --> openai
    llm_client --> anthropic
    llm_client --> gemini

    logger_mod_rs --> config_mod_rs
```

---

## Startup & CLI Flow

```mermaid
flowchart TD
    A["binary starts<br/>main()"] --> B["set panic hook"]
    B --> C["build tokio multi-thread runtime"]
    C --> D["parse CLI args via clap"]
    D --> E["resolve config path"]
    E --> F["ConfigManager.load_effective_config()"]
    F --> G["logger::init()"]
    G --> H["AuthManager::new()"]
    H --> I["ConversationRuntime::new()"]
    I --> J{subcommand?}

    J -->|none / prompt| K["InteractiveSurface"]
    J -->|exec| L["ExecSurface"]
    J -->|review| M["ExecSurface ephemeral"]
    J -->|resume| N["InteractiveSurface + resume_thread"]
    J -->|fork| O["InteractiveSurface + fork_thread"]
    J -->|app-server| P["AppServer stdio"]
    J -->|exec-server| Q["ExecServer stdio"]
    J -->|login| R["AuthManager login"]
    J -->|logout| S["AuthManager logout"]
    J -->|plugin / mcp / sandbox / features| T["runtime query + print JSON"]
```

---

## Request Lifecycle

```mermaid
sequenceDiagram
    participant User
    participant Surface as "Surface<br/>(Interactive/Exec)"
    participant Runtime as "ConversationRuntime"
    participant LLM as "LlmClient<br/>(UnifiedClient)"
    participant Tools as "ToolProvider"
    participant DB as "PersistenceManager<br/>(SQLite)"
    participant Bus as "EventBus"

    User->>Surface: prompt / keypress
    Surface->>Runtime: start_turn(TurnStartParams)
    Runtime->>DB: create_turn()
    Runtime->>Bus: publish(TurnStarted)
    Runtime->>DB: load thread history

    loop agent loop (max_iterations)
        Runtime->>LLM: stream(messages, tools)
        LLM-->>Runtime: StreamChunk tokens
        Runtime->>Bus: publish(ItemUpdated) per chunk
        Runtime->>DB: append_item(AgentMessage)

        alt tool calls present
            Runtime->>Tools: execute(ToolCall)
            Tools-->>Runtime: ToolResult
            Runtime->>Bus: publish(ItemCompleted ToolCall)
            Runtime->>DB: append_item(ToolResult)
        else no tool calls
            Runtime->>Bus: publish(TurnCompleted)
            Runtime->>DB: update_turn(Completed)
        end
    end

    Bus-->>Surface: RuntimeEvent stream
    Surface-->>User: render / print
```

---

## System Layer

### Config Resolution

Two config layers coexist:

| Layer | Type | File |
|---|---|---|
| **Legacy LLM Config** | `config::Config` | `src/config/mod.rs` |
| **Effective / Spec Config** | `system::config::EffectiveConfig` | `src/system/config.rs` |

```mermaid
flowchart LR
    yaml["config.yaml"] --> CM["ConfigManager"]
    env["ENV vars"] --> legacy["load_legacy_config()"]
    cli_flags["CLI flags<br/>--model, --sandbox…"] --> overrides["cli_overrides map"]
    CM --> merged["deep_merge JSON"]
    merged --> profile["optional profile overlay"]
    profile --> overrides
    overrides --> EC["EffectiveConfig"]
    legacy --> EC
```

`EffectiveConfig` embeds the legacy `Config` as `legacy_llm_config` so `ConversationRuntime` can access provider/model settings without a second config system.

---

### ConversationRuntime

The central coordinator in `src/system/runtime.rs`. It owns:

```mermaid
classDiagram
    class ConversationRuntime {
        +effective_config: EffectiveConfig
        +session: AccountSession
        -llm: Arc~UnifiedClient~
        -persistence: Arc~PersistenceManager~
        -event_bus: Arc~EventBus~
        -approval_manager: Arc~ApprovalManager~
        -sandbox: Arc~SandboxManager~
        -permissions: Arc~PermissionManager~
        -tool_providers: Vec~Arc dyn ToolProvider~
        +start_thread()
        +resume_thread()
        +fork_thread()
        +list_threads()
        +read_thread()
        +archive_thread()
        +compact_thread()
        +rollback_thread()
        +start_turn()
        +interrupt_turn()
        +steer_turn()
        +resolve_approval()
        +list_plugins()
        +list_skills()
        +list_mcp_servers()
        +list_models()
        +reset_memories()
        +event_bus() Arc~EventBus~
        +effective_config() EffectiveConfig
    }
```

`start_turn()` spawns an async task that drives the full agent loop — LLM calls, tool dispatch, event publishing and persistence — without blocking the caller.

---

### Persistence (SQLite)

Managed by `PersistenceManager` in `src/system/persistence.rs`.  Uses `rusqlite` with WAL journal mode.

```mermaid
erDiagram
    THREADS {
        TEXT thread_id PK
        TEXT title
        INTEGER created_at
        INTEGER updated_at
        TEXT cwd
        TEXT model_id
        TEXT provider_id
        TEXT status
        TEXT forked_from_id
        INTEGER archived
        INTEGER ephemeral
        TEXT memory_mode
        INTEGER last_sequence
    }
    TURNS {
        TEXT turn_id PK
        TEXT thread_id FK
        INTEGER created_at
        INTEGER updated_at
        TEXT status
        TEXT started_by_surface
        TEXT token_usage_json
    }
    ITEMS {
        TEXT item_id PK
        TEXT thread_id FK
        TEXT turn_id FK
        INTEGER created_at
        TEXT kind
        TEXT payload_json
        INTEGER item_order
        INTEGER tombstoned
    }
    LOGS {
        INTEGER id PK
        INTEGER created_at
        TEXT level
        TEXT message
    }

    THREADS ||--o{ TURNS : "has"
    THREADS ||--o{ ITEMS : "contains"
    TURNS ||--o{ ITEMS : "groups"
```

On startup, `PersistenceManager` calls `recover_incomplete_turns()` to mark any `Running` or `WaitingForApproval` turns as `Interrupted` (crash recovery).

---

### Tool System

```mermaid
classDiagram
    class ToolProvider {
        <<trait>>
        +get_kind() ToolProviderKind
        +list_tools(ctx) Vec~ToolDefinition~
        +execute(call, ctx) Result~ToolResult~
    }
    class ShellToolProvider {
        -sandbox: Arc~SandboxManager~
        -permissions: Arc~PermissionManager~
    }
    class FileToolProvider {
        -sandbox: Arc~SandboxManager~
        -permissions: Arc~PermissionManager~
    }
    class WebToolProvider
    class SearchToolProvider

    ToolProvider <|-- ShellToolProvider
    ToolProvider <|-- FileToolProvider
    ToolProvider <|-- WebToolProvider
    ToolProvider <|-- SearchToolProvider
```

| Tool | Provider | Approval Required |
|---|---|---|
| `shell_exec` | ShellToolProvider | ✅ Yes |
| `read_file` | FileToolProvider | ❌ No |
| `write_file` | FileToolProvider | ✅ Yes |
| `create_directory` | FileToolProvider | ✅ Yes |
| `grep_search` | SearchToolProvider | ❌ No |
| `web_fetch` | WebToolProvider | ❌ No |

All file/command operations pass through `SandboxManager`, which enforces `SandboxMode` (read-only, workspace-write, danger-full-access).

---

### Approval Pipeline

```mermaid
sequenceDiagram
    participant Runtime
    participant AM as "ApprovalManager"
    participant Bus as "EventBus"
    participant Surface

    Runtime->>AM: create_approval(request)
    AM-->>Runtime: PendingApproval
    Runtime->>Bus: publish(ApprovalRequested)
    Bus-->>Surface: ApprovalRequested event

    alt AutoReviewer
        AM->>AM: AutoReviewer.review()
        AM->>AM: resolve_approval(decision)
    else User reviewer (TUI)
        Surface->>Runtime: resolve_approval(A/D)
        Runtime->>AM: resolve_approval(decision)
    end

    AM->>AM: notify waiters
    Runtime->>Bus: publish(ApprovalResolved)
    Bus-->>Surface: ApprovalResolved event
```

Approvals time out after a configurable number of seconds and are auto-resolved as `TimedOut`.

---

### Event Bus

`EventBus` in `runtime.rs` uses a Tokio `broadcast` channel (capacity 1024). Every surface subscribes with an optional `thread_id` filter.

```mermaid
graph LR
    RT["ConversationRuntime"] -- "publish(RuntimeEvent)" --> EB["EventBus<br/>broadcast::channel"]
    EB --> TUI["InteractiveSurface<br/>(TUI subscriber)"]
    EB --> EX["ExecSurface<br/>(exec subscriber)"]
    EB --> AS["AppServer<br/>(all-threads subscriber)"]
```

`RuntimeEventKind` values:

- `ThreadStarted` / `TurnStarted` / `TurnCompleted` / `TurnFailed`
- `ItemStarted` / `ItemUpdated` / `ItemCompleted`
- `ApprovalRequested` / `ApprovalResolved`
- `Warning` / `Disconnected`

---

## LLM Layer

```mermaid
classDiagram
    class LlmClient {
        <<trait>>
        +complete(messages, tools, model, …) LlmResponse
        +stream(messages, tools, model, …) Receiver~StreamChunk~
    }
    class UnifiedClient {
        +provider: String
        +http: reqwest::Client
        +cfg: Config
    }
    class OllamaProvider {
        <<module ollama.rs>>
        +complete()
        +stream()
    }
    class OpenAiProvider {
        <<module openai.rs>>
        +complete()
        +stream()
    }
    class AnthropicProvider {
        <<module anthropic.rs>>
        +complete()
        +stream()
    }
    class GeminiProvider {
        <<module gemini.rs>>
        +complete()
        +stream()
    }

    LlmClient <|-- UnifiedClient
    UnifiedClient --> OllamaProvider : provider == "ollama"
    UnifiedClient --> OpenAiProvider : provider == "openai"
    UnifiedClient --> AnthropicProvider : provider == "anthropic"
    UnifiedClient --> GeminiProvider : provider == "gemini"
```

`StreamChunk` variants flowing from provider to runtime:

```
StreamChunk::Text(delta)
StreamChunk::ToolCallDelta { index, id, name, arguments_delta }
StreamChunk::Usage(TokenUsage)
StreamChunk::Done
```

---

## Surface Layer

### InteractiveSurface (TUI)

`InteractiveSurface` in `surfaces.rs` creates / resumes a thread then hands off to `run_interactive_tui()` in `interactive_tui.rs`.

```mermaid
stateDiagram-v2
    [*] --> Bootstrap: run_interactive_tui()
    Bootstrap --> EventLoop: load thread history
    EventLoop --> HandleKey: crossterm event (40ms poll)
    EventLoop --> HandleRuntimeEvent: EventBus broadcast
    HandleKey --> Composer: typing
    HandleKey --> ThreadList: navigation
    HandleKey --> Transcript: scrolling
    HandleKey --> Submit: Enter
    Submit --> Runtime: start_turn() or steer_turn()
    HandleRuntimeEvent --> Transcript: ItemUpdated → render delta
    HandleRuntimeEvent --> Approval: ApprovalRequested → modal
    HandleRuntimeEvent --> Ready: TurnCompleted
    EventLoop --> [*]: should_quit
```

Three panes, cycled with **Tab**:

| Pane | Focus | Keys |
|---|---|---|
| Threads | `FocusPane::Threads` | ↑↓ navigate, Enter open |
| Transcript | `FocusPane::Transcript` | ↑↓ scroll, PgUp/PgDn, End auto-scroll |
| Composer | `FocusPane::Composer` | Enter submit, Shift+Enter newline |

Slash commands: `/new`, `/fork`, `/quit`, `/exit`, `/interrupt`, `/threads`, `/open <id>`, `/resume <id>`, `/help`

---

### ExecSurface

Headless, non-interactive. Starts a turn, listens on EventBus, prints deltas to stdout (Human mode) or as JSONL.

```mermaid
flowchart LR
    CLI["exec subcommand"] --> ES["ExecSurface"]
    ES --> RT["Runtime.start_turn()"]
    RT --> EB["EventBus"]
    EB -->|ItemUpdated delta| stdout["stdout print"]
    EB -->|TurnCompleted| exit0["exit 0"]
    EB -->|TurnFailed| exit1["exit 1"]
    EB -->|ApprovalRequested| bail["bail! error"]
```

---

### AppServer (JSON-RPC)

Full JSON-RPC 2.0 server over stdio. Used by IDE extensions and GUI clients.

```mermaid
graph TD
    stdin["stdin (line-delimited JSON-RPC)"] --> AS["AppServer"]
    AS --> RT["ConversationRuntime"]
    RT --> EB["EventBus"]
    EB --> notification["JSON-RPC notification → stdout"]

    AS -->|"thread/start<br/>thread/resume<br/>thread/fork<br/>thread/list<br/>thread/read<br/>thread/archive<br/>thread/compact<br/>thread/rollback"| RT
    AS -->|"turn/start<br/>turn/interrupt<br/>turn/steer"| RT
    AS -->|"review/start"| RT
    AS -->|"approval/resolve"| RT
    AS -->|"fs/* commands"| FS["Local filesystem"]
    AS -->|"command/exec<br/>command/exec/write<br/>command/exec/terminate"| PT["ProcessTable<br/>(managed child processes)"]
    AS -->|"skills/list<br/>plugin/list<br/>app/list<br/>model/list"| RT
    AS -->|"config/read"| RT
    AS -->|"memory/reset"| RT
```

---

### ExecServer (JSON-RPC)

Lightweight process-management-only server. No `ConversationRuntime` dependency.

```mermaid
graph LR
    stdin2["stdin (JSON-RPC)"] --> XS["ExecServer"]
    XS -->|"process/start<br/>process/read<br/>process/write<br/>process/resize<br/>process/terminate"| PT2["ProcessTable"]
    XS -->|"fs/* commands"| FS2["Local filesystem"]
    PT2 --> child["child processes"]
    child -->|"process/output notification"| stdout2["stdout"]
```

---

## Data Model

```mermaid
classDiagram
    class ThreadMetadata {
        +thread_id: ThreadId
        +title: Option~String~
        +created_at: i64
        +updated_at: i64
        +cwd: Option~String~
        +model_id: ModelId
        +provider_id: ProviderId
        +status: ThreadStatus
        +forked_from_id: Option~ThreadId~
        +archived: bool
        +ephemeral: bool
        +memory_mode: MemoryMode
    }
    class TurnMetadata {
        +turn_id: TurnId
        +thread_id: ThreadId
        +created_at: i64
        +updated_at: i64
        +status: TurnStatus
        +started_by_surface: SurfaceKind
        +token_usage: TokenUsage
    }
    class ConversationItem {
        +item_id: ItemId
        +thread_id: ThreadId
        +turn_id: TurnId
        +created_at: i64
        +kind: ItemKind
        +payload: JsonValue
    }
    class RuntimeEvent {
        +event_id: String
        +kind: RuntimeEventKind
        +thread_id: Option~ThreadId~
        +turn_id: Option~TurnId~
        +sequence: i64
        +payload: JsonValue
        +emitted_at: i64
    }

    ThreadMetadata "1" --> "*" TurnMetadata
    ThreadMetadata "1" --> "*" ConversationItem
    TurnMetadata "1" --> "*" ConversationItem
    RuntimeEvent ..> ThreadMetadata : references
    RuntimeEvent ..> TurnMetadata : references
```

`ItemKind` values: `UserMessage`, `UserAttachment`, `AgentMessage`, `ReasoningText`, `ReasoningSummary`, `ToolCall`, `ToolResult`, `CommandExecution`, `CommandOutput`, `FileChange`, `Error`, `ReviewMarker`, `Status`

---

## Key Flows

### Interactive Turn Flow

```mermaid
sequenceDiagram
    participant U as User (keyboard)
    participant TUI as InteractiveTUI
    participant R as ConversationRuntime
    participant L as LlmClient
    participant DB as SQLite

    U->>TUI: type prompt + Enter
    TUI->>R: start_turn(TurnStartParams)
    R->>DB: create_turn(Running)
    R-->>TUI: TurnStartResult{turn_id}
    note over R: spawns async agent task

    loop agent task
        R->>L: stream(history + system_prompt + tools)
        L-->>R: Text chunks
        R-->>TUI: ItemUpdated{delta}
        TUI-->>U: render token stream
        R->>DB: append_item(AgentMessage delta)
    end

    R->>DB: update_turn(Completed)
    R-->>TUI: TurnCompleted
    TUI-->>U: status = "Ready"
```

---

### Tool Call Round-trip

```mermaid
sequenceDiagram
    participant R as Runtime
    participant L as LLM
    participant TP as ToolProvider
    participant AM as ApprovalManager

    L-->>R: StreamChunk::ToolCallDelta
    R->>R: assemble tool call args
    R->>TP: list_tools() → check definition
    
    alt requires_approval == true
        R->>AM: create_approval(request)
        AM-->>R: PendingApproval
        R->>R: wait_for_approval()
        alt Approved
            R->>TP: execute(call, ctx)
            TP-->>R: ToolResult
        else Denied
            R-->>L: ToolResult{error: "denied"}
        end
    else no approval needed
        R->>TP: execute(call, ctx)
        TP-->>R: ToolResult
    end

    R->>R: append tool result to history
    R->>L: next LLM call with tool result
```

---

### Approval Flow

```mermaid
stateDiagram-v2
    [*] --> Pending: create_approval()
    Pending --> AutoApproved: reviewer == AutoReviewer
    Pending --> WaitingUser: reviewer == User
    WaitingUser --> Approved: user presses A
    WaitingUser --> Denied: user presses D / Esc
    WaitingUser --> TimedOut: timeout_seconds elapsed
    AutoApproved --> [*]: action executes
    Approved --> [*]: action executes
    Denied --> [*]: error returned to LLM
    TimedOut --> [*]: error returned to LLM
```

---

*Generated from source at `src/` — Codezilla v2.0.0*
