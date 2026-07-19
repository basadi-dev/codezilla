# Memory Controller

Codezilla now has two memory paths:

1. Automatic runtime memory: completed turns are compacted into durable records, then searched and injected on later turns.
2. Explicit controller memory: a model or tool call emits a structured memory plan, and Codezilla applies it deterministically.

The post-trained model should act as a memory controller, not as the memory store. It should emit JSON matching `MemoryControllerPlan`:

```json
{
  "search": [
    {
      "query": "user database preferences",
      "limit": 6
    }
  ],
  "store": [
    {
      "kind": "preference",
      "scope": "global",
      "content": "User prefers PostgreSQL for backend projects.",
      "importance": 0.9
    }
  ],
  "update": [],
  "forget": [],
  "answerStrategy": "Use retrieved deployment and database preferences if relevant."
}
```

The runtime applies this via the `run_memory_plan` tool. Individual tools also exist:

- `search_memory`
- `save_memory`
- `update_memory`
- `forget_memory`
- `list_memory`

## Training Target

Fine-tune the controller on examples where it learns to:

- store durable user preferences, project facts, decisions, and task state
- search before answering when prior context may matter
- ignore short-lived or irrelevant chatter
- update stale or scoped memories instead of duplicating them
- forget memories when explicitly asked
- avoid treating retrieved memory as truth when the current user message contradicts it

## Runtime Contract

The controller must output valid JSON only. Codezilla validates and executes the plan; free-form text should not mutate memory.

Useful memory kinds:

- `preference`
- `fact`
- `decision`
- `project_context`
- `task`
- `summary`
- `episodic`

Scopes:

- `thread`: current conversation/project
- `global`: durable user-level preference or fact

## First Fine-Tuning Pass

1. Use `training/memory-controller/examples.jsonl` as seed data.
2. Generate more synthetic examples for store/search/update/forget/ignore/conflict cases.
3. Fine-tune a LoRA/QLoRA adapter on an instruction/chat model.
4. Wrap that model so it emits `MemoryControllerPlan`.
5. Call `run_memory_plan` before answering and after each completed turn.
