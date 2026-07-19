#!/usr/bin/env python3
"""Generate synthetic MemoryControllerPlan training examples.

The output format is OpenAI-style chat JSONL:
{"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant","content":"{...}"}]}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SYSTEM = "Emit only a MemoryControllerPlan JSON object."

PREFERENCES = [
    ("concise answers", "User prefers concise answers."),
    ("TypeScript examples", "User prefers TypeScript examples."),
    ("PostgreSQL for backend projects", "User prefers PostgreSQL for backend projects."),
    ("Hetzner for deployments", "User prefers Hetzner for deployments."),
    ("Rust for CLI tools", "User prefers Rust for CLI tools."),
]

PROJECT_OVERRIDES = [
    ("For this repo use SQLite, not Postgres.", "For this project, user wants SQLite rather than PostgreSQL."),
    ("For this service use Go instead of Python.", "For this project, user wants Go rather than Python."),
    ("In this app, keep the UI dense and utilitarian.", "For this app, user wants a dense utilitarian UI."),
]

DECISIONS = [
    ("We decided memory should be external, not stored in weights.", "Memory should be stored externally, not in model weights."),
    ("Let's use JSON memory operations and deterministic execution.", "Use JSON memory operations with deterministic runtime execution."),
    ("We decided to retrieve memory before every model turn.", "Retrieve relevant memory before every model turn."),
]

SEARCH_REQUESTS = [
    ("What database should I use?", "user database preferences current project database constraints"),
    ("How should I deploy this?", "user deployment preferences current project infrastructure"),
    ("What language should I write this in?", "user programming language preferences current project stack"),
]

NOISE = [
    "thanks",
    "nice, that worked",
    "ok continue",
    "cool",
]

UPDATES = [
    (
        "mem_pref_db",
        "Actually, I prefer SQLite for small backend projects now.",
        "preference",
        "global",
        "User prefers SQLite for small backend projects.",
    ),
    (
        "mem_project_lang",
        "Update that project language note: use Rust, not Go.",
        "project_context",
        "thread",
        "For this project, user wants Rust rather than Go.",
    ),
    (
        "mem_deploy",
        "Refine my deployment preference: Hetzner only for EU services.",
        "preference",
        "global",
        "User prefers Hetzner for EU service deployments.",
    ),
]

FORGETS = [
    ("mem_mongodb_pref", "Forget my MongoDB preference."),
    ("mem_old_deploy", "Forget my old deployment target."),
    ("mem_lang_pref", "Forget my previous language preference."),
]


def plan(**kwargs: object) -> str:
    payload = {
        "search": kwargs.get("search", []),
        "store": kwargs.get("store", []),
        "update": kwargs.get("update", []),
        "forget": kwargs.get("forget", []),
        "answerStrategy": kwargs.get("answerStrategy", ""),
    }
    return json.dumps(payload, separators=(",", ":"))


def row(user: str, assistant_plan: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_plan},
        ]
    }


def examples(seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    builders = [
        preference_example,
        project_override_example,
        decision_example,
        search_example,
        noise_example,
        forget_example,
        update_example,
        conflict_example,
    ]
    for _ in range(count):
        out.append(rng.choice(builders)(rng))
    return out


def preference_example(rng: random.Random) -> dict:
    phrase, memory = rng.choice(PREFERENCES)
    user = rng.choice([
        f"I prefer {phrase}.",
        f"Remember that I prefer {phrase}.",
        f"I usually want {phrase}.",
    ])
    return row(user, plan(
        store=[{"kind": "preference", "scope": "global", "content": memory, "importance": 0.85}],
        answerStrategy="Store durable user preference.",
    ))


def project_override_example(rng: random.Random) -> dict:
    user, memory = rng.choice(PROJECT_OVERRIDES)
    return row(user, plan(
        search=[{"query": "current project scoped preferences", "limit": 6}],
        store=[{"kind": "project_context", "scope": "thread", "content": memory, "importance": 0.85}],
        answerStrategy="Treat this as project-scoped unless user says it is global.",
    ))


def decision_example(rng: random.Random) -> dict:
    user, memory = rng.choice(DECISIONS)
    return row(user, plan(
        store=[{"kind": "decision", "scope": "thread", "content": memory, "importance": 0.9}],
        answerStrategy="Store architecture decision for future turns.",
    ))


def search_example(rng: random.Random) -> dict:
    user, query = rng.choice(SEARCH_REQUESTS)
    return row(user, plan(
        search=[{"query": query, "limit": 6}],
        answerStrategy="Use retrieved preferences and project constraints if relevant.",
    ))


def noise_example(rng: random.Random) -> dict:
    return row(rng.choice(NOISE), plan(answerStrategy="Do not store low-information chatter."))


def forget_example(rng: random.Random) -> dict:
    memory_id, user = rng.choice(FORGETS)
    subject = user.removeprefix("Forget my ").removesuffix(".")
    return row(user, plan(
        search=[{"query": subject, "limit": 10}],
        forget=[{"memoryId": memory_id}],
        answerStrategy="Forget the exact matching stale memory after retrieval.",
    ))


def update_example(rng: random.Random) -> dict:
    memory_id, user, kind, scope, content = rng.choice(UPDATES)
    return row(user, plan(
        search=[{"query": content, "limit": 6}],
        update=[{
            "memoryId": memory_id,
            "kind": kind,
            "scope": scope,
            "content": content,
            "importance": 0.9,
        }],
        answerStrategy="Update the existing memory rather than storing a duplicate.",
    ))


def conflict_example(rng: random.Random) -> dict:
    user = rng.choice([
        "For this repo use SQLite, even though I usually prefer PostgreSQL.",
        "For this service use Go, even though I usually prefer TypeScript examples.",
    ])
    return row(user, plan(
        search=[{"query": "global preference current project override", "limit": 6}],
        store=[{"kind": "project_context", "scope": "thread", "content": user, "importance": 0.9}],
        answerStrategy="Treat the current project instruction as more specific than any conflicting global preference.",
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("training/memory-controller/synthetic.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in examples(args.seed, args.count):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {args.count} examples to {args.output}")


if __name__ == "__main__":
    main()
