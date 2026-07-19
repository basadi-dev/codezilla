#!/usr/bin/env python3
"""Validate memory-controller chat JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VALID_SCOPES = {"thread", "global"}


def validate_plan(plan: dict, source: str) -> list[str]:
    errors: list[str] = []
    for key in ("search", "store", "update", "forget"):
        if key in plan and not isinstance(plan[key], list):
            errors.append(f"{source}: {key} must be a list")

    for i, op in enumerate(plan.get("search", [])):
        if not isinstance(op, dict) or not str(op.get("query", "")).strip():
            errors.append(f"{source}: search[{i}].query missing")

    for i, op in enumerate(plan.get("store", [])):
        if not isinstance(op, dict):
            errors.append(f"{source}: store[{i}] must be object")
            continue
        if not str(op.get("kind", "")).strip():
            errors.append(f"{source}: store[{i}].kind missing")
        if not str(op.get("content", "")).strip():
            errors.append(f"{source}: store[{i}].content missing")
        if op.get("scope", "thread") not in VALID_SCOPES:
            errors.append(f"{source}: store[{i}].scope invalid")
        importance = op.get("importance", 0.7)
        if not isinstance(importance, (int, float)) or not 0 <= importance <= 1:
            errors.append(f"{source}: store[{i}].importance must be 0..1")

    for i, op in enumerate(plan.get("update", [])):
        if not isinstance(op, dict) or not str(op.get("memoryId", op.get("memory_id", ""))).strip():
            errors.append(f"{source}: update[{i}].memoryId missing")
        if isinstance(op, dict) and op.get("scope") is not None and op.get("scope") not in VALID_SCOPES:
            errors.append(f"{source}: update[{i}].scope invalid")

    for i, op in enumerate(plan.get("forget", [])):
        if not isinstance(op, dict) or not str(op.get("memoryId", op.get("memory_id", ""))).strip():
            errors.append(f"{source}: forget[{i}].memoryId missing")

    return errors


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        source = f"{path}:{lineno}"
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"{source}: invalid JSONL row: {e}")
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            errors.append(f"{source}: messages must contain system/user/assistant")
            continue
        assistant = messages[-1]
        if assistant.get("role") != "assistant":
            errors.append(f"{source}: last message must be assistant")
            continue
        try:
            plan = json.loads(assistant.get("content", ""))
        except json.JSONDecodeError as e:
            errors.append(f"{source}: assistant content is not JSON plan: {e}")
            continue
        if not isinstance(plan, dict):
            errors.append(f"{source}: plan must be object")
            continue
        errors.extend(validate_plan(plan, source))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    for path in args.paths:
        errors.extend(validate_file(path))
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print(f"validated {len(args.paths)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
