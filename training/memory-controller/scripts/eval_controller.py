#!/usr/bin/env python3
"""Evaluate a memory-controller model against JSONL cases.

Supports Ollama's /api/chat endpoint by default. The model must answer with a
MemoryControllerPlan JSON object.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


SYSTEM = "Emit only a MemoryControllerPlan JSON object."


def call_ollama(base_url: str, model: str, prompt: str) -> dict:
    url = base_url.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data.get("message", {}).get("content", "")
    return json.loads(content)


def check_case(case: dict, plan: dict) -> list[str]:
    errors: list[str] = []
    expected = case.get("expected", {})

    for key, op_key in (("search_count", "search"), ("store_count", "store"), ("update_count", "update"), ("forget_count", "forget")):
        if key in expected and len(plan.get(op_key, [])) != expected[key]:
            errors.append(f"{op_key} count expected {expected[key]}, got {len(plan.get(op_key, []))}")

    if "search_contains" in expected:
        haystack = " ".join(op.get("query", "") for op in plan.get("search", []))
        for term in expected["search_contains"]:
            if term.lower() not in haystack.lower():
                errors.append(f"search query missing {term!r}")

    for i, want in enumerate(expected.get("store", [])):
        stores = plan.get("store", [])
        if not stores:
            errors.append("expected at least one store op")
            continue
        op = stores[min(i, len(stores) - 1)]
        if want.get("kind") and op.get("kind") != want["kind"]:
            errors.append(f"store[{i}].kind expected {want['kind']!r}, got {op.get('kind')!r}")
        if want.get("scope") and op.get("scope", "thread") != want["scope"]:
            errors.append(f"store[{i}].scope expected {want['scope']!r}, got {op.get('scope')!r}")
        for term in want.get("content_contains", []):
            if term.lower() not in op.get("content", "").lower():
                errors.append(f"store[{i}].content missing {term!r}")
        if op.get("importance", 0) < want.get("min_importance", 0):
            errors.append(f"store[{i}].importance below {want['min_importance']}")

    for i, want in enumerate(expected.get("update", [])):
        updates = plan.get("update", [])
        if not updates:
            errors.append("expected at least one update op")
            continue
        op = updates[min(i, len(updates) - 1)]
        if want.get("memoryId") and op.get("memoryId") != want["memoryId"]:
            errors.append(f"update[{i}].memoryId expected {want['memoryId']!r}, got {op.get('memoryId')!r}")
        if want.get("kind") and op.get("kind") != want["kind"]:
            errors.append(f"update[{i}].kind expected {want['kind']!r}, got {op.get('kind')!r}")
        if want.get("scope") and op.get("scope") != want["scope"]:
            errors.append(f"update[{i}].scope expected {want['scope']!r}, got {op.get('scope')!r}")
        for term in want.get("content_contains", []):
            if term.lower() not in op.get("content", "").lower():
                errors.append(f"update[{i}].content missing {term!r}")

    for i, want in enumerate(expected.get("forget", [])):
        forgets = plan.get("forget", [])
        if not forgets:
            errors.append("expected at least one forget op")
            continue
        op = forgets[min(i, len(forgets) - 1)]
        if want.get("memoryId") and op.get("memoryId") != want["memoryId"]:
            errors.append(f"forget[{i}].memoryId expected {want['memoryId']!r}, got {op.get('memoryId')!r}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--cases", type=Path, default=Path("training/memory-controller/eval-cases.jsonl"))
    args = parser.parse_args()

    failures = 0
    total = 0
    for raw in args.cases.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        total += 1
        case = json.loads(raw)
        try:
            plan = call_ollama(args.base_url, args.model, case["input"])
            errors = check_case(case, plan)
        except Exception as e:  # noqa: BLE001 - CLI diagnostic
            errors = [f"controller call failed: {e}"]
        if errors:
            failures += 1
            print(f"FAIL {case['name']}: {'; '.join(errors)}", file=sys.stderr)
        else:
            print(f"PASS {case['name']}")

    print(f"{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
