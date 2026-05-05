#!/usr/bin/env python3
"""Shared state helpers for OpenClaw speed autoresearch.

The runner should stay small: it launches turns and enforces liveness. This
module owns the durable research memory that makes those turns purposeful.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


RESULTS_HEADER = (
    "timestamp\trun_id\tstatus\ttarget\thypothesis\tttft_s\tprefill_tps\tdecode_tps\t"
    "wall_s\tmemory_gb\tcommit\tnotes\n"
)

DEFAULT_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "baseline-streaming-ttft",
        "status": "ready",
        "priority": 100,
        "lane": "current-stack",
        "target": "openclaw-model-proxy",
        "hypothesis": "Streaming TTFT and first useful status are the best first speed bottleneck signals.",
        "metric": "ttft_s",
        "benchmark_mode": "streaming-ttft",
        "guard_checks": ["no_sse_timeout", "no_reasoning_leak", "memory_ok"],
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode streaming-ttft",
    },
    {
        "id": "tool-roundtrip-overhead",
        "status": "ready",
        "priority": 90,
        "lane": "current-stack",
        "target": "tool-call-path",
        "hypothesis": "Tool-call round trips expose prompt/tool schema overhead better than tiny health probes.",
        "metric": "tool_roundtrip_s",
        "benchmark_mode": "tool-roundtrip",
        "guard_checks": ["one_narrow_tool", "no_loop", "context_within_limit"],
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode tool-roundtrip",
    },
    {
        "id": "prompt-size-pressure",
        "status": "ready",
        "priority": 85,
        "lane": "current-stack",
        "target": "prompt-context",
        "hypothesis": "Reducing stable prompt/tool context is likely to improve prefill and avoid preflight blocks.",
        "metric": "estimated_prompt_tokens",
        "benchmark_mode": "prompt-size",
        "guard_checks": ["semantic_preservation", "no_tool_regression"],
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode prompt-size",
    },
    {
        "id": "rapid-jang-prefill",
        "status": "ready",
        "priority": 80,
        "lane": "current-stack",
        "target": "openclaw/openclaw-rapid-launcher.py",
        "hypothesis": "One Rapid-MLX prefill/cache knob can reduce first-token latency without memory regression.",
        "metric": "prefill_or_wall_s",
        "benchmark_mode": "prefill-reuse",
        "guard_checks": ["tests_pass", "memory_ok", "no_crash"],
        "next_action": "read /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/openclaw-rapid-launcher.py",
    },
    {
        "id": "jang-bridge-loop-guard",
        "status": "ready",
        "priority": 70,
        "lane": "current-stack",
        "target": "openclaw/rapid-overlay/openclaw_rapid_jang.py",
        "hypothesis": "Gemma4 JANG loop behavior must stay guarded while Rapid-MLX owns the fast serving path.",
        "metric": "loop_or_reasoning_leak_count",
        "benchmark_mode": "tool-roundtrip",
        "guard_checks": ["no_thought_loop", "no_malformed_tool_json", "tests_pass"],
        "next_action": "read /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/rapid-overlay/openclaw_rapid_jang.py",
    },
)


def clean_tsv(value: object) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8", errors="replace") == text:
        return
    path.write_text(text, encoding="utf-8")


def ensure_research_state(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmarks").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "experiments").mkdir(exist_ok=True)
    if not (root / "results.tsv").exists():
        (root / "results.tsv").write_text(RESULTS_HEADER, encoding="utf-8")
    if not (root / "STRATEGY.md").exists():
        (root / "STRATEGY.md").write_text(strategy_template(), encoding="utf-8")
    for name in ("findings.jsonl", "experiments.jsonl", "rejections.jsonl"):
        path = root / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    tasks_path = root / "tasks.jsonl"
    if not tasks_path.exists() or tasks_path.stat().st_size == 0:
        write_jsonl(tasks_path, [dict(task) for task in DEFAULT_TASKS])
    else:
        existing = read_jsonl(tasks_path)
        existing_ids = {str(task.get("id", "")) for task in existing}
        missing = [dict(task) for task in DEFAULT_TASKS if str(task["id"]) not in existing_ids]
        if missing:
            write_jsonl(tasks_path, existing + missing)


def strategy_template() -> str:
    return """# OpenClaw Speed Strategy

Objective: optimize raw speed for the current Gemma 4 31B JANG OpenClaw setup while treating crashes, loops, memory pressure, and tool failures as hard guards.

## Current Best Understanding

- No accepted speed improvement yet in this strategy file.

## Top Hypotheses

1. Measure streaming TTFT and first useful status before changing knobs.
2. Measure tool-call round trip because it represents real agent UX better than tiny health prompts.
3. Reduce prompt/context overhead only if semantic and tool behavior guards hold.

## Rejected Or Exhausted

- Repeating quick-health benchmarks without a changed hypothesis is noise, not research progress.
"""


def append_result(
    root: Path,
    *,
    run_id: str,
    status: str,
    target: str,
    hypothesis: str,
    commit: str,
    notes: str,
    ttft_s: object = "",
    prefill_tps: object = "",
    decode_tps: object = "",
    wall_s: object = "",
    memory_gb: object = "",
) -> None:
    results = root / "results.tsv"
    if not results.exists():
        results.write_text(RESULTS_HEADER, encoding="utf-8")
    row = [
        time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        run_id,
        status,
        target,
        hypothesis,
        ttft_s,
        prefill_tps,
        decode_tps,
        wall_s,
        memory_gb,
        commit,
        notes,
    ]
    with results.open("a", encoding="utf-8") as file:
        file.write("\t".join(clean_tsv(item) for item in row) + "\n")


def result_rows_since(root: Path, before_line_count: int) -> list[dict[str, str]]:
    results = root / "results.tsv"
    if not results.exists():
        return []
    lines = results.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= before_line_count:
        return []
    headers = lines[0].split("\t") if lines else []
    rows: list[dict[str, str]] = []
    for line in lines[max(before_line_count, 1) :]:
        values = line.split("\t")
        rows.append({headers[index]: values[index] if index < len(values) else "" for index in range(len(headers))})
    return rows


def select_next_task(root: Path) -> dict[str, Any] | None:
    tasks = [task for task in read_jsonl(root / "tasks.jsonl") if task.get("status", "ready") in {"ready", "rework"}]
    if not tasks:
        return None
    return sorted(tasks, key=lambda task: int(task.get("priority", 0)), reverse=True)[0]


def task_summary(root: Path, limit: int = 3) -> str:
    tasks = [
        task
        for task in read_jsonl(root / "tasks.jsonl")
        if task.get("status", "ready") in {"ready", "in_progress", "rework"}
    ]
    tasks = sorted(tasks, key=lambda task: int(task.get("priority", 0)), reverse=True)[:limit]
    return "\n".join(
        f"- {task.get('id', 'task')}: metric={task.get('metric', 'unknown')} next={task.get('next_action', 'record evidence')}"
        for task in tasks
    )


def record_rejection(root: Path, *, cycle: int, task_id: str, reason: str, evidence: str) -> None:
    append_jsonl(
        root / "rejections.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cycle": cycle,
            "task_id": task_id,
            "reason": reason,
            "evidence": evidence,
        },
    )


def cycle_quality(
    root: Path,
    before: dict[str, Any],
    after: dict[str, Any],
    progress_reasons: list[str],
    issue: str,
) -> dict[str, Any]:
    if not progress_reasons:
        return {"score": 0, "status": "blocked", "reason": issue or "no durable artifact"}
    rows = result_rows_since(root, int(before.get("results_lines", 0)))
    targets = {row.get("target", "") for row in rows}
    if targets and targets <= {"quick-benchmark", "quick-health"} and len(progress_reasons) <= 2:
        return {"score": 1, "status": "noise", "reason": "quick health benchmark without comparison or synthesis"}
    if any(reason in progress_reasons for reason in ("repo patch", "findings update", "experiments update")):
        return {"score": 5, "status": "strong", "reason": "implementation or structured evidence artifact"}
    if any(reason in progress_reasons for reason in ("strategy update", "task queue update", "ideas update", "benchmark artifact")):
        return {"score": 3, "status": "useful", "reason": "research artifact recorded"}
    return {"score": 2, "status": "weak", "reason": "durable artifact recorded"}
