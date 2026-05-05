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
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode prefill-reuse",
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
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode tool-roundtrip",
    },
)

TASK_MIGRATIONS: dict[str, dict[str, Any]] = {
    "rapid-jang-prefill": {
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode prefill-reuse",
    },
    "jang-bridge-loop-guard": {
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode tool-roundtrip",
    },
}


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
    else:
        normalize_results_ledger(root / "results.tsv")
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
        changed = False
        for task in existing:
            migration = TASK_MIGRATIONS.get(str(task.get("id", "")))
            if not migration:
                continue
            for key, value in migration.items():
                if task.get(key) != value:
                    task[key] = value
                    changed = True
        if missing or changed:
            write_jsonl(tasks_path, existing + missing)


def normalize_results_ledger(path: Path) -> None:
    """Keep the ledger parseable while preserving historical evidence.

    Early bootstraps used a 5-column TSV. Current autoresearch uses the
    canonical 12-column ledger. Convert old rows and quarantine malformed rows
    instead of letting future quality scoring read a mixed schema.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        path.write_text(RESULTS_HEADER, encoding="utf-8")
        return
    expected = len(RESULTS_HEADER.rstrip("\n").split("\t"))
    header = lines[0].split("\t")
    converted = [RESULTS_HEADER.rstrip("\n")]
    quarantined: list[str] = []
    changed = header != RESULTS_HEADER.rstrip("\n").split("\t")
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) == expected:
            converted.append(line)
            continue
        if len(fields) == 5:
            timestamp, hypothesis, method, result, verdict = fields
            converted.append(
                "\t".join(
                    clean_tsv(item)
                    for item in [
                        timestamp,
                        f"legacy-{method.lower().replace(' ', '-')}",
                        "keep" if verdict.upper() == "PASS" else "blocked",
                        method,
                        hypothesis,
                        "",
                        "",
                        "",
                        result.removesuffix("s") if result.endswith("s") else result,
                        "",
                        "legacy",
                        f"legacy_result={result} verdict={verdict}",
                    ]
                )
            )
            changed = True
            continue
        quarantined.append(line)
        changed = True
    if not changed:
        return
    backup = path.with_suffix(f".tsv.backup-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.write_text("\n".join(converted) + "\n", encoding="utf-8")
    if quarantined:
        quarantine = path.with_suffix(f".tsv.quarantine-{time.strftime('%Y%m%d-%H%M%S')}")
        quarantine.write_text("\n".join(quarantined) + "\n", encoding="utf-8")


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


def all_result_rows(root: Path) -> list[dict[str, str]]:
    results = root / "results.tsv"
    if not results.exists():
        return []
    headers = RESULTS_HEADER.rstrip("\n").split("\t")
    rows: list[dict[str, str]] = []
    for line in results.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        values = line.split("\t")
        if len(values) != len(headers):
            continue
        rows.append({headers[index]: values[index] for index in range(len(headers))})
    return rows


def result_rows_since(root: Path, before_line_count: int) -> list[dict[str, str]]:
    results = root / "results.tsv"
    if not results.exists():
        return []
    lines = results.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= before_line_count:
        return []
    headers = RESULTS_HEADER.rstrip("\n").split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[max(before_line_count, 1) :]:
        values = line.split("\t")
        if len(values) != len(headers):
            continue
        rows.append({headers[index]: values[index] if index < len(values) else "" for index in range(len(headers))})
    return rows


def parse_float(value: object) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def malformed_result_rows_since(root: Path, before_line_count: int) -> int:
    results = root / "results.tsv"
    if not results.exists():
        return 0
    lines = results.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= before_line_count:
        return 0
    expected = len(RESULTS_HEADER.rstrip("\n").split("\t"))
    return sum(1 for line in lines[max(before_line_count, 1) :] if len(line.split("\t")) != expected)


def task_evidence_rows(root: Path, task: dict[str, Any]) -> list[dict[str, str]]:
    mode = str(task.get("benchmark_mode", ""))
    if not mode:
        return []
    start_line = int(task.get("evidence_start_line") or 0)
    rows = result_rows_since(root, start_line) if start_line else all_result_rows(root)
    return [row for row in rows if row.get("status") == "keep" and row.get("target") == mode]


def claim_task_evidence_window(root: Path, task: dict[str, Any] | None, start_line: int) -> dict[str, Any] | None:
    """Pin task evidence to rows created after the task is first selected.

    Without this, two tasks that share a benchmark mode can accidentally reuse
    old rows and advance without testing their own hypothesis.
    """
    if not task or task.get("status", "ready") not in {"ready", "rework"}:
        return task
    if task.get("evidence_start_line"):
        return task
    next_action = str(task.get("next_action", ""))
    if "openclaw-speed-research benchmark --mode" not in next_action:
        return task
    tasks = read_jsonl(root / "tasks.jsonl")
    changed = False
    updated_task = dict(task)
    for item in tasks:
        if item.get("id") != task.get("id"):
            continue
        item["evidence_start_line"] = start_line
        updated_task = dict(item)
        changed = True
        break
    if changed:
        write_jsonl(root / "tasks.jsonl", tasks)
    return updated_task


def append_strategy_note(root: Path, note: str) -> None:
    path = root / "STRATEGY.md"
    if not path.exists():
        path.write_text(strategy_template(), encoding="utf-8")
    text = path.read_text(encoding="utf-8", errors="replace")
    section = "## Accepted Baselines\n"
    if section not in text:
        text = text.rstrip() + "\n\n" + section + "\n"
    if note not in text:
        text = text.rstrip() + "\n\n" + note + "\n"
        path.write_text(text, encoding="utf-8")


def complete_task_from_evidence(
    root: Path,
    task: dict[str, Any] | None,
    *,
    min_samples: int = 3,
    commit: str = "unknown",
) -> dict[str, Any] | None:
    """Advance benchmark tasks once the supervisor has enough evidence.

    The model should propose ideas; the outer loop owns state transitions. That
    keeps overnight runs from repeating a completed benchmark forever.
    """
    if not task or task.get("status", "ready") not in {"ready", "rework"}:
        return None
    next_action = str(task.get("next_action", ""))
    if "openclaw-speed-research benchmark --mode" not in next_action:
        return None
    rows = task_evidence_rows(root, task)
    if len(rows) < min_samples:
        return None
    tasks = read_jsonl(root / "tasks.jsonl")
    updated = False
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ttft_values = [value for row in rows if (value := parse_float(row.get("ttft_s"))) is not None]
    wall_values = [value for row in rows if (value := parse_float(row.get("wall_s"))) is not None]
    decode_values = [value for row in rows if (value := parse_float(row.get("decode_tps"))) is not None]
    summary = {
        "timestamp": completed_at,
        "task_id": str(task.get("id", "unknown")),
        "status": "baseline-recorded",
        "benchmark_mode": str(task.get("benchmark_mode", "")),
        "sample_count": len(rows),
        "mean_ttft_s": mean(ttft_values),
        "mean_wall_s": mean(wall_values),
        "mean_decode_tps": mean(decode_values),
        "commit": commit,
    }
    for item in tasks:
        if item.get("id") != task.get("id"):
            continue
        if item.get("status") == "done":
            return None
        item["status"] = "done"
        item["completed_at"] = completed_at
        item["sample_count"] = len(rows)
        if summary["mean_ttft_s"] is not None:
            item["mean_ttft_s"] = summary["mean_ttft_s"]
        if summary["mean_wall_s"] is not None:
            item["mean_wall_s"] = summary["mean_wall_s"]
        if summary["mean_decode_tps"] is not None:
            item["mean_decode_tps"] = summary["mean_decode_tps"]
        updated = True
        break
    if not updated:
        return None
    write_jsonl(root / "tasks.jsonl", tasks)
    append_jsonl(root / "experiments.jsonl", summary)
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": completed_at,
            "task_id": summary["task_id"],
            "finding": (
                f"{summary['benchmark_mode']} baseline reached {summary['sample_count']} samples; "
                "supervisor advanced to the next queued task."
            ),
            "evidence": summary,
            "next": "select_next_ready_task",
        },
    )
    metric_bits = []
    if summary["mean_ttft_s"] is not None:
        metric_bits.append(f"mean_ttft_s={summary['mean_ttft_s']}")
    if summary["mean_wall_s"] is not None:
        metric_bits.append(f"mean_wall_s={summary['mean_wall_s']}")
    if summary["mean_decode_tps"] is not None:
        metric_bits.append(f"mean_decode_tps={summary['mean_decode_tps']}")
    metric_text = " ".join(metric_bits) or "metric=recorded"
    append_strategy_note(
        root,
        f"- `{summary['task_id']}`: {summary['sample_count']} samples for "
        f"`{summary['benchmark_mode']}`; {metric_text}.",
    )
    return summary


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
        f"- {task.get('id', 'task')}: type={task.get('task_type', 'benchmark')} "
        f"metric={task.get('metric', 'unknown')} next={task.get('next_action', 'record evidence')}"
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
    malformed_rows = malformed_result_rows_since(root, int(before.get("results_lines", 0)))
    if malformed_rows:
        return {"score": 0, "status": "blocked", "reason": f"malformed results.tsv rows={malformed_rows}"}
    rows = result_rows_since(root, int(before.get("results_lines", 0)))
    targets = {row.get("target", "") for row in rows}
    if targets and targets <= {"quick-benchmark", "quick-health"} and len(progress_reasons) <= 2:
        return {"score": 1, "status": "noise", "reason": "quick health benchmark without comparison or synthesis"}
    if any(reason in progress_reasons for reason in ("repo patch", "findings update", "experiments update")):
        return {"score": 5, "status": "strong", "reason": "implementation or structured evidence artifact"}
    if any(reason in progress_reasons for reason in ("strategy update", "task queue update", "ideas update", "benchmark artifact")):
        return {"score": 3, "status": "useful", "reason": "research artifact recorded"}
    return {"score": 2, "status": "weak", "reason": "durable artifact recorded"}
