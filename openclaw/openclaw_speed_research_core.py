#!/usr/bin/env python3
"""Shared state helpers for OpenClaw speed autoresearch.

The runner should stay small: it launches turns and enforces liveness. This
module owns the durable research memory that makes those turns purposeful.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


RESULTS_HEADER = (
    "timestamp\trun_id\tstatus\ttarget\thypothesis\tttft_s\tprefill_tps\tdecode_tps\t"
    "wall_s\tmemory_gb\tcommit\tnotes\n"
)
BENCHMARK_MANIFEST_VERSION = 1
DEFAULT_BENCHMARK_MANIFEST: dict[str, Any] = {
    "version": BENCHMARK_MANIFEST_VERSION,
    "locked": True,
    "success_schema": {
        "required": ["ok", "model", "mode", "wall_s", "memory_before_mb", "memory_after_mb", "timestamp"],
        "decode-sample": ["completion_tokens", "completion_token_source", "decode_tps"],
    },
    "modes": {
        "quick-health": {
            "prompt": "Reply with exactly: OK",
            "max_tokens": 12,
            "temperature": 0,
            "stream": False,
            "prompt_class": "health",
        },
        "streaming-ttft": {
            "prompt": "Reply with exactly: OK",
            "max_tokens": 12,
            "temperature": 0,
            "stream": True,
            "prompt_class": "health-stream",
        },
        "tool-roundtrip": {
            "prompt": "For OpenClaw speed research, reply with exactly TOOL_ROUNDTRIP_OK and no extra text.",
            "max_tokens": 24,
            "temperature": 0,
            "stream": False,
            "prompt_class": "tool-minimal",
        },
        "decode-sample": {
            "prompt": "Write one compact paragraph about reducing local LLM decode latency. Keep it practical.",
            "max_tokens": 96,
            "temperature": 0,
            "stream": False,
            "prompt_class": "normal-text",
            "requires_usage_completion_tokens": True,
        },
        "prefill-reuse": {
            "prompt": "Reply with one sentence about prefix-cache reuse in local agent harnesses.",
            "max_tokens": 48,
            "temperature": 0,
            "stream": False,
            "prompt_class": "cache-small",
        },
    },
}
DEFAULT_INSIGHT_RUBRIC: dict[str, Any] = {
    "version": 1,
    "required_fields": ["cause", "evidence", "proposed_change", "expected_metric_delta", "risk", "rollback"],
    "min_score_for_candidate": 5,
}
DEFAULT_REPLAY_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "decode-token-source-required",
        "failure": "word-count decode estimates polluted task advancement",
        "guard": "decode-sample evidence must include token_source=usage.completion_tokens",
    },
    {
        "id": "profile-variant-paired-control",
        "failure": "normal decode benchmark falsely stood in for no-drafter or block-size control",
        "guard": "profile variants require paired-control plan and restore_live_profile",
    },
    {
        "id": "malformed-tool-fallback",
        "failure": "malformed tool or hidden output wasted cycles",
        "guard": "supervisor fallback runs deterministic benchmark or synthesis",
    },
    {
        "id": "memory-pressure-breaker",
        "failure": "Metal/Python crash risk during long local 31B turns",
        "guard": "active memory circuit breaker stops unsafe turns",
    },
)

DEFAULT_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "decode-mtp-baseline",
        "status": "ready",
        "priority": 100,
        "lane": "mtp-decode",
        "target": "openclaw/openclaw-jang-vlm-server.py",
        "hypothesis": "The current Gemma 4 JANQ + official 4-bit assistant drafter path needs a fresh decode TPS baseline before any tuning.",
        "metric": "decode_tps",
        "benchmark_mode": "decode-sample",
        "guard_checks": ["memory_ok", "no_reasoning_leak", "no_sse_timeout"],
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
    },
    {
        "id": "mtp-acceptance-log-review",
        "status": "ready",
        "priority": 90,
        "lane": "mtp-decode",
        "target": "openclaw-model-proxy.log",
        "hypothesis": "MTP acceptance and round counts explain whether the drafter is accelerating or adding overhead.",
        "metric": "mean_accept",
        "guard_checks": ["one_narrow_tool", "no_loop", "context_within_limit"],
        "next_action": "tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log",
    },
    {
        "id": "no-drafter-control",
        "status": "ready",
        "priority": 90,
        "lane": "mtp-decode",
        "target": "OPENCLAW_JANG_DRAFT_MODEL",
        "hypothesis": "A no-drafter control is required to prove the assistant drafter improves wall-clock decode TPS on normal prompts.",
        "metric": "speedup_factor",
        "benchmark_mode": "decode-sample",
        "guard_checks": ["memory_ok", "restore_live_profile", "no_model_change"],
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
    },
    {
        "id": "drafter-block-sweep-plan",
        "status": "ready",
        "priority": 78,
        "lane": "mtp-decode",
        "task_type": "supervisor",
        "supervisor_action": "drafter-sweep-run",
        "target": "OPENCLAW_JANG_DRAFT_BLOCK_SIZE",
        "hypothesis": "MTP block size controls the acceptance/overhead tradeoff and must be swept on the same prompt set before promotion.",
        "metric": "decode_tps",
        "benchmark_mode": "decode-sample",
        "guard_checks": ["memory_ok", "restore_live_profile", "same_prompt_set"],
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-sweep-run --blocks 1,2,3,4",
    },
    {
        "id": "janq-dflash-drafter-fit-plan",
        "status": "ready",
        "priority": 86,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "drafter-fit-plan",
        "target": "/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json",
        "hypothesis": "A JANQ-specific DFlash drafter must pass structural, training-data, speed, and safety gates before it can enter normal TUI chat.",
        "metric": "drafter_fit_gate",
        "guard_checks": ["no_model_load", "no_opencode_changes", "normal_tui_default_remains_mtp"],
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-drafter-fit plan",
    },
    {
        "id": "drafter-calibration-review",
        "status": "ready",
        "priority": 84,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "focused-test",
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "hypothesis": "JANQ-specific calibration should only be promoted if it improves decode TPS over the official 4-bit drafter.",
        "metric": "acceptance_delta",
        "guard_checks": ["tests_pass", "memory_ok", "no_model_change"],
        "next_action": "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py",
    },
    {
        "id": "mtp-loop-overhead-map",
        "status": "ready",
        "priority": 76,
        "lane": "runtime-overhead",
        "target": "openclaw/openclaw-jang-vlm-server.py",
        "hypothesis": "Reaching 30+ tok/s likely requires reducing MTP verification/cache/rollback overhead after block-size tuning converges.",
        "metric": "decode_tps_delta",
        "guard_checks": ["one_narrow_tool", "no_live_profile_change", "tests_before_patch"],
        "next_action": "read exactly /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/openclaw-jang-vlm-server.py and identify the MTP loop boundaries",
    },
)

TASK_MIGRATIONS: dict[str, dict[str, Any]] = {
    "drafter-block-sweep-plan": {
        "status": "ready",
        "task_type": "supervisor",
        "supervisor_action": "drafter-sweep-run",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-sweep-run --blocks 1,2,3,4",
        "blocked_reason": "",
        "supervisor_summary": {},
    },
    "drafter-calibration-review": {
        "status": "ready",
        "task_type": "supervisor",
        "supervisor_action": "focused-test",
    },
    "implement-mtp-acceptance-report": {
        "status": "ready",
        "task_type": "supervisor",
        "supervisor_action": "mtp-report",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research mtp-report --lines 160",
        "blocked_reason": "",
        "supervisor_summary": {},
    },
    "implement-drafter-sweep-plan": {
        "status": "ready",
        "task_type": "supervisor",
        "supervisor_action": "drafter-sweep-run",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-sweep-run --blocks 1,2,3,4",
        "blocked_reason": "",
        "supervisor_summary": {},
    },
    "implement-janq-drafter-calibration-gate": {
        "status": "blocked",
        "blocked_reason": "superseded by supervisor-owned JANQ drafter-fit and calibration gates",
    },
    "dflash-janq-compatibility-spike": {
        "status": "blocked",
        "blocked_reason": "superseded by supervisor-owned JANQ DFlash drafter-fit plan",
    },
    "baseline-streaming-ttft": {
        "status": "blocked",
        "blocked_reason": "superseded by decode-mtp-baseline for current MTP decode research",
    },
    "tool-roundtrip-overhead": {
        "status": "blocked",
        "blocked_reason": "superseded by MTP acceptance/decode tasks",
    },
    "prompt-size-pressure": {
        "status": "blocked",
        "blocked_reason": "superseded by decode-first MTP research",
    },
    "rapid-jang-prefill": {
        "status": "blocked",
        "blocked_reason": "prefill is secondary to current decode TPS objective",
    },
    "jang-bridge-loop-guard": {
        "status": "blocked",
        "blocked_reason": "loop guards remain important but are not the current decode metric task",
    },
    "post-compact-prompt-shape": {
        "status": "blocked",
        "blocked_reason": "superseded by decode/MTP acceptance research",
    },
    "post-compact-prompt-size": {
        "status": "blocked",
        "blocked_reason": "superseded by decode/MTP acceptance research",
    },
    "prompt-shape-report": {
        "status": "blocked",
        "blocked_reason": "superseded by MTP acceptance report",
    },
    "prompt-size-after-synthesis": {
        "status": "blocked",
        "blocked_reason": "superseded by decode/MTP acceptance research",
    },
    "streaming-ttft-post-synthesis": {
        "status": "blocked",
        "blocked_reason": "superseded by decode/MTP acceptance research",
    },
    "implement-prompt-shape-compaction": {
        "status": "blocked",
        "blocked_reason": "superseded by decode/MTP implementation candidates",
    },
    "implement-rapid-profile-bandit-plan": {
        "status": "blocked",
        "blocked_reason": "superseded by drafter sweep plan",
    },
    "implement-speculative-pld-compat-probe": {
        "status": "blocked",
        "blocked_reason": "superseded by live MTP drafter and JANQ calibration gate",
    },
    "gemma4-mtp-drafter-compatibility": {
        "status": "blocked",
        "blocked_reason": "superseded by live Gemma 4 assistant drafter baseline and decode/MTP tasks",
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
    write_json_if_missing_or_stale(root / "benchmark-manifest.json", DEFAULT_BENCHMARK_MANIFEST, "version")
    write_json_if_missing_or_stale(root / "insight-rubric.json", DEFAULT_INSIGHT_RUBRIC, "version")
    ensure_replay_buffer(root / "replay-buffer.jsonl")
    tasks_path = root / "tasks.jsonl"
    if not tasks_path.exists() or tasks_path.stat().st_size == 0:
        write_jsonl(tasks_path, [dict(task) for task in DEFAULT_TASKS])
    else:
        existing = read_jsonl(tasks_path)
        existing_ids = {str(task.get("id", "")) for task in existing}
        missing = [dict(task) for task in DEFAULT_TASKS if str(task["id"]) not in existing_ids]
        changed = False
        for task in existing:
            task_id = str(task.get("id", ""))
            if task_id.startswith("implementation-bridge-") and task.get("status", "ready") in {"ready", "rework"}:
                bridge_update = {
                    "task_type": "supervisor",
                    "supervisor_action": "implementation-bridge",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research synthesize --kind frontier",
                    "blocked_reason": "",
                    "supervisor_summary": {},
                }
                for key, value in bridge_update.items():
                    if task.get(key) != value:
                        task[key] = value
                        changed = True
            migration = TASK_MIGRATIONS.get(str(task.get("id", "")))
            if not migration:
                continue
            for key, value in migration.items():
                if key == "status" and value == "ready" and task.get("status") not in {"ready", "rework"}:
                    continue
                if task.get(key) != value:
                    task[key] = value
                    changed = True
        if missing or changed:
            write_jsonl(tasks_path, existing + missing)


def write_json_if_missing_or_stale(path: Path, payload: dict[str, Any], version_key: str) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if isinstance(existing, dict) and existing.get(version_key) == payload.get(version_key):
            return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_replay_buffer(path: Path) -> None:
    existing_ids = {str(row.get("id", "")) for row in read_jsonl(path)}
    additions = [dict(row) for row in DEFAULT_REPLAY_CASES if str(row["id"]) not in existing_ids]
    if additions:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            for row in additions:
                file.write(json.dumps(row, sort_keys=True) + "\n")


def load_benchmark_manifest(root: Path) -> dict[str, Any]:
    ensure_research_state(root)
    path = root / "benchmark-manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = DEFAULT_BENCHMARK_MANIFEST
    return value if isinstance(value, dict) else DEFAULT_BENCHMARK_MANIFEST


def benchmark_spec(root: Path, mode: str) -> dict[str, Any]:
    manifest = load_benchmark_manifest(root)
    modes = manifest.get("modes", {})
    if not isinstance(modes, dict) or mode not in modes:
        mode = "quick-health"
    spec = modes.get(mode, {})
    if not isinstance(spec, dict):
        spec = DEFAULT_BENCHMARK_MANIFEST["modes"]["quick-health"]
    return {**spec, "mode": mode, "manifest_version": manifest.get("version", "unknown")}


def benchmark_result_schema_ok(root: Path, result: dict[str, Any]) -> tuple[bool, str]:
    manifest = load_benchmark_manifest(root)
    schema = manifest.get("success_schema", {})
    required = list(schema.get("required", [])) if isinstance(schema, dict) else []
    missing = [key for key in required if key not in result]
    mode = str(result.get("mode", ""))
    if isinstance(schema, dict):
        missing.extend(key for key in schema.get(mode, []) if key not in result)
    if missing:
        return False, f"missing result keys: {','.join(sorted(set(missing)))}"
    if mode == "decode-sample" and result.get("completion_token_source") != "usage.completion_tokens":
        return False, "decode-sample missing usage.completion_tokens source"
    if mode == "decode-sample":
        try:
            completion_tokens = int(result.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            completion_tokens = 0
        if completion_tokens < 64:
            return False, f"decode-sample too short for throughput evidence: completion_tokens={completion_tokens}"
    return True, ""


def score_insight(idea: dict[str, Any], rubric: dict[str, Any] | None = None) -> dict[str, Any]:
    active = rubric or DEFAULT_INSIGHT_RUBRIC
    required = [str(item) for item in active.get("required_fields", [])]
    missing = [field for field in required if not str(idea.get(field, "")).strip()]
    evidence_text = str(idea.get("evidence", ""))
    metric_text = str(idea.get("expected_metric_delta", ""))
    risk_text = str(idea.get("risk", ""))
    score = len(required) - len(missing)
    if any(token in evidence_text for token in ("tok/s", "mean_accept", "decode", "benchmark", "completion_tokens")):
        score += 1
    if any(token in metric_text for token in ("tok/s", "%", "decode_tps", "mean_accept")):
        score += 1
    if risk_text and str(idea.get("rollback", "")).strip():
        score += 1
    threshold = int(active.get("min_score_for_candidate", 5))
    return {
        "score": score,
        "threshold": threshold,
        "passed": score >= threshold and not missing,
        "missing": missing,
    }


def paired_profile_plan(task: dict[str, Any]) -> dict[str, Any]:
    target = str(task.get("target", "profile-variant"))
    return {
        "task_id": str(task.get("id", "unknown")),
        "target": target,
        "status": "plan-only",
        "control": {
            "label": "current-live-profile",
            "restore_before": True,
            "benchmark_mode": str(task.get("benchmark_mode", "decode-sample")),
            "samples": 3,
        },
        "variant": {
            "label": target,
            "set_env": {target: "<candidate-value>"},
            "benchmark_mode": str(task.get("benchmark_mode", "decode-sample")),
            "samples": 3,
        },
        "promotion_gate": {
            "min_decode_tps_delta": 0.5,
            "must_restore_live_profile": True,
            "must_pass_replay": True,
            "must_keep_model_id": True,
        },
        "rollback": "restore live OpenClaw model profile/env override before any further task",
    }


def replay_checks(root: Path) -> dict[str, Any]:
    ensure_research_state(root)
    rows = all_result_rows(root)
    def note_completion_tokens(row: dict[str, str]) -> int:
        match = re.search(r"completion_tokens=(\d+)", row.get("notes", ""))
        return int(match.group(1)) if match else 0

    bad_decode = [
        row.get("run_id", "")
        for row in rows
        if row.get("target") == "decode-sample"
        and row.get("status") == "keep"
        and row.get("decode_tps")
        and "completion_tokens=" in row.get("notes", "")
        and "token_source=usage.completion_tokens" not in row.get("notes", "")
    ]
    short_decode = [
        row.get("run_id", "")
        for row in rows
        if row.get("target") == "decode-sample"
        and row.get("status") == "keep"
        and "completion_tokens=" in row.get("notes", "")
        and note_completion_tokens(row) < 64
    ]
    legacy_decode = [
        row.get("run_id", "")
        for row in rows
        if row.get("target") == "decode-sample"
        and row.get("status") == "keep"
        and row.get("decode_tps")
        and "completion_tokens=" not in row.get("notes", "")
    ]
    tasks = read_jsonl(root / "tasks.jsonl")
    unsafe_profile_tasks = [
        str(task.get("id", ""))
        for task in tasks
        if task.get("status", "ready") in {"ready", "rework"}
        and str(task.get("target", "")).startswith("OPENCLAW_JANG_DRAFT_")
        and "restore_live_profile" not in {str(item) for item in task.get("guard_checks", [])}
    ]
    passed = not bad_decode and not short_decode and not unsafe_profile_tasks
    return {
        "ok": passed,
        "bad_decode_rows": bad_decode,
        "short_decode_rows": short_decode,
        "legacy_decode_rows_ignored": legacy_decode,
        "unsafe_profile_tasks": unsafe_profile_tasks,
        "cases": [row["id"] for row in DEFAULT_REPLAY_CASES],
    }


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

Objective: improve real decode tokens/sec for the current Gemma 4 31B JANG/JANQ OpenClaw setup with the Gemma 4 MTP assistant drafter, while treating crashes, loops, memory pressure, and tool failures as hard guards.

## Current Best Understanding

- Live baseline is the official quantized Gemma 4 assistant drafter at block size 2.
- Recent bounded decode measurements are about 14-15 tok/s, versus roughly 12.5 tok/s without a drafter.
- Earlier heuristic MTP scheduling, 3-bit drafter experiments, and pre-projection-only calibration did not beat the official 4-bit drafter.

## Top Hypotheses

1. Decode TPS will improve only if MTP acceptance rises enough to beat drafter overhead on normal prompts.
2. The best next experiments are no-drafter control, block-size sweep, drafter quantization/calibration, and log-based `mean_accept` analysis.
3. Rapid-MLX or MLX/VLM loop changes matter only if they reduce verification/drafter overhead without changing the selected target model.

## 30 Tok/S Ladder

1. Lock the honest baseline: live MTP, no-drafter control, and block-size sweep on the same prompt set.
2. Stop repeating exhausted block-size sweeps once block 2 remains the winner; move to acceptance and overhead diagnostics.
3. Raise acceptance: JANQ-specific drafter fit, calibration, quantization, and prompt-class-specific rejection analysis.
4. Reduce overhead: inspect MTP verify/cache/rollback loop boundaries and propose only small tested patches.
5. Explore step-change paths: DFlash compatibility and Rapid/MLX upstream gaps, but only after structural checks and replay guards.
6. Promote nothing to normal TUI chat unless paired benchmarks improve decode TPS and tool/thinking/stream/memory guards pass.

## Rejected Or Exhausted

- Repeating TTFT, tool-roundtrip, or prompt-size benchmarks without a decode/MTP hypothesis is noise for this phase.
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
    rows = [row for row in rows if row.get("status") == "keep" and row.get("target") == mode]
    if mode == "decode-sample":
        rows = [row for row in rows if "token_source=usage.completion_tokens" in row.get("notes", "")]
    return rows


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
