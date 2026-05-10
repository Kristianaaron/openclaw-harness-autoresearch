#!/usr/bin/env python3
"""Deterministic health/quality reviewer for OpenClaw autoresearch.

This is intentionally a sidecar. It reads the research workspace and writes
watchdog artifacts, but it does not mutate `tasks.jsonl` while autopilot owns
the workspace lock. That keeps the hot loop stable and gives the user a
Codex-like reviewer process that can run from cron/LaunchAgent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_WORKSPACE = Path.home() / ".openclaw" / "research" / "speed"
DEFAULT_TARGET_TPS = 30.0


def workspace_root() -> Path:
    return Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_DIR", DEFAULT_WORKSPACE)).expanduser()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def latest_json_artifact(root: Path, pattern: str) -> dict[str, Any]:
    paths = sorted((root / "benchmarks").glob(pattern), key=lambda item: item.stat().st_mtime_ns)
    if not paths:
        return {}
    payload = read_json(paths[-1])
    payload["_artifact_path"] = str(paths[-1])
    payload["_artifact_mtime"] = paths[-1].stat().st_mtime
    return payload


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def autopilot_lock(root: Path) -> dict[str, Any]:
    lock = root / "autopilot.lock"
    payload = read_json(lock)
    pid = int(payload.get("pid", 0) or 0)
    payload["path"] = str(lock)
    payload["active"] = process_alive(pid)
    return payload


def latest_autopilot_log(root: Path) -> dict[str, Any]:
    logs = sorted((root / "logs").glob("autopilot-*.log"), key=lambda item: item.stat().st_mtime_ns)
    if not logs:
        logs = sorted((root / "logs").glob("autopilot-speed-research-*.log"), key=lambda item: item.stat().st_mtime_ns)
    if not logs:
        return {}
    path = logs[-1]
    stat = path.stat()
    tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
    return {
        "path": str(path),
        "mtime": stat.st_mtime,
        "age_seconds": round(max(0.0, time.time() - stat.st_mtime), 3),
        "size": stat.st_size,
        "tail": tail,
    }


def result_rows(root: Path) -> list[dict[str, str]]:
    path = root / "results.tsv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
            fieldnames=[
                "timestamp",
                "run_id",
                "status",
                "target",
                "hypothesis",
                "ttft_s",
                "prefill_tps",
                "decode_tps",
                "wall_s",
                "memory_gb",
                "commit",
                "notes",
            ],
        )
        return [row for row in reader if row.get("timestamp") != "timestamp"]


def parse_note_fields(notes: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in str(notes).split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip().strip(",")
    return fields


def parse_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def recent_rows(root: Path, limit: int) -> list[dict[str, str]]:
    rows = result_rows(root)
    return rows[-max(1, limit) :]


def latest_decode_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    values: list[float] = []
    server_values: list[float] = []
    for row in rows:
        if row.get("status") != "keep" or row.get("target") != "decode-sample":
            continue
        value = parse_float(row.get("decode_tps"))
        if value is not None:
            values.append(value)
        server_value = parse_float(parse_note_fields(row.get("notes", "")).get("server_tok_s"))
        if server_value is not None:
            server_values.append(server_value)
    return {
        "samples": len(values),
        "mean_wall_decode_tps": round(sum(values) / len(values), 3) if values else None,
        "max_wall_decode_tps": round(max(values), 3) if values else None,
        "mean_server_decode_tps": round(sum(server_values) / len(server_values), 3) if server_values else None,
        "max_server_decode_tps": round(max(server_values), 3) if server_values else None,
    }


def latest_row_time(rows: list[dict[str, str]]) -> float | None:
    for row in reversed(rows):
        raw = row.get("timestamp", "")
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt).timestamp()
            except ValueError:
                continue
    return None


def count_recent_terminal_noise(rows: list[dict[str, str]]) -> dict[str, int]:
    terminal = 0
    bridge_zero = 0
    blocked = 0
    memory = 0
    for row in rows:
        notes = row.get("notes", "")
        target = row.get("target", "")
        run_id = row.get("run_id", "")
        if target == "synthesis-terminal" or "terminal_no_work=True" in notes:
            terminal += 1
        if run_id.startswith("supervisor-implementation-bridge-") and "ready_deterministic=0" in notes:
            bridge_zero += 1
        if row.get("status") == "blocked":
            blocked += 1
        if row.get("status") == "blocked" and any(fragment in notes.lower() for fragment in ("memory", "metal", "compressor", "swap")):
            memory += 1
    return {
        "terminal_synthesis_rows": terminal,
        "bridge_zero_rows": bridge_zero,
        "blocked_rows": blocked,
        "memory_block_rows": memory,
    }


def watchdog_review(
    root: Path,
    *,
    recent: int,
    target_tps: float,
    min_quality: float,
    min_scorecard: float,
    min_frontier: float,
    max_log_stale_seconds: float,
    max_result_stale_seconds: float,
) -> dict[str, Any]:
    rows = recent_rows(root, recent)
    lock = autopilot_lock(root)
    log = latest_autopilot_log(root)
    quality = latest_json_artifact(root, "quality-review-*.json")
    frontier = latest_json_artifact(root, "frontier-system-eval-*.json")
    handoff = latest_json_artifact(root, "implementation-handoff-audit-*.json")
    burn_in = latest_json_artifact(root, "stability-burn-in-*.json")
    canonical = frontier.get("canonical_state") if isinstance(frontier.get("canonical_state"), dict) else {}
    noise = canonical.get("noise") if isinstance(canonical.get("noise"), dict) else {}
    scorecard = quality.get("scorecard") if isinstance(quality.get("scorecard"), dict) else {}
    latest_time = latest_row_time(rows)
    result_age = round(max(0.0, time.time() - latest_time), 3) if latest_time else None
    decode = latest_decode_summary(rows)
    direct_noise = count_recent_terminal_noise(rows)
    active_noise = {
        "unresolved_blocked_rows": int(noise.get("unresolved_blocked_rows", 0) or 0),
        "terminal_synthesis_rows": int(noise.get("terminal_synthesis_rows", 0) or 0),
        "bridge_zero_rows": int(noise.get("bridge_zero_rows", 0) or 0),
        "memory_blocks": int(noise.get("memory_blocks", 0) or 0),
    }
    quality_score = parse_float(quality.get("quality_score"))
    scorecard_overall = parse_float(scorecard.get("overall"))
    frontier_overall = parse_float(frontier.get("overall"))
    handoff_score = parse_float(handoff.get("score"))
    gates = {
        "autopilot_lock_active_or_cleanly_absent": bool(lock.get("active")) or not (root / "autopilot.lock").exists(),
        "log_fresh": bool(log) and float(log.get("age_seconds", 999999)) <= max_log_stale_seconds,
        "results_fresh": result_age is not None and result_age <= max_result_stale_seconds,
        "quality_healthy": quality.get("verdict") == "healthy",
        "quality_score_high": quality_score is not None and quality_score >= min_quality,
        "scorecard_high": scorecard_overall is not None and scorecard_overall >= min_scorecard,
        "handoff_high": handoff_score is not None and handoff_score >= 95,
        "frontier_high": frontier_overall is not None and frontier_overall >= min_frontier,
        "zero_active_noise": all(value == 0 for value in active_noise.values()),
        "no_recent_memory_blocks": direct_noise["memory_block_rows"] == 0,
    }
    blockers = [name for name, ok in gates.items() if not ok]
    decode_mean = decode.get("mean_wall_decode_tps")
    canonical_state = str(canonical.get("state", "unknown"))
    ready_tasks = canonical.get("deterministic_ready_tasks", [])

    decision = "continue"
    severity = "healthy"
    next_command = ""
    reason = "all deterministic reviewer gates passed"
    if "log_fresh" in blockers or "results_fresh" in blockers:
        decision = "investigate-stall"
        severity = "critical"
        next_command = "tail -120 ~/.openclaw/research/speed/logs/$(ls -t ~/.openclaw/research/speed/logs/autopilot-*.log | head -1)"
        reason = "autopilot output stopped advancing inside the watchdog freshness window"
    elif "zero_active_noise" in blockers or scorecard_overall is not None and scorecard_overall < min_scorecard:
        decision = "repair-routing"
        severity = "degraded"
        next_command = "~/.openclaw/bin/openclaw-speed-research quality-review --recent-rows 120 --min-sweeps 3 --min-samples-per-block 3 --target-tps 30"
        reason = "quality/noise gate failed; route one deterministic repair before more research"
    elif frontier_overall is not None and frontier_overall < min_frontier:
        decision = "frontier-repair"
        severity = "degraded"
        next_command = "~/.openclaw/bin/openclaw-speed-research frontier-eval --recent-rows 120 --allow-fail"
        reason = "frontier eval fell below the configured floor"
    elif canonical_state in {"blocked_until_external_change", "plateau_detected"} and not ready_tasks:
        decision = "seed-next-candidate"
        severity = "attention"
        next_command = "~/.openclaw/bin/openclaw-speed-research synthesize --kind frontier"
        reason = "the loop has no deterministic ready work; seed exactly one next candidate path"
    elif decode_mean is not None and decode_mean < target_tps:
        decision = "continue-breakthrough-lane"
        severity = "healthy"
        next_command = "continue current autopilot; only promote paired decode improvements"
        reason = "decode is below target but the loop remains healthy and has deterministic work"

    return {
        "ok": severity == "healthy",
        "kind": "autoresearch-watchdog",
        "timestamp": int(time.time()),
        "workspace": str(root),
        "severity": severity,
        "decision": decision,
        "reason": reason,
        "next_command": next_command,
        "gates": gates,
        "blockers": blockers,
        "autopilot": lock,
        "latest_log": {key: value for key, value in log.items() if key != "tail"},
        "latest_log_tail": log.get("tail", [])[-20:],
        "result_age_seconds": result_age,
        "quality": {
            "artifact": quality.get("_artifact_path", ""),
            "verdict": quality.get("verdict", ""),
            "quality_score": quality_score,
            "scorecard_overall": scorecard_overall,
        },
        "frontier": {
            "artifact": frontier.get("_artifact_path", ""),
            "overall": frontier_overall,
            "readiness": frontier.get("readiness", ""),
            "certified": bool(frontier.get("frontier_certified")),
            "gaps": frontier.get("gaps", []),
        },
        "handoff": {
            "artifact": handoff.get("_artifact_path", ""),
            "score": handoff_score,
            "ok": bool(handoff.get("ok")),
        },
        "stability_burn_in": {
            "artifact": burn_in.get("_artifact_path", ""),
            "ok": bool(burn_in.get("ok")),
        },
        "canonical_state": {
            "state": canonical_state,
            "clean": bool(canonical.get("clean")),
            "ready_tasks": canonical.get("ready_tasks"),
            "deterministic_ready_tasks": ready_tasks,
            "breakthrough_lanes": canonical.get("breakthrough_lanes", []),
            "exhausted_lanes": canonical.get("exhausted_lanes", []),
            "noise": active_noise,
        },
        "direct_recent_noise": direct_noise,
        "decode": decode,
    }


def run_once(args: argparse.Namespace) -> int:
    root = workspace_root()
    report = watchdog_review(
        root,
        recent=args.recent_rows,
        target_tps=args.target_tps,
        min_quality=args.min_quality,
        min_scorecard=args.min_scorecard,
        min_frontier=args.min_frontier,
        max_log_stale_seconds=args.max_log_stale_seconds,
        max_result_stale_seconds=args.max_result_stale_seconds,
    )
    out_dir = root / "watchdog"
    write_json(out_dir / "autoresearch-watchdog-latest.json", report)
    write_json(out_dir / f"autoresearch-watchdog-{report['timestamp']}.json", report)
    append_jsonl(out_dir / "reviews.jsonl", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] or args.allow_degraded else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic OpenClaw autoresearch watchdog.")
    parser.add_argument("--recent-rows", type=int, default=120)
    parser.add_argument("--target-tps", type=float, default=DEFAULT_TARGET_TPS)
    parser.add_argument("--min-quality", type=float, default=95.0)
    parser.add_argument("--min-scorecard", type=float, default=95.0)
    parser.add_argument("--min-frontier", type=float, default=9.5)
    parser.add_argument("--max-log-stale-seconds", type=float, default=300.0)
    parser.add_argument("--max-result-stale-seconds", type=float, default=600.0)
    parser.add_argument("--allow-degraded", action="store_true")
    return run_once(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
