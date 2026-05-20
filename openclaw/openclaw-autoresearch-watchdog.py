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
import fcntl
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TARGET_TPS = 30.0
ARCHITECTURE_VERSION = 1


CORE_ARCHITECTURE_FILES = [
    "program.md",
    "research-profile.json",
    "insight-rubric.json",
    "benchmark-manifest.json",
    "tasks.jsonl",
    "results.tsv",
    "experiments.jsonl",
    "findings.jsonl",
]


def workspace_root() -> Path:
    explicit = os.environ.get("OPENCLAW_RESEARCH_DIR") or os.environ.get("OPENCLAW_SPEED_RESEARCH_DIR")
    if explicit:
        return Path(explicit).expanduser()
    name = os.environ.get("OPENCLAW_RESEARCH_NAME") or os.environ.get("OPENCLAW_RESEARCH_TOPIC") or "speed"
    slug = "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")[:80] or "research"
    return Path.home() / ".openclaw" / "research" / slug


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


def live_canonical_state(root: Path, *, recent_rows: int, target_tps: float) -> dict[str, Any]:
    """Recompute canonical state so the watchdog is not held hostage by stale artifacts."""
    helper_path = Path(__file__).with_name("openclaw-speed-research.py")
    if not helper_path.exists():
        helper_path = Path(__file__).with_name("openclaw-speed-research")
    if not helper_path.exists():
        return {}
    script_dir = str(helper_path.parent)
    inserted = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted = True
    try:
        loader = SourceFileLoader("openclaw_speed_research_live", str(helper_path))
        spec = importlib.util.spec_from_loader("openclaw_speed_research_live", loader)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        canonical = module.canonical_autoresearch_state(root, recent_rows=recent_rows, target_tps=target_tps)
        return canonical if isinstance(canonical, dict) else {}
    except Exception:
        return {}
    finally:
        if inserted:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass


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
    payload["exists"] = lock.exists()
    payload["active"] = process_alive(pid)
    return payload


def repair_stale_autopilot_lock(root: Path) -> dict[str, Any]:
    """Archive and remove an autopilot lock whose owner process is gone."""
    lock_path = root / "autopilot.lock"
    if not lock_path.exists():
        return {"repaired": False, "reason": "no stale lock"}

    try:
        handle = lock_path.open("r+", encoding="utf-8")
    except FileNotFoundError:
        return {"repaired": False, "reason": "no stale lock"}
    except OSError as error:
        return {"repaired": False, "reason": f"cannot open lock: {error}"}

    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"repaired": False, "reason": "lock currently owned"}

        handle.seek(0)
        try:
            payload = json.loads(handle.read().strip() or "{}")
        except json.JSONDecodeError:
            payload = {}
        pid = int(payload.get("pid", 0) or 0)
        if process_alive(pid):
            return {"repaired": False, "reason": "lock owner still alive"}

        timestamp = int(time.time())
        recovery = {
            "kind": "stale-autopilot-lock-recovery",
            "timestamp": timestamp,
            "repaired": True,
            "reason": "autopilot lock owner process is not alive",
            "lock": payload,
            "lock_path": str(lock_path),
        }
        archive = root / "watchdog" / "stale-locks" / f"autopilot-lock-{timestamp}.json"
        write_json(archive, recovery)
        append_jsonl(root / "watchdog" / "stale-locks.jsonl", recovery | {"archive": str(archive)})
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        recovery["archive"] = str(archive)
        return recovery
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


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


def architecture_contract(root: Path) -> dict[str, Any]:
    """Describe the live autoresearch system from local evidence.

    This gives the sidecar a Pi-style "North Star" and a Hermes-style promotion
    contract without asking an LLM to infer the architecture from memory.
    """

    files = {
        rel: {
            "exists": (root / rel).exists(),
            "size": (root / rel).stat().st_size if (root / rel).exists() else 0,
        }
        for rel in CORE_ARCHITECTURE_FILES
    }
    return {
        "version": ARCHITECTURE_VERSION,
        "north_star": "improve normal OpenClaw TUI decode speed while preserving stability, quality, and model identity",
        "primary_metric": "decode_tps",
        "target_tps": DEFAULT_TARGET_TPS,
        "immutable": [
            "OpenClaw only; never touch opencode",
            "do not change the target model unless explicitly requested",
            "do not mutate live profile/settings from watchdog",
            "do not count repeated synthesis or self-review as progress without new evidence",
            "architectural/source changes require patch-executor canary, tests, rollback, and approval gates",
        ],
        "mutable": [
            "task queue",
            "research strategy",
            "candidate artifacts",
            "self-improvement skills and rubrics after canary validation",
        ],
        "evidence_sources": [
            "results.tsv",
            "benchmarks/quality-review-*.json",
            "benchmarks/frontier-system-eval-*.json",
            "benchmarks/frontier-autonomy-score-*.json",
            "benchmarks/self-improvement-alive-eval-*.json",
            "benchmarks/implementation-handoff-audit-*.json",
            "benchmarks/stability-burn-in-*.json",
            "autopilot.lock",
            "logs/autopilot-*.log",
        ],
        "sidecar_authority": {
            "may_write": ["watchdog/*.json", "watchdog/reviews.jsonl"],
            "may_not_write": ["tasks.jsonl", "results.tsv", "research-profile.json", "program.md", "live OpenClaw profile"],
            "candidate_mode": "advisory-only",
        },
        "files": files,
    }


def evidence_linked_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Create sidecar ideas only from deterministic report evidence.

    These candidates are intentionally advisory. The autopilot/patch-executor
    owns live task mutation and promotion, which prevents sidecar idea noise.
    """

    canonical = report.get("canonical_state", {})
    decode = report.get("decode", {})
    quality = report.get("quality", {})
    frontier = report.get("frontier", {})
    ready = canonical.get("deterministic_ready_tasks") or []
    exhausted = canonical.get("exhausted_lanes") or []
    lanes = canonical.get("breakthrough_lanes") or []
    candidates: list[dict[str, Any]] = []
    base_gate = {
        "quality_healthy": report.get("gates", {}).get("quality_healthy") is True,
        "zero_active_noise": report.get("gates", {}).get("zero_active_noise") is True,
        "handoff_high": report.get("gates", {}).get("handoff_high") is True,
        "frontier_high": report.get("gates", {}).get("frontier_high") is True,
    }

    if report.get("severity") in {"critical", "degraded"}:
        candidates.append(
            {
                "id": "repair-before-new-research",
                "kind": "repair",
                "status": "advisory",
                "allowed_for_live_queue": False,
                "reason": "watchdog gates are not healthy; new ideas would add noise before repair",
                "evidence": {
                    "severity": report.get("severity"),
                    "blockers": report.get("blockers", []),
                    "quality": quality,
                    "frontier": frontier,
                },
                "next": report.get("next_command"),
            }
        )
        return candidates

    if ready:
        candidates.append(
            {
                "id": "continue-ready-deterministic-work",
                "kind": "continue",
                "status": "advisory",
                "allowed_for_live_queue": False,
                "reason": "deterministic ready work already exists; sidecar must not add duplicate tasks",
                "evidence": {
                    "ready_tasks": ready[:8],
                    "breakthrough_lanes": lanes,
                    "decode": decode,
                },
                "next": "let autopilot execute the existing deterministic task",
            }
        )

    if not ready and canonical.get("state") in {"blocked_until_external_change", "plateau_detected"}:
        candidates.append(
            {
                "id": "frontier-candidate-synthesis",
                "kind": "candidate-search",
                "status": "advisory",
                "allowed_for_live_queue": False,
                "reason": "clean plateau/external-blocker state needs exactly one next-candidate synthesis by autopilot",
                "evidence": {
                    "canonical_state": canonical.get("state"),
                    "exhausted_lanes": exhausted,
                    "quality": quality,
                    "frontier": frontier,
                },
                "next": "~/.openclaw/bin/openclaw-speed-research synthesize --kind frontier",
            }
        )

    if decode.get("mean_wall_decode_tps") is not None and float(decode["mean_wall_decode_tps"]) < DEFAULT_TARGET_TPS:
        candidates.append(
            {
                "id": "decode-breakthrough-track",
                "kind": "metric-focus",
                "status": "advisory",
                "allowed_for_live_queue": False,
                "reason": "decode remains below target; only paired benchmark wins should promote",
                "evidence": {
                    "decode": decode,
                    "breakthrough_lanes": lanes,
                    "exhausted_lanes": exhausted,
                    "gates": base_gate,
                },
                "next": "continue or synthesize only through autopilot gates; reject unpaired speed claims",
            }
        )

    return candidates


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
    autonomy = latest_json_artifact(root, "frontier-autonomy-score-*.json")
    alive = latest_json_artifact(root, "self-improvement-alive-eval-*.json")
    handoff = latest_json_artifact(root, "implementation-handoff-audit-*.json")
    burn_in = latest_json_artifact(root, "stability-burn-in-*.json")
    canonical = frontier.get("canonical_state") if isinstance(frontier.get("canonical_state"), dict) else {}
    live_canonical = live_canonical_state(root, recent_rows=recent, target_tps=target_tps)
    if live_canonical:
        canonical = live_canonical
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
    autonomy_total = parse_float(autonomy.get("total_score"))
    alive_total = parse_float(alive.get("total_score"))
    handoff_score = parse_float(handoff.get("score"))
    lock_exists = bool(lock.get("exists"))
    lock_active = bool(lock.get("active"))
    cleanly_idle = not lock_exists and not lock_active
    gates = {
        "autopilot_lock_active_or_cleanly_absent": lock_active or cleanly_idle,
        "log_fresh": cleanly_idle or (bool(log) and float(log.get("age_seconds", 999999)) <= max_log_stale_seconds),
        "results_fresh": cleanly_idle or (result_age is not None and result_age <= max_result_stale_seconds),
        "quality_healthy": quality.get("verdict") == "healthy",
        "quality_score_high": quality_score is not None and quality_score >= min_quality,
        "scorecard_high": scorecard_overall is not None and scorecard_overall >= min_scorecard,
        "handoff_high": handoff_score is not None and handoff_score >= 95,
        "frontier_high": frontier_overall is not None and frontier_overall >= min_frontier,
        "autonomy_score_high": autonomy_total is None or autonomy_total >= 99,
        "alive_score_high": alive_total is None or alive_total >= 95,
        "zero_active_noise": all(value == 0 for value in active_noise.values()),
        "no_recent_memory_blocks": direct_noise["memory_block_rows"] == 0,
    }
    noise_status = "clean" if gates["zero_active_noise"] else "active-noise"
    if noise_status == "clean" and direct_noise["terminal_synthesis_rows"]:
        noise_status = "clean-with-routed-history"
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
    elif cleanly_idle:
        decision = "idle-ready"
        severity = "healthy"
        next_command = "openclaw speed-research --max-hours 10 --cycles 80"
        reason = "no active autopilot lock; latest artifacts are healthy enough for a fresh run"
    elif "zero_active_noise" in blockers or scorecard_overall is not None and scorecard_overall < min_scorecard:
        decision = "repair-routing"
        severity = "degraded"
        next_command = "~/.openclaw/bin/openclaw-speed-research quality-review --recent-rows 120 --min-sweeps 3 --min-samples-per-block 3 --target-tps 30"
        reason = "quality/noise gate failed; route one deterministic repair before more research"
    elif "autonomy_score_high" in blockers:
        decision = "autonomy-repair"
        severity = "degraded"
        next_command = "~/.openclaw/bin/openclaw-speed-research frontier-autonomy-score --allow-fail"
        reason = "frontier autonomy score fell below the safe-action floor"
    elif "alive_score_high" in blockers:
        decision = "self-improvement-repair"
        severity = "degraded"
        next_command = "~/.openclaw/bin/openclaw-speed-research alive-eval --allow-fail"
        reason = "self-improvement alive score fell below the manual-check equivalence floor"
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
        "frontier_autonomy": {
            "artifact": autonomy.get("_artifact_path", ""),
            "total_score": autonomy_total,
            "decision": autonomy.get("decision", ""),
            "hard_gate_failures": autonomy.get("hard_gate_failures", []),
        },
        "self_improvement_alive": {
            "artifact": alive.get("_artifact_path", ""),
            "total_score": alive_total,
            "readiness": alive.get("readiness", ""),
            "hard_gate_failures": alive.get("hard_gate_failures", []),
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
        "noise_interpretation": {
            "status": noise_status,
            "active_noise_rows": sum(active_noise.values()),
            "raw_recent_terminal_rows": direct_noise["terminal_synthesis_rows"],
            "meaning": (
                "active canonical noise is zero; raw terminal rows are historical routed/discard rows"
                if noise_status == "clean-with-routed-history"
                else "active canonical noise is zero"
                if noise_status == "clean"
                else "active canonical noise requires repair before more research"
            ),
        },
        "decode": decode,
    }


def run_once(args: argparse.Namespace) -> int:
    root = workspace_root()
    stale_lock_recovery = repair_stale_autopilot_lock(root) if args.repair_stale_lock else {"repaired": False, "reason": "disabled"}
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
    report["architecture_contract"] = architecture_contract(root)
    report["advisory_candidates"] = evidence_linked_candidates(report)
    report["sidecar_safety"] = {
        "mode": "advisory-only",
        "live_task_mutation": False,
        "reason": "autopilot and patch-executor own task mutation and promotion gates",
        "noise_guard": "sidecar ideas are report artifacts only; duplicate live tasks are not written by watchdog",
    }
    report["stale_lock_recovery"] = stale_lock_recovery
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
    parser.add_argument("--repair-stale-lock", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-degraded", action="store_true")
    return run_once(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
