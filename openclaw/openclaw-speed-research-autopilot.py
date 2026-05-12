#!/usr/bin/env python3
"""Autonomous outer loop for OpenClaw speed autoresearch.

The TUI is interactive: once an assistant turn naturally stops, it waits for a
human. This helper keeps overnight speed research moving by launching repeated
bounded `openclaw agent` turns against the same session.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from openclaw_speed_research_core import (
    RESULTS_HEADER,
    all_result_rows,
    append_jsonl,
    append_result,
    claim_task_evidence_window,
    complete_task_from_evidence,
    cycle_quality,
    ensure_research_state,
    mark_lane_exhausted,
    paired_profile_plan,
    promotion_decision,
    read_jsonl,
    record_rejection,
    record_trajectory_case,
    replay_checks,
    select_next_task,
    task_contract_issues,
    task_summary,
    write_jsonl,
)

HOME = Path.home()
OPENCLAW_HOME = Path(os.environ.get("OPENCLAW_HOME", HOME / ".openclaw")).expanduser()
WORKSPACE = Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_DIR", OPENCLAW_HOME / "research" / "speed")).expanduser()
RESULTS = WORKSPACE / "results.tsv"
IDEAS = WORKSPACE / "ideas.md"
TASKS = WORKSPACE / "tasks.jsonl"
BENCHMARKS = WORKSPACE / "benchmarks"
STRATEGY = WORKSPACE / "STRATEGY.md"
FINDINGS = WORKSPACE / "findings.jsonl"
EXPERIMENTS = WORKSPACE / "experiments.jsonl"
REJECTIONS = WORKSPACE / "rejections.jsonl"
LOG_DIR = WORKSPACE / "logs"
PROGRAM = WORKSPACE / "program.md"
AUTOPILOT_LOCK = WORKSPACE / "autopilot.lock"
DEFAULT_REPO = "/Users/kristian/Documents/openclaw-harness-autoresearch"
DEFAULT_JANQ_TARGET_PATH = (
    "/Users/kristian/.cache/huggingface/hub/"
    "models--dealignai--Gemma-4-31B-JANG_4M-CRACK/"
    "snapshots/bb11360eacf55506f6e51eaacc6b0f65f9209b14"
)
DEFAULT_DFLASH_DRAFT_PATH = (
    "/Users/kristian/.cache/huggingface/hub/"
    "models--z-lab--gemma-4-31B-it-DFlash/"
    "snapshots/9e3bf61731945317dfb0dc2d130c383c9d051f76"
)
DEFAULT_DRAFTER_FIT_PLAN = "/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json"
BLOCKED_PATTERNS = (
    "OpenClaw blocked a broad local tool command",
    "blocked this request before model execution",
    "memory pressure is too high",
    "memory circuit breaker",
    "fatal process exit",
    "metal command buffer",
    "metal out of memory",
    "metal gpu",
    "SSE read timed out",
    "TOOL RESULT CAP",
    "TOOL RESULT SYNTHESIS GRACE",
    "timed out",
    "malformed hidden/tool output",
    "proxy suppressed",
)
EARLY_FAILURE_PATTERNS = (
    ("EMBEDDED FALLBACK", "gateway embedded fallback"),
    ("GatewayClientRequestError", "gateway client request error"),
    ("FailoverError: LLM request failed: network connection error", "model connection error"),
    ("embedded run agent end", "embedded agent failure"),
    ("rawError=Connection error", "model connection error"),
    ("Connection error.", "model connection error"),
)
BENCHMARK_MODES = {
    "quick-health",
    "streaming-ttft",
    "tool-roundtrip",
    "prompt-size",
    "prompt-shape",
    "decode-sample",
    "prefill-reuse",
}
MALFORMED_OR_TOOL_ISSUES = (
    "TOOL RESULT CAP",
    "TOOL RESULT SYNTHESIS GRACE",
    "malformed",
    "no durable artifact",
)
MEMORY_OR_CRASH_TERMS = (
    "memory",
    "metal",
    "fatal process exit",
    "sigabrt",
    "sigkill",
    "sigsegv",
    "crash",
)
ACTIVE_MEMORY_OR_CRASH_ROW_TERMS = (
    "memory gate",
    "memory pressure",
    "compressor_mb",
    "swap_used_mb",
    "metal",
    "fatal process exit",
    "sigabrt",
    "sigkill",
    "sigsegv",
    "crash",
)
ACTIVE_BAD_BEHAVIOR_ROW_TERMS = (
    "tool result cap",
    "tool result synthesis grace",
    "malformed hidden/tool output",
    "malformed tool",
    "no durable artifact",
    "stream timeout",
    "reasoning leak",
    "tool-call loop",
)
NEGATED_HARD_SIGNAL_PHRASES = (
    "no crash",
    "without crash",
    "without a crash",
    "not a crash",
    "no memory pressure",
    "no memory block",
    "no metal crash",
)
INTERRUPT_CONTEXT: dict[str, object] = {
    "args": None,
    "child_process": None,
    "current_session": "",
    "cycle": 0,
    "interrupted": False,
    "lock": None,
    "selected_task": None,
}


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def acquire_autopilot_lock(session: str) -> object | None:
    """Prevent two autopilots from mutating one research workspace at once."""
    AUTOPILOT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = AUTOPILOT_LOCK.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        reason = f"another autoresearch autopilot already owns {AUTOPILOT_LOCK}"
        if owner:
            reason += f" owner={owner}"
        append_result(
            WORKSPACE,
            run_id=f"autopilot-lock-{int(time.time())}",
            status="blocked",
            target="autoresearch-autopilot-lock",
            hypothesis="only one autoresearch supervisor may mutate tasks/results for a workspace at a time",
            commit=current_commit(),
            notes=clean_tsv(reason),
        )
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "session": session,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "workspace": str(WORKSPACE),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    return handle


def release_autopilot_lock(handle: object | None) -> None:
    """Release the workspace lock and remove our lock marker if it is ours."""
    if handle is None:
        return
    try:
        handle.seek(0)
        payload = json.loads(handle.read().strip() or "{}")
        if int(payload.get("pid", 0) or 0) == os.getpid():
            try:
                AUTOPILOT_LOCK.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                log(f"warning: failed to remove autopilot lock marker: {error}")
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    finally:
        try:
            handle.close()
        except Exception:
            pass


def results_line_count() -> int:
    try:
        return len(RESULTS.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def clean_tsv(value: object) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def handle_autopilot_interrupt(signum: int, _frame: object) -> None:
    INTERRUPT_CONTEXT["interrupted"] = True
    process = INTERRUPT_CONTEXT.get("child_process")
    if isinstance(process, subprocess.Popen) and process.poll() is None:
        stop_process_tree(process, terminate_grace_seconds=2)
    raise KeyboardInterrupt(signal.Signals(signum).name)


def install_interrupt_handlers() -> None:
    signal.signal(signal.SIGINT, handle_autopilot_interrupt)
    signal.signal(signal.SIGTERM, handle_autopilot_interrupt)


def finalize_autopilot_interrupt(reason: str = "user interrupt") -> None:
    if INTERRUPT_CONTEXT.get("interrupt_finalized"):
        return
    INTERRUPT_CONTEXT["interrupt_finalized"] = True
    process = INTERRUPT_CONTEXT.get("child_process")
    if isinstance(process, subprocess.Popen) and process.poll() is None:
        stop_process_tree(process, terminate_grace_seconds=2)
    cycle = int(INTERRUPT_CONTEXT.get("cycle") or 0)
    session = str(INTERRUPT_CONTEXT.get("current_session") or INTERRUPT_CONTEXT.get("session") or "unknown")
    selected_task = INTERRUPT_CONTEXT.get("selected_task")
    append_interrupt_checkpoint(
        cycle,
        session,
        reason,
        selected_task=selected_task if isinstance(selected_task, dict) else None,
    )
    args = INTERRUPT_CONTEXT.get("args")
    if args is not None and getattr(args, "stop_model_on_interrupt", True) and model_ready():
        previous_cooldown = getattr(args, "memory_cooldown_after_stop_seconds", 0)
        try:
            setattr(args, "memory_cooldown_after_stop_seconds", 0)
            stop_openclaw_model_for_memory_recovery(args, reason=f"autoresearch interrupted: {reason}")
        finally:
            setattr(args, "memory_cooldown_after_stop_seconds", previous_cooldown)
    lock = INTERRUPT_CONTEXT.get("lock")
    if lock is not None:
        release_autopilot_lock(lock)
        INTERRUPT_CONTEXT["lock"] = None


def calibration_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    runtime_site = Path(
        os.environ.get("OPENCLAW_JANG_TARGET", OPENCLAW_HOME / "runtime" / "rapid-mlx" / "site")
    ).expanduser()
    pythonpath = str(runtime_site)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return env


def missing_speculative_runtime_issue(output: str) -> str:
    lower = output.lower()
    if "no module named 'mlx_vlm.speculative'" in lower or 'no module named "mlx_vlm.speculative"' in lower:
        return "missing-runtime-module:mlx_vlm.speculative"
    if "no module named 'mlx_vlm'" in lower or 'no module named "mlx_vlm"' in lower:
        return "missing-runtime-module:mlx_vlm"
    return ""


def calibration_quantized_gradient_issue(output: str) -> str:
    lower = output.lower()
    if "no gradient wrt the quantized weights" in lower:
        return "calibration-quantized-gradient-unsupported"
    if "quantizedmatmul::vjp" in lower and "no gradient" in lower:
        return "calibration-quantized-gradient-unsupported"
    if "uantized weights" in lower and ("returncode" in lower or "probe_exit:2" in lower):
        return "calibration-quantized-gradient-unsupported"
    return ""


def calibration_memory_gate_issue(output: str) -> str:
    lower = output.lower()
    if "calibration memory gate blocked: after-load" in lower:
        return "calibration-memory-gate:after-load"
    if "calibration memory gate blocked" in lower:
        return "calibration-memory-gate"
    return ""


def is_memory_or_crash_issue(text: object) -> bool:
    lower = str(text).lower()
    return any(term in lower for term in MEMORY_OR_CRASH_TERMS)


def parse_json_object(text: str) -> dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def current_commit() -> str:
    repo = repo_path()
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=7", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def repo_path() -> Path:
    return Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", DEFAULT_REPO))


def repo_status_fingerprint() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path()), "status", "--porcelain"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def latest_mtime(path: Path) -> float:
    try:
        if path.is_file():
            return path.stat().st_mtime
        if not path.is_dir():
            return 0.0
        latest = 0.0
        for child in path.iterdir():
            try:
                latest = max(latest, child.stat().st_mtime)
            except OSError:
                continue
        return latest
    except OSError:
        return 0.0


def durable_snapshot() -> dict[str, object]:
    return {
        "results_lines": results_line_count(),
        "results_mtime": file_mtime(RESULTS),
        "ideas_mtime": file_mtime(IDEAS),
        "tasks_mtime": file_mtime(TASKS),
        "benchmarks_mtime": latest_mtime(BENCHMARKS),
        "strategy_mtime": file_mtime(STRATEGY),
        "findings_mtime": file_mtime(FINDINGS),
        "experiments_mtime": file_mtime(EXPERIMENTS),
        "rejections_mtime": file_mtime(REJECTIONS),
        "repo_status": repo_status_fingerprint(),
    }


def durable_progress(before: dict[str, object], after: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if int(after["results_lines"]) > int(before["results_lines"]):
        reasons.append("results row")
    if float(after["results_mtime"]) > float(before["results_mtime"]):
        reasons.append("results update")
    if float(after["ideas_mtime"]) > float(before["ideas_mtime"]):
        reasons.append("ideas update")
    if float(after["tasks_mtime"]) > float(before["tasks_mtime"]):
        reasons.append("task queue update")
    if float(after["benchmarks_mtime"]) > float(before["benchmarks_mtime"]):
        reasons.append("benchmark artifact")
    if float(after["strategy_mtime"]) > float(before["strategy_mtime"]):
        reasons.append("strategy update")
    if float(after["findings_mtime"]) > float(before["findings_mtime"]):
        reasons.append("findings update")
    if float(after["experiments_mtime"]) > float(before["experiments_mtime"]):
        reasons.append("experiments update")
    if float(after["rejections_mtime"]) > float(before["rejections_mtime"]):
        reasons.append("rejections update")
    if str(after["repo_status"]) != str(before["repo_status"]):
        reasons.append("repo patch")
    return reasons


def ensure_task_queue() -> None:
    ensure_research_state(WORKSPACE)


def next_task_summary(limit: int = 3) -> str:
    return task_summary(WORKSPACE, limit=limit)


def ready_tasks(*, task_type: str | None = None) -> list[dict[str, object]]:
    tasks = [
        task
        for task in read_jsonl(TASKS)
        if task.get("status", "ready") in {"ready", "rework"}
    ]
    if task_type is not None:
        tasks = [task for task in tasks if str(task.get("task_type", "research")) == task_type]
    return sorted(tasks, key=lambda task: int(task.get("priority", 0)), reverse=True)


def ready_implementation_tasks() -> list[dict[str, object]]:
    return ready_tasks(task_type="implementation")


def deterministic_ready_tasks() -> list[dict[str, object]]:
    return [task for task in ready_tasks() if task_runs_without_model(task)]


def select_next_runnable_task(args: argparse.Namespace) -> dict[str, object] | None:
    selected = select_next_task(WORKSPACE)
    if not selected:
        deterministic = deterministic_ready_tasks()
        return deterministic[0] if deterministic else None
    if not model_bound_defer_reason(args, selected):
        return selected
    deterministic = deterministic_ready_tasks()
    if deterministic:
        return deterministic[0]
    return selected


def ready_work_summary() -> dict[str, object]:
    tasks = ready_tasks()
    lanes = sorted({str(task.get("lane", "")) for task in tasks if str(task.get("lane", ""))})
    supervisor = sum(1 for task in tasks if str(task.get("task_type", "research")) == "supervisor")
    implementation = sum(1 for task in tasks if str(task.get("task_type", "research")) == "implementation")
    research = len(tasks) - supervisor - implementation
    return {
        "ready_tasks": len(tasks),
        "supervisor_tasks": supervisor,
        "implementation_tasks": implementation,
        "research_tasks": research,
        "lanes": lanes,
    }


def should_extend_cycle_budget(
    args: argparse.Namespace,
    *,
    deadline: float,
    progress_cycles: int,
    blocked_cycles: int,
    stalled_cycles: int,
) -> tuple[bool, str, dict[str, object]]:
    summary = ready_work_summary()
    if not getattr(args, "auto_extend_cycles", False):
        return False, "cycle budget reached and auto extension disabled", summary
    remaining = deadline - time.monotonic()
    if remaining <= max(1.0, float(getattr(args, "sleep_seconds", 0.0))):
        return False, "max-hours deadline is reached or too close for another useful cycle", summary
    rotate_after = int(getattr(args, "rotate_session_after_stalls", 0) or 0)
    unhealthy_stalls = max(3, rotate_after * 2 if rotate_after > 0 else 3)
    if progress_cycles <= 0 and stalled_cycles >= unhealthy_stalls:
        return False, f"no durable progress after {stalled_cycles} stalled cycles", summary
    if int(summary["ready_tasks"]) > 0:
        return True, f"ready work remains: {summary['ready_tasks']} tasks lanes={','.join(summary['lanes']) or 'none'}", summary
    if progress_cycles > 0 and blocked_cycles <= progress_cycles + max(2, rotate_after):
        return True, "cycle tranche completed with healthy progress; allow supervisor synthesis/refill", summary
    return False, "no ready tasks and recent progress is not healthy enough to extend", summary


def recovery_mode(stalled_cycles: int, last_issue: str) -> str:
    issue = last_issue.lower()
    if "memory" in issue or "metal" in issue or "crash" in issue or "fatal process exit" in issue:
        return "diagnose"
    if "tool result cap" in issue or "tool result synthesis grace" in issue:
        return "force-benchmark"
    if stalled_cycles <= 0:
        return "normal"
    if stalled_cycles == 1:
        return "force-artifact"
    if stalled_cycles == 2:
        return "force-benchmark"
    return "fresh-session"


def recovery_instruction(
    stalled_cycles: int,
    last_issue: str,
    selected_task: dict[str, object] | None = None,
) -> str:
    mode = recovery_mode(stalled_cycles, last_issue)
    if mode == "normal":
        return ""
    if mode == "force-artifact":
        return (
            "\n\nSupervisor recovery: the last cycle ended without a durable artifact. "
            "This cycle has a hard contract: before ending, create exactly one durable artifact: "
            "append a concise results.tsv row, append one grounded ideas.md note, update tasks.jsonl, "
            "run the quick benchmark, or make one source patch plus its focused test result. "
            "Use the smallest action that can satisfy that contract."
        )
    if mode == "force-benchmark":
        if (selected_task or {}).get("task_type") == "implementation":
            return (
                "\n\nSupervisor recovery: the implementation task stalled. "
                "Do not switch to a generic benchmark. Make the smallest source patch, run the focused test named "
                "by the task, or record a blocked row with the exact reason."
            )
        action = str((selected_task or {}).get("next_action", "")).strip()
        if "openclaw-speed-research benchmark --mode" not in action:
            action = "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample"
        return (
            "\n\nSupervisor recovery: two recent cycles did not produce useful durable progress. "
            "Your next tool call must be exactly "
            f"`{action}` "
            "unless memory pressure blocks it. If blocked, append one blocked row to results.tsv and end."
        )
    if mode == "diagnose":
        return (
            "\n\nSupervisor recovery: the last issue was memory, Metal, crash, timeout, or process health related. "
            "Do not start a heavy model action. Inspect one exact OpenClaw log tail or run one exact repo status/test "
            "command, then record a blocked or keep row with the evidence."
        )
    return (
        "\n\nSupervisor recovery: this is a fresh recovery session. Start with one narrow bootstrap action only: "
        "read results.tsv, run the quick benchmark, or run repo status. Record a durable result before ending."
    )


def append_supervisor_result(cycle: int, session: str, status: str, issue: str) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS.exists():
        RESULTS.write_text(RESULTS_HEADER, encoding="utf-8")
    row = [
        time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        f"autopilot-cycle-{cycle}",
        status,
        "autopilot",
        "overnight progress watchdog",
        "",
        "",
        "",
        "",
        "",
        current_commit(),
        f"session={session} issue={clean_tsv(issue)}",
    ]
    with RESULTS.open("a", encoding="utf-8") as file:
        file.write("\t".join(clean_tsv(item) for item in row) + "\n")


def append_quality_pause(cycle: int, session: str, reason: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    append_jsonl(
        FINDINGS,
        {
            "timestamp": timestamp,
            "task_id": "autopilot-quality-pause",
            "finding": "autoresearch paused instead of refilling another generic measurement loop",
            "reason": reason,
            "session": session,
            "next": "add a deterministic implementation, benchmark, or reviewer task before restarting",
        },
    )
    append_result(
        WORKSPACE,
        run_id=f"quality-pause-{cycle}",
        status="blocked",
        target="autoresearch-quality",
        hypothesis="exhausted synthesis should pause instead of researching for the sake of research",
        commit=current_commit(),
        notes=f"session={session} reason={clean_tsv(reason)}",
    )


def append_interrupt_checkpoint(
    cycle: int,
    session: str,
    reason: str,
    *,
    selected_task: dict[str, object] | None = None,
) -> None:
    """Record a user stop as a clean resume checkpoint, not research failure."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    task_id = str((selected_task or {}).get("id", "none"))
    summary = ready_work_summary()
    checkpoint = {
        "timestamp": timestamp,
        "cycle": cycle,
        "session": session,
        "reason": clean_tsv(reason),
        "active_task": task_id,
        "ready_tasks": summary.get("ready_tasks", 0),
        "deterministic_ready_tasks": summary.get("deterministic_ready_tasks", 0),
        "lanes": summary.get("lanes", []),
        "commit": current_commit(),
        "next": "resume with openclaw speed-research; do not treat this checkpoint as a blocker",
    }
    (WORKSPACE / "autopilot-interrupt.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_jsonl(
        FINDINGS,
        {
            **checkpoint,
            "task_id": "autopilot-user-interrupt",
            "finding": "autoresearch was stopped by the user and checkpointed cleanly for the next run",
        },
    )
    append_result(
        WORKSPACE,
        run_id=f"user-interrupt-{int(time.time())}",
        status="keep",
        target="autoresearch-user-interrupt",
        hypothesis="Ctrl+C should leave a neutral resume checkpoint instead of poisoning the next run",
        commit=current_commit(),
        notes=(
            f"session={session} cycle={cycle} active_task={clean_tsv(task_id)} "
            f"ready={checkpoint['ready_tasks']} deterministic={checkpoint['deterministic_ready_tasks']} "
            f"reason={clean_tsv(reason)}"
        ),
    )


TERMINAL_EXTERNAL_ACTIONS = {
    "calibration-memory-report",
    "implementation-bridge",
    "mtp-report",
    "runtime-overhead-map",
}
TERMINAL_EXTERNAL_LANES = {"exhaustion-report", "implementation-gate", "runtime-overhead"}
CORE_SPEED_LANES = {"drafter-calibration-memory", "frontier-dflash", "mtp-decode"}
LOW_SIGNAL_LOOP_TARGETS = {
    "mtp-acceptance-report",
    "synthesis",
    "synthesis-terminal",
}
LOW_SIGNAL_DECODE_REMEASURE_PREFIXES = (
    "lane-contract-decode-remeasure-ready-work-gap-",
    "lane-contract-decode-remeasure-calibration-block-",
    "handoff-audit-decode-remeasure-after-calibration-block-",
)
LOW_SIGNAL_ESCAPE_TARGETS = {
    "decode-sample",
    "frontier-source-scout",
    "frontier-agent-deliberation",
    "runtime-overhead-map",
    "autoresearch-implementation-handoff",
    "patch-execute",
}
LOW_SIGNAL_REMEASURE_ESCAPE_TARGETS = {
    "frontier-source-scout",
    "frontier-agent-deliberation",
    "runtime-overhead-map",
    "calibration-memory-report",
    "autoresearch-implementation-handoff",
    "autoresearch-low-signal-repair",
    "patch-execute",
}


def task_is_terminal_external(task: dict[str, object], *, recent_empty_bridge_count: int) -> bool:
    action = str(task.get("supervisor_action", "")).strip()
    lane = str(task.get("lane", "")).strip()
    task_id = str(task.get("id", "")).strip()
    if lane in CORE_SPEED_LANES:
        return False
    if action == "implementation-bridge":
        return recent_empty_bridge_count > 0 or task_id.startswith("handoff-audit-deterministic-bridge-")
    if action in TERMINAL_EXTERNAL_ACTIONS:
        return True
    return lane in TERMINAL_EXTERNAL_LANES


def recent_terminal_external_count(limit: int = 160) -> int:
    count = 0
    for task in read_jsonl(TASKS)[-limit:]:
        if task.get("status") not in {"done", "blocked"}:
            continue
        if task_is_terminal_external(task, recent_empty_bridge_count=1):
            count += 1
    return count


def external_change_required_status(args: argparse.Namespace) -> dict[str, object]:
    if not getattr(args, "stop_on_external_blocker", True):
        return {"should_stop": False, "reason": "disabled"}

    quality = latest_json_artifact("quality-review-*.json")
    frontier = latest_json_artifact("frontier-system-eval-*.json")
    handoff = latest_json_artifact("implementation-handoff-audit-*.json")
    canonical = quality.get("canonical_state") if isinstance(quality.get("canonical_state"), dict) else {}
    if not canonical:
        canonical = frontier.get("canonical_state") if isinstance(frontier.get("canonical_state"), dict) else {}

    canonical_state = str(canonical.get("state", "")).strip()
    external_states = {"blocked_until_external_change", "prerequisite_needed", "plateau_detected"}
    if canonical_state not in external_states:
        return {"should_stop": False, "reason": f"canonical_state={canonical_state or 'unknown'}"}
    if canonical.get("clean") is False:
        return {"should_stop": False, "reason": "canonical_state_not_clean"}

    quality_scorecard = quality.get("scorecard") if isinstance(quality.get("scorecard"), dict) else {}
    quality_values = [
        value
        for value in (quality.get("quality_score"), quality_scorecard.get("overall"))
        if isinstance(value, int | float)
    ]
    quality_score = float(max(quality_values)) if quality_values else 0.0
    min_quality = float(getattr(args, "external_blocker_min_quality", 90.0))
    if quality_score < min_quality:
        return {"should_stop": False, "reason": f"quality_score={quality_score}<min {min_quality}"}

    exhausted = active_exhausted_lanes()
    exhausted_core = sorted(exhausted & CORE_SPEED_LANES)
    min_exhausted = int(getattr(args, "external_blocker_min_exhausted_core_lanes", 2))
    if len(exhausted_core) < min_exhausted:
        return {
            "should_stop": False,
            "reason": f"exhausted_core_lanes={len(exhausted_core)}<min {min_exhausted}",
            "exhausted_lanes": sorted(exhausted),
        }

    recent_empty_bridges = len(recent_empty_implementation_bridges(limit=120))
    ready = ready_tasks()
    ready_impl = [task for task in ready if str(task.get("task_type", "research")) == "implementation"]
    if ready_impl:
        return {
            "should_stop": False,
            "reason": "ready implementation task exists",
            "ready_tasks": [str(task.get("id", "")) for task in ready_impl[:8]],
        }

    deterministic_ready = [task for task in ready if task_runs_without_model(task)]
    if deterministic_ready:
        return {
            "should_stop": False,
            "reason": "ready deterministic/prerequisite task exists",
            "ready_tasks": [str(task.get("id", "")) for task in deterministic_ready[:8]],
        }

    meaningful_ready = [
        task
        for task in ready
        if not task_is_terminal_external(task, recent_empty_bridge_count=recent_empty_bridges)
        and str(task.get("lane", "")) not in exhausted
    ]
    if meaningful_ready:
        return {
            "should_stop": False,
            "reason": "meaningful ready task exists",
            "ready_tasks": [str(task.get("id", "")) for task in meaningful_ready[:8]],
        }

    noise = canonical.get("noise") if isinstance(canonical.get("noise"), dict) else {}
    terminal_noise = int(noise.get("terminal_synthesis_rows") or 0) + int(noise.get("routed_terminal_synthesis_rows") or 0)
    terminal_count = terminal_noise + recent_terminal_external_count()
    min_terminal = int(getattr(args, "external_blocker_min_terminal_cycles", 3))
    if ready and terminal_count < min_terminal:
        return {
            "should_stop": False,
            "reason": f"terminal_evidence={terminal_count}<min {min_terminal}",
            "ready_tasks": [str(task.get("id", "")) for task in ready[:8]],
        }

    return {
        "should_stop": True,
        "reason": "external change required before more useful autonomous speed research",
        "canonical_state": canonical_state,
        "decode_mean_tps": canonical.get("decode_mean_tps"),
        "quality_score": quality_score,
        "handoff_score": handoff.get("score"),
        "ready_tasks": [str(task.get("id", "")) for task in ready[:8]],
        "exhausted_lanes": sorted(exhausted),
        "terminal_evidence": terminal_count,
        "next": canonical.get("next") or "add a new drafter candidate, trace source, or approved implementation direction",
    }


def recent_low_signal_mtp_loop_status(args: argparse.Namespace) -> dict[str, object]:
    """Detect stable but unproductive synthesis -> MTP-report churn.

    This is intentionally deterministic: repeated log-summary tasks can be
    useful once, but after a few unchanged cycles they stop being evidence and
    should route to a new benchmark, source scout, or patchable hypothesis.
    """
    rows = all_result_rows(WORKSPACE)[-max(12, int(getattr(args, "low_signal_window_rows", 40))) :]
    mtp_rows = [
        row
        for row in rows
        if row.get("target") == "mtp-acceptance-report" or row.get("run_id", "").startswith("mtp-report-")
    ]
    synthesis_rows = [
        row
        for row in rows
        if row.get("target") in {"synthesis", "synthesis-terminal"} or row.get("run_id", "").startswith("synthesis-")
    ]
    escape_rows = [
        row
        for row in rows
        if row.get("target") in LOW_SIGNAL_ESCAPE_TARGETS
        or row.get("run_id", "").startswith(("source-scout-", "frontier-agent-deliberation-", "runtime-overhead-map-"))
        or row.get("target") == "decode-sample"
    ]
    min_mtp = int(getattr(args, "low_signal_min_mtp_reports", 4))
    min_synthesis = int(getattr(args, "low_signal_min_synthesis_rows", 4))
    repeated_notes = len({re.sub(r"path=[^ ]+", "path=<artifact>", row.get("notes", "")) for row in mtp_rows[-min_mtp:]})
    loop = len(mtp_rows) >= min_mtp and len(synthesis_rows) >= min_synthesis and not escape_rows
    if not loop and len(mtp_rows) >= min_mtp + 2 and repeated_notes <= 2:
        loop = True
    return {
        "loop": loop,
        "mtp_reports": len(mtp_rows),
        "synthesis_rows": len(synthesis_rows),
        "escape_rows": len(escape_rows),
        "repeated_note_shapes": repeated_notes,
        "latest_mtp_run": mtp_rows[-1].get("run_id", "") if mtp_rows else "",
        "reason": (
            "repeated synthesis/MTP-report loop without new candidate evidence"
            if loop
            else "no low-signal MTP loop detected"
        ),
    }


def recent_low_signal_decode_remeasure_status(args: argparse.Namespace) -> dict[str, object]:
    """Detect clean-but-unproductive decode remeasure churn.

    A valid decode sample is evidence. Repeated synthesis -> decode benchmark
    with no escape task is not strategy. This catches the exact pattern where
    the loop keeps proving ~15 tok/s instead of routing to drafter-fit,
    source-scout, runtime-map, or a patchable contract.
    """
    rows = all_result_rows(WORKSPACE)[-max(12, int(getattr(args, "low_signal_window_rows", 40))) :]
    def is_remeasure_synthesis(row: dict[str, str]) -> bool:
        return row.get("target") in {"synthesis", "synthesis-terminal"} and any(
            prefix in row.get("notes", "") for prefix in LOW_SIGNAL_DECODE_REMEASURE_PREFIXES
        )

    def is_clean_decode(row: dict[str, str]) -> bool:
        return (
            row.get("status") == "keep"
            and row.get("target") == "decode-sample"
            and "measurement_quality=clean" in row.get("notes", "")
        )

    def is_escape(row: dict[str, str]) -> bool:
        return row.get("target") in LOW_SIGNAL_REMEASURE_ESCAPE_TARGETS or row.get("run_id", "").startswith(
            (
                "source-scout-",
                "frontier-agent-deliberation-",
                "runtime-overhead-map-",
                "calibration-memory-report-",
                "implementation-handoff-audit-",
                "patch-executor-",
            )
        )

    escape_indexes = [index for index, row in enumerate(rows) if is_escape(row)]
    latest_escape_index = escape_indexes[-1] if escape_indexes else -1
    post_escape_rows = rows[latest_escape_index + 1 :]
    remeasure_synthesis = [row for row in post_escape_rows if is_remeasure_synthesis(row)]
    clean_decode_rows = [row for row in post_escape_rows if is_clean_decode(row)]
    escape_rows = [row for row in rows if is_escape(row)]
    min_remeasure = int(getattr(args, "low_signal_min_decode_remeasures", 3))
    loop = len(remeasure_synthesis) >= min_remeasure and len(clean_decode_rows) >= min_remeasure and not escape_rows
    recent_ready_remeasure = [
        str(task.get("id", ""))
        for task in read_jsonl(TASKS)
        if task.get("status", "ready") in {"ready", "rework"}
        and str(task.get("id", "")).startswith(LOW_SIGNAL_DECODE_REMEASURE_PREFIXES)
    ]
    ready_escape_tasks = [
        str(task.get("id", ""))
        for task in read_jsonl(TASKS)
        if task.get("status", "ready") in {"ready", "rework"}
        and str(task.get("supervisor_action", ""))
        in {"source-scout", "runtime-overhead-map", "frontier-deliberation"}
    ]
    if not loop and len(remeasure_synthesis) >= min_remeasure + 1 and recent_ready_remeasure:
        loop = True
    if loop and ready_escape_tasks and not recent_ready_remeasure:
        loop = False
    return {
        "loop": loop,
        "decode_remeasure_synthesis": len(remeasure_synthesis),
        "clean_decode_rows": len(clean_decode_rows),
        "escape_rows": len(escape_rows),
        "rows_since_latest_escape": len(post_escape_rows),
        "ready_remeasure_tasks": recent_ready_remeasure[:8],
        "ready_escape_tasks": ready_escape_tasks[:8],
        "latest_decode_run": clean_decode_rows[-1].get("run_id", "") if clean_decode_rows else "",
        "reason": (
            "repeated clean decode remeasure loop without drafter-fit/runtime/source escape"
            if loop
            else "no low-signal decode remeasure loop detected"
        ),
    }


def block_ready_low_signal_mtp_tasks(reason: str) -> int:
    tasks = read_jsonl(TASKS)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    blocked = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        action = str(task.get("supervisor_action", ""))
        task_id = str(task.get("id", ""))
        if action != "mtp-report" and "mtp-acceptance-yield" not in task_id:
            continue
        task["status"] = "blocked"
        task["blocked_at"] = now
        task["blocked_reason"] = f"low-signal loop suppressed: {reason}"
        task["supervisor_summary"] = {
            **(task.get("supervisor_summary") if isinstance(task.get("supervisor_summary"), dict) else {}),
            "reason": "low_signal_mtp_loop_suppressed",
            "next": "route to source-scout, frontier-deliberation, runtime-overhead, or paired decode benchmark",
        }
        blocked += 1
    if blocked:
        write_jsonl(TASKS, tasks)
    return blocked


def block_ready_low_signal_decode_remeasure_tasks(reason: str) -> int:
    tasks = read_jsonl(TASKS)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    blocked = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        task_id = str(task.get("id", ""))
        if not task_id.startswith(LOW_SIGNAL_DECODE_REMEASURE_PREFIXES):
            continue
        task["status"] = "blocked"
        task["blocked_at"] = now
        task["blocked_reason"] = f"low-signal decode remeasure loop suppressed: {reason}"
        task["supervisor_summary"] = {
            **(task.get("supervisor_summary") if isinstance(task.get("supervisor_summary"), dict) else {}),
            "reason": "low_signal_decode_remeasure_loop_suppressed",
            "next": "route to source-scout, frontier-deliberation, runtime-overhead, or drafter-fit contract",
        }
        blocked += 1
    if blocked:
        write_jsonl(TASKS, tasks)
    return blocked


def low_signal_frontier_deliberation_task(cycle: int, status: dict[str, object]) -> dict[str, object]:
    return {
        "id": f"low-signal-frontier-deliberation-cycle-{cycle:03d}",
        "status": "ready",
        "priority": 99,
        "lane": "frontier-deliberation",
        "task_type": "supervisor",
        "supervisor_action": "frontier-deliberation",
        "target": "tasks.jsonl/results.tsv",
        "hypothesis": (
            "A repeated synthesis/MTP-report loop means the supervisor must select a new bounded path "
            "instead of collecting another equivalent log summary."
        ),
        "metric": "frontier_deliberation_contract",
        "guard_checks": ["no_model_load", "no_opencode_changes", "no_live_profile_change", "rollback_path"],
        "acceptance": "A frontier-agent-deliberation artifact selects one safe deterministic next task or records a closed blocker.",
        "rollback": "No source rollback needed; this only writes deliberation/task artifacts.",
        "evidence": status,
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research frontier-deliberation --allow-empty",
    }


def low_signal_source_scout_task(cycle: int, status: dict[str, object]) -> dict[str, object]:
    return {
        "id": f"low-signal-source-scout-cycle-{cycle:03d}",
        "status": "ready",
        "priority": 100,
        "lane": "frontier-deliberation",
        "task_type": "supervisor",
        "supervisor_action": "source-scout",
        "target": "external-speed-references",
        "topic": "JANQ Gemma4 MTP acceptance, drafter fit, Rapid-MLX decode speed, DFlash compatibility",
        "hypothesis": "A low-signal local loop should refresh bounded external evidence before creating another candidate.",
        "metric": "source_evidence_count",
        "guard_checks": ["allowlisted_hosts_only", "timeout_bounded", "no_model_load", "no_opencode_changes"],
        "acceptance": "A source-scout artifact records fetched/skipped references and concrete next-source gaps.",
        "rollback": "No rollback needed; this is read-only evidence collection.",
        "evidence": status,
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research source-scout --topic frontier-decode-speed",
    }


def low_signal_runtime_map_task(cycle: int, status: dict[str, object]) -> dict[str, object]:
    return {
        "id": f"low-signal-runtime-map-cycle-{cycle:03d}",
        "status": "ready",
        "priority": 98,
        "lane": "runtime-overhead",
        "task_type": "supervisor",
        "supervisor_action": "runtime-overhead-map",
        "target": "openclaw/openclaw-model-proxy.py",
        "hypothesis": "Repeated MTP summaries must be converted into a source/runtime boundary map before more reports run.",
        "metric": "server_wall_decode_gap",
        "guard_checks": ["no_model_load", "no_opencode_changes", "no_live_profile_change"],
        "acceptance": "A runtime-overhead map identifies a patchable boundary or records that this lane is clean.",
        "rollback": "No runtime rollback needed; this is a read-only supervisor artifact.",
        "evidence": status,
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
    }


def enqueue_low_signal_repair_tasks(cycle: int, status: dict[str, object]) -> int:
    existing = read_jsonl(TASKS)
    existing_ids = {str(task.get("id", "")) for task in existing}
    candidates = [
        low_signal_source_scout_task(cycle, status),
        low_signal_runtime_map_task(cycle, status),
        low_signal_frontier_deliberation_task(cycle, status),
    ]
    additions = [task for task in candidates if str(task["id"]) not in existing_ids]
    if additions:
        write_jsonl(TASKS, existing + additions)
    return len(additions)


def repair_low_signal_mtp_loop(cycle: int, session: str, status: dict[str, object]) -> dict[str, object]:
    blocked = block_ready_low_signal_mtp_tasks(str(status.get("reason", "")))
    seeded = enqueue_low_signal_repair_tasks(cycle, status)
    append_result(
        WORKSPACE,
        run_id=f"low-signal-loop-repair-{cycle}-{int(time.time())}",
        status="keep" if seeded or blocked else "blocked",
        target="autoresearch-low-signal-repair",
        hypothesis="the autonomous supervisor should break repeated synthesis/MTP-report loops without human intervention",
        commit=current_commit(),
        notes=(
            f"session={session} blocked_mtp_tasks={blocked} seeded_tasks={seeded} "
            f"status={clean_tsv(json.dumps(status, sort_keys=True))}"
        ),
    )
    append_jsonl(
        FINDINGS,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "autonomous-low-signal-loop-repair",
            "finding": "supervisor detected repeated MTP-report churn and routed to new bounded evidence paths",
            "blocked_mtp_tasks": blocked,
            "seeded_tasks": seeded,
            "evidence": status,
            "next": "run source-scout/runtime-map/frontier-deliberation before another MTP report",
        },
    )
    return {"blocked_tasks": blocked, "seeded_tasks": seeded}


def repair_low_signal_decode_remeasure_loop(cycle: int, session: str, status: dict[str, object]) -> dict[str, object]:
    blocked = block_ready_low_signal_decode_remeasure_tasks(str(status.get("reason", "")))
    seeded = enqueue_low_signal_repair_tasks(cycle, status)
    append_result(
        WORKSPACE,
        run_id=f"low-signal-decode-remeasure-repair-{cycle}-{int(time.time())}",
        status="keep" if seeded or blocked else "blocked",
        target="autoresearch-low-signal-repair",
        hypothesis="the autonomous supervisor should break repeated clean decode remeasure loops without human intervention",
        commit=current_commit(),
        notes=(
            f"session={session} blocked_remeasure_tasks={blocked} seeded_tasks={seeded} "
            f"status={clean_tsv(json.dumps(status, sort_keys=True))}"
        ),
    )
    append_jsonl(
        FINDINGS,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "autonomous-low-signal-decode-remeasure-repair",
            "finding": "supervisor detected repeated clean decode remeasure churn and routed to bounded escape paths",
            "blocked_remeasure_tasks": blocked,
            "seeded_tasks": seeded,
            "evidence": status,
            "next": "run source-scout/runtime-map/frontier-deliberation before another generic decode remeasure",
        },
    )
    return {"blocked_tasks": blocked, "seeded_tasks": seeded}


def append_external_change_required(cycle: int, session: str, status: dict[str, object]) -> None:
    reason = str(status.get("reason") or "external change required")
    next_step = str(status.get("next") or "add a new candidate/prerequisite before restarting")
    evidence = {
        "canonical_state": status.get("canonical_state"),
        "decode_mean_tps": status.get("decode_mean_tps"),
        "quality_score": status.get("quality_score"),
        "handoff_score": status.get("handoff_score"),
        "exhausted_lanes": status.get("exhausted_lanes"),
        "ready_tasks": status.get("ready_tasks"),
        "terminal_evidence": status.get("terminal_evidence"),
    }
    append_jsonl(
        FINDINGS,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "autoresearch-external-change-required",
            "finding": "autoresearch stopped cleanly because active speed lanes are exhausted",
            "reason": reason,
            "session": session,
            "evidence": evidence,
            "next": next_step,
        },
    )
    append_result(
        WORKSPACE,
        run_id=f"external-change-required-{cycle}",
        status="blocked",
        target="autoresearch-external-change-required",
        hypothesis="autoresearch should stop at a clean external blocker instead of repeating terminal synthesis",
        commit=current_commit(),
        notes=f"session={session} reason={clean_tsv(reason)} next={clean_tsv(next_step)} evidence={clean_tsv(json.dumps(evidence, sort_keys=True))}",
    )


def append_external_refocus(cycle: int, session: str, status: dict[str, object], seeded_tasks: int) -> None:
    reason = str(status.get("reason") or "external change required")
    next_step = str(status.get("next") or "continue with the next deterministic decode/MTP tranche")
    evidence = {
        "canonical_state": status.get("canonical_state"),
        "decode_mean_tps": status.get("decode_mean_tps"),
        "quality_score": status.get("quality_score"),
        "handoff_score": status.get("handoff_score"),
        "exhausted_lanes": status.get("exhausted_lanes"),
        "ready_tasks": status.get("ready_tasks"),
        "terminal_evidence": status.get("terminal_evidence"),
        "seeded_tasks": seeded_tasks,
    }
    append_jsonl(
        FINDINGS,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "autoresearch-external-refocus",
            "finding": "autoresearch converted a clean external blocker into the next autonomous decode/MTP tranche",
            "reason": reason,
            "session": session,
            "evidence": evidence,
            "next": next_step,
        },
    )
    append_result(
        WORKSPACE,
        run_id=f"external-refocus-{cycle}",
        status="keep",
        target="autoresearch-external-refocus",
        hypothesis="overnight autoresearch should refocus clean blockers instead of waiting for manual continuation",
        commit=current_commit(),
        notes=(
            f"session={session} seeded_tasks={seeded_tasks} reason={clean_tsv(reason)} "
            f"next={clean_tsv(next_step)} evidence={clean_tsv(json.dumps(evidence, sort_keys=True))}"
        ),
    )


def refill_before_external_stop(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    log_file: Path,
    status: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Give deterministic reviewers one chance to create useful work before stopping."""
    attempts: list[str] = []
    if not getattr(args, "external_blocker_refill_before_stop", True):
        return status, attempts
    if status.get("ready_tasks"):
        return status, attempts

    review_ok, review_issue = run_supervisor_quality_review(args, cycle, session, log_file)
    attempts.append(f"quality_review={review_ok}:{review_issue or 'ok'}")
    refreshed = external_change_required_status(args)
    if not refreshed.get("should_stop"):
        return refreshed, attempts

    synth_ok, synth_issue = run_supervisor_synthesis(args, cycle, session, log_file)
    attempts.append(f"synthesis={synth_ok}:{synth_issue or 'ok'}")
    review_ok, review_issue = run_supervisor_quality_review(args, cycle, session, log_file)
    attempts.append(f"post_synthesis_quality_review={review_ok}:{review_issue or 'ok'}")
    refreshed = external_change_required_status(args)
    if refreshed.get("should_stop"):
        refreshed = {**refreshed, "refill_attempts": attempts}
    return refreshed, attempts


def maybe_stop_for_external_change(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    log_file: Path | None = None,
) -> tuple[bool, dict[str, object]]:
    status = external_change_required_status(args)
    if status.get("should_stop"):
        if log_file is not None:
            status, attempts = refill_before_external_stop(args, cycle, session, log_file, status)
            if attempts:
                log(f"cycle={cycle} external_stop_refill attempts={'; '.join(attempts)}")
            if not status.get("should_stop"):
                log(
                    f"cycle={cycle} external_stop_refill_resolved "
                    f"reason={status.get('reason', 'ready work created')}"
                )
                return False, status
        action = str(getattr(args, "external_blocker_action", "refocus") or "refocus").strip().lower()
        if action != "stop":
            seeded = enqueue_recurring_decode_tasks(cycle, f"external blocker refocus: {status.get('reason', '')}")
            append_external_refocus(cycle, session, status, seeded)
            refocused = {
                **status,
                "should_stop": False,
                "reason": f"external blocker refocused into deterministic work: {status.get('reason', '')}",
                "seeded_tasks": seeded,
            }
            return False, refocused
        append_external_change_required(cycle, session, status)
        return True, status
    return False, status


def benchmark_mode_for_task(task: dict[str, object] | None) -> str:
    if not task or task.get("task_type") == "implementation":
        return ""
    if requires_profile_variant_runner(task):
        return ""
    mode = str(task.get("benchmark_mode") or "").strip()
    if mode in BENCHMARK_MODES:
        return mode
    action = str(task.get("next_action") or "")
    if "benchmark --quick" in action:
        return "quick-health"
    match = re.search(r"openclaw-speed-research\s+benchmark\s+--mode\s+([a-z-]+)", action)
    if match and match.group(1) in BENCHMARK_MODES:
        return match.group(1)
    return ""


def is_supervisor_benchmark_task(task: dict[str, object] | None) -> bool:
    return bool(benchmark_mode_for_task(task))


def is_supervisor_log_review_task(task: dict[str, object] | None) -> bool:
    if not task or task.get("task_type") == "implementation":
        return False
    action = str(task.get("next_action") or "").strip()
    return action == "tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log"


def is_supervisor_drafter_sweep_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    action = str(task.get("next_action", ""))
    return (
        task.get("supervisor_action") in {"drafter-sweep-plan", "drafter-sweep-run"}
        or "openclaw-speed-research drafter-sweep-plan" in action
        or "openclaw-speed-research drafter-sweep-run" in action
    )


def is_supervisor_mtp_report_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "mtp-report"
        or "openclaw-speed-research mtp-report" in str(task.get("next_action", ""))
    )


def is_supervisor_implementation_bridge_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "implementation-bridge"
        or str(task.get("id", "")).startswith("implementation-bridge-")
        or str(task.get("id", "")).startswith("handoff-audit-deterministic-bridge-")
        or str(task.get("id", "")).startswith("frontier-repair-implementation-bridge-")
    )


def recent_empty_implementation_bridges(limit: int = 80) -> list[dict[str, str]]:
    return [
        row
        for row in all_result_rows(WORKSPACE)[-max(1, limit) :]
        if row.get("run_id", "").startswith("supervisor-implementation-bridge-")
        and "ready_deterministic=0" in row.get("notes", "")
    ]


def bridge_only_deterministic_ready(tasks: list[dict[str, object]]) -> bool:
    return bool(tasks) and all(is_supervisor_implementation_bridge_task(task) for task in tasks)


def block_ready_empty_bridge_tasks() -> int:
    if not recent_empty_implementation_bridges(limit=120):
        return 0
    tasks = read_jsonl(TASKS)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    blocked = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        if not is_supervisor_implementation_bridge_task(task):
            continue
        task["status"] = "blocked"
        task["blocked_at"] = now
        task["blocked_reason"] = "empty implementation bridge already proved this bridge path has no downstream task"
        task["supervisor_summary"] = {
            **(task.get("supervisor_summary") if isinstance(task.get("supervisor_summary"), dict) else {}),
            "reason": "bridge_quarantined_after_empty_handoff",
            "next": "route to concrete prerequisite task instead of another implementation bridge",
        }
        blocked += 1
    if blocked:
        write_jsonl(TASKS, tasks)
    return blocked


def is_supervisor_patch_execute_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "patch-execute"
        or "openclaw-speed-research patch-execute" in str(task.get("next_action", ""))
    )


def is_supervisor_drafter_fit_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "drafter-fit-plan"
        or "openclaw-drafter-fit plan" in str(task.get("next_action", ""))
    )


def is_supervisor_drafter_bottleneck_review_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "drafter-bottleneck-review"
        or "openclaw-speed-research drafter-bottleneck-review" in str(task.get("next_action", ""))
    )


def is_supervisor_drafter_adapter_contract_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "drafter-adapter-method-contract"
        or "openclaw-speed-research drafter-adapter-method-contract" in str(task.get("next_action", ""))
    )


def is_supervisor_drafter_trace_gate_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "drafter-trace-gate"
        or "openclaw-speed-research drafter-trace-gate" in str(task.get("next_action", ""))
    )


def is_supervisor_drafter_trace_prerequisite_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "drafter-trace-prerequisite"
        or "openclaw-speed-research drafter-trace-prerequisite" in str(task.get("next_action", ""))
    )


def is_supervisor_drafter_trace_collect_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "drafter-trace-collect"
        or "openclaw-speed-research drafter-trace-collect" in str(task.get("next_action", ""))
    )


def is_supervisor_drafter_calibration_canary_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "drafter-calibration-canary"
        or "openclaw-speed-research drafter-calibration-canary" in str(task.get("next_action", ""))
    )


def is_supervisor_drafter_calibration_run_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "drafter-calibration-run"
        or "openclaw-mtp-drafter-calibrate.py" in str(task.get("next_action", ""))
    )


def is_supervisor_drafter_calibration_memory_stage_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "drafter-calibration-memory-stage"
        or "openclaw-speed-research drafter-calibration-memory-stage" in str(task.get("next_action", ""))
    )


def suppress_ready_calibration_canaries(reason: str, summary: dict[str, object]) -> int:
    tasks = read_jsonl(TASKS)
    has_stage = any(
        task.get("status", "ready") in {"ready", "rework"}
        and is_supervisor_drafter_calibration_memory_stage_task(task)
        for task in tasks
    )
    if not has_stage:
        return 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    suppressed = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        if not is_supervisor_drafter_calibration_canary_task(task):
            continue
        task["status"] = "done"
        task["completed_at"] = now
        task["completion_commit"] = current_commit()
        task["supervisor_summary"] = {
            **(task.get("supervisor_summary") if isinstance(task.get("supervisor_summary"), dict) else {}),
            "reason": reason,
            "suppressed_by": "active_calibration_memory_stage",
            "source_summary": summary,
        }
        suppressed += 1
    if suppressed:
        write_jsonl(TASKS, tasks)
        append_jsonl(
            FINDINGS,
            {
                "timestamp": now,
                "task_id": "calibration-canary-suppression",
                "finding": "suppressed ready calibration canaries because a calibration memory-stage is now queued",
                "suppressed": suppressed,
                "next": "select_calibration_memory_stage",
            },
        )
    return suppressed


def is_supervisor_dflash_compatibility_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    task_id = str(task.get("id", ""))
    return (
        task.get("supervisor_action") == "dflash-compatibility-gate"
        or task_id.startswith("deliberate-dflash-compatibility-")
        or task_id.startswith("plateau-dflash-compat-")
        or "openclaw-speed-research dflash-compatibility-gate" in str(task.get("next_action", ""))
    )


def is_supervisor_focused_test_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return task.get("supervisor_action") == "focused-test"


def is_supervisor_gepa_policy_canary_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "gepa-policy-canary"
        or "openclaw-speed-research gepa-policy-canary" in str(task.get("next_action", ""))
    )


def is_supervisor_runtime_overhead_map_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "runtime-overhead-map"
        or "openclaw-speed-research runtime-overhead-map" in str(task.get("next_action", ""))
    )


def is_supervisor_calibration_memory_report_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "calibration-memory-report"
        or "openclaw-speed-research calibration-memory-report" in str(task.get("next_action", ""))
    )


def is_supervisor_source_scout_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "source-scout"
        or "openclaw-speed-research source-scout" in str(task.get("next_action", ""))
    )


def is_supervisor_frontier_deliberation_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return (
        task.get("supervisor_action") == "frontier-deliberation"
        or "openclaw-speed-research frontier-deliberation" in str(task.get("next_action", ""))
    )


def requires_profile_variant_runner(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    if is_supervisor_drafter_sweep_task(task):
        return False
    guard_checks = {str(item) for item in task.get("guard_checks", []) if item}
    target = str(task.get("target", ""))
    return (
        "restore_live_profile" in guard_checks
        or "same_prompt_set" in guard_checks
        or target.startswith("OPENCLAW_JANG_DRAFT_")
    )


def task_runs_without_model(task: dict[str, object] | None) -> bool:
    if not task:
        return True
    return any(
        predicate(task)
        for predicate in (
            is_supervisor_log_review_task,
            is_supervisor_drafter_sweep_task,
            is_supervisor_mtp_report_task,
            is_supervisor_implementation_bridge_task,
            is_supervisor_patch_execute_task,
            is_supervisor_drafter_fit_task,
            is_supervisor_drafter_bottleneck_review_task,
            is_supervisor_drafter_adapter_contract_task,
            is_supervisor_drafter_trace_gate_task,
            is_supervisor_drafter_trace_prerequisite_task,
            is_supervisor_drafter_trace_collect_task,
            is_supervisor_drafter_calibration_canary_task,
            is_supervisor_drafter_calibration_memory_stage_task,
            is_supervisor_drafter_calibration_run_task,
            is_supervisor_dflash_compatibility_task,
            is_supervisor_focused_test_task,
            is_supervisor_gepa_policy_canary_task,
            is_supervisor_runtime_overhead_map_task,
            is_supervisor_calibration_memory_report_task,
            is_supervisor_source_scout_task,
            is_supervisor_frontier_deliberation_task,
            requires_profile_variant_runner,
            is_supervisor_benchmark_task,
        )
    )


def model_bound_defer_reason(args: argparse.Namespace, task: dict[str, object] | None) -> str:
    if task_runs_without_model(task):
        return ""
    if (
        task
        and task.get("task_type") == "implementation"
        and not getattr(args, "allow_implementation_model_turns", False)
    ):
        return "implementation task requires deterministic patch-executor path"
    if (
        task
        and task.get("task_type") == "implementation"
        and getattr(args, "allow_implementation_model_turns", False)
    ):
        if not model_ready():
            return "model endpoint offline after memory recovery"
        return ""
    if not getattr(args, "allow_model_bound_research_turns", False):
        return "model-bound research turn deferred for local 31B stability"
    if not model_ready():
        return "model endpoint offline after memory recovery"
    return ""


def recent_task_rejections(task_id: str, reason: str, limit: int = 3) -> int:
    if not task_id:
        return 0
    rows = read_jsonl(REJECTIONS)
    count = 0
    for row in reversed(rows):
        if row.get("task_id") != task_id:
            continue
        if reason and row.get("reason") != reason:
            continue
        count += 1
        if count >= limit:
            return count
    return count


def block_task_after_repeated_guard(task: dict[str, object] | None, reason: str, *, threshold: int = 3) -> bool:
    if not task or task.get("task_type") != "implementation":
        return False
    threshold_by_reason = {
        "OpenClaw blocked a broad local tool command": threshold,
        "TOOL RESULT SYNTHESIS GRACE": 1,
        "TOOL RESULT CAP": 1,
        "turn timeout": 1,
    }
    if reason not in threshold_by_reason:
        return False
    task_id = str(task.get("id", ""))
    active_threshold = threshold_by_reason[reason]
    if recent_task_rejections(task_id, reason, limit=active_threshold) < active_threshold:
        return False
    tasks = read_jsonl(TASKS)
    changed = False
    blocked_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for item in tasks:
        if item.get("id") != task_id:
            continue
        item["status"] = "blocked"
        item["blocked_at"] = blocked_at
        item["blocked_reason"] = reason
        item["next"] = "supervisor moved on after repeated implementation stalls; refine task into narrower exact file/tool steps"
        changed = True
        break
    if not changed:
        return False
    write_jsonl(TASKS, tasks)
    append_jsonl(
        FINDINGS,
        {
            "timestamp": blocked_at,
            "task_id": task_id,
            "finding": "implementation task stalled or hit a guard without durable evidence; supervisor marked it blocked and moved on",
            "reason": reason,
            "next": "select_next_ready_task",
        },
    )
    return True


def block_stale_rejected_implementation_tasks() -> int:
    blocked = 0
    for task in read_jsonl(TASKS):
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        if task.get("task_type") != "implementation":
            continue
        for reason in ("TOOL RESULT SYNTHESIS GRACE", "TOOL RESULT CAP", "turn timeout"):
            if block_task_after_repeated_guard(task, reason):
                blocked += 1
                break
        else:
            if block_task_after_repeated_guard(task, "OpenClaw blocked a broad local tool command", threshold=3):
                blocked += 1
    return blocked


def block_model_bound_implementation_tasks(args: argparse.Namespace) -> int:
    if getattr(args, "allow_implementation_model_turns", False):
        return 0
    tasks = read_jsonl(TASKS)
    blocked_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    blocked_ids: list[str] = []
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        if task.get("task_type") != "implementation":
            continue
        if task_runs_without_model(task):
            continue
        task["status"] = "blocked"
        task["blocked_at"] = blocked_at
        task["blocked_reason"] = "implementation task requires deterministic patch-executor path"
        task["next"] = (
            "convert implementation into a supervisor patch-executor task with canary, "
            "acceptance, and rollback gates before it can run"
        )
        blocked_ids.append(str(task.get("id", "unknown")))
    if not blocked_ids:
        return 0
    write_jsonl(TASKS, tasks)
    append_jsonl(
        FINDINGS,
        {
            "timestamp": blocked_at,
            "task_id": "implementation-model-turn-guard",
            "finding": "supervisor quarantined model-bound implementation tasks before selection",
            "blocked_tasks": blocked_ids,
            "reason": "implementation tasks must use deterministic patch-executor gates by default",
            "next": "run quality review or synthesis to seed a safe deterministic bridge",
        },
    )
    return len(blocked_ids)


def recent_hard_dflash_blocker(limit: int = 120) -> str:
    rows = all_result_rows(WORKSPACE)[-max(1, limit) :]
    for row in reversed(rows):
        if row.get("status") != "blocked" or not row.get("run_id", "").startswith("dflash-compatibility-gate-"):
            continue
        run_id = row.get("run_id", "")
        timestamp = run_id.rsplit("-", 1)[-1]
        artifact = WORKSPACE / "experiments" / f"dflash-compatibility-gate-{timestamp}.json"
        blockers: list[str] = []
        if artifact.exists():
            try:
                loaded = json.loads(artifact.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    blockers = [str(item) for item in loaded.get("blockers", []) if item]
            except (OSError, json.JSONDecodeError):
                blockers = []
        for blocker in blockers:
            if blocker.startswith("draft_model_type_mismatch=") or blocker in {
                "dflash_draft_config_missing",
                "draft_target_layer_ids_missing",
            }:
                return blocker
    return ""


def block_stale_hard_blocked_lane_tasks() -> int:
    dflash_blocker = recent_hard_dflash_blocker()
    if not dflash_blocker:
        return 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tasks = read_jsonl(TASKS)
    blocked = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        task_id = str(task.get("id", ""))
        lane = str(task.get("lane", ""))
        action = str(task.get("supervisor_action", ""))
        if lane != "frontier-dflash" and "dflash-compatibility" not in task_id and action != "dflash-compatibility-gate":
            continue
        task["status"] = "blocked"
        task["blocked_at"] = now
        task["blocked_reason"] = f"stale hard-blocked DFlash lane: {dflash_blocker}"
        task["supervisor_summary"] = {
            **(task.get("supervisor_summary") if isinstance(task.get("supervisor_summary"), dict) else {}),
            "reason": "quarantined_before_selection",
            "blocker": dflash_blocker,
            "next": "change_dflash_candidate_or_collect_janq_target_trace_data",
        }
        blocked += 1
    if blocked:
        write_jsonl(TASKS, tasks)
        mark_lane_exhausted(
            WORKSPACE,
            lane="frontier-dflash",
            reason="DFlash/JANQ compatibility is hard-blocked; suppress stale lane tasks until the draft candidate changes",
            evidence={"blocker": dflash_blocker, "blocked_tasks": blocked},
        )
        append_jsonl(
            FINDINGS,
            {
                "timestamp": now,
                "finding": "supervisor quarantined stale DFlash tasks before selection",
                "reason": dflash_blocker,
                "blocked_tasks": blocked,
                "next": "select_next_ready_task",
            },
        )
    return blocked


def block_ready_exhausted_lane_tasks() -> int:
    exhausted = active_exhausted_lanes()
    if not exhausted:
        return 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tasks = read_jsonl(TASKS)
    blocked = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        lane = str(task.get("lane", ""))
        if lane not in exhausted or lane == "exhaustion-report":
            continue
        task["status"] = "blocked"
        task["blocked_at"] = now
        task["blocked_reason"] = f"lane is exhausted: {lane}"
        task["supervisor_summary"] = {
            **(task.get("supervisor_summary") if isinstance(task.get("supervisor_summary"), dict) else {}),
            "reason": "blocked_exhausted_lane_before_selection",
            "lane": lane,
            "next": "route_to_non_exhausted_frontier_or_prerequisite_lane",
        }
        blocked += 1
    if blocked:
        write_jsonl(TASKS, tasks)
        append_jsonl(
            FINDINGS,
            {
                "timestamp": now,
                "finding": "supervisor blocked ready tasks from exhausted lanes before autonomy",
                "blocked_tasks": blocked,
                "exhausted_lanes": sorted(exhausted),
                "next": "run frontier certification again",
            },
        )
    return blocked


def block_stale_model_bound_causal_tasks() -> int:
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tasks = read_jsonl(TASKS)
    blocked = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        task_id = str(task.get("id", ""))
        if str(task.get("lane", "")) != "causal-repair" and not task_id.startswith("causal-review-"):
            continue
        if task_runs_without_model(task):
            continue
        task["status"] = "blocked"
        task["blocked_at"] = now
        task["blocked_reason"] = "stale model-bound causal review; deterministic causal-review command supersedes this task"
        task["supervisor_summary"] = {
            **(task.get("supervisor_summary") if isinstance(task.get("supervisor_summary"), dict) else {}),
            "reason": "quarantined_before_selection",
            "next": "run supervisor causal-review instead of a model-bound read task",
        }
        blocked += 1
    if blocked:
        write_jsonl(TASKS, tasks)
        append_jsonl(
            FINDINGS,
            {
                "timestamp": now,
                "finding": "supervisor quarantined stale model-bound causal review tasks before selection",
                "blocked_tasks": blocked,
                "next": "run deterministic quality review or select next ready task",
            },
        )
    return blocked


def memory_snapshot() -> dict[str, int]:
    snapshot = {"free_mb": 0, "compressor_mb": 0, "swap_used_mb": 0, "pressure_free_pct": 0}
    try:
        output = subprocess.check_output(["/usr/bin/vm_stat"], text=True, stderr=subprocess.DEVNULL)
        page_size = 16384
        free_pages = speculative_pages = compressor_pages = 0
        for line in output.splitlines():
            if "page size of" in line:
                digits = "".join(ch for ch in line.split("page size of", 1)[1] if ch.isdigit())
                if digits:
                    page_size = int(digits)
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":", 1)[1].strip().rstrip("."))
            elif line.startswith("Pages speculative:"):
                speculative_pages = int(line.split(":", 1)[1].strip().rstrip("."))
            elif line.startswith("Pages occupied by compressor:"):
                compressor_pages = int(line.split(":", 1)[1].strip().rstrip("."))
        snapshot["free_mb"] = int((free_pages + speculative_pages) * page_size / 1048576)
        snapshot["compressor_mb"] = int(compressor_pages * page_size / 1048576)
    except Exception:
        pass
    try:
        output = subprocess.check_output(["/usr/sbin/sysctl", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL)
        if "used = " in output:
            snapshot["swap_used_mb"] = int(float(output.split("used = ", 1)[1].split("M", 1)[0].strip()))
    except Exception:
        pass
    try:
        output = subprocess.check_output(
            ["/usr/bin/memory_pressure"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        for line in output.splitlines():
            if "System-wide memory free percentage:" in line:
                snapshot["pressure_free_pct"] = int(line.rsplit(" ", 1)[-1].strip("%"))
                break
    except Exception:
        pass
    return snapshot


def model_ready() -> bool:
    url = os.environ.get("OPENCLAW_SPEED_RESEARCH_MODEL_HEALTH_URL", "http://127.0.0.1:8091/v1/models")
    try:
        result = subprocess.run(
            ["/usr/bin/curl", "-fsS", "--max-time", "2", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def gateway_health_url(args: argparse.Namespace) -> str:
    configured = getattr(args, "gateway_health_url", "") or os.environ.get(
        "OPENCLAW_SPEED_RESEARCH_GATEWAY_HEALTH_URL",
        "",
    )
    return str(configured or f"http://127.0.0.1:{getattr(args, 'gateway_port', 18789)}/health")


def gateway_ready(args: argparse.Namespace) -> bool:
    try:
        result = subprocess.run(
            ["/usr/bin/curl", "-fsS", "--max-time", "2", gateway_health_url(args)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def gateway_listener_pids(args: argparse.Namespace) -> list[int]:
    try:
        result = subprocess.run(
            [
                "/usr/sbin/lsof",
                "-nP",
                f"-tiTCP:{getattr(args, 'gateway_port', 18789)}",
                "-sTCP:LISTEN",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def stop_gateway_listeners(args: argparse.Namespace, log_file: Path, *, reason: str) -> None:
    pids = gateway_listener_pids(args)
    if not pids:
        return
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\nGateway recovery stopping listeners reason={reason} pids={pids}\n")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not gateway_listener_pids(args):
            return
        time.sleep(0.25)
    for pid in gateway_listener_pids(args):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def start_gateway(args: argparse.Namespace, log_file: Path, *, reason: str) -> tuple[bool, str]:
    if gateway_ready(args):
        return True, ""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    gateway_log = OPENCLAW_HOME / "logs" / "gateway-autoresearch.log"
    gateway_err = OPENCLAW_HOME / "logs" / "gateway-autoresearch.err.log"
    gateway_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.openclaw_bin,
        "gateway",
        "run",
        "--port",
        str(args.gateway_port),
        "--token",
        os.environ.get("OPENCLAW_GATEWAY_TOKEN", "local-dev-token"),
    ]
    env = os.environ.copy()
    env.setdefault("OPENCLAW_GATEWAY_TOKEN", "local-dev-token")
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\nGateway recovery start reason={reason}\n")
        file.write("$ " + " ".join(cmd[:-1] + ["<token>"]) + "\n")
        file.write(f"stdout={gateway_log} stderr={gateway_err}\n")
        file.flush()
    try:
        with gateway_log.open("a", encoding="utf-8") as out, gateway_err.open("a", encoding="utf-8") as err:
            subprocess.Popen(cmd, env=env, text=True, stdout=out, stderr=err, start_new_session=True)
    except Exception as error:
        return False, f"gateway start failed: {error}"
    deadline = time.monotonic() + args.gateway_start_timeout_seconds
    while time.monotonic() < deadline:
        if gateway_ready(args):
            return True, ""
        time.sleep(0.5)
    return False, f"gateway did not become ready within {args.gateway_start_timeout_seconds:.0f}s"


def recover_gateway(args: argparse.Namespace, log_file: Path, *, reason: str, force_restart: bool = False) -> tuple[bool, str]:
    if gateway_ready(args) and not force_restart:
        return True, ""
    if force_restart or gateway_listener_pids(args):
        stop_gateway_listeners(args, log_file, reason=reason)
    ok, issue = start_gateway(args, log_file, reason=reason)
    if ok:
        append_jsonl(
            FINDINGS,
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "task_id": "gateway-recovery",
                "finding": "autoresearch recovered the OpenClaw gateway before continuing",
                "reason": reason,
                "health_url": gateway_health_url(args),
            },
        )
        append_result(
            WORKSPACE,
            run_id=f"gateway-recovery-{int(time.time())}",
            status="keep",
            target="openclaw-gateway",
            hypothesis="autoresearch must recover gateway failures instead of burning cycles",
            commit=current_commit(),
            notes=f"reason={clean_tsv(reason)} health_url={gateway_health_url(args)}",
        )
        return True, ""
    return False, issue


def ensure_gateway_for_agent(args: argparse.Namespace, log_file: Path) -> tuple[bool, str]:
    if gateway_ready(args):
        return True, ""
    return recover_gateway(args, log_file, reason="pre-agent gateway health check")


def is_gateway_issue(issue: str) -> bool:
    text = issue.lower()
    return "gateway" in text or "embedded fallback" in text or "websocket" in text


def stop_openclaw_model_for_memory_recovery(args: argparse.Namespace, *, reason: str) -> None:
    log(f"memory recovery stopping OpenClaw-owned model: {reason}")
    subprocess.run(
        [
            "/bin/zsh",
            "-lc",
            "fpath=(/Users/kristian/.zfunc $fpath); autoload -Uz openclaw; openclaw model-stop",
        ],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=max(10, int(getattr(args, "memory_model_stop_timeout_seconds", 60))),
        check=False,
    )
    cooldown = float(getattr(args, "memory_cooldown_after_stop_seconds", 0.0) or 0.0)
    if cooldown > 0:
        log(f"memory recovery cooldown after model-stop: {cooldown:.0f}s")
        time.sleep(cooldown)


def memory_gate_reason(args: argparse.Namespace, snap: dict[str, int], *, ready: bool) -> str:
    min_free_mb = args.ready_min_free_mb if ready else args.min_free_mb
    min_pressure_free_pct = int(getattr(args, "min_pressure_free_pct", 0) or 0)
    recovered = bool(
        ready
        and snap.get("pressure_free_pct", 0) >= getattr(args, "recovered_pressure_free_pct", 20)
        and snap.get("free_mb", 0) >= min_free_mb
    )
    if min_pressure_free_pct > 0 and snap.get("pressure_free_pct", 0) and snap["pressure_free_pct"] < min_pressure_free_pct:
        return (
            f"pressureFree={snap['pressure_free_pct']}%<{min_pressure_free_pct}% "
            f"free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB swap={snap['swap_used_mb']}MB ready={ready}"
        )
    if not recovered and snap["compressor_mb"] >= args.max_compressor_mb:
        return (
            f"compressor={snap['compressor_mb']}MB>={args.max_compressor_mb}MB "
            f"free={snap['free_mb']}MB swap={snap['swap_used_mb']}MB pressureFree={snap.get('pressure_free_pct', 0)}% ready={ready}"
        )
    if not recovered and snap["swap_used_mb"] >= args.max_swap_mb:
        return (
            f"swap={snap['swap_used_mb']}MB>={args.max_swap_mb}MB "
            f"free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB pressureFree={snap.get('pressure_free_pct', 0)}% ready={ready}"
        )
    if snap["free_mb"] and snap["free_mb"] < min_free_mb:
        return (
            f"free={snap['free_mb']}MB<{min_free_mb}MB "
            f"compressor={snap['compressor_mb']}MB swap={snap['swap_used_mb']}MB "
            f"pressureFree={snap.get('pressure_free_pct', 0)}% ready={ready}"
        )
    return ""


def wait_for_memory(args: argparse.Namespace) -> tuple[bool, str]:
    started = time.monotonic()
    last_reason = ""
    stopped_model = False
    stable_samples = 0
    required_stable_samples = max(1, int(getattr(args, "memory_stable_samples", 1) or 1))
    while True:
        snap = memory_snapshot()
        ready = model_ready()
        reason = memory_gate_reason(args, snap, ready=ready)
        if not reason:
            stable_samples += 1
            if stable_samples >= required_stable_samples:
                return True, ""
            log(
                f"memory gate stabilizing: sample={stable_samples}/{required_stable_samples} "
                f"free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB "
                f"swap={snap['swap_used_mb']}MB pressureFree={snap.get('pressure_free_pct', 0)}%"
            )
            time.sleep(max(0.5, float(getattr(args, "memory_stable_interval_seconds", 2.0))))
            continue
        stable_samples = 0
        last_reason = f"memory gate waiting: {reason}"
        if args.max_memory_wait_seconds > 0 and time.monotonic() - started >= args.max_memory_wait_seconds:
            if args.memory_stop_model_after_wait and ready and not stopped_model:
                stop_openclaw_model_for_memory_recovery(args, reason=f"wait budget exceeded: {last_reason}")
                stopped_model = True
                started = time.monotonic()
                continue
            return False, f"{last_reason}; exceeded {args.max_memory_wait_seconds:.0f}s wait budget"
        log(last_reason)
        time.sleep(args.memory_wait_seconds)


def active_memory_circuit_reason(args: argparse.Namespace, snap: dict[str, int]) -> str:
    """Abort an in-flight agent turn before macOS/Metal reaches crash territory."""
    compressor_limit = args.active_max_compressor_mb or args.max_compressor_mb
    swap_limit = args.active_max_swap_mb or args.max_swap_mb
    min_pressure_free_pct = int(getattr(args, "active_min_pressure_free_pct", 0) or 0)
    recovered = bool(
        snap.get("pressure_free_pct", 0) >= getattr(args, "recovered_pressure_free_pct", 20)
        and snap.get("free_mb", 0) >= args.active_min_free_mb
    )
    if min_pressure_free_pct > 0 and snap.get("pressure_free_pct", 0) and snap["pressure_free_pct"] < min_pressure_free_pct:
        return (
            f"pressureFree={snap['pressure_free_pct']}%<{min_pressure_free_pct}% "
            f"free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB swap={snap['swap_used_mb']}MB"
        )
    if not recovered and snap["compressor_mb"] >= compressor_limit:
        return (
            f"compressor={snap['compressor_mb']}MB>={compressor_limit}MB "
            f"free={snap['free_mb']}MB swap={snap['swap_used_mb']}MB pressureFree={snap.get('pressure_free_pct', 0)}%"
        )
    if not recovered and snap["swap_used_mb"] >= swap_limit:
        return (
            f"swap={snap['swap_used_mb']}MB>={swap_limit}MB "
            f"free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB pressureFree={snap.get('pressure_free_pct', 0)}%"
        )
    if (
        snap["free_mb"]
        and snap["free_mb"] < args.active_min_free_mb
        and (
            snap["compressor_mb"] >= args.active_low_free_pressure_compressor_mb
            or snap["swap_used_mb"] >= args.active_low_free_pressure_swap_mb
        )
    ):
        return (
            f"free={snap['free_mb']}MB<{args.active_min_free_mb}MB "
            f"compressor={snap['compressor_mb']}MB swap={snap['swap_used_mb']}MB "
            f"pressureFree={snap.get('pressure_free_pct', 0)}%"
        )
    return ""


def synthesis_task() -> dict[str, object]:
    return {
        "id": "synthesize-speed-ideas",
        "target": "ideas.md/STRATEGY.md/tasks.jsonl",
        "hypothesis": "Completed baselines must produce ranked ideas and new measurable OpenClaw speed tasks.",
        "metric": "ranked_ideas_and_seeded_tasks",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research synthesize --kind frontier",
    }


def recurring_decode_tasks(cycle: int) -> list[dict[str, object]]:
    suffix = f"cycle-{cycle:03d}-{int(time.time())}"
    return [
        {
            "id": f"decode-repeatability-{suffix}",
            "status": "ready",
            "priority": 88,
            "lane": "production-mtp",
            "target": "tui-decode-sample",
            "hypothesis": "Each new research tranche starts by remeasuring TUI-relevant real decode TPS on the live MTP setup.",
            "metric": "decode_tps",
            "benchmark_mode": "decode-sample",
            "guard_checks": ["memory_ok", "no_reasoning_leak", "no_sse_timeout"],
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
        },
        {
            "id": f"mtp-acceptance-review-{suffix}",
            "status": "ready",
            "priority": 84,
            "lane": "production-mtp",
            "target": "openclaw-model-proxy.log",
            "hypothesis": "Recent MTP acceptance logs should guide the next implementation candidate instead of repeating generic benchmarks.",
            "metric": "mean_accept",
            "guard_checks": ["one_narrow_tool", "no_loop"],
            "next_action": "tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log",
        },
        {
            "id": f"implementation-bridge-{suffix}",
            "status": "ready",
            "priority": 72,
            "lane": "implementation-gate",
            "task_type": "supervisor",
            "supervisor_action": "implementation-bridge",
            "target": "openclaw/openclaw-speed-research.py",
            "source_files": ["openclaw/openclaw-speed-research.py", "openclaw/test-speed-research.py"],
            "hypothesis": "When synthesis identifies a grounded TUI decode/MTP improvement, seed deterministic supervisor tasks instead of asking the LLM to improvise a patch.",
            "metric": "decode_tps_delta",
            "guard_checks": ["tests_pass", "no_opencode_changes", "memory_gate", "rollback_path"],
            "acceptance": "Supervisor records which deterministic follow-up tasks were ready or seeded; no generic implementation turn is required.",
            "rollback": "No source rollback needed; the bridge only advances the deterministic queue.",
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research synthesize --kind frontier",
        },
    ]


def enqueue_recurring_decode_tasks(cycle: int, reason: str) -> int:
    existing = read_jsonl(TASKS)
    existing_ids = {str(task.get("id", "")) for task in existing}
    additions = [task for task in recurring_decode_tasks(cycle) if str(task["id"]) not in existing_ids]
    if not additions:
        return 0
    write_jsonl(TASKS, existing + additions)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    append_jsonl(
        FINDINGS,
        {
            "timestamp": timestamp,
            "task_id": "supervisor-recurring-decode-refill",
            "finding": "supervisor refilled the queue with a new decode/MTP research and implementation tranche",
            "reason": reason,
            "seeded_tasks": [task["id"] for task in additions],
            "next": "continue_autopilot_loop",
        },
    )
    append_result(
        WORKSPACE,
        run_id=f"recurring-refill-{cycle}",
        status="keep",
        target="autopilot-refill",
        hypothesis="empty queues should bridge into another decode/MTP research and implementation cycle",
        commit=current_commit(),
        notes=f"seeded_tasks={len(additions)} reason={clean_tsv(reason)}",
    )
    return len(additions)


def continuation_prompt(
    cycle: int,
    stalled_cycles: int,
    last_issue: str = "",
    selected_task: dict[str, object] | None = None,
) -> str:
    pressure = ""
    if stalled_cycles >= 2:
        pressure = (
            "\n\nPrevious cycles did not record enough durable progress. In this turn, choose the smallest "
            "realistic OpenClaw/Gemma4 JANG MTP decode-speed experiment and record either a result "
            "row in results.tsv or a frontier idea in ideas.md before ending."
        )
    if last_issue:
        pressure += (
            f"\n\nLast cycle issue: {last_issue}. Recover with the next safe Bootstrap Ladder action. "
            "Your next tool call must be one of: read `/Users/kristian/.openclaw/research/speed/SUMMARY.md`, "
            "read `/Users/kristian/.openclaw/research/speed/results-recent.tsv`, run "
            "`/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --quick`, or run "
            "`git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`."
        )
    task_summary = next_task_summary()
    if task_summary:
        task_summary = f"\n\nCurrent machine-readable research queue:\n{task_summary}"
    if selected_task is None:
        selected_task = select_next_task(WORKSPACE)
    task_contract = ""
    if selected_task:
        task_contract = (
            "\n\nSelected task for this cycle:\n"
            f"- id: {selected_task.get('id', 'unknown')}\n"
            f"- target: {selected_task.get('target', 'unknown')}\n"
            f"- hypothesis: {selected_task.get('hypothesis', 'unknown')}\n"
            f"- metric: {selected_task.get('metric', 'unknown')}\n"
            f"- required next action: {selected_task.get('next_action', 'record evidence')}\n"
        )
        if selected_task.get("id") == "synthesize-speed-ideas":
            task_contract += (
                "\nThe benchmark queue is exhausted. Your next tool call must be exactly "
                "`/Users/kristian/.openclaw/bin/openclaw-speed-research synthesize --kind frontier`. "
                "Do not run another benchmark until synthesis has created new measurable tasks.\n"
            )
        if selected_task.get("task_type") == "implementation":
            task_contract += (
                "\nThis is an implementation task, not another measurement loop. Follow the implementation gate: "
                "read the implementation skill if needed, inspect only the listed target files, make the smallest "
                "OpenClaw-only patch, run the focused test, and record keep/discard/blocked evidence. "
                "Do not change live model settings, do not touch opencode, and do not bundle unrelated cleanup.\n"
                f"- source files: {', '.join(str(item) for item in selected_task.get('source_files', []))}\n"
                f"- acceptance: {selected_task.get('acceptance', 'focused tests and recorded evidence')}\n"
                f"- rollback: {selected_task.get('rollback', 'revert only this experiment')}\n"
            )
    return (
        "Continue OpenClaw Speed Autoresearch autonomously.\n\n"
        f"Workspace: {WORKSPACE}\n"
        f"Program: {PROGRAM}\n\n"
        "Use program.md as the installed policy, but do not read it this turn. Do one bounded unit of useful work. "
        "Primary scope: improve normal `openclaw tui` decode speed and visible response smoothness first. "
        "Autoresearch self-improvement is secondary and should only be done when it makes the TUI decode-speed loop safer, more deterministic, or more likely to produce a clean implementation.\n\n"
        "Prefer realistic TUI decode work over toy prompts: "
        "MTP acceptance, drafter block size, drafter quantization, JANQ calibration, MLX/VLM MTP loop overhead, "
        "Metal/KV/cache memory, or grounded frontier proposals toward 30+ and then 50-70 tok/s.\n\n"
        "Best next actions are: run a specific benchmark mode such as "
        "`/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample`, "
        "read recent MTP logs with `tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log`, "
        "read `/Users/kristian/.openclaw/research/speed/SUMMARY.md`, read one named OpenClaw source file, "
        "or run `git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`. "
        "Do not run setup commands, `find`, recursive `ls`, recursive grep, or broad local search.\n\n"
        "If implementing, first read `/Users/kristian/.openclaw/research/speed/implementation-skill.md`. "
        "Do not touch opencode. Do not change the primary model. Use one narrow tool call per assistant turn. "
        "Do not ask me whether to continue. If a test or benchmark cannot run safely, record blocked evidence "
        "and move to the next implementable item.\n\n"
        "Cycle contract: each cycle must finish with one durable artifact: a results.tsv row, ideas.md note, "
        "STRATEGY.md update, findings.jsonl entry, experiments.jsonl entry, tasks.jsonl update, benchmark JSON, "
        "source patch, test result, rejection entry, or explicit blocked row. "
        "Benchmark commands already write results.tsv and benchmark JSON; after running one, do not append another "
        "results row by hand. Use a later cycle for synthesis or task updates. "
        "Task completion is supervisor-owned: do not manually mark tasks done. "
        "Do not repeat quick-health, TTFT, prompt-size, tool-roundtrip, or autoresearch meta-work unless they directly support a TUI decode/MTP hypothesis. "
        "No quality artifact means the supervisor will narrow the next cycle automatically."
        f"{task_summary}"
        f"{task_contract}"
        f"\n\nAutopilot cycle: {cycle}.{pressure}{recovery_instruction(stalled_cycles, last_issue, selected_task)}"
    )


def summarize_issue(stdout: str, stderr: str, returncode: int) -> str:
    text = f"{stdout[-4000:]}\n{stderr[-4000:]}".lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in text:
            return pattern
    if returncode == 124:
        return "turn timeout"
    if returncode < 0:
        signal_name = {
            -6: "SIGABRT",
            -9: "SIGKILL",
            -11: "SIGSEGV",
            -15: "SIGTERM",
        }.get(returncode, f"signal {-returncode}")
        return f"fatal process exit via {signal_name}"
    if returncode != 0:
        return f"agent exit {returncode}"
    return ""


def early_failure_reason(text: str) -> str:
    for pattern, reason in EARLY_FAILURE_PATTERNS:
        if pattern in text:
            return reason
    return ""


def read_log_since(path: Path, offset: int, limit: int = 12000) -> str:
    try:
        size = path.stat().st_size
        start = min(offset, size)
        if size - start > limit:
            start = size - limit
        with path.open("rb") as file:
            file.seek(start)
            return file.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def session_tool_result_count(session: str) -> int:
    count = 0
    legacy = session_jsonl_path(session)
    if legacy.exists():
        try:
            count += sum(
                1
                for line in legacy.read_text(encoding="utf-8", errors="replace").splitlines()
                if '"role":"toolResult"' in line
            )
        except OSError:
            pass
    trajectory = session_trajectory_path(session)
    if trajectory.exists():
        try:
            for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
                if '"toolMetas"' in line:
                    count += line.count('"toolName"')
        except OSError:
            pass
    return count


def session_jsonl_path(session: str) -> Path:
    return OPENCLAW_HOME / "agents" / "main" / "sessions" / f"{session}.jsonl"


def session_pointer_path(session: str) -> Path:
    return OPENCLAW_HOME / "agents" / "main" / "sessions" / f"{session}.trajectory-path.json"


def session_trajectory_path(session: str) -> Path:
    return OPENCLAW_HOME / "agents" / "main" / "sessions" / f"{session}.trajectory.jsonl"


def session_paths(session: str) -> tuple[Path, ...]:
    return (session_jsonl_path(session), session_pointer_path(session), session_trajectory_path(session))


def session_exists(session: str) -> bool:
    return any(path.exists() for path in session_paths(session))


def session_mtime(session: str) -> float:
    return max(file_mtime(path) for path in session_paths(session))


def stop_process_tree(process: subprocess.Popen[str], *, terminate_grace_seconds: float = 8.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=terminate_grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
        process.wait(timeout=5)


def run_turn(
    args: argparse.Namespace,
    session: str,
    cycle: int,
    stalled_cycles: int,
    last_issue: str,
    log_file: Path,
    selected_task: dict[str, object] | None = None,
) -> tuple[int, str]:
    prompt = continuation_prompt(cycle, stalled_cycles, last_issue, selected_task)
    tool_result_cap = args.max_tool_results_per_turn
    if selected_task and selected_task.get("task_type") == "implementation":
        tool_result_cap = args.max_implementation_tool_results_per_turn
    gateway_ok, gateway_issue = ensure_gateway_for_agent(args, log_file)
    if not gateway_ok:
        return 124, f"gateway recovery failed before agent turn: {gateway_issue}"
    cmd = [
        args.openclaw_bin,
        "agent",
        "--session-id",
        session,
        "--message",
        prompt,
        "--timeout",
        str(args.turn_timeout_seconds),
        "--thinking",
        args.thinking,
        "--json",
    ]
    env = os.environ.copy()
    env.setdefault("OPENCLAW_GATEWAY_TOKEN", "local-dev-token")
    env.setdefault("OPENCLAW_AGENT_RUNTIME", "pi")
    env.setdefault("OPENCLAW_DISABLE_MLX_PROVIDER_PLUGIN_HOOKS", "1")
    started = time.monotonic()
    starting_tool_results = session_tool_result_count(session)
    last_session_mtime = session_mtime(session)
    first_new_tool_at = 0.0
    process: subprocess.Popen[str] | None = None
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n===== cycle {cycle} session {session} start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        file.write("$ " + " ".join(cmd[:5] + ["<prompt>", *cmd[6:]]) + "\n")
        file.flush()
        log_offset = file.tell()
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                text=True,
                stdout=file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            INTERRUPT_CONTEXT["child_process"] = process
            deadline = started + args.turn_timeout_seconds + args.turn_timeout_grace_seconds
            next_heartbeat = started + 30
            next_tool_check = started + 1
            next_failure_check = started + 2
            next_memory_check = started + max(1.0, args.active_memory_check_seconds)
            capped_by_tool_results = False
            while process.poll() is None:
                now = time.monotonic()
                current_session_mtime = session_mtime(session)
                if current_session_mtime > last_session_mtime:
                    last_session_mtime = current_session_mtime
                if now >= deadline:
                    file.write(f"\nTIMEOUT after {now - started:.1f}s\n")
                    file.flush()
                    stop_process_tree(process, terminate_grace_seconds=2)
                    tail = log_file.read_text(encoding="utf-8", errors="replace")[-8000:]
                    return 124, summarize_issue(tail, "", 124)
                if (
                    args.session_start_timeout_seconds > 0
                    and not session_exists(session)
                    and now - started >= args.session_start_timeout_seconds
                ):
                    file.write(f"\nSESSION START TIMEOUT after {now - started:.1f}s\n")
                    file.flush()
                    stop_process_tree(process)
                    return 124, "agent session bootstrap timeout"
                if (
                    args.session_idle_timeout_seconds > 0
                    and session_exists(session)
                    and last_session_mtime > 0
                    and now - last_session_mtime >= args.session_idle_timeout_seconds
                ):
                    file.write(f"\nSESSION IDLE TIMEOUT after {now - last_session_mtime:.1f}s without session update\n")
                    file.flush()
                    stop_process_tree(process)
                    return 124, "agent session idle timeout"
                if now >= next_heartbeat:
                    log(
                        f"cycle={cycle} session={session} still running "
                        f"elapsed={now - started:.0f}s timeout={args.turn_timeout_seconds}s"
                    )
                    next_heartbeat = now + 30
                if now >= next_failure_check:
                    failure = early_failure_reason(read_log_since(log_file, log_offset))
                    if failure:
                        file.write(f"\nEARLY FAILURE after {now - started:.1f}s: {failure}\n")
                        file.flush()
                        stop_process_tree(process)
                        if is_gateway_issue(failure):
                            recovered, recovery_issue = recover_gateway(
                                args,
                                log_file,
                                reason=failure,
                                force_restart=True,
                            )
                            if recovered:
                                return 124, "gateway recovered after early failure"
                            return 124, f"{failure}; gateway recovery failed: {recovery_issue}"
                        return 124, failure
                    next_failure_check = now + 2
                if args.active_memory_check_seconds > 0 and now >= next_memory_check:
                    memory_issue = active_memory_circuit_reason(args, memory_snapshot())
                    if memory_issue:
                        file.write(f"\nMEMORY CIRCUIT BREAKER after {now - started:.1f}s: {memory_issue}\n")
                        file.flush()
                        stop_process_tree(process, terminate_grace_seconds=2)
                        return 124, f"memory circuit breaker: {memory_issue}"
                    next_memory_check = now + args.active_memory_check_seconds
                if tool_result_cap > 0 and now >= next_tool_check:
                    tool_results = max(0, session_tool_result_count(session) - starting_tool_results)
                    if tool_results > 0 and first_new_tool_at == 0.0:
                        first_new_tool_at = now
                    if tool_results > tool_result_cap:
                        file.write(
                            f"\nTOOL RESULT CAP after {tool_results} tool results "
                            f"and {now - started:.1f}s\n"
                        )
                        file.flush()
                        capped_by_tool_results = True
                        stop_process_tree(process)
                        break
                    if (
                        tool_results >= tool_result_cap
                        and first_new_tool_at > 0
                        and now - first_new_tool_at >= args.tool_result_synthesis_grace_seconds
                    ):
                        file.write(
                            f"\nTOOL RESULT SYNTHESIS GRACE elapsed after {tool_results} tool results "
                            f"and {now - started:.1f}s\n"
                        )
                        file.flush()
                        capped_by_tool_results = True
                        stop_process_tree(process)
                        break
                    next_tool_check = now + 1
                time.sleep(1)
        except KeyboardInterrupt:
            file.write(f"\nUSER INTERRUPT after {time.monotonic() - started:.1f}s; stopping child process tree\n")
            file.flush()
            if process is not None:
                stop_process_tree(process, terminate_grace_seconds=2)
            raise
        finally:
            if process is not None and INTERRUPT_CONTEXT.get("child_process") is process:
                INTERRUPT_CONTEXT["child_process"] = None
            file.flush()
        if process is None:
            return 124, "agent process failed before launch"
        returncode = process.returncode if process.returncode is not None else 0
        if capped_by_tool_results and returncode != 0:
            returncode = 0
        file.write(f"\n===== cycle {cycle} exit {returncode} elapsed {time.monotonic() - started:.1f}s =====\n")
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-12000:]
        return returncode, summarize_issue(tail, "", returncode)


def complete_implementation_task(
    task: dict[str, object] | None,
    progress_reasons: list[str],
    *,
    commit: str,
) -> dict[str, object] | None:
    if not task or task.get("task_type") != "implementation":
        return None
    if task.get("status", "ready") not in {"ready", "rework"}:
        return None
    if "repo patch" not in progress_reasons:
        return None
    if not any(reason in progress_reasons for reason in ("results row", "benchmark artifact", "findings update", "experiments update")):
        return None
    tasks = read_jsonl(TASKS)
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    updated = False
    for item in tasks:
        if item.get("id") != task.get("id"):
            continue
        item["status"] = "done"
        item["completed_at"] = completed_at
        item["completion_commit"] = commit
        item["completion_artifacts"] = progress_reasons
        updated = True
        break
    if not updated:
        return None
    write_jsonl(TASKS, tasks)
    summary = {
        "timestamp": completed_at,
        "task_id": str(task.get("id", "unknown")),
        "status": "implementation-recorded",
        "target": str(task.get("target", "")),
        "artifacts": progress_reasons,
        "commit": commit,
    }
    append_jsonl(EXPERIMENTS, summary)
    append_jsonl(
        WORKSPACE / "promotion-decisions.jsonl",
        promotion_decision(task, summary, status="keep"),
    )
    append_jsonl(
        FINDINGS,
        {
            "timestamp": completed_at,
            "task_id": summary["task_id"],
            "finding": "implementation task produced a source patch plus durable evidence; supervisor marked it done",
            "evidence": summary,
            "next": "run_followup_benchmark_or_review",
        },
    )
    return summary


def complete_supervisor_task(
    task: dict[str, object] | None,
    *,
    status: str,
    summary: dict[str, object],
    commit: str,
) -> dict[str, object] | None:
    if not task or task.get("status", "ready") not in {"ready", "rework"}:
        return None
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    task_id = str(task.get("id", "unknown"))
    tasks = read_jsonl(TASKS)
    changed = False
    for item in tasks:
        if item.get("id") != task_id:
            continue
        item["status"] = "done" if status == "keep" else "blocked"
        item["completed_at" if status == "keep" else "blocked_at"] = completed_at
        if status == "keep":
            item.pop("blocked_at", None)
            item.pop("blocked_reason", None)
        else:
            item.pop("completed_at", None)
        item["completion_commit"] = commit
        item["supervisor_summary"] = summary
        changed = True
        break
    if not changed:
        return None
    write_jsonl(TASKS, tasks)
    row = {
        "timestamp": completed_at,
        "task_id": task_id,
        "status": "supervisor-task-done" if status == "keep" else "supervisor-task-blocked",
        "target": str(task.get("target", "")),
        "summary": summary,
        "commit": commit,
    }
    append_jsonl(EXPERIMENTS, row)
    append_jsonl(
        WORKSPACE / "promotion-decisions.jsonl",
        promotion_decision(task, summary, status=status),
    )
    append_jsonl(
        FINDINGS,
        {
            "timestamp": completed_at,
            "task_id": task_id,
            "finding": "supervisor executed deterministic task and advanced the queue without an LLM tool turn",
            "evidence": row,
            "next": "select_next_ready_task",
        },
    )
    return row


def mark_task_rework(
    task: dict[str, object] | None,
    *,
    reason: str,
    summary: dict[str, object],
    commit: str,
    max_reworks: int = 2,
) -> bool:
    if not task:
        return False
    task_id = str(task.get("id", ""))
    tasks = read_jsonl(TASKS)
    changed = False
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for item in tasks:
        if item.get("id") != task_id:
            continue
        attempts = int(item.get("rework_attempts") or 0) + 1
        item["rework_attempts"] = attempts
        item["last_rework_at"] = now
        item["last_rework_reason"] = reason
        item["supervisor_summary"] = summary
        if attempts <= max_reworks:
            item["status"] = "rework"
            item["priority"] = int(item.get("priority", 0)) + 5
            item["next_action"] = "repair the existing patch or record a blocked row; do not invent a new unrelated idea"
        else:
            item["status"] = "blocked"
            item["blocked_at"] = now
            item["blocked_reason"] = f"rework attempts exhausted: {reason}"
        changed = True
        break
    if not changed:
        return False
    write_jsonl(TASKS, tasks)
    append_jsonl(
        FINDINGS,
        {
            "timestamp": now,
            "task_id": task_id,
            "finding": "metric/idea may still be useful but guard failed; supervisor routed to bounded rework instead of losing the approach",
            "reason": reason,
            "summary": summary,
            "commit": commit,
        },
    )
    return True


def start_model_for_supervisor_benchmark(log_file: Path, timeout_seconds: float) -> None:
    if model_ready():
        return
    cmd = [
        "/bin/zsh",
        "-lc",
        "fpath=(/Users/kristian/.zfunc $fpath); autoload -Uz openclaw; openclaw model-start",
    ]
    with log_file.open("a", encoding="utf-8") as file:
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        subprocess.run(cmd, text=True, stdout=file, stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False)


def ensure_model_for_supervisor_task(
    log_file: Path,
    timeout_seconds: float,
    *,
    task_id: str,
    reason: str,
) -> tuple[bool, str]:
    """Start and verify the local model before deterministic model-bound work.

    Benchmark-like supervisor tasks are deterministic, but they are not useful
    unless the OpenAI-compatible model endpoint is actually alive. Failing here
    avoids burning a sweep artifact full of connection-refused samples.
    """
    try:
        start_model_for_supervisor_benchmark(log_file, timeout_seconds)
    except subprocess.TimeoutExpired:
        return False, "supervisor model start timeout"
    if model_ready():
        return True, ""
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            "\nSUPERVISOR MODEL PREFLIGHT FAILED "
            f"task={task_id} reason={reason}: endpoint unavailable after model-start\n"
        )
    return False, "model endpoint unavailable after model-start"


def run_supervisor_benchmark_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
    *,
    fallback: bool = False,
) -> tuple[int, str]:
    mode = benchmark_mode_for_task(task) or "decode-sample"
    cmd = [args.research_helper_bin, "benchmark"]
    if mode == "quick-health":
        cmd.append("--quick")
    else:
        cmd.extend(["--mode", mode])
    cmd.extend(["--timeout", str(args.supervisor_benchmark_timeout_seconds)])
    ready, start_issue = ensure_model_for_supervisor_task(
        log_file,
        args.model_start_timeout_seconds,
        task_id=str(task.get("id", "fallback")),
        reason="supervisor benchmark",
    )
    if not ready:
        return 124 if "timeout" in start_issue else 75, start_issue
    with log_file.open("a", encoding="utf-8") as file:
        label = "supervisor fallback benchmark" if fallback else "supervisor benchmark"
        file.write(f"\n===== cycle {cycle} session {session} {label} task={task.get('id', 'fallback')} =====\n")
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.supervisor_benchmark_timeout_seconds + 15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR BENCHMARK TIMEOUT\n")
            return 124, "supervisor benchmark timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout)
    if result.returncode != 0:
        reason = str((parsed or {}).get("reason") or f"supervisor benchmark exit {result.returncode}")
        complete_supervisor_task(
            task,
            status="blocked",
            summary={"mode": mode, "reason": reason, "result": parsed or {}},
            commit=current_commit(),
        )
        return result.returncode, reason
    if parsed and parsed.get("ok") is False:
        reason = str(parsed.get("reason") or "supervisor benchmark blocked")
        complete_supervisor_task(
            task,
            status="blocked",
            summary={"mode": mode, "reason": reason, "result": parsed},
            commit=current_commit(),
        )
        return 2, reason
    if parsed:
        append_jsonl(
            EXPERIMENTS,
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "task_id": str(task.get("id", "fallback")),
                "status": "supervisor-benchmark",
                "mode": mode,
                "fallback": fallback,
                "result": parsed,
            },
        )
        complete_supervisor_task(
            task,
            status="keep",
            summary={"mode": mode, "fallback": fallback, "result": parsed},
            commit=current_commit(),
        )
    return 0, ""


def parse_mtp_log_tail(text: str) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        if "mtp_rounds=" not in line and "tok_s=" not in line:
            continue
        tok_match = re.search(r"tok_s=([0-9.]+)", line)
        rounds_match = re.search(r"mtp_rounds=([0-9]+)", line)
        accept_match = re.search(r"mean_accept=([0-9.]+)", line)
        if not tok_match and not rounds_match and not accept_match:
            continue
        rows.append(
            {
                "tok_s": float(tok_match.group(1)) if tok_match else 0.0,
                "mtp_rounds": float(rounds_match.group(1)) if rounds_match else 0.0,
                "mean_accept": float(accept_match.group(1)) if accept_match else 0.0,
            }
        )
    if not rows:
        return {"sample_count": 0}
    tok_values = [row["tok_s"] for row in rows if row["tok_s"] > 0]
    round_values = [row["mtp_rounds"] for row in rows if row["mtp_rounds"] > 0]
    accept_values = [row["mean_accept"] for row in rows if row["mean_accept"] > 0]
    return {
        "sample_count": len(rows),
        "mean_tok_s": round(sum(tok_values) / len(tok_values), 3) if tok_values else None,
        "mean_mtp_rounds": round(sum(round_values) / len(round_values), 3) if round_values else None,
        "mean_accept": round(sum(accept_values) / len(accept_values), 3) if accept_values else None,
    }


def run_supervisor_log_review_task(
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    source = Path(os.environ.get("OPENCLAW_MODEL_PROXY_LOG", "/Users/kristian/.openclaw/logs/openclaw-model-proxy.log"))
    try:
        text = "\n".join(source.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
    except OSError as error:
        return 2, f"supervisor log review failed: {error}"
    summary = parse_mtp_log_tail(text)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n===== cycle {cycle} session {session} supervisor log review task={task.get('id', 'unknown')} =====\n")
        file.write(f"$ tail -n 80 {source}\n")
        file.write(json.dumps(summary, indent=2) + "\n")
    if int(summary.get("sample_count") or 0) <= 0:
        append_result(
            WORKSPACE,
            run_id=f"supervisor-log-review-{cycle}",
            status="blocked",
            target=str(task.get("target", "openclaw-model-proxy.log")),
            hypothesis=str(task.get("hypothesis", "parse recent MTP acceptance evidence")),
            commit=current_commit(),
            notes="no mtp/tok_s lines found in recent log tail",
        )
        return 2, "no MTP acceptance evidence in recent log tail"
    append_result(
        WORKSPACE,
        run_id=f"supervisor-log-review-{cycle}",
        status="keep",
        target=str(task.get("target", "openclaw-model-proxy.log")),
        hypothesis=str(task.get("hypothesis", "parse recent MTP acceptance evidence")),
        commit=current_commit(),
        notes=(
            f"samples={summary.get('sample_count')} mean_tok_s={summary.get('mean_tok_s')} "
            f"mean_mtp_rounds={summary.get('mean_mtp_rounds')} mean_accept={summary.get('mean_accept')}"
        ),
    )
    complete_supervisor_task(task, status="keep", summary=summary, commit=current_commit())
    return 0, ""


def run_supervisor_mtp_report_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    lines = str(task.get("lines") or os.environ.get("OPENCLAW_MTP_REPORT_LINES", "160"))
    cmd = [args.research_helper_bin, "mtp-report", "--lines", lines]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor mtp report "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR MTP REPORT TIMEOUT\n")
            return 124, "supervisor mtp report timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor mtp report exit {result.returncode}")
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    complete_supervisor_task(task, status="keep", summary=parsed, commit=current_commit())
    return 0, ""


def run_supervisor_profile_variant_guard(
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    reason = "profile variant task requires a dedicated paired-control runner; raw benchmark would be invalid"
    plan = paired_profile_plan(task)
    plan_path = WORKSPACE / "experiments" / f"paired-profile-plan-{task.get('id', 'unknown')}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "reason": reason,
        "target": str(task.get("target", "")),
        "guard_checks": task.get("guard_checks", []),
        "paired_plan": str(plan_path),
        "next": "implement bounded profile-variant runner with restore-before/after and same-prompt comparison",
    }
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor profile-variant guard "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write(json.dumps(summary, indent=2) + "\n")
    append_result(
        WORKSPACE,
        run_id=f"supervisor-profile-variant-{cycle}",
        status="blocked",
        target=str(task.get("target", "profile-variant")),
        hypothesis=str(task.get("hypothesis", "profile variant requires paired benchmark control")),
        commit=current_commit(),
        notes=reason,
    )
    complete_supervisor_task(task, status="blocked", summary=summary, commit=current_commit())
    return 2, reason


def run_supervisor_implementation_bridge(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    before_ids = {
        str(item.get("id", ""))
        for item in read_jsonl(TASKS)
        if item.get("status", "ready") in {"ready", "rework"}
    }
    ok, issue = run_supervisor_synthesis(args, cycle, session, log_file)
    stale_lane_blocked = block_stale_hard_blocked_lane_tasks()
    stale_causal_blocked = block_stale_model_bound_causal_tasks()
    after_ready = [
        item
        for item in read_jsonl(TASKS)
        if item.get("status", "ready") in {"ready", "rework"}
    ]
    contract_issues = {
        str(item.get("id", "")): task_contract_issues(WORKSPACE, item)
        for item in after_ready
    }
    deterministic = [
        str(item.get("id", ""))
        for item in after_ready
        if item.get("task_type") == "supervisor" and str(item.get("id", "")) != str(task.get("id", ""))
        and not contract_issues.get(str(item.get("id", "")), {}).get("blockers")
    ]
    seeded = [task_id for task_id in deterministic if task_id not in before_ids]
    blocked_contracts = {
        task_id: issue["blockers"]
        for task_id, issue in contract_issues.items()
        if issue.get("blockers")
    }
    terminal_no_work = issue == "supervisor synthesis terminal no-work" and not deterministic
    summary = {
        "synthesis_ok": ok,
        "issue": issue,
        "terminal_no_work": terminal_no_work,
        "seeded_deterministic_tasks": seeded,
        "ready_deterministic_tasks": deterministic[:12],
        "contract_blockers": blocked_contracts,
        "stale_lane_blocked": stale_lane_blocked,
        "stale_causal_blocked": stale_causal_blocked,
        "next": (
            "run implementation-handoff-audit to route one concrete prerequisite"
            if terminal_no_work
            else "select_next_ready_supervisor_task"
        ),
    }
    status = "keep" if (ok and deterministic) or terminal_no_work else "blocked"
    append_result(
        WORKSPACE,
        run_id=f"supervisor-implementation-bridge-{cycle}",
        status=status,
        target=str(task.get("target", "implementation-bridge")),
        hypothesis=str(task.get("hypothesis", "bridge synthesis into deterministic implementation tasks")),
        commit=current_commit(),
        notes=(
            f"seeded={len(seeded)} ready_deterministic={len(deterministic)} "
            f"terminal_no_work={terminal_no_work} issue={clean_tsv(issue)}"
        ),
    )
    complete_supervisor_task(task, status=status, summary=summary, commit=current_commit())
    return (0, "") if status == "keep" else (2, issue or "no deterministic implementation tasks available")


def run_supervisor_patch_execute_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    patch_file = str(task.get("patch_file") or task.get("patch") or "")
    if not patch_file:
        reason = "patch-execute task missing patch_file"
        complete_supervisor_task(task, status="blocked", summary={"reason": reason}, commit=current_commit())
        return 2, reason
    source_files = ",".join(str(item) for item in task.get("source_files", []) if item)
    tests = ";".join(str(item) for item in task.get("tests", []) if item)
    cmd = [
        args.research_helper_bin,
        "patch-execute",
        "--patch-file",
        patch_file,
        "--task-id",
        str(task.get("id", "patch-execute")),
        "--hypothesis",
        str(task.get("hypothesis", "canary-test allowlisted patch before promotion")),
        "--source-files",
        source_files,
    ]
    if tests:
        cmd.extend(["--tests", tests])
    if task.get("canary_only", False):
        cmd.append("--canary-only")
    if task.get("allow_architectural", False):
        cmd.append("--allow-architectural")
    crabbox_evidence_file = str(task.get("crabbox_evidence_file") or task.get("crabbox_evidence") or "")
    if crabbox_evidence_file:
        cmd.extend(["--crabbox-evidence-file", crabbox_evidence_file])
    if task.get("rollback_rehearsal_ok", False):
        cmd.append("--rollback-rehearsal-ok")
    approval_file = str(task.get("architectural_approval_file") or args.architectural_approval_file or "")
    if approval_file:
        cmd.extend(["--architectural-approval-file", approval_file])
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor patch execute "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.patch_execute_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR PATCH EXECUTOR TIMEOUT\n")
            return 124, "supervisor patch executor timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor patch executor exit {result.returncode}")
        if "canary tests failed" in reason or "patch apply failed in canary" in reason:
            if mark_task_rework(task, reason=reason, summary={"result": parsed}, commit=current_commit()):
                return result.returncode or 2, f"rework queued: {reason}"
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    complete_supervisor_task(task, status="keep", summary=parsed, commit=current_commit())
    return 0, ""


def run_supervisor_drafter_bottleneck_review_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    action = (
        "drafter-adapter-method-contract"
        if is_supervisor_drafter_adapter_contract_task(task)
        else "drafter-bottleneck-review"
    )
    cmd = [args.research_helper_bin, action, "--recent-rows", str(args.review_recent_rows)]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor drafter bottleneck "
            f"task={task.get('id', 'unknown')} action={action} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=45,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR DRAFTER BOTTLENECK TIMEOUT\n")
            return 124, "supervisor drafter bottleneck timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor {action} exit {result.returncode}")
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    complete_supervisor_task(task, status="keep", summary=parsed, commit=current_commit())
    return 0, ""


def run_supervisor_drafter_sweep_plan(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    blocks = str(task.get("blocks") or os.environ.get("OPENCLAW_DRAFTER_SWEEP_BLOCKS", "1,2,3,4"))
    samples = str(task.get("samples") or os.environ.get("OPENCLAW_DRAFTER_SWEEP_SAMPLES", "3"))
    retries = str(task.get("retries") or os.environ.get("OPENCLAW_DRAFTER_SWEEP_RETRIES", "2"))
    action = "drafter-sweep-plan" if task.get("supervisor_action") == "drafter-sweep-plan" else "drafter-sweep-run"
    cmd = [args.research_helper_bin, action, "--blocks", blocks, "--samples", samples]
    if action == "drafter-sweep-run":
        cmd.extend(["--retries", retries])
    block_count = len([part for part in blocks.split(",") if part.strip()])
    timeout_seconds = 30
    if action == "drafter-sweep-run":
        ready, start_issue = ensure_model_for_supervisor_task(
            log_file,
            args.model_start_timeout_seconds,
            task_id=str(task.get("id", "unknown")),
            reason="supervisor drafter sweep",
        )
        if not ready:
            return 124 if "timeout" in start_issue else 75, start_issue
        timeout_seconds = max(90, block_count * int(samples) * (int(retries) + 1) * 12)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor drafter sweep {action} "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write(f"SUPERVISOR DRAFTER SWEEP {action.upper()} TIMEOUT after {timeout_seconds}s\n")
            return 124, "supervisor drafter sweep timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor drafter sweep exit {result.returncode}")
        append_result(
            WORKSPACE,
            run_id=f"supervisor-drafter-sweep-{cycle}",
            status="blocked",
            target=str(task.get("target", "drafter-sweep-plan")),
            hypothesis=str(task.get("hypothesis", "run bounded drafter sweep")),
            commit=current_commit(),
            notes=reason,
        )
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    append_result(
        WORKSPACE,
        run_id=f"supervisor-drafter-sweep-{cycle}",
        status="keep",
        target=str(task.get("target", "drafter-sweep-plan")),
        hypothesis=str(task.get("hypothesis", "run bounded drafter sweep")),
        commit=current_commit(),
        notes=(
            f"action={action} blocks={blocks} samples={samples} "
            f"retries={retries if action == 'drafter-sweep-run' else ''} "
            f"decision={parsed.get('decision', '')} path={parsed.get('path', '')}"
        ),
    )
    complete_supervisor_task(task, status="keep", summary=parsed, commit=current_commit())
    return 0, ""


def run_supervisor_drafter_fit_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    target_path = str(task.get("target_path") or os.environ.get("OPENCLAW_JANQ_TARGET_PATH", DEFAULT_JANQ_TARGET_PATH))
    drafter_path = str(task.get("drafter_path") or os.environ.get("OPENCLAW_DFLASH_DRAFT_PATH", DEFAULT_DFLASH_DRAFT_PATH))
    output = str(task.get("output") or os.environ.get("OPENCLAW_DRAFTER_FIT_PLAN", DEFAULT_DRAFTER_FIT_PLAN))
    cmd = [
        args.drafter_fit_bin,
        "plan",
        "--target-path",
        target_path,
        "--drafter-path",
        drafter_path,
        "--output",
        output,
    ]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor drafter fit plan "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR DRAFTER FIT TIMEOUT\n")
            return 124, "supervisor drafter fit timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("decision") or parsed.get("reason") or f"supervisor drafter fit exit {result.returncode}")
        append_result(
            WORKSPACE,
            run_id=f"supervisor-drafter-fit-{cycle}",
            status="blocked",
            target=str(task.get("target", "drafter-fit-plan")),
            hypothesis=str(task.get("hypothesis", "create JANQ drafter fit plan")),
            commit=current_commit(),
            notes=reason,
        )
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    notes = (
        f"decision={parsed.get('decision', '')} "
        f"output={output} "
        f"min_speedup={((parsed.get('promotion_gate') or {}).get('minimum_speedup_vs_current', ''))} "
        f"min_accept={((parsed.get('promotion_gate') or {}).get('minimum_mean_accept', ''))}"
    )
    append_result(
        WORKSPACE,
        run_id=f"supervisor-drafter-fit-{cycle}",
        status="keep",
        target=str(task.get("target", "drafter-fit-plan")),
        hypothesis=str(task.get("hypothesis", "create JANQ drafter fit plan")),
        commit=current_commit(),
        notes=notes,
    )
    complete_supervisor_task(task, status="keep", summary=parsed, commit=current_commit())
    return 0, ""


def run_supervisor_drafter_trace_gate_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    plan = str(task.get("target") or DEFAULT_DRAFTER_FIT_PLAN)
    cmd = [args.research_helper_bin, "drafter-trace-gate", "--plan", plan]
    trace_data = str(task.get("trace_data", ""))
    if trace_data:
        cmd.extend(["--trace-data", trace_data])
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor drafter trace gate "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR DRAFTER TRACE GATE TIMEOUT\n")
            return 124, "supervisor drafter trace gate timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor drafter trace gate exit {result.returncode}")
        append_result(
            WORKSPACE,
            run_id=f"supervisor-drafter-trace-gate-{cycle}",
            status="blocked",
            target=str(task.get("target", "drafter-trace-gate")),
            hypothesis=str(task.get("hypothesis", "validate JANQ target-generated trace data")),
            commit=current_commit(),
            notes=reason,
        )
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    status = "keep" if parsed.get("status") == "keep" else "blocked"
    reason = str(parsed.get("reason") or "trace gate completed")
    complete_supervisor_task(task, status=status, summary=parsed, commit=current_commit())
    return 0, "" if status == "keep" else reason


def run_supervisor_drafter_trace_prerequisite_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    cmd = [args.research_helper_bin, "drafter-trace-prerequisite"]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor drafter trace prerequisite "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    status = "keep" if result.returncode == 0 and parsed.get("status") == "keep" else "blocked"
    reason = str(parsed.get("reason") or f"supervisor drafter trace prerequisite exit {result.returncode}")
    complete_supervisor_task(task, status=status, summary=parsed or {"reason": reason}, commit=current_commit())
    return 0, "" if status == "keep" else reason


def run_supervisor_drafter_trace_collect_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    cmd = [args.research_helper_bin, "drafter-trace-collect"]
    ready, start_issue = ensure_model_for_supervisor_task(
        log_file,
        args.model_start_timeout_seconds,
        task_id=str(task.get("id", "unknown")),
        reason="supervisor drafter trace collect",
    )
    if not ready:
        return 124 if "timeout" in start_issue else 75, start_issue
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor drafter trace collect "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR DRAFTER TRACE COLLECT TIMEOUT\n")
            return 124, "supervisor drafter trace collect timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    status = "keep" if result.returncode == 0 and parsed.get("status") == "keep" else "blocked"
    reason = str(parsed.get("reason") or f"supervisor drafter trace collect exit {result.returncode}")
    complete_supervisor_task(task, status=status, summary=parsed or {"reason": reason}, commit=current_commit())
    return 0, "" if status == "keep" else reason


def run_supervisor_drafter_calibration_canary_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    cmd = [args.research_helper_bin, "drafter-calibration-canary"]
    mode = str(task.get("calibration_mode", "")).strip()
    if mode:
        cmd.extend(["--calibration-mode", mode])
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor drafter calibration canary "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR DRAFTER CALIBRATION CANARY TIMEOUT\n")
            return 124, "supervisor drafter calibration canary timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    status = "keep" if result.returncode == 0 and parsed.get("status") == "keep" else "blocked"
    reason = str(parsed.get("decision") or parsed.get("reason") or f"supervisor drafter calibration canary exit {result.returncode}")
    if status != "keep":
        append_result(
            WORKSPACE,
            run_id=f"supervisor-drafter-calibration-canary-{cycle}",
            status="blocked",
            target=str(task.get("target", "drafter-calibration-canary")),
            hypothesis=str(task.get("hypothesis", "validate bounded JANQ drafter calibration canary")),
            commit=current_commit(),
            notes=clean_tsv(reason),
        )
    summary = parsed or {"reason": reason}
    complete_supervisor_task(task, status=status, summary=summary, commit=current_commit())
    if status == "keep" and str(parsed.get("decision", "")) == "ready-for-bounded-calibration":
        suppress_ready_calibration_canaries("ready-for-bounded-calibration", summary)
    return 0, "" if status == "keep" else reason


def run_supervisor_drafter_calibration_memory_stage_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    command = task.get("bounded_command")
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
        command = [args.research_helper_bin, "drafter-calibration-memory-stage", "--stage", str(task.get("stage", ""))]
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
        reason = "drafter calibration memory stage missing bounded_command"
        complete_supervisor_task(task, status="blocked", summary={"reason": reason}, commit=current_commit())
        return 2, reason
    if model_ready() and str(task.get("stage", "")) != "metadata":
        stop_openclaw_model_for_memory_recovery(
            args,
            reason=f"supervisor drafter calibration memory stage needs exclusive RAM: task={task.get('id', 'unknown')}",
        )
    ready, memory_issue = wait_for_memory(args)
    if not ready:
        complete_supervisor_task(
            task,
            status="blocked",
            summary={"reason": memory_issue, "deferred": True},
            commit=current_commit(),
        )
        append_result(
            WORKSPACE,
            run_id=f"supervisor-drafter-calibration-memory-stage-{cycle}",
            status="blocked",
            target=str(task.get("target", "drafter-calibration-memory-stage")),
            hypothesis=str(task.get("hypothesis", "run staged JANQ drafter calibration memory gate")),
            commit=current_commit(),
            notes=clean_tsv(memory_issue),
        )
        return 75, memory_issue
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor drafter calibration memory stage "
            f"task={task.get('id', 'unknown')} stage={task.get('stage', '')} =====\n"
        )
        file.write("$ " + " ".join(command) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                command,
                env=calibration_subprocess_env(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=2100,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR DRAFTER CALIBRATION MEMORY STAGE TIMEOUT\n")
            complete_supervisor_task(
                task,
                status="blocked",
                summary={"reason": "supervisor drafter calibration memory stage timeout"},
                commit=current_commit(),
            )
            return 124, "supervisor drafter calibration memory stage timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    status = "keep" if result.returncode == 0 and parsed.get("status") == "keep" else "blocked"
    memory_gate_issue = calibration_memory_gate_issue(result.stdout)
    runtime_issue = missing_speculative_runtime_issue(result.stdout)
    gradient_issue = calibration_quantized_gradient_issue(result.stdout)
    reason = (
        gradient_issue
        or runtime_issue
        or memory_gate_issue
        or str(parsed.get("decision") or parsed.get("reason") or f"supervisor drafter calibration memory stage exit {result.returncode}")
    )
    if gradient_issue:
        status = "blocked"
    append_result(
        WORKSPACE,
        run_id=f"supervisor-drafter-calibration-memory-stage-{cycle}",
        status=status,
        target=str(task.get("target", "drafter-calibration-memory-stage")),
        hypothesis=str(task.get("hypothesis", "run staged JANQ drafter calibration memory gate")),
        commit=current_commit(),
        notes=clean_tsv(
            f"stage={task.get('stage', '')} reason={reason} blocker={gradient_issue or ''} output_tail={result.stdout[-500:]}"
        ),
    )
    complete_supervisor_task(
        task,
        status=status,
        summary={
            "reason": reason,
            "returncode": result.returncode,
            "runtime_issue": runtime_issue,
            "memory_gate_issue": memory_gate_issue,
            "gradient_issue": gradient_issue,
            "parsed": parsed,
            "output_tail": result.stdout[-1200:],
            "command": command,
        },
        commit=current_commit(),
    )
    return 0, "" if status == "keep" else reason


def run_supervisor_drafter_calibration_run_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    command = task.get("bounded_command")
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
        reason = "drafter calibration run missing bounded_command"
        complete_supervisor_task(task, status="blocked", summary={"reason": reason}, commit=current_commit())
        return 2, reason
    if model_ready():
        stop_openclaw_model_for_memory_recovery(
            args,
            reason=f"supervisor drafter calibration run requires exclusive JANQ target load: task={task.get('id', 'unknown')}",
        )
    ready, memory_issue = wait_for_memory(args)
    if not ready:
        complete_supervisor_task(
            task,
            status="blocked",
            summary={"reason": memory_issue, "deferred": True},
            commit=current_commit(),
        )
        append_result(
            WORKSPACE,
            run_id=f"supervisor-drafter-calibration-run-{cycle}",
            status="blocked",
            target=str(task.get("target", "drafter-calibration-run")),
            hypothesis=str(task.get("hypothesis", "run bounded JANQ drafter calibration")),
            commit=current_commit(),
            notes=clean_tsv(memory_issue),
        )
        return 75, memory_issue
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor drafter calibration run "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(command) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                command,
                env=calibration_subprocess_env(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR DRAFTER CALIBRATION RUN TIMEOUT\n")
            complete_supervisor_task(
                task,
                status="blocked",
                summary={"reason": "supervisor drafter calibration run timeout"},
                commit=current_commit(),
            )
            return 124, "supervisor drafter calibration run timeout"
        file.write(result.stdout)
        file.flush()
    status = "keep" if result.returncode == 0 else "blocked"
    runtime_issue = missing_speculative_runtime_issue(result.stdout)
    memory_gate_issue = calibration_memory_gate_issue(result.stdout)
    gradient_issue = calibration_quantized_gradient_issue(result.stdout)
    reason = gradient_issue or runtime_issue or memory_gate_issue or f"supervisor drafter calibration run exit {result.returncode}"
    if runtime_issue:
        mark_lane_exhausted(
            WORKSPACE,
            lane="drafter-calibration-runtime",
            reason=runtime_issue,
            evidence={
                "task_id": task.get("id", "unknown"),
                "command": command,
                "output_tail": result.stdout[-1200:],
            },
        )
    if memory_gate_issue:
        mark_lane_exhausted(
            WORKSPACE,
            lane="drafter-calibration-memory",
            reason=memory_gate_issue,
            evidence={
                "task_id": task.get("id", "unknown"),
                "command": command,
                "output_tail": result.stdout[-1200:],
                "next": "suppress bounded calibration runs until the calibrator can avoid loading a second full JANQ target",
            },
        )
    if gradient_issue:
        mark_lane_exhausted(
            WORKSPACE,
            lane="drafter-calibration-gradient",
            reason=gradient_issue,
            evidence={
                "task_id": task.get("id", "unknown"),
                "command": command,
                "output_tail": result.stdout[-1200:],
                "next": "suppress calibration until a trainable adapter path avoids gradients through quantized weights",
            },
        )
    calibration_mode = str(task.get("calibration_mode", ""))
    append_result(
        WORKSPACE,
        run_id=f"supervisor-drafter-calibration-run-{cycle}",
        status=status,
        target=str(task.get("target", "drafter-calibration-run")),
        hypothesis=str(task.get("hypothesis", "run bounded JANQ drafter calibration")),
        commit=current_commit(),
        notes=clean_tsv(
            f"{reason} calibration_mode={calibration_mode} trace_distillation={bool(task.get('trace_distillation'))} "
            f"blocker={gradient_issue or ''} output_tail={result.stdout[-500:]}"
        ),
    )
    complete_supervisor_task(
        task,
        status=status,
        summary={
            "reason": reason,
            "returncode": result.returncode,
            "runtime_issue": runtime_issue,
            "memory_gate_issue": memory_gate_issue,
            "gradient_issue": gradient_issue,
            "calibration_mode": calibration_mode,
            "trace_distillation": bool(task.get("trace_distillation")),
            "output_tail": result.stdout[-1200:],
            "command": command,
        },
        commit=current_commit(),
    )
    return 0, "" if status == "keep" else reason


def run_supervisor_dflash_compatibility_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    cmd = [args.research_helper_bin, "dflash-compatibility-gate"]
    draft_path = str(task.get("draft_path") or task.get("drafter_path") or "")
    if draft_path:
        cmd.extend(["--draft-path", draft_path])
    plan = str(task.get("plan") or task.get("fit_plan") or "")
    if plan:
        cmd.extend(["--plan", plan])
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor dflash compatibility "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR DFLASH COMPATIBILITY TIMEOUT\n")
            return 124, "supervisor dflash compatibility timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor dflash compatibility exit {result.returncode}")
        append_result(
            WORKSPACE,
            run_id=f"supervisor-dflash-compatibility-{cycle}",
            status="blocked",
            target=str(task.get("target", "frontier-dflash")),
            hypothesis=str(task.get("hypothesis", "validate DFlash JANQ compatibility")),
            commit=current_commit(),
            notes=reason,
        )
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    status = "keep" if parsed.get("status") == "keep" else "blocked"
    blockers = len(parsed.get("blockers") or [])
    notes = f"decision={parsed.get('decision', '')} blockers={blockers}"
    complete_supervisor_task(task, status=status, summary=parsed, commit=current_commit())
    return 0, "" if status == "keep" else notes


def run_supervisor_focused_test_task(
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    action = str(task.get("next_action", ""))
    allowed = {
        "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py",
        "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-drafter-fit.py",
        "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-mtp-drafter-calibrate-guards.py",
        "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research-autopilot.py",
    }
    if action not in allowed:
        reason = f"focused test action is not allowlisted: {action}"
        append_result(
            WORKSPACE,
            run_id=f"supervisor-focused-test-{cycle}",
            status="blocked",
            target=str(task.get("target", "focused-test")),
            hypothesis=str(task.get("hypothesis", "run focused test")),
            commit=current_commit(),
            notes=reason,
        )
        complete_supervisor_task(task, status="blocked", summary={"reason": reason}, commit=current_commit())
        return 2, reason
    cmd = action.split()
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor focused test "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(repo_path()),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR FOCUSED TEST TIMEOUT\n")
            return 124, "supervisor focused test timeout"
        file.write(result.stdout)
        file.flush()
    status = "keep" if result.returncode == 0 else "blocked"
    notes = "focused test passed" if result.returncode == 0 else f"focused test exit {result.returncode}"
    append_result(
        WORKSPACE,
        run_id=f"supervisor-focused-test-{cycle}",
        status=status,
        target=str(task.get("target", "focused-test")),
        hypothesis=str(task.get("hypothesis", "run focused test")),
        commit=current_commit(),
        notes=notes,
    )
    complete_supervisor_task(
        task,
        status=status,
        summary={"returncode": result.returncode, "action": action, "notes": notes},
        commit=current_commit(),
    )
    return result.returncode, "" if result.returncode == 0 else notes


def run_supervisor_gepa_policy_canary_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    cmd = [
        args.research_helper_bin,
        "gepa-policy-canary",
        "--task-id",
        str(task.get("id", "")),
    ]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor GEPA policy canary "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.gepa_canary_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR GEPA POLICY CANARY TIMEOUT\n")
            return 124, "supervisor GEPA policy canary timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor GEPA policy canary exit {result.returncode}")
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    return 0, ""


def run_supervisor_runtime_overhead_map_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    cmd = [args.research_helper_bin, "runtime-overhead-map", "--recent-rows", str(args.review_recent_rows)]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor runtime overhead map "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=45,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR RUNTIME OVERHEAD MAP TIMEOUT\n")
            return 124, "supervisor runtime overhead map timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor runtime overhead map exit {result.returncode}")
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    complete_supervisor_task(task, status="keep", summary=parsed, commit=current_commit())
    return 0, ""


def run_supervisor_calibration_memory_report_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    cmd = [args.research_helper_bin, "calibration-memory-report"]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor calibration memory report "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=45,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR CALIBRATION MEMORY REPORT TIMEOUT\n")
            return 124, "supervisor calibration memory report timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor calibration memory report exit {result.returncode}")
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    complete_supervisor_task(task, status="keep", summary=parsed, commit=current_commit())
    return 0, ""


def run_supervisor_source_scout_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    topic = str(task.get("topic") or "frontier-decode-speed")
    cmd = [args.research_helper_bin, "source-scout", "--topic", topic]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor source scout "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=90,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR SOURCE SCOUT TIMEOUT\n")
            complete_supervisor_task(
                task,
                status="blocked",
                summary={"reason": "supervisor source scout timeout", "topic": topic},
                commit=current_commit(),
            )
            return 124, "supervisor source scout timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("reason") or f"supervisor source scout exit {result.returncode}")
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    complete_supervisor_task(task, status="keep", summary=parsed, commit=current_commit())
    return 0, ""


def run_supervisor_frontier_deliberation_task(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
) -> tuple[int, str]:
    cmd = [args.research_helper_bin, "frontier-deliberation", "--allow-empty"]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor frontier deliberation "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=90,
                check=False,
            )
        except subprocess.TimeoutExpired:
            file.write("SUPERVISOR FRONTIER DELIBERATION TIMEOUT\n")
            complete_supervisor_task(
                task,
                status="blocked",
                summary={"reason": "supervisor frontier deliberation timeout"},
                commit=current_commit(),
            )
            return 124, "supervisor frontier deliberation timeout"
        file.write(result.stdout)
        file.flush()
    parsed = parse_json_object(result.stdout) or {}
    if result.returncode != 0 or parsed.get("ok") is False:
        reason = str(parsed.get("next") or parsed.get("reason") or f"supervisor frontier deliberation exit {result.returncode}")
        complete_supervisor_task(task, status="blocked", summary={"reason": reason, "result": parsed}, commit=current_commit())
        return result.returncode or 2, reason
    complete_supervisor_task(task, status="keep", summary=parsed, commit=current_commit())
    return 0, ""


def should_run_deterministic_fallback(issue: str, quality: dict[str, object]) -> bool:
    text = f"{issue} {quality.get('reason', '')}"
    if is_gateway_issue(text) and "recovered" not in text.lower():
        return True
    return any(pattern.lower() in text.lower() for pattern in MALFORMED_OR_TOOL_ISSUES)


def run_supervisor_implementation_guard(
    cycle: int,
    session: str,
    task: dict[str, object],
    log_file: Path,
    reason: str,
) -> tuple[bool, str]:
    clean_reason = clean_tsv(reason or "implementation task did not produce safe durable evidence")
    summary = {
        "reason": clean_reason,
        "target": str(task.get("target", "")),
        "task_type": "implementation",
        "next": "move to the next ready task; implementation can be retried after the harness/model issue is fixed",
    }
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} supervisor implementation guard "
            f"task={task.get('id', 'unknown')} =====\n"
        )
        file.write(json.dumps(summary, indent=2) + "\n")
    append_result(
        WORKSPACE,
        run_id=f"supervisor-implementation-guard-{cycle}",
        status="blocked",
        target=str(task.get("target", "implementation")),
        hypothesis=str(task.get("hypothesis", "implementation task must produce a safe source patch plus evidence")),
        commit=current_commit(),
        notes=clean_reason,
    )
    complete_supervisor_task(task, status="blocked", summary=summary, commit=current_commit())
    return True, ""


def run_deterministic_fallback(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    selected_task: dict[str, object] | None,
    log_file: Path,
    reason: str = "",
) -> tuple[bool, str]:
    if selected_task and selected_task.get("task_type") == "implementation":
        return run_supervisor_implementation_guard(
            cycle,
            session,
            selected_task,
            log_file,
            reason or "implementation model/tool failure",
        )
    fallback_task = {
        "id": f"fallback-decode-sample-cycle-{cycle}",
        "benchmark_mode": "decode-sample",
        "target": "decode-sample",
        "hypothesis": "deterministic fallback benchmark after malformed or stalled model/tool turn",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
    }
    code, issue = run_supervisor_benchmark_task(args, cycle, session, fallback_task, log_file, fallback=True)
    return code == 0, issue


def run_supervisor_synthesis(args: argparse.Namespace, cycle: int, session: str, log_file: Path) -> tuple[bool, str]:
    """Create ranked ideas without spending a model turn.

    Empty queues are a supervisor state, not a reasoning problem. Running this
    deterministically prevents the agent from filling results.tsv with another
    easy benchmark just to prove it is still alive.
    """
    cmd = [args.research_helper_bin, "synthesize", "--kind", "frontier"]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n===== cycle {cycle} session {session} supervisor synthesis =====\n")
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=file,
                stderr=subprocess.STDOUT,
                timeout=args.synthesis_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "supervisor synthesis timeout"
    if result.returncode != 0:
        if result.returncode == 2:
            return False, "supervisor synthesis terminal no-work"
        return False, f"supervisor synthesis exit {result.returncode}"
    return True, ""


def run_supervisor_quality_review(args: argparse.Namespace, cycle: int, session: str, log_file: Path) -> tuple[bool, str]:
    tolerated_nonzero = {
        "implementation-handoff-audit",
        "frontier-eval",
        "gepa-escalation",
    }
    first_issue = ""
    commands = [
        [args.research_helper_bin, "environment-snapshot", "--label", f"review-cycle-{cycle}", "--allow-fail"],
        [args.research_helper_bin, "implementation-handoff-audit", "--min-score", "90"],
        [
            args.research_helper_bin,
            "quality-review",
            "--recent-rows",
            str(args.review_recent_rows),
            "--min-sweeps",
            str(args.review_min_sweeps),
            "--min-samples-per-block",
            str(args.review_min_samples_per_block),
            "--target-tps",
            str(args.review_target_tps),
        ],
    ]
    commands.append(
        [
            args.research_helper_bin,
            "frontier-review",
            "--recent-rows",
            str(args.review_recent_rows),
            "--min-samples",
            str(args.review_min_samples_per_block),
        ]
    )
    commands.append([args.research_helper_bin, "hypothesis-rank", "--limit", str(args.hypothesis_rank_limit)])
    commands.append([args.research_helper_bin, "causal-review", "--recent-rows", str(args.review_recent_rows)])
    commands.append(
        [
            args.research_helper_bin,
            "plateau-pivot",
            "--recent-rows",
            str(args.review_recent_rows),
            "--min-sweeps",
            str(args.review_min_sweeps),
            "--target-tps",
            str(args.review_target_tps),
        ]
    )
    commands.append([args.research_helper_bin, "evaluator-integrity"])
    commands.append([args.research_helper_bin, "implementation-handoff-audit", "--min-score", "90"])
    commands.append([args.research_helper_bin, "frontier-eval", "--recent-rows", str(args.review_recent_rows), "--allow-fail"])
    commands.append([args.research_helper_bin, "frontier-autonomy-score", "--recent-rows", str(args.review_recent_rows), "--allow-fail"])
    commands.append([args.research_helper_bin, "review-council", "--recent-rows", str(args.review_recent_rows), "--seed-next", "--allow-fail"])
    commands.append([args.research_helper_bin, "alive-eval", "--recent-rows", str(args.review_recent_rows), "--allow-fail"])
    commands.append([args.research_helper_bin, "implementation-handoff-audit", "--min-score", "90"])
    commands.append([args.research_helper_bin, "gepa-policy-promote", "--min-candidates", "3"])
    commands.append(
        [
            args.research_helper_bin,
            "gepa-escalation",
            "--recent-rows",
            str(args.review_recent_rows),
            "--min-blocked",
            str(args.gepa_min_blocked),
            "--min-rework",
            str(args.gepa_min_rework),
            "--min-trajectory",
            str(args.gepa_min_trajectory),
            "--min-low-quality",
            str(args.gepa_min_low_quality),
        ]
    )
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n===== cycle {cycle} session {session} supervisor quality review =====\n")
        for cmd in commands:
            file.write("$ " + " ".join(cmd) + "\n")
            file.flush()
            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    stdout=file,
                    stderr=subprocess.STDOUT,
                    timeout=args.quality_review_timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False, "supervisor quality/frontier review timeout"
            if result.returncode != 0:
                command_name = cmd[1] if len(cmd) > 1 else ""
                issue = f"{command_name} exit {result.returncode}"
                if command_name not in tolerated_nonzero:
                    return False, f"supervisor quality/frontier review {issue}"
                if not first_issue:
                    first_issue = issue
    return True, first_issue


def append_autonomous_repair_result(
    cycle: int,
    session: str,
    *,
    status: str,
    reason: str,
    attempts: int,
    deterministic_ready: int,
) -> None:
    append_result(
        WORKSPACE,
        run_id=f"autonomous-repair-{cycle}-{int(time.time())}",
        status=status,
        target="autoresearch-autonomous-repair",
        hypothesis="quality or stability drops should be repaired by the supervisor before human review is needed",
        commit=current_commit(),
        notes=(
            f"session={session} attempts={attempts} deterministic_ready={deterministic_ready} "
            f"reason={clean_tsv(reason)}"
        ),
    )
    append_jsonl(
        FINDINGS,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "autonomous-repair-owner",
            "finding": "supervisor owned a quality/stability repair loop instead of waiting for a human scan",
            "status": status,
            "reason": reason,
            "attempts": attempts,
            "deterministic_ready": deterministic_ready,
            "next": "continue the ready deterministic task queue" if status == "keep" else "run the next bounded repair attempt",
        },
    )


def run_autonomous_repair_loop(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    log_file: Path,
    *,
    reason: str,
) -> tuple[bool, str]:
    """Own local quality/stability recovery inside the supervisor.

    This is the replacement for the old human-in-the-loop pattern of
    "ask Codex to scan, explain, and patch."  The loop is deliberately bounded:
    review current evidence, curate/evolve policy lessons, synthesize exactly
    one deterministic repair/refocus path, re-review, then either continue with
    a clean ready task queue or record a bounded blocker.
    """

    if not getattr(args, "autonomous_repair", True):
        return False, "autonomous repair disabled"
    attempts = max(1, int(getattr(args, "autonomous_repair_attempts", 2)))
    last_issue = reason
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} autonomous repair owner "
            f"reason={clean_tsv(reason)} =====\n"
        )
    for attempt in range(1, attempts + 1):
        review_ok, review_issue = run_supervisor_quality_review(
            args,
            cycle,
            f"{session}-repair-{attempt}-pre",
            log_file,
        )
        self_ok, self_issue = run_supervisor_self_improvement(
            args,
            cycle,
            f"{session}-repair-{attempt}",
            log_file,
            reason=f"autonomous-repair:{reason}",
        )
        synth_ok, synth_issue = run_supervisor_synthesis(
            args,
            cycle,
            f"{session}-repair-{attempt}",
            log_file,
        )
        post_ok, post_issue = run_supervisor_quality_review(
            args,
            cycle,
            f"{session}-repair-{attempt}-post",
            log_file,
        )
        certification = frontier_certification_status(args)
        deterministic = deterministic_ready_tasks()
        last_issue = (
            post_issue
            or synth_issue
            or self_issue
            or review_issue
            or "; ".join(str(item) for item in certification.get("issues", []))
            or reason
        )
        if post_ok and self_ok and certification.get("ok") and deterministic:
            append_autonomous_repair_result(
                cycle,
                session,
                status="keep",
                reason=f"repaired after attempt {attempt}",
                attempts=attempt,
                deterministic_ready=len(deterministic),
            )
            return True, ""
        if synth_issue == "supervisor synthesis terminal no-work" and certification.get("ok") and deterministic:
            append_autonomous_repair_result(
                cycle,
                session,
                status="keep",
                reason=f"terminal synthesis but certified deterministic work exists after attempt {attempt}",
                attempts=attempt,
                deterministic_ready=len(deterministic),
            )
            return True, ""
        if review_ok and post_ok and synth_ok and deterministic:
            append_autonomous_repair_result(
                cycle,
                session,
                status="keep",
                reason=f"deterministic repair/refocus task queued after attempt {attempt}",
                attempts=attempt,
                deterministic_ready=len(deterministic),
            )
            return True, ""
    deterministic = deterministic_ready_tasks()
    append_autonomous_repair_result(
        cycle,
        session,
        status="blocked",
        reason=last_issue,
        attempts=attempts,
        deterministic_ready=len(deterministic),
    )
    return False, last_issue


def run_periodic_autonomy_watchdog(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    log_file: Path,
    *,
    reason: str,
) -> tuple[bool, str]:
    """Hourly supervisor check that owns repair instead of waiting for chat.

    The watchdog is deliberately evidence-first: run certification, detect
    low-signal loops, queue deterministic repair work, then re-certify.  Source
    patches still flow through patch-execute/Crabbox gates; this function only
    decides whether the loop should keep researching, repair, or route to the
    next safe action.
    """
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n===== cycle {cycle} session {session} autonomy watchdog reason={clean_tsv(reason)} =====\n")
    review_ok, review_issue = run_supervisor_quality_review(args, cycle, f"{session}-watchdog-review", log_file)
    low_signal = recent_low_signal_mtp_loop_status(args)
    if low_signal.get("loop"):
        repair = repair_low_signal_mtp_loop(cycle, session, low_signal)
        self_ok, self_issue = run_supervisor_self_improvement(
            args,
            cycle,
            f"{session}-watchdog-low-signal",
            log_file,
            reason=f"low-signal-loop:{low_signal.get('reason')}",
        )
        post_ok, post_issue = run_supervisor_quality_review(
            args,
            cycle,
            f"{session}-watchdog-post-low-signal",
            log_file,
        )
        certification = frontier_certification_status(args)
        deterministic = deterministic_ready_tasks()
        if self_ok and post_ok and (certification.get("ok") or deterministic):
            append_autonomous_repair_result(
                cycle,
                session,
                status="keep",
                reason=f"low-signal loop repaired: {repair}",
                attempts=1,
                deterministic_ready=len(deterministic),
            )
            return True, ""
        repair_reason = (
            f"low-signal repair incomplete: self_improvement={self_ok}:{self_issue or 'ok'} "
            f"post_review={post_ok}:{post_issue or 'ok'} "
            f"certification={certification.get('ok')} deterministic_ready={len(deterministic)}"
        )
        return run_autonomous_repair_loop(
            args,
            cycle,
            session,
            log_file,
            reason=repair_reason,
        )
    remeasure_loop = recent_low_signal_decode_remeasure_status(args)
    if remeasure_loop.get("loop"):
        repair = repair_low_signal_decode_remeasure_loop(cycle, session, remeasure_loop)
        self_ok, self_issue = run_supervisor_self_improvement(
            args,
            cycle,
            f"{session}-watchdog-remeasure-loop",
            log_file,
            reason=f"low-signal-decode-remeasure-loop:{remeasure_loop.get('reason')}",
        )
        post_ok, post_issue = run_supervisor_quality_review(
            args,
            cycle,
            f"{session}-watchdog-post-remeasure-loop",
            log_file,
        )
        certification = frontier_certification_status(args)
        deterministic = deterministic_ready_tasks()
        if self_ok and post_ok and (certification.get("ok") or deterministic):
            append_autonomous_repair_result(
                cycle,
                session,
                status="keep",
                reason=f"low-signal decode remeasure loop repaired: {repair}",
                attempts=1,
                deterministic_ready=len(deterministic),
            )
            return True, ""
        repair_reason = (
            f"low-signal decode remeasure repair incomplete: self_improvement={self_ok}:{self_issue or 'ok'} "
            f"post_review={post_ok}:{post_issue or 'ok'} "
            f"certification={certification.get('ok')} deterministic_ready={len(deterministic)}"
        )
        return run_autonomous_repair_loop(
            args,
            cycle,
            session,
            log_file,
            reason=repair_reason,
        )
    certification = frontier_certification_status(args)
    if review_ok and certification.get("ok"):
        append_result(
            WORKSPACE,
            run_id=f"autonomy-watchdog-{cycle}-{int(time.time())}",
            status="keep",
            target="autoresearch-autonomy-watchdog",
            hypothesis="periodic supervisor checks should certify health and route repair without human intervention",
            commit=current_commit(),
            notes=(
                f"session={session} reason={clean_tsv(reason)} frontier={certification.get('frontier_score')} "
                f"quality={certification.get('quality_score')} deterministic={len(certification.get('deterministic_ready_tasks', []))}"
            ),
        )
        return True, ""
    repair_ok, repair_issue = run_autonomous_repair_loop(
        args,
        cycle,
        session,
        log_file,
        reason=review_issue or "; ".join(str(item) for item in certification.get("issues", [])) or reason,
    )
    if repair_ok:
        return True, ""
    return False, repair_issue or review_issue or "autonomy watchdog repair failed"


def latest_json_artifact(pattern: str) -> dict[str, object]:
    paths = list(BENCHMARKS.glob(pattern))
    if not paths:
        return {}
    path = max(paths, key=lambda item: item.stat().st_mtime_ns)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(path), "ok": False, "error": "unreadable artifact"}
    return loaded if isinstance(loaded, dict) else {"path": str(path), "ok": False, "error": "non-object artifact"}


def active_exhausted_lanes() -> set[str]:
    return {
        str(row.get("lane", ""))
        for row in read_jsonl(WORKSPACE / "exhausted-approaches.jsonl")
        if row.get("lane") and not row.get("reopened")
    }


def frontier_certification_status(args: argparse.Namespace) -> dict[str, object]:
    frontier = latest_json_artifact("frontier-system-eval-*.json")
    autonomy = latest_json_artifact("frontier-autonomy-score-*.json")
    alive = latest_json_artifact("self-improvement-alive-eval-*.json")
    handoff = latest_json_artifact("implementation-handoff-audit-*.json")
    quality = latest_json_artifact("quality-review-*.json")
    replay = replay_checks(WORKSPACE)
    deterministic = deterministic_ready_tasks()
    empty_bridges = recent_empty_implementation_bridges(limit=120)
    bridge_ready = [task for task in deterministic if is_supervisor_implementation_bridge_task(task)]
    bridge_only_ready = bridge_only_deterministic_ready(deterministic)
    exhausted = active_exhausted_lanes()
    ready = ready_tasks()
    exhausted_ready = [
        str(task.get("id", ""))
        for task in ready
        if str(task.get("lane", "")) in exhausted and str(task.get("lane", "")) != "exhaustion-report"
    ]
    model_bound_causal = [
        str(task.get("id", ""))
        for task in ready
        if (
            str(task.get("lane", "")) == "causal-repair"
            or str(task.get("id", "")).startswith("causal-review-")
        )
        and not task_runs_without_model(task)
    ]
    frontier_score = float(frontier.get("overall") or 0.0)
    autonomy_score = float(autonomy.get("total_score") or 0.0) if autonomy else 0.0
    alive_score = float(alive.get("total_score") or 0.0) if alive else 0.0
    handoff_score = int(handoff.get("score") or 0)
    quality_scorecard = quality.get("scorecard") if isinstance(quality.get("scorecard"), dict) else {}
    quality_values = [
        value
        for value in (quality.get("quality_score"), quality_scorecard.get("overall"))
        if isinstance(value, int | float)
    ]
    quality_score = float(max(quality_values)) if quality_values else 0.0
    contract = frontier.get("task_contract") if isinstance(frontier.get("task_contract"), dict) else {}
    issues: list[str] = []
    if not replay.get("ok"):
        issues.append("replay guards failed")
    if frontier_score < float(args.frontier_certification_min_score):
        issues.append(f"frontier score {frontier_score}<min {args.frontier_certification_min_score}")
    if autonomy and autonomy_score < 99.0:
        issues.append(f"frontier autonomy score {autonomy_score}<min 99.0")
    if alive and alive_score < 95.0:
        issues.append(f"self-improvement alive score {alive_score}<min 95.0")
    if handoff_score < int(args.frontier_certification_min_handoff):
        issues.append(f"handoff score {handoff_score}<min {args.frontier_certification_min_handoff}")
    if quality_score < float(args.frontier_certification_min_quality):
        issues.append(f"quality score {quality_score}<min {args.frontier_certification_min_quality}")
    if not deterministic:
        issues.append("no deterministic ready task")
    if bridge_ready and empty_bridges:
        issues.append(
            "implementation bridge tasks remain ready after empty bridge rows: "
            + ",".join(str(row.get("run_id", "")) for row in empty_bridges[-3:])
        )
    if exhausted_ready:
        issues.append("ready tasks remain in exhausted lanes: " + ",".join(exhausted_ready[:6]))
    if model_bound_causal:
        issues.append("model-bound causal tasks remain ready: " + ",".join(model_bound_causal[:6]))
    if contract and not contract.get("ok"):
        issues.append("task contract is not clean")
    return {
        "ok": not issues,
        "issues": issues,
        "frontier_score": frontier_score,
        "frontier_autonomy_score": autonomy_score if autonomy else None,
        "self_improvement_alive_score": alive_score if alive else None,
        "handoff_score": handoff_score,
        "quality_score": quality_score,
        "deterministic_ready_tasks": [str(task.get("id", "")) for task in deterministic[:8]],
        "implementation_bridge_ready_tasks": [str(task.get("id", "")) for task in bridge_ready[:8]],
        "bridge_only_ready": bridge_only_ready,
        "recent_empty_bridges": len(empty_bridges),
        "exhausted_lanes": sorted(exhausted),
        "frontier": frontier,
        "frontier_autonomy": autonomy,
        "self_improvement_alive": alive,
        "handoff": handoff,
        "quality": quality,
    }


def latest_artifact_score(pattern: str, *keys: str) -> float:
    artifact = latest_json_artifact(pattern)
    for key in keys:
        value: object = artifact
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def recent_trigger_anomalies(args: argparse.Namespace) -> dict[str, object]:
    rows = all_result_rows(WORKSPACE)[-max(10, int(getattr(args, "autonomy_trigger_recent_rows", 40))) :]
    hard_terms = tuple(term.lower() for term in ACTIVE_MEMORY_OR_CRASH_ROW_TERMS + ACTIVE_BAD_BEHAVIOR_ROW_TERMS)
    blocked = [row for row in rows if row.get("status") == "blocked"]

    def has_active_hard_signal(row: dict[str, str]) -> bool:
        text = (row.get("notes", "") + " " + row.get("hypothesis", "")).lower()
        for term in hard_terms:
            if term not in text:
                continue
            if any(negated in text for negated in NEGATED_HARD_SIGNAL_PHRASES):
                continue
            return True
        return False

    hard = [
        row
        for row in rows
        if has_active_hard_signal(row)
    ]
    terminal = [
        row
        for row in rows
        if row.get("target") in {"synthesis-terminal", "autoresearch-quality-pause"}
        or "terminal_no_work=True" in row.get("notes", "")
    ]
    return {
        "blocked_rows": len(blocked),
        "hard_rows": len(hard),
        "terminal_rows": len(terminal),
        "latest_hard_run": hard[-1].get("run_id", "") if hard else "",
        "latest_blocked_run": blocked[-1].get("run_id", "") if blocked else "",
    }


def autonomy_trigger_status(
    args: argparse.Namespace,
    *,
    cycle: int,
    stalled_cycles: int,
    blocked_cycles: int,
    progress_cycles: int,
    last_issue: str,
) -> dict[str, object]:
    """Cheap every-cycle controller score.

    This is the fast tier of the autonomy system. It reads counters and latest
    artifacts only; it does not run model turns, quality review, or tests. When
    the score drops or a hard trigger fires, the caller escalates to the medium
    watchdog/review tier.
    """
    low_signal = recent_low_signal_mtp_loop_status(args)
    remeasure_loop = recent_low_signal_decode_remeasure_status(args)
    ready = ready_work_summary()
    deterministic = deterministic_ready_tasks()
    anomalies = recent_trigger_anomalies(args)
    quality_score = max(
        latest_artifact_score("quality-review-*.json", "quality_score"),
        latest_artifact_score("quality-review-*.json", "scorecard.overall"),
    )
    autonomy_score = latest_artifact_score("frontier-autonomy-score-*.json", "total_score")
    alive_score = latest_artifact_score("self-improvement-alive-eval-*.json", "total_score")
    council_score = latest_artifact_score("review-council-*.json", "scorecard.overall")
    frontier_score = latest_artifact_score("frontier-system-eval-*.json", "overall") * 10.0
    last_issue_lower = last_issue.lower()
    hard_memory_or_crash = any(term in last_issue_lower for term in MEMORY_OR_CRASH_TERMS)
    hard_bad_behavior = int(anomalies["hard_rows"] or 0) > 0
    has_low_signal = bool(low_signal.get("loop") or remeasure_loop.get("loop"))
    no_ready_work = int(ready["ready_tasks"] or 0) == 0
    no_deterministic_route = not deterministic

    components = {
        "stability": 25 if not hard_memory_or_crash and not hard_bad_behavior and stalled_cycles == 0 else 0,
        "quality": 20 if quality_score >= 99.0 else 12 if quality_score >= 90.0 else 0,
        "autonomy": 20 if autonomy_score >= 99.0 and alive_score >= 95.0 and council_score >= 95.0 else 10,
        "progress": 20 if not has_low_signal and not (no_ready_work and progress_cycles > 0) else 0,
        "routing": 15 if not no_deterministic_route or not no_ready_work else 5,
    }
    score = int(sum(components.values()))
    triggers: list[str] = []
    if hard_memory_or_crash:
        triggers.append("hard-memory-or-crash-signal")
    if hard_bad_behavior:
        triggers.append("hard-bad-behavior-row")
    if low_signal.get("loop"):
        triggers.append("low-signal-mtp-loop")
    if remeasure_loop.get("loop"):
        triggers.append("low-signal-decode-remeasure-loop")
    if quality_score and quality_score < 99.0:
        triggers.append(f"quality-below-99:{quality_score}")
    if autonomy_score and autonomy_score < 99.0:
        triggers.append(f"autonomy-below-99:{autonomy_score}")
    if alive_score and alive_score < 95.0:
        triggers.append(f"alive-below-95:{alive_score}")
    if council_score and council_score < 95.0:
        triggers.append(f"council-below-95:{council_score}")
    if no_ready_work and progress_cycles > 0:
        triggers.append("no-ready-work-after-progress")
    if no_deterministic_route and int(ready["ready_tasks"] or 0) > 0:
        triggers.append("ready-work-has-no-deterministic-route")

    min_score = int(getattr(args, "autonomy_trigger_min_score", 95))
    hard_score = int(getattr(args, "autonomy_trigger_hard_score", 80))
    if any(trigger.startswith("hard-") for trigger in triggers):
        action = "repair"
    elif score < hard_score:
        action = "repair"
    elif triggers or score < min_score:
        action = "review"
    else:
        action = "continue"
    return {
        "ok": action == "continue",
        "kind": "autonomy-trigger-status",
        "cycle": cycle,
        "score": score,
        "action": action,
        "components": components,
        "triggers": triggers,
        "ready": ready,
        "deterministic_ready": [str(task.get("id", "")) for task in deterministic[:8]],
        "low_signal": low_signal,
        "decode_remeasure": remeasure_loop,
        "anomalies": anomalies,
        "scores": {
            "quality": quality_score,
            "frontier": frontier_score,
            "autonomy": autonomy_score,
            "alive": alive_score,
            "council": council_score,
        },
        "primary_goal": "increase normal OpenClaw TUI decode_tps while preserving stability",
        "counters": {
            "stalled_cycles": stalled_cycles,
            "blocked_cycles": blocked_cycles,
            "progress_cycles": progress_cycles,
        },
        "last_issue": last_issue,
    }


def append_autonomy_trigger_result(cycle: int, session: str, status: dict[str, object]) -> None:
    append_result(
        WORKSPACE,
        run_id=f"autonomy-trigger-{cycle}-{int(time.time())}",
        status="keep" if status.get("action") in {"review", "repair"} else "discard",
        target="autoresearch-autonomy-trigger",
        hypothesis="tiered trigger scoring should call review/repair only when autonomous behavior falls below standard",
        commit=current_commit(),
        notes=(
            f"session={session} score={status.get('score')} action={status.get('action')} "
            f"triggers={','.join(str(item) for item in status.get('triggers', [])) or 'none'} "
            f"ready={status.get('ready')}"
        ),
    )


def run_autonomy_trigger_controller(
    args: argparse.Namespace,
    cycle: int,
    session: str,
    log_file: Path,
    *,
    stalled_cycles: int,
    blocked_cycles: int,
    progress_cycles: int,
    last_issue: str,
    last_trigger_at: float,
) -> tuple[bool, str, bool]:
    """Run the tiered autonomy controller.

    Returns (ok, issue, handled). handled=True means the caller should skip the
    normal cycle body because review/repair already acted.
    """
    if not getattr(args, "autonomy_trigger_controller", True):
        return True, "", False
    if int(getattr(args, "autonomy_trigger_interval_cycles", 1)) > 1:
        interval = int(getattr(args, "autonomy_trigger_interval_cycles", 1))
        if cycle % interval != 0:
            return True, "", False
    status = autonomy_trigger_status(
        args,
        cycle=cycle,
        stalled_cycles=stalled_cycles,
        blocked_cycles=blocked_cycles,
        progress_cycles=progress_cycles,
        last_issue=last_issue,
    )
    if status["action"] == "continue":
        return True, "", False
    triggers = [str(item) for item in status.get("triggers", [])]
    immediate = any(
        trigger.startswith("hard-memory-or-crash")
        or trigger in {"low-signal-mtp-loop", "low-signal-decode-remeasure-loop"}
        for trigger in triggers
    )
    review_interval = float(getattr(args, "autonomy_trigger_review_interval_seconds", 1800.0))
    review_due = review_interval <= 0 or time.monotonic() - last_trigger_at >= review_interval
    if not immediate and not review_due:
        return True, "", False
    append_autonomy_trigger_result(cycle, session, status)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(
            f"\n===== cycle {cycle} session {session} autonomy trigger "
            f"score={status['score']} action={status['action']} "
            f"triggers={','.join(str(item) for item in status['triggers']) or 'none'} =====\n"
        )
    ok, issue = run_periodic_autonomy_watchdog(
        args,
        cycle,
        session,
        log_file,
        reason=f"trigger-controller:{status['action']}:{','.join(str(item) for item in status['triggers'])}",
    )
    return ok, issue, True


def run_frontier_startup_certification(args: argparse.Namespace, log_file: Path) -> tuple[bool, str]:
    model_bound_impl_blocked = block_model_bound_implementation_tasks(args)
    stale_lane_blocked = block_stale_hard_blocked_lane_tasks()
    exhausted_lane_blocked = block_ready_exhausted_lane_tasks()
    stale_causal_blocked = block_stale_model_bound_causal_tasks()
    empty_bridge_blocked = block_ready_empty_bridge_tasks()
    run_supervisor_quality_review(args, 0, "startup-certification", log_file)
    status = frontier_certification_status(args)
    if status["ok"]:
        append_result(
            WORKSPACE,
            run_id=f"frontier-certification-{int(time.time())}",
            status="keep",
            target="autoresearch-frontier-certification",
            hypothesis="overnight autoresearch must certify deterministic quality before autonomous cycles",
            commit=current_commit(),
            notes=(
                f"frontier={status['frontier_score']} handoff={status['handoff_score']} "
                f"quality={status['quality_score']} deterministic={len(status['deterministic_ready_tasks'])} "
                f"stale_lane_blocked={stale_lane_blocked} exhausted_lane_blocked={exhausted_lane_blocked} "
                f"stale_causal_blocked={stale_causal_blocked} empty_bridge_blocked={empty_bridge_blocked} "
                f"model_bound_impl_blocked={model_bound_impl_blocked}"
            ),
        )
        return True, ""
    repair_issue = "; ".join(str(issue) for issue in status["issues"])
    run_supervisor_synthesis(args, 0, "startup-certification-repair", log_file)
    block_model_bound_implementation_tasks(args)
    block_stale_hard_blocked_lane_tasks()
    block_ready_exhausted_lane_tasks()
    block_stale_model_bound_causal_tasks()
    block_ready_empty_bridge_tasks()
    run_supervisor_quality_review(args, 0, "startup-certification-recheck", log_file)
    repaired = frontier_certification_status(args)
    if repaired["ok"]:
        append_result(
            WORKSPACE,
            run_id=f"frontier-certification-{int(time.time())}",
            status="keep",
            target="autoresearch-frontier-certification",
            hypothesis="overnight autoresearch repaired startup quality before autonomous cycles",
            commit=current_commit(),
            notes=(
                f"repaired=True frontier={repaired['frontier_score']} handoff={repaired['handoff_score']} "
                f"quality={repaired['quality_score']} deterministic={len(repaired['deterministic_ready_tasks'])}"
            ),
        )
        return True, ""
    reason = "; ".join(str(issue) for issue in repaired["issues"]) or repair_issue or "unknown certification failure"
    append_result(
        WORKSPACE,
        run_id=f"frontier-certification-{int(time.time())}",
        status="blocked",
        target="autoresearch-frontier-certification",
        hypothesis="overnight autoresearch should not start below frontier-quality threshold",
        commit=current_commit(),
        notes=f"startup certification failed after deterministic repair: {clean_tsv(reason)}",
    )
    append_jsonl(
        FINDINGS,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "frontier-certification",
            "finding": "autopilot refused to start a low-quality autonomous loop",
            "evidence": repaired,
            "next": "inspect certification issues and add a deterministic task or source patch before restarting",
        },
    )
    return False, reason


def run_supervisor_compaction(args: argparse.Namespace, log_file: Path) -> None:
    cmd = [args.research_helper_bin, "compact", "--recent-rows", str(args.compact_recent_rows)]
    with log_file.open("a", encoding="utf-8") as file:
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        subprocess.run(cmd, text=True, stdout=file, stderr=subprocess.STDOUT, timeout=30, check=False)


def run_supervisor_environment_snapshot(args: argparse.Namespace, log_file: Path, label: str) -> None:
    cmd = [args.research_helper_bin, "environment-snapshot", "--label", label, "--allow-fail"]
    with log_file.open("a", encoding="utf-8") as file:
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        subprocess.run(cmd, text=True, stdout=file, stderr=subprocess.STDOUT, timeout=30, check=False)


def run_supervisor_self_improvement(args: argparse.Namespace, cycle: int, session: str, log_file: Path, reason: str) -> tuple[bool, str]:
    if not args.self_improvement:
        return True, "disabled"
    commands = [
        [
            args.research_helper_bin,
            "self-improve",
            "--action",
            "curate",
            "--recent-rows",
            str(args.self_improvement_recent_rows),
        ]
    ]
    if getattr(args, "self_evolution", False):
        commands.append(
            [
                args.research_helper_bin,
                "self-improve",
                "--action",
                "evolve",
                "--recent-rows",
                str(args.self_improvement_recent_rows),
                "--max-variants-per-skill",
                str(args.self_evolution_max_variants_per_skill),
                "--min-score",
                str(args.self_evolution_min_score),
            ]
        )
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n===== cycle {cycle} session {session} supervisor self-improvement: {reason} =====\n")
        for cmd in commands:
            file.write("$ " + " ".join(cmd) + "\n")
            file.flush()
            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    stdout=file,
                    stderr=subprocess.STDOUT,
                    timeout=args.self_improvement_timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False, f"self-improvement {cmd[3]} timeout"
            if result.returncode != 0:
                return False, f"self-improvement {cmd[3]} exit {result.returncode}"
    return True, ""


def run_supervisor_reflection(args: argparse.Namespace, cycle: int, session: str, log_file: Path, reason: str) -> tuple[bool, str]:
    replay_result = replay_checks(WORKSPACE)
    ready_impl = ready_implementation_tasks()
    append_jsonl(
        FINDINGS,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "supervisor-reflection",
            "finding": "reflection checkpoint ran replay guards before synthesis",
            "reason": reason,
            "replay": replay_result,
            "ready_implementation_tasks": [str(task.get("id", "")) for task in ready_impl],
        },
    )
    if not replay_result["ok"]:
        return False, f"replay guards failed: {replay_result}"
    if ready_impl:
        return True, "synthesis deferred because ready implementation tasks exist"
    ok, issue = run_supervisor_synthesis(args, cycle, session, log_file)
    if not ok and issue == "supervisor synthesis terminal no-work":
        return True, "reflection skipped after clean terminal no-work"
    return ok, issue


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenClaw speed autoresearch in autonomous cycles.")
    parser.add_argument("--openclaw-bin", default=os.environ.get("OPENCLAW_REAL_BIN", "/opt/homebrew/bin/openclaw"))
    parser.add_argument(
        "--research-helper-bin",
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_HELPER", "/Users/kristian/.openclaw/bin/openclaw-speed-research"),
    )
    parser.add_argument(
        "--drafter-fit-bin",
        default=os.environ.get("OPENCLAW_DRAFTER_FIT_HELPER", "/Users/kristian/.openclaw/bin/openclaw-drafter-fit"),
    )
    parser.add_argument("--session", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_SESSION", "speed-research-auto"))
    parser.add_argument("--cycles", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_CYCLES", "48")))
    parser.add_argument("--max-hours", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_HOURS", "8")))
    parser.add_argument("--turn-timeout-seconds", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_TURN_TIMEOUT", "1200")))
    parser.add_argument("--turn-timeout-grace-seconds", type=int, default=30)
    parser.add_argument("--synthesis-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--patch-execute-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_PATCH_EXECUTE_TIMEOUT", "300")))
    parser.add_argument("--architectural-approval-file", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_ARCHITECTURAL_APPROVAL_FILE", ""))
    parser.add_argument("--supervisor-benchmark-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_SUPERVISOR_BENCHMARK_TIMEOUT", "180")))
    parser.add_argument("--model-start-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_MODEL_START_TIMEOUT", "420")))
    parser.add_argument("--gateway-port", type=int, default=int(os.environ.get("OPENCLAW_GATEWAY_PORT", "18789")))
    parser.add_argument("--gateway-health-url", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_GATEWAY_HEALTH_URL", ""))
    parser.add_argument("--gateway-start-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_GATEWAY_START_TIMEOUT", "30")))
    parser.add_argument("--reflection-interval", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_REFLECTION_INTERVAL", "4")))
    parser.add_argument("--quality-review-interval", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_INTERVAL", "6")))
    parser.add_argument("--quality-review-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_TIMEOUT", "45")))
    parser.add_argument("--review-recent-rows", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_RECENT_ROWS", "120")))
    parser.add_argument(
        "--self-improvement",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_SELF_IMPROVEMENT", "1") != "0",
        help="curate autoresearch lessons at deterministic supervisor checkpoints",
    )
    parser.add_argument(
        "--self-improvement-interval",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_SELF_IMPROVEMENT_INTERVAL", "0")),
        help="progress-cycle interval for self-improvement curation; 0 follows the quality-review interval",
    )
    parser.add_argument(
        "--self-improvement-recent-rows",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_SELF_IMPROVEMENT_ROWS", "160")),
    )
    parser.add_argument(
        "--self-improvement-timeout-seconds",
        type=float,
        default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_SELF_IMPROVEMENT_TIMEOUT", "30")),
    )
    parser.add_argument(
        "--self-evolution",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_SELF_EVOLUTION", "1") != "0",
        help="run canary-only skill evolution after self-improvement curation",
    )
    parser.add_argument(
        "--self-evolution-max-variants-per-skill",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_SELF_EVOLUTION_VARIANTS", "2")),
    )
    parser.add_argument(
        "--self-evolution-min-score",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_SELF_EVOLUTION_MIN_SCORE", "90")),
    )
    parser.add_argument("--review-min-sweeps", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_MIN_SWEEPS", "3")))
    parser.add_argument("--review-min-samples-per-block", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_MIN_SAMPLES_PER_BLOCK", "3")))
    parser.add_argument("--review-target-tps", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_TARGET_TPS", "30")))
    parser.add_argument("--hypothesis-rank-limit", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_HYPOTHESIS_RANK_LIMIT", "12")))
    parser.add_argument("--gepa-min-blocked", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_GEPA_MIN_BLOCKED", "3")))
    parser.add_argument("--gepa-min-rework", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_GEPA_MIN_REWORK", "2")))
    parser.add_argument("--gepa-min-trajectory", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_GEPA_MIN_TRAJECTORY", "2")))
    parser.add_argument("--gepa-min-low-quality", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_GEPA_MIN_LOW_QUALITY", "2")))
    parser.add_argument("--gepa-canary-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_GEPA_CANARY_TIMEOUT", "30")))
    parser.add_argument("--compact-recent-rows", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_COMPACT_ROWS", "24")))
    parser.add_argument("--max-tool-results-per-turn", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_TOOL_RESULTS", "1")))
    parser.add_argument(
        "--max-implementation-tool-results-per-turn",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_IMPLEMENTATION_TOOL_RESULTS", "4")),
        help="allow bounded read/patch/test/record cycles for implementation tasks",
    )
    parser.add_argument("--tool-result-synthesis-grace-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_TOOL_SYNTHESIS_GRACE", "20")))
    parser.add_argument("--session-start-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_SESSION_START_TIMEOUT", "90")))
    parser.add_argument("--session-idle-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_SESSION_IDLE_TIMEOUT", "180")))
    parser.add_argument("--task-min-samples", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_TASK_MIN_SAMPLES", "3")))
    parser.add_argument("--sleep-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_SLEEP", "8")))
    parser.add_argument("--thinking", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_THINKING", "off"))
    parser.add_argument("--min-free-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MIN_FREE_MB", "1024")))
    parser.add_argument("--ready-min-free-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_READY_MIN_FREE_MB", "0")))
    parser.add_argument("--min-pressure-free-pct", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MIN_PRESSURE_FREE_PCT", "3")))
    parser.add_argument("--max-compressor-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_COMPRESSOR_MB", "8192")))
    parser.add_argument("--max-swap-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_SWAP_MB", "8192")))
    parser.add_argument("--active-memory-check-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_MEMORY_CHECK_SECONDS", "2")))
    parser.add_argument("--active-min-free-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_MIN_FREE_MB", "128")))
    parser.add_argument("--active-min-pressure-free-pct", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_MIN_PRESSURE_FREE_PCT", "2")))
    parser.add_argument("--active-max-compressor-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_MAX_COMPRESSOR_MB", "6144")))
    parser.add_argument("--active-max-swap-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_MAX_SWAP_MB", "8192")))
    parser.add_argument("--active-low-free-pressure-compressor-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_LOW_FREE_PRESSURE_COMPRESSOR_MB", "4096")))
    parser.add_argument("--active-low-free-pressure-swap-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_LOW_FREE_PRESSURE_SWAP_MB", "2048")))
    parser.add_argument("--memory-stable-samples", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_MEMORY_STABLE_SAMPLES", "2")))
    parser.add_argument("--memory-stable-interval-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_MEMORY_STABLE_INTERVAL", "5")))
    parser.add_argument("--memory-wait-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-memory-wait-seconds",
        type=float,
        default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_MEMORY_WAIT", "1800")),
        help="record a blocked row and continue recovery after memory remains unsafe for this long",
    )
    parser.add_argument(
        "--memory-stop-model-after-wait",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_STOP_MODEL_AFTER_MEMORY_WAIT", "1") != "0",
        help="stop the OpenClaw-owned local model once memory stays unsafe beyond the wait budget",
    )
    parser.add_argument("--memory-model-stop-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_MODEL_STOP_TIMEOUT", "60")))
    parser.add_argument("--memory-cooldown-after-stop-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_MEMORY_COOLDOWN_AFTER_STOP", "60")))
    parser.add_argument("--crash-cooldown-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_CRASH_COOLDOWN_SECONDS", "180")))
    parser.add_argument(
        "--stop-model-on-memory-crash",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_STOP_MODEL_ON_MEMORY_CRASH", "1") != "0",
        help="stop OpenClaw's model process after memory, Metal, or fatal process failures before continuing",
    )
    parser.add_argument(
        "--stop-model-on-interrupt",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_STOP_MODEL_ON_INTERRUPT", "1") != "0",
        help="stop the OpenClaw-owned model and write a neutral resume checkpoint when Ctrl+C/SIGTERM stops research",
    )
    parser.add_argument(
        "--rotate-session-after-stalls",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_ROTATE_AFTER_STALLS", "3")),
        help="start a fresh recovery session after this many cycles without durable progress",
    )
    parser.add_argument(
        "--auto-extend-cycles",
        "--auto-extend",
        action=argparse.BooleanOptionalAction,
        dest="auto_extend_cycles",
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_EXTEND_CYCLES", "1") != "0",
        help="when the cycle tranche ends, keep going until max-hours if useful work remains or progress is healthy",
    )
    parser.add_argument(
        "--stop-on-external-blocker",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_STOP_ON_EXTERNAL_BLOCKER", "1") != "0",
        help="stop cleanly once evidence says active speed lanes require external change instead of more terminal synthesis",
    )
    parser.add_argument(
        "--external-blocker-min-quality",
        type=float,
        default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_EXTERNAL_BLOCKER_MIN_QUALITY", "90")),
        help="minimum quality score before an external blocker can stop the autonomous loop",
    )
    parser.add_argument(
        "--external-blocker-min-exhausted-core-lanes",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_EXTERNAL_BLOCKER_MIN_EXHAUSTED_LANES", "2")),
        help="core speed lanes that must be exhausted before external-blocker stop can fire",
    )
    parser.add_argument(
        "--external-blocker-min-terminal-cycles",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_EXTERNAL_BLOCKER_MIN_TERMINAL_CYCLES", "3")),
        help="terminal synthesis/report evidence required before stopping while terminal tasks remain queued",
    )
    parser.add_argument(
        "--external-blocker-refill-before-stop",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_EXTERNAL_REFILL_BEFORE_STOP", "1") != "0",
        help="run deterministic quality/synthesis refill before allowing an external-change stop",
    )
    parser.add_argument(
        "--external-blocker-action",
        choices=["refocus", "stop"],
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_EXTERNAL_BLOCKER_ACTION", "refocus"),
        help="what to do when clean evidence says current lanes are externally blocked; default refocus keeps overnight loops alive",
    )
    parser.add_argument(
        "--cycle-extension-size",
        "--extension-size",
        dest="cycle_extension_size",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_CYCLE_EXTENSION_SIZE", "0")),
        help="additional cycles to add per auto-extension; 0 reuses the original --cycles tranche size",
    )
    parser.add_argument(
        "--reuse-session",
        action="store_true",
        help="reuse one OpenClaw session instead of Ralph-style fresh sessions per cycle",
    )
    parser.add_argument(
        "--allow-model-bound-research-turns",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_ALLOW_MODEL_BOUND_TURNS", "0") == "1",
        help="allow non-supervisor autoresearch turns to call the local model; disabled by default for 31B stability",
    )
    parser.add_argument(
        "--allow-implementation-model-turns",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_ALLOW_IMPLEMENTATION_TURNS", "0") == "1",
        help="allow scoped implementation-gate tasks to use the local model; disabled by default so implementation flows through deterministic patch-executor gates",
    )
    parser.add_argument(
        "--certify-startup",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_CERTIFY_STARTUP", "1") != "0",
        help="run deterministic frontier-quality certification before autonomous cycles",
    )
    parser.add_argument(
        "--frontier-certification-min-score",
        type=float,
        default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_CERT_MIN_FRONTIER", "9.0")),
        help="minimum frontier eval score required before starting autonomous cycles",
    )
    parser.add_argument(
        "--frontier-certification-min-handoff",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_CERT_MIN_HANDOFF", "90")),
        help="minimum implementation handoff audit score required before starting autonomous cycles",
    )
    parser.add_argument(
        "--frontier-certification-min-quality",
        type=float,
        default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_CERT_MIN_QUALITY", "90")),
        help="minimum quality-review scorecard required before starting autonomous cycles",
    )
    parser.add_argument(
        "--autonomous-repair",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTONOMOUS_REPAIR", "1") != "0",
        help="own quality/stability repair inside the supervisor instead of pausing for human review",
    )
    parser.add_argument(
        "--autonomous-repair-attempts",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTONOMOUS_REPAIR_ATTEMPTS", "2")),
        help="bounded repair attempts before recording a blocker",
    )
    parser.add_argument(
        "--autonomy-watchdog-interval-seconds",
        type=float,
        default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTONOMY_WATCHDOG_INTERVAL", "3600")),
        help="wall-clock interval for autonomous health/quality/repair ownership checks",
    )
    parser.add_argument(
        "--autonomy-trigger-controller",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTONOMY_TRIGGER_CONTROLLER", "1") != "0",
        help="run cheap every-cycle trigger scoring and escalate to review/repair only when standards drop",
    )
    parser.add_argument(
        "--autonomy-trigger-interval-cycles",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTONOMY_TRIGGER_INTERVAL", "1")),
        help="cycle interval for the cheap trigger controller",
    )
    parser.add_argument(
        "--autonomy-trigger-review-interval-seconds",
        type=float,
        default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTONOMY_TRIGGER_REVIEW_INTERVAL", "1800")),
        help="minimum wall-clock interval between non-critical review/repair escalations from the cheap trigger controller",
    )
    parser.add_argument(
        "--autonomy-trigger-min-score",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTONOMY_TRIGGER_MIN_SCORE", "95")),
        help="trigger medium review below this cheap controller score",
    )
    parser.add_argument(
        "--autonomy-trigger-hard-score",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTONOMY_TRIGGER_HARD_SCORE", "80")),
        help="trigger repair below this cheap controller score",
    )
    parser.add_argument(
        "--autonomy-trigger-recent-rows",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTONOMY_TRIGGER_RECENT_ROWS", "40")),
        help="recent result rows inspected by cheap trigger scoring",
    )
    parser.add_argument(
        "--low-signal-guard",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_LOW_SIGNAL_GUARD", "1") != "0",
        help="detect and repair repeated synthesis/MTP-report loops before they waste overnight cycles",
    )
    parser.add_argument(
        "--low-signal-check-interval-cycles",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_LOW_SIGNAL_CHECK_INTERVAL", "12")),
        help="cycle interval for low-signal loop detection in addition to the wall-clock watchdog",
    )
    parser.add_argument(
        "--low-signal-window-rows",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_LOW_SIGNAL_WINDOW_ROWS", "40")),
    )
    parser.add_argument(
        "--low-signal-min-mtp-reports",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_LOW_SIGNAL_MIN_MTP", "4")),
    )
    parser.add_argument(
        "--low-signal-min-synthesis-rows",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_LOW_SIGNAL_MIN_SYNTHESIS", "4")),
    )
    parser.add_argument(
        "--low-signal-min-decode-remeasures",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_LOW_SIGNAL_MIN_DECODE_REMEASURES", "3")),
        help="clean decode remeasure/synthesis pairs before the supervisor pivots to an escape path",
    )
    args = parser.parse_args()
    INTERRUPT_CONTEXT.update(
        {
            "args": args,
            "session": args.session,
            "current_session": args.session,
            "cycle": 0,
            "selected_task": None,
        }
    )
    install_interrupt_handlers()
    if args.cycles <= 0:
        args.cycles = 1_000_000

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ensure_task_queue()
    autopilot_lock = acquire_autopilot_lock(args.session)
    if autopilot_lock is None:
        log(f"autopilot refused duplicate workspace run lock={AUTOPILOT_LOCK}")
        return 2
    INTERRUPT_CONTEXT["lock"] = autopilot_lock
    log_file = LOG_DIR / f"autopilot-{args.session}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    deadline = time.monotonic() + args.max_hours * 3600
    stalled_cycles = 0
    last_issue = ""
    current_session = args.session
    recovery_epoch = 0
    progress_cycles = 0
    blocked_cycles = 0
    crash_cooldown_until = 0.0
    last_autonomy_watchdog_at = time.monotonic()
    last_autonomy_trigger_at = time.monotonic()
    last_low_signal_repair_cycle = 0
    cycle_limit = args.cycles
    extension_size = args.cycle_extension_size if args.cycle_extension_size > 0 else max(1, args.cycles)
    log(
        f"autopilot start session={args.session} cycles={args.cycles} max_hours={args.max_hours} "
        f"auto_extend_cycles={args.auto_extend_cycles} extension_size={extension_size} log={log_file}"
    )
    run_supervisor_compaction(args, log_file)
    run_supervisor_environment_snapshot(args, log_file, "autopilot-start")
    self_improve_ok, self_improve_issue = run_supervisor_self_improvement(
        args,
        0,
        args.session,
        log_file,
        reason="autopilot-start",
    )
    if not self_improve_ok:
        log(f"startup self_improvement warning: {self_improve_issue}")
    replay_start = replay_checks(WORKSPACE)
    if not replay_start["ok"]:
        append_supervisor_result(0, args.session, "blocked", f"startup replay failed: {replay_start}")
        log(f"startup replay guards failed: {replay_start}")
        return 2
    if args.certify_startup:
        certified, certification_issue = run_frontier_startup_certification(args, log_file)
        if not certified:
            repair_ok, repair_issue = run_autonomous_repair_loop(
                args,
                0,
                args.session,
                log_file,
                reason=f"startup certification failed: {certification_issue}",
            )
            if not repair_ok:
                log(f"startup frontier certification failed after autonomous repair: {repair_issue}")
                return 2
            log("startup frontier certification recovered by autonomous repair owner")
    cycle = 0
    while True:
        INTERRUPT_CONTEXT["cycle"] = cycle
        INTERRUPT_CONTEXT["current_session"] = current_session
        INTERRUPT_CONTEXT["selected_task"] = None
        if cycle >= cycle_limit:
            extend, reason, summary = should_extend_cycle_budget(
                args,
                deadline=deadline,
                progress_cycles=progress_cycles,
                blocked_cycles=blocked_cycles,
                stalled_cycles=stalled_cycles,
            )
            if not extend:
                log(
                    f"cycle budget reached at cycle={cycle}; stopping: {reason} "
                    f"ready_tasks={summary['ready_tasks']} progress={progress_cycles} blocked={blocked_cycles}"
                )
                break
            old_limit = cycle_limit
            cycle_limit += extension_size
            log(
                f"cycle budget reached at cycle={cycle}; extending {old_limit}->{cycle_limit}: {reason} "
                f"progress={progress_cycles} blocked={blocked_cycles}"
            )
        cycle += 1
        if not args.reuse_session:
            current_session = f"{args.session}-cycle-{cycle:03d}"
        INTERRUPT_CONTEXT["cycle"] = cycle
        INTERRUPT_CONTEXT["current_session"] = current_session
        if time.monotonic() >= deadline:
            log("autopilot max-hours reached")
            break
        now = time.monotonic()
        if now < crash_cooldown_until:
            remaining = min(crash_cooldown_until - now, max(0.0, deadline - now))
            if remaining > 0:
                log(f"memory/crash cooldown active before cycle={cycle}: sleeping {remaining:.0f}s")
                time.sleep(min(remaining, max(1.0, args.sleep_seconds)))
                if time.monotonic() < crash_cooldown_until:
                    cycle -= 1
                    continue
        stale_blocked = block_stale_rejected_implementation_tasks()
        if stale_blocked:
            log(f"supervisor blocked stale rejected implementation tasks count={stale_blocked}")
        model_bound_impl_blocked = block_model_bound_implementation_tasks(args)
        if model_bound_impl_blocked:
            log(f"supervisor quarantined model-bound implementation tasks count={model_bound_impl_blocked}")
        stale_lane_blocked = block_stale_hard_blocked_lane_tasks()
        if stale_lane_blocked:
            log(f"supervisor quarantined stale hard-blocked lane tasks count={stale_lane_blocked}")
        stale_causal_blocked = block_stale_model_bound_causal_tasks()
        if stale_causal_blocked:
            log(f"supervisor quarantined stale model-bound causal tasks count={stale_causal_blocked}")
        external_stop, external_status = maybe_stop_for_external_change(args, cycle, current_session, log_file)
        if external_stop:
            log(
                f"cycle={cycle} external_change_required_stop reason={external_status.get('reason')} "
                f"exhausted={','.join(str(item) for item in external_status.get('exhausted_lanes', [])) or 'none'} "
                f"ready={','.join(str(item) for item in external_status.get('ready_tasks', [])) or 'none'}"
            )
            break
        if (
            args.autonomy_watchdog_interval_seconds > 0
            and time.monotonic() - last_autonomy_watchdog_at >= args.autonomy_watchdog_interval_seconds
        ):
            watchdog_ok, watchdog_issue = run_periodic_autonomy_watchdog(
                args,
                cycle,
                current_session,
                log_file,
                reason="wall-clock interval",
            )
            last_autonomy_watchdog_at = time.monotonic()
            log(f"cycle={cycle} autonomy_watchdog ok={watchdog_ok} issue={watchdog_issue or 'none'}")
            if watchdog_ok:
                time.sleep(args.sleep_seconds)
                continue
        trigger_ok, trigger_issue, trigger_handled = run_autonomy_trigger_controller(
            args,
            cycle,
            current_session,
            log_file,
            stalled_cycles=stalled_cycles,
            blocked_cycles=blocked_cycles,
            progress_cycles=progress_cycles,
            last_issue=last_issue,
            last_trigger_at=last_autonomy_trigger_at,
        )
        if trigger_handled:
            last_autonomy_trigger_at = time.monotonic()
            log(
                f"cycle={cycle} autonomy_trigger_controller ok={trigger_ok} "
                f"issue={trigger_issue or 'none'}"
            )
            if not trigger_ok:
                pause_reason = f"autonomy trigger controller could not repair cleanly: {trigger_issue or 'unknown issue'}"
                append_quality_pause(cycle, current_session, pause_reason)
                log(f"cycle={cycle} quality_pause reason={pause_reason}")
                break
            time.sleep(args.sleep_seconds)
            continue
        if (
            args.low_signal_guard
            and args.low_signal_check_interval_cycles > 0
            and cycle - last_low_signal_repair_cycle >= args.low_signal_check_interval_cycles
        ):
            low_signal = recent_low_signal_mtp_loop_status(args)
            remeasure_loop = recent_low_signal_decode_remeasure_status(args)
            if low_signal.get("loop") or remeasure_loop.get("loop"):
                loop_reason = "low-signal guard"
                if remeasure_loop.get("loop"):
                    loop_reason = "low-signal decode remeasure guard"
                watchdog_ok, watchdog_issue = run_periodic_autonomy_watchdog(
                    args,
                    cycle,
                    current_session,
                    log_file,
                    reason=loop_reason,
                )
                last_low_signal_repair_cycle = cycle
                log(
                    f"cycle={cycle} low_signal_watchdog ok={watchdog_ok} "
                    f"issue={watchdog_issue or 'none'}"
                )
                if not watchdog_ok:
                    pause_reason = f"low-signal watchdog could not repair cleanly: {watchdog_issue or 'unknown issue'}"
                    append_quality_pause(cycle, current_session, pause_reason)
                    log(f"cycle={cycle} quality_pause reason={pause_reason}")
                    break
                time.sleep(args.sleep_seconds)
                continue
        memory_ok, memory_issue = wait_for_memory(args)
        if not memory_ok:
            stalled_cycles += 1
            blocked_cycles += 1
            last_issue = memory_issue
            append_supervisor_result(cycle, current_session, "blocked", memory_issue)
            log(
                f"cycle={cycle} skipped model turn due to memory gate stalled_cycles={stalled_cycles} "
                f"issue={memory_issue}"
            )
            if (
                args.rotate_session_after_stalls > 0
                and stalled_cycles >= args.rotate_session_after_stalls
                and cycle < cycle_limit
            ):
                recovery_epoch += 1
                current_session = f"{args.session}-recovery-{recovery_epoch}"
                last_issue = f"rotated to fresh session after {stalled_cycles} stalled cycles"
                stalled_cycles = 0
                log(f"rotating to fresh recovery session={current_session}")
            time.sleep(args.sleep_seconds)
            continue
        before = durable_snapshot()
        selected_task = select_next_runnable_task(args)
        INTERRUPT_CONTEXT["selected_task"] = selected_task
        if selected_task is None:
            ok, issue = run_supervisor_synthesis(args, cycle, current_session, log_file)
            after = durable_snapshot()
            progress_reasons = durable_progress(before, after)
            deterministic_ready = deterministic_ready_tasks()
            if not ok and issue == "supervisor synthesis terminal no-work":
                review_ok, review_issue = run_supervisor_quality_review(args, cycle, current_session, log_file)
                after_review = durable_snapshot()
                review_progress = durable_progress(after, after_review)
                deterministic_ready = deterministic_ready_tasks()
                external_stop, external_status = maybe_stop_for_external_change(args, cycle, current_session, log_file)
                if external_stop:
                    log(
                        f"cycle={cycle} terminal_no_work_external_stop reason={external_status.get('reason')} "
                        f"artifact={','.join(review_progress) if review_progress else 'none'}"
                    )
                    break
                if deterministic_ready:
                    progress_cycles += 1
                    log(
                        f"cycle={cycle} terminal_no_work_review ok={review_ok} "
                        f"deterministic_ready={len(deterministic_ready)} "
                        f"artifact={','.join(review_progress) if review_progress else 'none'}"
                    )
                    time.sleep(args.sleep_seconds)
                    continue
                pause_reason = (
                    "synthesis reported terminal no-work and review found no deterministic ready task; "
                    f"{review_issue or 'research lanes are exhausted or waiting on new prerequisites'}"
                )
                append_quality_pause(cycle, current_session, pause_reason)
                log(f"cycle={cycle} terminal_no_work_pause reason={pause_reason}")
                break
            if ok and not deterministic_ready:
                review_ok, review_issue = run_supervisor_quality_review(args, cycle, current_session, log_file)
                after_review = durable_snapshot()
                review_progress = durable_progress(after, after_review)
                stale_causal_blocked = block_stale_model_bound_causal_tasks()
                if stale_causal_blocked:
                    review_progress.append("stale causal tasks quarantined")
                deterministic_ready = deterministic_ready_tasks()
                external_stop, external_status = maybe_stop_for_external_change(args, cycle, current_session, log_file)
                if external_stop:
                    log(
                        f"cycle={cycle} synthesis_empty_external_stop reason={external_status.get('reason')} "
                        f"artifact={','.join(review_progress) if review_progress else 'none'}"
                    )
                    break
                if deterministic_ready:
                    progress_cycles += 1
                    log(
                        f"cycle={cycle} synthesis_empty_review ok={review_ok} "
                        f"deterministic_ready={len(deterministic_ready)} "
                        f"artifact={','.join(review_progress) if review_progress else 'none'}"
                    )
                    time.sleep(args.sleep_seconds)
                    continue
                pause_reason = (
                    "synthesis produced no new deterministic ready tasks; "
                    f"review also found no ready deterministic task ({review_issue or 'no issue'}); "
                    "pausing to avoid low-quality repeated measurement"
                )
                append_quality_pause(cycle, current_session, pause_reason)
                log(f"cycle={cycle} quality_pause reason={pause_reason}")
                break
            progressed = ok and bool(progress_reasons)
            stalled_cycles = 0 if progressed else stalled_cycles + 1
            last_issue = "" if progressed else (issue or "supervisor synthesis made no durable progress")
            if progressed:
                progress_cycles += 1
            else:
                blocked_cycles += 1
                append_supervisor_result(cycle, current_session, "blocked", last_issue)
            log(
                f"cycle={cycle} supervisor_synthesis ok={ok} progressed={progressed} "
                f"deterministic_ready={len(deterministic_ready)} "
                f"artifact={','.join(progress_reasons) if progress_reasons else 'none'} "
                f"issue={last_issue or 'none'}"
            )
            time.sleep(args.sleep_seconds)
            continue
        selected_task = claim_task_evidence_window(WORKSPACE, selected_task, int(before["results_lines"]))
        INTERRUPT_CONTEXT["selected_task"] = selected_task
        before = durable_snapshot()
        defer_reason = model_bound_defer_reason(args, selected_task)
        if defer_reason:
            issue = f"{defer_reason}; routing to deterministic synthesis"
            ok, synth_issue = run_supervisor_synthesis(args, cycle, current_session, log_file)
            complete_supervisor_task(
                selected_task,
                status="blocked",
                summary={
                    "reason": defer_reason,
                    "next": "convert this into a deterministic supervisor task before retrying",
                },
                commit=current_commit(),
            )
            deterministic_ready = deterministic_ready_tasks()
            if not deterministic_ready:
                review_ok, review_issue = run_supervisor_quality_review(args, cycle, current_session, log_file)
                stale_causal_blocked = block_stale_model_bound_causal_tasks()
                if stale_causal_blocked:
                    log(f"cycle={cycle} deferred_task_review quarantined stale causal tasks count={stale_causal_blocked}")
                deterministic_ready = deterministic_ready_tasks()
                external_stop, external_status = maybe_stop_for_external_change(args, cycle, current_session, log_file)
                if external_stop:
                    log(f"cycle={cycle} deferred_task_external_stop reason={external_status.get('reason')}")
                    break
                if deterministic_ready:
                    append_supervisor_result(cycle, current_session, "blocked", issue)
                    log(
                        f"cycle={cycle} deferred_task_review ok={review_ok} "
                        f"deterministic_ready={len(deterministic_ready)} issue={review_issue or 'none'}"
                    )
                    time.sleep(args.sleep_seconds)
                    continue
                pause_reason = (
                    f"{defer_reason}; no deterministic ready tasks remained after synthesis, "
                    f"and review found no ready deterministic task ({review_issue or 'no issue'}), "
                    "so autoresearch paused instead of refilling generic cycles"
                )
                append_quality_pause(cycle, current_session, pause_reason)
                log(f"cycle={cycle} quality_pause reason={pause_reason}")
                break
            append_supervisor_result(cycle, current_session, "blocked", issue)
            after = durable_snapshot()
            progress_reasons = durable_progress(before, after)
            progressed = ok and bool(progress_reasons)
            if progressed:
                progress_cycles += 1
                stalled_cycles = 0
                last_issue = ""
            else:
                blocked_cycles += 1
                stalled_cycles += 1
                last_issue = synth_issue or issue
            log(
                f"cycle={cycle} skipped model-bound task={selected_task.get('id', 'unknown')} "
                f"because {defer_reason}; synthesis_ok={ok} deterministic_ready={len(deterministic_ready)} "
                f"progressed={progressed} issue={last_issue or 'none'}"
            )
            time.sleep(args.sleep_seconds)
            continue
        if is_supervisor_drafter_fit_task(selected_task):
            code, issue = run_supervisor_drafter_fit_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_drafter_bottleneck_review_task(selected_task) or is_supervisor_drafter_adapter_contract_task(selected_task):
            code, issue = run_supervisor_drafter_bottleneck_review_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_drafter_trace_gate_task(selected_task):
            code, issue = run_supervisor_drafter_trace_gate_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_drafter_trace_prerequisite_task(selected_task):
            code, issue = run_supervisor_drafter_trace_prerequisite_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_drafter_trace_collect_task(selected_task):
            code, issue = run_supervisor_drafter_trace_collect_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_drafter_calibration_canary_task(selected_task):
            code, issue = run_supervisor_drafter_calibration_canary_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_drafter_calibration_memory_stage_task(selected_task):
            code, issue = run_supervisor_drafter_calibration_memory_stage_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_drafter_calibration_run_task(selected_task):
            code, issue = run_supervisor_drafter_calibration_run_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_dflash_compatibility_task(selected_task):
            code, issue = run_supervisor_dflash_compatibility_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_drafter_sweep_task(selected_task):
            code, issue = run_supervisor_drafter_sweep_plan(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_mtp_report_task(selected_task):
            code, issue = run_supervisor_mtp_report_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_implementation_bridge_task(selected_task):
            code, issue = run_supervisor_implementation_bridge(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_patch_execute_task(selected_task):
            code, issue = run_supervisor_patch_execute_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_focused_test_task(selected_task):
            code, issue = run_supervisor_focused_test_task(cycle, current_session, selected_task, log_file)
        elif is_supervisor_gepa_policy_canary_task(selected_task):
            code, issue = run_supervisor_gepa_policy_canary_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_runtime_overhead_map_task(selected_task):
            code, issue = run_supervisor_runtime_overhead_map_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_calibration_memory_report_task(selected_task):
            code, issue = run_supervisor_calibration_memory_report_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_source_scout_task(selected_task):
            code, issue = run_supervisor_source_scout_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_frontier_deliberation_task(selected_task):
            code, issue = run_supervisor_frontier_deliberation_task(args, cycle, current_session, selected_task, log_file)
        elif requires_profile_variant_runner(selected_task):
            code, issue = run_supervisor_profile_variant_guard(cycle, current_session, selected_task, log_file)
        elif is_supervisor_benchmark_task(selected_task):
            code, issue = run_supervisor_benchmark_task(args, cycle, current_session, selected_task, log_file)
        elif is_supervisor_log_review_task(selected_task):
            code, issue = run_supervisor_log_review_task(cycle, current_session, selected_task, log_file)
        else:
            code, issue = run_turn(args, current_session, cycle, stalled_cycles, last_issue, log_file, selected_task)
        after = durable_snapshot()
        progress_reasons = durable_progress(before, after)
        quality = cycle_quality(WORKSPACE, before, after, progress_reasons, issue)
        progressed = int(quality["score"]) >= 2
        if not progressed and should_run_deterministic_fallback(issue, quality):
            fallback_ok, fallback_issue = run_deterministic_fallback(
                args,
                cycle,
                current_session,
                selected_task,
                log_file,
                reason=issue or str(quality["reason"]),
            )
            fallback_after = durable_snapshot()
            fallback_reasons = durable_progress(after, fallback_after)
            fallback_quality = cycle_quality(WORKSPACE, after, fallback_after, fallback_reasons, fallback_issue)
            if fallback_ok and int(fallback_quality["score"]) >= 2:
                after = fallback_after
                progress_reasons = progress_reasons + [f"fallback:{reason}" for reason in fallback_reasons]
                quality = fallback_quality
                issue = ""
                progressed = True
            elif fallback_issue:
                issue = f"{issue or quality['reason']}; fallback={fallback_issue}"
        advancement = None
        if progressed:
            advancement = complete_task_from_evidence(
                WORKSPACE,
                selected_task,
                min_samples=args.task_min_samples,
                commit=current_commit(),
            )
            if advancement is None:
                advancement = complete_implementation_task(
                    selected_task,
                    progress_reasons,
                    commit=current_commit(),
                )
        if progressed:
            progress_cycles += 1
        stalled_cycles = 0 if progressed else stalled_cycles + 1
        last_issue = "" if progressed else (issue or str(quality["reason"]))
        if not progressed:
            blocked_cycles += 1
        if is_memory_or_crash_issue(issue or last_issue or quality.get("reason", "")):
            if args.stop_model_on_memory_crash and model_ready():
                stop_openclaw_model_for_memory_recovery(
                    args,
                    reason=f"cycle={cycle} issue={issue or last_issue or quality.get('reason', '')}",
                )
            if args.crash_cooldown_seconds > 0:
                crash_cooldown_until = max(crash_cooldown_until, time.monotonic() + args.crash_cooldown_seconds)
                log(
                    f"memory/crash cooldown armed for {args.crash_cooldown_seconds:.0f}s "
                    f"after cycle={cycle} issue={issue or last_issue or quality.get('reason', '')}"
                )
        log(
            f"cycle={cycle} exit={code} progressed={progressed} "
            f"artifact={','.join(progress_reasons) if progress_reasons else 'none'} "
            f"results_lines={before['results_lines']}->{after['results_lines']} "
            f"stalled_cycles={stalled_cycles} issue={last_issue or 'none'} "
            f"quality={quality['status']}:{quality['score']} health={progress_cycles}/{cycle} blocked={blocked_cycles}"
        )
        if advancement:
            if "sample_count" in advancement:
                log(
                    f"cycle={cycle} advanced task={advancement['task_id']} "
                    f"samples={advancement['sample_count']} mode={advancement['benchmark_mode']}"
                )
            else:
                log(
                    f"cycle={cycle} advanced implementation task={advancement['task_id']} "
                    f"artifacts={','.join(str(item) for item in advancement.get('artifacts', []))}"
                )
        if (
            progressed
            and args.reflection_interval > 0
            and progress_cycles > 0
            and progress_cycles % args.reflection_interval == 0
        ):
            reflection_ok, reflection_issue = run_supervisor_reflection(
                args,
                cycle,
                current_session,
                log_file,
                reason=f"progress_cycles={progress_cycles}",
            )
            log(
                f"cycle={cycle} reflection ok={reflection_ok} issue={reflection_issue or 'none'}"
            )
        if (
            progressed
            and args.quality_review_interval > 0
            and progress_cycles > 0
            and progress_cycles % args.quality_review_interval == 0
        ):
            review_ok, review_issue = run_supervisor_quality_review(
                args,
                cycle,
                current_session,
                log_file,
            )
            log(f"cycle={cycle} quality_review ok={review_ok} issue={review_issue or 'none'}")
            if not review_ok or review_issue:
                repair_ok, repair_issue = run_autonomous_repair_loop(
                    args,
                    cycle,
                    current_session,
                    log_file,
                    reason=review_issue or "quality review failed",
                )
                log(
                    f"cycle={cycle} autonomous_repair ok={repair_ok} "
                    f"issue={repair_issue or 'none'}"
                )
            external_stop, external_status = maybe_stop_for_external_change(args, cycle, current_session, log_file)
            if external_stop:
                log(f"cycle={cycle} quality_review_external_stop reason={external_status.get('reason')}")
                break
        self_improvement_interval = args.self_improvement_interval or args.quality_review_interval
        if (
            progressed
            and args.self_improvement
            and self_improvement_interval > 0
            and progress_cycles > 0
            and progress_cycles % self_improvement_interval == 0
        ):
            self_improve_ok, self_improve_issue = run_supervisor_self_improvement(
                args,
                cycle,
                current_session,
                log_file,
                reason=f"progress_cycles={progress_cycles}",
            )
            log(f"cycle={cycle} self_improvement ok={self_improve_ok} issue={self_improve_issue or 'none'}")
        if code not in {0, 124} and not progressed:
            log(f"cycle action returned nonzero exit={code}; continuing after a short pause")
        elif code == 2 and progressed:
            log("cycle action returned terminal evidence exit=2; counted as durable progress")
        if not progressed:
            failed_task_id = str((selected_task or select_next_task(WORKSPACE) or {}).get("id", "unknown"))
            record_rejection(
                WORKSPACE,
                cycle=cycle,
                task_id=failed_task_id,
                reason=str(quality["reason"]),
                evidence=",".join(progress_reasons) if progress_reasons else last_issue,
            )
            record_trajectory_case(
                WORKSPACE,
                cycle=cycle,
                session=current_session,
                task_id=failed_task_id,
                reason=str(quality["reason"]),
                evidence=",".join(progress_reasons) if progress_reasons else last_issue,
            )
            append_supervisor_result(cycle, current_session, "blocked", last_issue)
            log(f"cycle={cycle} recorded supervisor blocked row for issue={last_issue}")
            repair_ok, repair_issue = run_autonomous_repair_loop(
                args,
                cycle,
                current_session,
                log_file,
                reason=last_issue or str(quality["reason"]),
            )
            log(
                f"cycle={cycle} autonomous_repair_after_block ok={repair_ok} "
                f"issue={repair_issue or 'none'}"
            )
            if block_task_after_repeated_guard(selected_task, last_issue):
                log(
                    f"cycle={cycle} blocked implementation task={selected_task.get('id', 'unknown')} "
                    "after repeated implementation stalls or guard hits; moving to next task"
                )
                stalled_cycles = 0
                last_issue = ""
        if (
            args.rotate_session_after_stalls > 0
            and stalled_cycles >= args.rotate_session_after_stalls
            and cycle < cycle_limit
        ):
            recovery_epoch += 1
            current_session = f"{args.session}-recovery-{recovery_epoch}"
            last_issue = f"rotated to fresh session after {stalled_cycles} stalled cycles"
            stalled_cycles = 0
            log(f"rotating to fresh recovery session={current_session}")
        time.sleep(args.sleep_seconds)
    log("autopilot done")
    release_autopilot_lock(autopilot_lock)
    INTERRUPT_CONTEXT["lock"] = None
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as error:
        finalize_autopilot_interrupt(str(error) or "user interrupt")
        log("autopilot interrupted safely; checkpoint written for next run")
        raise SystemExit(130)
