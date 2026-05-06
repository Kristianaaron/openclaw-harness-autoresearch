#!/usr/bin/env python3
"""Autonomous outer loop for OpenClaw speed autoresearch.

The TUI is interactive: once an assistant turn naturally stops, it waits for a
human. This helper keeps overnight speed research moving by launching repeated
bounded `openclaw agent` turns against the same session.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from openclaw_speed_research_core import (
    RESULTS_HEADER,
    append_jsonl,
    append_result,
    claim_task_evidence_window,
    complete_task_from_evidence,
    cycle_quality,
    ensure_research_state,
    paired_profile_plan,
    read_jsonl,
    record_rejection,
    replay_checks,
    select_next_task,
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


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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
    )


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


def is_supervisor_focused_test_task(task: dict[str, object] | None) -> bool:
    if not task:
        return False
    return task.get("supervisor_action") == "focused-test"


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


def memory_snapshot() -> dict[str, int]:
    snapshot = {"free_mb": 0, "compressor_mb": 0, "swap_used_mb": 0}
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


def memory_gate_reason(args: argparse.Namespace, snap: dict[str, int], *, ready: bool) -> str:
    min_free_mb = args.ready_min_free_mb if ready else args.min_free_mb
    if snap["compressor_mb"] >= args.max_compressor_mb:
        return (
            f"compressor={snap['compressor_mb']}MB>={args.max_compressor_mb}MB "
            f"free={snap['free_mb']}MB swap={snap['swap_used_mb']}MB ready={ready}"
        )
    if snap["swap_used_mb"] >= args.max_swap_mb:
        return (
            f"swap={snap['swap_used_mb']}MB>={args.max_swap_mb}MB "
            f"free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB ready={ready}"
        )
    if snap["free_mb"] and snap["free_mb"] < min_free_mb:
        return (
            f"free={snap['free_mb']}MB<{min_free_mb}MB "
            f"compressor={snap['compressor_mb']}MB swap={snap['swap_used_mb']}MB ready={ready}"
        )
    return ""


def wait_for_memory(args: argparse.Namespace) -> tuple[bool, str]:
    started = time.monotonic()
    last_reason = ""
    while True:
        snap = memory_snapshot()
        ready = model_ready()
        reason = memory_gate_reason(args, snap, ready=ready)
        if not reason:
            return True, ""
        last_reason = f"memory gate waiting: {reason}"
        if args.max_memory_wait_seconds > 0 and time.monotonic() - started >= args.max_memory_wait_seconds:
            return False, f"{last_reason}; exceeded {args.max_memory_wait_seconds:.0f}s wait budget"
        log(
            last_reason
        )
        time.sleep(args.memory_wait_seconds)


def active_memory_circuit_reason(args: argparse.Namespace, snap: dict[str, int]) -> str:
    """Abort an in-flight agent turn before macOS/Metal reaches crash territory."""
    compressor_limit = args.active_max_compressor_mb or args.max_compressor_mb
    swap_limit = args.active_max_swap_mb or args.max_swap_mb
    if snap["compressor_mb"] >= compressor_limit:
        return (
            f"compressor={snap['compressor_mb']}MB>={compressor_limit}MB "
            f"free={snap['free_mb']}MB swap={snap['swap_used_mb']}MB"
        )
    if snap["swap_used_mb"] >= swap_limit:
        return (
            f"swap={snap['swap_used_mb']}MB>={swap_limit}MB "
            f"free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB"
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
            f"compressor={snap['compressor_mb']}MB swap={snap['swap_used_mb']}MB"
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
            "metric": "decode_tps_delta_or_guardrail",
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
        finally:
            file.flush()
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
    try:
        start_model_for_supervisor_benchmark(log_file, args.model_start_timeout_seconds)
    except subprocess.TimeoutExpired:
        return 124, "supervisor model start timeout"
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
        return result.returncode, reason
    if parsed and parsed.get("ok") is False:
        return 2, str(parsed.get("reason") or "supervisor benchmark blocked")
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
    after_ready = [
        item
        for item in read_jsonl(TASKS)
        if item.get("status", "ready") in {"ready", "rework"}
    ]
    deterministic = [
        str(item.get("id", ""))
        for item in after_ready
        if item.get("task_type") == "supervisor" and str(item.get("id", "")) != str(task.get("id", ""))
    ]
    seeded = [task_id for task_id in deterministic if task_id not in before_ids]
    summary = {
        "synthesis_ok": ok,
        "issue": issue,
        "seeded_deterministic_tasks": seeded,
        "ready_deterministic_tasks": deterministic[:12],
        "next": "select_next_ready_supervisor_task",
    }
    status = "keep" if ok and deterministic else "blocked"
    append_result(
        WORKSPACE,
        run_id=f"supervisor-implementation-bridge-{cycle}",
        status=status,
        target=str(task.get("target", "implementation-bridge")),
        hypothesis=str(task.get("hypothesis", "bridge synthesis into deterministic implementation tasks")),
        commit=current_commit(),
        notes=(
            f"seeded={len(seeded)} ready_deterministic={len(deterministic)} "
            f"issue={clean_tsv(issue)}"
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
        return False, f"supervisor synthesis exit {result.returncode}"
    return True, ""


def run_supervisor_quality_review(args: argparse.Namespace, cycle: int, session: str, log_file: Path) -> tuple[bool, str]:
    cmd = [
        args.research_helper_bin,
        "quality-review",
        "--recent-rows",
        str(args.review_recent_rows),
        "--min-sweeps",
        str(args.review_min_sweeps),
    ]
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n===== cycle {cycle} session {session} supervisor quality review =====\n")
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
            return False, "supervisor quality review timeout"
    if result.returncode != 0:
        return False, f"supervisor quality review exit {result.returncode}"
    return True, ""


def run_supervisor_compaction(args: argparse.Namespace, log_file: Path) -> None:
    cmd = [args.research_helper_bin, "compact", "--recent-rows", str(args.compact_recent_rows)]
    with log_file.open("a", encoding="utf-8") as file:
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        subprocess.run(cmd, text=True, stdout=file, stderr=subprocess.STDOUT, timeout=30, check=False)


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
    return run_supervisor_synthesis(args, cycle, session, log_file)


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
    parser.add_argument("--supervisor-benchmark-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_SUPERVISOR_BENCHMARK_TIMEOUT", "180")))
    parser.add_argument("--model-start-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_MODEL_START_TIMEOUT", "420")))
    parser.add_argument("--gateway-port", type=int, default=int(os.environ.get("OPENCLAW_GATEWAY_PORT", "18789")))
    parser.add_argument("--gateway-health-url", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_GATEWAY_HEALTH_URL", ""))
    parser.add_argument("--gateway-start-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_GATEWAY_START_TIMEOUT", "30")))
    parser.add_argument("--reflection-interval", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_REFLECTION_INTERVAL", "4")))
    parser.add_argument("--quality-review-interval", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_INTERVAL", "6")))
    parser.add_argument("--quality-review-timeout-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_TIMEOUT", "45")))
    parser.add_argument("--review-recent-rows", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_RECENT_ROWS", "120")))
    parser.add_argument("--review-min-sweeps", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_REVIEW_MIN_SWEEPS", "3")))
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
    parser.add_argument("--max-compressor-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_COMPRESSOR_MB", "8192")))
    parser.add_argument("--max-swap-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_SWAP_MB", "8192")))
    parser.add_argument("--active-memory-check-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_MEMORY_CHECK_SECONDS", "5")))
    parser.add_argument("--active-min-free-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_MIN_FREE_MB", "128")))
    parser.add_argument("--active-max-compressor-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_MAX_COMPRESSOR_MB", "6144")))
    parser.add_argument("--active-max-swap-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_MAX_SWAP_MB", "8192")))
    parser.add_argument("--active-low-free-pressure-compressor-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_LOW_FREE_PRESSURE_COMPRESSOR_MB", "4096")))
    parser.add_argument("--active-low-free-pressure-swap-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_ACTIVE_LOW_FREE_PRESSURE_SWAP_MB", "2048")))
    parser.add_argument("--memory-wait-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-memory-wait-seconds",
        type=float,
        default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_MEMORY_WAIT", "1800")),
        help="record a blocked row and continue recovery after memory remains unsafe for this long",
    )
    parser.add_argument(
        "--rotate-session-after-stalls",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_ROTATE_AFTER_STALLS", "3")),
        help="start a fresh recovery session after this many cycles without durable progress",
    )
    parser.add_argument(
        "--reuse-session",
        action="store_true",
        help="reuse one OpenClaw session instead of Ralph-style fresh sessions per cycle",
    )
    args = parser.parse_args()
    if args.cycles <= 0:
        args.cycles = 1_000_000

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ensure_task_queue()
    log_file = LOG_DIR / f"autopilot-{args.session}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    deadline = time.monotonic() + args.max_hours * 3600
    stalled_cycles = 0
    last_issue = ""
    current_session = args.session
    recovery_epoch = 0
    progress_cycles = 0
    blocked_cycles = 0
    log(f"autopilot start session={args.session} cycles={args.cycles} max_hours={args.max_hours} log={log_file}")
    run_supervisor_compaction(args, log_file)
    replay_start = replay_checks(WORKSPACE)
    if not replay_start["ok"]:
        append_supervisor_result(0, args.session, "blocked", f"startup replay failed: {replay_start}")
        log(f"startup replay guards failed: {replay_start}")
        return 2
    for cycle in range(1, args.cycles + 1):
        if not args.reuse_session:
            current_session = f"{args.session}-cycle-{cycle:03d}"
        if time.monotonic() >= deadline:
            log("autopilot max-hours reached")
            break
        stale_blocked = block_stale_rejected_implementation_tasks()
        if stale_blocked:
            log(f"supervisor blocked stale rejected implementation tasks count={stale_blocked}")
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
                and cycle < args.cycles
            ):
                recovery_epoch += 1
                current_session = f"{args.session}-recovery-{recovery_epoch}"
                last_issue = f"rotated to fresh session after {stalled_cycles} stalled cycles"
                stalled_cycles = 0
                log(f"rotating to fresh recovery session={current_session}")
            time.sleep(args.sleep_seconds)
            continue
        before = durable_snapshot()
        selected_task = select_next_task(WORKSPACE)
        if selected_task is None:
            ok, issue = run_supervisor_synthesis(args, cycle, current_session, log_file)
            seeded = enqueue_recurring_decode_tasks(cycle, issue or "queue exhausted after synthesis")
            after = durable_snapshot()
            progress_reasons = durable_progress(before, after)
            progressed = ok and (bool(progress_reasons) or seeded > 0)
            stalled_cycles = 0 if progressed else stalled_cycles + 1
            last_issue = "" if progressed else (issue or "supervisor synthesis made no durable progress")
            if progressed:
                progress_cycles += 1
            else:
                blocked_cycles += 1
                append_supervisor_result(cycle, current_session, "blocked", last_issue)
            log(
                f"cycle={cycle} supervisor_synthesis ok={ok} progressed={progressed} "
                f"seeded_tasks={seeded} "
                f"artifact={','.join(progress_reasons) if progress_reasons else 'none'} "
                f"issue={last_issue or 'none'}"
            )
            time.sleep(args.sleep_seconds)
            continue
        selected_task = claim_task_evidence_window(WORKSPACE, selected_task, int(before["results_lines"]))
        before = durable_snapshot()
        if is_supervisor_drafter_fit_task(selected_task):
            code, issue = run_supervisor_drafter_fit_task(args, cycle, current_session, selected_task, log_file)
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
        if code not in {0, 124}:
            log(f"cycle action returned nonzero exit={code}; continuing after a short pause")
        if not progressed:
            record_rejection(
                WORKSPACE,
                cycle=cycle,
                task_id=str((selected_task or select_next_task(WORKSPACE) or {}).get("id", "unknown")),
                reason=str(quality["reason"]),
                evidence=",".join(progress_reasons) if progress_reasons else last_issue,
            )
            append_supervisor_result(cycle, current_session, "blocked", last_issue)
            log(f"cycle={cycle} recorded supervisor blocked row for issue={last_issue}")
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
            and cycle < args.cycles
        ):
            recovery_epoch += 1
            current_session = f"{args.session}-recovery-{recovery_epoch}"
            last_issue = f"rotated to fresh session after {stalled_cycles} stalled cycles"
            stalled_cycles = 0
            log(f"rotating to fresh recovery session={current_session}")
        time.sleep(args.sleep_seconds)
    log("autopilot done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
