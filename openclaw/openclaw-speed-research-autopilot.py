#!/usr/bin/env python3
"""Autonomous outer loop for OpenClaw speed autoresearch.

The TUI is interactive: once an assistant turn naturally stops, it waits for a
human. This helper keeps overnight speed research moving by launching repeated
bounded `openclaw agent` turns against the same session.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from openclaw_speed_research_core import (
    RESULTS_HEADER,
    append_jsonl,
    claim_task_evidence_window,
    complete_task_from_evidence,
    cycle_quality,
    ensure_research_state,
    read_jsonl,
    record_rejection,
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
)
EARLY_FAILURE_PATTERNS = (
    ("EMBEDDED FALLBACK", "gateway embedded fallback"),
    ("GatewayClientRequestError", "gateway client request error"),
    ("FailoverError: LLM request failed: network connection error", "model connection error"),
    ("embedded run agent end", "embedded agent failure"),
    ("rawError=Connection error", "model connection error"),
    ("Connection error.", "model connection error"),
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


def synthesis_task() -> dict[str, object]:
    return {
        "id": "synthesize-speed-ideas",
        "target": "ideas.md/STRATEGY.md/tasks.jsonl",
        "hypothesis": "Completed baselines must produce ranked ideas and new measurable OpenClaw speed tasks.",
        "metric": "ranked_ideas_and_seeded_tasks",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research synthesize --kind frontier",
    }


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
        "Prefer realistic decode work over toy prompts: "
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
        "Do not repeat quick-health, TTFT, prompt-size, or tool-roundtrip benchmarks unless they directly support a decode/MTP hypothesis. "
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
                        return 124, failure
                    next_failure_check = now + 2
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


def run_supervisor_compaction(args: argparse.Namespace, log_file: Path) -> None:
    cmd = [args.research_helper_bin, "compact", "--recent-rows", str(args.compact_recent_rows)]
    with log_file.open("a", encoding="utf-8") as file:
        file.write("$ " + " ".join(cmd) + "\n")
        file.flush()
        subprocess.run(cmd, text=True, stdout=file, stderr=subprocess.STDOUT, timeout=30, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenClaw speed autoresearch in autonomous cycles.")
    parser.add_argument("--openclaw-bin", default=os.environ.get("OPENCLAW_REAL_BIN", "/opt/homebrew/bin/openclaw"))
    parser.add_argument(
        "--research-helper-bin",
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_HELPER", "/Users/kristian/.openclaw/bin/openclaw-speed-research"),
    )
    parser.add_argument("--session", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_SESSION", "speed-research-auto"))
    parser.add_argument("--cycles", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_CYCLES", "48")))
    parser.add_argument("--max-hours", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_HOURS", "8")))
    parser.add_argument("--turn-timeout-seconds", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_TURN_TIMEOUT", "1200")))
    parser.add_argument("--turn-timeout-grace-seconds", type=int, default=30)
    parser.add_argument("--synthesis-timeout-seconds", type=float, default=60.0)
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
    synthesis_attempted = False
    log(f"autopilot start session={args.session} cycles={args.cycles} max_hours={args.max_hours} log={log_file}")
    run_supervisor_compaction(args, log_file)
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
            if synthesis_attempted:
                log("autopilot task queue exhausted after synthesis; stopping without benchmark churn")
                break
            ok, issue = run_supervisor_synthesis(args, cycle, current_session, log_file)
            synthesis_attempted = True
            after = durable_snapshot()
            progress_reasons = durable_progress(before, after)
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
                f"artifact={','.join(progress_reasons) if progress_reasons else 'none'} "
                f"issue={last_issue or 'none'}"
            )
            time.sleep(args.sleep_seconds)
            continue
        else:
            synthesis_attempted = False
        selected_task = claim_task_evidence_window(WORKSPACE, selected_task, int(before["results_lines"]))
        before = durable_snapshot()
        code, issue = run_turn(args, current_session, cycle, stalled_cycles, last_issue, log_file, selected_task)
        after = durable_snapshot()
        progress_reasons = durable_progress(before, after)
        quality = cycle_quality(WORKSPACE, before, after, progress_reasons, issue)
        progressed = int(quality["score"]) >= 2
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
        if code not in {0, 124}:
            log(f"agent turn returned nonzero exit={code}; continuing after a short pause")
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
