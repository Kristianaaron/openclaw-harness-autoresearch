#!/usr/bin/env python3
"""Autonomous outer loop for OpenClaw speed autoresearch.

The TUI is interactive: once an assistant turn naturally stops, it waits for a
human. This helper keeps overnight speed research moving by launching repeated
bounded `openclaw agent` turns against the same session.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from openclaw_speed_research_core import (
    RESULTS_HEADER,
    complete_task_from_evidence,
    cycle_quality,
    ensure_research_state,
    record_rejection,
    select_next_task,
    task_summary,
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
    "timed out",
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
        action = str((selected_task or {}).get("next_action", "")).strip()
        if "openclaw-speed-research benchmark --mode" not in action:
            action = "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode streaming-ttft"
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


def continuation_prompt(cycle: int, stalled_cycles: int, last_issue: str = "") -> str:
    pressure = ""
    if stalled_cycles >= 2:
        pressure = (
            "\n\nPrevious cycles did not record enough durable progress. In this turn, choose the smallest "
            "realistic OpenClaw/Rapid-MLX/Gemma4 JANG backend speed experiment and record either a result "
            "row in results.tsv or a frontier idea in ideas.md before ending."
        )
    if last_issue:
        pressure += (
            f"\n\nLast cycle issue: {last_issue}. Recover with the next safe Bootstrap Ladder action. "
            "Your next tool call must be one of: read `/Users/kristian/.openclaw/research/speed/README-openclaw-speed.md`, "
            "read `/Users/kristian/.openclaw/research/speed/results.tsv`, run "
            "`/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --quick`, or run "
            "`git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`."
        )
    task_summary = next_task_summary()
    if task_summary:
        task_summary = f"\n\nCurrent machine-readable research queue:\n{task_summary}"
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
    return (
        "Continue OpenClaw Speed Autoresearch autonomously.\n\n"
        f"Workspace: {WORKSPACE}\n"
        f"Program: {PROGRAM}\n\n"
        "Use program.md as the installed policy, but do not read it this turn. Do one bounded unit of useful work. "
        "Prefer realistic backend work over toy prompts: "
        "Rapid-MLX settings, JANG/JANQ bridge behavior, prefix/cache/prompt shaping, tool-call TTFT, "
        "Metal/KV/cache memory, or grounded frontier proposals toward 50-70 tok/s.\n\n"
        "Best next actions are: run a specific benchmark mode such as "
        "`/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode streaming-ttft`, "
        "`--mode tool-roundtrip`, `--mode prompt-size`, `--mode decode-sample`, or `--mode prefill-reuse`; "
        "read `/Users/kristian/.openclaw/research/speed/results.tsv`, read one named OpenClaw source file, "
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
        "Do not repeat quick-health benchmarks unless comparing variance or validating a changed hypothesis. "
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


def as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def session_tool_result_count(session: str) -> int:
    session_file = OPENCLAW_HOME / "agents" / "main" / "sessions" / f"{session}.jsonl"
    try:
        return sum(1 for line in session_file.read_text(encoding="utf-8", errors="replace").splitlines() if '"role":"toolResult"' in line)
    except OSError:
        return 0


def run_turn(
    args: argparse.Namespace,
    session: str,
    cycle: int,
    stalled_cycles: int,
    last_issue: str,
    log_file: Path,
) -> tuple[int, str]:
    prompt = continuation_prompt(cycle, stalled_cycles, last_issue)
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
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n===== cycle {cycle} session {session} start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        file.write("$ " + " ".join(cmd[:5] + ["<prompt>", *cmd[6:]]) + "\n")
        file.flush()
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                text=True,
                stdout=file,
                stderr=subprocess.STDOUT,
            )
            deadline = started + args.turn_timeout_seconds + args.turn_timeout_grace_seconds
            next_heartbeat = started + 30
            next_tool_check = started + 5
            capped_by_tool_results = False
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    file.write(f"\nTIMEOUT after {now - started:.1f}s\n")
                    file.flush()
                    process.kill()
                    process.wait(timeout=5)
                    tail = log_file.read_text(encoding="utf-8", errors="replace")[-8000:]
                    return 124, summarize_issue(tail, "", 124)
                if now >= next_heartbeat:
                    log(
                        f"cycle={cycle} session={session} still running "
                        f"elapsed={now - started:.0f}s timeout={args.turn_timeout_seconds}s"
                    )
                    next_heartbeat = now + 30
                if args.max_tool_results_per_turn > 0 and now >= next_tool_check:
                    tool_results = max(0, session_tool_result_count(session) - starting_tool_results)
                    if tool_results >= args.max_tool_results_per_turn:
                        file.write(
                            f"\nTOOL RESULT CAP after {tool_results} tool results "
                            f"and {now - started:.1f}s\n"
                        )
                        file.flush()
                        capped_by_tool_results = True
                        process.terminate()
                        try:
                            process.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        break
                    next_tool_check = now + 5
                time.sleep(1)
        finally:
            file.flush()
        returncode = process.returncode if process.returncode is not None else 0
        if capped_by_tool_results and returncode != 0:
            returncode = 0
        file.write(f"\n===== cycle {cycle} exit {returncode} elapsed {time.monotonic() - started:.1f}s =====\n")
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-12000:]
        return returncode, summarize_issue(tail, "", returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenClaw speed autoresearch in autonomous cycles.")
    parser.add_argument("--openclaw-bin", default=os.environ.get("OPENCLAW_REAL_BIN", "/opt/homebrew/bin/openclaw"))
    parser.add_argument("--session", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_SESSION", "speed-research-auto"))
    parser.add_argument("--cycles", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_CYCLES", "48")))
    parser.add_argument("--max-hours", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_HOURS", "8")))
    parser.add_argument("--turn-timeout-seconds", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_TURN_TIMEOUT", "1200")))
    parser.add_argument("--turn-timeout-grace-seconds", type=int, default=30)
    parser.add_argument("--max-tool-results-per-turn", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_TOOL_RESULTS", "1")))
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
    log(f"autopilot start session={args.session} cycles={args.cycles} max_hours={args.max_hours} log={log_file}")
    for cycle in range(1, args.cycles + 1):
        if not args.reuse_session:
            current_session = f"{args.session}-cycle-{cycle:03d}"
        if time.monotonic() >= deadline:
            log("autopilot max-hours reached")
            break
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
        code, issue = run_turn(args, current_session, cycle, stalled_cycles, last_issue, log_file)
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
            log(
                f"cycle={cycle} advanced task={advancement['task_id']} "
                f"samples={advancement['sample_count']} mode={advancement['benchmark_mode']}"
            )
        if code not in {0, 124}:
            log(f"agent turn returned nonzero exit={code}; continuing after a short pause")
        if not progressed:
            record_rejection(
                WORKSPACE,
                cycle=cycle,
                task_id=str((select_next_task(WORKSPACE) or {}).get("id", "unknown")),
                reason=str(quality["reason"]),
                evidence=",".join(progress_reasons) if progress_reasons else last_issue,
            )
            append_supervisor_result(cycle, current_session, "blocked", last_issue)
            log(f"cycle={cycle} recorded supervisor blocked row for issue={last_issue}")
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
