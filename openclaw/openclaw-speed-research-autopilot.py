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
import subprocess
import sys
import time
from pathlib import Path


HOME = Path.home()
OPENCLAW_HOME = Path(os.environ.get("OPENCLAW_HOME", HOME / ".openclaw")).expanduser()
WORKSPACE = Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_DIR", OPENCLAW_HOME / "research" / "speed")).expanduser()
RESULTS = WORKSPACE / "results.tsv"
IDEAS = WORKSPACE / "ideas.md"
LOG_DIR = WORKSPACE / "logs"
PROGRAM = WORKSPACE / "program.md"
BLOCKED_PATTERNS = (
    "OpenClaw blocked a broad local tool command",
    "blocked this request before model execution",
    "memory pressure is too high",
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


def wait_for_memory(args: argparse.Namespace) -> None:
    while True:
        snap = memory_snapshot()
        too_hot = (
            snap["compressor_mb"] >= args.max_compressor_mb
            or snap["swap_used_mb"] >= args.max_swap_mb
            or (snap["free_mb"] and snap["free_mb"] < args.min_free_mb)
        )
        if not too_hot:
            return
        log(
            "memory gate waiting: "
            f"free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB swap={snap['swap_used_mb']}MB"
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
    return (
        "Continue OpenClaw Speed Autoresearch autonomously.\n\n"
        f"Workspace: {WORKSPACE}\n"
        f"Program: {PROGRAM}\n\n"
        "Follow program.md exactly, including Bootstrap Ladder, Narrow Tool Catalog, Current Priority, Realistic Experiment Backlog, "
        "Implementation Gate, Frontier Speed Track, and Speed Targets.\n\n"
        "Do one bounded unit of useful work this turn. Prefer realistic backend work over toy prompts: "
        "Rapid-MLX settings, JANG/JANQ bridge behavior, prefix/cache/prompt shaping, tool-call TTFT, "
        "Metal/KV/cache memory, or grounded frontier proposals toward 50-70 tok/s.\n\n"
        "Allowed narrow bootstrap actions are: read program.md, read README-openclaw-speed.md, read results.tsv, "
        "run `/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --quick`, or run "
        "`git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`. "
        "Do not run setup commands, `find`, recursive `ls`, recursive grep, or broad local search.\n\n"
        "Do not touch opencode. Do not change the primary model. Use one narrow tool call per assistant turn. "
        "Do not ask me whether to continue. If a test or benchmark cannot run safely, record blocked evidence "
        "and move to the next implementable item."
        f"\n\nAutopilot cycle: {cycle}.{pressure}"
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
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n===== cycle {cycle} session {session} start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        file.write("$ " + " ".join(cmd[:-6] + ["--message", "<prompt>", *cmd[-4:]]) + "\n")
        file.flush()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.turn_timeout_seconds + args.turn_timeout_grace_seconds,
            )
        except subprocess.TimeoutExpired as error:
            file.write(f"TIMEOUT after {time.monotonic() - started:.1f}s\n")
            stdout = as_text(error.stdout)
            stderr = as_text(error.stderr)
            file.write(stdout[-4000:])
            file.write(stderr[-4000:])
            return 124, summarize_issue(stdout, stderr, 124)
        file.write(result.stdout[-12000:])
        if result.stderr:
            file.write("\n--- stderr ---\n")
            file.write(result.stderr[-12000:])
        file.write(f"\n===== cycle {cycle} exit {result.returncode} elapsed {time.monotonic() - started:.1f}s =====\n")
        return result.returncode, summarize_issue(result.stdout, result.stderr, result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenClaw speed autoresearch in autonomous cycles.")
    parser.add_argument("--openclaw-bin", default=os.environ.get("OPENCLAW_REAL_BIN", "/opt/homebrew/bin/openclaw"))
    parser.add_argument("--session", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_SESSION", "speed-research-auto"))
    parser.add_argument("--cycles", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_CYCLES", "48")))
    parser.add_argument("--max-hours", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_HOURS", "8")))
    parser.add_argument("--turn-timeout-seconds", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_TURN_TIMEOUT", "1200")))
    parser.add_argument("--turn-timeout-grace-seconds", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=float(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_SLEEP", "8")))
    parser.add_argument("--thinking", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_THINKING", "off"))
    parser.add_argument("--min-free-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MIN_FREE_MB", "1024")))
    parser.add_argument("--max-compressor-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_COMPRESSOR_MB", "8192")))
    parser.add_argument("--max-swap-mb", type=int, default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_MAX_SWAP_MB", "8192")))
    parser.add_argument("--memory-wait-seconds", type=float, default=60.0)
    parser.add_argument(
        "--rotate-session-after-stalls",
        type=int,
        default=int(os.environ.get("OPENCLAW_SPEED_RESEARCH_AUTO_ROTATE_AFTER_STALLS", "3")),
        help="start a fresh recovery session after this many cycles without durable progress",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"autopilot-{args.session}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    deadline = time.monotonic() + args.max_hours * 3600
    stalled_cycles = 0
    last_issue = ""
    current_session = args.session
    recovery_epoch = 0
    log(f"autopilot start session={args.session} cycles={args.cycles} max_hours={args.max_hours} log={log_file}")
    for cycle in range(1, args.cycles + 1):
        if time.monotonic() >= deadline:
            log("autopilot max-hours reached")
            break
        wait_for_memory(args)
        before_lines = results_line_count()
        before_results_mtime = file_mtime(RESULTS)
        before_ideas_mtime = file_mtime(IDEAS)
        code, issue = run_turn(args, current_session, cycle, stalled_cycles, last_issue, log_file)
        after_lines = results_line_count()
        progressed = after_lines > before_lines or file_mtime(RESULTS) > before_results_mtime or file_mtime(IDEAS) > before_ideas_mtime
        stalled_cycles = 0 if progressed else stalled_cycles + 1
        last_issue = "" if progressed else (issue or "no durable progress")
        log(
            f"cycle={cycle} exit={code} progressed={progressed} "
            f"results_lines={before_lines}->{after_lines} stalled_cycles={stalled_cycles} "
            f"issue={last_issue or 'none'}"
        )
        if code not in {0, 124}:
            log(f"agent turn returned nonzero exit={code}; continuing after a short pause")
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
