#!/usr/bin/env python3
"""Adaptive vMLX launcher for OpenClaw local models.

The OpenClaw model profile should stay readable and model-oriented. This small
launcher owns the performance policy: choose a fast prefill profile when memory
looks healthy, fall back to conservative flags if startup exits early, and keep
all vMLX/JANG feature flags in one place.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchProfile:
    name: str
    prefill_batch_size: int
    prefill_step_size: int
    completion_batch_size: int
    cache_memory_mb: int


PROFILES = {
    "safe": LaunchProfile("safe", 8, 2048, 16, 1024),
    "balanced": LaunchProfile("balanced", 16, 4096, 32, 2048),
    "turbo": LaunchProfile("turbo", 24, 8192, 32, 3072),
}


CHILD: subprocess.Popen[bytes] | None = None


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[openclaw-vmlx-launcher] ignoring invalid {name}={raw!r}", file=sys.stderr, flush=True)
        return default


def memory_snapshot() -> dict[str, int]:
    snapshot = {"free_mb": 0, "compressor_mb": 0, "swap_used_mb": 0, "pressure_free_pct": 0}
    try:
        vm_stat = subprocess.check_output(["/usr/bin/vm_stat"], text=True, stderr=subprocess.DEVNULL)
        page_size = 16384
        free_pages = 0
        speculative_pages = 0
        compressor_pages = 0
        for line in vm_stat.splitlines():
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
        pressure = subprocess.check_output(["/usr/bin/memory_pressure"], text=True, stderr=subprocess.DEVNULL)
        for line in pressure.splitlines():
            if "System-wide memory free percentage:" in line:
                value = line.rsplit(":", 1)[1].strip().rstrip("%")
                snapshot["pressure_free_pct"] = int(float(value))
                break
    except Exception:
        pass

    try:
        swap = subprocess.check_output(["/usr/sbin/sysctl", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL)
        marker = "used = "
        if marker in swap:
            value = swap.split(marker, 1)[1].split("M", 1)[0].strip()
            snapshot["swap_used_mb"] = int(float(value))
    except Exception:
        pass

    return snapshot


def choose_profile() -> LaunchProfile:
    requested = os.environ.get("OPENCLAW_VMLX_SPEED_PROFILE", "adaptive").lower()
    if requested in PROFILES:
        return PROFILES[requested]
    if requested != "adaptive":
        print(
            f"[openclaw-vmlx-launcher] unknown OPENCLAW_VMLX_SPEED_PROFILE={requested!r}; using adaptive",
            file=sys.stderr,
            flush=True,
        )

    snap = memory_snapshot()
    if snap["compressor_mb"] >= 8192 or snap["swap_used_mb"] >= 8192 or snap["free_mb"] < 4096:
        return PROFILES["safe"]
    if snap["free_mb"] >= 12288 and snap["compressor_mb"] < 4096 and snap["pressure_free_pct"] >= 20:
        return PROFILES["turbo"]
    return PROFILES["balanced"]


def health_ready(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def build_argv(args: argparse.Namespace, profile: LaunchProfile) -> list[str]:
    vmlx_bin = os.environ.get("OPENCLAW_VMLX_BIN", os.path.expanduser("~/.local/bin/vmlx"))
    max_num_seqs = env_int("OPENCLAW_VMLX_MAX_NUM_SEQS", 1)
    prefill_batch_size = env_int("OPENCLAW_VMLX_PREFILL_BATCH_SIZE", profile.prefill_batch_size)
    prefill_step_size = env_int("OPENCLAW_VMLX_PREFILL_STEP_SIZE", profile.prefill_step_size)
    completion_batch_size = env_int("OPENCLAW_VMLX_COMPLETION_BATCH_SIZE", profile.completion_batch_size)
    cache_memory_mb = env_int("OPENCLAW_VMLX_CACHE_MEMORY_MB", profile.cache_memory_mb)

    argv = [
        vmlx_bin,
        "serve",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-num-seqs",
        str(max_num_seqs),
        "--prefill-batch-size",
        str(prefill_batch_size),
        "--prefill-step-size",
        str(prefill_step_size),
        "--completion-batch-size",
        str(completion_batch_size),
        "--continuous-batching",
        "--cache-memory-mb",
        str(cache_memory_mb),
        "--enable-pld",
        "--stream-interval",
        "1",
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "gemma4",
        "--reasoning-parser",
        "gemma4",
        "--default-temperature",
        "0.3",
        "--default-top-p",
        "0.9",
        "--default-repetition-penalty",
        "1.05",
        "--default-enable-thinking",
        "false",
        "--log-level",
        os.environ.get("OPENCLAW_VMLX_LOG_LEVEL", "INFO"),
    ]
    if os.environ.get("OPENCLAW_VMLX_PIN_SYSTEM_PROMPT", "0").lower() in {"1", "true", "yes", "on"}:
        argv.append("--pin-system-prompt")
    return argv


def stop_child() -> None:
    global CHILD
    if CHILD is None or CHILD.poll() is not None:
        return
    CHILD.terminate()
    try:
        CHILD.wait(timeout=15)
    except subprocess.TimeoutExpired:
        CHILD.kill()
        CHILD.wait(timeout=5)


def handle_signal(_signum: int, _frame: object) -> None:
    stop_child()
    raise SystemExit(143)


def run_profile(args: argparse.Namespace, profile: LaunchProfile, startup_wait: int) -> bool:
    global CHILD
    argv = build_argv(args, profile)
    snap = memory_snapshot()
    print(
        "[openclaw-vmlx-launcher] starting "
        f"profile={profile.name} free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB "
        f"swap={snap['swap_used_mb']}MB pressureFree={snap['pressure_free_pct']}%",
        file=sys.stderr,
        flush=True,
    )
    print(f"[openclaw-vmlx-launcher] argv={' '.join(argv)}", file=sys.stderr, flush=True)
    CHILD = subprocess.Popen(argv)
    deadline = time.monotonic() + startup_wait
    while time.monotonic() < deadline:
        if health_ready(args.host, args.port):
            print(f"[openclaw-vmlx-launcher] ready profile={profile.name}", file=sys.stderr, flush=True)
            return True
        if CHILD.poll() is not None:
            print(
                f"[openclaw-vmlx-launcher] profile={profile.name} exited early code={CHILD.returncode}",
                file=sys.stderr,
                flush=True,
            )
            return False
        time.sleep(1)
    print(f"[openclaw-vmlx-launcher] startup timed out profile={profile.name}", file=sys.stderr, flush=True)
    stop_child()
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch vMLX for OpenClaw with adaptive speed flags.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--startup-wait-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    args = parse_args()
    first = choose_profile()
    fallback = PROFILES["safe"]
    if run_profile(args, first, args.startup_wait_seconds):
        assert CHILD is not None
        return CHILD.wait()
    if first.name != fallback.name:
        stop_child()
        print("[openclaw-vmlx-launcher] retrying with safe profile", file=sys.stderr, flush=True)
        if run_profile(args, fallback, args.startup_wait_seconds):
            assert CHILD is not None
            return CHILD.wait()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
