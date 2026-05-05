#!/usr/bin/env python3
"""OpenClaw Rapid-MLX launcher with JANG bootstrap.

The launcher does not patch Homebrew's Rapid-MLX install. It creates an
OpenClaw-owned dependency target for ``jang_tools``, adds a tiny Python overlay
via ``PYTHONPATH``, and then execs Rapid-MLX. Rapid remains the serving engine.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


OPENCLAW_DIR = Path.home() / ".openclaw"
RUNTIME_DIR = OPENCLAW_DIR / "runtime" / "rapid-mlx"
JANG_TARGET = RUNTIME_DIR / "site"
OVERLAY_DIR = OPENCLAW_DIR / "rapid-overlay"
CHILD: subprocess.Popen[bytes] | None = None
RAPID_SERVE_HELP: str | None = None
COMPATIBILITY_FAILED = False
STOPPING = False
MEMORY_BLOCKED = False
MEMORY_BLOCK_EXIT = 75


@dataclass(frozen=True)
class LaunchProfile:
    name: str
    prefill_batch_size: int
    prefill_step_size: int
    completion_batch_size: int
    cache_memory_mb: int
    chunked_prefill_tokens: int


PROFILES = {
    "safe": LaunchProfile("safe", 4, 1024, 8, 768, 1024),
    "balanced": LaunchProfile("balanced", 8, 2048, 16, 1536, 2048),
    "turbo": LaunchProfile("turbo", 12, 4096, 24, 2048, 4096),
}


def log(message: str) -> None:
    print(f"[openclaw-rapid-launcher] {message}", file=sys.stderr, flush=True)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        log(f"ignoring invalid {name}={raw!r}")
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        log(f"ignoring invalid {name}={raw!r}")
        return default


def memory_snapshot() -> dict[str, int]:
    snapshot = {"free_mb": 0, "compressor_mb": 0, "swap_used_mb": 0, "pressure_free_pct": 0}
    try:
        vm_stat = subprocess.check_output(["/usr/bin/vm_stat"], text=True, stderr=subprocess.DEVNULL)
        page_size = 16384
        free_pages = speculative_pages = compressor_pages = 0
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
                snapshot["pressure_free_pct"] = int(float(line.rsplit(":", 1)[1].strip().rstrip("%")))
                break
    except Exception:
        pass
    try:
        swap = subprocess.check_output(["/usr/sbin/sysctl", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL)
        if "used = " in swap:
            snapshot["swap_used_mb"] = int(float(swap.split("used = ", 1)[1].split("M", 1)[0].strip()))
    except Exception:
        pass
    return snapshot


def choose_profile() -> LaunchProfile:
    requested = os.environ.get("OPENCLAW_RAPID_SPEED_PROFILE", "adaptive").lower()
    if requested in PROFILES:
        return PROFILES[requested]
    snap = memory_snapshot()
    if snap["compressor_mb"] >= 8192 or snap["swap_used_mb"] >= 8192 or snap["free_mb"] < 4096:
        return PROFILES["safe"]
    if (
        os.environ.get("OPENCLAW_RAPID_ALLOW_TURBO", "0").lower() in {"1", "true", "yes", "on"}
        and snap["free_mb"] >= 12288
        and snap["compressor_mb"] < 4096
        and snap["pressure_free_pct"] >= 20
    ):
        return PROFILES["turbo"]
    return PROFILES["balanced"]


def memory_block_reason(phase: str, snap: dict[str, int] | None = None) -> str | None:
    snap = snap or memory_snapshot()
    min_free_mb = env_int("OPENCLAW_RAPID_MIN_FREE_MB", 2048)
    max_compressor_mb = env_int("OPENCLAW_RAPID_MAX_COMPRESSOR_MB", 8192)
    max_swap_mb = env_int("OPENCLAW_RAPID_MAX_SWAP_MB", 8192)
    if snap["compressor_mb"] >= max_compressor_mb:
        return (
            f"{phase}: compressor_mb={snap['compressor_mb']}>={max_compressor_mb} "
            f"free_mb={snap['free_mb']} swap_mb={snap['swap_used_mb']}"
        )
    if snap["swap_used_mb"] >= max_swap_mb:
        return (
            f"{phase}: swap_mb={snap['swap_used_mb']}>={max_swap_mb} "
            f"free_mb={snap['free_mb']} compressor_mb={snap['compressor_mb']}"
        )
    if snap["free_mb"] and snap["free_mb"] < min_free_mb:
        return (
            f"{phase}: free_mb={snap['free_mb']}<{min_free_mb} "
            f"compressor_mb={snap['compressor_mb']} swap_mb={snap['swap_used_mb']}"
        )
    return None


def require_memory_safe(phase: str) -> bool:
    global MEMORY_BLOCKED
    reason = memory_block_reason(phase)
    if not reason:
        return True
    MEMORY_BLOCKED = True
    log(f"memory circuit breaker blocked Rapid-MLX {reason}")
    return False


def rapid_python() -> str:
    explicit = os.environ.get("OPENCLAW_RAPID_PYTHON")
    if explicit:
        return explicit
    for candidate in (
        Path("/opt/homebrew/opt/rapid-mlx/libexec/bin/python"),
        Path("/opt/homebrew/opt/rapid-mlx/libexec/bin/python3.12"),
    ):
        if candidate.exists():
            return str(candidate)
    return "/opt/homebrew/bin/python3.12"


def rapid_bin() -> str:
    return os.environ.get("OPENCLAW_RAPID_BIN", "/opt/homebrew/bin/rapid-mlx")


def rapid_serve_help() -> str:
    global RAPID_SERVE_HELP
    if RAPID_SERVE_HELP is None:
        try:
            RAPID_SERVE_HELP = subprocess.check_output(
                [rapid_bin(), "serve", "--help"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
        except Exception as error:
            log(f"could not inspect Rapid-MLX serve flags: {error}")
            RAPID_SERVE_HELP = ""
    return RAPID_SERVE_HELP


def supports_flag(flag: str) -> bool:
    return flag in rapid_serve_help()


def extend_if_supported(argv: list[str], *items: str) -> None:
    if items and supports_flag(items[0]):
        argv.extend(items)
    elif items:
        log(f"Rapid-MLX does not support {items[0]}; skipping")


def ensure_jang_target() -> None:
    JANG_TARGET.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(JANG_TARGET)
    check = [
        rapid_python(),
        "-c",
        "import jang_tools; print(getattr(jang_tools, '__version__', 'unknown'))",
    ]
    if subprocess.run(check, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return
    package = os.environ.get("OPENCLAW_RAPID_JANG_PACKAGE", "jang>=2.5.8,<3")
    log(f"installing OpenClaw-managed JANG dependency target: {package}")
    cmd = [
        rapid_python(),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--target",
        str(JANG_TARGET),
        "--upgrade",
        "--no-deps",
        package,
    ]
    subprocess.check_call(cmd)
    if subprocess.run(check, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("jang_tools install completed but import still fails")


def health_ready(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _looks_repeated(text: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]+", "", text)
    if len(compact) >= 12:
        lowered = compact.lower()
        for size in range(1, min(16, len(lowered) // 4) + 1):
            unit = lowered[:size]
            repeats, remainder = divmod(len(lowered), size)
            if repeats >= 4 and remainder == 0 and unit * repeats == lowered:
                return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())
    if len(words) >= 8 and max(words.count(word) for word in set(words)) / len(words) >= 0.7:
        return True
    return False


def _is_exact_ok_smoke_response(content: str) -> bool:
    normalized = re.sub(r"[^A-Za-z]+", "", content).upper()
    return normalized == "OK"


def compatibility_ready(host: str, port: int, model: str) -> bool:
    if os.environ.get("OPENCLAW_RAPID_SKIP_COMPAT_SMOKE", "0").lower() in {"1", "true", "yes", "on"}:
        log("compatibility smoke skipped by OPENCLAW_RAPID_SKIP_COMPAT_SMOKE")
        return True
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 12,
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": False,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=75) as response:
            body = response.read()
    except Exception as error:
        log(f"compatibility smoke failed: {error}")
        return False
    try:
        completion = json.loads(body.decode("utf-8"))
        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        finish = choice.get("finish_reason") if isinstance(choice, dict) else None
    except Exception as error:
        log(f"compatibility smoke returned invalid JSON: {error}")
        return False
    if not isinstance(content, str) or not content.strip():
        log(f"compatibility smoke returned no assistant content; finish={finish!r}")
        return False
    if _looks_repeated(content):
        log(f"compatibility smoke detected repeated-token decode loop: {content[:120]!r}")
        return False
    if not _is_exact_ok_smoke_response(content):
        log(f"compatibility smoke expected exact OK but received: {content[:120]!r}")
        return False
    log(f"compatibility smoke passed: {content[:80]!r}")
    return True


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(OVERLAY_DIR), str(JANG_TARGET)]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_argv(args: argparse.Namespace, profile: LaunchProfile) -> list[str]:
    max_num_seqs = env_int("OPENCLAW_RAPID_MAX_NUM_SEQS", 1)
    prefill_batch_size = env_int("OPENCLAW_RAPID_PREFILL_BATCH_SIZE", profile.prefill_batch_size)
    prefill_step_size = env_int("OPENCLAW_RAPID_PREFILL_STEP_SIZE", profile.prefill_step_size)
    completion_batch_size = env_int("OPENCLAW_RAPID_COMPLETION_BATCH_SIZE", profile.completion_batch_size)
    cache_memory_mb = env_int("OPENCLAW_RAPID_CACHE_MEMORY_MB", profile.cache_memory_mb)
    chunked_prefill_tokens = env_int("OPENCLAW_RAPID_CHUNKED_PREFILL_TOKENS", profile.chunked_prefill_tokens)
    gpu_memory_utilization = env_float("OPENCLAW_RAPID_GPU_MEMORY_UTILIZATION", 0.82)
    prefix_cache_size = env_int("OPENCLAW_RAPID_PREFIX_CACHE_SIZE", 100)
    argv = [
        rapid_bin(),
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
        "--enable-prefix-cache",
        "--prefix-cache-size",
        str(prefix_cache_size),
        "--cache-memory-mb",
        str(cache_memory_mb),
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
        os.environ.get("OPENCLAW_RAPID_DEFAULT_TEMPERATURE", "0.0"),
        "--default-top-p",
        os.environ.get("OPENCLAW_RAPID_DEFAULT_TOP_P", "0.95"),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--log-level",
        os.environ.get("OPENCLAW_RAPID_LOG_LEVEL", "INFO"),
    ]
    extend_if_supported(argv, "--pin-system-prompt")
    extend_if_supported(argv, "--chunked-prefill-tokens", str(chunked_prefill_tokens))
    extend_if_supported(argv, "--default-repetition-penalty", os.environ.get("OPENCLAW_RAPID_REPETITION_PENALTY", "1.08"))
    if os.environ.get("OPENCLAW_RAPID_FORCE_MLLM", "0").lower() in {"1", "true", "yes", "on"}:
        extend_if_supported(argv, "--mllm")
    if os.environ.get("OPENCLAW_RAPID_ENABLE_PLD", "0").lower() in {"1", "true", "yes", "on"}:
        argv.append("--enable-pld")
    if os.environ.get("OPENCLAW_RAPID_ENABLE_TOOL_LOGITS_BIAS", "0").lower() in {"1", "true", "yes", "on"}:
        extend_if_supported(argv, "--enable-tool-logits-bias")
    if os.environ.get("OPENCLAW_RAPID_NO_GC_CONTROL", "0").lower() in {"1", "true", "yes", "on"}:
        extend_if_supported(argv, "--no-gc-control")
    if os.environ.get("OPENCLAW_RAPID_DISABLE_PREFIX_CACHE", "1").lower() in {"1", "true", "yes", "on"}:
        cleaned = []
        skip_next = False
        for item in argv:
            if skip_next:
                skip_next = False
                continue
            if item == "--prefix-cache-size":
                skip_next = True
                continue
            if item in {"--enable-prefix-cache", "--pin-system-prompt"}:
                continue
            cleaned.append(item)
        argv = cleaned
        argv.append("--disable-prefix-cache")
    if os.environ.get("OPENCLAW_RAPID_KV_TURBOQUANT", "0").lower() in {"1", "true", "yes", "on"}:
        argv.append("--kv-cache-turboquant")
    elif os.environ.get("OPENCLAW_RAPID_KV_QUANTIZATION", "0").lower() in {"1", "true", "yes", "on"}:
        argv.extend(["--kv-cache-quantization", "--kv-cache-quantization-bits", "4"])
    return argv


def stop_child() -> None:
    global CHILD, STOPPING
    if STOPPING:
        return
    STOPPING = True
    if CHILD is None or CHILD.poll() is not None:
        STOPPING = False


def child_exit_summary(returncode: int | None) -> str:
    if returncode is None:
        return "running"
    if returncode == 0:
        return "clean exit"
    if returncode < 0:
        signal_name = {
            -6: "SIGABRT",
            -9: "SIGKILL",
            -11: "SIGSEGV",
            -15: "SIGTERM",
        }.get(returncode, f"signal {-returncode}")
        return f"fatal process exit via {signal_name}"
    return f"process exit code {returncode}"


def wait_child_with_memory_guard(profile: LaunchProfile) -> int:
    assert CHILD is not None
    interval = max(1.0, env_float("OPENCLAW_RAPID_MEMORY_CHECK_INTERVAL_SECONDS", 3.0))
    next_check = time.monotonic() + interval
    while True:
        returncode = CHILD.poll()
        if returncode is not None:
            if returncode != 0:
                log(f"profile={profile.name} stopped: {child_exit_summary(returncode)}")
            return returncode
        now = time.monotonic()
        if now >= next_check:
            if not require_memory_safe("runtime"):
                log("memory circuit breaker stopping Rapid-MLX child before macOS/Metal crash")
                stop_child()
                return MEMORY_BLOCK_EXIT
            next_check = now + interval
        time.sleep(0.25)
        return
    CHILD.terminate()
    try:
        CHILD.wait(timeout=15)
    except subprocess.TimeoutExpired:
        CHILD.kill()
        deadline = time.monotonic() + 5
        while CHILD.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if CHILD.poll() is None:
            log(f"model server process {CHILD.pid} did not exit after SIGKILL")
    finally:
        STOPPING = False


def handle_signal(_signum: int, _frame: object) -> None:
    stop_child()
    raise SystemExit(143)


def run_profile(args: argparse.Namespace, profile: LaunchProfile, startup_wait: int) -> bool:
    global CHILD, COMPATIBILITY_FAILED
    ensure_jang_target()
    if not require_memory_safe("launch"):
        return False
    env = build_env()
    argv = build_argv(args, profile)
    snap = memory_snapshot()
    log(
        "starting "
        f"profile={profile.name} free={snap['free_mb']}MB compressor={snap['compressor_mb']}MB "
        f"swap={snap['swap_used_mb']}MB pressureFree={snap['pressure_free_pct']}%"
    )
    log(f"argv={' '.join(argv)}")
    CHILD = subprocess.Popen(argv, env=env)
    deadline = time.monotonic() + startup_wait
    while time.monotonic() < deadline:
        if health_ready(args.host, args.port):
            if not compatibility_ready(args.host, args.port, args.served_model_name):
                log(
                    "Rapid backend failed OpenClaw compatibility smoke; refusing to expose "
                    "a looping or broken model server."
                )
                COMPATIBILITY_FAILED = True
                stop_child()
                return False
            log(f"ready profile={profile.name}")
            return True
        if CHILD.poll() is not None:
            log(f"profile={profile.name} exited early: {child_exit_summary(CHILD.returncode)}")
            return False
        if not require_memory_safe("startup"):
            log("memory circuit breaker stopping startup before macOS/Metal crash")
            stop_child()
            return False
        time.sleep(1)
    log(f"startup timed out profile={profile.name}")
    stop_child()
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Rapid-MLX for OpenClaw with JANG support.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--startup-wait-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    args = parse_args()
    first = choose_profile()
    fallback = PROFILES["safe"]
    if run_profile(args, first, args.startup_wait_seconds):
        assert CHILD is not None
        return wait_child_with_memory_guard(first)
    if COMPATIBILITY_FAILED:
        return 1
    if MEMORY_BLOCKED:
        return MEMORY_BLOCK_EXIT
    if first.name != fallback.name:
        stop_child()
        log("retrying with safe profile")
        if run_profile(args, fallback, args.startup_wait_seconds):
            assert CHILD is not None
            return wait_child_with_memory_guard(fallback)
        if MEMORY_BLOCKED:
            return MEMORY_BLOCK_EXIT
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
