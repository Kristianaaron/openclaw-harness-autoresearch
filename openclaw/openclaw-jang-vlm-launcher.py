#!/usr/bin/env python3
"""Launch OpenClaw's production-safe JANG/VLM backend."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


OPENCLAW_DIR = Path.home() / ".openclaw"
RUNTIME_DIR = OPENCLAW_DIR / "runtime" / "rapid-mlx"
JANG_TARGET = RUNTIME_DIR / "site"
CHILD: subprocess.Popen[bytes] | None = None
STOPPING = False


def log(message: str) -> None:
    print(f"[openclaw-jang-vlm-launcher] {message}", file=sys.stderr, flush=True)


def rapid_python() -> str:
    explicit = os.environ.get("OPENCLAW_RAPID_PYTHON") or os.environ.get("OPENCLAW_JANG_PYTHON")
    if explicit:
        return explicit
    for candidate in (
        Path("/opt/homebrew/opt/rapid-mlx/libexec/bin/python"),
        Path("/opt/homebrew/opt/rapid-mlx/libexec/bin/python3.12"),
    ):
        if candidate.exists():
            return str(candidate)
    for candidate in sorted(Path("/opt/homebrew/Cellar/rapid-mlx").glob("*/libexec/bin/python"), reverse=True):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def ensure_jang_target() -> None:
    JANG_TARGET.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(JANG_TARGET)
    check = [rapid_python(), "-c", "import jang_tools, mlx_vlm"]
    if subprocess.run(check, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return
    package = os.environ.get("OPENCLAW_RAPID_JANG_PACKAGE", "jang>=2.5.8,<3")
    log(f"installing OpenClaw-managed JANG dependency target: {package}")
    subprocess.check_call(
        [
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
    )
    if subprocess.run(check, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("JANG dependency target installed but imports still fail")


def ensure_mtp_runtime() -> None:
    if (
        os.environ.get("OPENCLAW_JANG_DRAFT_KIND", "").strip().lower() == "dflash"
        or os.environ.get("OPENCLAW_JANG_DFLASH_DRAFT_MODEL")
    ):
        return
    if not os.environ.get("OPENCLAW_JANG_DRAFT_MODEL"):
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(JANG_TARGET)
    check = [
        rapid_python(),
        "-c",
        "import mlx_vlm.speculative.drafters.gemma4_assistant",
    ]
    if subprocess.run(check, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return
    package = os.environ.get(
        "OPENCLAW_JANG_MLX_VLM_PACKAGE",
        "git+https://github.com/Blaizzy/mlx-vlm.git@173829b1227d07b74bbbda419c6a90a28c409fe5",
    )
    log(f"installing OpenClaw-managed mlx-vlm MTP runtime: {package}")
    subprocess.check_call(
        [
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
    )
    if subprocess.run(check, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("mlx-vlm MTP runtime installed but Gemma4 assistant import still fails")


def ensure_dflash_runtime() -> None:
    if not (
        os.environ.get("OPENCLAW_JANG_DRAFT_KIND", "").strip().lower() == "dflash"
        or os.environ.get("OPENCLAW_JANG_DFLASH_DRAFT_MODEL")
    ):
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(JANG_TARGET)
    check = [rapid_python(), "-c", "import dflash.model_mlx"]
    if subprocess.run(check, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return
    package = os.environ.get("OPENCLAW_JANG_DFLASH_PACKAGE", "git+https://github.com/z-lab/dflash.git")
    log(f"installing OpenClaw-managed DFlash runtime: {package}")
    subprocess.check_call(
        [
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
    )
    if subprocess.run(check, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("DFlash runtime installed but import still fails")


def ensure_quantized_mtp_draft_model() -> None:
    draft_model = os.environ.get("OPENCLAW_JANG_DRAFT_MODEL")
    source_model = os.environ.get("OPENCLAW_JANG_DRAFT_SOURCE_MODEL")
    if not draft_model or not source_model:
        return
    draft_path = Path(draft_model).expanduser()
    if (draft_path / "model.safetensors").exists() and (draft_path / "config.json").exists():
        return
    draft_path.mkdir(parents=True, exist_ok=True)
    q_bits = os.environ.get("OPENCLAW_JANG_DRAFT_Q_BITS", "4")
    q_group_size = os.environ.get("OPENCLAW_JANG_DRAFT_Q_GROUP_SIZE", "64")
    log(
        "building quantized Gemma MTP drafter "
        f"source={source_model} target={draft_path} q_bits={q_bits}"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(JANG_TARGET)
    subprocess.check_call(
        [
            rapid_python(),
            "-m",
            "mlx_vlm.convert",
            "--hf-path",
            source_model,
            "--mlx-path",
            str(draft_path),
            "-q",
            "--q-bits",
            q_bits,
            "--q-group-size",
            q_group_size,
        ],
        env=env,
    )
    if not (draft_path / "model.safetensors").exists():
        raise RuntimeError(f"quantized Gemma MTP drafter was not created at {draft_path}")


def health_ready(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(JANG_TARGET)]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def stop_child() -> None:
    global CHILD, STOPPING
    if CHILD is None or CHILD.poll() is not None:
        return
    if STOPPING:
        return
    STOPPING = True
    CHILD.terminate()
    try:
        CHILD.wait(timeout=15)
    except subprocess.TimeoutExpired:
        CHILD.kill()
        try:
            CHILD.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log(f"child pid={CHILD.pid} did not exit after kill; leaving process supervisor to reap it")


def handle_signal(_signum: int, _frame: object) -> None:
    stop_child()
    raise SystemExit(143)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch OpenClaw JANG/VLM server.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--startup-wait-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    global CHILD
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    args = parse_args()
    ensure_jang_target()
    ensure_mtp_runtime()
    ensure_dflash_runtime()
    ensure_quantized_mtp_draft_model()
    argv = [
        rapid_python(),
        str(OPENCLAW_DIR / "servers/openclaw-jang-vlm-server.py"),
        "--model-path",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    log(f"starting argv={' '.join(argv)}")
    CHILD = subprocess.Popen(argv, env=build_env())
    deadline = time.monotonic() + args.startup_wait_seconds
    while time.monotonic() < deadline:
        if health_ready(args.host, args.port):
            log("ready")
            assert CHILD is not None
            return CHILD.wait()
        if CHILD.poll() is not None:
            log(f"server exited early code={CHILD.returncode}")
            return CHILD.returncode or 1
        time.sleep(1)
    log("startup timed out")
    stop_child()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
