#!/usr/bin/env python3
"""Warm OpenClaw's local model prefix cache before the first user prompt.

The first visible prompt in OpenClaw can include a large static system/context
prefix. For local MLX backends, paying that prefill cost on the first user turn
feels like a hang. This helper sends one tiny local completion against the active
profile so Rapid-MLX can cache the shared prefix before the TUI is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HOME = Path.home()
OPENCLAW_DIR = HOME / ".openclaw"
MODEL_HELPER = OPENCLAW_DIR / "bin/openclaw-model-profile"
SESSIONS_DIR = OPENCLAW_DIR / "agents/main/sessions"
WORKSPACE_DIR = OPENCLAW_DIR / "workspace"
STATE_PATH = OPENCLAW_DIR / "state/prefix-warm.json"
DEFAULT_TIMEOUT_SECONDS = 240.0
DEFAULT_MAX_SYSTEM_CHARS = 6000


def log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[openclaw-prefix-warmer] {message}", file=sys.stderr, flush=True)


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def load_active_profile() -> dict[str, Any]:
    if not MODEL_HELPER.exists():
        raise RuntimeError(f"missing model helper: {MODEL_HELPER}")
    result = subprocess.run(
        [str(MODEL_HELPER), "current"],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )
    profile = json.loads(result.stdout)
    if not isinstance(profile, dict):
        raise RuntimeError("active model profile was not a JSON object")
    return profile


def latest_trajectory_files(sessions_dir: Path = SESSIONS_DIR) -> list[Path]:
    if not sessions_dir.exists():
        return []
    files = [path for path in sessions_dir.glob("*.trajectory.jsonl") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files


def compiled_context_from_trajectory(path: Path) -> tuple[str, str] | None:
    system_prompt: str | None = None
    prompt = "[OpenClaw prefix warmup] Reply with exactly: OK"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("type") != "context.compiled":
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("systemPrompt"), str) and data["systemPrompt"].strip():
            system_prompt = data["systemPrompt"]
        if isinstance(data.get("prompt"), str) and data["prompt"].strip():
            prompt = data["prompt"]
    if not system_prompt:
        return None
    return system_prompt, prompt


def latest_compiled_context(sessions_dir: Path = SESSIONS_DIR) -> tuple[str, str, str] | None:
    max_system_chars = int_env("OPENCLAW_MODEL_PREFIX_WARM_MAX_SYSTEM_CHARS", DEFAULT_MAX_SYSTEM_CHARS)
    for path in latest_trajectory_files(sessions_dir):
        context = compiled_context_from_trajectory(path)
        if context:
            system_prompt, prompt = context
            if len(system_prompt) > max_system_chars:
                log(f"skipping large prefix source={path.name} system_chars={len(system_prompt)}>{max_system_chars}")
                continue
            return system_prompt, prompt, path.name
    return None


def fallback_workspace_context(workspace_dir: Path = WORKSPACE_DIR) -> tuple[str, str, str]:
    parts = [
        "You are a personal assistant running inside OpenClaw.",
        "Use the local OpenClaw tools carefully. Avoid loops, hidden reasoning leaks, and unbounded commands.",
    ]
    for name in ("AGENTS.md", "SOUL.md", "IDENTITY.md", "USER.md", "TOOLS.md"):
        path = workspace_dir / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        max_chars = 6000 if name == "AGENTS.md" else 2500
        parts.append(f"## {path}\n{text[:max_chars]}")
    return "\n\n".join(parts), "[OpenClaw prefix warmup] Reply with exactly: OK", "workspace-fallback"


def build_warmup_context() -> tuple[str, str, str]:
    compiled = latest_compiled_context()
    if compiled:
        return compiled
    return fallback_workspace_context()


def listener_pids(port: int) -> str:
    if port <= 0:
        return ""
    try:
        result = subprocess.run(
            ["/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return ""
    return ",".join(sorted(line.strip() for line in result.stdout.splitlines() if line.strip()))


def profile_port(profile: dict[str, Any]) -> int:
    server = profile.get("server")
    if isinstance(server, dict):
        try:
            return int(server.get("port") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def context_fingerprint(profile: dict[str, Any], system_prompt: str, source: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(profile.get("model") or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(profile.get("baseUrl") or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update(source.encode("utf-8"))
    digest.update(b"\0")
    digest.update(system_prompt.encode("utf-8"))
    return digest.hexdigest()


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_state(value: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def should_skip_warmup(
    *,
    state: dict[str, Any],
    fingerprint: str,
    pids: str,
    max_age_seconds: float,
) -> bool:
    if state.get("fingerprint") != fingerprint:
        return False
    if state.get("listenerPids") != pids:
        return False
    warmed_at = state.get("warmedAt")
    if not isinstance(warmed_at, (int, float)):
        return False
    return (time.time() - float(warmed_at)) < max_age_seconds


def http_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def warm_prefix(
    profile: dict[str, Any],
    system_prompt: str,
    prompt: str,
    *,
    timeout: float,
    max_tokens: int,
) -> tuple[float, int, bytes]:
    base_url = str(profile.get("baseUrl") or "").rstrip("/")
    model = str(profile.get("model") or "")
    if not base_url or not model:
        raise RuntimeError("active model profile needs baseUrl and model")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": False,
    }
    start = time.perf_counter()
    status, body = http_json(f"{base_url}/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - start
    if not 200 <= status < 300:
        raise RuntimeError(body.decode("utf-8", errors="replace")[:800] or f"HTTP {status}")
    return elapsed, status, body


def command_warm(args: argparse.Namespace) -> int:
    if not bool_env("OPENCLAW_MODEL_PREFIX_WARM_ENABLED", True):
        log("disabled by OPENCLAW_MODEL_PREFIX_WARM_ENABLED", quiet=args.quiet)
        return 0
    profile = load_active_profile()
    system_prompt, prompt, source = build_warmup_context()
    port = profile_port(profile)
    pids = listener_pids(port)
    fingerprint = context_fingerprint(profile, system_prompt, source)
    state = load_state()
    max_age = float_env("OPENCLAW_MODEL_PREFIX_WARM_MAX_AGE_SECONDS", 21600)
    if not args.force and should_skip_warmup(state=state, fingerprint=fingerprint, pids=pids, max_age_seconds=max_age):
        log(f"already warm for current model process ({source})", quiet=args.quiet)
        return 0
    timeout = args.timeout or float_env("OPENCLAW_MODEL_PREFIX_WARM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    max_tokens = args.max_tokens or int_env("OPENCLAW_MODEL_PREFIX_WARM_MAX_TOKENS", 1)
    log(f"warming prefix source={source} system_chars={len(system_prompt)}", quiet=args.quiet)
    elapsed, status, body = warm_prefix(profile, system_prompt, prompt, timeout=timeout, max_tokens=max_tokens)
    write_state(
        {
            "fingerprint": fingerprint,
            "listenerPids": pids,
            "model": profile.get("model"),
            "baseUrl": profile.get("baseUrl"),
            "source": source,
            "systemChars": len(system_prompt),
            "status": status,
            "responseBytes": len(body),
            "elapsedSeconds": round(elapsed, 3),
            "warmedAt": time.time(),
        }
    )
    log(f"warm complete in {elapsed:.2f}s ({source})", quiet=args.quiet)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm the OpenClaw local model prefix cache.")
    parser.add_argument("--force", action="store_true", help="warm even if the current model process was already warmed")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--timeout", type=float, help="warmup request timeout in seconds")
    parser.add_argument("--max-tokens", type=int, help="number of completion tokens for the warmup request")
    parser.set_defaults(func=command_warm)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
