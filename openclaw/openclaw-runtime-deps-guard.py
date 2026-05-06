#!/usr/bin/env python3
"""Repair OpenClaw bundled plugin runtime dependencies before agent startup.

OpenClaw can start a gateway and an embedded agent close together. If npm was
interrupted during bundled plugin runtime dependency staging, hidden npm staging
directories can be left under node_modules and later installs fail with
ENOTEMPTY before the model is called. This guard serializes repair and delegates
to OpenClaw's native plugin health command. OpenClaw 4.x exposed
`plugins deps`; OpenClaw 5.5 beta moved that surface to `plugins doctor`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


OPENCLAW_BIN = os.environ.get("OPENCLAW_REAL_BIN", "/opt/homebrew/bin/openclaw")
LOCK_TIMEOUT_SECONDS = 90
LOCK_STALE_SECONDS = 120


def log(message: str) -> None:
    print(f"OpenClaw runtime deps guard: {message}", file=sys.stderr, flush=True)


def run_openclaw(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [OPENCLAW_BIN, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def command_output(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout).strip()


def run_plugins_deps(*extra: str) -> dict[str, Any] | None:
    result = subprocess.run(
        [OPENCLAW_BIN, "plugins", "deps", "--json", *extra],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        detail = command_output(result)
        if "unknown option '--json'" in detail or "too many arguments for 'plugins'" in detail:
            return None
        raise RuntimeError(detail or f"openclaw plugins deps exited {result.returncode}")
    return json.loads(result.stdout)


def run_plugins_doctor() -> None:
    result = run_openclaw("plugins", "doctor", timeout=180)
    if result.returncode != 0:
        detail = command_output(result)
        raise RuntimeError(detail or f"openclaw plugins doctor exited {result.returncode}")


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(root: Path) -> Path:
    lock_dir = root / ".openclaw-runtime-deps-guard.lock"
    started = time.monotonic()
    while True:
        try:
            lock_dir.mkdir(parents=True)
            (lock_dir / "owner.json").write_text(json.dumps({"pid": os.getpid(), "createdAt": time.time()}) + "\n", encoding="utf-8")
            return lock_dir
        except FileExistsError:
            stale = False
            try:
                owner = json.loads((lock_dir / "owner.json").read_text(encoding="utf-8"))
                pid = int(owner.get("pid") or 0)
                created_at = float(owner.get("createdAt") or 0)
                stale = not process_alive(pid) or time.time() - created_at > LOCK_STALE_SECONDS
            except Exception:
                stale = True
            if stale:
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            if time.monotonic() - started > LOCK_TIMEOUT_SECONDS:
                raise RuntimeError(f"timed out waiting for runtime deps guard lock: {lock_dir}")
            time.sleep(0.2)


def is_npm_staging_dir(path: Path) -> bool:
    name = path.name
    return path.is_dir() and name.startswith(".") and "-" in name and name not in {".bin", ".cache"}


def cleanup_failed_npm_staging_dirs(install_root: Path) -> int:
    node_modules = install_root / "node_modules"
    if not node_modules.is_dir():
        return 0
    removed = 0
    for parent in [node_modules, *[entry for entry in node_modules.iterdir() if entry.is_dir() and entry.name.startswith("@")]]:
        for entry in parent.iterdir():
            if not is_npm_staging_dir(entry):
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed


def fallback_install_roots() -> list[Path]:
    roots: list[Path] = [Path.home() / ".openclaw"]
    try:
        real_bin = Path(OPENCLAW_BIN).resolve()
        for parent in real_bin.parents:
            if parent.name == "openclaw":
                roots.append(parent)
                break
    except Exception:
        pass
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def main() -> int:
    plan = run_plugins_deps()
    install_root = (
        Path(str(plan.get("installRoot") or "")).expanduser()
        if isinstance(plan, dict)
        else Path.home() / ".openclaw"
    )
    lock_dir = acquire_lock(install_root)
    try:
        roots = [install_root] if isinstance(plan, dict) else fallback_install_roots()
        removed = sum(cleanup_failed_npm_staging_dirs(root) for root in roots)
        if isinstance(plan, dict):
            repaired = run_plugins_deps("--repair")
            missing = repaired.get("missing") if isinstance(repaired, dict) else None
            if isinstance(missing, list) and missing:
                raise RuntimeError(f"runtime deps still missing after repair: {len(missing)}")
        else:
            run_plugins_doctor()
        if removed:
            log(f"removed {removed} stale npm staging director{'y' if removed == 1 else 'ies'}")
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
