#!/usr/bin/env python3
"""Static guard checks for the OpenClaw Rapid launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


LAUNCHER_PATH = Path(__file__).parent / "openclaw-rapid-launcher.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("openclaw_rapid_launcher", LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    launcher = load_launcher()
    assert launcher._is_exact_ok_smoke_response("OK")
    assert launcher._is_exact_ok_smoke_response("OK.")
    assert not launcher._is_exact_ok_smoke_response("okay")
    assert not launcher._is_exact_ok_smoke_response("eB1L deenedyto own que")
    assert launcher._looks_repeated("thought thought thought thought thought")
    print("ok openclaw Rapid launcher guards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
