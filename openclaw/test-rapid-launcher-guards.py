#!/usr/bin/env python3
"""Static guard checks for the OpenClaw Rapid launcher."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch


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
    with patch.dict(
        os.environ,
        {
            "OPENCLAW_RAPID_MIN_FREE_MB": "4096",
            "OPENCLAW_RAPID_MAX_COMPRESSOR_MB": "4096",
            "OPENCLAW_RAPID_MAX_SWAP_MB": "2048",
        },
        clear=False,
    ):
        assert launcher.memory_block_reason(
            "test", {"free_mb": 3000, "compressor_mb": 1, "swap_used_mb": 0, "pressure_free_pct": 0}
        ).startswith("test: free_mb=3000<4096")
        assert launcher.memory_block_reason(
            "test", {"free_mb": 8000, "compressor_mb": 5000, "swap_used_mb": 0, "pressure_free_pct": 0}
        ).startswith("test: compressor_mb=5000>=4096")
        assert launcher.memory_block_reason(
            "test", {"free_mb": 8000, "compressor_mb": 1, "swap_used_mb": 3000, "pressure_free_pct": 0}
        ).startswith("test: swap_mb=3000>=2048")
        assert launcher.memory_block_reason(
            "test", {"free_mb": 8000, "compressor_mb": 1, "swap_used_mb": 0, "pressure_free_pct": 0}
        ) is None
    assert "SIGABRT" in launcher.child_exit_summary(-6)
    assert "SIGSEGV" in launcher.child_exit_summary(-11)
    print("ok openclaw Rapid launcher guards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
