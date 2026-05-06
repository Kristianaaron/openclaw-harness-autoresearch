#!/usr/bin/env python3
"""Unit checks for the OpenClaw JANG/VLM launcher."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from unittest.mock import patch


LAUNCHER_PATH = Path(__file__).with_name("openclaw-jang-vlm-launcher.py")
spec = importlib.util.spec_from_file_location("openclaw_jang_vlm_launcher", LAUNCHER_PATH)
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_dflash_runtime_dispatch() -> None:
    with patch.dict(os.environ, {"OPENCLAW_JANG_DRAFT_KIND": "dflash", "OPENCLAW_JANG_DRAFT_MODEL": "mtp"}, clear=False):
        with patch.object(launcher.subprocess, "run") as run:
            launcher.ensure_mtp_runtime()
            run.assert_not_called()

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(launcher.subprocess, "run") as run:
            launcher.ensure_dflash_runtime()
            run.assert_not_called()

    ok = subprocess.CompletedProcess(args=["python"], returncode=0)
    with patch.dict(os.environ, {"OPENCLAW_JANG_DFLASH_DRAFT_MODEL": "z-lab/gemma-4-31B-it-DFlash"}, clear=False):
        with patch.object(launcher, "rapid_python", return_value="/usr/bin/python3"):
            with patch.object(launcher.subprocess, "run", return_value=ok) as run:
                launcher.ensure_dflash_runtime()
                assert "import dflash.model_mlx" in run.call_args.args[0]


if __name__ == "__main__":
    test_dflash_runtime_dispatch()
    print("ok openclaw JANG/VLM launcher")
