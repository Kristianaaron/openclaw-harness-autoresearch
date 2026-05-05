#!/usr/bin/env python3
"""Checks for the OpenClaw speed autoresearch helper."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


HELPER_PATH = Path(__file__).with_name("openclaw-speed-research.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("openclaw_speed_research", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    helper = load_helper()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "research" / "speed"
        with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_DIR": str(root)}, clear=False):
            assert helper.setup_workspace(Namespace(repo_url="file:///no/such/repo")) == 0
            assert (root / "program.md").exists()
            assert (root / "README-openclaw-speed.md").exists()
            assert (root / "results.tsv").read_text(encoding="utf-8").startswith("timestamp\trun_id\tstatus")
            assert (root / "sources" / "queue.md").exists()
            assert helper.add_source(
                Namespace(source="https://example.com/speed", kind="url", title="Speed note", note="check later")
            ) == 0
            assert helper.add_source(
                Namespace(source="https://example.com/speed", kind="url", title="Speed note duplicate", note="")
            ) == 0
            source_text = (root / "sources" / "queue.md").read_text(encoding="utf-8")
            assert "Speed note" in source_text
            assert "https://example.com/speed" in source_text
            assert source_text.count("https://example.com/speed") == 1
            prompt = helper.prompt_text(root)
            assert "OpenClaw Speed Autoresearch" in prompt
            assert "First assistant action" in prompt
            assert "Use one narrow tool call" in prompt
            program = (root / "program.md").read_text(encoding="utf-8")
            assert "Do not touch opencode" in program
            assert "## Tool Discipline" in program
            assert "## Current Priority" in program
            assert "Rapid-MLX serving behavior for Gemma 4 31B JANG/JANQ" in program
            assert "## Frontier Speed Track" in program
            assert "50-70 tok/s" in program
            assert "Lane A: current-stack work" in program
            assert "## Realistic Experiment Backlog" in program
            assert "## Speed Targets" in program
            assert "Do not spend rounds on toy prompts" in program
            assert "## Implementation Gate" in program
            assert (root / "experiments").is_dir()
            assert (root / "benchmarks").is_dir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
