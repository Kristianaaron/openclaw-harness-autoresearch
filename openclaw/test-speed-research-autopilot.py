#!/usr/bin/env python3
"""Checks for the autonomous OpenClaw speed research runner."""

from __future__ import annotations

import importlib.util
import tempfile
from argparse import Namespace
from pathlib import Path


HELPER_PATH = Path(__file__).with_name("openclaw-speed-research-autopilot.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("openclaw_speed_research_autopilot", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    helper = load_helper()
    prompt = helper.continuation_prompt(1, 0)
    assert "do not read it this turn" in prompt
    assert "Do not touch opencode" in prompt
    assert "openclaw-speed-research benchmark --quick" in prompt
    assert "Do not run setup commands" in prompt
    assert "implementation-skill.md" in prompt

    recovery_prompt = helper.continuation_prompt(3, 2, "OpenClaw blocked a broad local tool command")
    assert "Last cycle issue" in recovery_prompt
    assert "Your next tool call must be one of" in recovery_prompt
    safe_next_actions = recovery_prompt.split("Your next tool call must be one of", 1)[1].lower()
    assert "find" not in safe_next_actions

    assert helper.summarize_issue("x OpenClaw blocked a broad local tool command y", "", 0) == (
        "OpenClaw blocked a broad local tool command"
    )
    assert helper.summarize_issue("memory circuit breaker stopped backend", "", 0) == "memory circuit breaker"
    assert helper.summarize_issue("", "fatal process exit via SIGABRT", 1) == "fatal process exit"
    assert helper.summarize_issue("", "", 124) == "turn timeout"
    assert helper.summarize_issue("", "", 7) == "agent exit 7"
    assert helper.summarize_issue("all good", "", 0) == ""
    assert helper.summarize_issue("Warming up Metal shaders", "", 0) == ""
    assert helper.summarize_issue("Metal out of memory while compiling", "", 0) == "metal out of memory"
    assert helper.as_text(b"hello") == "hello"
    assert helper.as_text(None) == ""
    args = Namespace(min_free_mb=1024, ready_min_free_mb=0, max_compressor_mb=8192, max_swap_mb=8192)
    resident_snap = {"free_mb": 1396, "compressor_mb": 2088, "swap_used_mb": 1559}
    assert helper.memory_gate_reason(args, resident_snap, ready=True) == ""
    assert helper.memory_gate_reason(args, {"free_mb": 512, "compressor_mb": 0, "swap_used_mb": 0}, ready=False).startswith("free=512MB<1024MB")
    swap_hot = {"free_mb": 5000, "compressor_mb": 1000, "swap_used_mb": 9000}
    assert helper.memory_gate_reason(args, swap_hot, ready=True).startswith("swap=9000MB>=8192MB")
    low_free_resident = {"free_mb": 99, "compressor_mb": 2088, "swap_used_mb": 1559}
    assert helper.memory_gate_reason(args, low_free_resident, ready=True) == ""
    assert helper.continuation_prompt(4, 0, "rotated to fresh session after 3 stalled cycles").count(
        "Last cycle issue"
    ) == 1
    with tempfile.TemporaryDirectory() as tmp:
        helper.RESULTS = Path(tmp) / "results.tsv"
        helper.append_supervisor_result(7, "nightly", "blocked", "memory gate\tstill hot\nretry")
        text = helper.RESULTS.read_text(encoding="utf-8")
        assert text.startswith("timestamp\trun_id\tstatus")
        assert "autopilot-cycle-7" in text
        assert "session=nightly issue=memory gate still hot retry" in text
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
