#!/usr/bin/env python3
"""Checks for the OpenClaw speed autoresearch helper."""

from __future__ import annotations

import importlib.util
import json
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
            assert (root / "implementation-skill.md").exists()
            assert (root / "benchmark-manifest.json").exists()
            assert (root / "insight-rubric.json").exists()
            assert (root / "replay-buffer.jsonl").exists()
            manifest = json.loads((root / "benchmark-manifest.json").read_text(encoding="utf-8"))
            assert manifest["locked"] is True
            assert manifest["modes"]["decode-sample"]["max_tokens"] == 96
            assert manifest["modes"]["decode-sample"]["requires_usage_completion_tokens"] is True
            rubric = json.loads((root / "insight-rubric.json").read_text(encoding="utf-8"))
            assert "rollback" in rubric["required_fields"]
            replay_cases = (root / "replay-buffer.jsonl").read_text(encoding="utf-8")
            assert "decode-token-source-required" in replay_cases
            assert "profile-variant-paired-control" in replay_cases
            assert helper.replay(Namespace(allow_fail=False)) == 0
            assert helper.paired_plan(Namespace(task_id="no-drafter-control")) == 0
            paired_path = root / "experiments" / "paired-profile-plan-no-drafter-control.json"
            paired = json.loads(paired_path.read_text(encoding="utf-8"))
            assert paired["control"]["restore_before"] is True
            assert paired["promotion_gate"]["must_restore_live_profile"] is True
            assert helper.benchmark_prompt("decode-sample") == (
                "Write one compact paragraph about reducing local LLM decode latency. Keep it practical.",
                96,
            )
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
            assert "benchmark --mode decode-sample" in prompt
            assert "Use one narrow tool call" in prompt
            assert "SUMMARY.md" in prompt
            assert "results.tsv" not in prompt
            assert (root / "SUMMARY.md").exists()
            assert (root / "results-recent.tsv").exists()
            program = (root / "program.md").read_text(encoding="utf-8")
            assert "Do not touch opencode" in program
            assert "## Tool Discipline" in program
            assert "## Bootstrap Ladder" in program
            assert "## Narrow Tool Catalog" in program
            assert "find /Users" in program
            assert "## Starting Point" not in program
            assert "## Current Priority" in program
            assert "real OpenClaw decode tokens/sec" in program
            assert "openclaw-mtp-drafter-calibrate.py" in program
            assert "MTP acceptance" in program
            assert "## Frontier Speed Track" in program
            assert "50-70 tok/s" in program
            assert "Lane B: drafter alignment" in program
            assert "## Realistic Experiment Backlog" in program
            assert "No-drafter control" in program
            assert "Drafter block sweep" in program
            assert "## Speed Targets" in program
            assert "Current live baseline" in program
            assert "## Implementation Gate" in program
            assert "implementation-skill.md" in program
            implementation = (root / "implementation-skill.md").read_text(encoding="utf-8")
            assert "OpenClaw Speed Implementation Skill" in implementation
            assert "Do not touch opencode" in implementation
            assert "The patch is smaller than the problem it solves" in implementation
            assert (root / "experiments").is_dir()
            assert (root / "benchmarks").is_dir()
            assert (root / "STRATEGY.md").exists()
            assert (root / "tasks.jsonl").exists()
            assert (root / "findings.jsonl").exists()
            assert (root / "experiments.jsonl").exists()
            assert (root / "rejections.jsonl").exists()
            assert "Pre-Implementation Gate" in implementation
            assert helper.benchmark(
                Namespace(base_url="http://127.0.0.1:1/v1", model="", quick=False, mode="prompt-size", timeout=1.0)
            ) == 0
            assert helper.benchmark(
                Namespace(base_url="http://127.0.0.1:1/v1", model="", quick=False, mode="prompt-shape", timeout=1.0)
            ) == 0
            assert helper.completion_tokens_from_response(
                {"usage": {"completion_tokens": 96}},
                "short visible text",
            ) == (96, "usage.completion_tokens")
            fallback_tokens, fallback_source = helper.completion_tokens_from_response({}, "short visible text")
            assert fallback_tokens > 1
            assert fallback_source == "content_estimate"
            benchmark_rows = (root / "results.tsv").read_text(encoding="utf-8")
            assert "prompt-size" in benchmark_rows
            assert "prompt-shape" in benchmark_rows
            assert helper.synthesize(Namespace(kind="frontier")) == 0
            ideas = (root / "ideas.md").read_text(encoding="utf-8")
            assert "mtp-acceptance-bottleneck" in ideas
            assert "quality score" in ideas
            assert "cause:" in ideas
            assert "expected metric delta:" in ideas
            assert "rollback:" in ideas
            assert "drafter-block-and-quant-sweep" in ideas
            assert "janq-drafter-alignment" in ideas
            assert "mathematical handle" in ideas
            assert "Implementation Candidates" in ideas
            assert "implement-mtp-acceptance-report" in ideas
            tasks = (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert "decode-mtp-baseline" in tasks
            assert "mtp-acceptance-log-review" in tasks
            assert "post-mtp-acceptance-report" in tasks
            assert "decode-sample-baseline" in tasks
            assert "decode-sample-repeatability" in tasks
            assert "mtp-acceptance-report" in tasks
            assert "implement-mtp-acceptance-report" in tasks
            assert "implement-drafter-sweep-plan" in tasks
            assert "implement-janq-drafter-calibration-gate" in tasks
            assert '"task_type": "implementation"' in tasks
            assert "First tool call: read exactly" in tasks
            findings = (root / "findings.jsonl").read_text(encoding="utf-8")
            assert "synthesize-speed-ideas" in findings
            assert '"quality"' in findings
            assert "implementation_candidates" in findings
            assert "synthesis" in (root / "results.tsv").read_text(encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
