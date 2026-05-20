#!/usr/bin/env python3
"""Checks for the autonomous OpenClaw speed research runner."""

from __future__ import annotations

import importlib.util
import json
import os
import time
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


HELPER_PATH = Path(__file__).with_name("openclaw-speed-research-autopilot.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("openclaw_speed_research_autopilot", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_installed_helper_freshness(helper) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "repo" / "openclaw"
        installed = root / "bin"
        source.mkdir(parents=True)
        installed.mkdir()
        files = {
            "openclaw-speed-research.py": "print('helper')\n",
            "openclaw-speed-research-autopilot.py": "print('autopilot')\n",
            "openclaw_speed_research_core.py": "VALUE = 1\n",
            "openclaw_self_improvement.py": "VALUE = 2\n",
        }
        for name, text in files.items():
            (source / name).write_text(text, encoding="utf-8")
        (installed / "openclaw-speed-research").write_text(files["openclaw-speed-research.py"], encoding="utf-8")
        (installed / "openclaw-speed-research-autopilot").write_text(
            files["openclaw-speed-research-autopilot.py"],
            encoding="utf-8",
        )
        (installed / "openclaw_speed_research_core.py").write_text(
            files["openclaw_speed_research_core.py"],
            encoding="utf-8",
        )
        (installed / "openclaw_self_improvement.py").write_text(
            files["openclaw_self_improvement.py"],
            encoding="utf-8",
        )
        original_file = helper.__file__
        helper.__file__ = str(installed / "openclaw-speed-research-autopilot")
        try:
            with patch.dict(os.environ, {"OPENCLAW_HARNESS_REPO": str(root / "repo")}, clear=False):
                args = Namespace(research_helper_bin=str(installed / "openclaw-speed-research"))
                fresh = helper.installed_helper_freshness_report(args)
                assert fresh["ok"] is True, fresh
                (installed / "openclaw-speed-research").write_text("print('stale')\n", encoding="utf-8")
                stale = helper.installed_helper_freshness_report(args)
                assert stale["ok"] is False, stale
                assert "openclaw-speed-research" in stale["stale"], stale
        finally:
            helper.__file__ = original_file


def check_prompt_and_routing_guards(helper) -> None:
    prompt = helper.continuation_prompt(1, 0)
    assert "do not read it this turn" in prompt
    assert "Do not touch opencode" in prompt
    assert "benchmark --mode decode-sample" in prompt
    assert "Primary scope: improve normal `openclaw tui` decode speed" in prompt
    assert "Autoresearch self-improvement is secondary" in prompt
    assert "MTP acceptance" in prompt
    assert "openclaw-model-proxy.log" in prompt
    assert "Do not run setup commands" in prompt
    assert "implementation-skill.md" in prompt
    assert "Do not repeat quick-health, TTFT, prompt-size, tool-roundtrip, or autoresearch meta-work" in prompt
    assert "do not append another" in prompt
    synthesis_prompt = helper.continuation_prompt(10, 0, "", helper.synthesis_task())
    assert "benchmark queue is exhausted" in synthesis_prompt
    assert "synthesize --kind frontier" in synthesis_prompt
    assert "Do not run another benchmark until synthesis" in synthesis_prompt
    implementation_task = {
        "id": "implement-mtp-acceptance-report",
        "task_type": "implementation",
        "target": "openclaw/openclaw-speed-research.py",
        "metric": "mean_accept",
        "hypothesis": "record MTP acceptance evidence for decode tuning",
        "source_files": ["openclaw/openclaw-speed-research.py", "openclaw/test-speed-research.py"],
        "acceptance": "tests pass and decode artifacts include acceptance evidence",
        "rollback": "revert only this experiment",
        "next_action": "make one source patch",
    }
    implementation_prompt = helper.continuation_prompt(11, 0, "", implementation_task)
    assert "This is an implementation task" in implementation_prompt
    assert "OpenClaw-only patch" in implementation_prompt
    assert "openclaw/test-speed-research.py" in implementation_prompt

    recovery_prompt = helper.continuation_prompt(3, 2, "OpenClaw blocked a broad local tool command")
    assert "Last cycle issue" in recovery_prompt
    assert "Your next tool call must be one of" in recovery_prompt
    assert "Cycle contract" in recovery_prompt
    assert "Supervisor recovery" in recovery_prompt
    safe_next_actions = recovery_prompt.split("Your next tool call must be one of", 1)[1].lower()
    assert "find" not in safe_next_actions
    assert helper.recovery_mode(0, "") == "normal"
    assert helper.recovery_mode(1, "no durable progress") == "force-artifact"
    assert helper.recovery_mode(2, "no durable progress") == "force-benchmark"
    assert helper.recovery_mode(3, "no durable progress") == "fresh-session"
    assert helper.recovery_mode(1, "metal out of memory") == "diagnose"
    assert helper.recovery_mode(1, "TOOL RESULT CAP") == "force-benchmark"
    assert helper.benchmark_mode_for_task({"benchmark_mode": "decode-sample"}) == "decode-sample"
    assert helper.benchmark_mode_for_task(
        {"next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode prompt-shape"}
    ) == "prompt-shape"
    assert helper.benchmark_mode_for_task({"next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --quick"}) == "quick-health"
    assert helper.is_supervisor_benchmark_task({"benchmark_mode": "decode-sample"})
    assert not helper.is_supervisor_benchmark_task({"task_type": "implementation", "benchmark_mode": "decode-sample"})
    assert helper.requires_profile_variant_runner(
        {"target": "OPENCLAW_JANG_DRAFT_MODEL", "guard_checks": ["restore_live_profile"]}
    )
    assert helper.is_supervisor_drafter_sweep_task(
        {
            "task_type": "supervisor",
            "supervisor_action": "drafter-sweep-run",
            "target": "OPENCLAW_JANG_DRAFT_BLOCK_SIZE",
            "guard_checks": ["restore_live_profile"],
        }
    )
    assert not helper.requires_profile_variant_runner(
        {
            "task_type": "supervisor",
            "supervisor_action": "drafter-sweep-plan",
            "target": "OPENCLAW_JANG_DRAFT_BLOCK_SIZE",
            "guard_checks": ["restore_live_profile"],
        }
    )
    assert helper.is_supervisor_mtp_report_task(
        {"task_type": "supervisor", "supervisor_action": "mtp-report"}
    )
    assert helper.is_supervisor_implementation_bridge_task(
        {"id": "implementation-bridge-cycle-001", "task_type": "supervisor"}
    )
    assert helper.is_supervisor_patch_execute_task(
        {"task_type": "supervisor", "supervisor_action": "patch-execute"}
    )
    assert helper.is_supervisor_drafter_fit_task(
        {
            "task_type": "supervisor",
            "supervisor_action": "drafter-fit-plan",
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-drafter-fit plan",
        }
    )
    assert helper.is_supervisor_drafter_bottleneck_review_task(
        {
            "task_type": "supervisor",
            "supervisor_action": "drafter-bottleneck-review",
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-bottleneck-review",
        }
    )
    assert helper.is_supervisor_drafter_adapter_contract_task(
        {
            "task_type": "supervisor",
            "supervisor_action": "drafter-adapter-method-contract",
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-adapter-method-contract",
        }
    )
    assert helper.is_supervisor_drafter_trace_gate_task(
        {
            "task_type": "supervisor",
            "supervisor_action": "drafter-trace-gate",
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-trace-gate --plan plan.json",
        }
    )
    assert helper.is_supervisor_focused_test_task(
        {"task_type": "supervisor", "supervisor_action": "focused-test"}
    )
    assert helper.is_supervisor_gepa_policy_canary_task(
        {"task_type": "supervisor", "supervisor_action": "gepa-policy-canary"}
    )
    assert helper.is_supervisor_runtime_overhead_map_task(
        {"task_type": "supervisor", "supervisor_action": "runtime-overhead-map"}
    )
    assert helper.is_supervisor_runtime_overhead_map_task(
        {"next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map"}
    )
    assert helper.is_supervisor_calibration_memory_report_task(
        {"next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research calibration-memory-report"}
    )
    calibration_output = (
        '{"baseline_first_draft_acceptance": 0.25, '
        '"best_first_draft_acceptance": 0.5, '
        '"adapter_file": "openclaw-logit-bias-adapter.npz"}'
    )
    assert helper.calibration_float_metric(calibration_output, "baseline_first_draft_acceptance") == 0.25
    assert helper.calibration_float_metric(calibration_output, "best_first_draft_acceptance") == 0.5
    assert helper.calibration_string_metric(calibration_output, "adapter_file") == "openclaw-logit-bias-adapter.npz"
    assert helper.is_supervisor_source_scout_task(
        {"next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research source-scout --topic frontier-decode-speed"}
    )
    assert helper.task_runs_without_model(
        {"task_type": "supervisor", "supervisor_action": "runtime-overhead-map"}
    )
    assert helper.task_runs_without_model(
        {"task_type": "supervisor", "supervisor_action": "calibration-memory-report"}
    )
    assert helper.task_runs_without_model(
        {"task_type": "supervisor", "supervisor_action": "source-scout"}
    )
    assert helper.task_runs_without_model(
        {"task_type": "supervisor", "supervisor_action": "drafter-trace-gate"}
    )
    assert helper.task_runs_without_model(
        {"task_type": "supervisor", "supervisor_action": "drafter-bottleneck-review"}
    )
    assert helper.task_runs_without_model(
        {"task_type": "supervisor", "supervisor_action": "drafter-adapter-method-contract"}
    )
    assert helper.task_runs_without_model({"benchmark_mode": "decode-sample"})
    assert not helper.task_runs_without_model(
        {"id": "model-bound-research", "target": "notes.md", "next_action": "inspect and update notes"}
    )
    assert helper.model_bound_defer_reason(
        Namespace(allow_model_bound_research_turns=False),
        {"id": "model-bound-research", "target": "notes.md", "next_action": "inspect and update notes"},
    ) == "model-bound research turn deferred for local 31B stability"
    assert helper.model_bound_defer_reason(
        Namespace(allow_model_bound_research_turns=False),
        {"task_type": "supervisor", "supervisor_action": "runtime-overhead-map"},
    ) == ""
    with patch.object(helper, "model_ready", return_value=True):
        assert helper.model_bound_defer_reason(
            Namespace(allow_model_bound_research_turns=False, allow_implementation_model_turns=False),
            {
                "id": "implementation-drafter-adapter-method-current",
                "task_type": "implementation",
                "lane": "implementation-gate",
                "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
            },
        ) == "implementation task requires deterministic patch-executor path"
    with patch.object(helper, "model_ready", return_value=True):
        assert helper.model_bound_defer_reason(
            Namespace(allow_model_bound_research_turns=False, allow_implementation_model_turns=True),
            {
                "id": "implementation-drafter-adapter-method-current",
                "task_type": "implementation",
                "lane": "implementation-gate",
                "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
            },
        ) == ""
    with patch.object(helper, "model_ready", return_value=False):
        assert helper.model_bound_defer_reason(
            Namespace(allow_model_bound_research_turns=False, allow_implementation_model_turns=True),
            {
                "id": "implementation-drafter-adapter-method-current",
                "task_type": "implementation",
                "lane": "implementation-gate",
                "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
            },
        ) == "model endpoint offline after memory recovery"
    with patch.object(helper, "model_ready", return_value=False):
        assert helper.model_bound_defer_reason(
            Namespace(allow_model_bound_research_turns=True),
            {"id": "model-bound-research", "target": "notes.md", "next_action": "inspect and update notes"},
        ) == "model endpoint offline after memory recovery"
    assert not helper.is_supervisor_benchmark_task(
        {
            "benchmark_mode": "decode-sample",
            "target": "OPENCLAW_JANG_DRAFT_MODEL",
            "guard_checks": ["restore_live_profile"],
        }
    )
    assert helper.is_supervisor_log_review_task(
        {"next_action": "tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log"}
    )
    assert helper.parse_json_object('noise {"ok": true, "value": 3} tail') == {"ok": True, "value": 3}
    assert helper.parse_json_object("not json") is None
    mtp_summary = helper.parse_mtp_log_tail(
        "[server] chat completion elapsed=6.92s tok_s=13.9 mtp_rounds=58 mean_accept=0.60\n"
        "[server] chat completion elapsed=6.38s tok_s=15.0\n"
    )
    assert mtp_summary["sample_count"] == 2
    assert mtp_summary["mean_tok_s"] == 14.45
    assert mtp_summary["mean_accept"] == 0.6
    assert helper.should_run_deterministic_fallback("TOOL RESULT CAP", {"reason": ""})
    assert helper.should_run_deterministic_fallback("", {"reason": "no durable artifact"})
    assert not helper.should_run_deterministic_fallback("memory gate waiting", {"reason": "memory"})


def check_memory_and_failure_guards(helper) -> None:
    assert helper.summarize_issue("x OpenClaw blocked a broad local tool command y", "", 0) == (
        "OpenClaw blocked a broad local tool command"
    )
    assert helper.summarize_issue("memory circuit breaker stopped backend", "", 0) == "memory circuit breaker"
    assert helper.summarize_issue("", "fatal process exit via SIGABRT", 1) == "fatal process exit"
    assert helper.summarize_issue("", "", -6) == "fatal process exit via SIGABRT"
    assert helper.summarize_issue("", "", 124) == "turn timeout"
    assert helper.summarize_issue("", "", 7) == "agent exit 7"
    assert helper.summarize_issue("all good", "", 0) == ""
    assert helper.summarize_issue("TOOL RESULT CAP after 2 tool results", "", 0) == "TOOL RESULT CAP"
    assert helper.summarize_issue("Warming up Metal shaders", "", 0) == ""
    assert helper.summarize_issue("Metal out of memory while compiling", "", 0) == "metal out of memory"
    assert helper.summarize_issue("OpenClaw model emitted malformed hidden/tool output", "", 0) == (
        "malformed hidden/tool output"
    )
    assert helper.early_failure_reason("EMBEDDED FALLBACK: Gateway agent failed") == "gateway embedded fallback"
    assert helper.early_failure_reason("rawError=Connection error.") == "model connection error"
    assert helper.early_failure_reason("normal bounded result") == ""
    assert helper.as_text(b"hello") == "hello"
    assert helper.as_text(None) == ""
    args = Namespace(
        min_free_mb=1024,
        ready_min_free_mb=0,
        min_pressure_free_pct=3,
        max_compressor_mb=8192,
        max_swap_mb=8192,
    )
    resident_snap = {"free_mb": 1396, "compressor_mb": 2088, "swap_used_mb": 1559}
    assert helper.memory_gate_reason(args, resident_snap, ready=True) == ""
    pressure_snap = {"free_mb": 5000, "compressor_mb": 0, "swap_used_mb": 0, "pressure_free_pct": 1}
    assert helper.memory_gate_reason(args, pressure_snap, ready=False).startswith("pressureFree=1%<3%")
    assert helper.memory_gate_reason(args, {"free_mb": 512, "compressor_mb": 0, "swap_used_mb": 0}, ready=False).startswith("free=512MB<1024MB")
    swap_hot = {"free_mb": 5000, "compressor_mb": 1000, "swap_used_mb": 9000}
    assert helper.memory_gate_reason(args, swap_hot, ready=True).startswith("swap=9000MB>=8192MB")
    recovered_snap = {"free_mb": 24000, "compressor_mb": 21000, "swap_used_mb": 10500, "pressure_free_pct": 62}
    assert helper.memory_gate_reason(args, recovered_snap, ready=True) == ""
    resident_recovered_snap = {"free_mb": 450, "compressor_mb": 18942, "swap_used_mb": 12600, "pressure_free_pct": 65}
    assert helper.memory_gate_reason(args, resident_recovered_snap, ready=True) == ""
    assert helper.memory_gate_reason(args, resident_recovered_snap, ready=False).startswith("compressor=18942MB")
    low_free_resident = {"free_mb": 99, "compressor_mb": 2088, "swap_used_mb": 1559}
    assert helper.memory_gate_reason(args, low_free_resident, ready=True) == ""
    active_args = Namespace(
        active_min_free_mb=128,
        active_min_pressure_free_pct=2,
        active_max_compressor_mb=6144,
        active_max_swap_mb=8192,
        active_low_free_pressure_compressor_mb=4096,
        active_low_free_pressure_swap_mb=2048,
        max_compressor_mb=8192,
        max_swap_mb=8192,
    )
    assert helper.active_memory_circuit_reason(
        active_args,
        {"free_mb": 5000, "compressor_mb": 1000, "swap_used_mb": 0, "pressure_free_pct": 1},
    ).startswith("pressureFree=1%<2%")
    assert helper.active_memory_circuit_reason(
        active_args,
        {"free_mb": 5000, "compressor_mb": 7000, "swap_used_mb": 0},
    ).startswith("compressor=7000MB")
    assert helper.active_memory_circuit_reason(
        active_args,
        {"free_mb": 5000, "compressor_mb": 1000, "swap_used_mb": 9000},
    ).startswith("swap=9000MB")
    assert helper.active_memory_circuit_reason(active_args, recovered_snap) == ""
    assert helper.active_memory_circuit_reason(active_args, resident_recovered_snap) == ""
    assert helper.active_memory_circuit_reason(
        active_args,
        {"free_mb": 77, "compressor_mb": 1000, "swap_used_mb": 0},
    ) == ""
    assert helper.active_memory_circuit_reason(
        active_args,
        {"free_mb": 77, "compressor_mb": 5000, "swap_used_mb": 0},
    ).startswith("free=77MB")
    assert helper.active_memory_circuit_reason(
        active_args,
        {"free_mb": 1000, "compressor_mb": 1000, "swap_used_mb": 0},
    ) == ""
    stable_args = Namespace(
        min_free_mb=1024,
        ready_min_free_mb=0,
        min_pressure_free_pct=3,
        max_compressor_mb=8192,
        max_swap_mb=8192,
        max_memory_wait_seconds=5,
        memory_stop_model_after_wait=False,
        memory_stable_samples=2,
        memory_stable_interval_seconds=0,
        memory_wait_seconds=0,
    )
    stable_snap = {"free_mb": 5000, "compressor_mb": 1000, "swap_used_mb": 0, "pressure_free_pct": 5}
    with patch.object(helper, "memory_snapshot", side_effect=[stable_snap, stable_snap]):
        with patch.object(helper, "model_ready", return_value=False):
            with patch.object(helper.time, "sleep", return_value=None):
                assert helper.wait_for_memory(stable_args) == (True, "")
    assert helper.continuation_prompt(4, 0, "rotated to fresh session after 3 stalled cycles").count(
        "Last cycle issue"
    ) == 1


def configure_workspace(helper, workspace: Path) -> None:
    helper.WORKSPACE = workspace
    helper.RESULTS = helper.WORKSPACE / "results.tsv"
    helper.IDEAS = helper.WORKSPACE / "ideas.md"
    helper.TASKS = helper.WORKSPACE / "tasks.jsonl"
    helper.BENCHMARKS = helper.WORKSPACE / "benchmarks"
    helper.STRATEGY = helper.WORKSPACE / "STRATEGY.md"
    helper.FINDINGS = helper.WORKSPACE / "findings.jsonl"
    helper.EXPERIMENTS = helper.WORKSPACE / "experiments.jsonl"
    helper.REJECTIONS = helper.WORKSPACE / "rejections.jsonl"
    helper.OPENCLAW_HOME = workspace / "home"
    helper.AUTOPILOT_LOCK = helper.WORKSPACE / "autopilot.lock"


def check_terminal_bridge_no_work_is_neutral(helper) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        configure_workspace(helper, Path(tmp))
        helper.ensure_task_queue()
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "handoff-audit-deterministic-bridge-unit",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "implementation-bridge",
                    "target": "implementation-bridge",
                }
            ],
        )
        task = helper.read_jsonl(helper.TASKS)[0]
        with patch.object(
            helper,
            "run_supervisor_synthesis",
            return_value=(False, "supervisor synthesis terminal no-work"),
        ):
            code, issue = helper.run_supervisor_implementation_bridge(
                Namespace(),
                7,
                "unit",
                task,
                helper.WORKSPACE / "autopilot.log",
            )
        assert (code, issue) == (0, "")
        rows = helper.all_result_rows(helper.WORKSPACE)
        bridge_rows = [row for row in rows if row.get("run_id") == "supervisor-implementation-bridge-7"]
        assert bridge_rows and bridge_rows[-1]["status"] == "keep"
        assert "terminal_no_work=True" in bridge_rows[-1]["notes"]
        tasks = helper.read_jsonl(helper.TASKS)
        assert tasks[0]["status"] == "done"
        assert tasks[0]["supervisor_summary"]["terminal_no_work"] is True


def check_quality_review_repairs_before_scoring(helper) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        configure_workspace(helper, Path(tmp))
        helper.ensure_task_queue()
        commands: list[list[str]] = []

        class Result:
            returncode = 0

        def capture_command(cmd, **_kwargs):
            commands.append(list(cmd))
            return Result()

        args = Namespace(
            research_helper_bin="/Users/kristian/.openclaw/bin/openclaw-speed-research",
            review_recent_rows=120,
            review_min_sweeps=3,
            review_min_samples_per_block=3,
            review_target_tps=30.0,
            hypothesis_rank_limit=12,
            gepa_min_blocked=3,
            gepa_min_rework=2,
            gepa_min_trajectory=2,
            gepa_min_low_quality=2,
            quality_review_timeout_seconds=5,
        )
        with patch.object(helper.subprocess, "run", side_effect=capture_command):
            ok, issue = helper.run_supervisor_quality_review(args, 9, "unit", helper.WORKSPACE / "review.log")
        assert ok is True
        assert issue == ""
        names = [cmd[1] for cmd in commands]
        assert names.index("implementation-handoff-audit") < names.index("quality-review")
        assert names.index("quality-review") < names.index("frontier-eval")


def check_autonomous_repair_owner(helper) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        configure_workspace(helper, Path(tmp))
        helper.ensure_task_queue()
        args = Namespace(
            autonomous_repair=True,
            autonomous_repair_attempts=2,
            self_improvement=True,
            self_evolution=True,
            self_improvement_recent_rows=160,
            self_improvement_timeout_seconds=5,
            self_evolution_max_variants_per_skill=2,
            self_evolution_min_score=90,
        )
        with patch.object(
            helper,
            "run_supervisor_quality_review",
            side_effect=[(True, "implementation-handoff-audit exit 2"), (True, "")],
        ) as review_mock:
            with patch.object(helper, "run_supervisor_self_improvement", return_value=(True, "")) as improve_mock:
                with patch.object(helper, "run_supervisor_synthesis", return_value=(True, "")) as synth_mock:
                    with patch.object(helper, "frontier_certification_status", return_value={"ok": True, "issues": []}):
                        with patch.object(
                            helper,
                            "deterministic_ready_tasks",
                            return_value=[{"id": "deterministic-repair-task"}],
                        ):
                            ok, issue = helper.run_autonomous_repair_loop(
                                args,
                                17,
                                "unit",
                                helper.WORKSPACE / "autopilot.log",
                                reason="unit quality drop",
                            )
        assert ok is True
        assert issue == ""
        assert review_mock.call_count == 2
        assert improve_mock.call_count == 1
        assert synth_mock.call_count == 1
        results = helper.RESULTS.read_text(encoding="utf-8")
        assert "autoresearch-autonomous-repair" in results
        assert "deterministic_ready=1" in results
        findings = helper.FINDINGS.read_text(encoding="utf-8")
        assert "autonomous-repair-owner" in findings

        configure_workspace(helper, Path(tmp) / "score-only-terminal")
        helper.ensure_task_queue()
        with patch.object(helper, "run_supervisor_quality_review", return_value=(True, "")):
            with patch.object(helper, "run_supervisor_self_improvement", return_value=(True, "")):
                with patch.object(
                    helper,
                    "run_supervisor_synthesis",
                    return_value=(False, "supervisor synthesis terminal no-work"),
                ):
                    with patch.object(
                        helper,
                        "frontier_certification_status",
                        return_value={
                            "ok": False,
                            "issues": [
                                "frontier score 9.49<min 9.8",
                                "frontier autonomy score 65.0<min 99.0",
                                "quality score 74.0<min 90",
                            ],
                            "deterministic_ready_tasks": ["progress-owner"],
                        },
                    ):
                        with patch.object(
                            helper,
                            "deterministic_ready_tasks",
                            return_value=[{"id": "progress-owner"}],
                        ):
                            ok, issue = helper.run_autonomous_repair_loop(
                                args,
                                18,
                                "score-only-terminal",
                                helper.WORKSPACE / "autopilot.log",
                                reason="score-only startup gate",
                            )
        assert ok is True
        assert issue == ""
        assert "score-only" in helper.RESULTS.read_text(encoding="utf-8")


def main() -> int:
    helper = load_helper()
    check_installed_helper_freshness(helper)
    check_prompt_and_routing_guards(helper)
    check_memory_and_failure_guards(helper)
    check_terminal_bridge_no_work_is_neutral(helper)
    check_quality_review_repairs_before_scoring(helper)
    check_autonomous_repair_owner(helper)
    with tempfile.TemporaryDirectory() as tmp:
        configure_workspace(helper, Path(tmp))
        helper.ensure_task_queue()
        first_lock = helper.acquire_autopilot_lock("unit-primary")
        assert first_lock is not None
        assert helper.acquire_autopilot_lock("unit-duplicate") is None
        assert "autopilot-lock-" in helper.RESULTS.read_text(encoding="utf-8")
        first_lock.close()
        second_lock = helper.acquire_autopilot_lock("unit-after-release")
        assert second_lock is not None
        helper.release_autopilot_lock(second_lock)
        assert not helper.AUTOPILOT_LOCK.exists()
        helper.append_interrupt_checkpoint(
            2,
            "unit-session",
            "SIGINT",
            selected_task={"id": "active-drafter-task", "status": "ready"},
        )
        interrupt_rows = [
            row for row in helper.all_result_rows(helper.WORKSPACE) if row.get("target") == "autoresearch-user-interrupt"
        ]
        assert interrupt_rows and interrupt_rows[-1]["status"] == "keep"
        interrupt_checkpoint = json.loads((helper.WORKSPACE / "autopilot-interrupt.json").read_text(encoding="utf-8"))
        assert interrupt_checkpoint["active_task"] == "active-drafter-task"
        assert "autopilot-user-interrupt" in helper.FINDINGS.read_text(encoding="utf-8")
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "model-bound-high-priority",
                    "status": "ready",
                    "priority": 99,
                    "target": "notes.md",
                    "next_action": "inspect notes",
                },
                {
                    "id": "deterministic-lower-priority",
                    "status": "ready",
                    "priority": 50,
                    "task_type": "supervisor",
                    "supervisor_action": "source-scout",
                    "target": "frontier-decode-speed-unit",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research source-scout --topic frontier-decode-speed-unit",
                },
            ],
        )
        selected = helper.select_next_runnable_task(Namespace(allow_model_bound_research_turns=False))
        assert selected["id"] != "model-bound-high-priority"
        assert helper.task_runs_without_model(selected)
        helper.ensure_task_queue()
        task_text = helper.TASKS.read_text(encoding="utf-8")
        assert "decode-mtp-baseline" in task_text
        assert "decode-sample" in helper.next_task_summary()
        seeded = helper.enqueue_recurring_decode_tasks(42, "test empty queue")
        assert seeded == 3
        recurring_tasks = helper.TASKS.read_text(encoding="utf-8")
        assert "decode-repeatability-cycle-042" in recurring_tasks
        assert "TUI-relevant real decode TPS" in recurring_tasks
        assert "mtp-acceptance-review-cycle-042" in recurring_tasks
        assert "implementation-bridge-cycle-042" in recurring_tasks
        assert '"supervisor_action": "implementation-bridge"' in recurring_tasks
        assert "autopilot-refill" in helper.RESULTS.read_text(encoding="utf-8")
        helper.write_jsonl(
            helper.TASKS,
            [
                {"id": "freeform", "status": "ready", "next_action": "inspect a broad folder"},
                {"id": "bounded-benchmark", "status": "ready", "benchmark_mode": "decode-sample"},
            ],
        )
        deterministic = helper.deterministic_ready_tasks()
        deterministic_ids = [task["id"] for task in deterministic]
        assert "bounded-benchmark" in deterministic_ids
        assert "freeform" not in deterministic_ids
        (helper.BENCHMARKS).mkdir(parents=True, exist_ok=True)
        (helper.BENCHMARKS / "frontier-system-eval-1.json").write_text(
            json.dumps({"overall": 9.4, "task_contract": {"ok": True}}),
            encoding="utf-8",
        )
        (helper.BENCHMARKS / "implementation-handoff-audit-1.json").write_text(
            json.dumps({"score": 100}),
            encoding="utf-8",
        )
        (helper.BENCHMARKS / "quality-review-1.json").write_text(
            json.dumps({"quality_score": 90, "scorecard": {"overall": 89.8}, "verdict": "healthy"}),
            encoding="utf-8",
        )
        (helper.BENCHMARKS / "frontier-autonomy-score-1.json").write_text(
            json.dumps(
                {
                    "total_score": 80,
                    "decision": "repair",
                    "hard_gate_failures": ["scorecard_at_least_continue_threshold"],
                }
            ),
            encoding="utf-8",
        )
        certification_args = Namespace(
            frontier_certification_min_score=9.0,
            frontier_certification_min_handoff=90,
            frontier_certification_min_quality=90,
        )
        certified = helper.frontier_certification_status(certification_args)
        assert certified["ok"] is True
        assert certified["autonomy_research_continue_ok"] is True
        assert helper.certification_allows_ready_work(
            {
                "issues": [
                    "frontier score 9.49<min 9.8",
                    "frontier autonomy score 65.0<min 99.0",
                    "quality score 74.0<min 90",
                ],
                "deterministic_ready_tasks": ["progress-owner"],
            }
        ) is True
        assert helper.certification_allows_ready_work(
            {
                "issues": ["task contract is not clean"],
                "deterministic_ready_tasks": ["progress-owner"],
            }
        ) is False
        assert helper.certification_allows_ready_work(
            {
                "issues": ["quality score 74.0<min 90"],
                "deterministic_ready_tasks": [],
            }
        ) is False
        (helper.BENCHMARKS / "frontier-autonomy-score-2.json").write_text(
            json.dumps(
                {
                    "total_score": 80,
                    "decision": "repair",
                    "hard_gate_failures": ["zero_active_noise"],
                }
            ),
            encoding="utf-8",
        )
        noisy_autonomy = helper.frontier_certification_status(certification_args)
        assert noisy_autonomy["ok"] is False
        assert any("frontier autonomy score" in issue for issue in noisy_autonomy["issues"])
        (helper.BENCHMARKS / "frontier-autonomy-score-3.json").write_text(
            json.dumps(
                {
                    "total_score": 100,
                    "decision": "continue",
                    "hard_gate_failures": [],
                }
            ),
            encoding="utf-8",
        )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "handoff-audit-deterministic-bridge-unit",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "implementation-bridge",
                }
            ],
        )
        helper.append_result(
            helper.WORKSPACE,
            run_id="supervisor-implementation-bridge-unit",
            status="blocked",
            target="implementation-bridge",
            hypothesis="unit empty bridge",
            commit="abc123",
            notes="seeded=0 ready_deterministic=0 issue=no deterministic implementation tasks available",
        )
        bridge_only = helper.frontier_certification_status(certification_args)
        assert bridge_only["ok"] is False
        assert "handoff-audit-deterministic-bridge-unit" in bridge_only["implementation_bridge_ready_tasks"]
        assert any("implementation bridge tasks remain ready" in issue for issue in bridge_only["issues"])
        assert helper.block_ready_empty_bridge_tasks() == 1
        helper.append_jsonl(
            helper.WORKSPACE / "exhausted-approaches.jsonl",
            {"lane": "frontier-dflash", "reason": "unit-test"},
        )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "stale-dflash",
                    "status": "ready",
                    "lane": "frontier-dflash",
                    "task_type": "supervisor",
                    "supervisor_action": "dflash-compatibility-gate",
                }
            ],
        )
        uncertified = helper.frontier_certification_status(certification_args)
        assert uncertified["ok"] is False
        assert any("exhausted lanes" in issue for issue in uncertified["issues"])
        assert helper.block_ready_exhausted_lane_tasks() == 1
        assert helper.frontier_certification_status(certification_args)["ok"] is True
        helper.append_quality_pause(43, "nightly", "unit-test exhausted synthesis")
        assert "quality-pause-43" in helper.RESULTS.read_text(encoding="utf-8")
        assert "autopilot-quality-pause" in helper.FINDINGS.read_text(encoding="utf-8")
        helper.write_jsonl(
            helper.WORKSPACE / "exhausted-approaches.jsonl",
            [
                {"lane": "frontier-dflash", "reason": "unit-test"},
                {"lane": "mtp-decode", "reason": "unit-test"},
                {"lane": "drafter-calibration-memory", "reason": "unit-test"},
            ],
        )
        (helper.BENCHMARKS / "quality-review-external.json").write_text(
            json.dumps(
                {
                    "quality_score": 97,
                    "scorecard": {"overall": 96.8},
                    "canonical_state": {
                        "state": "blocked_until_external_change",
                        "clean": True,
                        "decode_mean_tps": 15.0,
                        "noise": {"terminal_synthesis_rows": 3, "routed_terminal_synthesis_rows": 1},
                        "next": "add a new drafter candidate",
                    },
                }
            ),
            encoding="utf-8",
        )
        (helper.BENCHMARKS / "frontier-system-eval-external.json").write_text(
            json.dumps({"overall": 9.3, "task_contract": {"ok": True}}),
            encoding="utf-8",
        )
        (helper.BENCHMARKS / "implementation-handoff-audit-external.json").write_text(
            json.dumps({"score": 82}),
            encoding="utf-8",
        )
        external_args = Namespace(
            stop_on_external_blocker=True,
            external_blocker_min_quality=90,
            external_blocker_min_exhausted_core_lanes=2,
            external_blocker_min_terminal_cycles=3,
        )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "terminal-runtime-overhead",
                    "status": "ready",
                    "lane": "runtime-overhead",
                    "task_type": "supervisor",
                    "supervisor_action": "runtime-overhead-map",
                }
            ],
        )
        external_status = helper.external_change_required_status(external_args)
        assert external_status["should_stop"] is False
        assert external_status["reason"] == "ready deterministic/prerequisite task exists"
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "terminal-runtime-overhead",
                    "status": "done",
                    "lane": "runtime-overhead",
                    "task_type": "supervisor",
                    "supervisor_action": "runtime-overhead-map",
                }
            ],
        )
        external_status = helper.external_change_required_status(external_args)
        assert external_status["should_stop"] is True
        helper.append_external_change_required(44, "nightly", external_status)
        assert "external-change-required-44" in helper.RESULTS.read_text(encoding="utf-8")
        assert "autoresearch-external-change-required" in helper.FINDINGS.read_text(encoding="utf-8")
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "frontier-repair-exhaustion-mtp-report",
                    "status": "ready",
                    "lane": "exhaustion-report",
                    "task_type": "supervisor",
                    "supervisor_action": "mtp-report",
                }
            ],
        )
        mtp_status = helper.external_change_required_status(external_args)
        assert mtp_status["should_stop"] is False
        assert mtp_status["reason"] == "ready deterministic/prerequisite task exists"
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "meaningful-trace-prereq",
                    "status": "ready",
                    "lane": "drafter-trace",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-trace-gate",
                }
            ],
        )
        assert helper.external_change_required_status(external_args)["should_stop"] is False
        for artifact in helper.BENCHMARKS.glob("*.json"):
            artifact.unlink()
        helper.write_jsonl(helper.WORKSPACE / "exhausted-approaches.jsonl", [])
        extend_args = Namespace(auto_extend_cycles=True, sleep_seconds=0, rotate_session_after_stalls=3)
        extend, reason, summary = helper.should_extend_cycle_budget(
            extend_args,
            deadline=helper.time.monotonic() + 120,
            progress_cycles=1,
            blocked_cycles=0,
            stalled_cycles=0,
        )
        assert extend
        assert "ready work remains" in reason
        assert summary["ready_tasks"] >= 1
        helper.write_jsonl(helper.TASKS, [])
        extend, reason, summary = helper.should_extend_cycle_budget(
            extend_args,
            deadline=helper.time.monotonic() + 120,
            progress_cycles=3,
            blocked_cycles=1,
            stalled_cycles=0,
        )
        assert extend
        assert "supervisor synthesis" in reason
        extend, reason, summary = helper.should_extend_cycle_budget(
            extend_args,
            deadline=helper.time.monotonic() + 120,
            progress_cycles=0,
            blocked_cycles=3,
            stalled_cycles=6,
        )
        assert not extend
        assert "no durable progress" in reason
        disabled_args = Namespace(auto_extend_cycles=False, sleep_seconds=0, rotate_session_after_stalls=3)
        extend, reason, summary = helper.should_extend_cycle_budget(
            disabled_args,
            deadline=helper.time.monotonic() + 120,
            progress_cycles=3,
            blocked_cycles=0,
            stalled_cycles=0,
        )
        assert not extend
        assert "disabled" in reason
        helper.write_jsonl(helper.TASKS, [])
        helper.ensure_task_queue()
        selected = helper.select_next_task(helper.WORKSPACE)
        assert selected["id"] == "decode-mtp-baseline"
        with helper.RESULTS.open("a", encoding="utf-8") as file:
            for index, decode_tps in enumerate(("14.100", "14.600"), start=1):
                file.write(
                    f"2026-05-05T00:00:0{index}+0000\tbenchmark-{index}\tkeep\tdecode-sample\t"
                    f"bounded OpenClaw decode-sample probe\t\t\t{decode_tps}\t6.9\t1.5\tabc123\tmodel=local completion_tokens=96 token_source=usage.completion_tokens\n"
                )
        assert helper.complete_task_from_evidence(helper.WORKSPACE, selected, min_samples=3, commit="abc123") is None
        with helper.RESULTS.open("a", encoding="utf-8") as file:
            file.write(
                "2026-05-05T00:00:03+0000\tbenchmark-3\tkeep\tdecode-sample\t"
                "bounded OpenClaw decode-sample probe\t\t\t15.000\t6.6\t1.5\tabc123\tmodel=local completion_tokens=96 token_source=usage.completion_tokens\n"
            )
        advancement = helper.complete_task_from_evidence(helper.WORKSPACE, selected, min_samples=3, commit="abc123")
        assert advancement is not None
        assert advancement["task_id"] == "decode-mtp-baseline"
        assert advancement["sample_count"] == 3
        assert advancement["mean_decode_tps"] == 14.567
        assert helper.select_next_task(helper.WORKSPACE)["id"] == "mtp-acceptance-log-review"
        original_tasks = helper.read_jsonl(helper.TASKS)
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "review-drafter-calibration-canary-test",
                    "status": "ready",
                    "priority": 98,
                    "lane": "drafter-alignment",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-calibration-canary",
                    "next_action": "openclaw-speed-research drafter-calibration-canary",
                },
                {
                    "id": "drafter-calibration-memory-stage-metadata-test",
                    "status": "ready",
                    "priority": 98,
                    "lane": "drafter-alignment",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-calibration-memory-stage",
                    "stage": "metadata",
                    "next_action": "openclaw-speed-research drafter-calibration-memory-stage --stage metadata",
                },
                {
                    "id": "handoff-audit-decode-remeasure-after-calibration-block-test",
                    "status": "ready",
                    "priority": 140,
                    "lane": "runtime-overhead",
                    "task_type": "supervisor",
                    "benchmark_mode": "decode-sample",
                    "next_action": "openclaw-speed-research benchmark --mode decode-sample",
                },
            ],
        )
        assert helper.select_next_task(helper.WORKSPACE)["id"] == "drafter-calibration-memory-stage-metadata-test"
        assert helper.suppress_ready_calibration_canaries("unit-test-stage-ready", {"status": "keep"}) == 1
        queue_after_stage_lock = helper.read_jsonl(helper.TASKS)
        assert any(
            task["id"] == "review-drafter-calibration-canary-test" and task["status"] == "done"
            for task in queue_after_stage_lock
        )
        assert any(
            task["id"] == "drafter-calibration-memory-stage-metadata-test" and task["status"] == "ready"
            for task in queue_after_stage_lock
        )
        for index in range(2):
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"calibration-memory-report-loop-{index}",
                status="blocked",
                target="calibration-memory-report",
                hypothesis="Calibration plateau should become a no-model root-cause report.",
                commit="unit",
                notes="blocker=calibration-quantized-gradient-unsupported",
            )
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"drafter-adapter-method-contract-loop-{index}",
                status="keep",
                target="janq-drafter-adapter-method",
                hypothesis="convert repeated quantized-gradient failures into adapter/logit implementation",
                commit="unit",
                notes="state=adapter_calibration_memory_blocked ok=False terminal_routed=True",
            )
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"supervisor-focused-test-loop-{index}",
                status="keep",
                target="openclaw/openclaw-mtp-drafter-calibrate.py",
                hypothesis="add a canary-only adapter/logit-distillation calibration path",
                commit="unit",
                notes="focused test passed",
            )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "drafter-adapter-method-contract-current",
                    "status": "ready",
                    "priority": 100,
                    "lane": "implementation-gate",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-adapter-method-contract",
                    "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
                    "source_files": ["openclaw/openclaw-mtp-drafter-calibrate.py"],
                    "hypothesis": "repeat adapter/logit contract",
                    "metric": "adapter_method_contract",
                    "acceptance": "contract artifact exists",
                    "rollback": "discard contract",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-adapter-method-contract",
                },
                {
                    "id": "mtp-acceptance-yield-escape",
                    "status": "ready",
                    "priority": 90,
                    "lane": "frontier-deliberation",
                    "task_type": "supervisor",
                    "supervisor_action": "mtp-report",
                    "target": "openclaw-model-proxy.log",
                    "source_files": ["openclaw/openclaw-model-proxy.py"],
                    "hypothesis": "escape repeated adapter blocker by measuring acceptance yield",
                    "metric": "mean_accept",
                    "acceptance": "acceptance-yield artifact exists",
                    "rollback": "read-only task",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research mtp-report --lines 320",
                },
            ],
        )
        assert helper.select_next_task(helper.WORKSPACE)["id"] == "mtp-acceptance-yield-escape"
        helper.write_jsonl(helper.TASKS, original_tasks)
        selected_tool = helper.select_next_task(helper.WORKSPACE)
        before_claim = helper.durable_snapshot()
        selected_tool = helper.claim_task_evidence_window(helper.WORKSPACE, selected_tool, helper.results_line_count())
        assert "evidence_start_line" not in selected_tool
        after_claim = helper.durable_snapshot()
        claim_progress = helper.durable_progress(before_claim, after_claim)
        assert "task queue update" not in claim_progress
        quality = helper.cycle_quality(helper.WORKSPACE, after_claim, after_claim, [], "")
        assert quality["status"] == "blocked"
        assert helper.complete_task_from_evidence(helper.WORKSPACE, selected_tool, min_samples=3, commit="abc123") is None
        recovery_after_advance = helper.continuation_prompt(9, 2, "no durable progress")
        assert "tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log" in recovery_after_advance
        assert "baseline-recorded" in helper.EXPERIMENTS.read_text(encoding="utf-8")
        assert "supervisor advanced" in helper.FINDINGS.read_text(encoding="utf-8")
        assert "Accepted Baselines" in helper.STRATEGY.read_text(encoding="utf-8")
        invalid_decode_task = {
            "id": "invalid-decode",
            "status": "ready",
            "benchmark_mode": "decode-sample",
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
        }
        helper.write_jsonl(helper.TASKS, [invalid_decode_task])
        helper.RESULTS.write_text(
            helper.RESULTS_HEADER
            + "2026-05-05T00:00:04+0000\tbenchmark-old\tkeep\tdecode-sample\th\t\t\t10.0\t7.0\t1.5\tabc123\tmodel=local preview=old\n",
            encoding="utf-8",
        )
        assert helper.complete_task_from_evidence(helper.WORKSPACE, invalid_decode_task, min_samples=1, commit="abc123") is None
        before = helper.durable_snapshot()
        helper.IDEAS.write_text("# idea\n", encoding="utf-8")
        helper.BENCHMARKS.mkdir(exist_ok=True)
        (helper.BENCHMARKS / "one.json").write_text("{}", encoding="utf-8")
        after = helper.durable_snapshot()
        progress = helper.durable_progress(before, after)
        assert "ideas update" in progress
        assert "benchmark artifact" in progress
        quick_before = dict(before)
        quick_before["results_lines"] = 1
        helper.RESULTS.write_text(
            helper.RESULTS_HEADER
            + "2026-05-05T00:00:00+0000\tbenchmark-x\tkeep\tquick-health\th\t\t\t\t1\t\tc\tn\n",
            encoding="utf-8",
        )
        quick_after = helper.durable_snapshot()
        quality = helper.cycle_quality(helper.WORKSPACE, quick_before, quick_after, ["results row"], "")
        assert quality["status"] == "noise"
        helper.RESULTS.write_text(
            helper.RESULTS_HEADER
            + (
                "2026-05-05T00:00:01+0000\tdrafter-calibration-canary-old\tkeep\t"
                "janq-drafter-calibration-canary\th\t\t\t\t\t\tc\t"
                "decision=ready-for-bounded-calibration calibration_mode=adapter-logit-distillation "
                "seeded_stage_task=0 seeded_run_task=1\n"
            ),
            encoding="utf-8",
        )
        repeat_before = helper.durable_snapshot()
        with helper.RESULTS.open("a", encoding="utf-8") as file:
            file.write(
                "2026-05-05T00:00:02+0000\tdrafter-calibration-canary-new\tkeep\t"
                "janq-drafter-calibration-canary\th\t\t\t\t\t\tc\t"
                "decision=ready-for-bounded-calibration calibration_mode=adapter-logit-distillation "
                "seeded_stage_task=0 seeded_run_task=1\n"
            )
        repeat_after = helper.durable_snapshot()
        quality = helper.cycle_quality(helper.WORKSPACE, repeat_before, repeat_after, ["results row", "benchmark artifact"], "")
        assert quality["status"] == "noise"
        assert "repeated semantic action" in quality["reason"]
        with helper.RESULTS.open("a", encoding="utf-8") as file:
            file.write(
                "2026-05-05T00:00:03+0000\tdrafter-calibration-evaluation-unit\tkeep\t"
                "janq-drafter-calibration-evaluation\th\t\t\t\t\t\tc\t"
                "decision=reject-no-lift calibration_mode=adapter-logit-distillation "
                "baseline_acceptance=1.0 best_acceptance=1.0 acceptance_lift=0.0\n"
            )
        evaluation_after = helper.durable_snapshot()
        quality = helper.cycle_quality(helper.WORKSPACE, repeat_after, evaluation_after, ["results row"], "")
        assert quality["status"] != "noise"
        exhausted_task = {
            "id": "drafter-calibration-run-exhausted-unit",
            "status": "ready",
            "task_type": "supervisor",
            "supervisor_action": "drafter-calibration-run",
            "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
            "hypothesis": "unit exhausted fingerprint",
            "calibration_mode": "adapter-logit-distillation",
            "next_action": "openclaw-mtp-drafter-calibrate.py --calibration-mode adapter-logit-distillation",
        }
        helper.write_jsonl(helper.TASKS, [exhausted_task])
        suppressed, suppress_reason = helper.suppress_exhausted_calibration_task(
            exhausted_task,
            999,
            "unit",
            helper.LOG_DIR / "unit-suppression.log",
        )
        assert suppressed
        assert "exhausted calibration candidate fingerprint" in suppress_reason
        task_after = helper.read_jsonl(helper.TASKS)[0]
        assert task_after["status"] == "done"
        assert task_after["supervisor_summary"]["next"] == "material_candidate_change_required"
        malformed_before = dict(quick_after)
        malformed_before["results_lines"] = helper.results_line_count()
        with helper.RESULTS.open("a", encoding="utf-8") as file:
            file.write("bad\trow\n")
        malformed_after = helper.durable_snapshot()
        quality = helper.cycle_quality(helper.WORKSPACE, malformed_before, malformed_after, ["results row"], "")
        assert quality["status"] == "blocked"
        assert "malformed results.tsv" in quality["reason"]
        session_dir = helper.OPENCLAW_HOME / "agents" / "main" / "sessions"
        session_dir.mkdir(parents=True)
        session_file = session_dir / "cycle-count.jsonl"
        session_file.write_text(
            '{"message":{"role":"toolResult"}}\n'
            '{"message":{"role":"toolResult"}}\n'
            '{"message":{"role":"assistant"}}\n',
            encoding="utf-8",
        )
        assert helper.session_tool_result_count("cycle-count") == 2
        trajectory = session_dir / "trajectory-only.trajectory.jsonl"
        trajectory.write_text(
            '{"type":"trace.artifacts","data":{"toolMetas":[{"toolName":"read"},{"toolName":"exec"}]}}\n',
            encoding="utf-8",
        )
        assert helper.session_tool_result_count("trajectory-only") == 2
        assert helper.session_jsonl_path("cycle-count") == session_file
        assert helper.session_exists("cycle-count")
        assert helper.session_exists("trajectory-only")
        assert helper.session_mtime("cycle-count") > 0
        helper.append_supervisor_result(7, "nightly", "blocked", "memory gate\tstill hot\nretry")
        text = helper.RESULTS.read_text(encoding="utf-8")
        assert text.startswith("timestamp\trun_id\tstatus")
        assert "autopilot-cycle-7" in text
        assert "session=nightly issue=memory gate still hot retry" in text
        gateway_args = Namespace(
            openclaw_bin="/opt/homebrew/bin/openclaw",
            gateway_port=18789,
            gateway_health_url="",
            gateway_start_timeout_seconds=1,
        )
        with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_GATEWAY_HEALTH_URL": ""}, clear=False):
            assert helper.gateway_health_url(gateway_args) == "http://127.0.0.1:18789/health"
        assert helper.is_gateway_issue("gateway embedded fallback")
        assert helper.should_run_deterministic_fallback(
            "gateway recovery failed before agent turn",
            {"reason": "no durable artifact"},
        )
        with patch.object(helper, "gateway_ready", return_value=False):
            with patch.object(helper, "gateway_listener_pids", return_value=[]):
                with patch.object(helper, "start_gateway", return_value=(True, "")):
                    ok, issue = helper.recover_gateway(
                        gateway_args,
                        Path(tmp) / "autopilot.log",
                        reason="gateway embedded fallback",
                    )
        assert ok
        assert issue == ""
        assert "gateway-recovery" in helper.RESULTS.read_text(encoding="utf-8")
        assert "autoresearch recovered the OpenClaw gateway" in helper.FINDINGS.read_text(encoding="utf-8")
        before_failed_gateway_rows = helper.results_line_count()
        with patch.object(helper, "gateway_ready", return_value=False):
            with patch.object(helper, "gateway_listener_pids", return_value=[]):
                with patch.object(helper, "start_gateway", return_value=(False, "gateway did not become ready")):
                    ok, issue = helper.recover_gateway(
                        gateway_args,
                        Path(tmp) / "autopilot.log",
                        reason="gateway embedded fallback",
                    )
        assert not ok
        assert issue == "gateway did not become ready"
        assert helper.results_line_count() == before_failed_gateway_rows
        repeated_task = {
            "id": "blocked-impl",
            "status": "ready",
            "task_type": "implementation",
            "target": "openclaw/example.py",
        }
        helper.write_jsonl(helper.TASKS, [repeated_task])
        for cycle in range(1, 4):
            helper.record_rejection(
                helper.WORKSPACE,
                cycle=cycle,
                task_id="blocked-impl",
                reason="OpenClaw blocked a broad local tool command",
                evidence="blocked",
            )
        assert helper.block_task_after_repeated_guard(
            repeated_task,
            "OpenClaw blocked a broad local tool command",
            threshold=3,
        )
        blocked_text = helper.TASKS.read_text(encoding="utf-8")
        assert '"status": "blocked"' in blocked_text
        assert "stalled or hit a guard" in helper.FINDINGS.read_text(encoding="utf-8")
        tool_grace_task = {
            "id": "tool-grace-impl",
            "status": "ready",
            "task_type": "implementation",
            "target": "openclaw/example.py",
        }
        helper.write_jsonl(helper.TASKS, [tool_grace_task])
        helper.record_rejection(
            helper.WORKSPACE,
            cycle=4,
            task_id="tool-grace-impl",
            reason="TOOL RESULT SYNTHESIS GRACE",
            evidence="too many tools",
        )
        assert helper.block_task_after_repeated_guard(tool_grace_task, "TOOL RESULT SYNTHESIS GRACE")
        assert '"status": "blocked"' in helper.TASKS.read_text(encoding="utf-8")
        stale_task = {
            "id": "stale-impl",
            "status": "ready",
            "task_type": "implementation",
            "target": "openclaw/example.py",
        }
        helper.write_jsonl(helper.TASKS, [stale_task])
        helper.record_rejection(
            helper.WORKSPACE,
            cycle=5,
            task_id="stale-impl",
            reason="turn timeout",
            evidence="timeout",
        )
        assert helper.block_stale_rejected_implementation_tasks() == 1
        assert '"status": "blocked"' in helper.TASKS.read_text(encoding="utf-8")
        model_bound_impl = {
            "id": "model-bound-impl",
            "status": "ready",
            "task_type": "implementation",
            "target": "openclaw/example.py",
            "next_action": "inspect and patch the implementation",
        }
        helper.write_jsonl(helper.TASKS, [model_bound_impl])
        assert (
            helper.block_model_bound_implementation_tasks(
                Namespace(allow_implementation_model_turns=False)
            )
            == 1
        )
        blocked_model_bound_text = helper.TASKS.read_text(encoding="utf-8")
        assert '"status": "blocked"' in blocked_model_bound_text
        assert "deterministic patch-executor path" in blocked_model_bound_text
        helper.write_jsonl(helper.TASKS, [model_bound_impl])
        assert (
            helper.block_model_bound_implementation_tasks(
                Namespace(allow_implementation_model_turns=True)
            )
            == 0
        )
        assert '"status": "ready"' in helper.TASKS.read_text(encoding="utf-8")
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "log-review",
                    "status": "ready",
                    "target": "openclaw-model-proxy.log",
                    "hypothesis": "parse MTP logs",
                    "next_action": "tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log",
                }
            ],
        )
        proxy_log = Path(tmp) / "openclaw-model-proxy.log"
        proxy_log.write_text(
            "[openclaw-jang-vlm-server] chat completion: prompt=29 completion=96 "
            "elapsed=6.92s tok_s=13.9 mtp_rounds=58 mean_accept=0.60\n",
            encoding="utf-8",
        )
        original_proxy_env = os.environ.get("OPENCLAW_MODEL_PROXY_LOG")
        os.environ["OPENCLAW_MODEL_PROXY_LOG"] = str(proxy_log)
        try:
            code, issue = helper.run_supervisor_log_review_task(
                6,
                "test-session",
                helper.read_jsonl(helper.TASKS)[0],
                Path(tmp) / "autopilot.log",
            )
        finally:
            if original_proxy_env is None:
                os.environ.pop("OPENCLAW_MODEL_PROXY_LOG", None)
            else:
                os.environ["OPENCLAW_MODEL_PROXY_LOG"] = original_proxy_env
        assert code == 0
        assert issue == ""
        assert "supervisor-log-review-6" in helper.RESULTS.read_text(encoding="utf-8")
        assert '"status": "done"' in helper.TASKS.read_text(encoding="utf-8")
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "profile-variant",
                    "status": "ready",
                    "target": "OPENCLAW_JANG_DRAFT_MODEL",
                    "hypothesis": "paired no-drafter control",
                    "guard_checks": ["restore_live_profile"],
                }
            ],
        )
        code, issue = helper.run_supervisor_profile_variant_guard(
            7,
            "test-session",
            helper.read_jsonl(helper.TASKS)[0],
            Path(tmp) / "autopilot.log",
        )
        assert code == 2
        assert "dedicated paired-control runner" in issue
        assert "supervisor-profile-variant-7" in helper.RESULTS.read_text(encoding="utf-8")
        assert '"status": "blocked"' in helper.TASKS.read_text(encoding="utf-8")
        paired_plan = helper.WORKSPACE / "experiments" / "paired-profile-plan-profile-variant.json"
        assert paired_plan.exists()
        assert "must_restore_live_profile" in paired_plan.read_text(encoding="utf-8")
        sweep_helper = Path(tmp) / "sweep-helper.py"
        sweep_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "path = pathlib.Path(sys.argv[sys.argv.index('--blocks') + 1].replace(',', '-') + '.json')\n"
            "print(json.dumps({'ok': True, 'path': str(path), 'plan': {'promotion_gate': {'must_not_change_live_profile': True}}}))\n",
            encoding="utf-8",
        )
        sweep_helper.chmod(0o700)
        sweep_args = Namespace(research_helper_bin=str(sweep_helper), model_start_timeout_seconds=1)
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "sweep",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-sweep-plan",
                    "target": "OPENCLAW_JANG_DRAFT_BLOCK_SIZE",
                    "hypothesis": "plan block sweep",
                    "blocks": "1,2",
                }
            ],
        )
        code, issue = helper.run_supervisor_drafter_sweep_plan(
            sweep_args,
            8,
            "test-session",
            helper.read_jsonl(helper.TASKS)[0],
            Path(tmp) / "autopilot.log",
        )
        assert code == 0
        assert issue == ""
        assert "supervisor-drafter-sweep-8" in helper.RESULTS.read_text(encoding="utf-8")
        assert '"status": "done"' in helper.TASKS.read_text(encoding="utf-8")
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "sweep-run",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-sweep-run",
                    "target": "OPENCLAW_JANG_DRAFT_BLOCK_SIZE",
                    "hypothesis": "run block sweep",
                    "blocks": "1,2",
                    "samples": 1,
                }
            ],
        )
        original_model_ready = helper.model_ready
        original_start_model = helper.start_model_for_supervisor_benchmark
        start_calls = []
        helper.model_ready = lambda: True
        helper.start_model_for_supervisor_benchmark = lambda log_file, timeout: start_calls.append((str(log_file), timeout))
        try:
            code, issue = helper.run_supervisor_drafter_sweep_plan(
                sweep_args,
                9,
                "test-session",
                helper.read_jsonl(helper.TASKS)[0],
                Path(tmp) / "autopilot.log",
            )
        finally:
            helper.model_ready = original_model_ready
            helper.start_model_for_supervisor_benchmark = original_start_model
        assert code == 0
        assert issue == ""
        assert start_calls
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "sweep-run-offline",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-sweep-run",
                    "target": "OPENCLAW_JANG_DRAFT_BLOCK_SIZE",
                    "hypothesis": "run block sweep",
                    "blocks": "1,2",
                    "samples": 1,
                }
            ],
        )
        helper.model_ready = lambda: False
        helper.start_model_for_supervisor_benchmark = lambda log_file, timeout: None
        try:
            code, issue = helper.run_supervisor_drafter_sweep_plan(
                sweep_args,
                10,
                "test-session",
                helper.read_jsonl(helper.TASKS)[0],
                Path(tmp) / "autopilot.log",
            )
        finally:
            helper.model_ready = original_model_ready
            helper.start_model_for_supervisor_benchmark = original_start_model
        assert code == 75
        assert issue == "model endpoint unavailable after model-start"
        assert '"status": "ready"' in helper.TASKS.read_text(encoding="utf-8")
        fit_helper = Path(tmp) / "fit-helper.py"
        fit_plan = Path(tmp) / "fit-plan.json"
        fit_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "payload = {'ok': True, 'decision': 'ready-for-target-generated-trace-data', 'promotion_gate': {'minimum_speedup_vs_current': 1.35, 'minimum_mean_accept': 2.25}}\n"
            "out.parent.mkdir(parents=True, exist_ok=True)\n"
            "out.write_text(json.dumps(payload), encoding='utf-8')\n"
            "print(json.dumps(payload))\n",
            encoding="utf-8",
        )
        fit_helper.chmod(0o700)
        fit_args = Namespace(drafter_fit_bin=str(fit_helper))
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "fit",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-fit-plan",
                    "target": str(fit_plan),
                    "target_path": str(Path(tmp) / "target"),
                    "drafter_path": str(Path(tmp) / "draft"),
                    "output": str(fit_plan),
                    "hypothesis": "plan JANQ fit",
                }
            ],
        )
        code, issue = helper.run_supervisor_drafter_fit_task(
            fit_args,
            9,
            "test-session",
            helper.read_jsonl(helper.TASKS)[0],
            Path(tmp) / "autopilot.log",
        )
        assert code == 0
        assert issue == ""
        assert fit_plan.exists()
        assert "supervisor-drafter-fit-9" in helper.RESULTS.read_text(encoding="utf-8")
        benchmark_helper = Path(tmp) / "benchmark-helper.py"
        benchmark_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'ok': True, 'mode': 'decode-sample', 'decode_tps': 14.5, 'completion_tokens': 96}))\n",
            encoding="utf-8",
        )
        benchmark_helper.chmod(0o700)
        benchmark_task = {
            "id": "lane-contract-decode-remeasure-unit",
            "status": "ready",
            "lane": "runtime-overhead",
            "task_type": "supervisor",
            "benchmark_mode": "decode-sample",
            "target": "decode-sample",
            "hypothesis": "unit benchmark task should be completed by the supervisor",
        }
        helper.write_jsonl(helper.TASKS, [benchmark_task])
        with patch.object(helper, "ensure_model_for_supervisor_task", return_value=(True, "")):
            code, issue = helper.run_supervisor_benchmark_task(
                Namespace(
                    research_helper_bin=str(benchmark_helper),
                    supervisor_benchmark_timeout_seconds=10,
                    model_start_timeout_seconds=1.0,
                ),
                10,
                "test-session",
                helper.read_jsonl(helper.TASKS)[0],
                Path(tmp) / "autopilot.log",
            )
        assert code == 0
        assert issue == ""
        completed_benchmark_task = helper.read_jsonl(helper.TASKS)[0]
        assert completed_benchmark_task["status"] == "done"
        assert completed_benchmark_task["supervisor_summary"]["result"]["decode_tps"] == 14.5
        trace_helper = Path(tmp) / "trace-helper.py"
        trace_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'ok': True, 'status': 'blocked', 'reason': 'target-generated-trace-data-missing'}))\n",
            encoding="utf-8",
        )
        trace_helper.chmod(0o700)
        trace_args = Namespace(research_helper_bin=str(trace_helper))
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "trace-gate",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-trace-gate",
                    "target": str(fit_plan),
                    "hypothesis": "validate JANQ trace data",
                }
            ],
        )
        code, issue = helper.run_supervisor_drafter_trace_gate_task(
            trace_args,
            10,
            "test-session",
            helper.read_jsonl(helper.TASKS)[0],
            Path(tmp) / "autopilot.log",
        )
        assert code == 0
        assert issue == "target-generated-trace-data-missing"
        assert helper.read_jsonl(helper.TASKS)[0]["status"] == "blocked"
        dflash_helper = Path(tmp) / "dflash-helper.py"
        dflash_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'ok': True, 'status': 'keep', 'decision': 'canary-plan-ready', 'blockers': []}))\n",
            encoding="utf-8",
        )
        dflash_helper.chmod(0o700)
        dflash_args = Namespace(research_helper_bin=str(dflash_helper))
        legacy_dflash_task = {
            "id": "deliberate-dflash-compatibility-1",
            "status": "ready",
            "lane": "frontier-dflash",
            "task_type": "research",
            "target": "dflash.model_mlx/openclaw-jang-vlm-server.py",
            "hypothesis": "legacy model-bound DFlash task should be supervisor-routed",
        }
        assert helper.is_supervisor_dflash_compatibility_task(legacy_dflash_task)
        assert helper.task_runs_without_model(legacy_dflash_task)
        helper.write_jsonl(helper.TASKS, [legacy_dflash_task])
        code, issue = helper.run_supervisor_dflash_compatibility_task(
            dflash_args,
            10,
            "test-session",
            helper.read_jsonl(helper.TASKS)[0],
            Path(tmp) / "autopilot.log",
        )
        assert code == 0
        assert issue == ""
        assert helper.read_jsonl(helper.TASKS)[0]["status"] == "done"
        dflash_blocked_report = helper.WORKSPACE / "experiments" / "dflash-compatibility-gate-123.json"
        dflash_blocked_report.parent.mkdir(parents=True, exist_ok=True)
        dflash_blocked_report.write_text(
            '{"status":"blocked","blockers":["draft_model_type_mismatch=qwen3"]}',
            encoding="utf-8",
        )
        helper.append_result(
            helper.WORKSPACE,
            run_id="dflash-compatibility-gate-123",
            status="blocked",
            target="frontier-dflash",
            hypothesis="DFlash gate blocks a mismatched draft model",
            commit="abc123",
            notes="decision=blocked",
        )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "deliberate-dflash-compatibility-stale",
                    "status": "ready",
                    "lane": "frontier-dflash",
                    "task_type": "supervisor",
                    "supervisor_action": "dflash-compatibility-gate",
                    "priority": 99,
                },
                {
                    "id": "safe-runtime-map",
                    "status": "ready",
                    "lane": "runtime-overhead",
                    "task_type": "supervisor",
                    "supervisor_action": "runtime-overhead-map",
                    "priority": 50,
                },
            ],
        )
        assert helper.recent_hard_dflash_blocker() == "draft_model_type_mismatch=qwen3"
        assert helper.block_stale_hard_blocked_lane_tasks() == 1
        quarantined_tasks = helper.read_jsonl(helper.TASKS)
        assert quarantined_tasks[0]["status"] == "blocked"
        assert quarantined_tasks[1]["status"] == "ready"
        assert "frontier-dflash" in helper.read_jsonl(helper.WORKSPACE / "exhausted-approaches.jsonl")[0]["lane"]
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "causal-review-old",
                    "status": "ready",
                    "lane": "causal-repair",
                    "priority": 99,
                    "next_action": "read exactly promotion-decisions.jsonl",
                },
                {
                    "id": "supervisor-mtp-report",
                    "status": "ready",
                    "lane": "production-mtp",
                    "task_type": "supervisor",
                    "supervisor_action": "mtp-report",
                    "priority": 50,
                },
            ],
        )
        assert helper.block_stale_model_bound_causal_tasks() == 1
        causal_tasks = helper.read_jsonl(helper.TASKS)
        assert causal_tasks[0]["status"] == "blocked"
        assert causal_tasks[1]["status"] == "ready"
        patch_helper = Path(tmp) / "patch-helper.py"
        patch_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'ok': True, 'promoted': False, 'classification': {'impact': 'safe'}}))\n",
            encoding="utf-8",
        )
        patch_helper.chmod(0o700)
        patch_file = Path(tmp) / "proposal.patch"
        patch_file.write_text("diff --git a/openclaw/x.py b/openclaw/x.py\n", encoding="utf-8")
        patch_args = Namespace(
            research_helper_bin=str(patch_helper),
            patch_execute_timeout_seconds=5,
            architectural_approval_file=str(Path(tmp) / "arch-approval.txt"),
        )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "patch",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "patch-execute",
                    "patch_file": str(patch_file),
                    "source_files": ["openclaw/x.py"],
                    "tests": ["python3 openclaw/test-speed-research.py"],
                    "hypothesis": "canary patch",
                    "allow_architectural": True,
                }
            ],
        )
        code, issue = helper.run_supervisor_patch_execute_task(
            patch_args,
            10,
            "test-session",
            helper.read_jsonl(helper.TASKS)[0],
            Path(tmp) / "autopilot.log",
        )
        assert code == 0
        assert issue == ""
        assert '"status": "done"' in helper.TASKS.read_text(encoding="utf-8")
        patch_log = (Path(tmp) / "autopilot.log").read_text(encoding="utf-8")
        assert "--allow-architectural" in patch_log
        assert "--architectural-approval-file" in patch_log
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "focused",
                    "status": "ready",
                    "blocked_at": "2026-05-10T00:00:00+0000",
                    "blocked_reason": "stale pre-revive blocker",
                    "task_type": "supervisor",
                    "supervisor_action": "focused-test",
                    "target": "openclaw/test-drafter-fit.py",
                    "hypothesis": "focused tests pass",
                    "next_action": "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-drafter-fit.py",
                }
            ],
        )
        code, issue = helper.run_supervisor_focused_test_task(
            10,
            "test-session",
            helper.read_jsonl(helper.TASKS)[0],
            Path(tmp) / "autopilot.log",
        )
        assert code == 0
        assert issue == ""
        assert "supervisor-focused-test-10" in helper.RESULTS.read_text(encoding="utf-8")
        focused_task = helper.read_jsonl(helper.TASKS)[0]
        assert focused_task["status"] == "done"
        assert "completed_at" in focused_task
        assert "blocked_at" not in focused_task
        assert "blocked_reason" not in focused_task
        gepa_helper = Path(tmp) / "gepa-helper.py"
        gepa_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "task_id = sys.argv[sys.argv.index('--task-id') + 1]\n"
            f"pathlib.Path({str(helper.RESULTS)!r}).write_text("
            f"{helper.RESULTS_HEADER!r} + "
            "'2026-05-05T00:00:00+0000\\tgepa-policy-canary-test\\tkeep\\tautoresearch-gepa-policy-canary\\th\\t\\t\\t\\t\\t\\tabc123\\ttask_id=' + task_id + '\\n', "
            "encoding='utf-8')\n"
            "print(json.dumps({'ok': True, 'path': '/tmp/gepa-canary.json'}))\n",
            encoding="utf-8",
        )
        gepa_helper.chmod(0o700)
        gepa_args = Namespace(research_helper_bin=str(gepa_helper), gepa_canary_timeout_seconds=5)
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "gepa",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "gepa-policy-canary",
                    "target": "program.md",
                }
            ],
        )
        code, issue = helper.run_supervisor_gepa_policy_canary_task(
            gepa_args,
            11,
            "test-session",
            helper.read_jsonl(helper.TASKS)[0],
            Path(tmp) / "autopilot.log",
        )
        assert code == 0
        assert issue == ""
        assert "gepa-policy-canary-test" in helper.RESULTS.read_text(encoding="utf-8")
        synth_helper = Path(tmp) / "synthesize-helper.py"
        synth_marker = Path(tmp) / "synth-marker.txt"
        synth_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            f"pathlib.Path({str(synth_marker)!r}).write_text(' '.join(sys.argv[1:]), encoding='utf-8')\n",
            encoding="utf-8",
        )
        synth_helper.chmod(0o700)
        synth_args = Namespace(research_helper_bin=str(synth_helper), synthesis_timeout_seconds=5)
        ok, issue = helper.run_supervisor_synthesis(synth_args, 8, "nightly", Path(tmp) / "autopilot.log")
        assert ok
        assert issue == ""
        assert synth_marker.read_text(encoding="utf-8") == "synthesize --kind frontier"
        timeout_helper = Path(tmp) / "synthesize-timeout-helper.py"
        timeout_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import time\n"
            "time.sleep(2)\n",
            encoding="utf-8",
        )
        timeout_helper.chmod(0o700)
        timeout_args = Namespace(research_helper_bin=str(timeout_helper), synthesis_timeout_seconds=0.05)
        ok, issue = helper.run_supervisor_synthesis(timeout_args, 9, "nightly", Path(tmp) / "autopilot.log")
        assert ok
        assert "synthesis timeout" in issue
        tasks = helper.read_jsonl(helper.TASKS)
        recovery = [task for task in tasks if str(task.get("id", "")).startswith("synthesis-timeout-recovery-")]
        assert len(recovery) == 1
        assert recovery[0]["supervisor_action"] == "drafter-bottleneck-review"
        assert "autoresearch-synthesis-timeout-recovery" in helper.RESULTS.read_text(encoding="utf-8")
        before_task_count = len(tasks)
        ok, issue = helper.run_supervisor_synthesis(timeout_args, 10, "nightly", Path(tmp) / "autopilot.log")
        assert ok
        assert "existing synthesis-timeout recovery task" in issue
        assert len(helper.read_jsonl(helper.TASKS)) == before_task_count
        helper.append_result(
            helper.WORKSPACE,
            run_id="drafter-material-candidate-exhausted-unit",
            status="blocked",
            target="janq-drafter-material-candidate",
            hypothesis="unit material candidates are exhausted",
            commit="unit-test",
            notes="decision=all-material-drafter-candidates-exhausted",
        )
        ok, issue = helper.run_supervisor_synthesis(timeout_args, 11, "nightly", Path(tmp) / "autopilot.log")
        assert ok
        assert "synthesis timeout" in issue
        tasks = helper.read_jsonl(helper.TASKS)
        stale_recovery = [
            task for task in tasks if str(task.get("id", "")).startswith("synthesis-timeout-recovery-")
        ]
        assert stale_recovery and all(task["status"] == "blocked" for task in stale_recovery)
        breakouts = [task for task in tasks if str(task.get("id", "")).startswith("material-exhaustion-breakout-")]
        assert len(breakouts) == 1
        assert breakouts[0]["supervisor_action"] == "focused-test"
        assert breakouts[0]["metric"] == "material_exhaustion_breakout_contract"
        helper.append_result(
            helper.WORKSPACE,
            run_id="supervisor-focused-test-material-exhaustion-unit",
            status="keep",
            target="openclaw/openclaw-mtp-drafter-calibrate.py",
            hypothesis="All current JANQ drafter material candidates are exhausted.",
            commit="unit-test",
            notes="focused test passed",
        )
        ok, issue = helper.run_supervisor_synthesis(timeout_args, 12, "nightly", Path(tmp) / "autopilot.log")
        assert ok
        assert "material exhaustion breakout already proved" in issue
        tasks = helper.read_jsonl(helper.TASKS)
        breakouts_after = [
            task for task in tasks if str(task.get("id", "")).startswith("material-exhaustion-breakout-")
        ]
        assert len(breakouts_after) == 1
        assert breakouts_after[0]["status"] == "blocked"
        assert "autoresearch-material-exhaustion-terminal" in helper.RESULTS.read_text(encoding="utf-8")
        review_helper = Path(tmp) / "review-helper.py"
        review_marker = Path(tmp) / "review-marker.txt"
        review_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            f"pathlib.Path({str(review_marker)!r}).write_text(' '.join(sys.argv[1:]), encoding='utf-8')\n",
            encoding="utf-8",
        )
        review_helper.chmod(0o700)
        review_args = Namespace(
            research_helper_bin=str(review_helper),
            review_recent_rows=120,
            review_min_sweeps=3,
            review_min_samples_per_block=3,
            review_target_tps=30.0,
            hypothesis_rank_limit=12,
            gepa_min_blocked=3,
            gepa_min_rework=2,
            gepa_min_trajectory=2,
            gepa_min_low_quality=2,
            quality_review_timeout_seconds=5,
        )
        ok, issue = helper.run_supervisor_quality_review(review_args, 8, "nightly", Path(tmp) / "autopilot.log")
        assert ok
        assert issue == ""
        review_log = (Path(tmp) / "autopilot.log").read_text(encoding="utf-8")
        assert "quality-review --recent-rows 120 --min-sweeps 3 --min-samples-per-block 3 --target-tps 30.0" in review_log
        assert "environment-snapshot --label review-cycle-8 --allow-fail" in review_log
        assert "frontier-review --recent-rows 120 --min-samples 3" in review_log
        assert "hypothesis-rank --limit 12" in review_log
        assert "causal-review --recent-rows 120" in review_log
        assert "plateau-pivot --recent-rows 120 --min-sweeps 3 --target-tps 30.0" in review_log
        assert "evaluator-integrity" in review_log
        assert "implementation-handoff-audit --min-score 90" in review_log
        assert review_log.count("$ " + str(review_helper) + " implementation-handoff-audit --min-score 90") == 1
        assert review_log.count("SKIP duplicate review command: " + str(review_helper) + " implementation-handoff-audit --min-score 90") == 2
        assert "frontier-eval --recent-rows 120 --allow-fail" in review_log
        assert "gepa-policy-promote --min-candidates 3" in review_log
        assert "gepa-escalation --recent-rows 120 --min-blocked 3 --min-rework 2 --min-trajectory 2 --min-low-quality 2" in review_log
        review_timeout_helper = Path(tmp) / "review-timeout-helper.py"
        review_timeout_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import time\n"
            "time.sleep(2)\n",
            encoding="utf-8",
        )
        review_timeout_helper.chmod(0o700)
        review_timeout_args = Namespace(
            **{
                **vars(review_args),
                "research_helper_bin": str(review_timeout_helper),
                "quality_review_timeout_seconds": 0.05,
                "quality_review_total_timeout_seconds": 1.0,
            }
        )
        ok, issue = helper.run_supervisor_quality_review(
            review_timeout_args,
            10,
            "nightly",
            Path(tmp) / "autopilot.log",
        )
        assert not ok
        assert issue == "supervisor quality/frontier review timeout: environment-snapshot"
        assert "SUPERVISOR QUALITY REVIEW TIMEOUT command=environment-snapshot" in (
            Path(tmp) / "autopilot.log"
        ).read_text(encoding="utf-8")
        refill_args = Namespace(**{**vars(review_args), "external_blocker_refill_before_stop": True})
        with patch.object(
            helper,
            "external_change_required_status",
            side_effect=[
                {"should_stop": True, "ready_tasks": [], "reason": "external change required"},
                {
                    "should_stop": False,
                    "ready_tasks": ["refilled-task"],
                    "reason": "ready deterministic/prerequisite task exists",
                },
            ],
        ):
            with patch.object(helper, "run_supervisor_quality_review", return_value=(True, "")) as review_mock:
                with patch.object(helper, "run_supervisor_synthesis", side_effect=AssertionError("synthesis not needed")):
                    should_stop, refill_status = helper.maybe_stop_for_external_change(
                        refill_args,
                        13,
                        "nightly",
                        Path(tmp) / "autopilot.log",
                    )
        assert not should_stop
        assert refill_status["ready_tasks"] == ["refilled-task"]
        assert review_mock.call_count == 1
        refocus_args = Namespace(**{**vars(review_args), "external_blocker_refill_before_stop": True})
        with patch.object(
            helper,
            "external_change_required_status",
            side_effect=[
                {"should_stop": True, "ready_tasks": [], "reason": "external change required"},
                {"should_stop": True, "ready_tasks": [], "reason": "external change required"},
                {"should_stop": True, "ready_tasks": [], "reason": "external change required"},
            ],
        ):
            with patch.object(helper, "run_supervisor_quality_review", return_value=(True, "")):
                with patch.object(helper, "run_supervisor_synthesis", return_value=(True, "")):
                    should_stop, refocus_status = helper.maybe_stop_for_external_change(
                        refocus_args,
                        14,
                        "nightly",
                        Path(tmp) / "autopilot.log",
                    )
        assert not should_stop
        assert refocus_status["seeded_tasks"] == 3
        assert "autoresearch-external-refocus" in helper.RESULTS.read_text(encoding="utf-8")
        assert "decode-repeatability-cycle-014" in helper.TASKS.read_text(encoding="utf-8")
        stop_args = Namespace(
            **{
                **vars(review_args),
                "external_blocker_refill_before_stop": False,
                "external_blocker_action": "stop",
            }
        )
        with patch.object(
            helper,
            "external_change_required_status",
            return_value={"should_stop": True, "ready_tasks": [], "reason": "external change required"},
        ):
            should_stop, stop_status = helper.maybe_stop_for_external_change(
                stop_args,
                15,
                "nightly",
                Path(tmp) / "autopilot.log",
            )
        assert should_stop
        assert stop_status["reason"] == "external change required"
        assert "autoresearch-external-change-required" in helper.RESULTS.read_text(encoding="utf-8")
        review_fail_helper = Path(tmp) / "review-fail-helper.py"
        review_fail_log = Path(tmp) / "review-fail-marker.txt"
        review_fail_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            f"path = pathlib.Path({str(review_fail_log)!r})\n"
            "path.write_text(path.read_text(encoding='utf-8') + ' '.join(sys.argv[1:]) + '\\n' if path.exists() else ' '.join(sys.argv[1:]) + '\\n', encoding='utf-8')\n"
            "raise SystemExit(2 if len(sys.argv) > 1 and sys.argv[1] == 'implementation-handoff-audit' else 0)\n",
            encoding="utf-8",
        )
        review_fail_helper.chmod(0o700)
        review_fail_args = Namespace(**{**vars(review_args), "research_helper_bin": str(review_fail_helper)})
        ok, issue = helper.run_supervisor_quality_review(
            review_fail_args,
            12,
            "nightly",
            Path(tmp) / "autopilot.log",
        )
        assert ok
        assert issue == "implementation-handoff-audit exit 2"
        review_fail_text = review_fail_log.read_text(encoding="utf-8")
        assert "implementation-handoff-audit --min-score 90" in review_fail_text
        assert "frontier-eval --recent-rows 120 --allow-fail" in review_fail_text
        self_improve_marker = Path(tmp) / "self-improve-marker.txt"
        self_improve_helper = Path(tmp) / "self-improve-helper.py"
        self_improve_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            f"pathlib.Path({str(self_improve_marker)!r}).write_text(' '.join(sys.argv[1:]), encoding='utf-8')\n",
            encoding="utf-8",
        )
        self_improve_helper.chmod(0o700)
        self_improve_args = Namespace(
            research_helper_bin=str(self_improve_helper),
            self_improvement=True,
            self_evolution=False,
            self_improvement_recent_rows=160,
            self_improvement_timeout_seconds=5,
            self_evolution_max_variants_per_skill=2,
            self_evolution_min_score=90,
        )
        ok, issue = helper.run_supervisor_self_improvement(
            self_improve_args,
            8,
            "nightly",
            Path(tmp) / "autopilot.log",
            reason="unit",
        )
        assert ok
        assert issue == ""
        assert self_improve_marker.read_text(encoding="utf-8") == "self-improve --action curate --recent-rows 160"
        self_improve_marker.unlink()
        evolution_args = Namespace(**{**vars(self_improve_args), "self_evolution": True})
        ok, issue = helper.run_supervisor_self_improvement(
            evolution_args,
            9,
            "nightly",
            Path(tmp) / "autopilot.log",
            reason="unit-evolve",
        )
        assert ok
        assert issue == ""
        assert (
            self_improve_marker.read_text(encoding="utf-8")
            == "self-improve --action evolve --recent-rows 160 --max-variants-per-skill 2 --min-score 90"
        )
        disabled_args = Namespace(**{**vars(self_improve_args), "self_improvement": False})
        ok, issue = helper.run_supervisor_self_improvement(
            disabled_args,
            10,
            "nightly",
            Path(tmp) / "autopilot.log",
            reason="disabled",
        )
        assert ok
        assert issue == "disabled"
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "causal-review-unit",
                    "status": "ready",
                    "lane": "causal-repair",
                    "next_action": "read exactly promotion-decisions.jsonl",
                }
            ],
        )
        assert helper.block_stale_model_bound_causal_tasks() == 1
        assert '"status": "blocked"' in helper.TASKS.read_text(encoding="utf-8")
        ok, issue = helper.run_supervisor_reflection(
            synth_args,
            9,
            "nightly",
            Path(tmp) / "autopilot.log",
            reason="unit-test",
        )
        assert ok
        assert issue == ""
        assert "supervisor-reflection" in helper.FINDINGS.read_text(encoding="utf-8")
        stalled_impl = {
            "id": "stalled-impl",
            "status": "ready",
            "task_type": "implementation",
            "target": "openclaw/example.py",
        }
        helper.write_jsonl(helper.TASKS, [stalled_impl])
        ok, issue = helper.run_supervisor_reflection(
            synth_args,
            10,
            "nightly",
            Path(tmp) / "autopilot.log",
            reason="ready-implementation",
        )
        assert ok
        assert issue == "synthesis deferred because ready implementation tasks exist"
        assert "stalled-impl" in helper.FINDINGS.read_text(encoding="utf-8")
        ok, issue = helper.run_deterministic_fallback(
            synth_args,
            11,
            "nightly",
            stalled_impl,
            Path(tmp) / "autopilot.log",
            reason="malformed hidden/tool output",
        )
        assert ok
        assert issue == ""
        assert "supervisor-implementation-guard-11" in helper.RESULTS.read_text(encoding="utf-8")
        assert '"status": "blocked"' in helper.TASKS.read_text(encoding="utf-8")
        tasks = [
            {
                "id": "impl",
                "status": "ready",
                "task_type": "implementation",
                "target": "openclaw/example.py",
            }
        ]
        helper.write_jsonl(helper.TASKS, tasks)
        impl_summary = helper.complete_implementation_task(
            tasks[0],
            ["repo patch", "findings update"],
            commit="abc123",
        )
        assert impl_summary is not None
        assert impl_summary["task_id"] == "impl"
        assert '"status": "done"' in helper.TASKS.read_text(encoding="utf-8")
        assert "implementation-recorded" in helper.EXPERIMENTS.read_text(encoding="utf-8")
        assert "impl" in (helper.WORKSPACE / "promotion-decisions.jsonl").read_text(encoding="utf-8")
        collect_helper = Path(tmp) / "trace-collect-helper.py"
        collect_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "print(json.dumps({'status': 'keep', 'reason': 'target-generated-trace-data-present'}))\n",
            encoding="utf-8",
        )
        collect_helper.chmod(0o700)
        collect_task = {
            "id": "trace-collect",
            "status": "ready",
            "task_type": "supervisor",
            "supervisor_action": "drafter-trace-collect",
            "next_action": "openclaw-speed-research drafter-trace-collect",
        }
        assert helper.is_supervisor_drafter_trace_collect_task(collect_task)
        with patch.object(helper, "ensure_model_for_supervisor_task", return_value=(True, "")):
            code, issue = helper.run_supervisor_drafter_trace_collect_task(
                Namespace(research_helper_bin=str(collect_helper), model_start_timeout_seconds=1.0),
                13,
                "nightly",
                collect_task,
                Path(tmp) / "autopilot.log",
            )
        assert code == 0
        assert issue == ""
        canary_helper = Path(tmp) / "calibration-canary-helper.py"
        canary_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'status': 'keep', 'decision': 'ready-for-bounded-calibration'}))\n",
            encoding="utf-8",
        )
        canary_helper.chmod(0o700)
        canary_task = {
            "id": "calibration-canary",
            "status": "ready",
            "task_type": "supervisor",
            "supervisor_action": "drafter-calibration-canary",
            "next_action": "openclaw-speed-research drafter-calibration-canary",
        }
        assert helper.is_supervisor_drafter_calibration_canary_task(canary_task)
        assert helper.task_runs_without_model(canary_task)
        helper.write_jsonl(
            helper.TASKS,
            [
                canary_task,
                {
                    "id": "calibration-stage-ready",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-calibration-memory-stage",
                    "stage": "metadata",
                    "next_action": "openclaw-speed-research drafter-calibration-memory-stage --stage metadata",
                },
            ],
        )
        code, issue = helper.run_supervisor_drafter_calibration_canary_task(
            Namespace(research_helper_bin=str(canary_helper)),
            14,
            "nightly",
            canary_task,
            Path(tmp) / "autopilot.log",
        )
        assert code == 0
        assert issue == ""
        queue_after_canary = helper.read_jsonl(helper.TASKS)
        assert any(task["id"] == "calibration-stage-ready" and task["status"] == "ready" for task in queue_after_canary)
        assert all(
            task["status"] == "done"
            for task in queue_after_canary
            if helper.is_supervisor_drafter_calibration_canary_task(task)
        )
        calibration_stage_helper = Path(tmp) / "calibration-stage-helper.py"
        calibration_stage_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'status': 'keep', 'decision': 'advance', 'stage': 'metadata'}))\n",
            encoding="utf-8",
        )
        calibration_stage_helper.chmod(0o700)
        calibration_stage_task = {
            "id": "calibration-stage",
            "status": "ready",
            "task_type": "supervisor",
            "supervisor_action": "drafter-calibration-memory-stage",
            "stage": "metadata",
            "bounded_command": [str(calibration_stage_helper)],
            "next_action": "openclaw-speed-research drafter-calibration-memory-stage --stage metadata",
        }
        assert helper.is_supervisor_drafter_calibration_memory_stage_task(calibration_stage_task)
        assert helper.task_runs_without_model(calibration_stage_task)
        with patch.object(helper, "model_ready", return_value=False), patch.object(helper, "wait_for_memory", return_value=(True, "")):
            code, issue = helper.run_supervisor_drafter_calibration_memory_stage_task(
                Namespace(),
                141,
                "nightly",
                calibration_stage_task,
                Path(tmp) / "autopilot.log",
            )
        assert code == 0
        assert issue == ""
        calibration_run_helper = Path(tmp) / "calibration-run-helper.py"
        calibration_run_helper.write_text(
            "#!/usr/bin/env python3\n"
            "print('bounded calibration complete')\n",
            encoding="utf-8",
        )
        calibration_run_helper.chmod(0o700)
        calibration_run_task = {
            "id": "calibration-run",
            "status": "ready",
            "task_type": "supervisor",
            "supervisor_action": "drafter-calibration-run",
            "bounded_command": [str(calibration_run_helper)],
            "next_action": str(calibration_run_helper),
        }
        assert helper.is_supervisor_drafter_calibration_run_task(calibration_run_task)
        assert helper.task_runs_without_model(calibration_run_task)
        with patch.object(helper, "model_ready", return_value=False), patch.object(helper, "wait_for_memory", return_value=(True, "")):
            code, issue = helper.run_supervisor_drafter_calibration_run_task(
                Namespace(),
                15,
                "nightly",
                calibration_run_task,
                Path(tmp) / "autopilot.log",
        )
        assert code == 0
        assert issue == ""
        missing_module_helper = Path(tmp) / "calibration-run-missing-module.py"
        missing_module_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "print(\"ModuleNotFoundError: No module named 'mlx_vlm.speculative'\")\n"
            "sys.exit(2)\n",
            encoding="utf-8",
        )
        missing_module_helper.chmod(0o700)
        missing_module_task = {
            "id": "calibration-run-missing-module",
            "status": "ready",
            "task_type": "supervisor",
            "supervisor_action": "drafter-calibration-run",
            "bounded_command": [str(missing_module_helper)],
            "next_action": str(missing_module_helper),
        }
        with patch.object(helper, "model_ready", return_value=False), patch.object(helper, "wait_for_memory", return_value=(True, "")):
            code, issue = helper.run_supervisor_drafter_calibration_run_task(
                Namespace(),
                16,
                "nightly",
                missing_module_task,
                Path(tmp) / "autopilot.log",
            )
        assert code == 0
        assert issue == "missing-runtime-module:mlx_vlm.speculative"
        assert "drafter-calibration-runtime" in (helper.WORKSPACE / "exhausted-approaches.jsonl").read_text(
            encoding="utf-8"
        )
        memory_gate_helper = Path(tmp) / "calibration-run-memory-gate.py"
        memory_gate_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "print('calibration memory gate blocked: after-load: free=795MB<16384MB')\n"
            "sys.exit(2)\n",
            encoding="utf-8",
        )
        memory_gate_helper.chmod(0o700)
        memory_gate_task = {
            "id": "calibration-run-memory-gate",
            "status": "ready",
            "task_type": "supervisor",
            "supervisor_action": "drafter-calibration-run",
            "bounded_command": [str(memory_gate_helper)],
            "next_action": str(memory_gate_helper),
        }
        with patch.object(helper, "model_ready", return_value=False), patch.object(helper, "wait_for_memory", return_value=(True, "")):
            code, issue = helper.run_supervisor_drafter_calibration_run_task(
                Namespace(),
                17,
                "nightly",
                memory_gate_task,
                Path(tmp) / "autopilot.log",
            )
        assert code == 0
        assert issue == "calibration-memory-gate:after-load"
        exhausted = (helper.WORKSPACE / "exhausted-approaches.jsonl").read_text(encoding="utf-8")
        assert "drafter-calibration-memory" in exhausted
        bench_helper = Path(tmp) / "benchmark-helper.py"
        bench_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            f"pathlib.Path({str(helper.RESULTS)!r}).write_text("
            f"{helper.RESULTS_HEADER!r} + "
            "'2026-05-05T00:00:00+0000\\tbenchmark-test\\tkeep\\tdecode-sample\\th\\t\\t\\t16.0\\t6.0\\t1.5\\tabc123\\tn\\n', "
            "encoding='utf-8')\n"
            "print(json.dumps({'ok': True, 'mode': 'decode-sample', 'decode_tps': 16.0}))\n",
            encoding="utf-8",
        )
        bench_helper.chmod(0o700)
        original_model_ready = helper.model_ready
        helper.model_ready = lambda: True
        try:
            bench_args = Namespace(
                research_helper_bin=str(bench_helper),
                supervisor_benchmark_timeout_seconds=5,
                model_start_timeout_seconds=5,
            )
            code, issue = helper.run_supervisor_benchmark_task(
                bench_args,
                9,
                "bench-session",
                {"id": "bench-task", "benchmark_mode": "decode-sample"},
                Path(tmp) / "autopilot.log",
            )
        finally:
            helper.model_ready = original_model_ready
        assert code == 0
        assert issue == ""
        assert "benchmark-test" in helper.RESULTS.read_text(encoding="utf-8")
        assert "supervisor-benchmark" in helper.EXPERIMENTS.read_text(encoding="utf-8")
        refocus_workspace = Path(tmp) / "actual-external-refocus"
        configure_workspace(helper, refocus_workspace)
        helper.WORKSPACE.mkdir(parents=True, exist_ok=True)
        helper.BENCHMARKS.mkdir(parents=True, exist_ok=True)
        helper.RESULTS.write_text(helper.RESULTS_HEADER, encoding="utf-8")
        helper.write_jsonl(helper.TASKS, [])
        helper.write_jsonl(
            helper.WORKSPACE / "exhausted-approaches.jsonl",
            [
                {"lane": "drafter-calibration-memory", "reason": "unit exhausted"},
                {"lane": "frontier-dflash", "reason": "unit exhausted"},
            ],
        )
        canonical = {
            "state": "prerequisite_needed",
            "clean": True,
            "next": "run deterministic prerequisite",
            "decode_mean_tps": 13.4,
            "noise": {"terminal_synthesis_rows": 3, "routed_terminal_synthesis_rows": 0},
        }
        (helper.BENCHMARKS / "quality-review-999.json").write_text(
            json.dumps(
                {
                    "quality_score": 100,
                    "scorecard": {"overall": 99.7},
                    "canonical_state": canonical,
                }
            ),
            encoding="utf-8",
        )
        (helper.BENCHMARKS / "frontier-system-eval-999.json").write_text(
            json.dumps({"overall": 9.7, "canonical_state": canonical}),
            encoding="utf-8",
        )
        (helper.BENCHMARKS / "implementation-handoff-audit-999.json").write_text(
            json.dumps({"ok": True, "score": 100}),
            encoding="utf-8",
        )
        noop_helper = Path(tmp) / "noop-research-helper.py"
        noop_helper.write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n", encoding="utf-8")
        noop_helper.chmod(0o700)
        actual_refocus_args = Namespace(
            research_helper_bin=str(noop_helper),
            review_recent_rows=120,
            review_min_sweeps=3,
            review_min_samples_per_block=3,
            review_target_tps=30.0,
            hypothesis_rank_limit=12,
            gepa_min_blocked=3,
            gepa_min_rework=2,
            gepa_min_trajectory=2,
            gepa_min_low_quality=2,
            quality_review_timeout_seconds=5,
            synthesis_timeout_seconds=5,
            stop_on_external_blocker=True,
            external_blocker_refill_before_stop=True,
            external_blocker_action="refocus",
            external_blocker_min_quality=90.0,
            external_blocker_min_exhausted_core_lanes=2,
            external_blocker_min_terminal_cycles=3,
        )
        actual_status = helper.external_change_required_status(actual_refocus_args)
        assert actual_status["should_stop"]
        should_stop, refocused = helper.maybe_stop_for_external_change(
            actual_refocus_args,
            21,
            "actual-refocus",
            helper.WORKSPACE / "autopilot.log",
        )
        assert not should_stop
        assert refocused["seeded_tasks"] == 3
        actual_tasks = helper.TASKS.read_text(encoding="utf-8")
        assert "decode-repeatability-cycle-021" in actual_tasks
        assert "mtp-acceptance-review-cycle-021" in actual_tasks
        assert "implementation-bridge-cycle-021" in actual_tasks
        assert "autoresearch-external-refocus" in helper.RESULTS.read_text(encoding="utf-8")
        assert "autoresearch-external-change-required" not in helper.RESULTS.read_text(encoding="utf-8")
        low_signal_workspace = Path(tmp) / "low-signal-loop"
        configure_workspace(helper, low_signal_workspace)
        helper.WORKSPACE.mkdir(parents=True, exist_ok=True)
        helper.BENCHMARKS.mkdir(parents=True, exist_ok=True)
        helper.RESULTS.write_text(helper.RESULTS_HEADER, encoding="utf-8")
        for index in range(4):
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"synthesis-low-signal-{index}",
                status="keep",
                target="synthesis",
                hypothesis="seeded another MTP report",
                commit="abc123",
                notes="seeded_tasks=1 deliberation_actions=agent-deliberation-mtp-acceptance-yield",
            )
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"mtp-report-low-signal-{index}",
                status="keep",
                target="mtp-acceptance-report",
                hypothesis="recent OpenClaw server logs should expose drafter acceptance evidence",
                commit="abc123",
                notes="samples=5 mtp_samples=3 mean_server_tok_s=3.5 mean_accept=0.87 path=/tmp/report.json",
            )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "agent-deliberation-mtp-acceptance-yield-loop",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "mtp-report",
                    "lane": "frontier-deliberation",
                    "next_action": "openclaw-speed-research mtp-report --lines 320",
                }
            ],
        )
        low_signal_args = Namespace(
            low_signal_window_rows=20,
            low_signal_min_mtp_reports=4,
            low_signal_min_synthesis_rows=4,
        )
        low_signal_status = helper.recent_low_signal_mtp_loop_status(low_signal_args)
        assert low_signal_status["loop"]
        repair = helper.repair_low_signal_mtp_loop(22, "low-signal", low_signal_status)
        assert repair["blocked_tasks"] == 1
        assert repair["seeded_tasks"] == 3
        low_signal_tasks = helper.read_jsonl(helper.TASKS)
        assert any(task["status"] == "blocked" for task in low_signal_tasks if "mtp-acceptance-yield-loop" in task["id"])
        assert any(task.get("supervisor_action") == "source-scout" for task in low_signal_tasks)
        assert any(task.get("supervisor_action") == "runtime-overhead-map" for task in low_signal_tasks)
        deliberation_tasks = [task for task in low_signal_tasks if task.get("supervisor_action") == "frontier-deliberation"]
        assert len(deliberation_tasks) == 1
        assert helper.task_runs_without_model(deliberation_tasks[0])
        deliberation_helper = Path(tmp) / "frontier-deliberation-helper.py"
        deliberation_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'ok': True, 'seeded': 1, 'seeded_tasks': ['next-safe-task']}))\n",
            encoding="utf-8",
        )
        deliberation_helper.chmod(0o700)
        code, issue = helper.run_supervisor_frontier_deliberation_task(
            Namespace(research_helper_bin=str(deliberation_helper)),
            23,
            "low-signal",
            deliberation_tasks[0],
            helper.WORKSPACE / "autopilot.log",
        )
        assert code == 0
        assert issue == ""
        assert '"status": "done"' in helper.TASKS.read_text(encoding="utf-8")

        post_escape_workspace = Path(tmp) / "low-signal-post-escape-loop"
        configure_workspace(helper, post_escape_workspace)
        helper.WORKSPACE.mkdir(parents=True, exist_ok=True)
        helper.BENCHMARKS.mkdir(parents=True, exist_ok=True)
        helper.RESULTS.write_text(helper.RESULTS_HEADER, encoding="utf-8")
        helper.append_result(
            helper.WORKSPACE,
            run_id="runtime-overhead-map-earlier-escape",
            status="keep",
            target="runtime-overhead-map",
            hypothesis="bounded runtime escape was attempted before the loop restarted",
            commit="abc123",
            notes="gap=tooling overhead map complete",
        )
        for index in range(3):
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"synthesis-post-escape-{index}",
                status="keep",
                target="synthesis",
                hypothesis="seeded another post-escape MTP report",
                commit="abc123",
                notes="seeded_tasks=1 deliberation_actions=agent-deliberation-mtp-acceptance-yield",
            )
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"mtp-report-post-escape-{index}",
                status="keep",
                target="mtp-acceptance-report",
                hypothesis="recent OpenClaw server logs should expose drafter acceptance evidence",
                commit="abc123",
                notes="samples=5 mtp_samples=3 mean_server_tok_s=3.5 mean_accept=0.87 path=/tmp/report.json",
            )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "agent-deliberation-mtp-acceptance-yield-post-escape",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "mtp-report",
                    "lane": "frontier-deliberation",
                    "next_action": "openclaw-speed-research mtp-report --lines 320",
                }
            ],
        )
        post_escape_args = Namespace(
            low_signal_window_rows=20,
            low_signal_min_mtp_reports=3,
            low_signal_min_synthesis_rows=3,
        )
        post_escape_status = helper.recent_low_signal_mtp_loop_status(post_escape_args)
        assert post_escape_status["loop"], post_escape_status
        assert post_escape_status["escape_rows"] == 1, post_escape_status
        assert post_escape_status["rows_since_latest_escape"] == 6, post_escape_status
        post_escape_repair = helper.repair_low_signal_mtp_loop(24, "post-escape-low-signal", post_escape_status)
        assert post_escape_repair["blocked_tasks"] == 1
        assert post_escape_repair["seeded_tasks"] == 3

        pending_escape_workspace = Path(tmp) / "low-signal-pending-escape"
        configure_workspace(helper, pending_escape_workspace)
        helper.WORKSPACE.mkdir(parents=True, exist_ok=True)
        helper.BENCHMARKS.mkdir(parents=True, exist_ok=True)
        helper.RESULTS.write_text(helper.RESULTS_HEADER, encoding="utf-8")
        for index in range(3):
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"synthesis-pending-escape-{index}",
                status="keep",
                target="synthesis",
                hypothesis="seeded another MTP report before pending escape work",
                commit="abc123",
                notes="seeded_tasks=1 deliberation_actions=agent-deliberation-mtp-acceptance-yield",
            )
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"mtp-report-pending-escape-{index}",
                status="keep",
                target="mtp-acceptance-report",
                hypothesis="recent OpenClaw server logs should expose drafter acceptance evidence",
                commit="abc123",
                notes="samples=5 mtp_samples=3 mean_server_tok_s=3.5 mean_accept=0.87 path=/tmp/report.json",
            )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "low-signal-source-scout-cycle-024",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "source-scout",
                    "lane": "frontier-evidence",
                    "next_action": "openclaw-speed-research source-scout --topic rapid-mlx",
                }
            ],
        )
        pending_escape_status = helper.recent_low_signal_mtp_loop_status(post_escape_args)
        assert not pending_escape_status["loop"], pending_escape_status
        assert pending_escape_status["ready_escape_tasks"] == ["low-signal-source-scout-cycle-024"], pending_escape_status

        remeasure_workspace = Path(tmp) / "low-signal-decode-remeasure"
        configure_workspace(helper, remeasure_workspace)
        helper.WORKSPACE.mkdir(parents=True, exist_ok=True)
        helper.BENCHMARKS.mkdir(parents=True, exist_ok=True)
        helper.RESULTS.write_text(helper.RESULTS_HEADER, encoding="utf-8")
        for index in range(3):
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"synthesis-remeasure-{index}",
                status="keep",
                target="synthesis",
                hypothesis="seeded another decode remeasure",
                commit="abc123",
                notes="ideas=5 seeded_tasks=1 contract_actions=lane-contract-decode-remeasure-ready-work-gap-123",
            )
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"benchmark-remeasure-{index}",
                status="keep",
                target="decode-sample",
                hypothesis="bounded OpenClaw decode-sample probe",
                commit="abc123",
                decode_tps="15.1",
                wall_s="6.3",
                notes="completion_tokens=96 measurement_quality=clean server_tok_s=15.2",
            )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "lane-contract-decode-remeasure-ready-work-gap-loop",
                    "status": "ready",
                    "task_type": "supervisor",
                    "lane": "runtime-overhead",
                    "benchmark_mode": "decode-sample",
                    "next_action": "openclaw-speed-research benchmark --mode decode-sample",
                }
            ],
        )
        remeasure_args = Namespace(
            low_signal_window_rows=20,
            low_signal_min_mtp_reports=4,
            low_signal_min_synthesis_rows=4,
            low_signal_min_decode_remeasures=3,
            autonomy_trigger_recent_rows=20,
            autonomy_trigger_min_score=95,
            autonomy_trigger_hard_score=80,
        )
        remeasure_status = helper.recent_low_signal_decode_remeasure_status(remeasure_args)
        assert remeasure_status["loop"], remeasure_status
        helper.append_result(
            helper.WORKSPACE,
            run_id="implementation-handoff-should-not-mask-remeasure-loop",
            status="keep",
            target="autoresearch-implementation-handoff",
            hypothesis="handoff audit is a review artifact, not an escape from decode remeasure churn",
            commit="abc123",
            notes="ok=True score=100 ready_deterministic=1",
        )
        masked_remeasure_status = helper.recent_low_signal_decode_remeasure_status(remeasure_args)
        assert masked_remeasure_status["loop"], masked_remeasure_status
        trigger_status = helper.autonomy_trigger_status(
            remeasure_args,
            cycle=24,
            stalled_cycles=0,
            blocked_cycles=0,
            progress_cycles=6,
            last_issue="",
        )
        assert trigger_status["action"] in {"review", "repair"}, trigger_status
        assert "low-signal-decode-remeasure-loop" in trigger_status["triggers"], trigger_status
        remeasure_repair = helper.repair_low_signal_decode_remeasure_loop(24, "low-signal-remeasure", remeasure_status)
        assert remeasure_repair["blocked_tasks"] == 1
        assert remeasure_repair["seeded_tasks"] == 3
        remeasure_tasks = helper.read_jsonl(helper.TASKS)
        assert any(
            task["status"] == "blocked"
            for task in remeasure_tasks
            if task["id"] == "lane-contract-decode-remeasure-ready-work-gap-loop"
        )
        assert any(task.get("supervisor_action") == "frontier-deliberation" for task in remeasure_tasks)
        assert "autoresearch-low-signal-repair" in helper.RESULTS.read_text(encoding="utf-8")
        post_repair_trigger = helper.autonomy_trigger_status(
            remeasure_args,
            cycle=25,
            stalled_cycles=0,
            blocked_cycles=0,
            progress_cycles=7,
            last_issue="",
        )
        assert "low-signal-decode-remeasure-loop" not in post_repair_trigger["triggers"], post_repair_trigger

        trigger_workspace = Path(tmp) / "autonomy-trigger-cadence"
        configure_workspace(helper, trigger_workspace)
        helper.WORKSPACE.mkdir(parents=True, exist_ok=True)
        helper.BENCHMARKS.mkdir(parents=True, exist_ok=True)
        helper.RESULTS.write_text(helper.RESULTS_HEADER, encoding="utf-8")
        helper.append_result(
            helper.WORKSPACE,
            run_id="frontier-review-memory-lane",
            status="keep",
            target="autoresearch-frontier",
            hypothesis="frontier review mentions drafter-calibration-memory without a crash",
            commit="abc123",
            notes="ready=2 exhausted=drafter-calibration-memory,frontier-dflash",
        )
        benign_args = Namespace(
            low_signal_window_rows=20,
            low_signal_min_mtp_reports=4,
            low_signal_min_synthesis_rows=4,
            low_signal_min_decode_remeasures=3,
            autonomy_trigger_recent_rows=20,
            autonomy_trigger_min_score=95,
            autonomy_trigger_hard_score=80,
            autonomy_trigger_controller=True,
            autonomy_trigger_interval_cycles=1,
            autonomy_trigger_review_interval_seconds=1800,
        )
        benign_anomalies = helper.recent_trigger_anomalies(benign_args)
        assert benign_anomalies["hard_rows"] == 0, benign_anomalies

        trigger_log = helper.WORKSPACE / "autopilot.log"
        calls = []
        original_watchdog = helper.run_periodic_autonomy_watchdog

        def fake_watchdog(*_args, **_kwargs):
            calls.append((_args, _kwargs))
            return True, ""

        helper.run_periodic_autonomy_watchdog = fake_watchdog
        try:
            ok, issue, handled = helper.run_autonomy_trigger_controller(
                benign_args,
                30,
                "trigger-cadence",
                trigger_log,
                stalled_cycles=0,
                blocked_cycles=0,
                progress_cycles=1,
                last_issue="",
                last_trigger_at=time.monotonic(),
            )
            assert (ok, issue, handled) == (True, "", False)
            assert calls == []
            ok, issue, handled = helper.run_autonomy_trigger_controller(
                benign_args,
                31,
                "trigger-cadence",
                trigger_log,
                stalled_cycles=0,
                blocked_cycles=0,
                progress_cycles=1,
                last_issue="",
                last_trigger_at=time.monotonic() - 1801,
            )
            assert (ok, issue, handled) == (True, "", True)
            assert len(calls) == 1
        finally:
            helper.run_periodic_autonomy_watchdog = original_watchdog

        watchdog_workspace = Path(tmp) / "low-signal-watchdog"
        configure_workspace(helper, watchdog_workspace)
        helper.WORKSPACE.mkdir(parents=True, exist_ok=True)
        helper.BENCHMARKS.mkdir(parents=True, exist_ok=True)
        helper.RESULTS.write_text(helper.RESULTS_HEADER, encoding="utf-8")
        for index in range(4):
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"synthesis-watchdog-{index}",
                status="keep",
                target="synthesis",
                hypothesis="seeded another MTP report",
                commit="abc123",
                notes="seeded_tasks=1 deliberation_actions=agent-deliberation-mtp-acceptance-yield",
            )
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"mtp-report-watchdog-{index}",
                status="keep",
                target="mtp-acceptance-report",
                hypothesis="recent OpenClaw server logs should expose drafter acceptance evidence",
                commit="abc123",
                notes="samples=5 mtp_samples=3 mean_server_tok_s=3.5 mean_accept=0.87 path=/tmp/report.json",
            )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "agent-deliberation-mtp-acceptance-yield-watchdog",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "mtp-report",
                    "lane": "frontier-deliberation",
                    "next_action": "openclaw-speed-research mtp-report --lines 320",
                }
            ],
        )
        watchdog_args = Namespace(
            low_signal_window_rows=20,
            low_signal_min_mtp_reports=4,
            low_signal_min_synthesis_rows=4,
        )
        with (
            patch.object(helper, "run_supervisor_quality_review", return_value=(True, "")) as quality_mock,
            patch.object(helper, "run_supervisor_self_improvement", return_value=(True, "")) as self_mock,
            patch.object(helper, "frontier_certification_status", return_value={"ok": True}),
            patch.object(helper, "run_autonomous_repair_loop", return_value=(False, "should-not-run")) as repair_mock,
        ):
            ok, issue = helper.run_periodic_autonomy_watchdog(
                watchdog_args,
                31,
                "watchdog-success",
                helper.WORKSPACE / "autopilot.log",
                reason="unit low signal",
            )
        assert ok
        assert issue == ""
        assert quality_mock.call_count == 2
        assert self_mock.call_count == 1
        assert repair_mock.call_count == 0
        assert "autoresearch-autonomous-repair" in helper.RESULTS.read_text(encoding="utf-8")

        watchdog_failure_workspace = Path(tmp) / "low-signal-watchdog-failure"
        configure_workspace(helper, watchdog_failure_workspace)
        helper.WORKSPACE.mkdir(parents=True, exist_ok=True)
        helper.BENCHMARKS.mkdir(parents=True, exist_ok=True)
        helper.RESULTS.write_text(helper.RESULTS_HEADER, encoding="utf-8")
        for index in range(4):
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"synthesis-watchdog-failure-{index}",
                status="keep",
                target="synthesis",
                hypothesis="seeded another MTP report",
                commit="abc123",
                notes="seeded_tasks=1 deliberation_actions=agent-deliberation-mtp-acceptance-yield",
            )
            helper.append_result(
                helper.WORKSPACE,
                run_id=f"mtp-report-watchdog-failure-{index}",
                status="keep",
                target="mtp-acceptance-report",
                hypothesis="recent OpenClaw server logs should expose drafter acceptance evidence",
                commit="abc123",
                notes="samples=5 mtp_samples=3 mean_server_tok_s=3.5 mean_accept=0.87 path=/tmp/report.json",
            )
        helper.write_jsonl(
            helper.TASKS,
            [
                {
                    "id": "agent-deliberation-mtp-acceptance-yield-watchdog-failure",
                    "status": "ready",
                    "task_type": "supervisor",
                    "supervisor_action": "mtp-report",
                    "lane": "frontier-deliberation",
                    "next_action": "openclaw-speed-research mtp-report --lines 320",
                }
            ],
        )
        with (
            patch.object(helper, "run_supervisor_quality_review", return_value=(False, "review failed")),
            patch.object(helper, "run_supervisor_self_improvement", return_value=(False, "self-improvement failed")),
            patch.object(helper, "frontier_certification_status", return_value={"ok": False, "issues": ["not certified"]}),
            patch.object(helper, "run_autonomous_repair_loop", return_value=(True, "")) as repair_mock,
        ):
            ok, issue = helper.run_periodic_autonomy_watchdog(
                watchdog_args,
                32,
                "watchdog-failure",
                helper.WORKSPACE / "autopilot.log",
                reason="unit low signal",
            )
        assert ok
        assert issue == ""
        assert repair_mock.call_count == 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
