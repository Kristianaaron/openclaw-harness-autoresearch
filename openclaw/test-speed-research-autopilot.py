#!/usr/bin/env python3
"""Checks for the autonomous OpenClaw speed research runner."""

from __future__ import annotations

import importlib.util
import os
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
    assert helper.task_runs_without_model(
        {"task_type": "supervisor", "supervisor_action": "runtime-overhead-map"}
    )
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


def main() -> int:
    helper = load_helper()
    check_prompt_and_routing_guards(helper)
    check_memory_and_failure_guards(helper)
    with tempfile.TemporaryDirectory() as tmp:
        configure_workspace(helper, Path(tmp))
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
        malformed_before = dict(quick_after)
        malformed_before["results_lines"] = 2
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
        sweep_args = Namespace(research_helper_bin=str(sweep_helper))
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
        assert "frontier-review --recent-rows 120 --min-samples 3" in review_log
        assert "hypothesis-rank --limit 12" in review_log
        assert "causal-review --recent-rows 120" in review_log
        assert "gepa-policy-promote --min-candidates 3" in review_log
        assert "gepa-escalation --recent-rows 120 --min-blocked 3 --min-rework 2 --min-trajectory 2 --min-low-quality 2" in review_log
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
