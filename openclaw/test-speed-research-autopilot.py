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
    assert "benchmark --mode decode-sample" in prompt
    assert "MTP acceptance" in prompt
    assert "openclaw-model-proxy.log" in prompt
    assert "Do not run setup commands" in prompt
    assert "implementation-skill.md" in prompt
    assert "Do not repeat quick-health, TTFT, prompt-size, or tool-roundtrip benchmarks" in prompt
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

    assert helper.summarize_issue("x OpenClaw blocked a broad local tool command y", "", 0) == (
        "OpenClaw blocked a broad local tool command"
    )
    assert helper.summarize_issue("memory circuit breaker stopped backend", "", 0) == "memory circuit breaker"
    assert helper.summarize_issue("", "fatal process exit via SIGABRT", 1) == "fatal process exit"
    assert helper.summarize_issue("", "", 124) == "turn timeout"
    assert helper.summarize_issue("", "", 7) == "agent exit 7"
    assert helper.summarize_issue("all good", "", 0) == ""
    assert helper.summarize_issue("TOOL RESULT CAP after 2 tool results", "", 0) == "TOOL RESULT CAP"
    assert helper.summarize_issue("Warming up Metal shaders", "", 0) == ""
    assert helper.summarize_issue("Metal out of memory while compiling", "", 0) == "metal out of memory"
    assert helper.early_failure_reason("EMBEDDED FALLBACK: Gateway agent failed") == "gateway embedded fallback"
    assert helper.early_failure_reason("rawError=Connection error.") == "model connection error"
    assert helper.early_failure_reason("normal bounded result") == ""
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
        helper.WORKSPACE = Path(tmp)
        helper.RESULTS = helper.WORKSPACE / "results.tsv"
        helper.IDEAS = helper.WORKSPACE / "ideas.md"
        helper.TASKS = helper.WORKSPACE / "tasks.jsonl"
        helper.BENCHMARKS = helper.WORKSPACE / "benchmarks"
        helper.STRATEGY = helper.WORKSPACE / "STRATEGY.md"
        helper.FINDINGS = helper.WORKSPACE / "findings.jsonl"
        helper.EXPERIMENTS = helper.WORKSPACE / "experiments.jsonl"
        helper.REJECTIONS = helper.WORKSPACE / "rejections.jsonl"
        helper.OPENCLAW_HOME = Path(tmp) / "home"
        helper.ensure_task_queue()
        task_text = helper.TASKS.read_text(encoding="utf-8")
        assert "decode-mtp-baseline" in task_text
        assert "decode-sample" in helper.next_task_summary()
        seeded = helper.enqueue_recurring_decode_tasks(42, "test empty queue")
        assert seeded == 3
        recurring_tasks = helper.TASKS.read_text(encoding="utf-8")
        assert "decode-repeatability-cycle-042" in recurring_tasks
        assert "mtp-acceptance-review-cycle-042" in recurring_tasks
        assert "implementation-bridge-cycle-042" in recurring_tasks
        assert "autopilot-refill" in helper.RESULTS.read_text(encoding="utf-8")
        selected = helper.select_next_task(helper.WORKSPACE)
        assert selected["id"] == "decode-mtp-baseline"
        with helper.RESULTS.open("a", encoding="utf-8") as file:
            for index, decode_tps in enumerate(("14.100", "14.600"), start=1):
                file.write(
                    f"2026-05-05T00:00:0{index}+0000\tbenchmark-{index}\tkeep\tdecode-sample\t"
                    f"bounded OpenClaw decode-sample probe\t\t\t{decode_tps}\t6.9\t1.5\tabc123\tmodel=local\n"
                )
        assert helper.complete_task_from_evidence(helper.WORKSPACE, selected, min_samples=3, commit="abc123") is None
        with helper.RESULTS.open("a", encoding="utf-8") as file:
            file.write(
                "2026-05-05T00:00:03+0000\tbenchmark-3\tkeep\tdecode-sample\t"
                "bounded OpenClaw decode-sample probe\t\t\t15.000\t6.6\t1.5\tabc123\tmodel=local\n"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
