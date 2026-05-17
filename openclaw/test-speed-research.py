#!/usr/bin/env python3
"""Checks for the OpenClaw speed autoresearch helper."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


HELPER_PATH = Path(__file__).with_name("openclaw-speed-research.py")
CALIBRATOR_PATH = Path(__file__).with_name("openclaw-mtp-drafter-calibrate.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("openclaw_speed_research", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_calibrator():
    spec = importlib.util.spec_from_file_location("openclaw_mtp_drafter_calibrate", CALIBRATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {CALIBRATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    helper = load_helper()
    calibrator = load_calibrator()
    gate_args = Namespace(min_free_mb=16384, min_pressure_free_percent=20, max_compressor_mb=2048, max_swap_mb=2048)
    with patch.object(
        calibrator,
        "memory_snapshot",
        return_value={"free_mb": 795, "compressor_mb": 1200, "swap_used_mb": 1600, "pressure_free_percent": 71},
    ):
        assert calibrator.memory_block_reason(gate_args, phase="after-load") == ""
    with patch.object(
        calibrator,
        "memory_snapshot",
        return_value={"free_mb": 795, "compressor_mb": 1200, "swap_used_mb": 1600, "pressure_free_percent": 5},
    ):
        assert "pressureFree=5%<20%" in calibrator.memory_block_reason(gate_args, phase="after-load")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "research" / "speed"
        with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_DIR": str(root)}, clear=False):
            assert helper.setup_workspace(Namespace(repo_url="file:///no/such/repo")) == 0
            assert (root / "program.md").exists()
            assert (root / "README-openclaw-speed.md").exists()
            assert (root / "implementation-skill.md").exists()
            assert (root / "benchmark-manifest.json").exists()
            runtime_root = Path(tmp) / "runtime-clean-research" / "speed"
            helper.ensure_research_state(runtime_root)
            helper.append_result(
                runtime_root,
                run_id="runtime-overhead-map-unit-a",
                status="keep",
                target="runtime-overhead-map",
                hypothesis="unit",
                commit="unit",
                notes="contaminated=0",
            )
            helper.append_result(
                runtime_root,
                run_id="runtime-overhead-map-unit-b",
                status="keep",
                target="runtime-overhead-map",
                hypothesis="unit",
                commit="unit",
                notes="contaminated=0",
            )
            stale_runtime_task = {
                "id": "unit-stale-runtime-overhead",
                "status": "ready",
                "lane": "runtime-overhead",
                "task_type": "supervisor",
                "supervisor_action": "runtime-overhead-map",
                "target": "openclaw/openclaw-jang-vlm-server.py",
                "hypothesis": "repeated clean runtime maps should retire runtime-overhead churn",
                "metric": "server_wall_decode_gap",
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
            }
            helper.write_jsonl(runtime_root / "tasks.jsonl", [stale_runtime_task])
            assert helper.runtime_overhead_repeated_clean(runtime_root, helper.result_rows(runtime_root)) is True
            assert helper.filter_seedable_tasks(runtime_root, [stale_runtime_task]) == []
            assert helper.compact_repeated_runtime_overhead_tasks(runtime_root, helper.result_rows(runtime_root)) >= 1
            assert not [
                task
                for task in helper.read_jsonl(runtime_root / "tasks.jsonl")
                if task.get("id") == "unit-stale-runtime-overhead" and task.get("status", "ready") in {"ready", "rework"}
            ]
            duplicate_task = {
                "id": "unit-duplicate-task",
                "status": "ready",
                "lane": "implementation-gate",
                "task_type": "supervisor",
                "supervisor_action": "focused-test",
                "target": "openclaw/test-speed-research.py",
                "hypothesis": "duplicate active task ids should be compacted before they create false progress",
                "metric": "test_pass",
                "next_action": "python3 openclaw/test-speed-research.py",
            }
            assert helper.upsert_tasks(root, [duplicate_task, {**duplicate_task, "priority": 99}]) == 1
            assert helper.upsert_tasks(root, [{**duplicate_task, "priority": 98}]) == 0
            active_duplicates = [
                task
                for task in helper.read_jsonl(root / "tasks.jsonl")
                if task.get("id") == "unit-duplicate-task" and task.get("status", "ready") in {"ready", "rework"}
            ]
            assert len(active_duplicates) == 1
            semantic_duplicate = {
                "id": "unit-duplicate-task-new-name",
                "status": "ready",
                "lane": "implementation-gate",
                "task_type": "supervisor",
                "supervisor_action": "focused-test",
                "target": "openclaw/test-speed-research.py",
                "hypothesis": "same operation under a new id should not enter the ready queue",
                "metric": "test_pass",
                "next_action": "python3 openclaw/test-speed-research.py",
            }
            assert helper.semantic_task_key(semantic_duplicate) == helper.semantic_task_key(duplicate_task)
            assert helper.upsert_tasks(root, [semantic_duplicate]) == 0
            assert not [
                task
                for task in helper.read_jsonl(root / "tasks.jsonl")
                if task.get("id") == "unit-duplicate-task-new-name" and task.get("status", "ready") in {"ready", "rework"}
            ]
            memory_root = Path(tmp) / "operational-memory-research" / "speed"
            helper.ensure_research_state(memory_root)
            helper.append_result(
                memory_root,
                run_id="benchmark-decode-baseline",
                status="keep",
                target="decode-sample",
                hypothesis="baseline",
                commit="unit",
                decode_tps=14.0,
                wall_s=6.8,
                notes="server_tok_s=14.0 measurement_quality=clean",
            )
            for index in range(2):
                helper.append_result(
                    memory_root,
                    run_id=f"mtp-report-loop-{index}",
                    status="keep",
                    target="mtp-acceptance-report",
                    hypothesis="acceptance report",
                    commit="unit",
                    notes="mean_accept=0.80 mean_server_tok_s=7.0",
                )
            mtp_task = {
                "id": "agent-deliberation-mtp-acceptance-yield-unit",
                "status": "ready",
                "priority": 90,
                "lane": "production-mtp",
                "task_type": "supervisor",
                "supervisor_action": "mtp-report",
                "target": "openclaw-model-proxy.log",
                "hypothesis": "MTP acceptance report should wait for fresh decode after repeated reports",
                "metric": "mean_accept",
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research mtp-report --lines 240",
            }
            memory = helper.operational_strategy_memory(memory_root)
            assert (
                memory["semantic_tasks"]["production-mtp:mtp-report:mean_accept"]["state"]
                == "waiting_for_prerequisite"
            )
            assert helper.filter_seedable_tasks(memory_root, [mtp_task]) == []
            helper.append_result(
                memory_root,
                run_id="benchmark-decode-refresh",
                status="keep",
                target="decode-sample",
                hypothesis="fresh decode unlocks acceptance report",
                commit="unit",
                decode_tps=14.2,
                wall_s=6.7,
                notes="server_tok_s=14.2 measurement_quality=clean",
            )
            assert helper.filter_seedable_tasks(memory_root, [mtp_task]) == [mtp_task]
            loop_root = Path(tmp) / "adapter-loop-research" / "speed"
            helper.ensure_research_state(loop_root)
            for index in range(2):
                helper.append_result(
                    loop_root,
                    run_id=f"calibration-memory-report-loop-{index}",
                    status="blocked",
                    target="calibration-memory-report",
                    hypothesis="Calibration plateau should become a no-model root-cause report.",
                    commit="unit",
                    notes=f"blocker={helper.CALIBRATION_QUANTIZED_GRADIENT_BLOCKER}",
                )
                helper.append_result(
                    loop_root,
                    run_id=f"drafter-adapter-method-contract-loop-{index}",
                    status="keep",
                    target="janq-drafter-adapter-method",
                    hypothesis="convert repeated quantized-gradient failures into adapter/logit implementation",
                    commit="unit",
                    notes="state=adapter_calibration_memory_blocked ok=False terminal_routed=True",
                )
                helper.append_result(
                    loop_root,
                    run_id=f"supervisor-focused-test-{index}",
                    status="keep",
                    target="openclaw/openclaw-mtp-drafter-calibrate.py",
                    hypothesis="add a canary-only adapter/logit-distillation calibration path",
                    commit="unit",
                    notes="focused test passed",
                )
            adapter_task = helper.drafter_adapter_method_contract_task(
                123,
                task_id="drafter-adapter-method-contract-current",
            )
            memory_report_task = {
                "id": "calibration-memory-report-current",
                "status": "ready",
                "lane": "drafter-alignment",
                "task_type": "supervisor",
                "supervisor_action": "calibration-memory-report",
                "target": "calibration-memory-report",
                "hypothesis": "repeat the same blocker report",
                "metric": "calibration_memory_root_cause",
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research calibration-memory-report",
            }
            escape_task = helper.mtp_acceptance_yield_task(123, evidence={"unit": True})
            assert helper.adapter_logit_loop_saturated(loop_root)
            assert helper.drafter_bottleneck_state(loop_root)["state"] == "adapter_logit_loop_exhausted"
            assert helper.filter_seedable_tasks(loop_root, [adapter_task, memory_report_task, escape_task]) == []
            immediate_escape = helper.drafter_bottleneck_next_tasks(
                loop_root,
                helper.result_rows(loop_root),
                123,
                reason="unit immediate adapter-loop escape",
            )
            assert len(immediate_escape) == 1
            assert immediate_escape[0]["id"].startswith("agent-deliberation-quant-safe-drafter-candidate-")
            for index in range(2):
                helper.append_result(
                    loop_root,
                    run_id=f"source-scout-loop-{index}",
                    status="keep",
                    target="frontier-source-scout",
                    hypothesis="frontier scout loop",
                    commit="unit",
                    notes="topic=JANQ Gemma4 drafter fit fetched=9 attempted=10",
                )
                helper.append_result(
                    loop_root,
                    run_id=f"mtp-report-source-loop-{index}",
                    status="keep",
                    target="mtp-acceptance-report",
                    hypothesis="frontier MTP loop",
                    commit="unit",
                    notes="samples=4 mtp_samples=2 mean_accept=0.80",
                )
            assert helper.repeated_frontier_escape_evidence(loop_root)["ready"] is True
            source_task = helper.source_scout_task(124, evidence={"unit": True})
            mtp_task = helper.mtp_acceptance_yield_task(124, evidence={"unit": True})
            assert helper.filter_seedable_tasks(loop_root, [source_task, escape_task]) == []
            assert helper.filter_seedable_tasks(loop_root, [mtp_task]) == []
            candidate_tasks = helper.frontier_escape_candidate_tasks(
                loop_root,
                helper.result_rows(loop_root),
                125,
            )
            assert len(candidate_tasks) == 1
            assert candidate_tasks[0]["id"].startswith("agent-deliberation-quant-safe-drafter-candidate-")
            assert candidate_tasks[0]["metric"] == "quant_safe_drafter_candidate_gate"
            assert candidate_tasks[0]["crabbox_required"] is True
            deliberation_report, deliberation_candidate = helper.frontier_agent_deliberation(
                loop_root,
                helper.result_rows(loop_root),
                126,
            )
            assert deliberation_candidate
            assert deliberation_candidate[0]["id"].startswith("agent-deliberation-quant-safe-drafter-candidate-")
            assert "quantization-safe drafter candidate" in deliberation_report["architect"]["selected_reason"]
            direct_escape = helper.drafter_bottleneck_next_tasks(
                loop_root,
                helper.result_rows(loop_root),
                127,
                reason="unit direct escape",
            )
            assert len(direct_escape) == 1
            assert direct_escape[0]["id"].startswith("agent-deliberation-quant-safe-drafter-candidate-")
            assert "drafter-trace-prerequisite" in direct_escape[0]["next_action"]
            helper.upsert_tasks(loop_root, direct_escape)
            active_candidate_state = helper.drafter_bottleneck_state(loop_root)
            assert active_candidate_state["state"] == "quant_safe_candidate_ready"
            assert active_candidate_state["next_step"] == "run_quant_safe_candidate_gate"

            terminal_source_root = Path(tmp) / "terminal-source-scout-suppression"
            helper.ensure_research_state(terminal_source_root)
            helper.append_result(
                terminal_source_root,
                run_id="decode-before-terminal-block",
                status="keep",
                target="decode-sample",
                hypothesis="unit decode",
                commit="abc123",
                notes="mode=decode-sample server_decode_tps=14.0 draft_block_size=2 contaminated=0",
            )
            helper.append_result(
                terminal_source_root,
                run_id="drafter-calibration-memory-stage-micro-step-unit",
                status="blocked",
                target="openclaw/openclaw-mtp-drafter-calibrate.py",
                hypothesis="unit terminal blocker",
                commit="abc123",
                notes="stage=micro-step reason=calibration-quantized-gradient-unsupported",
            )
            terminal_source = helper.source_scout_task(128, evidence={"unit": True})
            terminal_mtp = helper.mtp_acceptance_yield_task(128, evidence={"unit": True})
            assert helper.filter_seedable_tasks(terminal_source_root, [terminal_source, terminal_mtp]) == []
            helper.write_jsonl(terminal_source_root / "tasks.jsonl", [terminal_source, terminal_mtp])
            assert helper.block_operational_strategy_ready_tasks(terminal_source_root) >= 2
            terminal_tasks = helper.read_jsonl(terminal_source_root / "tasks.jsonl")
            assert all(
                task.get("status") == "blocked"
                for task in terminal_tasks
                if str(task.get("id", "")).startswith(
                    ("agent-deliberation-source-scout-", "agent-deliberation-mtp-acceptance-yield-")
                )
            )
            assert not [
                task
                for task in terminal_tasks
                if task.get("status") in {"ready", "rework"}
                and str(task.get("id", "")).startswith(
                    ("agent-deliberation-source-scout-", "agent-deliberation-mtp-acceptance-yield-")
                )
            ]
            adapter_memory_root = Path(tmp) / "adapter-memory-source-scout-suppression"
            helper.ensure_research_state(adapter_memory_root)
            helper.append_result(
                adapter_memory_root,
                run_id="decode-before-adapter-memory-block",
                status="keep",
                target="decode-sample",
                hypothesis="unit decode",
                commit="abc123",
                notes="mode=decode-sample server_decode_tps=14.0 draft_block_size=2 contaminated=0",
            )
            helper.append_result(
                adapter_memory_root,
                run_id="calibration-memory-report-unit",
                status="keep",
                target="calibration-memory-report",
                hypothesis="unit adapter memory blocker",
                commit="abc123",
                notes="state=adapter_calibration_memory_blocked blocker=calibration-memory-after-load",
            )
            adapter_memory_source = helper.source_scout_task(129, evidence={"unit": True})
            adapter_memory_mtp = helper.mtp_acceptance_yield_task(129, evidence={"unit": True})
            assert helper.filter_seedable_tasks(adapter_memory_root, [adapter_memory_source, adapter_memory_mtp]) == []
            helper.write_jsonl(adapter_memory_root / "tasks.jsonl", [adapter_memory_source, adapter_memory_mtp])
            assert helper.block_operational_strategy_ready_tasks(adapter_memory_root) >= 2
            assert all(
                task.get("status") == "blocked"
                for task in helper.read_jsonl(adapter_memory_root / "tasks.jsonl")
                if str(task.get("id", "")).startswith(
                    ("agent-deliberation-source-scout-", "agent-deliberation-mtp-acceptance-yield-")
                )
            )
            calibration_report_root = Path(tmp) / "calibration-report-none-source-suppression"
            helper.ensure_research_state(calibration_report_root)
            helper.append_result(
                calibration_report_root,
                run_id="decode-before-calibration-report-none",
                status="keep",
                target="decode-sample",
                hypothesis="unit decode",
                commit="abc123",
                decode_tps=15.6,
                notes="mode=decode-sample server_decode_tps=15.6 draft_block_size=2 contaminated=0",
            )
            helper.mark_lane_exhausted(
                calibration_report_root,
                lane="drafter-calibration-gradient",
                reason="calibration-quantized-gradient-unsupported",
                evidence={"unit": True},
            )
            helper.mark_lane_exhausted(
                calibration_report_root,
                lane="drafter-calibration-memory",
                reason="calibration-memory-gate:after-load",
                evidence={"unit": True},
            )
            helper.append_result(
                calibration_report_root,
                run_id="calibration-memory-report-none-unit",
                status="keep",
                target="calibration-memory-report",
                hypothesis="Calibration plateau should become a no-model root-cause report.",
                commit="abc123",
                notes="blocker=none hit_count=177 plateau=false",
            )
            report_none_source = helper.source_scout_task(130, evidence={"unit": True})
            report_none_mtp = helper.mtp_acceptance_yield_task(130, evidence={"unit": True})
            memory = helper.operational_strategy_memory(calibration_report_root)
            assert (
                memory["semantic_tasks"]["frontier-deliberation:source-scout:source_evidence_count"][
                    "required_evidence"
                ]
                == "drafter_candidate_or_calibration_canary"
            )
            assert helper.filter_seedable_tasks(calibration_report_root, [report_none_source, report_none_mtp]) == []
            helper.write_jsonl(calibration_report_root / "tasks.jsonl", [report_none_source, report_none_mtp])
            assert helper.block_operational_strategy_ready_tasks(calibration_report_root) >= 2
            assert all(
                task.get("status") == "blocked"
                for task in helper.read_jsonl(calibration_report_root / "tasks.jsonl")
                if str(task.get("id", "")).startswith(
                    ("agent-deliberation-source-scout-", "agent-deliberation-mtp-acceptance-yield-")
                )
            )
            progress_memory_root = Path(tmp) / "progress-memory-source-suppression"
            helper.ensure_research_state(progress_memory_root)
            (progress_memory_root / "progress-memory.json").write_text(
                json.dumps(
                    {
                        "current_bottleneck": {"next_step": "seed_frontier_deliberation_escape"},
                        "not_progress": ["Repeating source-scout or MTP reports after calibration-memory reports."],
                    }
                ),
                encoding="utf-8",
            )
            progress_source = helper.source_scout_task(131, evidence={"unit": True})
            progress_mtp = helper.mtp_acceptance_yield_task(131, evidence={"unit": True})
            assert helper.task_operational_blocker(progress_memory_root, progress_source)
            assert helper.task_operational_blocker(progress_memory_root, progress_mtp)
            helper.write_jsonl(progress_memory_root / "tasks.jsonl", [progress_source, progress_mtp])
            assert helper.block_operational_strategy_ready_tasks(progress_memory_root) >= 2
            assert all(
                task.get("status") == "blocked"
                for task in helper.read_jsonl(progress_memory_root / "tasks.jsonl")
                if str(task.get("id", "")).startswith(
                    ("agent-deliberation-source-scout-", "agent-deliberation-mtp-acceptance-yield-")
                )
            )
            quant_safe_route_root = Path(tmp) / "quant-safe-route"
            quant_safe_home = Path(tmp) / "quant-safe-home"
            trace_dir = quant_safe_home / "drafter-fit"
            trace_dir.mkdir(parents=True)
            (trace_dir / "target-generated-traces.jsonl").write_text(
                '{"prompt":"hello","completion":"world"}\n',
                encoding="utf-8",
            )
            helper.ensure_research_state(quant_safe_route_root)
            helper.write_jsonl(
                quant_safe_route_root / "tasks.jsonl",
                [
                    helper.quant_safe_drafter_candidate_task(
                        132,
                        evidence={"unit": True},
                    )
                ],
            )
            with patch.dict(
                os.environ,
                {
                    "OPENCLAW_HOME": str(quant_safe_home),
                    "OPENCLAW_SPEED_RESEARCH_DIR": str(quant_safe_route_root),
                },
                clear=False,
            ):
                assert helper.should_seed_quant_safe_drafter_canary(quant_safe_route_root)
                assert helper.drafter_trace_prerequisite(Namespace()) == 0
                ready_after = [
                    task
                    for task in helper.read_jsonl(quant_safe_route_root / "tasks.jsonl")
                    if task.get("status") in {"ready", "rework"}
                ]
                assert any(
                    str(task.get("id", "")).startswith("adapter-drafter-calibration-canary-")
                    and task.get("calibration_mode") == helper.CALIBRATION_ADAPTER_MODE
                    for task in ready_after
                )
            assert helper.actionable_blocked_rows(
                [
                    {"status": "blocked", "target": "autoresearch-quality"},
                    {"status": "blocked", "target": "autoresearch-frontier-eval"},
                    {"status": "blocked", "target": "autoresearch-stability-burn-in"},
                    {"status": "blocked", "target": "autoresearch-sota-autonomy-eval"},
                    {
                        "status": "blocked",
                        "target": "janq-drafter-calibration-memory-stage",
                        "notes": "[QuantizedMatmul::vjp] no gradient wrt the quantized weights.",
                    },
                    {"status": "blocked", "target": "decode-sample"},
                ]
            ) == [{"status": "blocked", "target": "decode-sample"}]
            assert helper.is_memory_safety_blocked_row(
                {
                    "status": "blocked",
                    "target": "autoresearch-external-change-required",
                    "notes": "exhausted_lanes=[drafter-calibration-memory]",
                }
            ) is False
            assert helper.is_memory_safety_blocked_row(
                {"status": "blocked", "target": "decode-sample", "notes": "memory pressure"}
            ) is True
            assert (root / "insight-rubric.json").exists()
            assert (root / "research-profile.json").exists()
            assert (root / "evaluator-policy.json").exists()
            assert (root / "lane-contracts.json").exists()
            assert (root / "replay-buffer.jsonl").exists()
            manifest = json.loads((root / "benchmark-manifest.json").read_text(encoding="utf-8"))
            assert manifest["locked"] is True
            assert manifest["modes"]["decode-sample"]["max_tokens"] == 96
            assert manifest["modes"]["decode-sample"]["requires_usage_completion_tokens"] is True
            rubric = json.loads((root / "insight-rubric.json").read_text(encoding="utf-8"))
            assert "rollback" in rubric["required_fields"]
            profile = json.loads((root / "research-profile.json").read_text(encoding="utf-8"))
            assert profile["name"] == "openclaw-speed"
            assert "decode_tps" in profile["metrics"]["primary"]
            assert "policy-optimization" in profile["scope"]["allowed_lanes"]
            assert "frontier-expansion" in profile["scope"]["allowed_lanes"]
            assert "frontier_candidate_gate" in profile["metrics"]["secondary"]
            assert "trace_distillation_repair_gate" in profile["metrics"]["secondary"]
            assert "adapter_method_contract" in profile["metrics"]["secondary"]
            assert "autoresearch_quality_delta" in profile["metrics"]["secondary"]
            policy = json.loads((root / "evaluator-policy.json").read_text(encoding="utf-8"))
            assert "benchmark-manifest.json" in policy["immutable_paths"]
            assert "replay-buffer.jsonl" in policy["immutable_paths"]
            lane_contracts = json.loads((root / "lane-contracts.json").read_text(encoding="utf-8"))
            assert "drafter-alignment" in lane_contracts["lanes"]
            assert "frontier-expansion" in lane_contracts["lanes"]
            assert "calibration-memory-after-load" in lane_contracts["lanes"]["drafter-alignment"]["hard_blockers"]
            assert (
                "calibration-quantized-gradient-unsupported"
                in lane_contracts["lanes"]["drafter-alignment"]["hard_blockers"]
            )
            replay_cases = (root / "replay-buffer.jsonl").read_text(encoding="utf-8")
            assert "decode-token-source-required" in replay_cases
            assert "profile-variant-paired-control" in replay_cases
            assert helper.replay(Namespace(allow_fail=False)) == 0
            (root / "benchmarks").mkdir(parents=True, exist_ok=True)
            (root / "benchmarks" / "quality-review-999.json").write_text(
                json.dumps(
                    {
                        "kind": "quality-review",
                        "verdict": "healthy",
                        "quality_score": 100,
                        "scorecard": {
                            "overall": 98.0,
                            "interpretation": "high_quality_exhaustion_or_prerequisite_route",
                        },
                        "canonical_state": {
                            "clean": True,
                            "noise": {
                                "unresolved_blocked_rows": 0,
                                "terminal_synthesis_rows": 0,
                                "bridge_zero_rows": 0,
                                "memory_blocks": 0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "benchmarks" / "implementation-handoff-audit-999.json").write_text(
                json.dumps(
                    {
                        "kind": "implementation-handoff-audit",
                        "ok": True,
                        "score": 100,
                        "gaps": [],
                        "gates": {
                            "deterministic_ready_task": True,
                            "implementation_candidates_present": True,
                            "scoped_candidates_have_guards": True,
                            "ready_contracts_clean": True,
                            "patch_executor_contract_ready": True,
                            "safe_patch_tests_allowlisted": True,
                            "patch_executor_blocks_secret_content": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            helper.write_jsonl(
                root / "tasks.jsonl",
                [
                    helper.drafter_calibration_memory_stage_task(
                        999,
                        stage="target-load",
                        task_id="burn-in-memory-stage",
                        bounded_command=["openclaw-speed-research", "drafter-calibration-memory-stage", "--stage", "target-load"],
                    )
                ],
            )
            with patch.object(
                helper,
                "memory_snapshot",
                return_value={"free_mb": 16384, "compressor_mb": 1024, "swap_used_mb": 0},
            ):
                burn = helper.stability_burn_in_report(root)
            assert burn["ok"] is True
            assert all(burn["gates"].values())
            assert burn["gates"]["no_model_bound_implementation_ready"] is True
            assert burn["gates"]["ready_tasks_guarded"] is True
            (root / "benchmarks" / "stability-burn-in-999.json").write_text(
                json.dumps(burn),
                encoding="utf-8",
            )
            assert helper.environment_snapshot_command(
                Namespace(label="unit", repo="/Users/kristian/Documents/openclaw-harness-autoresearch", allow_fail=False)
            ) == 0
            assert (root / "snapshots").is_dir()
            assert "environment-snapshot" in (root / "results.tsv").read_text(encoding="utf-8")
            assert helper.evaluator_integrity_command(Namespace(allow_fail=False)) == 0
            assert helper.paired_plan(Namespace(task_id="no-drafter-control")) == 0
            paired_path = root / "experiments" / "paired-profile-plan-no-drafter-control.json"
            paired = json.loads(paired_path.read_text(encoding="utf-8"))
            assert paired["control"]["restore_before"] is True
            assert paired["promotion_gate"]["must_restore_live_profile"] is True
            assert helper.benchmark_prompt("decode-sample") == (
                "Write one compact paragraph about reducing local LLM decode latency. Keep it practical.",
                96,
            )
            parsed_log = helper.parse_generation_log_metrics(
                "[openclaw-jang-vlm-server] chat completion: prompt=10 completion=96 "
                "elapsed=6.00s tok_s=16.0 mtp_rounds=24 mean_accept=0.75\n"
            )
            assert parsed_log["server_tok_s"] == 16.0
            assert parsed_log["mtp_rounds"] == 24
            assert parsed_log["mean_accept"] == 0.75
            summary = helper.parse_generation_log_summary(
                "chat completion: prompt=10 completion=96 elapsed=6.00s tok_s=16.0 mtp_rounds=24 mean_accept=0.75\n"
                "stream chat completion: prompt=12 completion=96 elapsed=8.00s tok_s=12.0 mtp_rounds=48 mean_accept=0.25\n"
            )
            assert summary["sample_count"] == 2
            assert summary["mean_server_tok_s"] == 14.0
            assert summary["mean_accept"] == 0.5
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
            assert "normal `openclaw tui` decode speed" in prompt
            assert "Improve autoresearch itself only when it helps" in prompt
            assert "First assistant action" in prompt
            assert "RUN_MEMORY.md" in prompt
            assert "benchmark --mode decode-sample" in prompt
            assert "Use one narrow tool call" in prompt
            assert "SUMMARY.md" in prompt
            assert "results.tsv" not in prompt
            assert (root / "RUN_MEMORY.md").exists()
            assert (root / "PROGRESS.md").exists()
            assert (root / "restart-context.json").exists()
            assert (root / "progress-memory.json").exists()
            assert (root / "SUMMARY.md").exists()
            assert (root / "results-recent.tsv").exists()
            run_memory = (root / "RUN_MEMORY.md").read_text(encoding="utf-8")
            assert "OpenClaw Speed Research Run Memory" in run_memory
            assert "Progress Memory Lane" in run_memory
            assert "Live Speed vs Candidate Speed" in run_memory
            assert "Breakthrough Truth" in run_memory
            assert "Autonomy, Modularity, And Self-Improvement" in run_memory
            assert "What Has Been Worked On" in run_memory
            assert "Next Clear Tests" in run_memory
            assert "candidate needs paired live TUI decode benchmark before promotion" in run_memory
            assert "candidate_fixture_is_proof: False" in run_memory
            assert "not_progress:" in run_memory
            assert "creative_mode:" in run_memory
            assert "zero_active_noise_gate:" in run_memory
            assert "Do Not Re-discover" in run_memory
            restart_context = json.loads((root / "restart-context.json").read_text(encoding="utf-8"))
            assert restart_context["primary_metric"] == "normal OpenClaw TUI decode tok/s"
            assert "canonical_state" in restart_context
            progress_memory = json.loads((root / "progress-memory.json").read_text(encoding="utf-8"))
            progress_text = (root / "PROGRESS.md").read_text(encoding="utf-8")
            assert "OpenClaw Research Progress Memory" in progress_text
            assert "Current Owner" in progress_text
            assert "Not Progress" in progress_text
            assert "Required Next Evidence" in progress_text
            assert progress_memory["current_owner"]["next_action"]
            assert progress_memory["candidate_fixture_is_proof"] is False
            assert (
                "Repeating source-scout or MTP reports after calibration-memory reports."
                in progress_memory["not_progress"]
            )
            assert restart_context["speed_context"]["candidate_fixture_decode_tps"] == 45.5
            assert restart_context["speed_context"]["paired_live_candidate_evidence"] is False
            assert restart_context["breakthrough_context"]["achieved"] is False
            assert restart_context["breakthrough_context"]["candidate_fixture_is_proof"] is False
            assert restart_context["breakthrough_context"]["now"]
            assert restart_context["progress_memory"]["current_owner"]["next_action"]
            capability = restart_context["system_capability_context"]
            assert capability["creative_problem_solving"]["allowed"] is True
            assert capability["modularity"]["topic_portable"] is True
            assert "opencode" in capability["modularity"]["forbidden"]
            assert capability["bad_behavior_guards"]["zero_active_noise"] is True
            assert restart_context["ready_tasks"][0]["next_action"]
            assert restart_context["next_clear_tests"]
            summary = (root / "SUMMARY.md").read_text(encoding="utf-8")
            assert "run memory:" in summary
            assert "restart context:" in summary
            program = (root / "program.md").read_text(encoding="utf-8")
            assert "Do not touch opencode" in program
            assert "## Tool Discipline" in program
            assert "## Bootstrap Ladder" in program
            assert "## Narrow Tool Catalog" in program
            assert "find /Users" in program
            assert "## Starting Point" not in program
            assert "## Current Priority" in program
            assert "real OpenClaw TUI decode tokens/sec" in program
            assert "Autoresearch self-improvement is not the primary benchmark" in program
            assert "TUI-relevant decode benchmark result" in program
            assert "openclaw-mtp-drafter-calibrate.py" in program
            assert "MTP acceptance" in program
            assert "## Frontier Speed Track" in program
            assert "50-70 tok/s" in program
            assert "Lane B: drafter alignment" in program
            assert "DFlash compatibility" in program
            assert "## 30 Tok/S Investigation Ladder" in program
            assert "## Karpathy Compatibility Layer" in program
            assert "Immutable evaluator policy" in program
            assert "stop repeating that sweep" in program
            assert "z-lab/gemma-4-31B-it-DFlash" in program
            assert "## Realistic Experiment Backlog" in program
            assert "No-drafter control" in program
            assert "Drafter block sweep" in program
            assert "## Speed Targets" in program
            assert "Current live baseline" in program
            assert "Autoresearch quality metrics are tertiary" in program
            assert "## Implementation Gate" in program
            assert "implementation-skill.md" in program
            assert "## Dynamic Policy Optimization" in program
            assert "GEPA policy optimization is a supervisor reflex" in program
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
            assert (root / "journal.jsonl").exists()
            assert (root / "trajectory-corpus.jsonl").exists()
            assert (root / "gepa-candidates.jsonl").exists()
            assert (root / "hypothesis-rank.jsonl").exists()
            assert (root / "promotion-decisions.jsonl").exists()
            assert (root / "causal-reviews.jsonl").exists()
            assert (root / "exhausted-approaches.jsonl").exists()
            tasks = helper.read_jsonl(root / "tasks.jsonl")
            assert any(task.get("id") == "mtp-loop-overhead-map" for task in tasks)
            assert any(task.get("id") == "janq-dflash-drafter-fit-plan" and task.get("priority") == 86 for task in tasks)
            assert helper.task_contract_report(root)["ok"] is True
            helper.write_jsonl(
                root / "tasks.jsonl",
                [
                    {
                        "id": "bad-opencode-task",
                        "status": "ready",
                        "target": "opencode/config",
                        "hypothesis": "bad task should be blocked by research profile",
                        "metric": "decode_tps",
                        "next_action": "edit opencode/config",
                    }
                ],
            )
            bad_contract = helper.task_contract_report(root)
            assert bad_contract["ok"] is False
            assert "forbidden scope reference: opencode" in bad_contract["issues"][0]["blockers"]
            helper.write_jsonl(
                root / "tasks.jsonl",
                [
                    {
                        "id": "deliberate-dflash-compatibility-legacy",
                        "status": "ready",
                        "priority": 93,
                        "lane": "frontier-dflash",
                        "task_type": "research",
                        "target": "dflash.model_mlx/openclaw-jang-vlm-server.py",
                        "hypothesis": "legacy DFlash task",
                        "metric": "compatibility_decision_then_decode_tps",
                        "next_action": "read implementation-skill.md",
                    }
                ],
            )
            helper.ensure_research_state(root)
            migrated = helper.read_jsonl(root / "tasks.jsonl")[0]
            assert migrated["supervisor_action"] == "dflash-compatibility-gate"
            assert "openclaw-speed-research dflash-compatibility-gate" in migrated["next_action"]
            helper.write_jsonl(root / "tasks.jsonl", tasks)
            helper.write_jsonl(root / "tasks.jsonl", [{"id": "old-blocked", "status": "blocked"}])
            with (root / "results.tsv").open("a", encoding="utf-8") as file:
                for index in range(2):
                    file.write(
                        f"2026-05-05T00:09:0{index}+0000\tdrafter-sweep-run-seed-{index}\tkeep\t"
                        "OPENCLAW_JANG_DRAFT_BLOCK_SIZE\tpaired drafter block-size sweep\t\t\t14.2\t\t\tabc123\t"
                        "decision=keep-current control_block=2 winner_block=2 delta_vs_control=0.0\n"
                    )
            assert helper.quality_review(
                Namespace(recent_rows=20, min_sweeps=3, min_samples_per_block=3, target_tps=30.0)
            ) == 0
            seeded_review_tasks = (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert "review-drafter-sweep-next" in seeded_review_tasks
            stale_speed_root = Path(tmp) / "stale-speed-evidence" / "speed"
            helper.ensure_research_state(stale_speed_root)
            helper.mark_lane_exhausted(stale_speed_root, lane="mtp-decode", reason="unit settled", evidence={})
            helper.write_jsonl(stale_speed_root / "tasks.jsonl", [])
            for index in range(4):
                helper.append_result(
                    stale_speed_root,
                    run_id=f"mtp-report-stale-speed-{index}",
                    status="keep",
                    target="mtp-acceptance-report",
                    hypothesis="unit stale speed evidence",
                    commit="abc123",
                    notes="samples=9 mtp_samples=6 mean_server_tok_s=5.733 mean_accept=0.805 path=/tmp/report.json",
                )
            with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_DIR": str(stale_speed_root)}, clear=False):
                assert helper.quality_review(
                    Namespace(recent_rows=20, min_sweeps=3, min_samples_per_block=3, target_tps=30.0)
                ) == 0
            stale_review = json.loads(
                max(
                    (stale_speed_root / "benchmarks").glob("quality-review-*.json"),
                    key=lambda path: path.stat().st_mtime_ns,
                ).read_text(encoding="utf-8")
            )
            assert stale_review["stale_speed_evidence"] is True
            assert stale_review["gates"]["fresh_decode_metric"] is False
            stale_tasks = helper.read_jsonl(stale_speed_root / "tasks.jsonl")
            assert any(task.get("benchmark_mode") == "decode-sample" for task in stale_tasks)
            helper.write_jsonl(root / "tasks.jsonl", tasks)
            helper.write_jsonl(
                root / "tasks.jsonl",
                [
                    {
                        "id": "drafter-block-sweep-plan",
                        "status": "done",
                        "priority": 78,
                        "supervisor_action": "drafter-sweep-run",
                    }
                ],
            )
            helper.ensure_research_state(root)
            migrated_tasks = helper.read_jsonl(root / "tasks.jsonl")
            assert next(task for task in migrated_tasks if task.get("id") == "drafter-block-sweep-plan")[
                "status"
            ] == "done"
            assert "Pre-Implementation Gate" in implementation
            assert helper.benchmark(
                Namespace(base_url="http://127.0.0.1:1/v1", model="", quick=False, mode="prompt-size", timeout=1.0)
            ) == 0
            assert helper.benchmark(
                Namespace(base_url="http://127.0.0.1:1/v1", model="", quick=False, mode="prompt-shape", timeout=1.0)
            ) == 0
            assert helper.drafter_sweep_plan(Namespace(blocks="2,3,4", samples=2, min_delta=0.5)) == 0
            sweep_paths = list((root / "experiments").glob("mtp-drafter-sweep-plan-*.json"))
            assert sweep_paths
            sweep = json.loads(sweep_paths[-1].read_text(encoding="utf-8"))
            assert sweep["promotion_gate"]["must_not_change_live_profile"] is True
            assert "--draft-block-size 3" in json.dumps(sweep)
            assert helper.completion_tokens_from_response(
                {"usage": {"completion_tokens": 96}},
                "short visible text",
            ) == (96, "usage.completion_tokens")
            assert helper.benchmark_result_schema_ok(
                root,
                {
                    "ok": True,
                    "model": "local",
                    "mode": "decode-sample",
                    "wall_s": 0.5,
                    "memory_before_mb": {},
                    "memory_after_mb": {},
                    "timestamp": 1,
                    "completion_tokens": 1,
                    "completion_token_source": "usage.completion_tokens",
                    "decode_tps": 2.0,
                },
            )[0] is False
            fallback_tokens, fallback_source = helper.completion_tokens_from_response({}, "short visible text")
            assert fallback_tokens > 1
            assert fallback_source == "content_estimate"
            benchmark_rows = (root / "results.tsv").read_text(encoding="utf-8")
            assert "prompt-size" in benchmark_rows
            assert "prompt-shape" in benchmark_rows
            fake_log = root / "fake-proxy.log"
            fake_log.write_text("before\n", encoding="utf-8")

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def read(self):
                    return b'{"data":[{"id":"local-model"}]}'

            def fake_model_request(_base_url, _payload, _timeout):
                with fake_log.open("a", encoding="utf-8") as file:
                    file.write(
                        "chat completion: prompt=10 completion=96 elapsed=6.00s "
                        "tok_s=16.0 mtp_rounds=24 mean_accept=0.75\n"
                    )
                return (
                    6.0,
                    b'{"choices":[{"message":{"content":"done"}}],"usage":{"completion_tokens":96}}',
                )

            with patch.dict(os.environ, {"OPENCLAW_MODEL_PROXY_LOG": str(fake_log)}, clear=False):
                with patch.object(helper.urllib.request, "urlopen", return_value=FakeResponse()):
                    with patch.object(helper, "model_request", side_effect=fake_model_request):
                        assert helper.benchmark(
                            Namespace(
                                base_url="http://127.0.0.1:8091/v1",
                                model="",
                                quick=False,
                                mode="decode-sample",
                                timeout=1.0,
                                draft_block_size=3,
                            )
                        ) == 0
                        assert helper.drafter_sweep_run(
                            Namespace(
                                base_url="http://127.0.0.1:8091/v1",
                                model="",
                                blocks="1,2",
                                samples=1,
                                min_delta=0.5,
                                control_block=2,
                                timeout=1.0,
                                retries=2,
                            )
                        ) == 0
                    short_then_good_calls = {"count": 0}

                    def fake_short_then_good(_base_url, _payload, _timeout):
                        short_then_good_calls["count"] += 1
                        if short_then_good_calls["count"] == 1:
                            return (
                                0.1,
                                b'{"choices":[{"message":{"content":"x"}}],"usage":{"completion_tokens":1}}',
                            )
                        with fake_log.open("a", encoding="utf-8") as file:
                            file.write(
                                "chat completion: prompt=10 completion=96 elapsed=6.00s "
                                "tok_s=16.0 mtp_rounds=24 mean_accept=0.75\n"
                            )
                        return (
                            6.0,
                            b'{"choices":[{"message":{"content":"done"}}],"usage":{"completion_tokens":96}}',
                        )

                    with patch.object(helper, "model_request", side_effect=fake_short_then_good):
                        before_retry_rows = (root / "results.tsv").read_text(encoding="utf-8")
                        assert helper.drafter_sweep_run(
                            Namespace(
                                base_url="http://127.0.0.1:8091/v1",
                                model="",
                                blocks="2",
                                samples=1,
                                min_delta=0.5,
                                control_block=2,
                                timeout=1.0,
                                retries=2,
                            )
                        ) == 0
                        after_retry_rows = (root / "results.tsv").read_text(encoding="utf-8")
                        assert after_retry_rows.count("schema_issue=decode-sample too short") == before_retry_rows.count(
                            "schema_issue=decode-sample too short"
                        )
                    trace_home = root / "home"
                    trace_output = trace_home / "drafter-fit" / "target-generated-traces.jsonl"

                    def fake_trace_request(_base_url, payload, _timeout):
                        prompt = payload["messages"][0]["content"]
                        content = f"target completion for {prompt[:24]}"
                        return (
                            0.2,
                            json.dumps(
                                {
                                    "choices": [{"message": {"content": content}}],
                                    "usage": {"completion_tokens": 12},
                                }
                            ).encode("utf-8"),
                        )

                    with patch.dict(os.environ, {"OPENCLAW_HOME": str(trace_home)}, clear=False):
                        with patch.object(helper, "model_request", side_effect=fake_trace_request):
                            assert helper.drafter_trace_collect(
                                Namespace(
                                    base_url="http://127.0.0.1:8091/v1",
                                    model="",
                                    output=str(trace_output),
                                    samples=4,
                                    min_traces=4,
                                    max_tokens=48,
                                    timeout=1.0,
                                    min_free_mb=0,
                                    force=False,
                                )
                            ) == 0
                        trace_rows = [json.loads(line) for line in trace_output.read_text(encoding="utf-8").splitlines()]
                        assert len(trace_rows) == 4
                        assert trace_rows[0]["schema_version"] == 1
                        assert trace_rows[0]["completion_tokens"] == 12
                        assert "drafter-trace-collect" in (root / "results.tsv").read_text(encoding="utf-8")
                        assert helper.drafter_trace_prerequisite(Namespace()) == 0
                        prereq_report = json.loads(
                            sorted((root / "benchmarks").glob("drafter-trace-prerequisite-*.json"))[-1].read_text(
                                encoding="utf-8"
                            )
                        )
                        assert prereq_report["status"] == "keep"
                        assert prereq_report["trace_data"]
                        task_text = (root / "tasks.jsonl").read_text(encoding="utf-8")
                        assert "drafter-calibration-canary-" in task_text
                        assert "drafter-calibration-canary" in task_text
                        plan_path = trace_home / "drafter-fit" / "gemma4-janq-dflash-fit-plan.json"
                        plan_path.write_text(
                            json.dumps(
                                {
                                    "decision": "ready-for-target-generated-trace-data",
                                    "target_path": "/tmp/janq-target",
                                }
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        assert helper.drafter_calibration_canary(
                            Namespace(
                                plan=str(plan_path),
                                trace_data=str(trace_output),
                                output_dir=str(trace_home / "drafter-fit"),
                                min_traces=4,
                                max_prompts=4,
                                test_timeout=1.0,
                                skip_test=True,
                            )
                        ) == 0
                        canary_report = json.loads(
                            sorted((root / "benchmarks").glob("drafter-calibration-canary-*.json"))[-1].read_text(
                                encoding="utf-8"
                            )
                        )
                        assert canary_report["decision"] == "ready-for-bounded-calibration"
                        assert canary_report["trace_rows"] == 4
                        assert canary_report["runtime_import_ok"] is True
                        assert canary_report["bounded_calibration_command"][0] == helper.calibration_python()
                        assert Path(canary_report["prompts_file"]).exists()
                        task_text = (root / "tasks.jsonl").read_text(encoding="utf-8")
                        assert "drafter-calibration-memory-stage-metadata-" in task_text
                        assert "drafter-calibration-memory-stage" in task_text
                        fake_calibrator = trace_home / "fake-calibrator.py"
                        fake_calibrator.write_text(
                            "#!/usr/bin/env python3\n"
                            "import argparse, json, pathlib, time\n"
                            "p=argparse.ArgumentParser(); p.add_argument('--output-path', required=True); "
                            "p.add_argument('--probe-stage', required=True); p.add_argument('--target-path'); "
                            "p.add_argument('--drafter-path'); p.add_argument('--prompts-file'); "
                            "p.add_argument('--train-samples'); p.add_argument('--eval-samples'); "
                            "p.add_argument('--positions-per-prompt'); p.add_argument('--steps'); "
                            "p.add_argument('--eval-every'); p.add_argument('--min-free-mb'); "
                            "p.add_argument('--max-compressor-mb'); p.add_argument('--max-swap-mb'); "
                            "p.add_argument('--min-pressure-free-percent'); p.add_argument('--gpu-memory-utilization'); "
                            "p.add_argument('--mlx-cache-gb'); p.add_argument('--target-trace-policy'); a=p.parse_args(); "
                            "out=pathlib.Path(a.output_path); out.mkdir(parents=True, exist_ok=True); "
                            "payload={'ok': True, 'status': 'keep', 'stage': a.probe_stage, 'timestamp': int(time.time())}; "
                            "(out / f'openclaw-calibration-probe-{a.probe_stage}.json').write_text(json.dumps(payload)); "
                            "print(json.dumps(payload))\n",
                            encoding="utf-8",
                        )
                        fake_calibrator.chmod(0o700)
                        with patch.dict(os.environ, {"OPENCLAW_MTP_CALIBRATOR_SCRIPT": str(fake_calibrator)}, clear=False):
                            assert helper.drafter_calibration_memory_stage(
                                Namespace(
                                    stage="metadata",
                                    plan=str(plan_path),
                                    trace_data=str(trace_output),
                                    output_dir=str(trace_home / "drafter-fit"),
                                    min_traces=4,
                                    max_prompts=4,
                                )
                            ) == 0
                        stage_report = json.loads(
                            sorted((root / "benchmarks").glob("drafter-calibration-memory-stage-metadata-*.json"))[
                                -1
                            ].read_text(encoding="utf-8")
                        )
                        assert stage_report["decision"] == "advance"
                        task_text = (root / "tasks.jsonl").read_text(encoding="utf-8")
                        assert "drafter-calibration-memory-stage-drafter-load-" in task_text
                        fake_python = trace_home / "missing-speculative-python"
                        fake_python.write_text(
                            "#!/usr/bin/env python3\n"
                            "import sys\n"
                            "print(\"ModuleNotFoundError: No module named 'mlx_vlm.speculative'\")\n"
                            "sys.exit(1)\n",
                            encoding="utf-8",
                        )
                        fake_python.chmod(0o700)
                        with patch.dict(os.environ, {"OPENCLAW_CALIBRATION_PYTHON": str(fake_python)}, clear=False):
                            assert helper.calibration_python() == str(fake_python)
                            assert (
                                helper.calibration_runtime_import_issue(helper.calibration_python())
                                == "missing-runtime-module:mlx_vlm.speculative.drafters"
                            )
                        blocked_root = Path(tmp) / "calibration-block-root"
                        helper.ensure_research_state(blocked_root)
                        helper.append_result(
                            blocked_root,
                            run_id="supervisor-drafter-calibration-run-unit",
                            status="blocked",
                            target="openclaw/openclaw-mtp-drafter-calibrate.py",
                            hypothesis="unit calibration blocker",
                            commit="abc123",
                            notes="calibration memory gate blocked: after-load: free=795MB<16384MB",
                        )
                        assert helper.recent_calibration_run_hard_blocker(blocked_root) == "calibration-memory-after-load"
                        assert helper.should_seed_drafter_calibration_canary(blocked_root)
                        assert not helper.should_seed_drafter_calibration_run(blocked_root)
                        blocked_root_contracts = helper.ensure_lane_contracts(blocked_root)
                        assert "drafter-alignment" in blocked_root_contracts["lanes"]
                        filtered_calibration = helper.filter_seedable_tasks(
                            blocked_root,
                            [
                                helper.drafter_calibration_run_task(
                                    123456,
                                    task_id="drafter-calibration-run-blocked",
                                    bounded_command=["python3", "calibrate.py"],
                                )
                            ],
                        )
                        assert filtered_calibration == []
                        contract_fallback = helper.lane_contract_fallback_tasks(
                            blocked_root,
                            helper.result_rows(blocked_root),
                            123457,
                            reason="unit no ready task",
                        )
                        if contract_fallback:
                            assert len(contract_fallback) == 1
                            assert contract_fallback[0]["benchmark_mode"] == "decode-sample"
                            assert "calibration-memory-after-load" in contract_fallback[0]["hypothesis"]
                        else:
                            assert any(
                                task.get("status", "ready") in {"ready", "rework"}
                                and task.get("benchmark_mode") == "decode-sample"
                                for task in helper.read_jsonl(blocked_root / "tasks.jsonl")
                            )
                        adapter_block_root = Path(tmp) / "adapter-memory-block-root"
                        helper.ensure_research_state(adapter_block_root)
                        helper.write_jsonl(
                            adapter_block_root / "tasks.jsonl",
                            [
                                helper.drafter_calibration_memory_stage_task(
                                    123458,
                                    stage="micro-step",
                                    task_id="adapter-memory-blocked-stage",
                                    bounded_command=["python3", "calibrate.py"],
                                    calibration_mode_value=helper.CALIBRATION_ADAPTER_MODE,
                                )
                            ],
                        )
                        blocked_stage_tasks = helper.read_jsonl(adapter_block_root / "tasks.jsonl")
                        blocked_stage_tasks[0]["status"] = "blocked"
                        blocked_stage_tasks[0]["supervisor_summary"] = {
                            "reason": "calibration-memory-gate:after-load",
                            "memory_gate_issue": "calibration-memory-gate:after-load",
                            "output_tail": "calibration memory gate blocked: after-load: compressor=9000MB>=8192MB",
                        }
                        helper.write_jsonl(adapter_block_root / "tasks.jsonl", blocked_stage_tasks)
                        assert (
                            helper.latest_calibration_stage_issue(
                                adapter_block_root,
                                calibration_mode_filter=helper.CALIBRATION_ADAPTER_MODE,
                            )
                            == "calibration-memory-after-load"
                        )
                        bottleneck = helper.drafter_bottleneck_state(adapter_block_root)
                        assert bottleneck["state"] == "adapter_calibration_memory_blocked"
                        assert bottleneck["next_step"] == "seed_adapter_calibration_memory_report"
                        report_tasks = helper.drafter_bottleneck_next_tasks(
                            adapter_block_root,
                            helper.result_rows(adapter_block_root),
                            123459,
                            reason="unit adapter memory blocker",
                        )
                        assert len(report_tasks) == 1
                        assert report_tasks[0]["supervisor_action"] == "calibration-memory-report"
                        helper.append_result(
                            adapter_block_root,
                            run_id="calibration-memory-report-adapter-memory-unit",
                            status="keep",
                            target="calibration-memory-report",
                            hypothesis="unit adapter memory report exhausted",
                            commit="abc123",
                            notes="state=adapter_calibration_memory_blocked blocker=calibration-memory-after-load",
                        )
                        after_report_tasks = helper.drafter_bottleneck_next_tasks(
                            adapter_block_root,
                            helper.result_rows(adapter_block_root),
                            123460,
                            reason="unit adapter memory report exhausted",
                        )
                        assert len(after_report_tasks) == 1
                        assert after_report_tasks[0]["id"].startswith("agent-deliberation-quant-safe-drafter-candidate-")
                        with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_DIR": str(adapter_block_root)}, clear=False):
                            assert helper.drafter_adapter_method_contract(Namespace(recent_rows=240)) == 0
                        adapter_contract_rows = helper.result_rows(adapter_block_root)
                        assert adapter_contract_rows[-1]["target"] == "janq-drafter-adapter-method"
                        assert adapter_contract_rows[-1]["status"] == "keep"
                        assert "terminal_routed=True" in adapter_contract_rows[-1]["notes"]
                        adapter_contract_tasks = helper.read_jsonl(adapter_block_root / "tasks.jsonl")
                        assert any(
                            str(task.get("id", "")).startswith("implementation-drafter-adapter-method-")
                            and task.get("status") == "ready"
                            and task.get("contract_path")
                            for task in adapter_contract_tasks
                        )
                        legacy_adapter_blocker = [
                            {
                                "timestamp": "2026-05-10T00:00:00+0000",
                                "run_id": "drafter-adapter-method-contract-unit",
                                "status": "blocked",
                                "target": "janq-drafter-adapter-method",
                                "hypothesis": "old adapter contract blocker",
                                "notes": "state=adapter_calibration_memory_blocked ok=False path=/tmp/contract.json",
                            }
                        ]
                        assert helper.actionable_blocked_rows(legacy_adapter_blocker) == []
                        clean_report_root = Path(tmp) / "calibration-memory-report-clean-root"
                        helper.ensure_research_state(clean_report_root)
                        with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_DIR": str(clean_report_root)}, clear=False):
                            assert helper.calibration_memory_report(Namespace()) == 0
                        clean_report_rows = helper.result_rows(clean_report_root)
                        assert clean_report_rows[-1]["target"] == "calibration-memory-report"
                        assert clean_report_rows[-1]["status"] == "keep"
                        assert "blocker=none" in clean_report_rows[-1]["notes"]
                        assert helper.actionable_blocked_rows(clean_report_rows) == []
                        helper.append_result(
                            clean_report_root,
                            run_id="old-calibration-memory-report-none",
                            status="blocked",
                            target="calibration-memory-report",
                            hypothesis="old inverted no-blocker report",
                            commit="abc123",
                            notes="blocker=none hit_count=177 plateau=false best_clean_decode_tps=",
                        )
                        helper.append_result(
                            clean_report_root,
                            run_id="old-review-council-repair",
                            status="blocked",
                            target="autoresearch-review-council",
                            hypothesis="old repair council row",
                            commit="abc123",
                            notes="decision=repair ok=False seeded_tasks=0",
                        )
                        helper.append_result(
                            clean_report_root,
                            run_id="frontier-system-eval-clean",
                            status="keep",
                            target="autoresearch-frontier-eval",
                            hypothesis="clean frontier checkpoint",
                            commit="abc123",
                            notes="overall=10.0 readiness=frontier",
                        )
                        helper.append_result(
                            clean_report_root,
                            run_id="review-council-clean",
                            status="keep",
                            target="autoresearch-review-council",
                            hypothesis="clean council checkpoint",
                            commit="abc123",
                            notes="decision=continue ok=True seeded_tasks=0",
                        )
                        assert helper.unresolved_actionable_blocked_rows(helper.result_rows(clean_report_root)) == []
                        gradient_blocked_root = Path(tmp) / "calibration-gradient-block-root"
                        helper.ensure_research_state(gradient_blocked_root)
                        helper.append_result(
                            gradient_blocked_root,
                            run_id="supervisor-drafter-calibration-memory-stage-unit",
                            status="blocked",
                            target="janq-drafter-calibration-memory-stage",
                            hypothesis="unit gradient blocker",
                            commit="abc123",
                            notes=(
                                "stage=micro-step reason=calibration-quantized-gradient-unsupported "
                                "output_tail=[QuantizedMatmul::vjp] no gradient wrt the quantized weights."
                            ),
                        )
                        assert (
                            helper.recent_calibration_run_hard_blocker(gradient_blocked_root)
                            == "calibration-quantized-gradient-unsupported"
                        )
                        assert (
                            helper.calibration_quantized_gradient_issue("calibration-quantized-gradient-unsupported")
                            == "calibration-quantized-gradient-unsupported"
                        )
                        literal_blocked_root = Path(tmp) / "literal-calibration-gradient-block-root"
                        helper.ensure_research_state(literal_blocked_root)
                        helper.append_result(
                            literal_blocked_root,
                            run_id="drafter-calibration-memory-stage-micro-step-literal",
                            status="blocked",
                            target="janq-drafter-calibration-memory-stage",
                            hypothesis="unit literal gradient blocker",
                            commit="abc123",
                            notes=(
                                "stage=micro-step decision=terminal-blocker "
                                "failures=probe_exit:2,calibration-quantized-gradient-unsupported "
                                "blocker=calibration-quantized-gradient-unsupported"
                            ),
                        )
                        assert (
                            helper.recent_calibration_run_hard_blocker(literal_blocked_root)
                            == "calibration-quantized-gradient-unsupported"
                        )
                        assert helper.filter_seedable_tasks(
                            literal_blocked_root,
                            [
                                helper.drafter_calibration_canary_task(
                                    123457,
                                    task_id="lane-contract-drafter-calibration-canary-literal-blocked",
                                )
                            ],
                        ) == []
                        assert helper.recent_terminal_calibration_block_rows(literal_blocked_root)
                        helper.append_result(
                            literal_blocked_root,
                            run_id="supervisor-drafter-calibration-memory-stage-wrapper",
                            status="blocked",
                            target="openclaw/openclaw-mtp-drafter-calibrate.py",
                            hypothesis="unit wrapper row",
                            commit="abc123",
                            notes=(
                                'stage=micro-step reason=terminal-blocker blocker= '
                                'output_tail=projection.scales\\n", "returncode": 2'
                            ),
                        )
                        wrapper_row = helper.result_rows(literal_blocked_root)[-1]
                        assert helper.is_known_terminal_calibration_blocked_row(wrapper_row)
                        assert (
                            helper.recent_calibration_run_hard_blocker(literal_blocked_root)
                            == "calibration-quantized-gradient-unsupported"
                        )
                        adapter_memory_rows = [
                            {
                                "run_id": "drafter-calibration-memory-stage-micro-step-unit",
                                "status": "blocked",
                                "target": "janq-drafter-calibration-memory-stage",
                                "notes": (
                                    "stage=micro-step decision=blocked failures=probe_exit:2 "
                                    "calibration_mode=adapter-logit-distillation blocker="
                                ),
                            },
                            {
                                "run_id": "supervisor-drafter-calibration-memory-stage-unit",
                                "status": "blocked",
                                "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
                                "notes": "stage=micro-step reason=calibration-memory-gate:after-load",
                            },
                        ]
                        assert all(helper.is_known_calibration_memory_blocked_row(row) for row in adapter_memory_rows)
                        assert helper.actionable_blocked_rows(adapter_memory_rows) == []
                        helper.append_result(
                            literal_blocked_root,
                            run_id="drafter-calibration-memory-stage-micro-step-literal-repeat",
                            status="blocked",
                            target="janq-drafter-calibration-memory-stage",
                            hypothesis="unit literal gradient blocker repeat",
                            commit="abc123",
                            notes=(
                                "stage=micro-step decision=terminal-blocker "
                                "failures=probe_exit:2,calibration-quantized-gradient-unsupported "
                                "blocker=calibration-quantized-gradient-unsupported"
                            ),
                        )
                        with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_DIR": str(literal_blocked_root)}, clear=False):
                            assert helper.quality_review(
                                Namespace(recent_rows=80, min_sweeps=3, min_samples_per_block=3, target_tps=30.0)
                            ) == 0
                        literal_review = json.loads(
                            max(
                                (literal_blocked_root / "benchmarks").glob("quality-review-*.json"),
                                key=lambda path: path.stat().st_mtime_ns,
                            ).read_text(encoding="utf-8")
                        )
                        assert literal_review["verdict"] == "needs-repair"
                        assert literal_review["quality_score"] <= 74
                        literal_tasks = (literal_blocked_root / "tasks.jsonl").read_text(encoding="utf-8")
                        assert "lane-contract-drafter-calibration-canary" not in literal_tasks
                        helper.append_result(
                            gradient_blocked_root,
                            run_id="supervisor-drafter-calibration-memory-stage-truncated-unit",
                            status="blocked",
                            target="janq-drafter-calibration-memory-stage",
                            hypothesis="unit truncated gradient blocker",
                            commit="abc123",
                            notes='stage=micro-step reason=blocked output_tail=uantized weights.\\n", "returncode": 2',
                        )
                        assert (
                            helper.recent_calibration_run_hard_blocker(gradient_blocked_root)
                            == "calibration-quantized-gradient-unsupported"
                        )
                        assert not helper.should_seed_drafter_calibration_canary(gradient_blocked_root)
                        assert not helper.should_seed_drafter_calibration_run(gradient_blocked_root)
                        stage_mode_root = Path(tmp) / "stage-mode-root"
                        helper.ensure_research_state(stage_mode_root)
                        for index, stage_name in enumerate(helper.CALIBRATION_MEMORY_STAGES):
                            helper.append_result(
                                stage_mode_root,
                                run_id=f"drafter-calibration-memory-stage-{stage_name}-{index}",
                                status="keep",
                                target="janq-drafter-calibration-memory-stage",
                                hypothesis="unit direct stage",
                                commit="abc123",
                                notes=(
                                    f"stage={stage_name} decision=advance "
                                    f"calibration_mode={helper.CALIBRATION_DIRECT_MODE}"
                                ),
                            )
                        assert helper.first_seedable_calibration_memory_stage(stage_mode_root) == ""
                        assert (
                            helper.first_seedable_calibration_memory_stage(
                                stage_mode_root,
                                calibration_mode_filter=helper.CALIBRATION_ADAPTER_MODE,
                            )
                            == "metadata"
                        )
                        certified_scorecard = helper.research_quality_scorecard(
                            blocked_rows=0,
                            missing_required_blocks=["2", "3", "4"],
                            sweep_rows=3,
                            min_sweeps=3,
                            repeated_block2=True,
                            repeated_keep_current=True,
                            plateau_below_target=False,
                            exhaustion_candidate=False,
                            frontier_ready=[],
                            seeded_tasks=[],
                            ready_tasks=[
                                {
                                    "id": "agent-deliberation-handoff-recovery-contract-unit",
                                    "acceptance": "guarded",
                                    "rollback": "discard",
                                    "guard_checks": ["tests_pass"],
                                }
                            ],
                            contaminated_rows=0,
                            clean_runtime_maps=0,
                            variance={},
                            artifact_check={"artifact_suspected": False},
                            contract_ok=True,
                            dflash_suppressed=True,
                            repeated_dflash_synthesis=0,
                            duplicate_stage_tasks=0,
                            best_mean=None,
                            target_tps=30.0,
                            server_decode_values=[],
                            canonical_state="prerequisite_needed",
                        )
                        assert certified_scorecard["overall"] >= 99.0
                        low_signal_scorecard = helper.research_quality_scorecard(
                            blocked_rows=0,
                            missing_required_blocks=["2", "3", "4"],
                            sweep_rows=3,
                            min_sweeps=3,
                            repeated_block2=True,
                            repeated_keep_current=True,
                            plateau_below_target=False,
                            exhaustion_candidate=False,
                            frontier_ready=["runtime-overhead"],
                            seeded_tasks=[],
                            ready_tasks=[
                                {
                                    "id": "low-signal-source-scout-cycle-000",
                                    "acceptance": "bounded external source evidence updates the next route",
                                    "rollback": "delete task and keep prior queue",
                                    "guard_checks": ["no_model_load", "no_live_profile_change"],
                                },
                                {
                                    "id": "low-signal-runtime-map-cycle-000",
                                    "acceptance": "runtime map finds a patchable boundary or closes cleanly",
                                    "rollback": "delete task and keep prior queue",
                                    "guard_checks": ["no_model_load", "no_live_profile_change"],
                                },
                                {
                                    "id": "low-signal-frontier-deliberation-cycle-000",
                                    "acceptance": "deliberation selects one safe deterministic next task",
                                    "rollback": "delete task and keep prior queue",
                                    "guard_checks": ["no_model_load", "no_live_profile_change"],
                                },
                            ],
                            contaminated_rows=0,
                            clean_runtime_maps=1,
                            variance={},
                            artifact_check={"artifact_suspected": False},
                            contract_ok=True,
                            dflash_suppressed=True,
                            repeated_dflash_synthesis=0,
                            duplicate_stage_tasks=0,
                            best_mean=None,
                            target_tps=30.0,
                            server_decode_values=[],
                            canonical_state="breakthrough_lane_active",
                        )
                        assert low_signal_scorecard["overall"] >= 99.0
                        assert low_signal_scorecard["components"]["causal"] >= 98.0
                        assert helper.actionable_blocked_rows(helper.result_rows(gradient_blocked_root)) == []
                        assert helper.filter_seedable_tasks(
                            gradient_blocked_root,
                            [
                                helper.drafter_calibration_canary_task(
                                    123457,
                                    task_id="drafter-calibration-canary-gradient-blocked",
                                ),
                                helper.drafter_calibration_memory_stage_task(
                                    123457,
                                    stage="micro-step",
                                    task_id="drafter-calibration-memory-stage-micro-step-blocked",
                                    bounded_command=["python3", "calibrate.py"],
                                ),
                            ],
                        ) == []
                        gradient_expansion = helper.frontier_expansion_tasks(
                            gradient_blocked_root,
                            helper.result_rows(gradient_blocked_root),
                            123457,
                        )
                        assert len(gradient_expansion) == 1
                        assert gradient_expansion[0]["lane"] == "frontier-expansion"
                        assert gradient_expansion[0]["supervisor_action"] == "focused-test"
                        assert gradient_expansion[0]["metric"] == "frontier_candidate_gate"
                        assert "no_model_load" in gradient_expansion[0]["guard_checks"]
                        assert "JANQ calibration is blocked" in gradient_expansion[0]["hypothesis"]
                        assert not helper.task_contract_issues(gradient_blocked_root, gradient_expansion[0])["blockers"]
                        helper.upsert_tasks(gradient_blocked_root, gradient_expansion)
                        consumed_gradient_tasks = helper.read_jsonl(gradient_blocked_root / "tasks.jsonl")
                        for task in consumed_gradient_tasks:
                            if task["id"] == gradient_expansion[0]["id"]:
                                task["status"] = "done"
                        helper.write_jsonl(gradient_blocked_root / "tasks.jsonl", consumed_gradient_tasks)
                        assert (
                            helper.frontier_expansion_tasks(
                                gradient_blocked_root,
                                helper.result_rows(gradient_blocked_root),
                                123458,
                            )
                            == []
                        )
                        helper.upsert_tasks(
                            gradient_blocked_root,
                            [
                                helper.drafter_calibration_memory_stage_task(
                                    123458,
                                    stage="micro-step",
                                    task_id="drafter-calibration-memory-stage-micro-step-stale",
                                    bounded_command=["python3", "calibrate.py"],
                                )
                            ],
                        )
                        assert helper.compact_terminal_calibration_tasks(gradient_blocked_root) == 1
                        assert not helper.active_calibration_memory_stage_tasks(gradient_blocked_root)
                        helper.upsert_tasks(
                            gradient_blocked_root,
                            [
                                helper.drafter_calibration_canary_task(
                                    123459,
                                    task_id="adapter-drafter-calibration-canary-current",
                                    calibration_mode_value=helper.CALIBRATION_ADAPTER_MODE,
                                )
                            ],
                        )
                        assert helper.compact_terminal_calibration_tasks(gradient_blocked_root) == 0
                        adapter_tasks = helper.read_jsonl(gradient_blocked_root / "tasks.jsonl")
                        adapter_task = next(
                            task for task in adapter_tasks if task["id"] == "adapter-drafter-calibration-canary-current"
                        )
                        assert adapter_task["status"] == "ready"
                        assert (
                            helper.lane_contract_fallback_tasks(
                                gradient_blocked_root,
                                helper.result_rows(gradient_blocked_root),
                                123457,
                                reason="unit gradient blocker",
                            )
                            == []
                        )
                        for task in adapter_tasks:
                            if task["id"] == "adapter-drafter-calibration-canary-current":
                                task["status"] = "done"
                        helper.write_jsonl(gradient_blocked_root / "tasks.jsonl", adapter_tasks)
                        gradient_fallback = helper.lane_contract_fallback_tasks(
                            gradient_blocked_root,
                            helper.result_rows(gradient_blocked_root),
                            123457,
                            reason="unit gradient blocker",
                        )
                        assert len(gradient_fallback) == 1
                        assert gradient_fallback[0]["id"].startswith("drafter-trace-distillation-run-")
                        assert gradient_fallback[0]["supervisor_action"] == "drafter-calibration-run"
                        assert gradient_fallback[0]["trace_distillation"] is True
                        assert "--target-trace-policy" in gradient_fallback[0]["bounded_command"]
                        assert "stop-gradient" in gradient_fallback[0]["bounded_command"]
                        assert not helper.task_contract_issues(gradient_blocked_root, gradient_fallback[0])["blockers"]
                        helper.upsert_tasks(gradient_blocked_root, gradient_fallback)
                        distillation_task_text = (gradient_blocked_root / "tasks.jsonl").read_text(encoding="utf-8")
                        assert "trace-distillation" in distillation_task_text
                        consumed_distillation_tasks = helper.read_jsonl(gradient_blocked_root / "tasks.jsonl")
                        for task in consumed_distillation_tasks:
                            if task["id"] == gradient_fallback[0]["id"]:
                                task["status"] = "blocked"
                                task["supervisor_summary"] = {
                                    "reason": (
                                        "calibration-quantized-gradient-unsupported "
                                        "calibration_mode=trace-distillation trace_distillation=True "
                                        "[QuantizedMatmul::vjp] no gradient wrt the quantized weights."
                                    )
                                }
                        helper.write_jsonl(gradient_blocked_root / "tasks.jsonl", consumed_distillation_tasks)
                        helper.append_result(
                            gradient_blocked_root,
                            run_id="supervisor-drafter-calibration-run-trace-distillation-unit",
                            status="blocked",
                            target="openclaw/openclaw-mtp-drafter-calibrate.py",
                            hypothesis="unit trace-distillation blocker",
                            commit="abc123",
                            notes=(
                                "calibration-quantized-gradient-unsupported "
                                "calibration_mode=trace-distillation trace_distillation=True "
                                "target_gradient_policy=stop-gradient "
                                "[QuantizedMatmul::vjp] no gradient wrt the quantized weights."
                            ),
                        )
                        assert helper.trace_distillation_proof_failed(gradient_blocked_root)
                        assert helper.filter_seedable_tasks(
                            gradient_blocked_root,
                            [
                                helper.drafter_trace_distillation_run_task(
                                    123458,
                                    task_id="drafter-trace-distillation-run-repeat",
                                    bounded_command=["python3", "calibrate.py"],
                                )
                            ],
                        ) == []
                        gradient_repair_fallback = helper.lane_contract_fallback_tasks(
                            gradient_blocked_root,
                            helper.result_rows(gradient_blocked_root),
                            123458,
                            reason="unit gradient blocker",
                        )
                        assert len(gradient_repair_fallback) == 1
                        assert gradient_repair_fallback[0]["id"].startswith("trace-distillation-gradient-repair-")
                        assert gradient_repair_fallback[0]["supervisor_action"] == "focused-test"
                        assert "test-mtp-drafter-calibrate-guards.py" in gradient_repair_fallback[0]["next_action"]
                        assert not helper.task_contract_issues(gradient_blocked_root, gradient_repair_fallback[0])["blockers"]
                        helper.upsert_tasks(gradient_blocked_root, gradient_repair_fallback)
                        active_repair_state = helper.drafter_bottleneck_state(gradient_blocked_root)
                        assert active_repair_state["state"] == "trace_distillation_repair_active"
                        assert active_repair_state["next_step"] == "wait_for_trace_distillation_repair"
                        helper.append_result(
                            gradient_blocked_root,
                            run_id="supervisor-drafter-fit-plan-trace-distillation-unit",
                            status="keep",
                            target="/tmp/fit-plan.json",
                            hypothesis="unit plan ready",
                            commit="abc123",
                            notes="decision=ready-for-target-generated-trace-data",
                        )
                        handoff_repair = helper.concrete_handoff_prerequisite_tasks(
                            gradient_blocked_root,
                            helper.result_rows(gradient_blocked_root),
                            123459,
                        )
                        assert handoff_repair == []
                        repair_done_root = Path(tmp) / "trace-repair-done-root"
                        helper.ensure_research_state(repair_done_root)
                        helper.append_result(
                            repair_done_root,
                            run_id="supervisor-drafter-calibration-run-trace-distillation-unit",
                            status="blocked",
                            target="openclaw/openclaw-mtp-drafter-calibrate.py",
                            hypothesis="unit trace-distillation blocker",
                            commit="abc123",
                            notes=(
                                "calibration-quantized-gradient-unsupported "
                                "calibration_mode=trace-distillation trace_distillation=True "
                                "[QuantizedMatmul::vjp] no gradient wrt the quantized weights."
                            ),
                        )
                        first_repair = helper.lane_contract_fallback_tasks(
                            repair_done_root,
                            helper.result_rows(repair_done_root),
                            123460,
                            reason="unit trace repair first",
                        )
                        assert first_repair[0]["id"] == "trace-distillation-gradient-repair-current"
                        helper.upsert_tasks(repair_done_root, first_repair)
                        repair_done_tasks = helper.read_jsonl(repair_done_root / "tasks.jsonl")
                        for task in repair_done_tasks:
                            if task["id"] == first_repair[0]["id"]:
                                task["status"] = "done"
                        helper.write_jsonl(repair_done_root / "tasks.jsonl", repair_done_tasks)
                        assert helper.trace_distillation_repair_attempted(repair_done_root)
                        after_repair = helper.lane_contract_fallback_tasks(
                            repair_done_root,
                            helper.result_rows(repair_done_root),
                            123461,
                            reason="unit trace repair already passed",
                        )
                        assert len(after_repair) == 1
                        assert after_repair[0]["id"].startswith("frontier-expansion-janq-adapter-path-")
                        assert not after_repair[0]["id"].startswith("trace-distillation-gradient-repair-")
                        helper.upsert_tasks(repair_done_root, after_repair)
                        adapter_path_tasks = helper.read_jsonl(repair_done_root / "tasks.jsonl")
                        for task in adapter_path_tasks:
                            if task["id"] == after_repair[0]["id"]:
                                task["status"] = "done"
                        helper.write_jsonl(repair_done_root / "tasks.jsonl", adapter_path_tasks)
                        adapter_bridge = helper.lane_contract_fallback_tasks(
                            repair_done_root,
                            helper.result_rows(repair_done_root),
                            123462,
                            reason="unit trace expansion already passed",
                        )
                        assert len(adapter_bridge) == 1
                        assert adapter_bridge[0]["id"] == "trace-distillation-adapter-bridge-current"
                        assert adapter_bridge[0]["supervisor_action"] == "focused-test"
                        assert not helper.task_contract_issues(repair_done_root, adapter_bridge[0])["blockers"]
                        helper.upsert_tasks(repair_done_root, adapter_bridge)
                        bridge_tasks = helper.read_jsonl(repair_done_root / "tasks.jsonl")
                        for task in bridge_tasks:
                            if task["id"] == adapter_bridge[0]["id"]:
                                task["status"] = "done"
                        helper.write_jsonl(repair_done_root / "tasks.jsonl", bridge_tasks)
                        adapter_contract = helper.lane_contract_fallback_tasks(
                            repair_done_root,
                            helper.result_rows(repair_done_root),
                            123463,
                            reason="unit adapter bridge already passed",
                        )
                        assert len(adapter_contract) == 1
                        assert adapter_contract[0]["id"] == "drafter-adapter-method-contract-current"
                        assert adapter_contract[0]["supervisor_action"] == "drafter-adapter-method-contract"
                        assert adapter_contract[0]["lane"] == "implementation-gate"
                        assert not helper.task_contract_issues(repair_done_root, adapter_contract[0])["blockers"]
                        helper.upsert_tasks(repair_done_root, adapter_contract)
                        helper.upsert_tasks(repair_done_root, adapter_contract)
                        assert (
                            len(
                                [
                                    task
                                    for task in helper.read_jsonl(repair_done_root / "tasks.jsonl")
                                    if str(task.get("id", "")).startswith("drafter-adapter-method-contract-")
                                    and task.get("status") == "ready"
                                ]
                            )
                            == 1
                        )
                        active_contract_state = helper.drafter_bottleneck_state(repair_done_root)
                        assert active_contract_state["state"] == "adapter_method_contract_active"
                        assert active_contract_state["next_step"] == "wait_for_adapter_method_contract"
                        revived_root = Path(tmp) / "revived-adapter-contract"
                        shutil.copytree(repair_done_root, revived_root)
                        revived_tasks = helper.read_jsonl(revived_root / "tasks.jsonl")
                        for task in revived_tasks:
                            if task["id"] == "drafter-adapter-method-contract-current":
                                task["status"] = "blocked"
                                task["blocked_at"] = "2026-05-10T00:00:00+0000"
                                task["blocked_reason"] = "unit stale blocked reason"
                                task["completed_at"] = "2026-05-09T00:00:00+0000"
                                task["supervisor_summary"] = {"notes": "stale"}
                        helper.write_jsonl(revived_root / "tasks.jsonl", revived_tasks)
                        revived_seed = helper.lane_contract_fallback_tasks(
                            revived_root,
                            helper.result_rows(revived_root),
                            123464,
                            reason="unit revive blocked adapter contract",
                        )
                        assert len(revived_seed) == 1
                        assert revived_seed[0]["id"] == "drafter-adapter-method-contract-current"
                        assert helper.upsert_tasks(revived_root, revived_seed) == 1
                        assert any(
                            task["id"] == "drafter-adapter-method-contract-current"
                            and task.get("status") == "ready"
                            and task.get("revived_at")
                            and "blocked_at" not in task
                            and "blocked_reason" not in task
                            and "completed_at" not in task
                            and "supervisor_summary" not in task
                            for task in helper.read_jsonl(revived_root / "tasks.jsonl")
                        )
                        with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_DIR": str(repair_done_root)}, clear=False):
                            assert helper.drafter_bottleneck_review(Namespace(recent_rows=240)) == 0
                            assert helper.drafter_adapter_method_contract(Namespace(recent_rows=240)) == 0
                        contract_paths = list((repair_done_root / "experiments").glob("drafter-adapter-method-contract-*.json"))
                        assert contract_paths
                        contract = json.loads(contract_paths[-1].read_text(encoding="utf-8"))
                        assert contract["kind"] == "drafter-adapter-method-contract"
                        assert "openclaw/openclaw-mtp-drafter-calibrate.py" in contract["allowed_source_files"]
                        implementation_tasks = helper.read_jsonl(repair_done_root / "tasks.jsonl")
                        assert any(str(task.get("id", "")).startswith("implementation-drafter-adapter-method-") for task in implementation_tasks)
                        for index in range(260):
                            helper.append_result(
                                repair_done_root,
                                run_id=f"decode-noise-after-adapter-contract-{index}",
                                status="keep",
                                target="decode-sample",
                                hypothesis="unit decode rows should not erase the JANQ bottleneck route",
                                commit="abc123",
                                notes="mode=decode-sample wall_decode_tps=16.0 server_decode_tps=16.0 draft_block_size=2 contaminated=0",
                            )
                        historical_state = helper.drafter_bottleneck_state(repair_done_root)
                        assert historical_state["state"] in {
                            "adapter_method_contract_ready",
                            "adapter_method_implementation_active",
                        }
                        assert historical_state["next_step"] in {
                            "seed_adapter_method_implementation",
                            "wait_for_adapter_method_implementation",
                        }
                        implementation_done_tasks = helper.read_jsonl(repair_done_root / "tasks.jsonl")
                        for task in implementation_done_tasks:
                            if str(task.get("id", "")).startswith("implementation-drafter-adapter-method-"):
                                task["status"] = "done"
                                task["completed_at"] = "2026-05-10T00:02:00+0000"
                        helper.write_jsonl(repair_done_root / "tasks.jsonl", implementation_done_tasks)
                        helper.append_result(
                            repair_done_root,
                            run_id="supervisor-focused-test-adapter-method-unit",
                            status="keep",
                            target="openclaw/openclaw-mtp-drafter-calibrate.py",
                            hypothesis="adapter method implementation unit pass",
                            commit="abc123",
                            notes="focused test passed adapter",
                        )
                        adapter_state = helper.drafter_bottleneck_state(repair_done_root)
                        assert adapter_state["state"] == "adapter_method_implementation_done"
                        assert adapter_state["next_step"] == "seed_adapter_calibration_canary"
                        adapter_tasks = helper.drafter_bottleneck_next_tasks(
                            repair_done_root,
                            helper.result_rows(repair_done_root),
                            123466,
                            reason="unit adapter route",
                        )
                        assert len(adapter_tasks) == 1
                        assert adapter_tasks[0]["id"] == "adapter-drafter-calibration-canary-current"
                        assert adapter_tasks[0]["calibration_mode"] == "adapter-logit-distillation"
                        synthesized_route = helper.synthesis_deliberate_action_tasks(
                            repair_done_root,
                            helper.result_rows(repair_done_root),
                            123465,
                        )
                        assert all(
                            not str(task.get("id", "")).startswith("deliberate-drafter-fit-plan-")
                            for task in synthesized_route
                        )
                        old_bridge_row = [
                            {
                                "timestamp": "2026-05-10T00:00:00+0000",
                                "run_id": "supervisor-implementation-bridge-unit",
                                "status": "blocked",
                                "target": "openclaw/openclaw-speed-research.py",
                                "hypothesis": "old bridge blocker",
                                "notes": "seeded=0 ready_deterministic=1 terminal_no_work=False issue=terminal",
                            },
                            {
                                "timestamp": "2026-05-10T00:01:00+0000",
                                "run_id": "quality-review-unit",
                                "status": "keep",
                                "target": "autoresearch-quality",
                                "hypothesis": "clean quality checkpoint",
                                "notes": "verdict=healthy score=98 scorecard_overall=97.7",
                            },
                        ]
                        assert helper.unresolved_actionable_blocked_rows(old_bridge_row) == []
                        quality_args = Namespace(
                            recent_rows=120,
                            min_sweeps=3,
                            min_samples_per_block=3,
                            target_tps=30.0,
                        )
                        assert helper.quality_review(quality_args) == 0
                        for index in range(3):
                            helper.append_result(
                                blocked_root,
                                run_id=f"synthesis-fallback-{index}",
                                status="keep",
                                target="synthesis",
                                hypothesis="unit repeated fallback",
                                commit="abc123",
                                notes=(
                                    "ideas=5 seeded_tasks=1 kind=frontier deliberate_actions= "
                                    f"contract_actions=lane-contract-decode-remeasure-calibration-block-{index}"
                                ),
                            )
                        assert helper.recent_lane_contract_decode_fallback_count(blocked_root) == 3
                        overhead_fallback = helper.lane_contract_fallback_tasks(
                            blocked_root,
                            helper.result_rows(blocked_root),
                            123458,
                            reason="unit repeated no ready task",
                        )
                        if overhead_fallback:
                            assert len(overhead_fallback) == 1
                            assert overhead_fallback[0]["supervisor_action"] == "runtime-overhead-map"
                        else:
                            assert any(
                                task.get("status", "ready") in {"ready", "rework"}
                                and task.get("supervisor_action") == "runtime-overhead-map"
                                for task in helper.read_jsonl(blocked_root / "tasks.jsonl")
                            )
                            overhead_fallback = [
                                task
                                for task in helper.read_jsonl(blocked_root / "tasks.jsonl")
                                if task.get("status", "ready") in {"ready", "rework"}
                                and task.get("supervisor_action") == "runtime-overhead-map"
                            ][:1]
                        helper.upsert_tasks(blocked_root, overhead_fallback)
                        fallback_tasks = helper.read_jsonl(blocked_root / "tasks.jsonl")
                        for task in fallback_tasks:
                            if task["id"] == overhead_fallback[0]["id"]:
                                task["status"] = "done"
                        helper.write_jsonl(blocked_root / "tasks.jsonl", fallback_tasks)
                        helper.append_result(
                            blocked_root,
                            run_id="runtime-overhead-map-after-fallback-unit",
                            status="keep",
                            target="runtime-overhead-map",
                            hypothesis="unit overhead map",
                            commit="abc123",
                            notes="contaminated=0 mean_server_tps=14.0 mean_clean_wall_tps=14.0 hit_count=12",
                        )
                        mtp_fallback = helper.lane_contract_fallback_tasks(
                            blocked_root,
                            helper.result_rows(blocked_root),
                            123459,
                            reason="unit repeated no ready task",
                        )
                        assert len(mtp_fallback) == 1
                        assert mtp_fallback[0].get("supervisor_action") in {
                            "mtp-report",
                            "runtime-overhead-map",
                        } or mtp_fallback[0].get("benchmark_mode") == "decode-sample"
                        helper.upsert_tasks(blocked_root, mtp_fallback)
                        fallback_tasks = helper.read_jsonl(blocked_root / "tasks.jsonl")
                        for task in fallback_tasks:
                            if task["id"] == mtp_fallback[0]["id"]:
                                task["status"] = "done"
                        helper.write_jsonl(blocked_root / "tasks.jsonl", fallback_tasks)
                        helper.append_result(
                            blocked_root,
                            run_id="mtp-report-after-fallback-unit",
                            status="keep",
                            target="mtp-acceptance-report",
                            hypothesis="unit mtp report",
                            commit="abc123",
                            notes="samples=2 mtp_samples=2 mean_server_tok_s=14.0 mean_accept=0.6",
                        )
                        fallback_tasks = helper.read_jsonl(blocked_root / "tasks.jsonl")
                        fallback_tasks.append(
                            {
                                "id": "lane-contract-mtp-report-after-fallback-unit",
                                "status": "done",
                                "lane": "exhaustion-report",
                                "task_type": "supervisor",
                                "supervisor_action": "mtp-report",
                            }
                        )
                        helper.write_jsonl(blocked_root / "tasks.jsonl", fallback_tasks)
                        exhausted_fallback = helper.lane_contract_fallback_tasks(
                            blocked_root,
                            helper.result_rows(blocked_root),
                            123460,
                            reason="unit repeated no ready task",
                        )
                        assert exhausted_fallback == []
                        assert "lane-contract-fallback-exhausted" in (
                            blocked_root / "findings.jsonl"
                        ).read_text(encoding="utf-8")
                        dflash_blocked_root = Path(tmp) / "dflash-block-root"
                        helper.ensure_research_state(dflash_blocked_root)
                        dflash_artifact = dflash_blocked_root / "experiments" / "dflash-compatibility-gate-123.json"
                        dflash_artifact.write_text(
                            json.dumps({"blockers": ["draft_model_type_mismatch=gemma4!=gemma4-janq"]}),
                            encoding="utf-8",
                        )
                        helper.append_result(
                            dflash_blocked_root,
                            run_id="dflash-compatibility-gate-123",
                            status="blocked",
                            target="frontier-dflash",
                            hypothesis="unit dflash blocker",
                            commit="abc123",
                            notes="decision=blocked draft_model_type_mismatch=gemma4!=gemma4-janq",
                        )
                        dflash_expansion = helper.frontier_expansion_tasks(
                            dflash_blocked_root,
                            helper.result_rows(dflash_blocked_root),
                            123458,
                        )
                        assert len(dflash_expansion) == 1
                        assert dflash_expansion[0]["id"].startswith("frontier-expansion-dflash-candidate-search-")
                        assert "same-tokenizer JANQ-compatible" in dflash_expansion[0]["hypothesis"]
                        mtp_exhausted_root = Path(tmp) / "mtp-exhausted-root"
                        helper.ensure_research_state(mtp_exhausted_root)
                        helper.mark_lane_exhausted(
                            mtp_exhausted_root,
                            lane="mtp-decode",
                            reason="unit block-size settled below target",
                            evidence={"winner_block": 2},
                        )
                        mtp_expansion = helper.frontier_expansion_tasks(
                            mtp_exhausted_root,
                            helper.result_rows(mtp_exhausted_root),
                            123459,
                        )
                        assert len(mtp_expansion) == 1
                        assert mtp_expansion[0]["id"].startswith("frontier-expansion-mtp-verify-cache-")
                        assert "MTP verify/cache/rollback" in mtp_expansion[0]["hypothesis"]
                        for index in range(4):
                            helper.append_result(
                                blocked_root,
                                run_id=f"benchmark-plateau-{index}",
                                status="keep",
                                target="decode-sample",
                                hypothesis="unit clean plateau decode",
                                commit="abc123",
                                decode_tps=14.0 + index / 10,
                                wall_s=6.8,
                                notes="server_tok_s=14.5 server_elapsed_s=6.6 measurement_quality=clean",
                            )
                        helper.append_result(
                            blocked_root,
                            run_id="supervisor-drafter-fit-unit",
                            status="keep",
                            target="/tmp/fit-plan.json",
                            hypothesis="unit plan ready",
                            commit="abc123",
                            notes="decision=ready-for-target-generated-trace-data",
                        )
                        plateau = helper.recent_calibration_fallback_plateau(blocked_root)
                        assert plateau
                        assert plateau["best_clean_decode_tps"] == 14.5
                        assert plateau["no_model_escalations_done"] is True
                        handoff_after_plateau = helper.concrete_handoff_prerequisite_tasks(
                            blocked_root,
                            helper.result_rows(blocked_root),
                            123461,
                        )
                        assert len(handoff_after_plateau) == 1
                        assert handoff_after_plateau[0]["supervisor_action"] == "calibration-memory-report"
                        assert "handoff-audit-calibration-plateau" in (
                            blocked_root / "findings.jsonl"
                        ).read_text(encoding="utf-8")
            benchmark_json = sorted((root / "benchmarks").glob("benchmark-*-decode-sample.json"))[-1]
            benchmark_data = json.loads(benchmark_json.read_text(encoding="utf-8"))
            assert benchmark_data["draft_block_size"] in {1, 2, 3}
            assert benchmark_data["mtp"]["mean_accept"] == 0.75
            assert benchmark_data["measurement_quality"] == "clean"
            assert "mean_accept=0.75" in (root / "results.tsv").read_text(encoding="utf-8")
            assert "server_elapsed_s=6.0" in (root / "results.tsv").read_text(encoding="utf-8")
            sweep_run_paths = list((root / "experiments").glob("mtp-drafter-sweep-run-*.json"))
            assert sweep_run_paths
            sweep_run = json.loads(sweep_run_paths[-1].read_text(encoding="utf-8"))
            assert sweep_run["decision"] in {"keep-current", "promotion-ready"}
            assert sweep_run["promotion_gate"]["must_restore_live_profile"] is True
            with (root / "results.tsv").open("a", encoding="utf-8") as file:
                for index in range(3):
                    file.write(
                        f"2026-05-05T00:10:0{index}+0000\tdrafter-sweep-run-review-{index}\tkeep\t"
                        "OPENCLAW_JANG_DRAFT_BLOCK_SIZE\tpaired drafter block-size sweep\t\t\t14.2\t\t\tabc123\t"
                        "decision=keep-current control_block=2 winner_block=2 delta_vs_control=0.0\n"
                    )
                for block, tps in (("2", "14.2"), ("3", "12.5"), ("4", "10.5")):
                    for index in range(3):
                        file.write(
                            f"2026-05-05T00:11:{block}{index}+0000\tbenchmark-review-{block}-{index}\tkeep\t"
                            "decode-sample\tbounded OpenClaw decode-sample probe\t\t\t"
                            f"{tps}\t6.8\t1.2\tabc123\tmodel=local completion_tokens=96 "
                            f"token_source=usage.completion_tokens draft_block_size={block}\n"
                        )
            assert helper.quality_review(
                Namespace(recent_rows=80, min_sweeps=3, min_samples_per_block=3, target_tps=30.0)
            ) == 0
            assert len(helper.completed_drafter_sweep_rows(root, recent_rows=1, min_sweeps=3)) >= 3
            assert helper.plateau_pivot(Namespace(recent_rows=80, min_sweeps=3, target_tps=30.0)) == 0
            review_paths = list((root / "benchmarks").glob("quality-review-*.json"))
            assert review_paths
            review = json.loads(max(review_paths, key=lambda path: path.stat().st_mtime_ns).read_text(encoding="utf-8"))
            assert review["repeated_block2_winner"] is True
            assert review["verdict"] == "converged-below-target"
            assert review["scorecard"]["overall"] >= 90
            assert review["scorecard"]["components"]["causal"] >= 90
            assert review["scorecard"]["components"]["next_action"] >= 90
            assert review["quality_score"] >= review["legacy_quality_score"]
            assert review["gates"]["required_block_coverage"] is True
            assert review["best_block"] == "2"
            assert review["target_tps"] == 30.0
            assert "variance" in review
            assert review["gates"]["no_measurement_artifact"] is True
            assert review["gates"]["no_contaminated_wall_clock"] is True
            assert "mtp-decode" in (root / "exhausted-approaches.jsonl").read_text(encoding="utf-8")
            task_text = (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert "review-mtp-loop-overhead-next" in task_text or "mtp-loop-overhead-map" in task_text
            plateau_paths = list((root / "benchmarks").glob("plateau-pivot-*.json"))
            assert plateau_paths
            plateau = json.loads(plateau_paths[-1].read_text(encoding="utf-8"))
            assert plateau["state"] == "pivot"
            with (root / "results.tsv").open("a", encoding="utf-8") as file:
                file.write(
                    "2026-05-05T00:11:50+0000\tmtp-report-unit\tkeep\tmtp-acceptance-report\t"
                    "unit mtp report\t\t\t\t\t\tabc123\tsamples=3 mtp_samples=3 mean_accept=0.7\n"
                )
                for index in range(2):
                    file.write(
                        f"2026-05-05T00:11:5{index + 1}+0000\truntime-overhead-map-unit-{index}\tkeep\t"
                        "runtime-overhead-map\tclean runtime map\t\t\t\t\t\tabc123\t"
                        "contaminated=0 mean_server_tps=14.0 mean_clean_wall_tps=13.9 hit_count=10\n"
                    )
            tasks_without_active_calibration = helper.read_jsonl(root / "tasks.jsonl")
            for task in tasks_without_active_calibration:
                if str(task.get("supervisor_action", "")).startswith("drafter-calibration"):
                    task["status"] = "done"
            helper.write_jsonl(root / "tasks.jsonl", tasks_without_active_calibration)
            clean_routed = helper.synthesis_deliberate_action_tasks(root, helper.result_rows(root), 123456)
            assert clean_routed
            assert clean_routed[0]["id"].startswith("deliberate-drafter-fit-plan-")
            assert "openclaw-drafter-fit plan" in clean_routed[0]["next_action"]
            with (root / "results.tsv").open("a", encoding="utf-8") as file:
                file.write(
                    "2026-05-05T00:12:00+0000\tsupervisor-drafter-fit-unit\tkeep\t"
                    "/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json\t"
                    "unit drafter plan\t\t\t\t\t\tabc123\t"
                    "decision=ready-for-target-generated-trace-data output=plan.json min_speedup=1.35 min_accept=2.25\n"
                )
            trace_routed = helper.synthesis_deliberate_action_tasks(root, helper.result_rows(root), 123457)
            assert trace_routed
            assert trace_routed[0]["id"].startswith("deliberate-drafter-trace-gate-")
            assert "openclaw-speed-research drafter-trace-gate" in trace_routed[0]["next_action"]
            assert helper.quality_review(
                Namespace(recent_rows=80, min_sweeps=3, min_samples_per_block=3, target_tps=30.0)
            ) == 0
            clean_review = json.loads(
                max((root / "benchmarks").glob("quality-review-*.json"), key=lambda path: path.stat().st_mtime_ns).read_text()
            )
            assert clean_review["clean_runtime_overhead_maps"] >= 2
            assert clean_review["gates"]["runtime_overhead_not_repeated"] is True
            assert clean_review["compacted_runtime_tasks"] >= 1
            assert clean_review["scorecard"]["components"]["novelty"] >= 70
            task_text = (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert "review-janq-drafter-fit-next" in task_text or "janq-dflash-drafter-fit-plan" in task_text
            with (root / "results.tsv").open("a", encoding="utf-8") as file:
                for index in range(3):
                    file.write(
                        f"2026-05-05T00:12:0{index}+0000\tfallback-decode-{index}\tkeep\t"
                        "decode-sample\tdeterministic fallback benchmark\t\t\t1.9\t50.0\t4.0\tabc123\t"
                        "model=local completion_tokens=96 token_source=usage.completion_tokens "
                        "server_tok_s=15.7 server_elapsed_s=6.1 measurement_quality=contaminated\n"
                    )
            assert helper.quality_review(
                Namespace(recent_rows=80, min_sweeps=3, min_samples_per_block=3, target_tps=30.0)
            ) == 0
            contaminated_review = json.loads(
                max((root / "benchmarks").glob("quality-review-*.json"), key=lambda path: path.stat().st_mtime_ns).read_text()
            )
            assert contaminated_review["contaminated_decode_rows"] >= 3
            assert contaminated_review["mean_server_decode_tps"] >= 15.7
            assert contaminated_review["gates"]["no_contaminated_wall_clock"] is False
            assert "runtime-overhead-map" in (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert helper.runtime_overhead_map(Namespace(recent_rows=80)) == 0
            overhead = json.loads(sorted((root / "benchmarks").glob("runtime-overhead-map-*.json"))[-1].read_text())
            assert overhead["contaminated_decode_rows"] >= 3
            assert overhead["max_server_decode_tps"] >= 15.7
            assert helper.frontier_review(Namespace(recent_rows=80, min_samples=3)) == 0
            frontier_paths = list((root / "benchmarks").glob("frontier-review-*.json"))
            assert frontier_paths
            frontier = json.loads(frontier_paths[-1].read_text(encoding="utf-8"))
            assert frontier["variance"]["best_variant"] == "2"
            assert "mtp-decode" in frontier["exhausted_lanes"]
            helper.append_jsonl(
                root / "trajectory-corpus.jsonl",
                {"task_id": "unit-gepa", "reason": "repeated blocked trajectory", "evidence": "unit"},
            )
            assert helper.gepa_escalation(
                Namespace(recent_rows=80, min_blocked=1, min_rework=1, min_trajectory=1, min_low_quality=1)
            ) == 0
            gepa_paths = list((root / "benchmarks").glob("gepa-escalation-*.json"))
            assert gepa_paths
            gepa = json.loads(gepa_paths[-1].read_text(encoding="utf-8"))
            assert gepa["needed"] is False
            assert gepa["non_gepa_ready_work_exists"] is True
            original_tasks = helper.read_jsonl(root / "tasks.jsonl")
            temporarily_done_tasks = [dict(task) for task in original_tasks]
            for task in temporarily_done_tasks:
                if task.get("status", "ready") in {"ready", "rework"}:
                    task["status"] = "done"
            helper.write_jsonl(root / "tasks.jsonl", temporarily_done_tasks)
            assert helper.gepa_escalation(
                Namespace(recent_rows=80, min_blocked=1, min_rework=1, min_trajectory=1, min_low_quality=1)
            ) == 0
            gepa = json.loads(max((root / "benchmarks").glob("gepa-escalation-*.json"), key=lambda path: path.stat().st_mtime_ns).read_text())
            assert gepa["needed"] is True
            assert gepa["candidate"]["target"] in {"program.md", "insight-rubric.json", "STRATEGY.md", "tasks.jsonl"}
            assert "gepa-policy-canary" in (root / "tasks.jsonl").read_text(encoding="utf-8")
            canary_task = next(
                task
                for task in helper.read_jsonl(root / "tasks.jsonl")
                if task.get("id") == gepa["candidate"]["id"]
            )
            helper.write_jsonl(root / "tasks.jsonl", [*original_tasks, canary_task])
            assert helper.gepa_policy_canary(Namespace(task_id=gepa["candidate"]["id"])) == 0
            assert list((root / "gepa-canaries").glob("*.json"))
            gepa_candidates_text = (root / "gepa-candidates.jsonl").read_text(encoding="utf-8")
            assert "GEPA policy candidates remain canary-only" in gepa_candidates_text
            assert "actionable_side_information" in gepa_candidates_text
            assert "pareto_objectives" in gepa_candidates_text
            assert "Preserve OpenClaw-only scope" in gepa_candidates_text
            assert helper.gepa_policy_promote(Namespace(min_candidates=99)) == 0
            assert root.joinpath("results.tsv").read_text(encoding="utf-8").splitlines()[-1].split("\t")[2] == "keep"
            for index in range(3):
                helper.append_jsonl(
                    root / "gepa-candidates.jsonl",
                    {
                        "path": f"/tmp/gepa-{index}.json",
                        "candidate": {"target": "insight-rubric.json"},
                        "promotion": {"auto_promote": False},
                    },
                )
            assert helper.gepa_policy_promote(Namespace(min_candidates=3)) == 0
            promoted_rubric = json.loads((root / "insight-rubric.json").read_text(encoding="utf-8"))
            assert "measurement_quality" in promoted_rubric["promotion_required_fields"]
            assert "decode_claim_missing_server_tok_s" in promoted_rubric["reject_if"]
            assert helper.hypothesis_rank(Namespace(limit=5)) == 0
            rank_paths = list((root / "benchmarks").glob("hypothesis-rank-*.json"))
            assert rank_paths
            rank = json.loads(rank_paths[-1].read_text(encoding="utf-8"))
            assert rank["ranked"]
            assert rank["ranked"][0]["score"] >= rank["ranked"][-1]["score"]
            helper.append_jsonl(
                root / "promotion-decisions.jsonl",
                {
                    "task_id": "low-confidence",
                    "decision": "keep-canary",
                    "confidence": 0.5,
                    "target": "program.md",
                },
            )
            assert helper.causal_review(Namespace(recent_rows=80)) == 0
            causal_paths = list((root / "benchmarks").glob("causal-review-*.json"))
            assert causal_paths
            causal = json.loads(causal_paths[-1].read_text(encoding="utf-8"))
            assert "low-confidence" in causal["low_confidence_kept"]
            assert causal["low_confidence_action"].startswith("recorded_in_causal_review_only")
            causal_task_ids = [
                str(task.get("id", ""))
                for task in helper.read_jsonl(root / "tasks.jsonl")
                if str(task.get("lane", "")) == "causal-repair"
            ]
            assert not any(task_id.startswith("causal-review-low-confidence") for task_id in causal_task_ids)
            assert helper.synthesize(Namespace(kind="frontier")) == 0
            ideas = (root / "ideas.md").read_text(encoding="utf-8")
            assert "mtp-acceptance-bottleneck" in ideas
            assert "quality score" in ideas
            assert "cause:" in ideas
            assert "expected metric delta:" in ideas
            assert "rollback:" in ideas
            assert "drafter-block-and-quant-sweep" in ideas
            assert "janq-drafter-alignment" in ideas
            assert "dflash-janq-compatibility" in ideas
            assert "mathematical handle" in ideas
            assert "Implementation Candidates" in ideas
            assert "implement-mtp-acceptance-report" in ideas
            task_rows = helper.read_jsonl(root / "tasks.jsonl")
            tasks = (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert "decode-mtp-baseline" in tasks
            assert "mtp-acceptance-log-review" in tasks
            assert "post-mtp-acceptance-report" in tasks
            assert any(
                helper.semantic_task_key(task) == "production-mtp:decode-sample:decode_tps"
                and task.get("status", "ready") in {"ready", "rework"}
                for task in task_rows
            )
            assert "mtp-acceptance-report" in tasks
            assert "implement-mtp-acceptance-report" in tasks
            assert "implement-drafter-sweep-plan" in tasks
            assert "implement-janq-drafter-calibration-gate" in tasks
            assert "janq-dflash-drafter-fit-plan" in tasks
            assert '"supervisor_action": "drafter-sweep-run"' in tasks
            assert "openclaw-speed-research mtp-report" in tasks
            findings = (root / "findings.jsonl").read_text(encoding="utf-8")
            assert "synthesize-speed-ideas" in findings
            assert '"quality"' in findings
            assert "implementation_candidates" in findings
            assert "synthesis" in (root / "results.tsv").read_text(encoding="utf-8")
            assert helper.implementation_handoff_audit(Namespace(min_score=90)) == 0
            repo = Path(tmp) / "repo"
            (repo / "openclaw").mkdir(parents=True)
            (repo / "openclaw" / "openclaw-jang-vlm-server.py").write_text(
                "\n".join(
                    [
                        "from dflash.model_mlx import load_draft",
                        "from dflash.model_mlx import stream_generate",
                        "def validate_dflash_compatibility(): pass",
                        "def should_use_dflash(): pass",
                        "def record_dflash_acceptance(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            (repo / "openclaw" / "openclaw-jang-vlm-launcher.py").write_text(
                "def ensure_dflash_runtime():\n    return 'import dflash.model_mlx'\n",
                encoding="utf-8",
            )
            draft = Path(tmp) / "draft"
            draft.mkdir()
            (draft / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "gemma4",
                        "dflash_config": {"target_layer_ids": [1, 12, 23, 35, 46, 57]},
                    }
                ),
                encoding="utf-8",
            )
            fit_plan = Path(tmp) / "fit-plan.json"
            fit_plan.write_text(
                json.dumps({"decision": "ready-for-target-generated-trace-data"}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_REPO": str(repo)}, clear=False):
                assert helper.dflash_compatibility_gate(
                    Namespace(draft_path=str(draft), plan=str(fit_plan))
                ) == 0
            dflash_reports = list((root / "experiments").glob("dflash-compatibility-gate-*.json"))
            assert dflash_reports
            dflash_report = json.loads(dflash_reports[-1].read_text(encoding="utf-8"))
            assert dflash_report["status"] == "keep"
            assert dflash_report["evidence"]["server_hooks"]["acceptance_metrics"] is True
            (draft / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "dflash_config": {"target_layer_ids": [1, 12, 23, 35, 46, 57]},
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_REPO": str(repo)}, clear=False):
                assert helper.dflash_compatibility_gate(
                    Namespace(draft_path=str(draft), plan=str(fit_plan))
                ) == 0
            mismatch_report = json.loads(sorted((root / "experiments").glob("dflash-compatibility-gate-*.json"))[-1].read_text())
            assert mismatch_report["status"] == "blocked"
            assert "draft_model_type_mismatch=qwen3" in mismatch_report["blockers"]
            assert helper.dflash_lane_is_blocked(root, recent_rows=20) is True
            assert helper.suppress_hard_blocked_dflash_lane(root, recent_rows=20) is True
            assert "frontier-dflash" in helper.exhausted_lanes(root)
            assert helper.quality_review(
                Namespace(recent_rows=120, min_sweeps=3, min_samples_per_block=3, target_tps=30.0)
            ) == 0
            dflash_review = json.loads(
                max((root / "benchmarks").glob("quality-review-*.json"), key=lambda path: path.stat().st_mtime_ns).read_text()
            )
            assert dflash_review["scorecard"]["signals"]["dflash_suppressed"] is True
            assert dflash_review["scorecard"]["components"]["convergence"] >= 90
            helper.write_jsonl(
                root / "tasks.jsonl",
                [task for task in helper.read_jsonl(root / "tasks.jsonl") if "dflash" not in str(task.get("id", ""))],
            )
            no_dflash = helper.synthesis_deliberate_action_tasks(root, helper.result_rows(root), 123458)
            assert not any("dflash" in str(task.get("id", "")) for task in no_dflash)
            helper.write_jsonl(root / "tasks.jsonl", [])
            assert helper.mark_lane_exhausted(
                root,
                lane="frontier-dflash",
                reason="regression-test exhausted lane",
                evidence={"source": "unit-test"},
            ) is False
            exhausted_only = helper.filter_seedable_tasks(
                root,
                [helper.dflash_compatibility_task(123460, task_id="deliberate-dflash-compatibility-exhausted")],
            )
            assert exhausted_only == []
            helper.write_jsonl(
                root / "tasks.jsonl",
                [
                    helper.drafter_calibration_memory_stage_task(
                        123461,
                        stage="metadata",
                        task_id="drafter-calibration-memory-stage-metadata-a",
                        bounded_command=["true"],
                    ),
                    helper.drafter_calibration_memory_stage_task(
                        123462,
                        stage="metadata",
                        task_id="drafter-calibration-memory-stage-metadata-b",
                        bounded_command=["true"],
                    ),
                ],
            )
            assert helper.calibration_stage_duplicate_count(root) == 1
            assert helper.compact_duplicate_calibration_stage_tasks(root) == 1
            assert helper.calibration_stage_duplicate_count(root) == 0
            assert len(helper.active_calibration_memory_stage_tasks(root)) == 1
            assert helper.filter_seedable_tasks(
                root,
                [helper.drafter_calibration_canary_task(123463, task_id="drafter-calibration-canary-suppressed")],
            ) == []
            helper.write_jsonl(
                root / "tasks.jsonl",
                [
                    helper.drafter_calibration_canary_task(
                        123463,
                        task_id="drafter-calibration-canary-stale",
                    ),
                    *helper.active_calibration_memory_stage_tasks(root),
                ],
            )
            assert helper.compact_stale_calibration_canary_tasks(root) == 1
            compacted_queue = helper.read_jsonl(root / "tasks.jsonl")
            assert any(
                task["id"] == "drafter-calibration-canary-stale" and task["status"] == "done"
                for task in compacted_queue
            )
            assert any(
                helper.calibration_memory_stage_name(task) and task["status"] == "ready"
                for task in compacted_queue
            )
            helper.write_jsonl(root / "tasks.jsonl", [])
            assert helper.mark_lane_exhausted(
                root,
                lane="frontier-dflash",
                reason="regression-test dflash blocked",
                evidence={"source": "unit-test"},
            ) is False
            blocked_fallback = helper.filter_seedable_tasks(
                root,
                [
                    helper.lane_contract_decode_task(
                        123464,
                        task_id="lane-contract-decode-remeasure-dflash-block-123464",
                        priority=99,
                        reason="unit-test",
                    )
                ],
            )
            assert blocked_fallback == []
            helper.write_jsonl(
                root / "tasks.jsonl",
                [
                    helper.drafter_calibration_memory_stage_task(
                        123465,
                        stage="metadata",
                        task_id="drafter-calibration-memory-stage-metadata-ready",
                        bounded_command=["true"],
                    )
                ],
            )
            assert helper.lane_contract_fallback_tasks(
                root,
                [],
                123466,
                reason="unit-test dflash blocked with active calibration stage",
            ) == []
            helper.write_jsonl(root / "tasks.jsonl", [])
            handoff_paths = list((root / "benchmarks").glob("implementation-handoff-audit-*.json"))
            assert handoff_paths
            handoff = json.loads(handoff_paths[-1].read_text(encoding="utf-8"))
            assert handoff["score"] >= 90
            assert handoff["gates"]["patch_executor_contract_ready"] is True
            assert handoff["gates"]["scoped_candidates_have_guards"] is True
            exhausted_tasks = helper.read_jsonl(root / "tasks.jsonl")
            for task in exhausted_tasks:
                task["status"] = "done"
            helper.write_jsonl(root / "tasks.jsonl", exhausted_tasks)
            assert helper.synthesize(Namespace(kind="frontier")) == 0
            deliberate_tasks = (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert "deliberate-dflash-compatibility-" not in deliberate_tasks
            if "deliberate-drafter-trace-gate-" not in deliberate_tasks:
                helper.write_jsonl(
                    root / "tasks.jsonl",
                    [helper.drafter_trace_gate_task(123459, task_id="deliberate-drafter-trace-gate-test")],
                )
                deliberate_tasks = (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert "deliberate_actions" in (root / "findings.jsonl").read_text(encoding="utf-8")
            helper.append_result(
                root,
                run_id="external-change-required-unit",
                status="blocked",
                target="autoresearch-external-change-required",
                hypothesis="old external blocker should not poison a fresh ready prerequisite",
                commit="unit-test",
                notes='evidence={"exhausted_lanes":["drafter-calibration-memory"],"ready_tasks":["deliberate-drafter-trace-gate-test"]}',
            )
            queued = helper.read_jsonl(root / "tasks.jsonl")
            queued.append(
                {
                    "id": "implementation-drafter-adapter-method-current",
                    "status": "blocked",
                    "lane": "implementation-gate",
                    "task_type": "implementation",
                    "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
                }
            )
            helper.write_jsonl(root / "tasks.jsonl", queued)
            helper.append_result(
                root,
                run_id="supervisor-implementation-guard-unit",
                status="blocked",
                target="openclaw/openclaw-mtp-drafter-calibrate.py",
                hypothesis="implementation model turns should be routed to deterministic patch executor paths",
                commit="unit-test",
                notes="malformed hidden/tool output",
            )
            canonical_with_external = helper.canonical_autoresearch_state(root, recent_rows=120)
            assert canonical_with_external["noise"]["unresolved_blocked_rows"] == 0
            assert canonical_with_external["noise"]["memory_blocks"] == 0
            assert "routed_blocked_rows" not in canonical_with_external["noise"]
            assert canonical_with_external["resolved_debt"]["routed_blocked_rows"] >= 2
            assert canonical_with_external["state"] == "breakthrough_lane_active"
            assert helper.frontier_eval(Namespace(recent_rows=120, min_score=8.0, allow_fail=False)) == 0
            eval_paths = list((root / "benchmarks").glob("frontier-system-eval-*.json"))
            assert eval_paths
            eval_report = json.loads(eval_paths[-1].read_text(encoding="utf-8"))
            assert eval_report["scores"]["karpathy_core_loop"] >= 8.0
            assert eval_report["scores"]["research_quality"] >= 9.4
            assert eval_report["latest_quality_scorecard_overall"] >= 94
            assert eval_report["latest_quality_verdict"] == "converged-below-target"
            assert eval_report["readiness"] == "frontier-candidate"
            assert eval_report["frontier_certified"] is False
            assert eval_report["frontier_requirements"]["latest_quality_healthy"] is False
            assert eval_report["canonical_state"]["state"] == "breakthrough_lane_active"
            assert eval_report["canonical_state"]["noise"]["unresolved_blocked_rows"] == 0
            assert eval_report["latest_quality_artifact"].endswith(".json")
            assert any("decode still below practical floor" in gap for gap in eval_report["gaps"])
            assert "deliberate-drafter-trace-gate-" in "\n".join(eval_report["deterministic_ready_tasks"])
            assert helper.frontier_eval(Namespace(recent_rows=120, min_score=9.0, allow_fail=True)) == 0
            repair_tasks = (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert "frontier-repair-measurement-artifact-" not in repair_tasks
            all_done_tasks = helper.read_jsonl(root / "tasks.jsonl")
            for task in all_done_tasks:
                task["status"] = "done"
            helper.write_jsonl(root / "tasks.jsonl", all_done_tasks)
            assert helper.frontier_eval(Namespace(recent_rows=120, min_score=9.0, allow_fail=True)) == 0
            empty_queue_repair = [
                task
                for task in helper.read_jsonl(root / "tasks.jsonl")
                if helper.is_deterministic_research_task(task) and task.get("status", "ready") == "ready"
            ]
            assert empty_queue_repair
            repaired_tasks = helper.read_jsonl(root / "tasks.jsonl")
            for task in repaired_tasks:
                task["status"] = "done"
            helper.write_jsonl(root / "tasks.jsonl", repaired_tasks)
            assert helper.implementation_handoff_audit(Namespace(min_score=90)) == 0
            handoff_report = json.loads(
                max(
                    (root / "benchmarks").glob("implementation-handoff-audit-*.json"),
                    key=lambda path: path.stat().st_mtime_ns,
                ).read_text(encoding="utf-8")
            )
            assert handoff_report["seeded_bridge"] is False
            assert handoff_report["seeded_prerequisite"] or handoff_report["seeded_expansion"]
            assert handoff_report["ready_deterministic_tasks"]
            bridge_tasks = helper.read_jsonl(root / "tasks.jsonl")
            seeded_ready = [
                task
                for task in bridge_tasks
                if task.get("status") == "ready" and helper.is_deterministic_research_task(task)
            ]
            assert seeded_ready
            assert all(task.get("acceptance") for task in seeded_ready)
            assert all(task.get("rollback") for task in seeded_ready)
            for task in bridge_tasks:
                task["status"] = "done"
            helper.write_jsonl(root / "tasks.jsonl", bridge_tasks)
            helper.append_result(
                root,
                run_id="supervisor-implementation-bridge-unit",
                status="blocked",
                target="implementation-bridge",
                hypothesis="unit empty bridge",
                commit="abc123",
                notes="seeded=0 ready_deterministic=0 issue=no deterministic implementation tasks available",
            )
            assert helper.implementation_handoff_audit(Namespace(min_score=90)) == 0
            handoff_after_empty = json.loads(
                max(
                    (root / "benchmarks").glob("implementation-handoff-audit-*.json"),
                    key=lambda path: path.stat().st_mtime_ns,
                ).read_text(encoding="utf-8")
            )
            assert handoff_after_empty["seeded_bridge"] is False
            assert handoff_after_empty["seeded_prerequisite"] is True
            handoff_ready = "\n".join(handoff_after_empty["ready_deterministic_tasks"])
            assert (
                "handoff-audit-drafter-calibration-canary-" in handoff_ready
                or "adapter-drafter-calibration-canary-current" in handoff_ready
                or "frontier-expansion-dflash-candidate-search-" in handoff_ready
                or "frontier-expansion-janq-adapter-path-" in handoff_ready
                or "frontier-expansion-mtp-verify-cache-" in handoff_ready
            )
            handoff_tasks = helper.read_jsonl(root / "tasks.jsonl")
            for task in handoff_tasks:
                task["status"] = "done"
            helper.write_jsonl(root / "tasks.jsonl", handoff_tasks)
            helper.append_result(
                root,
                run_id="supervisor-implementation-bridge-terminal",
                status="keep",
                target="implementation-bridge",
                hypothesis="unit terminal bridge",
                commit="abc123",
                notes="seeded=0 ready_deterministic=0 terminal_no_work=True issue=supervisor synthesis terminal no-work",
            )
            with patch.object(helper, "concrete_handoff_prerequisite_tasks", return_value=[]):
                with patch.object(helper, "lane_contract_fallback_tasks", return_value=[]):
                    assert helper.implementation_handoff_audit(Namespace(min_score=90)) == 0
            terminal_handoff = json.loads(
                max(
                    (root / "benchmarks").glob("implementation-handoff-audit-*.json"),
                    key=lambda path: path.stat().st_mtime_ns,
                ).read_text(encoding="utf-8")
            )
            assert terminal_handoff["ok"] is True
            assert terminal_handoff["terminal_handoff_exhausted"] is False
            assert terminal_handoff["seeded_prerequisite"] is True
            assert terminal_handoff["seeded_expansion"] is True
            assert terminal_handoff["ready_deterministic_tasks"][0].startswith("frontier-expansion-")
            handoff_rows = [
                row
                for row in helper.result_rows(root)
                if row.get("run_id", "").startswith("implementation-handoff-audit-")
            ]
            assert handoff_rows[-1]["status"] == "keep"
            assert "seeded_expansion=True" in handoff_rows[-1]["notes"]
            completed_handoff_tasks = helper.read_jsonl(root / "tasks.jsonl")
            for task in completed_handoff_tasks:
                task["status"] = "done"
            helper.write_jsonl(root / "tasks.jsonl", completed_handoff_tasks)
            helper.append_result(
                root,
                run_id="supervisor-implementation-bridge-deliberation",
                status="keep",
                target="implementation-bridge",
                hypothesis="unit terminal deliberation",
                commit="abc123",
                notes="seeded=0 ready_deterministic=0 terminal_no_work=True issue=supervisor synthesis terminal no-work",
            )
            with patch.object(helper, "concrete_handoff_prerequisite_tasks", return_value=[]):
                with patch.object(helper, "frontier_expansion_tasks", return_value=[]):
                    with patch.object(helper, "lane_contract_fallback_tasks", return_value=[]):
                        assert helper.implementation_handoff_audit(Namespace(min_score=90)) == 0
            deliberation_handoff = json.loads(
                max(
                    (root / "benchmarks").glob("implementation-handoff-audit-*.json"),
                    key=lambda path: path.stat().st_mtime_ns,
                ).read_text(encoding="utf-8")
            )
            assert deliberation_handoff["terminal_handoff_exhausted"] is False
            assert deliberation_handoff["seeded_deliberation"] is True
            deliberation_tasks = [
                task for task in helper.read_jsonl(root / "tasks.jsonl") if str(task.get("id", "")).startswith("agent-deliberation-")
            ]
            assert deliberation_tasks
            deliberation_artifact = json.loads(
                max(
                    (root / "benchmarks").glob("frontier-agent-deliberation-*.json"),
                    key=lambda path: path.stat().st_mtime_ns,
                ).read_text(encoding="utf-8")
            )
            assert (
                deliberation_artifact["gates"]["canonical_clean_or_repairable_terminal"] is True
                or deliberation_artifact.get("recovery_task_seeded") == 1
            )
            assert (
                deliberation_artifact["gates"]["zero_unsafe_noise"] is True
                or deliberation_artifact.get("recovery_task_seeded") == 1
            )
            assert deliberation_artifact["architect"]["contract_complete"] is True
            with patch.object(
                helper,
                "drafter_bottleneck_state",
                return_value={"state": "adapter_method_implementation_done", "next_step": "seed_adapter_calibration_canary"},
            ):
                adapter_canary = helper.drafter_calibration_canary_task(
                    123457,
                    task_id="adapter-drafter-calibration-canary-current",
                    calibration_mode_value=helper.CALIBRATION_ADAPTER_MODE,
                )
                with patch.object(helper, "drafter_bottleneck_next_tasks", return_value=[adapter_canary]):
                    breakout_report, breakout_tasks = helper.frontier_agent_deliberation(root, helper.result_rows(root), 123457)
            assert breakout_tasks
            assert breakout_tasks[0]["supervisor_action"] == "drafter-calibration-canary"
            assert breakout_tasks[0]["calibration_mode"] == helper.CALIBRATION_ADAPTER_MODE
            assert "canonical JANQ drafter bottleneck" in breakout_report["architect"]["selected_reason"]
            with patch.object(
                helper,
                "recent_calibration_run_hard_blocker",
                return_value=helper.CALIBRATION_QUANTIZED_GRADIENT_BLOCKER,
            ):
                assert helper.filter_seedable_tasks(root, [breakout_tasks[0]]) == [breakout_tasks[0]]
            with tempfile.TemporaryDirectory() as deliberation_tmp:
                deliberation_root = Path(deliberation_tmp) / "research" / "speed"
                helper.ensure_research_state(deliberation_root)
                (deliberation_root / "benchmarks" / "source-scout-unit.json").write_text(
                    json.dumps({"ok": True, "findings": [{"status": "fetched", "url": "https://github.com/karpathy/autoresearch"}]}),
                    encoding="utf-8",
                )
                with patch.object(
                    helper,
                    "drafter_bottleneck_state",
                    return_value={"state": "no_terminal_quantized_blocker", "next_step": "continue_current_lane_contract"},
                ):
                    mtp_report, mtp_tasks = helper.frontier_agent_deliberation(
                        deliberation_root,
                        helper.result_rows(deliberation_root),
                        123458,
                    )
            assert mtp_tasks
            assert mtp_tasks[0]["supervisor_action"] == "mtp-report"
            assert mtp_tasks[0]["metric"] == "mean_accept"
            assert mtp_report["architect"]["selected_task_id"].startswith("agent-deliberation-mtp-acceptance-yield-")
            patch_repo = Path(tmp) / "patch-repo"
            (patch_repo / "openclaw").mkdir(parents=True)
            (patch_repo / "openclaw" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
            (patch_repo / "openclaw" / "openclaw-model-proxy.py").write_text("MODE = 'old'\n", encoding="utf-8")
            (patch_repo / "openclaw" / "openclaw-mtp-drafter-calibrate.py").write_text("MODE = 'old'\n", encoding="utf-8")
            (patch_repo / "openclaw" / "test-speed-research.py").write_text("print('ok')\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=patch_repo, stdout=subprocess.DEVNULL, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=patch_repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=patch_repo, check=True)
            subprocess.run(["git", "add", "."], cwd=patch_repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=patch_repo, stdout=subprocess.DEVNULL, check=True)
            (patch_repo / "openclaw" / "sample.py").write_text("VALUE = 2\n", encoding="utf-8")
            patch_file = root / "patches" / "sample.patch"
            patch_file.parent.mkdir(parents=True, exist_ok=True)
            diff = subprocess.run(["git", "diff"], cwd=patch_repo, text=True, stdout=subprocess.PIPE, check=True)
            patch_file.write_text(diff.stdout, encoding="utf-8")
            subprocess.run(["git", "checkout", "--", "openclaw/sample.py"], cwd=patch_repo, check=True)
            assert helper.classify_patch(diff.stdout, source_files=["openclaw/sample.py"])["impact"] == "safe"
            high_risk_task = helper.agent_deliberation_task(
                123456,
                slug="unit-high-risk",
                priority=99,
                target="openclaw/openclaw-jang-vlm-server.py",
                hypothesis="unit high-risk deliberation path",
                acceptance="unit acceptance",
                evidence={},
            )
            assert high_risk_task["risk_tier"] == "high-risk"
            assert high_risk_task["crabbox_required"] is True
            assert high_risk_task["promotion_blocked_until_crabbox"] is True
            assert "crabbox_static_ssh_mac" in high_risk_task["guard_checks"]
            sota_report = helper.sota_autonomy_eval_report(root)
            assert sota_report["components"]["sandbox_governance"] == 20
            assert sota_report["gates"]["high_risk_classification_requires_crabbox"] is True
            assert sota_report["gates"]["agent_high_risk_task_requires_crabbox"] is True
            assert sota_report["modularity"]["lane_contract_count"] >= 4
            secret_diff = (
                "diff --git a/openclaw/sample.py b/openclaw/sample.py\n"
                "--- a/openclaw/sample.py\n"
                "+++ b/openclaw/sample.py\n"
                "@@ -1 +1 @@\n"
                "-VALUE = 1\n"
                "+DUMMY_PASSWORD = 'placeholder-not-real-value-1234567890'\n"
            )
            secret_classification = helper.classify_patch(secret_diff, source_files=["openclaw/sample.py"])
            assert secret_classification["allowed"] is False
            assert "patch adds possible secret material" in secret_classification["reasons"]
            assert helper.patch_execute(
                Namespace(
                    patch_file=str(patch_file),
                    task_id="unit-patch",
                    hypothesis="unit patch",
                    source_files="openclaw/sample.py",
                    tests="python3 openclaw/test-speed-research.py",
                    repo=str(patch_repo),
                    test_timeout=30.0,
                    canary_only=True,
                    keep_canary=False,
                    allow_architectural=False,
                    architectural_approval_file="",
                )
            ) == 0
            assert "patch-executor" in (root / "results.tsv").read_text(encoding="utf-8")
            (patch_repo / "openclaw" / "dirty.py").write_text("DIRTY = True\n", encoding="utf-8")
            assert helper.git_dirty_files(patch_repo)
            assert helper.patch_execute(
                Namespace(
                    patch_file=str(patch_file),
                    task_id="dirty-main-patch",
                    hypothesis="dirty main patch",
                    source_files="openclaw/sample.py",
                    tests="python3 openclaw/test-speed-research.py",
                    repo=str(patch_repo),
                    test_timeout=30.0,
                    canary_only=False,
                    keep_canary=False,
                    allow_architectural=False,
                    architectural_approval_file="",
                )
            ) == 2
            dirty_artifact = sorted((root / "experiments").glob("patch-executor-*-dirty-main-patch.json"))[-1]
            dirty_data = json.loads(dirty_artifact.read_text(encoding="utf-8"))
            assert dirty_data["reason"] == "main repo has uncommitted changes; refusing autonomous promotion"
            assert (patch_repo / "openclaw" / "sample.py").read_text(encoding="utf-8") == "VALUE = 1\n"
            (patch_repo / "openclaw" / "dirty.py").unlink()
            (root / "benchmarks" / "quality-review-1000.json").write_text(
                json.dumps(
                    {
                        "kind": "quality-review",
                        "verdict": "healthy",
                        "quality_score": 100,
                        "scorecard": {
                            "overall": 99.7,
                            "interpretation": "high_quality_exhaustion_or_prerequisite_route",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "benchmarks" / "frontier-system-eval-1000.json").write_text(
                json.dumps(
                    {
                        "kind": "frontier-system-eval",
                        "overall": 10.0,
                        "readiness": "frontier",
                        "frontier_certified": True,
                        "gaps": [],
                        "canonical_state": {
                            "clean": True,
                            "noise": {
                                "unresolved_blocked_rows": 0,
                                "terminal_synthesis_rows": 0,
                                "bridge_zero_rows": 0,
                                "memory_blocks": 0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            helper.append_result(
                root,
                run_id="frontier-system-eval-1000",
                status="keep",
                target="autoresearch-frontier-eval",
                hypothesis="unit clean checkpoint",
                commit="unit",
                notes="overall=10.0 readiness=frontier",
            )
            autonomy = helper.frontier_autonomy_score_report(root, promotion=False)
            assert autonomy["total_score"] == 100, autonomy
            assert autonomy["hard_gate_failures"] == []
            council_report = helper.review_council_report(root, recent_rows=120, target_tps=30.0, seed_next=True)
            assert council_report["roles"]["prober"]["questions"]
            assert council_report["roles"]["consultant"]["next_leverage"]
            assert council_report["progress_memory"]["objective"]
            assert council_report["deterministic_path"]["phase"]
            assert council_report["gates"]["progress_memory_present"] is True
            assert council_report["gates"]["progress_docs_present"] is True
            assert council_report["roles"]["consultant"]["progress_memory_lane"]["phase"]
            assert council_report["roles"]["skeptic"]["falsification_gates"]["zero_active_noise"] is True
            assert council_report["roles"]["gatekeeper"]["promotion_allowed"] is False
            assert council_report["scorecard"]["overall"] < 95, council_report["scorecard"]
            assert "safety_gates_clear" in council_report["scorecard"]["hard_gate_failures"]
            clean_council = dict(council_report)
            clean_council["gates"] = {
                key: True
                for key in (
                    "quality_artifact_present",
                    "frontier_artifact_present",
                    "handoff_artifact_present",
                    "autonomy_artifact_present",
                    "canonical_clean",
                    "zero_active_noise",
                    "no_bad_behavior_rows",
                    "task_contract_clean",
                    "handoff_clean",
                    "autonomy_clean",
                    "measurement_clean",
                    "progress_memory_present",
                    "progress_docs_present",
                    "progress_owner_ready_or_repairable",
                )
            }
            clean_score = helper.score_review_council_artifact(clean_council)
            assert clean_score["overall"] >= 95, clean_score
            assert clean_score["hard_gate_failures"] == [], clean_score
            autonomy_only_council = dict(clean_council)
            autonomy_only_council["gates"] = dict(clean_council["gates"])
            autonomy_only_council["gates"]["autonomy_clean"] = False
            autonomy_only_score = helper.score_review_council_artifact(autonomy_only_council)
            assert autonomy_only_score["overall"] >= 95, autonomy_only_score
            assert autonomy_only_score["hard_gate_failures"] == [], autonomy_only_score
            assert autonomy_only_score["signals"]["promotion_only_failed_safety_gates"] == ["autonomy_clean"]
            assert council_report["roles"]["strategist"]["decision"] in {
                "continue",
                "seed-frontier-deliberation",
                "observe",
                "repair",
            }
            assert helper.review_council(Namespace(recent_rows=120, target_tps=30.0, seed_next=True, allow_fail=True)) == 0
            council_paths = list((root / "benchmarks").glob("review-council-*.json"))
            assert council_paths
            assert "autoresearch-review-council" in (root / "results.tsv").read_text(encoding="utf-8")
            alive_after_council = helper.self_improvement_alive_report(root, recent_rows=120)
            assert alive_after_council["gates"]["review_council_quality"] is True
            bad_council = dict(council_report)
            bad_council["roles"] = {}
            bad_score = helper.score_review_council_artifact(bad_council)
            assert bad_score["overall"] < 95
            assert bad_score["hard_gate_failures"]
            (patch_repo / "openclaw" / "openclaw-model-proxy.py").write_text("MODE = 'new'\n", encoding="utf-8")
            arch_patch_file = root / "patches" / "architectural.patch"
            arch_diff = subprocess.run(["git", "diff"], cwd=patch_repo, text=True, stdout=subprocess.PIPE, check=True)
            arch_patch_file.write_text(arch_diff.stdout, encoding="utf-8")
            subprocess.run(["git", "checkout", "--", "openclaw/openclaw-model-proxy.py"], cwd=patch_repo, check=True)
            arch_blocked = helper.classify_patch(
                arch_diff.stdout,
                source_files=["openclaw/openclaw-model-proxy.py"],
                allow_architectural=False,
            )
            assert arch_blocked["allowed"] is False
            arch_allowed = helper.classify_patch(
                arch_diff.stdout,
                source_files=["openclaw/openclaw-model-proxy.py"],
                allow_architectural=True,
            )
            assert arch_allowed["impact"] == "architectural"
            assert arch_allowed["auto_promote"] is False
            assert arch_allowed["approval_required"] is True
            (patch_repo / "openclaw" / "openclaw-mtp-drafter-calibrate.py").write_text("MODE = 'new'\n", encoding="utf-8")
            high_risk_patch_file = root / "patches" / "high-risk.patch"
            high_risk_diff = subprocess.run(["git", "diff"], cwd=patch_repo, text=True, stdout=subprocess.PIPE, check=True)
            high_risk_patch_file.write_text(high_risk_diff.stdout, encoding="utf-8")
            subprocess.run(["git", "checkout", "--", "openclaw/openclaw-mtp-drafter-calibrate.py"], cwd=patch_repo, check=True)
            high_risk_classification = helper.classify_patch(
                high_risk_diff.stdout,
                source_files=["openclaw/openclaw-mtp-drafter-calibrate.py"],
                allow_architectural=False,
            )
            assert high_risk_classification["impact"] == "high-risk"
            assert high_risk_classification["allowed"] is True
            assert high_risk_classification["crabbox_required"] is True
            assert high_risk_classification["auto_promote"] is False
            assert helper.patch_execute(
                Namespace(
                    patch_file=str(high_risk_patch_file),
                    task_id="high-risk-patch",
                    hypothesis="high-risk patch",
                    source_files="openclaw/openclaw-mtp-drafter-calibrate.py",
                    tests="python3 openclaw/test-speed-research.py",
                    repo=str(patch_repo),
                    test_timeout=30.0,
                    canary_only=False,
                    keep_canary=False,
                    allow_architectural=False,
                    crabbox_evidence_file="",
                    rollback_rehearsal_ok=False,
                    architectural_approval_file="",
                )
            ) == 0
            high_risk_artifact = sorted((root / "experiments").glob("patch-executor-*-high-risk-patch.json"))[-1]
            high_risk_data = json.loads(high_risk_artifact.read_text(encoding="utf-8"))
            assert high_risk_data["held_for_crabbox"] is True
            assert high_risk_data["classification"]["impact"] == "high-risk"
            assert high_risk_data["promoted"] is False
            assert (patch_repo / "openclaw" / "openclaw-mtp-drafter-calibrate.py").read_text(encoding="utf-8") == "MODE = 'old'\n"
            assert helper.patch_execute(
                Namespace(
                    patch_file=str(arch_patch_file),
                    task_id="arch-patch",
                    hypothesis="architectural patch",
                    source_files="openclaw/openclaw-model-proxy.py",
                    tests="python3 openclaw/test-speed-research.py",
                    repo=str(patch_repo),
                    test_timeout=30.0,
                    canary_only=False,
                    keep_canary=False,
                    allow_architectural=True,
                    crabbox_evidence_file="",
                    rollback_rehearsal_ok=False,
                    architectural_approval_file="",
                )
            ) == 0
            arch_artifact = sorted((root / "experiments").glob("patch-executor-*-arch-patch.json"))[-1]
            arch_data = json.loads(arch_artifact.read_text(encoding="utf-8"))
            assert arch_data["held_for_crabbox"] is True
            assert arch_data["promoted"] is False
            assert (patch_repo / "openclaw" / "openclaw-model-proxy.py").read_text(encoding="utf-8") == "MODE = 'old'\n"
            crabbox_evidence = root / "crabbox-evidence.json"
            crabbox_evidence.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "runner": "static-ssh-mac",
                        "patch_sha256": helper.text_sha256(arch_diff.stdout),
                        "tests": [{"command": "python3 openclaw/test-speed-research.py", "ok": True}],
                        "full_suite": {"ok": True},
                        "rollback_rehearsal_ok": True,
                        "run_id": "run_unit_crabbox",
                    }
                ),
                encoding="utf-8",
            )
            assert helper.patch_execute(
                Namespace(
                    patch_file=str(arch_patch_file),
                    task_id="arch-patch",
                    hypothesis="architectural patch",
                    source_files="openclaw/openclaw-model-proxy.py",
                    tests="python3 openclaw/test-speed-research.py",
                    repo=str(patch_repo),
                    test_timeout=30.0,
                    canary_only=False,
                    keep_canary=False,
                    allow_architectural=True,
                    crabbox_evidence_file=str(crabbox_evidence),
                    rollback_rehearsal_ok=True,
                    architectural_approval_file="",
                )
            ) == 0
            assert (patch_repo / "openclaw" / "openclaw-model-proxy.py").read_text(encoding="utf-8") == "MODE = 'new'\n"
            stable_builds = helper.read_jsonl(root / "stable-builds.jsonl")
            assert stable_builds
            assert stable_builds[-1]["frontier_autonomy_score"] == 100
            bad_patch = "diff --git a/.env b/.env\n--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-A=1\n+A=2\n"
            bad_patch_file = root / "patches" / "bad.patch"
            bad_patch_file.write_text(bad_patch, encoding="utf-8")
            assert helper.patch_execute(
                Namespace(
                    patch_file=str(bad_patch_file),
                    task_id="bad-patch",
                    hypothesis="bad patch",
                    source_files="",
                    tests="python3 openclaw/test-speed-research.py",
                    repo=str(patch_repo),
                    test_timeout=30.0,
                    canary_only=True,
                    keep_canary=False,
                    allow_architectural=False,
                    architectural_approval_file="",
                )
            ) == 2
            secret_patch_file = root / "patches" / "secret.patch"
            secret_patch_file.write_text(secret_diff, encoding="utf-8")
            assert helper.patch_execute(
                Namespace(
                    patch_file=str(secret_patch_file),
                    task_id="secret-patch",
                    hypothesis="secret patch",
                    source_files="openclaw/sample.py",
                    tests="python3 openclaw/test-speed-research.py",
                    repo=str(patch_repo),
                    test_timeout=30.0,
                    canary_only=True,
                    keep_canary=False,
                    allow_architectural=False,
                    architectural_approval_file="",
                )
            ) == 2
        with tempfile.TemporaryDirectory() as handoff_tmp:
            handoff_root = Path(handoff_tmp) / "research" / "speed"
            handoff_home = Path(handoff_tmp) / "home"
            helper.ensure_research_state(handoff_root)
            traces = handoff_home / "drafter-fit" / "target-generated-traces.jsonl"
            traces.parent.mkdir(parents=True, exist_ok=True)
            traces.write_text(
                json.dumps({"prompt": "hello", "completion": "world", "completion_tokens": 2}) + "\n",
                encoding="utf-8",
            )
            helper.append_result(
                handoff_root,
                run_id="supervisor-drafter-fit-plan-unit",
                status="keep",
                target="drafter-fit-plan",
                hypothesis="unit fit plan",
                commit="unit",
                notes="decision=ready-for-target-generated-trace-data",
            )
            helper.append_result(
                handoff_root,
                run_id="drafter-sweep-run-unit",
                status="keep",
                target="decode-sample",
                hypothesis="unit decode",
                commit="unit",
                notes="decision=keep-current winner_block=2 wall_decode_tps=14.0",
            )
            with patch.dict(os.environ, {"OPENCLAW_HOME": str(handoff_home)}, clear=False):
                handoff_tasks = helper.synthesis_deliberate_action_tasks(
                    handoff_root,
                    helper.result_rows(handoff_root),
                    1234567,
                )
                assert handoff_tasks
                assert handoff_tasks[0]["supervisor_action"] == "drafter-calibration-canary"
                assert not any(task.get("supervisor_action") == "mtp-report" for task in handoff_tasks)
                helper.upsert_tasks(
                    handoff_root,
                    [
                        helper.calibration_memory_report_task(
                            1234568,
                            task_id="calibration-memory-report-active",
                        )
                    ],
                )
                assert helper.synthesis_deliberate_action_tasks(
                    handoff_root,
                    helper.result_rows(handoff_root),
                    1234569,
                ) == []
                report, deliberation_tasks = helper.frontier_agent_deliberation(
                    handoff_root,
                    helper.result_rows(handoff_root),
                    1234570,
                )
                assert deliberation_tasks == []
                assert "already active" in report["architect"]["selected_reason"]
        with tempfile.TemporaryDirectory() as progress_tmp:
            progress_root = Path(progress_tmp) / "research" / "speed"
            helper.ensure_research_state(progress_root)
            completed = helper.drafter_adapter_method_implementation_task(
                1778978000,
                task_id="implementation-drafter-adapter-method-completed",
            )
            completed["status"] = "done"
            stale = helper.drafter_adapter_method_implementation_task(
                1778979999,
                task_id="implementation-drafter-adapter-method-stale",
            )
            helper.write_jsonl(progress_root / "tasks.jsonl", [completed, stale])
            assert helper.adapter_method_implementation_completed(progress_root)
            assert helper.drafter_bottleneck_state(progress_root)["next_step"] == "seed_adapter_calibration_canary"
            blocked = helper.block_operational_strategy_ready_tasks(progress_root)
            assert blocked >= 1
            stale_after = [
                task
                for task in helper.read_jsonl(progress_root / "tasks.jsonl")
                if task["id"] == "implementation-drafter-adapter-method-stale"
            ][0]
            assert stale_after["status"] == "blocked"
            assert "already passed" in stale_after["blocked_reason"]
        with tempfile.TemporaryDirectory() as autonomy_tmp:
            autonomy_root = Path(autonomy_tmp) / "research" / "speed"
            helper.ensure_research_state(autonomy_root)
            benchmarks = autonomy_root / "benchmarks"
            benchmarks.mkdir(parents=True, exist_ok=True)
            (benchmarks / "quality-review-1.json").write_text(
                json.dumps({"ok": True, "quality_score": 100, "verdict": "healthy", "scorecard": {"overall": 98.3}}),
                encoding="utf-8",
            )
            (benchmarks / "frontier-system-eval-1.json").write_text(
                json.dumps({"ok": True, "overall": 10.0, "readiness": "frontier", "frontier_certified": True}),
                encoding="utf-8",
            )
            (benchmarks / "implementation-handoff-audit-1.json").write_text(
                json.dumps({"ok": True, "score": 100}),
                encoding="utf-8",
            )
            (benchmarks / "stability-burn-in-1.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            report = helper.frontier_autonomy_score_report(autonomy_root, promotion=False)
            assert report["ok"] is True
            assert report["total_score"] == 100
            promotion_report = helper.frontier_autonomy_score_report(autonomy_root, promotion=True)
            assert promotion_report["ok"] is False
            assert "scorecard_at_least_99" in promotion_report["hard_gate_failures"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
