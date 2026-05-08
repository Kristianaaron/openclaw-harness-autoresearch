#!/usr/bin/env python3
"""Checks for the OpenClaw speed autoresearch helper."""

from __future__ import annotations

import importlib.util
import json
import os
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
            assert helper.actionable_blocked_rows(
                [
                    {"status": "blocked", "target": "autoresearch-quality"},
                    {"status": "blocked", "target": "autoresearch-frontier-eval"},
                    {"status": "blocked", "target": "decode-sample"},
                ]
            ) == [{"status": "blocked", "target": "decode-sample"}]
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
            policy = json.loads((root / "evaluator-policy.json").read_text(encoding="utf-8"))
            assert "benchmark-manifest.json" in policy["immutable_paths"]
            assert "replay-buffer.jsonl" in policy["immutable_paths"]
            lane_contracts = json.loads((root / "lane-contracts.json").read_text(encoding="utf-8"))
            assert "drafter-alignment" in lane_contracts["lanes"]
            assert "calibration-memory-after-load" in lane_contracts["lanes"]["drafter-alignment"]["hard_blockers"]
            replay_cases = (root / "replay-buffer.jsonl").read_text(encoding="utf-8")
            assert "decode-token-source-required" in replay_cases
            assert "profile-variant-paired-control" in replay_cases
            assert helper.replay(Namespace(allow_fail=False)) == 0
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
            assert "GEPA-style policy optimization is a supervisor reflex" in program
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
                            "p.add_argument('--mlx-cache-gb'); a=p.parse_args(); "
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
                        assert len(contract_fallback) == 1
                        assert contract_fallback[0]["benchmark_mode"] == "decode-sample"
                        assert "calibration-memory-after-load" in contract_fallback[0]["hypothesis"]
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
                        assert len(overhead_fallback) == 1
                        assert overhead_fallback[0]["supervisor_action"] == "runtime-overhead-map"
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
                        assert mtp_fallback[0]["supervisor_action"] == "mtp-report"
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
            assert "review-mtp-loop-overhead-next" in (root / "tasks.jsonl").read_text(encoding="utf-8")
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
            assert clean_review["gates"]["runtime_overhead_not_repeated"] is False
            assert clean_review["scorecard"]["components"]["novelty"] >= 70
            assert "review-janq-drafter-fit-next" in (root / "tasks.jsonl").read_text(encoding="utf-8")
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
            assert gepa["needed"] is True
            assert gepa["candidate"]["target"] in {"program.md", "insight-rubric.json", "STRATEGY.md", "tasks.jsonl"}
            assert "gepa-policy-canary" in (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert helper.gepa_policy_canary(Namespace(task_id=gepa["candidate"]["id"])) == 0
            assert list((root / "gepa-canaries").glob("*.json"))
            assert "GEPA policy candidates remain canary-only" in (root / "gepa-candidates.jsonl").read_text(
                encoding="utf-8"
            )
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
            assert "dflash-janq-compatibility-spike" in tasks
            assert '"supervisor_action": "drafter-sweep-run"' in tasks
            assert '"supervisor_action": "dflash-compatibility-gate"' in tasks
            assert "openclaw-speed-research mtp-report" in tasks
            assert "openclaw-speed-research dflash-compatibility-gate" in tasks
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
            assert helper.frontier_eval(Namespace(recent_rows=120, min_score=8.0, allow_fail=False)) == 0
            eval_paths = list((root / "benchmarks").glob("frontier-system-eval-*.json"))
            assert eval_paths
            eval_report = json.loads(eval_paths[-1].read_text(encoding="utf-8"))
            assert eval_report["scores"]["karpathy_core_loop"] >= 8.0
            assert eval_report["scores"]["research_quality"] < 9.0
            assert eval_report["latest_quality_scorecard_overall"] <= 74
            assert eval_report["latest_quality_verdict"] == "needs-repair"
            assert eval_report["readiness"] == "needs-targeted-work"
            assert eval_report["frontier_certified"] is False
            assert eval_report["frontier_requirements"]["latest_quality_healthy"] is False
            assert any("latest quality review verdict is needs-repair" in gap for gap in eval_report["gaps"])
            assert "deliberate-drafter-trace-gate-" in "\n".join(eval_report["deterministic_ready_tasks"])
            assert helper.frontier_eval(Namespace(recent_rows=120, min_score=9.0, allow_fail=True)) == 0
            repair_tasks = (root / "tasks.jsonl").read_text(encoding="utf-8")
            assert "frontier-repair-measurement-artifact-" in repair_tasks
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
            assert handoff_report["seeded_bridge"] is True
            assert "handoff-audit-deterministic-bridge" in "\n".join(handoff_report["ready_deterministic_tasks"])
            bridge_tasks = helper.read_jsonl(root / "tasks.jsonl")
            bridge = next(task for task in bridge_tasks if str(task["id"]).startswith("handoff-audit-deterministic-bridge-"))
            assert "canary_only" in bridge["guard_checks"]
            assert bridge["acceptance"]
            assert bridge["rollback"]
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
            assert "handoff-audit-drafter-calibration-canary-" in "\n".join(
                handoff_after_empty["ready_deterministic_tasks"]
            )
            patch_repo = Path(tmp) / "patch-repo"
            (patch_repo / "openclaw").mkdir(parents=True)
            (patch_repo / "openclaw" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
            (patch_repo / "openclaw" / "openclaw-model-proxy.py").write_text("MODE = 'old'\n", encoding="utf-8")
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
                    architectural_approval_file="",
                )
            ) == 0
            arch_artifact = sorted((root / "experiments").glob("patch-executor-*-arch-patch.json"))[-1]
            arch_data = json.loads(arch_artifact.read_text(encoding="utf-8"))
            assert arch_data["held_for_approval"] is True
            assert arch_data["promoted"] is False
            assert (patch_repo / "openclaw" / "openclaw-model-proxy.py").read_text(encoding="utf-8") == "MODE = 'old'\n"
            approval_file = root / "architectural-approval.txt"
            approval_file.write_text("APPROVE_ARCHITECTURAL_PATCH=arch-patch\n", encoding="utf-8")
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
                    architectural_approval_file=str(approval_file),
                )
            ) == 0
            assert (patch_repo / "openclaw" / "openclaw-model-proxy.py").read_text(encoding="utf-8") == "MODE = 'new'\n"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
