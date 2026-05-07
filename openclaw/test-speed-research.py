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
            assert (root / "research-profile.json").exists()
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
            review_paths = list((root / "benchmarks").glob("quality-review-*.json"))
            assert review_paths
            review = json.loads(review_paths[-1].read_text(encoding="utf-8"))
            assert review["repeated_block2_winner"] is True
            assert review["verdict"] == "converged-below-target"
            assert review["gates"]["required_block_coverage"] is True
            assert review["best_block"] == "2"
            assert review["target_tps"] == 30.0
            assert "variance" in review
            assert review["gates"]["no_measurement_artifact"] is True
            assert review["gates"]["no_contaminated_wall_clock"] is True
            assert "mtp-decode" in (root / "exhausted-approaches.jsonl").read_text(encoding="utf-8")
            assert "review-mtp-loop-overhead-next" in (root / "tasks.jsonl").read_text(encoding="utf-8")
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
            contaminated_review = json.loads(sorted((root / "benchmarks").glob("quality-review-*.json"))[-1].read_text())
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
            assert causal["seeded_repair_tasks"] >= 1
            assert "causal-repair" in (root / "tasks.jsonl").read_text(encoding="utf-8")
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
            assert "openclaw-speed-research mtp-report" in tasks
            assert "First tool call: read exactly" in tasks
            findings = (root / "findings.jsonl").read_text(encoding="utf-8")
            assert "synthesize-speed-ideas" in findings
            assert '"quality"' in findings
            assert "implementation_candidates" in findings
            assert "synthesis" in (root / "results.tsv").read_text(encoding="utf-8")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
