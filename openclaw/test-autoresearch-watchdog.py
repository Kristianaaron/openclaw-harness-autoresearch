#!/usr/bin/env python3
"""Checks for the deterministic autoresearch watchdog sidecar."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


HELPER_PATH = Path(__file__).with_name("openclaw-autoresearch-watchdog.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("openclaw_autoresearch_watchdog", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_results(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(rows), encoding="utf-8")


def healthy_workspace(root: Path) -> None:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "autopilot-unit.log").write_text("cycle=1 ok\ncycle=2 ok\n", encoding="utf-8")
    write_json(root / "autopilot.lock", {"pid": 12345, "session": "unit"})
    write_results(
        root / "results.tsv",
        [
            "2026-05-10T20:00:00+0100\tbenchmark-1\tkeep\tdecode-sample\th\t\t\t15.2\t6.3\t\tabc\tserver_tok_s=15.4 measurement_quality=clean\n",
            "2026-05-10T20:01:00+0100\tquality-review-1\tkeep\tautoresearch-quality\th\t\t\t\t\t\tabc\tverdict=healthy score=100 scorecard_overall=99.7\n",
            "2026-05-10T20:02:00+0100\tfrontier-system-eval-1\tkeep\tautoresearch-frontier-eval\th\t\t\t\t\t\tabc\toverall=10.0 readiness=frontier gaps=0\n",
        ],
    )
    write_json(
        root / "benchmarks" / "quality-review-1.json",
        {
            "verdict": "healthy",
            "quality_score": 100,
            "scorecard": {"overall": 99.7},
        },
    )
    write_json(
        root / "benchmarks" / "frontier-system-eval-1.json",
        {
            "overall": 10.0,
            "readiness": "frontier",
            "frontier_certified": True,
            "gaps": [],
            "canonical_state": {
                "state": "breakthrough_lane_active",
                "clean": True,
                "ready_tasks": 1,
                "deterministic_ready_tasks": ["decode-remeasure"],
                "breakthrough_lanes": ["runtime-overhead"],
                "exhausted_lanes": ["frontier-dflash"],
                "noise": {
                    "unresolved_blocked_rows": 0,
                    "terminal_synthesis_rows": 0,
                    "bridge_zero_rows": 0,
                    "memory_blocks": 0,
                },
            },
        },
    )
    write_json(root / "benchmarks" / "implementation-handoff-audit-1.json", {"ok": True, "score": 100})
    write_json(root / "benchmarks" / "stability-burn-in-1.json", {"ok": True})


def main() -> int:
    helper = load_helper()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        healthy_workspace(root)
        with (
            patch.object(helper, "process_alive", return_value=True),
            patch.object(helper, "live_canonical_state", return_value={}),
            patch.object(helper.time, "time", return_value=1778439780),
        ):
            report = helper.watchdog_review(
                root,
                recent=120,
                target_tps=30,
                min_quality=95,
                min_scorecard=95,
                min_frontier=9.5,
                max_log_stale_seconds=300,
                max_result_stale_seconds=600,
            )
        assert report["severity"] == "healthy"
        assert report["decision"] == "continue-breakthrough-lane"
        assert report["gates"]["zero_active_noise"] is True
        assert report["decode"]["mean_wall_decode_tps"] == 15.2
        assert report["noise_interpretation"]["status"] == "clean"
        contract = helper.architecture_contract(root)
        assert contract["sidecar_authority"]["candidate_mode"] == "advisory-only"
        assert "tasks.jsonl" in contract["sidecar_authority"]["may_not_write"]
        candidates = helper.evidence_linked_candidates(report)
        assert candidates
        assert all(candidate["allowed_for_live_queue"] is False for candidate in candidates)
        assert candidates[0]["id"] == "continue-ready-deterministic-work"

        write_json(
            root / "benchmarks" / "quality-review-2.json",
            {
                "verdict": "healthy",
                "quality_score": 100,
                "scorecard": {"overall": 92.8},
            },
        )
        with (
            patch.object(helper, "process_alive", return_value=True),
            patch.object(helper, "live_canonical_state", return_value={}),
            patch.object(helper.time, "time", return_value=1778439780),
        ):
            degraded = helper.watchdog_review(
                root,
                recent=120,
                target_tps=30,
                min_quality=95,
                min_scorecard=95,
                min_frontier=9.5,
                max_log_stale_seconds=300,
                max_result_stale_seconds=600,
            )
        assert degraded["severity"] == "degraded"
        assert degraded["decision"] == "repair-routing"
        assert "scorecard_high" in degraded["blockers"]
        repair_candidates = helper.evidence_linked_candidates(degraded)
        assert repair_candidates[0]["id"] == "repair-before-new-research"
        assert repair_candidates[0]["allowed_for_live_queue"] is False

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        healthy_workspace(root)
        write_json(
            root / "benchmarks" / "frontier-system-eval-2.json",
            {
                "overall": 9.7,
                "readiness": "frontier-candidate",
                "frontier_certified": False,
                "gaps": [],
                "canonical_state": {
                    "state": "blocked_until_external_change",
                    "clean": True,
                    "ready_tasks": 0,
                    "deterministic_ready_tasks": [],
                    "breakthrough_lanes": [],
                    "exhausted_lanes": ["frontier-dflash", "mtp-decode"],
                    "noise": {
                        "unresolved_blocked_rows": 0,
                        "terminal_synthesis_rows": 0,
                        "bridge_zero_rows": 0,
                        "memory_blocks": 0,
                    },
                },
            },
        )
        with (
            patch.object(helper, "process_alive", return_value=True),
            patch.object(helper, "live_canonical_state", return_value={}),
            patch.object(helper.time, "time", return_value=1778439780),
        ):
            blocked = helper.watchdog_review(
                root,
                recent=120,
                target_tps=30,
                min_quality=95,
                min_scorecard=95,
                min_frontier=9.5,
                max_log_stale_seconds=300,
                max_result_stale_seconds=600,
            )
        assert blocked["severity"] == "attention"
        assert blocked["decision"] == "seed-next-candidate"
        assert "synthesize --kind frontier" in blocked["next_command"]
        blocked_candidates = helper.evidence_linked_candidates(blocked)
        assert any(candidate["id"] == "frontier-candidate-synthesis" for candidate in blocked_candidates)
        assert all(candidate["status"] == "advisory" for candidate in blocked_candidates)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        healthy_workspace(root)
        write_json(root / "autopilot.lock", {"pid": 999999, "session": "dead"})
        with patch.object(helper, "process_alive", return_value=False):
            recovery = helper.repair_stale_autopilot_lock(root)
        assert recovery["repaired"] is True
        assert not (root / "autopilot.lock").exists()
        assert Path(recovery["archive"]).exists()
        with (
            patch.object(helper, "live_canonical_state", return_value={}),
            patch.object(helper.time, "time", return_value=1778439780),
        ):
            idle = helper.watchdog_review(
                root,
                recent=120,
                target_tps=30,
                min_quality=95,
                min_scorecard=95,
                min_frontier=9.5,
                max_log_stale_seconds=1,
                max_result_stale_seconds=1,
            )
        assert idle["severity"] == "healthy"
        assert idle["decision"] == "idle-ready"
        assert idle["gates"]["log_fresh"] is True
        assert idle["gates"]["results_fresh"] is True

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        healthy_workspace(root)
        write_json(
            root / "benchmarks" / "frontier-system-eval-2.json",
            {
                "overall": 9.85,
                "readiness": "frontier-candidate",
                "frontier_certified": False,
                "gaps": [],
                "canonical_state": {
                    "state": "blocked_until_external_change",
                    "clean": True,
                    "ready_tasks": 0,
                    "deterministic_ready_tasks": [],
                    "breakthrough_lanes": [],
                    "exhausted_lanes": ["frontier-dflash", "mtp-decode"],
                    "noise": {
                        "unresolved_blocked_rows": 0,
                        "terminal_synthesis_rows": 4,
                        "bridge_zero_rows": 1,
                        "memory_blocks": 0,
                    },
                },
            },
        )
        live_clean = {
            "state": "breakthrough_lane_active",
            "clean": True,
            "ready_tasks": 1,
            "deterministic_ready_tasks": ["decode-remeasure-after-calibration-block"],
            "breakthrough_lanes": ["runtime-overhead"],
            "exhausted_lanes": ["frontier-dflash", "mtp-decode"],
            "noise": {
                "unresolved_blocked_rows": 0,
                "terminal_synthesis_rows": 0,
                "bridge_zero_rows": 0,
                "memory_blocks": 0,
            },
        }
        with (
            patch.object(helper, "process_alive", return_value=True),
            patch.object(helper, "live_canonical_state", return_value=live_clean),
            patch.object(helper.time, "time", return_value=1778439780),
        ):
            refreshed = helper.watchdog_review(
                root,
                recent=120,
                target_tps=30,
                min_quality=95,
                min_scorecard=95,
                min_frontier=9.5,
                max_log_stale_seconds=300,
                max_result_stale_seconds=600,
            )
        assert refreshed["gates"]["zero_active_noise"] is True
        assert refreshed["canonical_state"]["ready_tasks"] == 1
        assert refreshed["canonical_state"]["noise"]["terminal_synthesis_rows"] == 0

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        healthy_workspace(root)
        write_json(root / "autopilot.lock", {"pid": 12345, "session": "alive"})
        with patch.object(helper, "process_alive", return_value=True):
            alive = helper.repair_stale_autopilot_lock(root)
        assert alive["repaired"] is False
        assert alive["reason"] == "lock owner still alive"
        assert (root / "autopilot.lock").exists()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        healthy_workspace(root)
        write_json(root / "autopilot.lock", {"pid": 999999, "session": "locked"})
        with (
            patch.object(helper, "process_alive", return_value=False),
            patch.object(helper.fcntl, "flock", side_effect=BlockingIOError),
        ):
            locked = helper.repair_stale_autopilot_lock(root)
        assert locked["repaired"] is False
        assert locked["reason"] == "lock currently owned"
        assert (root / "autopilot.lock").exists()

    print("autoresearch watchdog checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
