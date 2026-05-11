#!/usr/bin/env python3
"""Adversarial checks for frontier autonomy scoring and promotion gates."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path


HELPER_PATH = Path(__file__).with_name("openclaw-speed-research.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("openclaw_speed_research", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def seed_certified_workspace(helper, root: Path) -> None:
    assert helper.setup_workspace(Namespace(repo_url="file:///no/such/repo")) == 0
    write_json(
        root / "benchmarks" / "quality-review-1.json",
        {
            "kind": "quality-review",
            "verdict": "healthy",
            "quality_score": 100,
            "scorecard": {"overall": 99.7, "interpretation": "high_quality_exhaustion_or_prerequisite_route"},
        },
    )
    write_json(
        root / "benchmarks" / "frontier-system-eval-1.json",
        {
            "kind": "frontier-system-eval",
            "overall": 10.0,
            "readiness": "frontier",
            "frontier_certified": True,
            "canonical_state": {
                "clean": True,
                "noise": {
                    "unresolved_blocked_rows": 0,
                    "terminal_synthesis_rows": 0,
                    "bridge_zero_rows": 0,
                    "memory_blocks": 0,
                },
            },
        },
    )
    write_json(
        root / "benchmarks" / "implementation-handoff-audit-1.json",
        {"kind": "implementation-handoff-audit", "ok": True, "score": 100},
    )
    write_json(
        root / "benchmarks" / "stability-burn-in-1.json",
        {
            "kind": "stability-burn-in",
            "ok": True,
            "gates": {"replay_ok": True, "zero_active_noise": True},
            "canonical_state": {
                "clean": True,
                "noise": {
                    "unresolved_blocked_rows": 0,
                    "terminal_synthesis_rows": 0,
                    "bridge_zero_rows": 0,
                    "memory_blocks": 0,
                },
            },
        },
    )
    helper.append_result(
        root,
        run_id="frontier-system-eval-1",
        status="keep",
        target="autoresearch-frontier-eval",
        hypothesis="certified checkpoint",
        commit="unit",
        notes="overall=10 readiness=frontier",
    )


def main() -> int:
    helper = load_helper()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "research" / "speed"
        old = os.environ.get("OPENCLAW_SPEED_RESEARCH_DIR")
        os.environ["OPENCLAW_SPEED_RESEARCH_DIR"] = str(root)
        try:
            seed_certified_workspace(helper, root)
            perfect = helper.frontier_autonomy_score_report(root)
            assert perfect["total_score"] == 100
            assert perfect["hard_gate_failures"] == []

            helper.append_result(
                root,
                run_id="tool-loop-after-cert",
                status="blocked",
                target="tool-call-loop",
                hypothesis="bad behavior must block autonomy",
                commit="unit",
                notes="tool-call loop detected",
            )
            noisy = helper.frontier_autonomy_score_report(root)
            assert noisy["total_score"] < 100
            assert "no_bad_behavior_rows" in noisy["hard_gate_failures"]

            (root / "results.tsv").write_text(helper.RESULTS_HEADER, encoding="utf-8")
            helper.append_result(
                root,
                run_id="frontier-system-eval-2",
                status="keep",
                target="autoresearch-frontier-eval",
                hypothesis="fresh checkpoint clears routed historical bad behavior",
                commit="unit",
                notes="overall=10 readiness=frontier",
            )
            patch_text = (
                "diff --git a/openclaw/openclaw-model-proxy.py b/openclaw/openclaw-model-proxy.py\n"
                "--- a/openclaw/openclaw-model-proxy.py\n"
                "+++ b/openclaw/openclaw-model-proxy.py\n"
                "@@ -1 +1 @@\n"
                "-MODE = 'old'\n"
                "+MODE = 'new'\n"
            )
            classification = helper.classify_patch(
                patch_text,
                source_files=["openclaw/openclaw-model-proxy.py"],
                allow_architectural=True,
            )
            assert classification["architectural"] is True
            missing_crabbox = helper.frontier_autonomy_score_report(
                root,
                promotion=True,
                classification=classification,
                patch_tests=[{"ok": True}],
                crabbox_evidence={},
                rollback_rehearsal_ok=True,
            )
            assert "crabbox_evidence_complete" in missing_crabbox["hard_gate_failures"]
            missing_canary = helper.frontier_autonomy_score_report(
                root,
                promotion=True,
                classification=classification,
                patch_tests=None,
                crabbox_evidence={},
                rollback_rehearsal_ok=True,
            )
            assert "patch_tests_complete" in missing_canary["hard_gate_failures"]

            evidence_path = root / "sandbox-runs" / "crabbox-ok.json"
            write_json(
                evidence_path,
                {
                    "ok": True,
                    "runner": "static-ssh-mac",
                    "patch_sha256": helper.text_sha256(patch_text),
                    "tests": [{"ok": True, "command": "python3 openclaw/test-speed-research.py"}],
                    "full_suite": {"ok": True},
                    "rollback_rehearsal_ok": True,
                    "run_id": "run_unit",
                },
            )
            crabbox = helper.load_crabbox_evidence(str(evidence_path), patch_sha256=helper.text_sha256(patch_text))
            assert crabbox["_valid_for_architectural_promotion"] is True
            promoted = helper.frontier_autonomy_score_report(
                root,
                promotion=True,
                classification=classification,
                patch_tests=[{"ok": True}],
                crabbox_evidence=crabbox,
                rollback_rehearsal_ok=True,
            )
            assert promoted["total_score"] == 100, promoted
            assert promoted["decision"] == "promote"
            helper.append_result(
                root,
                run_id="python-crash-after-cert",
                status="keep",
                target="runtime-stability",
                hypothesis="crash signal must fail closed even if not blocked",
                commit="unit",
                notes="python crash observed during model run",
            )
            crashy = helper.frontier_autonomy_score_report(root)
            assert "no_bad_behavior_rows" in crashy["hard_gate_failures"]
        finally:
            if old is None:
                os.environ.pop("OPENCLAW_SPEED_RESEARCH_DIR", None)
            else:
                os.environ["OPENCLAW_SPEED_RESEARCH_DIR"] = old
    print("frontier autonomy policy checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
