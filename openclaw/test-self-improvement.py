#!/usr/bin/env python3
"""Checks for OpenClaw self-improvement memory."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("openclaw_self_improvement.py")
HELPER_PATH = Path(__file__).with_name("openclaw-speed-research.py")
WRAPPER_PATH = Path(__file__).with_name("openclaw-wrapper.zsh")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_results(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.tsv").write_text(
        "timestamp\trun_id\tstatus\ttarget\thypothesis\tttft_s\tprefill_tps\tdecode_tps\twall_s\tmemory_gb\tcommit\tnotes\n"
        "2026-05-08T10:00:00+0100\tquality-review-1\tblocked\tautoresearch-quality\treviewer should catch noise\t\t\t\t\t\tabc\tverdict=needs-repair repeated synthesis\n"
        "2026-05-08T10:01:00+0100\thandoff-1\tblocked\tautoresearch-implementation-handoff\tresearch should become patch\t\t\t\t\t\tabc\tready_deterministic=0 candidates=5\n"
        "2026-05-08T10:02:00+0100\teval-1\tkeep\tautoresearch-frontier-eval\tscore loop\t\t\t\t\t\tabc\toverall=8.73 quality=7.48 handoff=7.95\n"
        "2026-05-08T10:03:00+0100\tbenchmark-1\tkeep\tdecode-sample\tbounded decode\t\t\t14.2\t6.8\t\tabc\tserver_tok_s=14.3 mean_accept=0.8\n",
        encoding="utf-8",
    )


def main() -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    sim = load_module(MODULE_PATH, "openclaw_self_improvement_test")
    helper = load_module(HELPER_PATH, "openclaw_speed_research_test")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "research" / "speed"
        write_results(root)

        sim.ensure_self_improvement_state(root)
        status = sim.status(root)
        assert status["ok"] is True
        assert set(status["skills"]) == {
            "decode-speed-research",
            "implementation-gate",
            "reviewer-quality",
            "self-improvement-curator",
        }
        assert (root / "self-improvement" / "lessons.jsonl").exists()
        assert (root / "self-improvement" / "trajectories.jsonl").exists()
        assert (root / "self-improvement" / "evolution-proposals.jsonl").exists()
        assert (root / "self-improvement" / "evolution-eval-cases.jsonl").exists()
        assert (root / "self-improvement" / "evolution-variants.jsonl").exists()
        assert (root / "self-improvement" / "evolution-decisions.jsonl").exists()
        assert (root / "self-improvement" / "evolution-shadow-reviews.jsonl").exists()
        assert (root / "self-improvement" / "evolution-promotions.jsonl").exists()
        assert (root / "self-improvement" / "evolution-rollbacks.jsonl").exists()
        assert (root / "self-improvement" / "usage.json").exists()
        assert (root / "self-improvement" / "curator-state.json").exists()

        lessons = sim.derive_lessons(root, recent_rows=20)
        assert len(lessons) == 4
        assert {lesson.lane for lesson in lessons} == {
            "decode-speed-research",
            "implementation-gate",
            "reviewer-quality",
            "system-eval",
        }

        recorded = sim.record_lessons(root, lessons)
        assert recorded["added"] == 4
        assert sim.record_lessons(root, lessons)["added"] == 0
        repeated = sim.build_lesson(
            source="different-run-id",
            lane="reviewer-quality",
            finding="Quality review still finds repair-worthy noise, so reviewer criteria should drive the next queue route.",
            evidence="new evidence from a later cycle",
            next_action="Prefer a concrete prerequisite, implementation handoff, or lane retirement over another generic synthesis.",
            confidence=0.4,
            tags=["noise", "reviewer", "quality"],
        )
        assert repeated.id == lessons[0].id
        assert sim.record_lessons(root, [repeated])["added"] == 0

        summary = sim.curate(root, recent_rows=20)
        assert summary["derived"] == 4
        assert summary["recorded"]["added"] == 0
        assert summary["trajectories"]["added"] == 4
        assert summary["proposals"]["added"] == 3
        assert summary["usage"]["updated"] == 4
        assert summary["snapshot"]
        assert Path(summary["snapshot"]).exists()
        assert (Path(summary["snapshot"]) / "manifest.json").exists()
        reviewer_lessons = (
            root / "self-improvement" / "skills" / "reviewer-quality" / "references" / "lessons.md"
        ).read_text(encoding="utf-8")
        assert "Frontier score should be treated as a routing signal" in reviewer_lessons
        assert "Quality review still finds repair-worthy noise" in reviewer_lessons

        state = json.loads((root / "self-improvement" / "curator-state.json").read_text(encoding="utf-8"))
        assert state["run_count"] == 1
        assert state["last_health"]["trajectory_total"] == 4
        assert state["last_health"]["proposal_total"] == 3
        usage = json.loads((root / "self-improvement" / "usage.json").read_text(encoding="utf-8"))
        assert usage["self-improvement-curator"]["use_count"] == 1
        trajectories = sim.read_jsonl(root / "self-improvement" / "trajectories.jsonl")
        assert {item["target"] for item in trajectories} == {
            "autoresearch-quality",
            "autoresearch-implementation-handoff",
            "autoresearch-frontier-eval",
            "decode-sample",
        }
        proposals = sim.read_jsonl(root / "self-improvement" / "evolution-proposals.jsonl")
        assert {item["target"] for item in proposals} == {
            "decode-speed-research",
            "reviewer-quality",
            "implementation-gate",
        }

        active_skill_before = (
            root / "self-improvement" / "skills" / "reviewer-quality" / "SKILL.md"
        ).read_text(encoding="utf-8")
        evolution = sim.run_evolution(
            root,
            recent_rows=20,
            max_variants_per_skill=2,
            min_score=90,
            shadow_min_score=90,
            stage_min_wins=1,
        )
        assert evolution["ok"] is True
        assert evolution["promotion"] == "manual-review-only"
        assert evolution["active_skill_mutated"] is False
        assert evolution["cases"]["total"] >= 4
        assert evolution["variants"]["generated"] >= 4
        assert evolution["decisions"]["held_for_review"] >= 1
        assert evolution["decisions"]["rejected"] == 0
        assert evolution["shadow_reviews"]["generated"] >= 1
        assert evolution["shadow_reviews"]["wins"] >= 1
        assert evolution["staged_authority"]["generated"] >= 1
        assert evolution["rollbacks"]["generated"] >= 1
        assert "quality-review-blocked" in evolution["rollbacks"]["reasons"]
        active_skill_after = (
            root / "self-improvement" / "skills" / "reviewer-quality" / "SKILL.md"
        ).read_text(encoding="utf-8")
        assert active_skill_after == active_skill_before
        eval_cases = sim.read_jsonl(root / "self-improvement" / "evolution-eval-cases.jsonl")
        assert {item["target_skill"] for item in eval_cases} >= {
            "decode-speed-research",
            "implementation-gate",
            "reviewer-quality",
        }
        decisions = sim.read_jsonl(root / "self-improvement" / "evolution-decisions.jsonl")
        assert all(item["active_skill_mutated"] is False for item in decisions)
        held = [item for item in decisions if item["decision"] == "hold_for_review"]
        assert held
        for item in held:
            manifest = Path(item["canary_path"]) / "manifest.json"
            assert manifest.exists()
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            assert payload["active_skill_mutated"] is False
            assert payload["promotion"] == "manual-review-only"
        shadow_reviews = sim.read_jsonl(root / "self-improvement" / "evolution-shadow-reviews.jsonl")
        assert all(item["active_skill_mutated"] is False for item in shadow_reviews)
        assert all(item["effective_authority"] == "none" for item in shadow_reviews)
        promotions = sim.read_jsonl(root / "self-improvement" / "evolution-promotions.jsonl")
        assert all(item["active_skill_mutated"] is False for item in promotions)
        assert {item["effective_authority"] for item in promotions} <= {"advisory", "shadow", "canary", "none"}
        rollbacks = sim.read_jsonl(root / "self-improvement" / "evolution-rollbacks.jsonl")
        assert rollbacks
        assert all(item["active_skill_mutated"] is False for item in rollbacks)
        second_evolution = sim.run_evolution(
            root,
            recent_rows=20,
            max_variants_per_skill=2,
            min_score=90,
            shadow_min_score=90,
            stage_min_wins=2,
        )
        assert second_evolution["shadow_reviews"]["generated"] >= 1
        assert second_evolution["staged_authority"]["advisory"] >= 1
        assert second_evolution["active_skill_mutated"] is False
        status_after_evolution = sim.status(root)
        assert status_after_evolution["shadow_reviews"] >= len(shadow_reviews)
        assert status_after_evolution["promotions"] >= len(promotions)
        assert status_after_evolution["rollbacks"] >= len(rollbacks)
        assert status_after_evolution["rolled_back_promotions"] >= 1
        write_json = lambda path, payload: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text(  # noqa: E731
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )
        write_json(
            root / "benchmarks" / "quality-review-1.json",
            {
                "kind": "quality-review",
                "verdict": "healthy",
                "quality_score": 100,
                "scorecard": {"overall": 99.5, "interpretation": "high_quality_exhaustion_or_prerequisite_route"},
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
        write_json(root / "benchmarks" / "implementation-handoff-audit-1.json", {"ok": True, "score": 100})
        write_json(root / "benchmarks" / "frontier-autonomy-score-1.json", {"ok": True, "total_score": 100})
        write_json(
            root / "benchmarks" / "stability-burn-in-1.json",
            {"ok": True, "gates": {"replay_ok": True, "zero_active_noise": True}},
        )
        helper.write_jsonl(
            root / "tasks.jsonl",
            [
                {
                    "id": "alive-eval-deterministic-next-action",
                    "status": "ready",
                    "task_type": "supervisor",
                    "lane": "reviewer-quality",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research quality-review --allow-fail",
                }
            ],
        )
        (root / "results.tsv").write_text(helper.RESULTS_HEADER, encoding="utf-8")
        helper.append_result(
            root,
            run_id="frontier-system-eval-1",
            status="keep",
            target="autoresearch-frontier-eval",
            hypothesis="certified checkpoint",
            commit="unit",
            notes="overall=10 readiness=frontier",
        )
        alive = helper.self_improvement_alive_report(root, recent_rows=120)
        assert alive["ok"] is True, alive
        assert alive["total_score"] >= 95
        assert alive["readiness"] == "frontier-alive"
        assert alive["verdict"] == "certified"
        assert alive["gates"]["evidence_requirements_complete"] is True
        assert all(section["passed"] for section in alive["evidence_requirements"].values())
        assert alive["evidence_requirements"]["observe"]["evidence"]["quality"]
        assert alive["evidence_requirements"]["diagnose"]["evidence"]["autonomy"]
        assert alive["evidence_requirements"]["route_and_repair"]["evidence"]["deterministic_ready_tasks"]
        assert alive["gates"]["no_active_skill_mutation"] is True
        assert alive["components"]["evolve"] == 20

        bad_variant = {
            "id": "bad",
            "target_skill": "reviewer-quality",
            "body": "No constraints, no gates, too vague.",
            "case_ids": [],
            "constraints": [],
        }
        bad_score = sim.score_skill_variant(root, bad_variant, eval_cases)
        assert bad_score["decision"] == "reject"
        assert bad_score["blockers"]

        with patch.object(helper, "load_self_improvement_module", side_effect=RuntimeError("missing")):
            helper.ensure_optional_self_improvement_state(root)

        with patch.dict(os.environ, {"OPENCLAW_SPEED_RESEARCH_DIR": str(root)}, clear=False):
            assert helper.self_improve(type("Args", (), {"action": "status", "recent_rows": 20})()) == 0
            assert helper.self_improve(type("Args", (), {"action": "derive-lessons", "recent_rows": 20})()) == 0
            assert helper.self_improve(type("Args", (), {"action": "curate", "recent_rows": 20})()) == 0
            assert helper.self_improve(
                type(
                    "Args",
                    (),
                    {
                        "action": "evolve",
                        "recent_rows": 20,
                        "max_variants_per_skill": 1,
                        "min_score": 90,
                        "shadow_min_score": 90,
                        "stage_min_wins": 1,
                        "stage_max_effective_authority": "advisory",
                    },
                )()
            ) == 0

        wrapper = WRAPPER_PATH.read_text(encoding="utf-8")
        assert "speed-research-self-improve|research-speed-self-improve" in wrapper
        assert "setup|prompt|benchmark|record|synthesize|compact|self-improve" in wrapper

    print("self-improvement checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
