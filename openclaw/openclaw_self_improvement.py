#!/usr/bin/env python3
"""Hermes-style self-improvement memory for OpenClaw autoresearch.

This module intentionally avoids live model/runtime mutation. It turns the
research ledger into durable procedural memory, snapshots that memory before
curation, and keeps source changes behind the existing patch/canary gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SELF_DIR = "self-improvement"
LESSONS = "lessons.jsonl"
CURATOR_STATE = "curator-state.json"
USAGE = "usage.json"
TRAJECTORIES = "trajectories.jsonl"
PROPOSALS = "evolution-proposals.jsonl"
EVAL_CASES = "evolution-eval-cases.jsonl"
VARIANTS = "evolution-variants.jsonl"
DECISIONS = "evolution-decisions.jsonl"
SHADOW_REVIEWS = "evolution-shadow-reviews.jsonl"
PROMOTIONS = "evolution-promotions.jsonl"
ROLLBACKS = "evolution-rollbacks.jsonl"
SKILLS_DIR = "skills"
SNAPSHOTS_DIR = "snapshots"
CANARIES_DIR = "evolution-canaries"

AUTHORITY_STAGES = ("canary", "shadow", "advisory", "task_seed", "supervisor_route")
DEFAULT_MAX_EFFECTIVE_AUTHORITY = "advisory"

EVOLUTION_TARGETS = {
    "decode-speed-research",
    "implementation-gate",
    "reviewer-quality",
    "self-improvement-curator",
}

DEFAULT_SKILLS: dict[str, dict[str, str]] = {
    "decode-speed-research": {
        "description": "Procedural memory for improving OpenClaw TUI decode speed.",
        "body": (
            "# Decode Speed Research\n\n"
            "Use this skill when the active research objective is faster normal "
            "OpenClaw TUI decode for the selected model.\n\n"
            "## Procedure\n\n"
            "1. Prefer paired decode benchmarks over synthetic one-off rows.\n"
            "2. Separate server decode TPS from wall-clock/proxy contamination.\n"
            "3. Track MTP acceptance before proposing drafter changes.\n"
            "4. Retire settled knobs and move to the next bottleneck.\n"
            "5. Only promote if speed improves and stream/tool/reasoning guards pass.\n"
        ),
    },
    "implementation-gate": {
        "description": "Procedural memory for safe OpenClaw autoresearch patches.",
        "body": (
            "# Implementation Gate\n\n"
            "Use this skill when research evidence should become a patch.\n\n"
            "## Procedure\n\n"
            "1. Name the exact source files and failure mode.\n"
            "2. Prefer the smallest canary-only patch.\n"
            "3. Run allowlisted focused tests, not broad live model tests.\n"
            "4. Record rollback before promotion.\n"
            "5. Block architectural changes unless explicitly approved.\n"
        ),
    },
    "reviewer-quality": {
        "description": "Procedural memory for detecting noisy or fake research progress.",
        "body": (
            "# Reviewer Quality\n\n"
            "Use this skill when judging whether an autoresearch cycle made real "
            "progress.\n\n"
            "## Procedure\n\n"
            "1. Reject duplicate synthesis that adds no new evidence.\n"
            "2. Reward prerequisite routing and lane retirement when a lane is blocked.\n"
            "3. Require actionable next moves, not generic benchmark repetition.\n"
            "4. Prefer explicit blockers over vague healthy narration.\n"
            "5. Keep decode TPS as the primary metric unless the user changes scope.\n"
        ),
    },
    "self-improvement-curator": {
        "description": "Sidecar memory for stable OpenClaw self-improvement.",
        "body": (
            "# Self-Improvement Curator\n\n"
            "Use this skill when maintaining OpenClaw's own autoresearch memory.\n\n"
            "## Procedure\n\n"
            "1. Treat memory as evidence, not authority.\n"
            "2. Capture trajectories before proposing changes.\n"
            "3. Prefer skill/rubric updates before source-code changes.\n"
            "4. Require constraints, acceptance tests, and rollback for every proposal.\n"
            "5. Never mutate live model/runtime paths from the sidecar.\n"
        ),
    },
}


@dataclass(frozen=True)
class Lesson:
    id: str
    created_at: str
    source: str
    lane: str
    finding: str
    evidence: str
    next_action: str
    confidence: float
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "source": self.source,
            "lane": self.lane,
            "finding": self.finding,
            "evidence": self.evidence,
            "next_action": self.next_action,
            "confidence": self.confidence,
            "tags": list(self.tags),
        }


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def self_root(root: Path) -> Path:
    return root / SELF_DIR


def skill_dir(root: Path, name: str) -> Path:
    return self_root(root) / SKILLS_DIR / name


def lesson_id(*parts: object) -> str:
    raw = "\n".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def ensure_self_improvement_state(root: Path) -> None:
    base = self_root(root)
    (base / SKILLS_DIR).mkdir(parents=True, exist_ok=True)
    (base / SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
    for name in (
        LESSONS,
        TRAJECTORIES,
        PROPOSALS,
        EVAL_CASES,
        VARIANTS,
        DECISIONS,
        SHADOW_REVIEWS,
        PROMOTIONS,
        ROLLBACKS,
    ):
        if not (base / name).exists():
            (base / name).write_text("", encoding="utf-8")
    usage_path = base / USAGE
    if not usage_path.exists():
        write_json(usage_path, {})
    state_path = base / CURATOR_STATE
    if not state_path.exists():
        write_json(
            state_path,
            {
                "version": 1,
                "paused": False,
                "run_count": 0,
                "last_run_at": None,
                "last_summary": None,
                "last_health": None,
            },
        )
    for name, spec in DEFAULT_SKILLS.items():
        write_skill_if_missing(root, name, spec["description"], spec["body"])


def write_skill_if_missing(root: Path, name: str, description: str, body: str) -> None:
    path = skill_dir(root, name) / "SKILL.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(description)[1:-1]}\n"
        "version: 1.0.0\n"
        "metadata:\n"
        "  openclaw:\n"
        "    self_improvement: true\n"
        "---\n\n"
        f"{body.rstrip()}\n",
        encoding="utf-8",
    )


def parse_results(path: Path, limit: int = 160) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= 1:
        return []
    recent = lines[-limit:]
    if not recent[0].startswith("timestamp\t"):
        recent = [lines[0], *recent]
    reader = csv.DictReader(recent, delimiter="\t")
    return [dict(row) for row in reader if row]


def parse_note_fields(notes: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in str(notes or "").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key:
            fields[key.strip()] = value.strip().strip(",")
    return fields


def existing_lesson_ids(root: Path) -> set[str]:
    return {str(row.get("id", "")) for row in read_jsonl(self_root(root) / LESSONS)}


def bump_skill_usage(root: Path, skill_names: list[str], reason: str) -> dict[str, Any]:
    ensure_self_improvement_state(root)
    path = self_root(root) / USAGE
    usage = read_json(path, {})
    if not isinstance(usage, dict):
        usage = {}
    timestamp = now()
    for name in sorted(set(skill_names)):
        if name not in DEFAULT_SKILLS:
            continue
        record = usage.get(name)
        if not isinstance(record, dict):
            record = {"created_at": timestamp, "use_count": 0}
        record["use_count"] = int(record.get("use_count") or 0) + 1
        record["last_used_at"] = timestamp
        record["last_reason"] = reason[:160]
        record["state"] = record.get("state") or "active"
        usage[name] = record
    write_json(path, usage)
    return {"updated": len(skill_names), "skills": sorted(set(skill_names))}


def build_lesson(source: str, lane: str, finding: str, evidence: str, next_action: str, confidence: float, tags: list[str]) -> Lesson:
    return Lesson(
        id=lesson_id(lane, finding, next_action, ",".join(sorted(set(tags)))),
        created_at=now(),
        source=source,
        lane=lane,
        finding=finding,
        evidence=evidence[:700],
        next_action=next_action,
        confidence=max(0.0, min(1.0, float(confidence))),
        tags=tuple(sorted(set(tags))),
    )


def latest_rows_by_target(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        target = row.get("target") or row.get("run_id") or "unknown"
        latest[target] = row
    return latest


def capture_trajectory_cases(root: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Persist compact research trajectories for later GEPA-style reflection."""
    ensure_self_improvement_state(root)
    targets = {
        "autoresearch-quality",
        "autoresearch-frontier-eval",
        "autoresearch-implementation-handoff",
        "decode-sample",
        "mtp-acceptance-report",
    }
    existing = read_jsonl(self_root(root) / TRAJECTORIES)
    seen = {str(item.get("id", "")) for item in existing}
    latest = latest_rows_by_target([row for row in rows if row.get("target") in targets])
    added = 0
    timestamp = now()
    for target, row in latest.items():
        fields = parse_note_fields(row.get("notes", ""))
        case_id = lesson_id("trajectory", target, row.get("run_id", ""), row.get("status", ""), row.get("notes", ""))
        if case_id in seen:
            continue
        existing.append(
            {
                "id": case_id,
                "created_at": timestamp,
                "target": target,
                "run_id": row.get("run_id", ""),
                "status": row.get("status", ""),
                "hypothesis": row.get("hypothesis", ""),
                "metrics": {
                    "decode_tps": row.get("decode_tps", ""),
                    "wall_s": row.get("wall_s", ""),
                    "ttft_s": row.get("ttft_s", ""),
                    "server_tok_s": fields.get("server_tok_s", ""),
                    "mean_accept": fields.get("mean_accept", ""),
                    "quality": fields.get("quality", fields.get("score", "")),
                    "handoff": fields.get("handoff", ""),
                    "overall": fields.get("overall", ""),
                },
                "notes": row.get("notes", "")[:900],
            }
        )
        seen.add(case_id)
        added += 1
    write_jsonl(self_root(root) / TRAJECTORIES, existing)
    return {"added": added, "total": len(existing)}


def derive_lessons(root: Path, recent_rows: int = 160) -> list[Lesson]:
    ensure_self_improvement_state(root)
    rows = parse_results(root / "results.tsv", limit=recent_rows)
    lessons: list[Lesson] = []
    quality_blocked = [row for row in rows if row.get("target") == "autoresearch-quality" and row.get("status") == "blocked"]
    handoff_blocked = [row for row in rows if row.get("target") == "autoresearch-implementation-handoff" and row.get("status") == "blocked"]
    eval_rows = [row for row in rows if row.get("target") == "autoresearch-frontier-eval"]
    mtp_rows = [row for row in rows if row.get("target") in {"decode-sample", "mtp-acceptance-report"}]

    if quality_blocked:
        latest = quality_blocked[-1]
        lessons.append(
            build_lesson(
                source=latest.get("run_id", "quality-review"),
                lane="reviewer-quality",
                finding="Quality review still finds repair-worthy noise, so reviewer criteria should drive the next queue route.",
                evidence=latest.get("notes", ""),
                next_action="Prefer a concrete prerequisite, implementation handoff, or lane retirement over another generic synthesis.",
                confidence=0.82,
                tags=["quality", "reviewer", "noise"],
            )
        )

    if handoff_blocked:
        latest = handoff_blocked[-1]
        lessons.append(
            build_lesson(
                source=latest.get("run_id", "implementation-handoff-audit"),
                lane="implementation-gate",
                finding="Research evidence is not consistently becoming deterministic implementation work.",
                evidence=latest.get("notes", ""),
                next_action="Seed one scoped canary-only bridge task with explicit source files, tests, acceptance, and rollback.",
                confidence=0.86,
                tags=["implementation", "handoff", "canary"],
            )
        )

    if eval_rows:
        latest = eval_rows[-1]
        notes = latest.get("notes", "")
        lessons.append(
            build_lesson(
                source=latest.get("run_id", "frontier-system-eval"),
                lane="system-eval",
                finding="Frontier score should be treated as a routing signal, not as proof that speed improved.",
                evidence=notes,
                next_action="Route the weakest subscore to a deterministic task; keep decode TPS as the primary objective.",
                confidence=0.78,
                tags=["eval", "routing", "frontier"],
            )
        )

    if mtp_rows:
        latest = mtp_rows[-1]
        notes = latest.get("notes", "")
        lessons.append(
            build_lesson(
                source=latest.get("run_id", "mtp-evidence"),
                lane="decode-speed-research",
                finding="Current decode work must distinguish measured server TPS from acceptance and wall-clock contamination.",
                evidence=notes,
                next_action="Use paired decode and MTP acceptance evidence before promoting drafter or runtime changes.",
                confidence=0.8,
                tags=["decode", "mtp", "measurement"],
            )
        )

    return lessons


def proposal_id(target: str, hypothesis: str, constraints: list[str]) -> str:
    return lesson_id("proposal", target, hypothesis, ",".join(sorted(constraints)))


def derive_evolution_proposals(root: Path, lessons: list[Lesson], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Create constrained candidate improvements without applying them."""
    proposals: list[dict[str, Any]] = []
    timestamp = now()
    latest = latest_rows_by_target(rows)
    eval_row = latest.get("autoresearch-frontier-eval", {})
    eval_fields = parse_note_fields(eval_row.get("notes", ""))
    decode_rows = [row for row in rows if row.get("target") == "decode-sample"]
    latest_decode = decode_rows[-1] if decode_rows else {}
    decode_notes = parse_note_fields(latest_decode.get("notes", ""))
    constraints = [
        "no live model/backend mutation",
        "must pass focused tests",
        "must preserve OpenClaw-only boundary",
        "must keep rollback path",
    ]
    if latest_decode:
        proposals.append(
            {
                "id": proposal_id("decode-speed-research", "Improve drafter-alignment evidence before runtime changes.", constraints),
                "created_at": timestamp,
                "tier": 1,
                "target": "decode-speed-research",
                "status": "candidate",
                "hypothesis": "Improve drafter-alignment evidence before runtime changes.",
                "evidence": latest_decode.get("notes", "")[:900],
                "metrics": {
                    "decode_tps": latest_decode.get("decode_tps", ""),
                    "server_tok_s": decode_notes.get("server_tok_s", ""),
                    "mean_accept": decode_notes.get("mean_accept", ""),
                },
                "constraints": constraints + [
                    "promotion requires paired decode TPS improvement",
                    "promotion requires stream/tool/reasoning guards",
                ],
                "next_action": "Use JANQ trace and MTP acceptance evidence to select the next bounded drafter-fit task.",
                "rollback": "Do not promote any drafter/runtime change; discard proposal if paired benchmarks do not improve.",
            }
        )
    if eval_row:
        proposals.append(
            {
                "id": proposal_id("reviewer-quality", "Use frontier eval gaps as reviewer routing cases.", constraints),
                "created_at": timestamp,
                "tier": 1,
                "target": "reviewer-quality",
                "status": "candidate",
                "hypothesis": "Use frontier eval gaps as reviewer routing cases.",
                "evidence": eval_row.get("notes", "")[:900],
                "metrics": {
                    "overall": eval_fields.get("overall", ""),
                    "quality": eval_fields.get("quality", ""),
                    "handoff": eval_fields.get("handoff", ""),
                },
                "constraints": constraints + ["only update skill/rubric memory unless a deterministic patch task exists"],
                "next_action": "Convert repeated eval gaps into one reviewer lesson or one canary-only bridge task.",
                "rollback": "Remove the proposal or generated skill reference entry; do not touch runtime code.",
            }
        )
    for lesson in lessons:
        if lesson.lane == "implementation-gate":
            proposals.append(
                {
                    "id": proposal_id("implementation-gate", lesson.finding, constraints),
                    "created_at": timestamp,
                    "tier": 1,
                    "target": "implementation-gate",
                    "status": "candidate",
                    "hypothesis": lesson.finding,
                    "evidence": lesson.evidence,
                    "metrics": {},
                    "constraints": constraints + ["architectural changes require explicit approval"],
                    "next_action": lesson.next_action,
                    "rollback": "Revert only the canary patch or leave the task unpromoted.",
                }
            )
    return proposals


def record_evolution_proposals(root: Path, proposals: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_self_improvement_state(root)
    path = self_root(root) / PROPOSALS
    rows = read_jsonl(path)
    seen = {str(row.get("id", "")) for row in rows}
    added = 0
    for proposal in proposals:
        if proposal["id"] in seen:
            continue
        rows.append(proposal)
        seen.add(proposal["id"])
        added += 1
    write_jsonl(path, rows)
    return {"added": added, "total": len(rows)}


def skill_body(root: Path, name: str) -> str:
    path = skill_dir(root, name) / "SKILL.md"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def skill_hash(root: Path, name: str) -> str:
    return hashlib.sha256(skill_body(root, name).encode("utf-8")).hexdigest()


def case_target_for_row(row: dict[str, Any]) -> str:
    target = str(row.get("target", ""))
    status = str(row.get("status", ""))
    notes = str(row.get("notes", ""))
    if target == "decode-sample" or "server_tok_s" in notes or "mean_accept" in notes:
        return "decode-speed-research"
    if target == "autoresearch-implementation-handoff":
        return "implementation-gate"
    if target in {"autoresearch-quality", "autoresearch-frontier-eval"}:
        return "reviewer-quality"
    if status == "blocked":
        return "reviewer-quality"
    return "self-improvement-curator"


def build_evolution_eval_cases(root: Path, recent_rows: int = 160) -> list[dict[str, Any]]:
    """Create a compact eval set from real trajectories and lessons.

    These cases are intentionally small and deterministic. They let the sidecar
    compare candidate skill updates without changing the active skill files.
    """
    ensure_self_improvement_state(root)
    rows = parse_results(root / "results.tsv", limit=recent_rows)
    trajectories = read_jsonl(self_root(root) / TRAJECTORIES)
    lessons = read_jsonl(self_root(root) / LESSONS)
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_case(target: str, source: str, failure: str, expected: str, evidence: str, weight: float = 1.0) -> None:
        if target not in EVOLUTION_TARGETS:
            return
        case_id = lesson_id("eval-case", target, source, failure, expected)
        if case_id in seen:
            return
        seen.add(case_id)
        cases.append(
            {
                "id": case_id,
                "created_at": now(),
                "target_skill": target,
                "source": source[:160],
                "failure_mode": failure[:240],
                "expected_behavior": expected[:300],
                "evidence": evidence[:900],
                "weight": max(0.1, min(3.0, float(weight))),
            }
        )

    for row in rows[-recent_rows:]:
        target = case_target_for_row(row)
        status = str(row.get("status", ""))
        notes = str(row.get("notes", ""))
        run_id = str(row.get("run_id", "results-row"))
        if target == "decode-speed-research":
            add_case(
                target,
                run_id,
                "decode evidence can be contaminated or incomplete",
                "separate server TPS, wall-clock TPS, MTP acceptance, and promotion gates",
                notes,
                1.4,
            )
        elif target == "implementation-gate":
            add_case(
                target,
                run_id,
                "research did not become a deterministic safe patch",
                "require source files, canary tests, acceptance criteria, rollback, and approval gates",
                notes,
                1.5,
            )
        elif target == "reviewer-quality" and (status == "blocked" or "quality" in notes or "overall" in notes):
            add_case(
                target,
                run_id,
                "review cycle risks fake progress or repeated synthesis",
                "route to a concrete prerequisite, lane retirement, or implementation handoff",
                notes,
                1.3,
            )

    for item in trajectories[-recent_rows:]:
        target = case_target_for_row(item)
        add_case(
            target,
            str(item.get("run_id", item.get("id", "trajectory"))),
            str(item.get("status", "trajectory")),
            "apply the matching procedural skill and produce one bounded next action",
            str(item.get("notes", item.get("hypothesis", ""))),
            1.0,
        )

    for lesson in lessons[-recent_rows:]:
        target = {
            "decode-speed-research": "decode-speed-research",
            "implementation-gate": "implementation-gate",
            "reviewer-quality": "reviewer-quality",
            "system-eval": "reviewer-quality",
        }.get(str(lesson.get("lane", "")), "self-improvement-curator")
        add_case(
            target,
            str(lesson.get("source", "lesson")),
            str(lesson.get("finding", "lesson")),
            str(lesson.get("next_action", "preserve the lesson in future routing")),
            str(lesson.get("evidence", "")),
            float(lesson.get("confidence") or 0.8),
        )

    return cases


def record_evolution_eval_cases(root: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_self_improvement_state(root)
    path = self_root(root) / EVAL_CASES
    rows = read_jsonl(path)
    seen = {str(row.get("id", "")) for row in rows}
    added = 0
    for case in cases:
        if case["id"] in seen:
            continue
        rows.append(case)
        seen.add(case["id"])
        added += 1
    write_jsonl(path, rows)
    return {"added": added, "total": len(rows)}


def variant_addendum(kind: str, cases: list[dict[str, Any]]) -> str:
    by_failure = []
    for case in cases[:4]:
        by_failure.append(
            f"- When `{case['failure_mode']}`, expected behavior is: {case['expected_behavior']}"
        )
    case_text = "\n".join(by_failure) or "- No case evidence available; do not promote."
    if kind == "metric-first":
        focus = "Rank decisions by the user's current primary metric before generic health narration."
    elif kind == "failure-router":
        focus = "Route repeated failures to prerequisite, retirement, or canary implementation instead of another synthesis loop."
    else:
        focus = "Preserve the existing skill and add only evidence-backed operational checks."
    return (
        "\n\n## Evolution Canary Addendum\n\n"
        "This addendum is a canary candidate, not active runtime policy.\n\n"
        f"Focus: {focus}\n\n"
        "Evidence cases:\n"
        f"{case_text}\n\n"
        "Promotion gates:\n"
        "- replay checks pass\n"
        "- no live model/backend mutation\n"
        "- no opencode changes\n"
        "- rollback is documented\n"
        "- active skill purpose is preserved\n"
    )


def generate_skill_variants(
    root: Path,
    cases: list[dict[str, Any]],
    *,
    max_variants_per_skill: int = 2,
) -> list[dict[str, Any]]:
    ensure_self_improvement_state(root)
    timestamp = now()
    by_skill: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_skill.setdefault(str(case.get("target_skill", "")), []).append(case)

    variants: list[dict[str, Any]] = []
    kinds = ["metric-first", "failure-router", "minimal-guard"]
    for skill in sorted(EVOLUTION_TARGETS):
        skill_cases = by_skill.get(skill, [])
        if not skill_cases:
            continue
        current = skill_body(root, skill)
        if not current:
            continue
        for kind in kinds[: max(1, int(max_variants_per_skill))]:
            addendum = variant_addendum(kind, skill_cases)
            proposed = current.rstrip() + addendum + "\n"
            variant_id = lesson_id("variant", skill, kind, skill_hash(root, skill), [case["id"] for case in skill_cases[:8]])
            variants.append(
                {
                    "id": variant_id,
                    "created_at": timestamp,
                    "target_skill": skill,
                    "kind": kind,
                    "status": "candidate",
                    "base_hash": skill_hash(root, skill),
                    "case_ids": [str(case["id"]) for case in skill_cases[:8]],
                    "case_count": len(skill_cases),
                    "body": proposed,
                    "constraints": [
                        "canary-only",
                        "no active skill mutation",
                        "no live model/backend mutation",
                        "no opencode changes",
                        "size limit <= 15KB",
                        "human review before promotion",
                    ],
                    "rollback": "Discard the canary variant; active SKILL.md is untouched.",
                }
            )
    return variants


def score_skill_variant(root: Path, variant: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    body = str(variant.get("body", ""))
    skill = str(variant.get("target_skill", ""))
    blockers: list[str] = []
    warnings: list[str] = []
    if skill not in EVOLUTION_TARGETS:
        blockers.append("unknown target skill")
    if not skill_body(root, skill):
        blockers.append("missing base skill")
    if len(body.encode("utf-8")) > 15_000:
        blockers.append("variant exceeds 15KB skill limit")
    for required in ("no live model/backend mutation", "no opencode changes", "rollback"):
        if required not in body and required not in " ".join(str(item) for item in variant.get("constraints", [])):
            blockers.append(f"missing constraint: {required}")
    if "canary candidate" not in body.lower():
        blockers.append("variant does not declare canary-only status")
    case_ids = set(str(item) for item in variant.get("case_ids", []))
    covered_cases = [case for case in cases if str(case.get("id")) in case_ids]
    if not covered_cases:
        blockers.append("variant has no eval cases")
    if "Promotion gates:" not in body:
        warnings.append("promotion gates section missing")
    if "Evidence cases:" not in body:
        warnings.append("evidence cases section missing")

    score = 100
    score -= 30 * len(blockers)
    score -= 5 * len(warnings)
    score += min(8, len(covered_cases))
    score = max(0, min(100, score))
    decision = "hold_for_review" if score >= 90 and not blockers else "reject"
    return {
        "variant_id": variant.get("id"),
        "target_skill": skill,
        "score": score,
        "decision": decision,
        "blockers": blockers,
        "warnings": warnings,
        "covered_cases": len(covered_cases),
    }


def latest_decision_by_variant(root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(self_root(root) / DECISIONS):
        variant_id = str(row.get("variant_id", ""))
        if variant_id:
            latest[variant_id] = row
    return latest


def variant_metadata_by_id(root: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id", "")): row for row in read_jsonl(self_root(root) / VARIANTS) if row.get("id")}


def eval_cases_by_id(root: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id", "")): row for row in read_jsonl(self_root(root) / EVAL_CASES) if row.get("id")}


def latest_shadow_reviews_by_variant(root: Path) -> dict[str, list[dict[str, Any]]]:
    reviews: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(self_root(root) / SHADOW_REVIEWS):
        variant_id = str(row.get("variant_id", ""))
        if variant_id:
            reviews.setdefault(variant_id, []).append(row)
    return reviews


def canary_skill_body(canary_path: str, target_skill: str) -> str:
    if not canary_path:
        return ""
    path = Path(canary_path) / "skills" / target_skill / "SKILL.md"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def shadow_case_coverage_score(body: str, cases: list[dict[str, Any]], *, is_canary: bool) -> dict[str, Any]:
    text = body.lower()
    score = 55.0
    matched = 0
    missing: list[str] = []
    required_terms = {
        "decode-speed-research": ("decode", "paired", "acceptance", "promotion"),
        "implementation-gate": ("source", "canary", "rollback", "approval"),
        "reviewer-quality": ("duplicate", "evidence", "route", "actionable"),
        "self-improvement-curator": ("trajectory", "evidence", "authority", "mutation"),
    }
    for case in cases:
        target = str(case.get("target_skill", "self-improvement-curator"))
        terms = required_terms.get(target, required_terms["self-improvement-curator"])
        case_hits = sum(1 for term in terms if term in text)
        if case_hits >= max(2, len(terms) - 1):
            matched += 1
        else:
            missing.append(str(case.get("id", "")))
        score += min(8.0, case_hits * 1.4) * float(case.get("weight") or 1.0)
    if "no opencode changes" in text:
        score += 4.0
    if "no live model/backend mutation" in text:
        score += 4.0
    if "rollback" in text:
        score += 4.0
    if "promotion gates" in text:
        score += 4.0
    if is_canary and "canary candidate" in text:
        score += 4.0
    return {
        "score": round(max(0.0, min(100.0, score)), 2),
        "matched_cases": matched,
        "missing_case_ids": missing[:8],
    }


def shadow_review_variants(root: Path, *, min_score: int = 90) -> dict[str, Any]:
    """Replay held canaries against current cases without giving them authority."""
    ensure_self_improvement_state(root)
    case_map = eval_cases_by_id(root)
    variants = variant_metadata_by_id(root)
    latest_decisions = latest_decision_by_variant(root)
    existing = read_jsonl(self_root(root) / SHADOW_REVIEWS)
    seen = {str(row.get("id", "")) for row in existing}
    generated = 0
    wins = 0
    losses = 0
    timestamp = now()
    for variant_id, decision in sorted(latest_decisions.items()):
        if decision.get("decision") != "hold_for_review":
            continue
        metadata = variants.get(variant_id, {})
        target_skill = str(decision.get("target_skill") or metadata.get("target_skill") or "")
        if target_skill not in EVOLUTION_TARGETS:
            continue
        canary_body = canary_skill_body(str(decision.get("canary_path", "")), target_skill)
        stable_body = skill_body(root, target_skill)
        case_ids = [str(item) for item in metadata.get("case_ids", [])]
        cases = [case_map[case_id] for case_id in case_ids if case_id in case_map]
        if not canary_body or not stable_body or not cases:
            status = "loss"
            canary_score = 0.0
            stable_score = 100.0 if stable_body else 0.0
            missing = case_ids[:8]
        else:
            canary = shadow_case_coverage_score(canary_body, cases, is_canary=True)
            stable = shadow_case_coverage_score(stable_body, cases, is_canary=False)
            canary_score = float(canary["score"])
            stable_score = float(stable["score"])
            missing = list(canary["missing_case_ids"])
            status = "win" if canary_score >= float(min_score) and canary_score >= stable_score else "loss"
        review_id = lesson_id(
            "shadow-review",
            timestamp,
            len(existing),
            variant_id,
            decision.get("id", ""),
            canary_score,
            stable_score,
            status,
        )
        if review_id in seen:
            continue
        existing.append(
            {
                "id": review_id,
                "created_at": timestamp,
                "variant_id": variant_id,
                "target_skill": target_skill,
                "status": status,
                "shadow_score": canary_score,
                "stable_score": stable_score,
                "delta": round(canary_score - stable_score, 2),
                "min_score": int(min_score),
                "missing_case_ids": missing,
                "stage_before": "canary",
                "stage_after": "shadow" if status == "win" else "canary",
                "effective_authority": "none",
                "active_skill_mutated": False,
            }
        )
        seen.add(review_id)
        generated += 1
        if status == "win":
            wins += 1
        else:
            losses += 1
    write_jsonl(self_root(root) / SHADOW_REVIEWS, existing)
    return {"generated": generated, "wins": wins, "losses": losses, "total": len(existing)}


def authority_stage_for_wins(wins: int) -> str:
    if wins >= 4:
        return "supervisor_route"
    if wins >= 3:
        return "task_seed"
    if wins >= 2:
        return "advisory"
    if wins >= 1:
        return "shadow"
    return "canary"


def capped_authority(stage: str, max_effective_authority: str) -> str:
    if stage not in AUTHORITY_STAGES:
        return "none"
    if max_effective_authority not in AUTHORITY_STAGES:
        return "none"
    stage_index = AUTHORITY_STAGES.index(stage)
    max_index = AUTHORITY_STAGES.index(max_effective_authority)
    if stage_index > max_index:
        return max_effective_authority
    return stage


def stage_evolution_authority(
    root: Path,
    *,
    min_wins: int = 2,
    min_score: int = 90,
    max_effective_authority: str = DEFAULT_MAX_EFFECTIVE_AUTHORITY,
) -> dict[str, Any]:
    """Create non-mutating promotion records from repeated shadow wins."""
    ensure_self_improvement_state(root)
    reviews = latest_shadow_reviews_by_variant(root)
    variants = variant_metadata_by_id(root)
    promotions = read_jsonl(self_root(root) / PROMOTIONS)
    seen = {str(row.get("id", "")) for row in promotions}
    generated = 0
    advisory = 0
    timestamp = now()
    for variant_id, variant_reviews in sorted(reviews.items()):
        wins = [row for row in variant_reviews if row.get("status") == "win" and float(row.get("shadow_score") or 0) >= min_score]
        if len(wins) < int(min_wins):
            continue
        target_skill = str(variants.get(variant_id, {}).get("target_skill") or wins[-1].get("target_skill") or "")
        proposed_stage = authority_stage_for_wins(len(wins))
        effective = capped_authority(proposed_stage, max_effective_authority)
        if effective in {"task_seed", "supervisor_route"}:
            effective = "advisory"
        promotion_id = lesson_id("promotion", variant_id, len(wins), proposed_stage, effective)
        if promotion_id in seen:
            continue
        promotions.append(
            {
                "id": promotion_id,
                "created_at": timestamp,
                "variant_id": variant_id,
                "target_skill": target_skill,
                "wins": len(wins),
                "min_wins": int(min_wins),
                "min_score": int(min_score),
                "proposed_stage": proposed_stage,
                "effective_authority": effective,
                "promotion_mode": "manual-review-only",
                "approval_required": effective != "none",
                "active_skill_mutated": False,
                "rollback": "Disable this promotion record; active skills and runtime were not changed.",
            }
        )
        seen.add(promotion_id)
        generated += 1
        if effective == "advisory":
            advisory += 1
    write_jsonl(self_root(root) / PROMOTIONS, promotions)
    return {"generated": generated, "advisory": advisory, "total": len(promotions)}


def recent_unhealthy_signals(root: Path, *, recent_rows: int = 160) -> list[str]:
    reasons: list[str] = []
    for row in parse_results(root / "results.tsv", limit=recent_rows):
        target = str(row.get("target", ""))
        status = str(row.get("status", ""))
        notes = str(row.get("notes", "")).lower()
        if target == "autoresearch-quality" and status == "blocked":
            reasons.append("quality-review-blocked")
        if "memory" in notes and ("blocked" in notes or "crash" in notes or "pressure" in notes):
            reasons.append("memory-safety-signal")
        if "python" in notes and "crash" in notes:
            reasons.append("python-crash-signal")
        if "metal" in notes and ("crash" in notes or "error" in notes):
            reasons.append("metal-safety-signal")
    return sorted(set(reasons))


def rollback_unhealthy_promotions(root: Path, *, recent_rows: int = 160) -> dict[str, Any]:
    """Disable staged authority when current run health is not clean."""
    ensure_self_improvement_state(root)
    reasons = recent_unhealthy_signals(root, recent_rows=recent_rows)
    if not reasons:
        return {"generated": 0, "reasons": [], "total": len(read_jsonl(self_root(root) / ROLLBACKS))}
    promotions = read_jsonl(self_root(root) / PROMOTIONS)
    active = [row for row in promotions if row.get("effective_authority") not in {"", "none"}]
    rollbacks = read_jsonl(self_root(root) / ROLLBACKS)
    seen = {str(row.get("id", "")) for row in rollbacks}
    generated = 0
    timestamp = now()
    for promotion in active:
        rollback_id = lesson_id("rollback", promotion.get("id", ""), ",".join(reasons))
        if rollback_id in seen:
            continue
        rollbacks.append(
            {
                "id": rollback_id,
                "created_at": timestamp,
                "promotion_id": promotion.get("id", ""),
                "variant_id": promotion.get("variant_id", ""),
                "target_skill": promotion.get("target_skill", ""),
                "reason": ",".join(reasons),
                "previous_effective_authority": promotion.get("effective_authority", ""),
                "new_effective_authority": "none",
                "active_skill_mutated": False,
            }
        )
        seen.add(rollback_id)
        generated += 1
    write_jsonl(self_root(root) / ROLLBACKS, rollbacks)
    return {"generated": generated, "reasons": reasons, "total": len(rollbacks)}


def rolled_back_promotion_ids(root: Path) -> set[str]:
    return {
        str(row.get("promotion_id", ""))
        for row in read_jsonl(self_root(root) / ROLLBACKS)
        if row.get("promotion_id")
    }


def write_variant_canary(root: Path, variant: dict[str, Any], score: dict[str, Any]) -> str:
    base = self_root(root) / CANARIES_DIR / str(variant["id"])
    target = base / "skills" / str(variant["target_skill"]) / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(target, str(variant["body"]))
    write_json(
        base / "manifest.json",
        {
            "created_at": now(),
            "variant_id": variant["id"],
            "target_skill": variant["target_skill"],
            "base_hash": variant["base_hash"],
            "decision": score,
            "promotion": "manual-review-only",
            "active_skill_mutated": False,
            "rollback": variant["rollback"],
        },
    )
    return str(base)


def run_evolution(
    root: Path,
    *,
    recent_rows: int = 160,
    max_variants_per_skill: int = 2,
    min_score: int = 90,
    shadow_min_score: int = 90,
    stage_min_wins: int = 2,
    stage_max_effective_authority: str = DEFAULT_MAX_EFFECTIVE_AUTHORITY,
) -> dict[str, Any]:
    """Run canary-only skill evolution, shadow review, and staged advisory records."""
    ensure_self_improvement_state(root)
    snapshot = snapshot_self_improvement(root, reason="pre-evolution-run")
    rows = parse_results(root / "results.tsv", limit=recent_rows)
    capture_trajectory_cases(root, rows)
    lessons = derive_lessons(root, recent_rows=recent_rows)
    record_lessons(root, lessons)
    cases = build_evolution_eval_cases(root, recent_rows=recent_rows)
    case_summary = record_evolution_eval_cases(root, cases)
    variants = generate_skill_variants(root, cases, max_variants_per_skill=max_variants_per_skill)

    existing_variants = read_jsonl(self_root(root) / VARIANTS)
    existing_variant_ids = {str(row.get("id", "")) for row in existing_variants}
    existing_decisions = read_jsonl(self_root(root) / DECISIONS)
    decisions: list[dict[str, Any]] = []
    added_variants = 0
    held_for_review = 0
    rejected = 0
    for variant in variants:
        score = score_skill_variant(root, variant, cases)
        if int(score["score"]) < int(min_score):
            score["decision"] = "reject"
        if variant["id"] not in existing_variant_ids:
            existing_variants.append({key: value for key, value in variant.items() if key != "body"})
            existing_variant_ids.add(str(variant["id"]))
            added_variants += 1
        canary_path = ""
        if score["decision"] == "hold_for_review":
            canary_path = write_variant_canary(root, variant, score)
            held_for_review += 1
        else:
            rejected += 1
        decision = {
            "id": lesson_id("decision", variant["id"], score["decision"], score["score"]),
            "created_at": now(),
            "variant_id": variant["id"],
            "target_skill": variant["target_skill"],
            "decision": score["decision"],
            "score": score["score"],
            "blockers": score["blockers"],
            "warnings": score["warnings"],
            "covered_cases": score["covered_cases"],
            "canary_path": canary_path,
            "active_skill_mutated": False,
        }
        decisions.append(decision)
        if decision["id"] not in {str(row.get("id", "")) for row in existing_decisions}:
            existing_decisions.append(decision)

    write_jsonl(self_root(root) / VARIANTS, existing_variants)
    write_jsonl(self_root(root) / DECISIONS, existing_decisions)
    shadow_reviews = shadow_review_variants(root, min_score=shadow_min_score)
    staged_authority = stage_evolution_authority(
        root,
        min_wins=stage_min_wins,
        min_score=shadow_min_score,
        max_effective_authority=stage_max_effective_authority,
    )
    rollbacks = rollback_unhealthy_promotions(root, recent_rows=recent_rows)
    bump_skill_usage(root, ["self-improvement-curator"], reason="evolve")
    report = {
        "ok": True,
        "snapshot": str(snapshot) if snapshot else None,
        "cases": case_summary,
        "variants": {"generated": len(variants), "added": added_variants},
        "decisions": {
            "generated": len(decisions),
            "held_for_review": held_for_review,
            "rejected": rejected,
        },
        "shadow_reviews": shadow_reviews,
        "staged_authority": staged_authority,
        "rollbacks": rollbacks,
        "promotion": "manual-review-only",
        "active_skill_mutated": False,
        "next": "review advisory/shadow wins, then promote through patch/canary gate only if desired",
    }
    state_path = self_root(root) / CURATOR_STATE
    state = read_json(state_path, {})
    state["last_evolution"] = {**report, "ran_at": now()}
    write_json(state_path, state)
    return report


def record_lessons(root: Path, lessons: list[Lesson]) -> dict[str, Any]:
    ensure_self_improvement_state(root)
    path = self_root(root) / LESSONS
    rows = read_jsonl(path)
    seen = {str(row.get("id", "")) for row in rows}
    added = 0
    for lesson in lessons:
        if lesson.id in seen:
            continue
        rows.append(lesson.to_dict())
        seen.add(lesson.id)
        added += 1
    write_jsonl(path, rows)
    return {"added": added, "total": len(rows)}


def snapshot_self_improvement(root: Path, reason: str = "curate") -> Path | None:
    ensure_self_improvement_state(root)
    base = self_root(root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base_name = f"{stamp}-{safe_slug(reason)}"
    dest = base / SNAPSHOTS_DIR / base_name
    suffix = 1
    while dest.exists():
        suffix += 1
        dest = base / SNAPSHOTS_DIR / f"{base_name}-{suffix:02d}"
    source_paths = [
        base / LESSONS,
        base / CURATOR_STATE,
        base / USAGE,
        base / TRAJECTORIES,
        base / PROPOSALS,
        base / EVAL_CASES,
        base / VARIANTS,
        base / DECISIONS,
        base / SHADOW_REVIEWS,
        base / PROMOTIONS,
        base / ROLLBACKS,
        base / SKILLS_DIR,
    ]
    dest.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": now(), "reason": reason, "files": []}
    for source in source_paths:
        if not source.exists():
            continue
        target = dest / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        manifest["files"].append(source.name)
    write_json(dest / "manifest.json", manifest)
    return dest


def safe_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.lower()).strip("-")
    return slug[:60] or "snapshot"


def append_skill_lessons(root: Path, lessons: list[Lesson]) -> dict[str, int]:
    updates: dict[str, int] = {}
    by_lane: dict[str, list[Lesson]] = {}
    lane_to_skill = {
        "decode-speed-research": "decode-speed-research",
        "reviewer-quality": "reviewer-quality",
        "implementation-gate": "implementation-gate",
        "system-eval": "reviewer-quality",
    }
    for lesson in lessons:
        skill = lane_to_skill.get(lesson.lane)
        if not skill:
            continue
        by_lane.setdefault(skill, []).append(lesson)

    for skill, skill_lessons in by_lane.items():
        ref = skill_dir(root, skill) / "references" / "lessons.md"
        existing = ref.read_text(encoding="utf-8", errors="replace") if ref.exists() else "# Lessons\n\n"
        additions: list[str] = []
        for lesson in skill_lessons:
            marker = f"<!-- lesson:{lesson.id} -->"
            if marker in existing:
                continue
            additions.append(
                f"{marker}\n"
                f"## {lesson.created_at} - {lesson.finding}\n\n"
                f"- source: `{lesson.source}`\n"
                f"- confidence: {lesson.confidence:.2f}\n"
                f"- evidence: {lesson.evidence}\n"
                f"- next action: {lesson.next_action}\n"
                f"- tags: {', '.join(lesson.tags)}\n"
            )
        if additions:
            ref.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(ref, existing.rstrip() + "\n\n" + "\n\n".join(additions) + "\n")
            updates[skill] = len(additions)
    return updates


def curate(root: Path, recent_rows: int = 160) -> dict[str, Any]:
    ensure_self_improvement_state(root)
    rows = parse_results(root / "results.tsv", limit=recent_rows)
    lessons = derive_lessons(root, recent_rows=recent_rows)
    snapshot = snapshot_self_improvement(root, reason="pre-curator-run")
    recorded = record_lessons(root, lessons)
    updates = append_skill_lessons(root, lessons)
    used_skills = sorted(
        {
            "self-improvement-curator",
            *[
                {
                    "decode-speed-research": "decode-speed-research",
                    "implementation-gate": "implementation-gate",
                    "reviewer-quality": "reviewer-quality",
                    "system-eval": "reviewer-quality",
                }.get(lesson.lane, "self-improvement-curator")
                for lesson in lessons
            ],
        }
    )
    usage = bump_skill_usage(root, used_skills, reason="curate")
    trajectories = capture_trajectory_cases(root, rows)
    proposals = record_evolution_proposals(root, derive_evolution_proposals(root, lessons, rows))
    state_path = self_root(root) / CURATOR_STATE
    state = read_json(state_path, {})
    state.update(
        {
            "version": 1,
            "run_count": int(state.get("run_count") or 0) + 1,
            "last_run_at": now(),
            "last_summary": {
                "derived": len(lessons),
                "recorded": recorded,
                "skill_updates": updates,
                "usage": usage,
                "trajectories": trajectories,
                "proposals": proposals,
                "snapshot": str(snapshot) if snapshot else None,
            },
            "last_health": {
                "lessons_total": recorded["total"],
                "trajectory_total": trajectories["total"],
                "proposal_total": proposals["total"],
                "skill_count": len(DEFAULT_SKILLS),
            },
        }
    )
    write_json(state_path, state)
    return state["last_summary"]


def status(root: Path) -> dict[str, Any]:
    ensure_self_improvement_state(root)
    base = self_root(root)
    lessons = read_jsonl(base / LESSONS)
    trajectories = read_jsonl(base / TRAJECTORIES)
    proposals = read_jsonl(base / PROPOSALS)
    eval_cases = read_jsonl(base / EVAL_CASES)
    variants = read_jsonl(base / VARIANTS)
    decisions = read_jsonl(base / DECISIONS)
    shadow_reviews = read_jsonl(base / SHADOW_REVIEWS)
    promotions = read_jsonl(base / PROMOTIONS)
    rollbacks = read_jsonl(base / ROLLBACKS)
    usage = read_json(base / USAGE, {})
    state = read_json(base / CURATOR_STATE, {})
    skills = sorted(path.parent.name for path in (base / SKILLS_DIR).glob("*/SKILL.md"))
    held_variants = [row for row in decisions if row.get("decision") == "hold_for_review"]
    rolled_back = rolled_back_promotion_ids(root)
    advisory_promotions = [
        row
        for row in promotions
        if row.get("effective_authority") == "advisory" and str(row.get("id", "")) not in rolled_back
    ]
    return {
        "ok": True,
        "path": str(base),
        "lessons": len(lessons),
        "trajectories": len(trajectories),
        "proposals": len(proposals),
        "eval_cases": len(eval_cases),
        "variants": len(variants),
        "decisions": len(decisions),
        "held_variants": len(held_variants),
        "shadow_reviews": len(shadow_reviews),
        "promotions": len(promotions),
        "advisory_promotions": len(advisory_promotions),
        "rolled_back_promotions": len(rolled_back),
        "rollbacks": len(rollbacks),
        "usage": usage,
        "skills": skills,
        "run_count": int(state.get("run_count") or 0),
        "last_run_at": state.get("last_run_at"),
        "last_summary": state.get("last_summary"),
        "last_health": state.get("last_health"),
        "last_evolution": state.get("last_evolution"),
    }
