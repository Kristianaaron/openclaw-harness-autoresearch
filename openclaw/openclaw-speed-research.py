#!/usr/bin/env python3
"""OpenClaw speed autoresearch workspace helper.

This adapts the karpathy/autoresearch loop to OpenClaw runtime speed work:
small scoped experiments, bounded benchmarks, TSV results, and keep/discard
discipline. It intentionally does not start a model by itself.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from openclaw_speed_research_core import (
    RESULTS_HEADER,
    append_result,
    append_jsonl,
    benchmark_result_schema_ok,
    benchmark_spec,
    causal_review_report,
    decode_measurement_signal,
    ensure_research_state,
    environment_snapshot,
    evaluator_integrity_report,
    exhausted_lanes,
    gepa_escalation_report,
    gepa_policy_promotion_report,
    latest_decode_mean,
    mark_lane_exhausted,
    measurement_artifact_analysis,
    load_benchmark_manifest,
    paired_profile_plan,
    read_jsonl,
    rank_tasks,
    replay_checks,
    score_insight,
    seed_gepa_canary_task,
    suppress_stale_gepa_policy_canaries,
    task_contract_issues,
    task_contract_report,
    variance_analysis,
    write_gepa_policy_canary,
    write_jsonl,
)
DEFAULT_REPO_URL = "https://github.com/karpathy/autoresearch.git"
DEFAULT_MODEL_URL = "http://127.0.0.1:8091/v1"
DEFAULT_PROXY_LOG = "/Users/kristian/.openclaw/logs/openclaw-model-proxy.log"
DEFAULT_JANQ_TARGET_PATH = (
    "/Users/kristian/.cache/huggingface/hub/"
    "models--dealignai--Gemma-4-31B-JANG_4M-CRACK/"
    "snapshots/bb11360eacf55506f6e51eaacc6b0f65f9209b14"
)
DEFAULT_MTP_DRAFT_PATH = "/Users/kristian/.openclaw/models/gemma-4-31B-it-assistant-mlx-4bit"
DEFAULT_RUNTIME_SITE = Path.home() / ".openclaw" / "runtime" / "rapid-mlx" / "site"
DEFAULT_MTP_CALIBRATOR_SCRIPT = (
    "/Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/openclaw-mtp-drafter-calibrate.py"
)
CALIBRATION_MEMORY_STAGES = ("metadata", "drafter-load", "target-load", "combined-load", "micro-step")
CALIBRATION_QUANTIZED_GRADIENT_BLOCKER = "calibration-quantized-gradient-unsupported"
CALIBRATION_TRACE_DISTILLATION_MODE = "trace-distillation"
CALIBRATION_DIRECT_MODE = "direct-pre-projection"
CALIBRATION_ADAPTER_MODE = "adapter-logit-distillation"
CALIBRATION_MODES = {CALIBRATION_DIRECT_MODE, CALIBRATION_ADAPTER_MODE}
ADAPTER_LOGIT_LOOP_THRESHOLD = 2
CALIBRATION_CANARY_TERMINAL_BLOCKERS = {
    "calibration-runtime-missing-speculative",
    CALIBRATION_QUANTIZED_GRADIENT_BLOCKER,
}
DEFAULT_PATCH_TESTS = (
    "python3 openclaw/test-speed-research.py",
    "python3 openclaw/test-speed-research-autopilot.py",
)
FRONTIER_SOURCE_URLS = (
    "https://github.com/karpathy/autoresearch",
    "https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/GEPA/overview.md",
    "https://github.com/NousResearch/hermes-agent",
    "https://github.com/raullenchai/Rapid-MLX",
    "https://github.com/z-lab/dflash",
    "https://ai.google.dev/gemma/docs/mtp/mtp",
    "https://huggingface.co/dealignai/Gemma-4-31B-JANG_4M-CRACK",
    "https://www.reddit.com/r/LocalLLaMA/search.json?q=Gemma%204%20MTP%20drafter%20decode%20speed&restrict_sr=1&sort=new",
    "https://x.com/search?q=Gemma%204%20MTP%20drafter%20decode%20speed&src=typed_query",
)
SOURCE_SCOUT_ALLOWED_HOSTS = {
    "ai.google.dev",
    "github.com",
    "huggingface.co",
    "raw.githubusercontent.com",
    "reddit.com",
    "www.reddit.com",
    "x.com",
}
FULL_AUTONOMY_TESTS = (
    "python3 -m compileall -q openclaw",
    "python3 openclaw/test-speed-research.py",
    "python3 openclaw/test-speed-research-autopilot.py",
    "python3 openclaw/test-autoresearch-watchdog.py",
    "python3 openclaw/test-self-improvement.py",
    "python3 openclaw/test-drafter-fit.py",
    "python3 openclaw/test-mtp-drafter-calibrate-guards.py",
    "python3 openclaw/test-rapid-launcher-guards.py",
    "python3 openclaw/test-jang-vlm-server-guards.py",
    "python3 openclaw/test-vmlx-proxy-guards.py",
)
DEFAULT_AUTONOMY_POLICY: dict[str, Any] = {
    "version": 1,
    "mode": "sandbox-auto-promote",
    "crabbox_runner": "static-ssh-mac",
    "thresholds": {
        "total": 100,
        "quality_score": 99,
        "scorecard_overall": 99,
        "frontier_overall": 9.8,
        "handoff_score": 100,
    },
    "forbidden_domains": ["opencode", "model-change", "secrets", "private-config", "live-profile-mutation"],
    "architectural_requires_crabbox": True,
    "full_suite_tests": list(FULL_AUTONOMY_TESTS),
    "stable_build_registry": "stable-builds.jsonl",
}
DEFAULT_LANE_CONTRACTS: dict[str, Any] = {
    "version": 1,
    "lanes": {
        "production-mtp": {
            "purpose": "Measure and improve the normal OpenClaw TUI MTP decode path.",
            "memory_class": "live_model",
            "fallback": "decode-sample",
            "promotion_gate": "paired benchmark must improve normal TUI decode TPS and preserve stream/tool guards",
        },
        "runtime-overhead": {
            "purpose": "Separate backend decode speed from proxy, stream, tool, and watchdog overhead.",
            "memory_class": "live_model_or_read_only",
            "fallback": "decode-sample",
            "promotion_gate": "artifact must name a patchable boundary before source changes are proposed",
        },
        "drafter-alignment": {
            "purpose": "Fit the drafter to the JANQ target without changing the target model.",
            "memory_class": "bounded_training",
            "fallback": "trace-prerequisite-or-decode-sample",
            "hard_blockers": [
                "calibration-memory-after-load",
                "calibration-runtime-missing-speculative",
                CALIBRATION_QUANTIZED_GRADIENT_BLOCKER,
            ],
            "promotion_gate": "target-generated traces, calibration canary, acceptance lift, and paired decode benchmark",
        },
        "frontier-dflash": {
            "purpose": "Evaluate DFlash compatibility only after structural JANQ compatibility is proven.",
            "memory_class": "no_model_load_until_compatible",
            "fallback": "runtime-overhead",
            "hard_blockers": [
                "draft_model_type_mismatch",
                "dflash_draft_config_missing",
                "draft_target_layer_ids_missing",
                "lane-exhausted",
            ],
            "promotion_gate": "compatibility artifact, canary, loop guards, then paired TUI decode improvement",
        },
        "mtp-decode": {
            "purpose": "Bounded MTP sweep lane; retire once block-size evidence converges.",
            "memory_class": "live_model",
            "fallback": "production-mtp",
            "promotion_gate": "winner must beat control outside observed variance",
        },
        "implementation-gate": {
            "purpose": "Convert evidence into canary-only source patches with rollback.",
            "memory_class": "no_model_load",
            "fallback": "deterministic-bridge",
            "promotion_gate": "allowed paths, py_compile/tests, no opencode or secret files, rollback recorded",
        },
        "frontier-expansion": {
            "purpose": "Generate the next bounded candidate path after active speed lanes exhaust.",
            "memory_class": "no_model_load",
            "fallback": "implementation-gate",
            "promotion_gate": "one canary-only candidate with acceptance, rollback, and no live profile mutation",
        },
    },
}
DEFAULT_TRACE_PROMPTS = (
    (
        "coding-small",
        "In one concise paragraph, explain how to reduce repeated tool-call loops in a local coding agent.",
    ),
    (
        "speed-small",
        "Give a short implementation note for improving perceived latency in a local MLX model server.",
    ),
    (
        "tool-json-small",
        "Return a compact JSON object with keys diagnosis, safe_next_step, and rollback for a stalled benchmark loop.",
    ),
    (
        "reasoning-guard-small",
        "Summarize how to keep model reasoning separate from tool JSON in a terminal agent harness.",
    ),
    (
        "memory-small",
        "Write a short checklist for avoiding macOS memory pressure while testing a 31B local model.",
    ),
    (
        "research-small",
        "Propose one measurable experiment for improving decode tokens per second without changing the target model.",
    ),
)
ALLOWED_PATCH_PREFIXES = (
    "openclaw/",
    "docs/case-studies/",
)
ARCHITECTURAL_PATCH_PREFIXES = (
    "openclaw/openclaw-wrapper.zsh",
    "openclaw/openclaw-model-proxy.py",
    "openclaw/openclaw-jang-vlm-server.py",
    "openclaw/openclaw-jang-vlm-launcher.py",
    "openclaw/model-profiles",
)
HIGH_RISK_PATCH_PREFIXES = (
    *ARCHITECTURAL_PATCH_PREFIXES,
    "openclaw/openclaw-speed-research-autopilot.py",
    "openclaw/openclaw-autoresearch-watchdog.py",
    "openclaw/openclaw-mtp-drafter-calibrate.py",
    "openclaw/openclaw-drafter-fit.py",
    "openclaw/openclaw-rapid-mlx-launcher.py",
)
DENIED_PATCH_FRAGMENTS = (
    ".env",
    "opencode",
    "token",
    "secret",
    "password",
    "id_rsa",
    ".pem",
    ".key",
)
SECRET_PATCH_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)github_pat_[A-Za-z0-9_]{20,}"),
)


def home() -> Path:
    return Path(os.environ.get("OPENCLAW_HOME", Path.home() / ".openclaw")).expanduser()


def workspace_root() -> Path:
    return Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_DIR", home() / "research" / "speed")).expanduser()


def run(argv: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def calibration_python() -> str:
    explicit = os.environ.get("OPENCLAW_CALIBRATION_PYTHON") or os.environ.get("OPENCLAW_RAPID_PYTHON")
    if explicit:
        return explicit
    for candidate in (
        Path("/opt/homebrew/opt/rapid-mlx/libexec/bin/python"),
        Path("/opt/homebrew/opt/rapid-mlx/libexec/bin/python3.12"),
    ):
        if candidate.exists():
            return str(candidate)
    for candidate in sorted(Path("/opt/homebrew/Cellar/rapid-mlx").glob("*/libexec/bin/python"), reverse=True):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def calibration_python_env() -> dict[str, str]:
    env = os.environ.copy()
    runtime_site = Path(os.environ.get("OPENCLAW_JANG_TARGET", DEFAULT_RUNTIME_SITE)).expanduser()
    pythonpath = str(runtime_site)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return env


def calibration_runtime_import_issue(python_bin: str) -> str:
    code = (
        "import importlib; "
        "importlib.import_module('mlx'); "
        "importlib.import_module('jang_tools.loader'); "
        "importlib.import_module('mlx_vlm.speculative.drafters')"
    )
    try:
        result = subprocess.run(
            [python_bin, "-c", code],
            env=calibration_python_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"calibration-runtime-check-failed:{type(error).__name__}"
    if result.returncode == 0:
        return ""
    tail = result.stdout[-500:].replace("\n", " ").strip()
    if "mlx_vlm.speculative" in tail:
        return "missing-runtime-module:mlx_vlm.speculative.drafters"
    if "jang_tools" in tail:
        return "missing-runtime-module:jang_tools.loader"
    if "No module named 'mlx'" in tail or 'No module named "mlx"' in tail:
        return "missing-runtime-module:mlx"
    return f"calibration-runtime-import-exit:{result.returncode}:{tail}"


def git_available() -> bool:
    try:
        run(["git", "--version"])
        return True
    except Exception:
        return False


def write_if_missing(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def latest_json_artifact(root: Path, pattern: str) -> dict[str, Any]:
    """Read the newest benchmark artifact matching a pattern."""
    paths = list((root / "benchmarks").glob(pattern))
    if not paths:
        return {}
    latest = max(paths, key=lambda item: item.stat().st_mtime_ns)
    try:
        loaded = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    loaded["_artifact_path"] = str(latest)
    return loaded


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def load_autonomy_policy(root: Path) -> dict[str, Any]:
    path = root / "autonomy-policy.json"
    if not path.exists():
        write_if_missing(path, json.dumps(DEFAULT_AUTONOMY_POLICY, indent=2, sort_keys=True) + "\n")
        return dict(DEFAULT_AUTONOMY_POLICY)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_AUTONOMY_POLICY)
    if not isinstance(loaded, dict):
        return dict(DEFAULT_AUTONOMY_POLICY)
    merged = dict(DEFAULT_AUTONOMY_POLICY)
    merged.update(loaded)
    thresholds = dict(DEFAULT_AUTONOMY_POLICY["thresholds"])
    if isinstance(loaded.get("thresholds"), dict):
        thresholds.update(loaded["thresholds"])
    merged["thresholds"] = thresholds
    return merged


def ensure_lane_contracts(root: Path) -> dict[str, Any]:
    """Keep lane routing explicit so the loop has a deterministic fallback."""
    path = root / "lane-contracts.json"
    if not path.exists():
        write_if_missing(path, json.dumps(DEFAULT_LANE_CONTRACTS, indent=2, sort_keys=True) + "\n")
        return DEFAULT_LANE_CONTRACTS
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        loaded = {}
    if not isinstance(loaded, dict) or not isinstance(loaded.get("lanes"), dict):
        write_if_changed(path, json.dumps(DEFAULT_LANE_CONTRACTS, indent=2, sort_keys=True) + "\n")
        return DEFAULT_LANE_CONTRACTS
    changed = False
    lanes = loaded.setdefault("lanes", {})
    for lane, contract in DEFAULT_LANE_CONTRACTS["lanes"].items():
        if lane not in lanes:
            lanes[lane] = contract
            changed = True
            continue
        existing_contract = lanes.get(lane)
        if not isinstance(existing_contract, dict):
            lanes[lane] = contract
            changed = True
            continue
        for key, value in contract.items():
            if key not in existing_contract:
                existing_contract[key] = value
                changed = True
            elif key == "hard_blockers" and isinstance(existing_contract.get(key), list) and isinstance(value, list):
                for blocker in value:
                    if blocker not in existing_contract[key]:
                        existing_contract[key].append(blocker)
                        changed = True
    if changed:
        loaded.setdefault("version", DEFAULT_LANE_CONTRACTS["version"])
        write_if_changed(path, json.dumps(loaded, indent=2, sort_keys=True) + "\n")
    return loaded


def ensure_section(path: Path, marker: str, section: str) -> None:
    if not path.exists():
        path.write_text(section, encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")


def upsert_section(path: Path, heading: str, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = section.strip() + "\n"
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        path.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
        return
    next_start = text.find("\n## ", start + 1)
    if next_start < 0:
        next_text = text[:start].rstrip() + "\n\n" + section
    else:
        next_text = text[:start].rstrip() + "\n\n" + section + "\n" + text[next_start + 1 :].lstrip()
    if next_text != text:
        path.write_text(next_text, encoding="utf-8")


def remove_section(path: Path, heading: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return
    next_start = text.find("\n## ", start + 1)
    if next_start < 0:
        next_text = text[:start].rstrip() + "\n"
    else:
        next_text = text[:start].rstrip() + "\n\n" + text[next_start + 1 :].lstrip()
    if next_text != text:
        path.write_text(next_text, encoding="utf-8")


def normalize_source_queue(path: Path) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    header: list[str] = []
    entries: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("## "):
            if current:
                entries.append(current)
            current = [line]
        elif current is None:
            header.append(line)
        else:
            current.append(line)
    if current:
        entries.append(current)

    seen: set[str] = set()
    unique: list[list[str]] = []
    for entry in entries:
        source = next((line.removeprefix("- source:").strip() for line in entry if line.startswith("- source:")), "")
        key = source or "\n".join(entry).strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)

    next_lines = list(header)
    while next_lines and next_lines[-1] == "":
        next_lines.pop()
    next_lines.append("")
    for entry in unique:
        next_lines.extend(entry)
        next_lines.append("")
    next_text = "\n".join(next_lines).rstrip() + "\n"
    if next_text != path.read_text(encoding="utf-8"):
        path.write_text(next_text, encoding="utf-8")


def current_commit(path: Path) -> str:
    try:
        result = run(["git", "rev-parse", "--short=7", "HEAD"], cwd=path)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def clone_or_update_reference(root: Path, repo_url: str = DEFAULT_REPO_URL) -> str:
    reference_dir = root / "reference" / "autoresearch"
    if not git_available():
        return "git unavailable; skipped reference clone"
    if reference_dir.exists():
        result = run(["git", "pull", "--ff-only"], cwd=reference_dir, check=False)
        if result.returncode == 0:
            return f"updated karpathy/autoresearch reference at {reference_dir}"
        return f"reference exists at {reference_dir}; pull skipped: {result.stderr.strip()[:240]}"
    reference_dir.parent.mkdir(parents=True, exist_ok=True)
    result = run(["git", "clone", "--depth", "1", repo_url, str(reference_dir)], check=False)
    if result.returncode != 0:
        return f"reference clone skipped: {result.stderr.strip()[:240]}"
    return f"cloned karpathy/autoresearch reference at {reference_dir}"


def program_md() -> str:
    return """# OpenClaw Speed Autoresearch

This workspace adapts the `karpathy/autoresearch` method to OpenClaw runtime speed and reliability research. The current overnight objective is to improve real decode tokens/sec for Gemma 4 31B JANG/JANQ with the Gemma 4 MTP assistant drafter path. The goal is not to train a model or change the user's target model. Reliability, memory stability, tool-call behavior, and loop resistance remain hard guards while decode TPS is optimized.

## Scope

You are working on OpenClaw only. Do not touch opencode. Do not change the user's model choice unless the user explicitly asks. The active model target is Gemma 4 31B JANG through the OpenClaw model profile layer.

In-scope files are the OpenClaw harness files in the setup repository, especially:

- `openclaw/openclaw-model-proxy.py`
- `openclaw/openclaw-jang-vlm-server.py`
- `openclaw/openclaw-jang-vlm-launcher.py`
- `openclaw/openclaw-mtp-drafter-calibrate.py`
- `openclaw/model-profiles.example.json`
- OpenClaw tests under `openclaw/test-*.py`

Live OpenClaw state is under `~/.openclaw`. Treat it as deployment/runtime state, not the source of truth. Do not edit opencode files or opencode state.

## Research Method

Use the autoresearch loop, but never begin with open-ended filesystem discovery.

1. Follow the Bootstrap Ladder below until it is complete.
2. Pick one concrete speed or reliability hypothesis from the Realistic Experiment Backlog.
3. Inspect exactly one named source file, config file, log tail, or benchmark output.
4. Make the smallest source change that tests the hypothesis.
5. Run focused tests first.
6. Deploy to `~/.openclaw` only when tests pass.
7. Run a bounded benchmark if memory pressure is acceptable.
8. Record the result in `results.tsv`.
9. Keep the change if it improves speed/reliability without degrading UX. Revert your own failed experiment if it does not.
10. Continue automatically until interrupted by the user.

Do not ask the user to continue after each experiment. Do not ask the user to manually test unless permissions or hardware state make testing impossible.

## Current Priority

Focus on improvements that make this exact OpenClaw setup faster and more reliable:

- MTP decode behavior for Gemma 4 31B JANG/JANQ.
- Drafter acceptance and overhead for the official Gemma 4 assistant drafter.
- Drafter block size, quantization, calibration, and deterministic sampling settings.
- OpenClaw model proxy behavior only where buffering/status changes affect measured decode.
- Rapid-MLX or MLX/VLM implementation ideas only when they can plausibly reduce decode-loop overhead.

Think broadly, but every useful idea must become one of: a source patch, a benchmark result, a rejected experiment with evidence, or a source note for later.

Do not spend rounds on toy prompts such as repeated single words except as a one-time health check. Do not optimize for synthetic repeated-token numbers; use comparable normal-text/code prompts and record the prompt class.

## Frontier Speed Track

After the current OpenClaw/Rapid/JANG knobs have been measured and locally optimized, deliberately explore higher-upside paths toward 50-70 tok/s. This track is allowed to think beyond the current constraints, but it must stay grounded in implementable OpenClaw architecture.

Use two lanes:

- Lane A: current-stack work. Keep improving the existing Gemma 4 31B JANG Rapid-MLX backend without changing the model.
- Lane B: frontier proposals. Research and prototype architecture that could plausibly unlock a step-change in speed while preserving OpenClaw UX and model behavior.

Frontier proposal areas include:

- Rapid-MLX scheduler, prefix cache, paged/cache reuse, chunked prefill, speculative decode, PLD, draft-token strategies, and future Rapid upstream features.
- JANG/JANQ loader or kernel changes that reduce full-prompt handoff cost.
- Prompt architecture changes that move stable tool/system context into reusable prefixes or compact runtime state.
- Agent orchestration changes that avoid model turns for deterministic bookkeeping, result recording, or known-safe shell probes.
- Draft-model or same-tokenizer assist paths if they preserve the user's selected primary model and pass acceptance tests.
- Native MLX/Metal bottlenecks, memory layout, KV quantization, cache residency, and decode batching choices.

For every frontier idea, record a note with: expected speed impact, feasibility, risk to reliability, files or upstream projects involved, smallest prototype, and whether it is local-only, upstream-dependent, or requires a separate draft/helper model. Do not implement speculative ideas blindly. Promote only the ideas with a plausible path to a tested OpenClaw patch.

## Tool Discipline

This is a local 31B MLX workflow. Every tool result is expensive on the next turn.

- Use exactly one tool call per assistant turn.
- Read one specific file or run one bounded command at a time.
- Never use broad commands such as `find ~`, `find /`, `find /Users`, `ls -R`, recursive grep over home, or whole-disk search.
- Prefer the setup repository path and explicit files listed above.
- After each tool result, summarize the useful finding in your own words before choosing the next tool.
- If a command is blocked as broad, immediately retry with one narrower path; do not keep emitting more broad commands.
- Keep `results.tsv` entries short.

## Benchmarks

Use quick benchmarks before heavy ones. Prefer prompts that do not trigger broad tools. Measure:

- startup readiness time
- time to first visible token
- prefill tokens/sec when available from logs
- decode tokens/sec when available from logs
- wall time
- memory before/start/load/generate/stop

Use `openclaw speed-research-benchmark --quick` for a bounded probe. The nested form `openclaw speed-research benchmark --quick` is also accepted for convenience. If the model is stopped and memory pressure is high, record the blocker instead of starting a 31B model.

## Realistic Experiment Backlog

Prioritize these experiment types over generic LLM speed prompts:

- OpenClaw bootstrap turn: first tool call latency when asked to read `program.md`.
- Tool-call round trip: read one explicit OpenClaw source file, summarize it, then write one result row.
- Prompt-size impact: compare shaped prompt tokens and TTFT before/after context compaction or history limits.
- Rapid-MLX backend settings: test one setting at a time, such as prefill step size, batch size, KV cache quantization, prefix cache policy, stream interval, or max tool tokens.
- JANG/JANQ bridge behavior: inspect whether the bridge prevents repeated tokens, reasoning leakage, and expensive unnecessary full-prompt paths.
- Perceived latency: verify heartbeat/status behavior during long prefill and time to first useful TUI status.
- Memory safety: record Metal active/peak memory, RSS, and whether compressor/swap rises.

Each experiment must name the exact file or runtime knob under test and the exact command used to measure it.

## Speed Targets

Treat these as directional targets, not promises:

- Immediate usability target: first useful status in under 5s and follow-up agent turns under 15s when prefix/cache is warm.
- Current-stack stretch target: realistic OpenClaw decode above 20 tok/s and much lower repeated prefill.
- Frontier target: investigate paths that could reach 50-70 tok/s perceived or measured decode on suitable workloads.

Prefer perceived speed wins that help agentic work: faster first tool call, less repeated prefill, better streaming status, fewer unnecessary model turns, and fewer wasted tokens.

## Implementation Gate

Before keeping an idea, prove it can be implemented in this setup:

1. Identify the specific source file or model profile knob.
2. Make the smallest source/config change.
3. Run focused tests.
4. Deploy only if tests pass.
5. Run a realistic benchmark.
6. Record `keep`, `discard`, or `blocked` in `results.tsv` with the reason.

If a change requires an upstream Rapid-MLX feature that is not present locally, record the gap clearly and move to the next implementable improvement.

## Reliability Rules

- No silent hangs: every long action needs heartbeat/status in logs or output.
- No runaway tool loops: tool-call streams must be bounded and recoverable.
- No reasoning marker leaks into TUI content.
- No duplicate model services.
- Closing OpenClaw must release OpenClaw-owned model/gateway processes.
- Keep native OpenClaw behavior. Do not strip features to fake speed.

## Results

Append every experiment to `results.tsv` as TSV, not CSV:

`timestamp run_id status target hypothesis ttft_s prefill_tps decode_tps wall_s memory_gb commit notes`

Statuses:

- `keep`: improvement or reliability hardening retained
- `discard`: tested and reverted
- `blocked`: could not run due to memory, dependency, or hardware state
- `crash`: run crashed and was not kept

## Bootstrap Ladder

The wrapper and autopilot own workspace bootstrap. Do not spend a model turn reading this full `program.md` unless a human explicitly asks for it.

In a fresh agent run, perform one of these narrow actions:

1. Run exactly `/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample`.
2. Read exactly `/Users/kristian/.openclaw/research/speed/results.tsv`.
3. Run exactly `git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`.

After that, choose the smallest realistic OpenClaw speed/reliability experiment and record the result. Do not run setup commands during bootstrap. The workspace already exists. Do not use `find`, recursive `ls`, recursive grep, or broad local search. Do not read `reference/autoresearch/program.md` during bootstrap; it is method reference only.

## Narrow Tool Catalog

Allowed narrow actions are:

- `read` a single explicit file path from Scope or `sources/queue.md`.
- `exec` one exact command against the OpenClaw source repo, such as `git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`.
- `exec` one exact test file, such as `python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py`.
- `exec` one exact benchmark helper, such as `/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample`.
- `exec` one exact log tail, such as `tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log`.

Forbidden actions include `find ~`, `find /`, `find /Users`, `ls -R`, `grep -R`, recursive `rg` over home, `mdfind`, and any broad command intended to discover files. If you need a file, use the explicit paths in this program.
"""


def readme_md() -> str:
    return """# OpenClaw Speed Research Workspace

This is an OpenClaw-specific adaptation of Karpathy's `autoresearch` pattern.

The original repo is cloned under `reference/autoresearch` for method reference. Its default CUDA training workload is not used on this Mac. The useful part is the operating loop: fixed-budget experiments, clear metrics, TSV logging, and keep/discard discipline.

Useful commands:

```bash
openclaw speed-research-setup
openclaw speed-research-prompt
openclaw speed-research-auto
openclaw speed-research-benchmark --mode decode-sample
```

Implementation is intentionally a separate gated phase. Before keeping production changes, read `implementation-skill.md` and follow its plan/test/deploy/record checklist.

The research agent should improve OpenClaw speed and reliability without touching opencode.
    """


def implementation_skill_md() -> str:
    return """# OpenClaw Speed Implementation Skill

Use this skill when a research finding is ready to become a source change. The goal is clean, minimal implementation, not speculative patching.

## Entry Criteria

Start implementation only when there is one accepted insight with evidence from `results.tsv`, a benchmark JSON file, a log excerpt, or a source note in `ideas.md`.

Do not implement from a vague hunch. If evidence is missing, run one narrow experiment first.

## Pre-Implementation Gate

Before editing source, state:

- The OpenClaw subsystem being changed and how it affects runtime, memory, tool calling, context size, and live deployment.
- Why the design is minimal, readable, and efficient enough for another software engineer to maintain.
- The exact metric, guardrail, failure mode, and rollback path.
- The prompt-size, memory, timeout, tool-loop, and context-regression blind spots.
- That opencode is out of scope and untouched.

## Implementation Loop

1. State the accepted insight in one sentence.
2. Identify the exact file or model-profile knob to change.
3. Define the expected behavior and the failure mode being prevented or improved.
4. Make the smallest cohesive patch as a patch file.
5. Run `openclaw-speed-research patch-execute` so the patch is classified, applied in a canary worktree, tested, and only then promoted.
6. Add or update the narrowest relevant test.
7. Deploy to `~/.openclaw` only after tests pass.
8. Run a bounded benchmark or record why it is blocked.
9. Record `keep`, `discard`, or `blocked` in `results.tsv`.
10. Revert your own failed experiment if it does not improve speed, reliability, or maintainability.

## Patch Executor Rules

- Safe and moderate patches can auto-promote only after canary apply and allowlisted tests pass.
- Architectural patches are held after canary unless `allow_architectural` is set and an explicit approval file contains the task id.
- Destructive patches are blocked before canary.
- Patch files must touch allowlisted OpenClaw paths only.
- Patch files must not touch opencode, `.env`, secrets, passwords, tokens, private keys, or private runtime config.
- A canary artifact must record impact classification, changed files, tests, promotion decision, and rollback state.

## Code Quality Rules

- Prefer existing OpenClaw harness patterns over new abstractions.
- Keep changes local to one responsibility: proxy guard, launcher guard, wrapper UX, research program, or profile configuration.
- Do not layer duplicate guard logic when one shared helper can express the rule clearly.
- Do not hide failures. Convert crashes, timeouts, and memory pressure into explicit logged states and result rows.
- Do not strip native OpenClaw behavior to fake speed.
- Do not touch opencode.
- Do not change the primary model unless the user explicitly asks.

## Review Checklist

Before marking a change `keep`, verify:

- The patch is smaller than the problem it solves.
- The names explain the intent without long comments.
- The failure path is visible in logs or terminal output.
- The test would fail without the change.
- The deployment path is explicit.
- The benchmark or blocker is recorded.
- The environment snapshot and evaluator-integrity gates pass.
- A staged diff secret scan found no `.env`, tokens, passwords, keys, private config, or sensitive logs.

## Rollback

If a change causes regressions, revert only the files changed by that experiment, record `discard` with the reason, and continue with the next smallest hypothesis.
"""


def tool_discipline_section() -> str:
    return """## Tool Discipline

This is a local 31B MLX workflow. Every tool result is expensive on the next turn.

- Use exactly one tool call per assistant turn.
- Read one specific file or run one bounded command at a time.
- Never use broad commands such as `find ~`, `find /`, `find /Users`, `ls -R`, recursive grep over home, or whole-disk search.
- Prefer the setup repository path and explicit files listed above.
- If you need repository state, use exactly `git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`.
- If you need recent proxy logs, use exactly `tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log`.
- After each tool result, summarize the useful finding in your own words before choosing the next tool.
- If a command is blocked as broad, immediately retry with one narrower path; do not keep emitting more broad commands.
- Keep `results.tsv` entries short.
"""


def research_method_section() -> str:
    return """## Research Method

Use a Ralph-style continuation loop with Karpathy-style measurable experiments and Hermes-style self-evolution gates. Never begin with open-ended filesystem discovery.

Each cycle follows one exact state transition:

`select task -> source/evidence check -> baseline -> probe or patch -> focused test -> benchmark -> analyze -> keep/discard/rework`

Progress requires a quality artifact: `STRATEGY.md`, `findings.jsonl`, `experiments.jsonl`, `rejections.jsonl`, `tasks.jsonl`, benchmark JSON with comparison, source patch with tests, or a blocker with evidence.

The supervisor owns three non-negotiable honesty gates:

- Environment snapshot: record commit, profile hashes, evaluator hashes, and redacted runtime env before autonomous work.
- Immutable evaluator policy: benchmark definitions and replay guards are frozen unless a canary/promote route explicitly changes policy.
- Plateau pivot: once a rung is settled below target, stop repeating it and move to drafter fit, DFlash compatibility, runtime overhead, or an exhaustion report.

Do not ask the user to continue after each experiment. Do not ask the user to manually test unless permissions or hardware state make testing impossible.
"""


def bootstrap_ladder_section() -> str:
    return """## Bootstrap Ladder

The wrapper and autopilot own workspace bootstrap. Do not spend a model turn reading this full `program.md` unless a human explicitly asks for it.

In a fresh agent run, perform one of these narrow actions:

1. Run exactly `/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample`.
2. Read exactly `/Users/kristian/.openclaw/research/speed/results.tsv`.
3. Run exactly `git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`.

After that, choose the smallest realistic OpenClaw speed/reliability experiment and record the result. Do not run setup commands during bootstrap. The workspace already exists. Do not use `find`, recursive `ls`, recursive grep, or broad local search. Do not read `reference/autoresearch/program.md` during bootstrap; it is method reference only.
"""


def narrow_tool_catalog_section() -> str:
    return """## Narrow Tool Catalog

Allowed narrow actions are:

- `read` a single explicit file path from Scope or `sources/queue.md`.
- `exec` one exact command against the OpenClaw source repo, such as `git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`.
- `exec` one exact test file, such as `python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py`.
- `exec` one exact benchmark helper, such as `/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample`.
- `exec` one exact log tail, such as `tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log`.

Forbidden actions include `find ~`, `find /`, `find /Users`, `ls -R`, `grep -R`, recursive `rg` over home, `mdfind`, and any broad command intended to discover files. If you need a file, use the explicit paths in this program.
"""


def current_priority_section() -> str:
    return """## Current Priority

Focus the overnight run on one user-facing metric first: **real OpenClaw TUI decode tokens/sec and visible response smoothness for Gemma 4 31B JANG/JANQ with the Gemma 4 MTP assistant drafter path**.

The scope order is strict:

1. Improve normal `openclaw tui` chat decode speed and perceived response quality.
2. Preserve reliability: no reasoning-marker leaks, no tool-call loops, no silent stream stalls, no Python/Metal memory crashes.
3. Improve autoresearch only where it helps the first two goals: better task selection, better implementation handoff, fewer wasted cycles, and safer overnight operation.

Autoresearch self-improvement is not the primary benchmark. It is a support system for finding, testing, and safely implementing changes that make the regular TUI experience faster.

The active production baseline is:

- Target model: `mlx/Gemma-4-31B-JANG_4M-CRACK`.
- Serving path: `openclaw/openclaw-jang-vlm-server.py` behind `openclaw/openclaw-model-proxy.py`.
- Drafter path: quantized Gemma 4 assistant under `{openclawDir}/models/gemma-4-31B-it-assistant-mlx-4bit`.
- Known measured baseline: about 14-15 decode tok/s on bounded decode prompts, with MTP `mean_accept` often below 1 on longer normal text.
- Goal: improve measured TUI decode TPS first; TTFT and prefill are secondary unless they block fair decode measurement or make the TUI feel frozen before decode starts.

Primary research questions:

- Why is MTP acceptance low for this JANQ target, and can acceptance be raised without changing the target model?
- Can drafter quantization, block size, calibration, sampling/logit settings, cache handling, or the MLX MTP loop improve wall-clock decode TPS?
- Can DFlash, Rapid-MLX, or upstream MLX/VLM implementation details reduce drafter overhead while preserving OpenClaw behavior?

In-scope source files and knobs:

- `openclaw/openclaw-jang-vlm-server.py`
- `openclaw/openclaw-jang-vlm-launcher.py`
- `openclaw/openclaw-mtp-drafter-calibrate.py`
- `openclaw/model-profiles.example.json`
- `openclaw/openclaw-model-proxy.py` only when proxy streaming or shaping affects measured decode
- Live profile env vars for `OPENCLAW_JANG_DRAFT_MODEL`, `OPENCLAW_JANG_DRAFT_BLOCK_SIZE`, drafter quantization, temperature, top-p, and repetition penalty
- DFlash reference only: `https://github.com/z-lab/dflash` and `z-lab/gemma-4-31B-it-DFlash`, gated behind compatibility proof before any live runtime change

Think broadly, but every useful idea must become one of: a TUI-relevant decode benchmark result, MTP acceptance measurement, drafter calibration/quantization experiment, rejected experiment with evidence, or a small source patch with tests.

Do not spend rounds on generic prefill, prompt-shape, tool UX, or autoresearch meta-work unless it directly improves TUI decode experiments or prevents autoresearch from safely producing TUI decode improvements. Do not optimize for synthetic repeated-token prompts; use deterministic normal-text, code, shell-list, and agent-summary prompts.
"""


def frontier_speed_track_section() -> str:
    return """## Frontier Speed Track

Deliberately explore higher-upside paths toward 30+ tok/s decode first, then 50-70 tok/s if evidence supports it. This track must stay grounded in implementable OpenClaw architecture and must not change the selected target model.

Use three lanes:

- Lane A: production MTP path. Improve the current JANG/VLM server plus quantized Gemma assistant drafter path.
- Lane B: drafter alignment. Research and prototype ways to make the assistant drafter better match JANQ target logits/hidden states.
- Lane C: runtime overhead. Research Rapid-MLX, MLX-VLM, and MLX decode-loop changes that reduce MTP verification/drafter overhead.

Frontier proposal areas include:

- JANQ-specific drafter calibration beyond `pre_projection.weight`: adapters, post-projection tuning, selective layer tuning, distillation targets, acceptance-weighted losses, and multi-position trace datasets.
- Drafter quantization recipes: q-bits, q-group-size, mixed-bit predicates, BF16 vs 4-bit vs 3-bit tradeoffs, and whether quantization changes acceptance or only overhead.
- MTP scheduler policy: fixed block size, acceptance-aware block sizing, prompt-class-specific block size, and early stop on acceptance collapse.
- MLX/VLM MTP implementation: eval boundaries, cache rollback cost, shared-KV slicing, prompt-cache reuse, target/drafter stream synchronization, and opportunities to upstream a cleaner faster loop.
- Rapid-MLX compatibility: whether separate assistant drafters can be supported natively rather than only built-in MTP heads.
- DFlash compatibility: whether `dflash.model_mlx.stream_generate` can safely wrap the JANG-loaded Gemma4 target, capture the required hidden layers, and preserve OpenClaw streaming/tool/reasoning guards.
- Benchmark design: separating decode wall time from prefill, extracting `mtp_rounds` and `mean_accept` from logs, and comparing against no-drafter baseline.

For every frontier idea, record: expected decode TPS impact, expected acceptance impact, feasibility, reliability risk, files/upstream projects involved, smallest prototype, and rollback path. Promote only ideas with a plausible path to a tested OpenClaw patch.

## 30 Tok/S Investigation Ladder

The harness should climb this ladder autonomously instead of repeatedly proving the same block-size result:

1. Baseline honestly: live MTP decode, no-drafter control, and block-size sweep on the same deterministic prompt set.
2. Converge or reject easy knobs: if block size 2 remains the winner and block sizes 1/3/4 are slower or invalid, stop repeating that sweep except as an occasional regression check.
3. Diagnose the bottleneck: decide whether the current limit is low acceptance, drafter cost, target verification cost, cache rollback, Python loop overhead, proxy buffering, or memory pressure.
4. Raise acceptance: investigate JANQ-specific drafter fit, calibration targets beyond `pre_projection.weight`, quantization, logit/sampling settings, and prompt-class-specific rejection patterns.
5. Reduce overhead: inspect the exact MLX/VLM MTP loop boundaries and propose only small patches that reduce verification/cache/rollback work while preserving streaming/tool/reasoning guards.
6. Explore step-change paths: DFlash and Rapid/MLX compatibility are frontier lanes, not assumptions. Prove structural compatibility first, then canary, then benchmark, then promote only if normal TUI decode improves.
7. Promote safely: no live TUI change is kept unless paired benchmarks beat the current block-2 baseline and replay checks pass for tool calls, reasoning separation, stream stalls, and memory.

If a rung is exhausted, record the evidence and move upward. Do not spend overnight cycles re-running a settled rung unless a new source change makes the old evidence stale.

## Karpathy Compatibility Layer

This workspace keeps Karpathy's native loop structure while adapting it to OpenClaw:

- Human-written policy remains in Markdown: `program.md`, `STRATEGY.md`, and `implementation-skill.md`.
- The evaluator is frozen: `benchmark-manifest.json` and `replay-buffer.jsonl` define the scoring surface and must not drift during a run.
- Experiments are fixed-budget and measurable: every kept or rejected change needs a bounded benchmark, test, or blocker row.
- Keep/discard is mechanical: safe improvements can move forward, failures are recorded and rolled back, and architectural patches wait for explicit approval.
- The supervisor may add deterministic structure around the loop, but it must not hide failures behind model narration.
"""


def realistic_experiment_backlog_section() -> str:
    return """## Realistic Experiment Backlog

Prioritize decode/MTP experiments over generic LLM speed prompts:

- Decode baseline: run bounded normal-text decode prompts and record decode TPS, wall time, `mtp_rounds`, `mean_accept`, drafter path, block size, temperature, repetition penalty, and commit.
- No-drafter control: temporarily disable `OPENCLAW_JANG_DRAFT_MODEL`, benchmark the same prompts, restore MTP, and record the real speedup factor.
- Drafter block sweep: test block sizes 2, 3, 4, and 6 on the same deterministic prompt set. Keep only settings that improve aggregate decode TPS.
- Drafter quantization sweep: compare BF16, 4-bit, 3-bit, q-group-size, and mixed-bit candidates if memory allows. Reject quantization that lowers decode TPS or acceptance.
- JANQ calibration: use `openclaw/openclaw-mtp-drafter-calibrate.py` to test small calibration ideas. Promote only if benchmarked decode TPS beats the current official 4-bit drafter.
- Acceptance diagnostics: parse logs for `mtp_rounds` and `mean_accept`; identify prompts/classes with acceptance collapse and record why.
- MTP loop overhead: inspect `mlx_vlm.generate._mtp_rounds` behavior and compare with OpenClaw server usage. Look for avoidable eval/cache/rollback overhead.
- DFlash compatibility spike: inspect `dflash.model_mlx` against `mlx_vlm.models.gemma4.gemma4.Model` and the JANQ target loader. Do not install or promote DFlash into the live server until a no-load structural check and a bounded canary pass.
- Deterministic decode settings: test temperature, top-p, repetition penalty, and logit processors for acceptance and loop safety. Keep deterministic settings unless quality/reliability regresses.
- Proxy streaming control: verify OpenClaw proxy is not hiding decode gains by buffering content. Measure time to first visible token separately from decode TPS.
- Memory safety during decode: record Metal/RSS/compressor before and after drafter experiments; discard anything that increases crash risk.

Each experiment must name the exact file or runtime knob under test, the exact benchmark command, the prompt set, and the keep/discard decision.
"""


def speed_targets_section() -> str:
    return """## Speed Targets

Treat these as directional targets, not promises:

- Current live baseline: about 14-15 decode tok/s with the official quantized 4-bit assistant drafter at block size 2.
- Immediate target: stable measured decode above 18 tok/s without lower-quality output, reasoning loops, or extra memory pressure.
- Current-stack stretch target: realistic OpenClaw decode above 20 tok/s on normal deterministic prompts.
- Frontier target: investigate paths that could reach 30+ tok/s, then 50-70 tok/s if drafter acceptance and runtime overhead evidence supports it.

Primary metric is TUI-relevant decode TPS from comparable normal-text/code prompts. Secondary metrics are MTP mean acceptance, MTP rounds, wall time, first visible token, stream smoothness, and memory pressure.
Autoresearch quality metrics are tertiary: fewer stalled cycles, higher-quality implementation candidates, and cleaner handoff artifacts. They matter only when they help produce safer TUI decode-speed improvements.
"""


def strategy_decode_focus_section() -> str:
    return """## Decode MTP Focus

Primary metric for the current overnight run: real OpenClaw TUI decode tokens/sec and visible response smoothness for `mlx/Gemma-4-31B-JANG_4M-CRACK` with the Gemma 4 assistant drafter path.

Autoresearch self-improvement is secondary. Improve the research loop only when it makes the TUI decode-speed loop more deterministic, safer, or more likely to produce a clean implementation.

Current baseline:

- Official quantized Gemma 4 assistant drafter, block size 2.
- Recent bounded decode measurements: about 14-15 tok/s.
- No-drafter control seen earlier: about 12.5 tok/s.
- Heuristic MTP scheduling, 3-bit drafter, and pre-projection-only calibration did not beat the official q4 drafter.

Research order:

1. Measure decode-sample repeatability.
2. Extract MTP `mean_accept` and `mtp_rounds` from recent logs.
3. Compare no-drafter control against live q4 drafter.
4. Sweep drafter block size and quantization only with fixed prompts and rollback.
5. Promote JANQ drafter calibration only if wall-clock decode TPS improves.

TTFT, prefill, prompt-shape, tool-roundtrip, and autoresearch workflow metrics are secondary unless they block fair TUI decode measurement or safe overnight execution.
"""


def strategy_current_best_section() -> str:
    return """## Current Best Understanding

- Live baseline is the official quantized Gemma 4 assistant drafter at block size 2.
- Recent bounded decode measurements are about 14-15 tok/s, versus roughly 12.5 tok/s without a drafter.
- Earlier heuristic MTP scheduling, 3-bit drafter experiments, and pre-projection-only calibration did not beat the official 4-bit drafter.
"""


def strategy_top_hypotheses_section() -> str:
    return """## Top Hypotheses

1. Decode TPS will improve only if MTP acceptance rises enough to beat drafter overhead on normal prompts.
2. The best next experiments are no-drafter control, block-size sweep, drafter quantization/calibration, and log-based `mean_accept` analysis.
3. Rapid-MLX or MLX/VLM loop changes matter only if they reduce verification/drafter overhead without changing the selected target model.
"""


def strategy_rejected_section() -> str:
    return """## Rejected Or Exhausted

- Repeating TTFT, tool-roundtrip, or prompt-size benchmarks without a decode/MTP hypothesis is noise for this phase.
- Earlier blocked prompt-shape/profile-bandit/speculative-compat tasks are superseded by the live MTP drafter baseline and decode/MTP task queue.
"""


def strategy_current_synthesis_section() -> str:
    return """## Current Synthesis

- Current research phase is decode/MTP optimization, not generic speed archaeology.
- Top production idea: explain and improve MTP `mean_accept` for the live q4 assistant drafter.
- Top experiment idea: paired no-drafter, block-size, and quantization comparisons on the same prompt set.
- Top frontier idea: JANQ-specific drafter alignment or MLX/VLM loop changes only if wall-clock decode TPS improves.
"""


def refresh_strategy_objective(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    objective_lines = {
        "Objective: optimize raw speed for the current Gemma 4 31B JANG OpenClaw setup while treating crashes, loops, memory pressure, and tool failures as hard guards.",
        "Objective: improve real decode tokens/sec for the current Gemma 4 31B JANG/JANQ OpenClaw setup with the Gemma 4 MTP assistant drafter, while treating crashes, loops, memory pressure, and tool failures as hard guards.",
        "Objective: improve real OpenClaw TUI decode tokens/sec and visible response smoothness for the current Gemma 4 31B JANG/JANQ setup with the Gemma 4 MTP assistant drafter, while treating crashes, loops, memory pressure, and tool failures as hard guards. Autoresearch self-improvement is secondary and exists to make that TUI decode loop safer and more productive.",
    }
    new = "Objective: improve real OpenClaw TUI decode tokens/sec and visible response smoothness for the current Gemma 4 31B JANG/JANQ setup with the Gemma 4 MTP assistant drafter, while treating crashes, loops, memory pressure, and tool failures as hard guards. Autoresearch self-improvement is secondary and exists to make that TUI decode loop safer and more productive."
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line in objective_lines:
            lines[index] = new
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return


def implementation_gate_section() -> str:
    return """## Implementation Gate

Research and implementation are separate phases. Before keeping any source/config change, read `/Users/kristian/.openclaw/research/speed/implementation-skill.md` and follow it.

Before entering implementation, pass the pre-implementation gate:

1. Name the OpenClaw subsystem touched and how it affects runtime, memory, tool calling, context size, and live deployment.
2. State why the patch is the smallest clean change and how a normal developer can understand it.
3. Define the speed metric, reliability guard, rollback path, and exact failure mode being improved.
4. Identify prompt-size, memory, timeout, tool-loop, and context-regression blind spots.
5. Confirm opencode is untouched and out of scope.

Before implementing an idea, prove it belongs in this setup:

1. Identify the specific source file or model profile knob.
2. Name the evidence from `results.tsv`, a benchmark JSON file, a log excerpt, or `ideas.md`.
3. Write a one-sentence expected behavior.
4. Generate the smallest source/config patch file.
5. Run the patch through `openclaw-speed-research patch-execute`.
6. Add or update the narrowest relevant test.
7. Deploy only if the canary and focused tests pass.
8. Run a realistic benchmark or record why it is blocked.
9. Record `keep`, `discard`, or `blocked` in `results.tsv` with the reason.

If a change requires an upstream Rapid-MLX feature that is not present locally, record the gap clearly and move to the next implementable improvement.

Do not bundle unrelated cleanup with speed experiments. Do not keep a patch that only rearranges code without measured speed, reliability, or maintainability value.

Before pushing, run a staged diff secret scan and confirm no `.env`, passwords, tokens, keys, private config, or sensitive logs are included.
"""


def dynamic_policy_optimization_section() -> str:
    return """## Dynamic Policy Optimization

GEPA policy optimization is a supervisor reflex, not a default mode.

The deterministic supervisor may run `gepa-escalation` after quality and frontier review. It should only escalate when recent evidence shows repeated blocked rows, rework tasks, trajectory failures, low-quality reviews, or measurement artifacts.

Escalation creates a bounded policy canary for `program.md`, `STRATEGY.md`, `insight-rubric.json`, `implementation-skill.md`, or `tasks.jsonl`. It must not mutate live runtime code, model profiles, opencode, or the selected model.

The canary must include actionable side information from actual failures, explicit Pareto objectives, and one narrow text-policy delta. Promotion requires replay checks, quality review, no new broad tool commands, no memory/Metal regression, and the existing patch/approval gates. If those gates are not met, keep the canary as evidence and continue the default supervisor route.
"""


def prompt_text(root: Path) -> str:
    compact_workspace(root)
    return f"""OpenClaw Speed Autoresearch bootstrap.

Workspace: {root}

Primary scope: improve normal `openclaw tui` decode speed and visible response smoothness first. Improve autoresearch itself only when it helps produce safer, better TUI decode-speed changes.

First assistant action: read exactly:
`{root / 'RUN_MEMORY.md'}`

Then read exactly:
`{root / 'SUMMARY.md'}`

Only after those two reads, run this narrow benchmark command when the run memory says measurement is needed:
`/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample`

Do not read the full `program.md` unless a human explicitly asks. It is installed policy, not first-turn context.

Hard constraints:
- OpenClaw only. Do not touch opencode.
- Do not change the model unless the user explicitly asks.
- Continue the loop without asking me to manually continue.
- Use one narrow tool call per assistant turn.
- Never use broad local search commands.
- Do not run setup commands during bootstrap.
"""


def load_self_improvement_module():
    try:
        import openclaw_self_improvement
    except ImportError as exc:
        raise RuntimeError(
            "OpenClaw self-improvement support is not installed. Copy "
            "openclaw_self_improvement.py next to openclaw-speed-research, "
            "or run from the setup repository."
        ) from exc
    return openclaw_self_improvement


def ensure_optional_self_improvement_state(root: Path) -> None:
    try:
        self_improvement = load_self_improvement_module()
    except RuntimeError:
        return
    self_improvement.ensure_self_improvement_state(root)


def setup_workspace(args: argparse.Namespace) -> int:
    root = workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    ensure_research_state(root)
    ensure_optional_self_improvement_state(root)
    ensure_lane_contracts(root)
    (root / "sources").mkdir(exist_ok=True)
    clone_status = clone_or_update_reference(root, args.repo_url)
    write_if_changed(root / "program.md", program_md())
    remove_section(root / "program.md", "Starting Point")
    upsert_section(root / "program.md", "Research Method", research_method_section())
    upsert_section(root / "program.md", "Bootstrap Ladder", bootstrap_ladder_section())
    upsert_section(root / "program.md", "Narrow Tool Catalog", narrow_tool_catalog_section())
    upsert_section(root / "program.md", "Current Priority", current_priority_section())
    upsert_section(root / "program.md", "Frontier Speed Track", frontier_speed_track_section())
    ensure_section(root / "program.md", "## Tool Discipline", tool_discipline_section())
    upsert_section(root / "program.md", "Realistic Experiment Backlog", realistic_experiment_backlog_section())
    upsert_section(root / "program.md", "Speed Targets", speed_targets_section())
    upsert_section(root / "program.md", "Implementation Gate", implementation_gate_section())
    upsert_section(root / "program.md", "Dynamic Policy Optimization", dynamic_policy_optimization_section())
    refresh_strategy_objective(root / "STRATEGY.md")
    upsert_section(root / "STRATEGY.md", "Current Best Understanding", strategy_current_best_section())
    upsert_section(root / "STRATEGY.md", "Top Hypotheses", strategy_top_hypotheses_section())
    upsert_section(root / "STRATEGY.md", "Rejected Or Exhausted", strategy_rejected_section())
    upsert_section(root / "STRATEGY.md", "Current Synthesis", strategy_current_synthesis_section())
    upsert_section(root / "STRATEGY.md", "Decode MTP Focus", strategy_decode_focus_section())
    write_if_changed(root / "README-openclaw-speed.md", readme_md())
    write_if_changed(root / "implementation-skill.md", implementation_skill_md())
    write_if_missing(root / "ideas.md", "# Speed Research Ideas\n\n")
    write_if_missing(
        root / "sources" / "queue.md",
        "# Research Source Queue\n\n"
        "Add notes, URLs, article excerpts, image URLs, and local file paths here. "
        "The OpenClaw speed researcher should read this file at the start of each loop "
        "and incorporate relevant sources without broad speculative searches.\n\n",
    )
    normalize_source_queue(root / "sources" / "queue.md")
    write_if_missing(
        root / ".gitignore",
        "logs/\nbenchmarks/*.json\nexperiments/*.json\n*.tmp\n",
    )
    compact_workspace(root)
    print(root)
    print(clone_status)
    return 0


def self_improve(args: argparse.Namespace) -> int:
    self_improvement = load_self_improvement_module()
    root = workspace_root()
    ensure_research_state(root)
    self_improvement.ensure_self_improvement_state(root)
    if args.action == "status":
        print(json.dumps(self_improvement.status(root), indent=2, sort_keys=True))
        return 0
    if args.action == "derive-lessons":
        lessons = self_improvement.derive_lessons(root, recent_rows=args.recent_rows)
        recorded = self_improvement.record_lessons(root, lessons)
        print(json.dumps({"ok": True, "derived": len(lessons), **recorded}, indent=2, sort_keys=True))
        return 0
    if args.action == "curate":
        summary = self_improvement.curate(root, recent_rows=args.recent_rows)
        print(json.dumps({"ok": True, **summary}, indent=2, sort_keys=True))
        return 0
    if args.action == "evolve":
        report = self_improvement.run_evolution(
            root,
            recent_rows=args.recent_rows,
            max_variants_per_skill=args.max_variants_per_skill,
            min_score=args.min_score,
            shadow_min_score=args.shadow_min_score,
            stage_min_wins=args.stage_min_wins,
            stage_max_effective_authority=args.stage_max_effective_authority,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unknown self-improve action: {args.action}")


def latest_watchdog_artifact(root: Path) -> dict[str, Any]:
    path = root / "watchdog" / "autoresearch-watchdog-latest.json"
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def self_improvement_alive_report(root: Path, *, recent_rows: int = 160) -> dict[str, Any]:
    """Score whether the sidecar can replace manual health-check/repair work.

    The score is intentionally artifact-based. LLM ideas can feed the sidecar,
    but this eval only rewards durable evidence: review artifacts, canary-only
    evolution, shadow/replay records, rollback containment, and zero active
    mutation of live skills or runtimes.
    """
    ensure_research_state(root)
    try:
        self_improvement = load_self_improvement_module()
        self_improvement.ensure_self_improvement_state(root)
        status = self_improvement.status(root)
    except RuntimeError as error:
        return {
            "ok": False,
            "kind": "self-improvement-alive-eval",
            "timestamp": int(time.time()),
            "total_score": 0,
            "readiness": "missing",
            "reason": str(error),
            "gates": {"sidecar_installed": False},
            "hard_gate_failures": ["sidecar_installed"],
            "components": {},
            "next": "install openclaw_self_improvement.py next to the research helper",
        }
    required_skills = {
        "decode-speed-research",
        "implementation-gate",
        "reviewer-quality",
        "self-improvement-curator",
    }
    skills = set(str(item) for item in status.get("skills", []))
    latest_quality = latest_json_artifact(root, "quality-review-*.json")
    latest_frontier = latest_json_artifact(root, "frontier-system-eval-*.json")
    latest_autonomy = latest_json_artifact(root, "frontier-autonomy-score-*.json")
    latest_handoff = latest_json_artifact(root, "implementation-handoff-audit-*.json")
    latest_burn_in = latest_json_artifact(root, "stability-burn-in-*.json")
    latest_council = latest_json_artifact(root, "review-council-*.json")
    latest_watchdog = latest_watchdog_artifact(root)
    canonical = canonical_autoresearch_state(root, recent_rows=recent_rows)
    noise = canonical.get("noise") if isinstance(canonical.get("noise"), dict) else {}
    bad_rows = recent_bad_behavior_rows(root, recent_rows=recent_rows)
    tasks = read_jsonl(root / "tasks.jsonl")
    ready_tasks = [task for task in tasks if task.get("status", "ready") in {"ready", "rework"}]
    deterministic_ready = [task for task in ready_tasks if is_deterministic_research_task(task)]
    deterministic_ready_ids = [str(task.get("id", "")) for task in deterministic_ready[:8]]
    last_evolution = status.get("last_evolution") if isinstance(status.get("last_evolution"), dict) else {}
    last_summary = status.get("last_summary") if isinstance(status.get("last_summary"), dict) else {}
    usage = status.get("usage") if isinstance(status.get("usage"), dict) else {}
    all_usage_recorded = all(
        isinstance(usage.get(skill), dict) and int(usage[skill].get("use_count") or 0) > 0
        for skill in required_skills
    )
    decisions = int(status.get("decisions") or 0)
    held_variants = int(status.get("held_variants") or 0)
    shadow_reviews = int(status.get("shadow_reviews") or 0)
    promotions = int(status.get("promotions") or 0)
    rollbacks = int(status.get("rollbacks") or 0)
    rolled_back = int(status.get("rolled_back_promotions") or 0)
    active_skill_mutated = bool(last_evolution.get("active_skill_mutated"))
    evolution_generated = isinstance(last_evolution.get("decisions"), dict) and int(
        last_evolution["decisions"].get("generated") or 0
    ) > 0
    watchdog_decision = str(latest_watchdog.get("decision", ""))
    council_scorecard = latest_council.get("scorecard") if isinstance(latest_council.get("scorecard"), dict) else {}
    council_overall = float(council_scorecard.get("overall") or 0)
    council_failures = (
        council_scorecard.get("hard_gate_failures")
        if isinstance(council_scorecard.get("hard_gate_failures"), list)
        else []
    )
    accepted_watchdog_decisions = {
        "healthy",
        "seed-next-candidate",
        "repair-routing",
        "autonomy-repair",
        "frontier-repair",
        "self-improvement-repair",
    }
    gates = {
        "sidecar_installed": True,
        "required_skills_present": required_skills.issubset(skills),
        "durable_memory_present": int(status.get("lessons") or 0) > 0
        and int(status.get("trajectories") or 0) > 0
        and int(status.get("proposals") or 0) > 0,
        "eval_cases_present": int(status.get("eval_cases") or 0) > 0,
        "evolution_canaries_present": decisions > 0 and held_variants > 0,
        "shadow_review_present": shadow_reviews > 0,
        "staged_or_rollback_awareness": promotions > 0 or rollbacks > 0 or rolled_back > 0,
        "no_active_skill_mutation": not active_skill_mutated,
        "review_artifacts_present": bool(latest_quality) and bool(latest_frontier) and bool(latest_handoff),
        "review_council_present": bool(latest_council) or bool(deterministic_ready),
        "review_council_quality": bool(deterministic_ready)
        or (
            bool(latest_council)
            and bool(latest_council.get("ok"))
            and council_overall >= 95.0
            and not council_failures
        ),
        "autonomy_gate_present": bool(latest_autonomy),
        "stability_evidence_present": bool(latest_burn_in),
        "watchdog_or_route_present": bool(latest_watchdog) or bool(deterministic_ready),
        "canonical_clean": bool(canonical.get("clean")),
        "zero_active_noise": all(int(noise.get(key, 0) or 0) == 0 for key in noise),
        "no_bad_behavior_rows": not bad_rows,
        "skills_used": all_usage_recorded,
    }
    evidence_requirements = {
        "observe": {
            "required": [
                "quality_artifact",
                "frontier_artifact",
                "handoff_artifact",
                "council_or_route",
                "council_quality_or_route",
                "watchdog_or_route",
            ],
            "present": {
                "quality_artifact": bool(latest_quality),
                "frontier_artifact": bool(latest_frontier),
                "handoff_artifact": bool(latest_handoff),
                "council_or_route": gates["review_council_present"],
                "council_quality_or_route": gates["review_council_quality"],
                "watchdog_or_route": gates["watchdog_or_route_present"],
            },
            "evidence": {
                "quality": latest_quality.get("_artifact_path", ""),
                "frontier": latest_frontier.get("_artifact_path", ""),
                "handoff": latest_handoff.get("_artifact_path", ""),
                "review_council": latest_council.get("_artifact_path", ""),
                "review_council_score": council_overall,
                "review_council_failures": council_failures,
                "watchdog_decision": watchdog_decision,
                "deterministic_ready_tasks": deterministic_ready_ids,
            },
        },
        "diagnose": {
            "required": ["autonomy_artifact", "canonical_clean", "zero_active_noise"],
            "present": {
                "autonomy_artifact": bool(latest_autonomy),
                "canonical_clean": gates["canonical_clean"],
                "zero_active_noise": gates["zero_active_noise"],
            },
            "evidence": {
                "autonomy": latest_autonomy.get("_artifact_path", ""),
                "canonical_noise": noise,
            },
        },
        "route_and_repair": {
            "required": ["deterministic_ready_task_or_accepted_watchdog_decision"],
            "present": {
                "deterministic_ready_task_or_accepted_watchdog_decision": bool(deterministic_ready)
                or watchdog_decision in accepted_watchdog_decisions,
            },
            "evidence": {
                "watchdog_decision": watchdog_decision,
                "accepted_watchdog_decisions": sorted(accepted_watchdog_decisions),
                "deterministic_ready_tasks": deterministic_ready_ids,
            },
        },
        "evolve": {
            "required": ["durable_memory", "eval_cases", "canary_variants", "shadow_reviews"],
            "present": {
                "durable_memory": gates["durable_memory_present"],
                "eval_cases": gates["eval_cases_present"],
                "canary_variants": gates["evolution_canaries_present"],
                "shadow_reviews": gates["shadow_review_present"],
            },
            "evidence": {
                "lessons": status.get("lessons", 0),
                "trajectories": status.get("trajectories", 0),
                "proposals": status.get("proposals", 0),
                "eval_cases": status.get("eval_cases", 0),
                "decisions": decisions,
                "held_variants": held_variants,
                "shadow_reviews": shadow_reviews,
                "last_evolution_generated": evolution_generated,
            },
        },
        "containment": {
            "required": ["no_active_skill_mutation", "rollback_or_staging_awareness", "no_bad_behavior_rows"],
            "present": {
                "no_active_skill_mutation": gates["no_active_skill_mutation"],
                "rollback_or_staging_awareness": gates["staged_or_rollback_awareness"],
                "no_bad_behavior_rows": gates["no_bad_behavior_rows"],
            },
            "evidence": {
                "active_skill_mutated": active_skill_mutated,
                "promotions": promotions,
                "rollbacks": rollbacks,
                "rolled_back_promotions": rolled_back,
                "bad_behavior_rows": [row.get("run_id", "") for row in bad_rows[:8]],
            },
        },
    }
    for requirement in evidence_requirements.values():
        present = requirement.get("present") if isinstance(requirement.get("present"), dict) else {}
        requirement["passed"] = all(bool(value) for value in present.values())
    gates["evidence_requirements_complete"] = all(
        bool(requirement.get("passed")) for requirement in evidence_requirements.values()
    )
    components = {
        "observe": 20
        if evidence_requirements["observe"]["passed"]
        else 12
        if gates["review_artifacts_present"]
        else 0,
        "diagnose": 20
        if evidence_requirements["diagnose"]["passed"]
        else 10
        if gates["canonical_clean"]
        else 0,
        "route_and_repair": 20
        if evidence_requirements["route_and_repair"]["passed"]
        else 0,
        "evolve": 20
        if evidence_requirements["evolve"]["passed"]
        else 12
        if gates["durable_memory_present"] and gates["eval_cases_present"]
        else 0,
        "containment": 20
        if evidence_requirements["containment"]["passed"]
        else 10
        if gates["no_active_skill_mutation"]
        else 0,
    }
    total = int(sum(components.values()))
    hard_gate_failures = [
        key
        for key in (
            "sidecar_installed",
            "required_skills_present",
            "no_active_skill_mutation",
            "canonical_clean",
            "zero_active_noise",
            "no_bad_behavior_rows",
            "evidence_requirements_complete",
        )
        if not gates.get(key)
    ]
    readiness = (
        "frontier-alive"
        if total >= 95 and not hard_gate_failures
        else "operational"
        if total >= 80 and not hard_gate_failures
        else "warming"
        if total >= 60
        else "needs-repair"
    )
    return {
        "ok": total >= 95 and not hard_gate_failures,
        "kind": "self-improvement-alive-eval",
        "timestamp": int(time.time()),
        "total_score": total,
        "readiness": readiness,
        "verdict": "certified" if total >= 95 and not hard_gate_failures else "blocked",
        "components": components,
        "gates": gates,
        "hard_gate_failures": hard_gate_failures,
        "evidence_requirements": evidence_requirements,
        "status": {
            "lessons": status.get("lessons", 0),
            "trajectories": status.get("trajectories", 0),
            "proposals": status.get("proposals", 0),
            "eval_cases": status.get("eval_cases", 0),
            "decisions": decisions,
            "held_variants": held_variants,
            "shadow_reviews": shadow_reviews,
            "promotions": promotions,
            "rollbacks": rollbacks,
            "rolled_back_promotions": rolled_back,
            "skills": sorted(skills),
        },
        "evidence": {
            "quality": latest_quality.get("_artifact_path", ""),
            "frontier": latest_frontier.get("_artifact_path", ""),
            "autonomy": latest_autonomy.get("_artifact_path", ""),
            "handoff": latest_handoff.get("_artifact_path", ""),
            "burn_in": latest_burn_in.get("_artifact_path", ""),
            "review_council": latest_council.get("_artifact_path", ""),
            "watchdog_decision": watchdog_decision,
            "deterministic_ready_tasks": deterministic_ready_ids,
            "bad_behavior_rows": [row.get("run_id", "") for row in bad_rows[:8]],
            "last_evolution_generated": evolution_generated,
            "last_summary": last_summary,
        },
        "next": (
            "allow autonomous repair/promotion gates to proceed"
            if total >= 95 and not hard_gate_failures
            else "run self-improve curate/evolve plus quality/frontier/autonomy review before promotion"
        ),
    }


def alive_eval(args: argparse.Namespace) -> int:
    root = workspace_root()
    report = self_improvement_alive_report(root, recent_rows=args.recent_rows)
    path = root / "benchmarks" / f"self-improvement-alive-eval-{report['timestamp']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_result(
        root,
        run_id=f"self-improvement-alive-eval-{report['timestamp']}",
        status="keep" if report["ok"] else "blocked",
        target="autoresearch-self-improvement-alive",
        hypothesis="self-improvement should autonomously observe, diagnose, route, canary, review, and contain its own upgrades",
        commit=current_commit(repo_root()),
        notes=(
            f"score={report['total_score']} readiness={report['readiness']} "
            f"failures={','.join(report['hard_gate_failures'])} path={path}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0 if report["ok"] or args.allow_fail else 2


def add_source(args: argparse.Namespace) -> int:
    root = workspace_root()
    sources = root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    queue = sources / "queue.md"
    write_if_missing(queue, "# Research Source Queue\n\n")
    title = args.title or args.source[:80]
    kind = args.kind
    with queue.open("a", encoding="utf-8") as file:
        file.write(f"\n## {title}\n\n")
        file.write(f"- kind: {kind}\n")
        file.write(f"- source: {args.source}\n")
        if args.note:
            file.write(f"- note: {args.note}\n")
    normalize_source_queue(queue)
    print(queue)
    return 0


def source_scout(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    topic = args.topic or "frontier-decode-speed"
    timestamp = int(time.time())
    urls = list(FRONTIER_SOURCE_URLS)
    queue = root / "sources" / "queue.md"
    if queue.exists():
        for match in re.finditer(r"https?://[^\s)>\"]+", queue.read_text(encoding="utf-8", errors="replace")):
            url = match.group(0).rstrip(".,")
            if url not in urls:
                urls.append(url)
    findings: list[dict[str, Any]] = []
    for url in urls[: max(1, int(args.max_sources))]:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower()
        if host not in SOURCE_SCOUT_ALLOWED_HOSTS:
            findings.append({"url": url, "status": "skipped", "reason": f"host not allowlisted: {host}"})
            continue
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "OpenClaw-Autoresearch-SourceScout/1.0"},
            )
            with urllib.request.urlopen(request, timeout=float(args.timeout)) as response:
                raw = response.read(60000).decode("utf-8", errors="replace")
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
            title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
            findings.append(
                {
                    "url": url,
                    "status": "fetched",
                    "host": host,
                    "title": title[:180],
                    "snippet": text[:700],
                }
            )
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            findings.append({"url": url, "status": "unavailable", "host": host, "reason": str(error)[:240]})
    fetched = sum(1 for item in findings if item.get("status") == "fetched")
    report = {
        "ok": True,
        "kind": "source-scout",
        "timestamp": timestamp,
        "topic": topic,
        "fetched": fetched,
        "attempted": len(findings),
        "allowlisted_hosts": sorted(SOURCE_SCOUT_ALLOWED_HOSTS),
        "findings": findings,
        "next": (
            "feed fetched references into frontier-agent-deliberation"
            if fetched
            else "continue with local evidence; source scout failed closed without blocking the loop"
        ),
    }
    path = root / "benchmarks" / f"source-scout-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "source-scout",
            "finding": "source scout gathered bounded allowlisted references for frontier deliberation",
            "evidence": {"path": str(path), "fetched": fetched, "attempted": len(findings), "topic": topic},
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"source-scout-{timestamp}",
        status="keep",
        target="frontier-source-scout",
        hypothesis="frontier deliberation should use current allowlisted references without broad local searches or runtime mutation",
        commit=current_commit(repo_root()),
        notes=f"topic={topic} fetched={fetched} attempted={len(findings)} path={path}",
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0


def model_request(base_url: str, payload: dict[str, Any], timeout: float) -> tuple[float, bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return time.monotonic() - start, response.read()


def stream_model_request(base_url: str, payload: dict[str, Any], timeout: float) -> tuple[float, float, str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    first_token_at = 0.0
    chunks: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            choice = parsed.get("choices", [{}])[0]
            delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
            text = delta.get("content") or delta.get("reasoning_content") or ""
            if text:
                if not first_token_at:
                    first_token_at = time.monotonic()
                chunks.append(str(text))
    finished = time.monotonic()
    return (first_token_at - start if first_token_at else finished - start), finished - start, "".join(chunks)


def memory_snapshot() -> dict[str, int]:
    snapshot = {"free_mb": 0, "compressor_mb": 0, "swap_used_mb": 0}
    try:
        output = subprocess.check_output(["/usr/bin/vm_stat"], text=True, stderr=subprocess.DEVNULL)
        page_size = 16384
        free_pages = speculative_pages = compressor_pages = 0
        for line in output.splitlines():
            if "page size of" in line:
                digits = "".join(ch for ch in line.split("page size of", 1)[1] if ch.isdigit())
                if digits:
                    page_size = int(digits)
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":", 1)[1].strip().rstrip("."))
            elif line.startswith("Pages speculative:"):
                speculative_pages = int(line.split(":", 1)[1].strip().rstrip("."))
            elif line.startswith("Pages occupied by compressor:"):
                compressor_pages = int(line.split(":", 1)[1].strip().rstrip("."))
        snapshot["free_mb"] = int((free_pages + speculative_pages) * page_size / 1048576)
        snapshot["compressor_mb"] = int(compressor_pages * page_size / 1048576)
    except Exception:
        pass
    try:
        output = subprocess.check_output(["/usr/sbin/sysctl", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL)
        if "used = " in output:
            snapshot["swap_used_mb"] = int(float(output.split("used = ", 1)[1].split("M", 1)[0].strip()))
    except Exception:
        pass
    return snapshot


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.8))


def completion_tokens_from_response(parsed: dict[str, Any], content: str) -> tuple[int, str]:
    usage = parsed.get("usage")
    if isinstance(usage, dict):
        value = usage.get("completion_tokens")
        if isinstance(value, int) and value > 0:
            return value, "usage.completion_tokens"
        if isinstance(value, str) and value.isdigit() and int(value) > 0:
            return int(value), "usage.completion_tokens"
    return estimate_tokens(content), "content_estimate"


def log_path() -> Path:
    return Path(os.environ.get("OPENCLAW_MODEL_PROXY_LOG", DEFAULT_PROXY_LOG)).expanduser()


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def read_since(path: Path, offset: int, limit: int = 20000) -> str:
    try:
        size = path.stat().st_size
        start = min(max(0, offset), size)
        if size - start > limit:
            start = size - limit
        with path.open("rb") as file:
            file.seek(start)
            return file.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def parse_generation_log_metrics(text: str) -> dict[str, Any]:
    pattern = re.compile(
        r"(?P<stream>stream\s+)?chat completion: "
        r"prompt=(?P<prompt>\d+) completion=(?P<completion>\d+) "
        r"elapsed=(?P<elapsed>[0-9.]+)s tok_s=(?P<tok_s>[0-9.]+)"
        r"(?: mtp_rounds=(?P<rounds>\d+) mean_accept=(?P<accept>[0-9.]+))?"
    )
    matches = [match for match in pattern.finditer(text)]
    if not matches:
        return {"available": False}
    match = matches[-1]
    result: dict[str, Any] = {
        "available": True,
        "stream": bool(match.group("stream")),
        "prompt_tokens": int(match.group("prompt")),
        "completion_tokens": int(match.group("completion")),
        "elapsed_s": float(match.group("elapsed")),
        "server_tok_s": float(match.group("tok_s")),
    }
    if match.group("rounds"):
        result["mtp_rounds"] = int(match.group("rounds"))
    if match.group("accept"):
        result["mean_accept"] = float(match.group("accept"))
    return result


def parse_generation_log_summary(text: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"(?P<stream>stream\s+)?chat completion: "
        r"prompt=(?P<prompt>\d+) completion=(?P<completion>\d+) "
        r"elapsed=(?P<elapsed>[0-9.]+)s tok_s=(?P<tok_s>[0-9.]+)"
        r"(?: mtp_rounds=(?P<rounds>\d+) mean_accept=(?P<accept>[0-9.]+))?"
    )
    for match in pattern.finditer(text):
        row: dict[str, Any] = {
            "stream": bool(match.group("stream")),
            "prompt_tokens": int(match.group("prompt")),
            "completion_tokens": int(match.group("completion")),
            "elapsed_s": float(match.group("elapsed")),
            "server_tok_s": float(match.group("tok_s")),
        }
        if match.group("rounds"):
            row["mtp_rounds"] = int(match.group("rounds"))
        if match.group("accept"):
            row["mean_accept"] = float(match.group("accept"))
        rows.append(row)
    with_accept = [row for row in rows if "mean_accept" in row]
    tok_values = [float(row["server_tok_s"]) for row in rows]
    round_values = [float(row["mtp_rounds"]) for row in with_accept]
    accept_values = [float(row["mean_accept"]) for row in with_accept]
    return {
        "ok": bool(rows),
        "sample_count": len(rows),
        "mtp_sample_count": len(with_accept),
        "mean_server_tok_s": round(sum(tok_values) / len(tok_values), 3) if tok_values else "",
        "mean_mtp_rounds": round(sum(round_values) / len(round_values), 3) if round_values else "",
        "mean_accept": round(sum(accept_values) / len(accept_values), 3) if accept_values else "",
        "latest": rows[-1] if rows else {},
    }


def result_rows(root: Path) -> list[dict[str, str]]:
    results = root / "results.tsv"
    if not results.exists():
        return []
    headers = RESULTS_HEADER.rstrip("\n").split("\t")
    rows: list[dict[str, str]] = []
    for line in results.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        values = line.split("\t")
        if len(values) == len(headers):
            rows.append(dict(zip(headers, values)))
    return rows


CERTIFICATION_TARGETS = {
    "autoresearch-quality",
    "autoresearch-frontier-eval",
    "autoresearch-frontier-certification",
    "autoresearch-self-improvement-alive",
    "autoresearch-stability-burn-in",
    "autoresearch-sota-autonomy-eval",
    "frontier-autonomy-score",
}


def is_certification_blocked_row(row: dict[str, str]) -> bool:
    return row.get("status") == "blocked" and row.get("target") in CERTIFICATION_TARGETS


def calibration_quantized_gradient_issue(text: object) -> str:
    lower = str(text).lower()
    if CALIBRATION_QUANTIZED_GRADIENT_BLOCKER in lower:
        return CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
    if "calibration quantized drafter gradient blocked" in lower:
        return CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
    if "differentiating quantized drafter weights" in lower:
        return CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
    if "no gradient wrt the quantized weights" in lower:
        return CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
    if "quantizedmatmul::vjp" in lower and "no gradient" in lower:
        return CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
    if "uantized weights" in lower and ("returncode" in lower or "probe_exit:2" in lower):
        return CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
    return ""


def is_known_terminal_calibration_blocked_row(row: dict[str, str]) -> bool:
    if row.get("status") != "blocked":
        return False
    target = row.get("target", "")
    run_id = row.get("run_id", "")
    if "calibration" not in target and "calibration" not in run_id:
        return False
    notes = row.get("notes", "")
    return (
        CALIBRATION_QUANTIZED_GRADIENT_BLOCKER in notes
        or calibration_quantized_gradient_issue(notes) == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
        or (
            run_id.startswith("supervisor-drafter-calibration-memory-stage-")
            and "reason=terminal-blocker" in notes
            and "projection.scales" in notes
            and '"returncode": 2' in notes
        )
        or (
            row.get("run_id", "").startswith("drafter-calibration-memory-stage-micro-step-")
            and "decision=blocked" in notes
            and "failures=1" in notes
        )
    )


def is_known_calibration_memory_blocked_row(row: dict[str, str]) -> bool:
    if row.get("status") != "blocked":
        return False
    target = row.get("target", "")
    run_id = row.get("run_id", "")
    if "calibration" not in target and "calibration" not in run_id:
        return False
    notes = row.get("notes", "").lower()
    if "calibration-memory-gate:after-load" in notes or "calibration memory gate blocked: after-load" in notes:
        return True
    return (
        "calibration_mode=adapter-logit-distillation" in notes
        and "decision=blocked" in notes
        and "probe_exit:2" in notes
        and run_id.startswith("drafter-calibration-memory-stage-")
    )


def is_known_adapter_method_blocked_row(row: dict[str, str]) -> bool:
    if row.get("status") != "blocked":
        return False
    if row.get("target") != "janq-drafter-adapter-method":
        return False
    notes = row.get("notes", "")
    fields = parse_note_fields(notes)
    return (
        row.get("run_id", "").startswith("drafter-adapter-method-contract-")
        and fields.get("ok") in {"False", "false"}
        and fields.get("state")
        in {
            "adapter_calibration_memory_blocked",
            "adapter_calibration_blocked",
            "adapter_calibration_attempted",
        }
    )


def adapter_logit_loop_evidence(root: Path, *, recent_rows: int = 240) -> dict[str, Any]:
    """Return deterministic evidence that the adapter/logit lane is cycling.

    Safety stays deterministic, but strategy should not treat repeated
    blocked-report -> contract -> focused-test rows as progress. Once this
    pattern repeats, direct adapter/logit reseeding is suppressed until a new
    patch candidate or external evidence changes the lane.
    """

    window = result_rows(root)[-max(1, recent_rows) :]
    blocker_reports = 0
    adapter_contracts = 0
    adapter_focused_tests = 0
    for row in window:
        run_id = row.get("run_id", "")
        target = row.get("target", "")
        notes = row.get("notes", "")
        hypothesis = row.get("hypothesis", "").lower()
        if (
            row.get("status") == "blocked"
            and target == "calibration-memory-report"
            and CALIBRATION_QUANTIZED_GRADIENT_BLOCKER in notes
        ):
            blocker_reports += 1
        if run_id.startswith("drafter-adapter-method-contract-") and target == "janq-drafter-adapter-method":
            adapter_contracts += 1
        if run_id.startswith("supervisor-focused-test-") and (
            "adapter/logit" in hypothesis
            or "adapter-logit" in hypothesis
            or "logit-distillation" in hypothesis
        ):
            adapter_focused_tests += 1
    saturated = (
        blocker_reports >= ADAPTER_LOGIT_LOOP_THRESHOLD
        and adapter_contracts >= ADAPTER_LOGIT_LOOP_THRESHOLD
        and adapter_focused_tests >= ADAPTER_LOGIT_LOOP_THRESHOLD
    )
    return {
        "saturated": saturated,
        "blocker_reports": blocker_reports,
        "adapter_contracts": adapter_contracts,
        "adapter_focused_tests": adapter_focused_tests,
    }


def adapter_logit_loop_saturated(root: Path, *, recent_rows: int = 240) -> bool:
    return bool(adapter_logit_loop_evidence(root, recent_rows=recent_rows)["saturated"])


def is_adapter_logit_loop_task(task: dict[str, Any]) -> bool:
    task_id = str(task.get("id", ""))
    action = str(task.get("supervisor_action", ""))
    target = str(task.get("target", ""))
    hypothesis = str(task.get("hypothesis", "")).lower()
    next_action = str(task.get("next_action", "")).lower()
    return (
        action in {"drafter-adapter-method-contract", "calibration-memory-report"}
        or task_id.startswith("drafter-adapter-method-contract-")
        or task_id.startswith("agent-deliberation-adapter-logit-contract-")
        or task_id.startswith("implementation-drafter-adapter-method-")
        or task_id.startswith("calibration-memory-report-")
        or "calibration-memory-report" in next_action
        or (
            action == "focused-test"
            and target == "openclaw/openclaw-mtp-drafter-calibrate.py"
            and (
                "adapter/logit" in hypothesis
                or "adapter-logit" in hypothesis
                or "logit-distillation" in hypothesis
            )
        )
    )


def is_memory_safety_blocked_row(row: dict[str, str]) -> bool:
    if row.get("status") != "blocked":
        return False
    if row.get("target") == "autoresearch-external-change-required":
        return False
    notes = row.get("notes", "").lower()
    crash_terms = (
        "memory gate",
        "memory pressure",
        "memory_guard",
        "memory/crash",
        "memory crash",
        "metal crash",
        "metal error",
        "python crash",
        "mlx crash",
        "crash",
    )
    return any(term in notes for term in crash_terms)


def actionable_blocked_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("status") == "blocked"
        and not is_certification_blocked_row(row)
        and not is_known_terminal_calibration_blocked_row(row)
        and not is_known_calibration_memory_blocked_row(row)
        and not is_known_adapter_method_blocked_row(row)
        and not (
            row.get("target") == "calibration-memory-report"
            and "blocker=none" in row.get("notes", "")
        )
    ]


def unresolved_actionable_blocked_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return blockers that are still relevant inside the review window.

    Failed handoff/bridge rows are useful evidence, but once a later handoff
    audit passes with a clean deterministic task, those older rows are resolved
    debt. Keeping them as active blockers makes the reviewer look broken and
    drives duplicate repair work.
    """
    blocked = actionable_blocked_rows(rows)
    latest_clean_handoff_index = -1
    latest_clean_review_index = -1
    latest_clean_frontier_index = -1
    latest_clean_council_index = -1
    for index, row in enumerate(rows):
        if row.get("target") != "autoresearch-implementation-handoff":
            continue
        if row.get("status") != "keep":
            continue
        fields = parse_note_fields(row.get("notes", ""))
        if fields.get("ok") not in {"True", "true"}:
            continue
        try:
            score = float(fields.get("score", "0") or 0)
        except ValueError:
            score = 0.0
        if score >= 90:
            latest_clean_handoff_index = index
    for index, row in enumerate(rows):
        if row.get("target") != "autoresearch-quality" or row.get("status") != "keep":
            continue
        fields = parse_note_fields(row.get("notes", ""))
        try:
            score = float(fields.get("score", "0") or 0)
        except ValueError:
            score = 0.0
        if score >= 90 and fields.get("verdict") in {"healthy", "converged-below-target", "exhaustion-candidate"}:
            latest_clean_review_index = index
    for index, row in enumerate(rows):
        if row.get("status") != "keep":
            continue
        target = row.get("target", "")
        fields = parse_note_fields(row.get("notes", ""))
        if target == "autoresearch-frontier-eval":
            try:
                overall = float(fields.get("overall", "0") or 0)
            except ValueError:
                overall = 0.0
            if overall >= 9.8 and fields.get("readiness") == "frontier":
                latest_clean_frontier_index = index
        elif target == "frontier-autonomy-score":
            try:
                score = float(fields.get("score", "0") or 0)
            except ValueError:
                score = 0.0
            if score >= 99 and fields.get("decision") == "continue":
                latest_clean_frontier_index = index
        elif target == "autoresearch-review-council":
            if fields.get("ok") in {"True", "true"} and fields.get("decision") == "continue":
                latest_clean_council_index = index

    latest_clean_checkpoint_index = max(
        latest_clean_handoff_index,
        latest_clean_review_index,
        latest_clean_frontier_index,
        latest_clean_council_index,
    )
    if latest_clean_checkpoint_index < 0:
        return blocked

    latest_progress_index = latest_clean_checkpoint_index
    for index, row in enumerate(rows):
        if row.get("status") != "keep":
            continue
        if row.get("target") in CERTIFICATION_TARGETS:
            continue
        if row.get("target") in {"synthesis", "autopilot"}:
            continue
        latest_progress_index = max(latest_progress_index, index)

    latest_adapter_route_index = -1
    for index, row in enumerate(rows):
        if row.get("status") != "keep":
            continue
        run_id = row.get("run_id", "")
        if run_id.startswith(("drafter-bottleneck-review-", "drafter-adapter-method-contract-")):
            latest_adapter_route_index = index

    unresolved: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if row not in blocked:
            continue
        run_id = row.get("run_id", "")
        target = row.get("target", "")
        notes = row.get("notes", "")
        if (
            index < latest_adapter_route_index
            and run_id.startswith("drafter-adapter-method-contract-")
            and target == "janq-drafter-adapter-method"
        ):
            continue
        if (
            index < latest_progress_index
            and target == "autopilot"
            and "model-bound research turn deferred" in notes
        ):
            continue
        if index < latest_clean_checkpoint_index and (
            row.get("target") == "autoresearch-implementation-handoff"
            or row.get("target") == "autoresearch-review-council"
            or row.get("target") == "frontier-autonomy-score"
            or row.get("target") == "autoresearch-self-improvement-alive"
            or row.get("run_id", "").startswith("supervisor-implementation-bridge-")
        ):
            continue
        if index < latest_progress_index and (
            (
                row.get("target") == "synthesis"
                and "seeded_tasks=0" in row.get("notes", "")
            )
            or (
                row.get("target") == "autopilot"
                and "supervisor synthesis" in row.get("notes", "")
            )
        ):
            continue
        unresolved.append(row)
    return unresolved


def canonical_autoresearch_state(root: Path, *, recent_rows: int = 120, target_tps: float = 30.0) -> dict[str, Any]:
    """Classify the loop once so reviewers do not infer conflicting states."""
    ensure_research_state(root)
    rows = result_rows(root)
    recent = rows[-max(1, int(recent_rows)) :]
    raw_blocked = unresolved_actionable_blocked_rows(recent)
    tasks = read_jsonl(root / "tasks.jsonl")
    ready = [task for task in tasks if task.get("status", "ready") in {"ready", "rework"}]
    deterministic = [task for task in ready if is_deterministic_research_task(task)]
    deterministic_ids = [str(task.get("id", "")) for task in deterministic]
    ready_lanes = sorted({str(task.get("lane", "")) for task in ready if str(task.get("lane", ""))})
    exhausted = set(exhausted_lanes(root))
    frontier_lanes = {"runtime-overhead", "drafter-alignment", "frontier-dflash", "frontier-expansion"}
    breakthrough_lanes = sorted(lane for lane in ready_lanes if lane in frontier_lanes and lane not in exhausted)
    decode_values = [
        float(signal["wall_decode_tps"])
        for signal in (
            decode_measurement_signal(row)
            for row in recent
            if row.get("status") == "keep" and row.get("target") == "decode-sample"
        )
        if not signal["contaminated"] and signal.get("wall_decode_tps") is not None
    ]
    decode_mean = mean_float(decode_values)
    decode_spread = round(max(decode_values) - min(decode_values), 3) if len(decode_values) >= 2 else None
    plateau = bool(
        len(decode_values) >= 5
        and decode_mean is not None
        and decode_mean < target_tps
        and decode_spread is not None
        and decode_spread <= 2.0
    )
    terminal_synthesis_rows = [
        row
        for row in recent
        if row.get("target") == "synthesis-terminal" or "terminal_no_work=True" in row.get("notes", "")
    ]
    bridge_zero = recent_empty_bridge_rows(root, recent, recent_rows=len(recent) or 1)
    memory_blocks = [row for row in raw_blocked if is_memory_safety_blocked_row(row)]
    repair_ready = [
        task_id
        for task_id in deterministic_ids
        if any(
            fragment in task_id
            for fragment in (
                "handoff-audit-",
                "agent-deliberation",
                "frontier-repair-",
                "review-drafter-calibration-canary",
                "review-janq-drafter-fit",
                "runtime-overhead",
                "frontier-expansion",
                "trace-distillation-adapter-bridge",
                "drafter-adapter-method-contract",
                "implementation-drafter-adapter-method",
                "calibration-memory-report",
                "exhaustion",
            )
        )
    ]
    routed_blockers: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    for row in raw_blocked:
        notes = row.get("notes", "").lower()
        target = row.get("target", "")
        run_id = row.get("run_id", "")
        is_memory_block = is_memory_safety_blocked_row(row)
        external_blocker_routed = (
            target == "autoresearch-external-change-required"
            and bool(repair_ready or breakthrough_lanes or deterministic_ids)
        )
        dflash_exhausted = target == "frontier-dflash" and "frontier-dflash" in exhausted
        causal_routed = (
            target == "autoresearch-causal-review"
            and ("regression=true" in notes or "low_confidence" in notes or "low-confidence" in notes)
            and bool(breakthrough_lanes or repair_ready)
        )
        deterministic_routed = bool(repair_ready and not is_memory_block)
        dflash_compatibility_routed = (
            run_id.startswith("dflash-compatibility-gate-")
            and ("draft_model_type_mismatch" in notes or "decision=blocked" in notes)
            and bool("frontier-dflash" in exhausted or repair_ready or breakthrough_lanes)
        )
        implementation_model_guard_routed = (
            run_id.startswith("supervisor-implementation-guard-")
            and target.startswith("openclaw/")
            and any(
                marker in notes
                for marker in (
                    "malformed hidden/tool output",
                    "tool result synthesis grace",
                    "tool result cap",
                    "implementation task requires deterministic patch-executor path",
                    "turn timeout",
                )
            )
            and any_task_has_prefix(root, "implementation-drafter-adapter-method-")
        )
        if not is_memory_block and (
            deterministic_routed
            or dflash_exhausted
            or causal_routed
            or dflash_compatibility_routed
            or implementation_model_guard_routed
            or external_blocker_routed
        ):
            routed_blockers.append(row)
        else:
            unresolved.append(row)
    if memory_blocks:
        state = "memory_guarded"
    elif breakthrough_lanes:
        state = "breakthrough_lane_active"
    elif repair_ready:
        state = "prerequisite_needed"
    elif plateau:
        state = "plateau_detected"
    elif not ready and exhausted:
        state = "blocked_until_external_change"
    elif not unresolved:
        state = "frontier_healthy"
    else:
        state = "needs_repair"
    terminal_state_exhausted = not ready and bool(exhausted)
    routed_terminal_synthesis_rows = (
        terminal_synthesis_rows
        if deterministic_ids or breakthrough_lanes or repair_ready or terminal_state_exhausted
        else []
    )
    unresolved_terminal_synthesis_rows = (
        [] if routed_terminal_synthesis_rows else terminal_synthesis_rows
    )
    routed_bridge_zero_rows = (
        bridge_zero
        if deterministic_ids or breakthrough_lanes or repair_ready or routed_blockers or terminal_state_exhausted
        else []
    )
    unresolved_bridge_zero_rows = [] if routed_bridge_zero_rows else bridge_zero
    noise = {
        "unresolved_blocked_rows": len(unresolved),
        "terminal_synthesis_rows": len(unresolved_terminal_synthesis_rows),
        "bridge_zero_rows": len(unresolved_bridge_zero_rows),
        "memory_blocks": len(memory_blocks),
    }
    resolved_debt = {
        "routed_blocked_rows": len(routed_blockers),
        "routed_terminal_synthesis_rows": len(routed_terminal_synthesis_rows),
        "routed_bridge_zero_rows": len(routed_bridge_zero_rows),
    }
    return {
        "version": 1,
        "timestamp": int(time.time()),
        "state": state,
        "clean": not unresolved and not memory_blocks,
        "ready_tasks": len(ready),
        "deterministic_ready_tasks": deterministic_ids[:12],
        "ready_lanes": ready_lanes,
        "breakthrough_lanes": breakthrough_lanes,
        "exhausted_lanes": sorted(exhausted),
        "decode_mean_tps": decode_mean,
        "decode_sample_count": len(decode_values),
        "decode_spread_tps": decode_spread,
        "plateau_detected": plateau,
        "unresolved_blocked_rows": unresolved,
        "routed_blocked_rows": routed_blockers,
        "noise": noise,
        "resolved_debt": resolved_debt,
        "next": (
            "run deterministic prerequisite"
            if repair_ready
            else "run breakthrough lane"
            if breakthrough_lanes
            else "declare external blocker or add new candidate"
            if state == "blocked_until_external_change"
            else "continue measurement"
        ),
    }


def clipped_text(value: object, limit: int = 220) -> str:
    text = str(value or "").replace("\n", " ").replace("\t", " ").strip()
    return text[:limit].rstrip()


def latest_keep_row(rows: list[dict[str, str]], target: str) -> dict[str, str]:
    for row in reversed(rows):
        if row.get("status") == "keep" and row.get("target") == target:
            return row
    return {}


def latest_keep_row_prefix(rows: list[dict[str, str]], prefix: str) -> dict[str, str]:
    for row in reversed(rows):
        if row.get("status") == "keep" and row.get("run_id", "").startswith(prefix):
            return row
    return {}


def restart_context_payload(root: Path, *, recent_rows: int = 120) -> dict[str, Any]:
    """Build a compact restart handoff so new sessions do not rediscover old work."""
    ensure_research_state(root)
    rows = result_rows(root)
    recent = rows[-max(1, int(recent_rows)) :]
    canonical = canonical_autoresearch_state(root, recent_rows=recent_rows)
    tasks = read_jsonl(root / "tasks.jsonl")
    ready_tasks = [task for task in tasks if task.get("status", "ready") in {"ready", "rework"}]
    deterministic_ready = [task for task in ready_tasks if is_deterministic_research_task(task)]
    exhausted = exhausted_lanes(root)
    latest_decode = latest_keep_row(recent, "decode-sample")
    latest_mtp = latest_keep_row(recent, "mtp-acceptance-report")
    latest_quality = latest_keep_row(recent, "autoresearch-quality")
    latest_frontier = latest_keep_row(recent, "autoresearch-frontier-eval")
    latest_autonomy = latest_keep_row(recent, "frontier-autonomy-score")
    latest_handoff = latest_keep_row(recent, "autoresearch-implementation-handoff")
    latest_adapter_contract = latest_keep_row_prefix(recent, "drafter-adapter-method-contract-")
    return {
        "version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workspace": str(root),
        "mission": (
            "Improve normal OpenClaw TUI decode speed for Gemma-4-31B-JANG/JANQ while preserving "
            "model choice, stability, tool/reasoning guards, and zero active noise."
        ),
        "primary_metric": "normal OpenClaw TUI decode tok/s",
        "target_tps": 30,
        "aspirational_tps": "50-70 if hardware/runtime/drafter evidence supports it",
        "canonical_state": {
            "state": canonical.get("state"),
            "clean": canonical.get("clean"),
            "next": canonical.get("next"),
            "noise": canonical.get("noise", {}),
            "decode_mean_tps": canonical.get("decode_mean_tps"),
            "ready_lanes": canonical.get("ready_lanes", []),
            "breakthrough_lanes": canonical.get("breakthrough_lanes", []),
            "exhausted_lanes": canonical.get("exhausted_lanes", []),
        },
        "latest_signals": {
            "decode_sample": {
                "decode_tps": latest_decode.get("decode_tps", ""),
                "notes": clipped_text(latest_decode.get("notes", "")),
            },
            "mtp_acceptance": {"notes": clipped_text(latest_mtp.get("notes", ""))},
            "quality": {"notes": clipped_text(latest_quality.get("notes", ""))},
            "frontier_eval": {"notes": clipped_text(latest_frontier.get("notes", ""))},
            "autonomy": {"notes": clipped_text(latest_autonomy.get("notes", ""))},
            "handoff": {"notes": clipped_text(latest_handoff.get("notes", ""))},
            "adapter_contract": {"notes": clipped_text(latest_adapter_contract.get("notes", ""))},
        },
        "ready_tasks": [
            {
                "id": str(task.get("id", "")),
                "lane": str(task.get("lane", "")),
                "supervisor_action": str(task.get("supervisor_action", "")),
                "target": str(task.get("target", "")),
                "priority": task.get("priority", ""),
                "hypothesis": clipped_text(task.get("hypothesis", ""), 260),
            }
            for task in deterministic_ready[:8]
        ],
        "do_not_rediscover": [
            "Do not re-run broad DFlash compatibility unless the draft candidate changes.",
            "Do not repeat block-size sweeps as progress after block 2 has converged unless new evidence appears.",
            "Do not re-record terminal adapter/calibration blockers as active noise; route them to an executable canary or candidate change.",
            "Do not treat clean-but-no-ready-work as success; seed a deterministic next action or declare an external blocker.",
        ],
        "exhausted_lane_reasons": {
            lane: clipped_text(data.get("reason", ""), 260)
            for lane, data in exhausted.items()
            if isinstance(data, dict)
        },
        "restart_instructions": [
            "Read this file before SUMMARY.md on every fresh research session.",
            "Prefer the first ready deterministic task over synthesis.",
            "If there are no ready deterministic tasks, run quality-review then synthesize --kind frontier once.",
            "Record new evidence instead of repeating the last terminal report.",
        ],
    }


def render_run_memory(memory: dict[str, Any]) -> str:
    state = memory.get("canonical_state", {}) if isinstance(memory.get("canonical_state"), dict) else {}
    signals = memory.get("latest_signals", {}) if isinstance(memory.get("latest_signals"), dict) else {}
    lines = [
        "# OpenClaw Speed Research Run Memory",
        "",
        "This is the compact restart handoff. Read it before `SUMMARY.md` so a new session continues from the current frontier instead of rediscovering old work.",
        "",
        "## Mission",
        "",
        f"- {memory.get('mission', '')}",
        f"- primary_metric: {memory.get('primary_metric', '')}",
        f"- target_tps: {memory.get('target_tps', '')}",
        f"- aspirational_tps: {memory.get('aspirational_tps', '')}",
        "",
        "## Current State",
        "",
        f"- state: {state.get('state', '')}",
        f"- clean: {state.get('clean', '')}",
        f"- next: {state.get('next', '')}",
        f"- decode_mean_tps: {state.get('decode_mean_tps', '')}",
        f"- ready_lanes: {', '.join(state.get('ready_lanes', []) or []) or 'none'}",
        f"- breakthrough_lanes: {', '.join(state.get('breakthrough_lanes', []) or []) or 'none'}",
        f"- exhausted_lanes: {', '.join(state.get('exhausted_lanes', []) or []) or 'none'}",
        f"- active_noise: {json.dumps(state.get('noise', {}), sort_keys=True)}",
        "",
        "## Latest Signals",
        "",
    ]
    for name in ("decode_sample", "mtp_acceptance", "quality", "frontier_eval", "autonomy", "handoff", "adapter_contract"):
        signal = signals.get(name, {}) if isinstance(signals.get(name), dict) else {}
        bits = []
        if signal.get("decode_tps"):
            bits.append(f"decode_tps={signal['decode_tps']}")
        if signal.get("notes"):
            bits.append(str(signal["notes"]))
        lines.append(f"- {name}: {'; '.join(bits) if bits else 'no recent signal'}")
    lines.extend(["", "## Ready Deterministic Tasks", ""])
    ready_tasks = memory.get("ready_tasks", []) if isinstance(memory.get("ready_tasks"), list) else []
    if ready_tasks:
        for task in ready_tasks:
            lines.append(
                f"- {task.get('id', '')}: lane={task.get('lane', '')} action={task.get('supervisor_action', '')} "
                f"target={task.get('target', '')} hypothesis={task.get('hypothesis', '')}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Do Not Re-discover", ""])
    for item in memory.get("do_not_rediscover", []) or []:
        lines.append(f"- {item}")
    exhausted_reasons = memory.get("exhausted_lane_reasons", {})
    if isinstance(exhausted_reasons, dict) and exhausted_reasons:
        lines.extend(["", "## Exhausted Lane Reasons", ""])
        for lane, reason in sorted(exhausted_reasons.items()):
            lines.append(f"- {lane}: {reason or 'recorded exhausted'}")
    lines.extend(["", "## Restart Instructions", ""])
    for item in memory.get("restart_instructions", []) or []:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_restart_context(root: Path, *, recent_rows: int = 120) -> dict[str, Any]:
    memory = restart_context_payload(root, recent_rows=recent_rows)
    write_if_changed(root / "restart-context.json", json.dumps(memory, indent=2, sort_keys=True) + "\n")
    write_if_changed(root / "RUN_MEMORY.md", render_run_memory(memory))
    return memory


def compact_workspace(root: Path, *, recent_rows: int = 24) -> dict[str, Any]:
    ensure_research_state(root)
    rows = result_rows(root)
    recent = rows[-recent_rows:]
    recent_path = root / "results-recent.tsv"
    headers = RESULTS_HEADER.rstrip("\n").split("\t")
    recent_text = RESULTS_HEADER
    for row in recent:
        recent_text += "\t".join(row.get(header, "") for header in headers) + "\n"
    write_if_changed(recent_path, recent_text)

    targets: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("status") == "keep":
            targets[row.get("target", "unknown")] = row
    ready_tasks = [task for task in read_jsonl(root / "tasks.jsonl") if task.get("status", "ready") in {"ready", "rework"}]
    blocked_tasks = [task for task in read_jsonl(root / "tasks.jsonl") if task.get("status") == "blocked"]
    summary_lines = [
        "# OpenClaw Speed Research Summary",
        "",
        "This file is the compact entrypoint for autoresearch. Read this instead of the full results.tsv ledger.",
        "",
        "## Latest Metrics",
        "",
    ]
    for target in sorted(targets):
        row = targets[target]
        bits = []
        if row.get("ttft_s"):
            bits.append(f"ttft_s={row['ttft_s']}")
        if row.get("decode_tps"):
            bits.append(f"decode_tps={row['decode_tps']}")
        if row.get("wall_s"):
            bits.append(f"wall_s={row['wall_s']}")
        notes = row.get("notes", "")[:160]
        summary_lines.append(f"- {target}: {' '.join(bits) or 'recorded'} {notes}".rstrip())
    summary_lines.extend(["", "## Queue", ""])
    summary_lines.append(f"- ready_tasks={len(ready_tasks)}")
    summary_lines.append(f"- blocked_tasks={len(blocked_tasks)}")
    for task in ready_tasks[:5]:
        summary_lines.append(
            f"- ready: {task.get('id', 'task')} type={task.get('task_type', 'benchmark')} metric={task.get('metric', 'unknown')}"
        )
    summary_lines.extend(
        [
            "",
            "## Files",
            "",
            f"- recent results: {recent_path}",
            f"- run memory: {root / 'RUN_MEMORY.md'}",
            f"- restart context: {root / 'restart-context.json'}",
            f"- full ledger: {root / 'results.tsv'}",
            f"- tasks: {root / 'tasks.jsonl'}",
        ]
    )
    write_if_changed(root / "SUMMARY.md", "\n".join(summary_lines).rstrip() + "\n")
    memory = write_restart_context(root, recent_rows=max(120, recent_rows))
    return {
        "ok": True,
        "recent_rows": len(recent),
        "ready_tasks": len(ready_tasks),
        "blocked_tasks": len(blocked_tasks),
        "run_memory": str(root / "RUN_MEMORY.md"),
        "restart_context": str(root / "restart-context.json"),
        "memory_state": (memory.get("canonical_state") or {}).get("state"),
    }


def float_values(rows: list[dict[str, str]], target: str, key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("status") != "keep" or row.get("target") != target:
            continue
        try:
            text = row.get(key, "").strip()
            if text:
                values.append(float(text))
        except ValueError:
            continue
    return values


def mean_value(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def parse_note_fields(notes: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in notes.replace(",", " ").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def calibration_mode(value: Any = None) -> str:
    mode = str(value or CALIBRATION_DIRECT_MODE)
    return mode if mode in CALIBRATION_MODES else CALIBRATION_DIRECT_MODE


def calibration_task_mode(task: dict[str, Any]) -> str:
    return calibration_mode(task.get("calibration_mode"))


def calibration_memory_stage_name(task: dict[str, Any]) -> str:
    if str(task.get("supervisor_action", "")) != "drafter-calibration-memory-stage":
        return ""
    stage = str(task.get("stage", ""))
    if stage:
        return stage
    task_id = str(task.get("id", ""))
    prefix = "drafter-calibration-memory-stage-"
    if task_id.startswith(prefix):
        remainder = task_id[len(prefix) :]
        for candidate in CALIBRATION_MEMORY_STAGES:
            if remainder.startswith(candidate):
                return candidate
    return ""


def calibration_memory_stage_key(task: dict[str, Any]) -> str:
    stage = calibration_memory_stage_name(task)
    if not stage:
        return ""
    return f"{calibration_task_mode(task)}:{stage}"


def active_calibration_memory_stage_tasks(root: Path) -> list[dict[str, Any]]:
    return [
        task
        for task in read_jsonl(root / "tasks.jsonl")
        if task.get("status", "ready") in {"ready", "rework"} and calibration_memory_stage_name(task)
    ]


def completed_calibration_memory_stages(
    root: Path,
    *,
    recent_rows: int = 600,
    calibration_mode_filter: str = CALIBRATION_DIRECT_MODE,
) -> set[str]:
    mode_filter = calibration_mode(calibration_mode_filter)
    completed: set[str] = set()
    for row in result_rows(root)[-max(1, recent_rows) :]:
        if row.get("status") != "keep":
            continue
        run_id = row.get("run_id", "")
        notes = row.get("notes", "")
        fields = parse_note_fields(notes)
        row_mode = calibration_mode(fields.get("calibration_mode"))
        if row_mode != mode_filter:
            continue
        for stage in CALIBRATION_MEMORY_STAGES:
            if run_id.startswith(f"drafter-calibration-memory-stage-{stage}-"):
                completed.add(stage)
            elif run_id.startswith("supervisor-drafter-calibration-memory-stage-"):
                if fields.get("stage") == stage:
                    completed.add(stage)
    return completed


def completed_calibration_memory_stage_keys(root: Path, *, recent_rows: int = 600) -> set[str]:
    keys: set[str] = set()
    for mode in CALIBRATION_MODES:
        keys.update(
            f"{mode}:{stage}"
            for stage in completed_calibration_memory_stages(
                root,
                recent_rows=recent_rows,
                calibration_mode_filter=mode,
            )
        )
    return keys


def first_seedable_calibration_memory_stage(
    root: Path,
    *,
    calibration_mode_filter: str = CALIBRATION_DIRECT_MODE,
) -> str:
    mode_filter = calibration_mode(calibration_mode_filter)
    active = {
        calibration_memory_stage_name(task)
        for task in active_calibration_memory_stage_tasks(root)
        if calibration_task_mode(task) == mode_filter
    }
    if active:
        return ""
    completed = completed_calibration_memory_stages(
        root,
        calibration_mode_filter=mode_filter,
    )
    for stage in CALIBRATION_MEMORY_STAGES:
        if stage not in completed:
            return stage
    return ""


def calibration_stage_duplicate_count(root: Path) -> int:
    counts: dict[str, int] = {}
    for task in active_calibration_memory_stage_tasks(root):
        stage = calibration_memory_stage_key(task)
        counts[stage] = counts.get(stage, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def calibration_stage_task_issue(task: dict[str, Any]) -> str:
    """Return the terminal issue recorded on a staged calibration task."""
    summary = task.get("supervisor_summary") if isinstance(task.get("supervisor_summary"), dict) else {}
    text = " ".join(
        str(part)
        for part in (
            task.get("blocked_reason", ""),
            summary.get("reason", "") if isinstance(summary, dict) else "",
            summary.get("memory_gate_issue", "") if isinstance(summary, dict) else "",
            summary.get("output_tail", "") if isinstance(summary, dict) else "",
            json.dumps(summary, sort_keys=True) if isinstance(summary, dict) else "",
        )
    ).lower()
    gradient_issue = calibration_quantized_gradient_issue(text)
    if gradient_issue:
        return gradient_issue
    if "calibration-memory-gate:after-load" in text or "calibration memory gate blocked: after-load" in text:
        return "calibration-memory-after-load"
    if "missing-runtime-module:mlx_vlm.speculative" in text or "no module named 'mlx_vlm.speculative'" in text:
        return "calibration-runtime-missing-speculative"
    if task.get("status") == "blocked":
        return "calibration-stage-blocked"
    return ""


def latest_calibration_stage_issue(root: Path, *, calibration_mode_filter: str) -> str:
    """Return the latest blocked staged-calibration issue for one calibration mode."""
    mode_filter = calibration_mode(calibration_mode_filter)
    for task in reversed(read_jsonl(root / "tasks.jsonl")):
        if str(task.get("supervisor_action", "")) != "drafter-calibration-memory-stage":
            continue
        if calibration_task_mode(task) != mode_filter:
            continue
        status = task.get("status", "ready")
        if status == "blocked":
            return calibration_stage_task_issue(task)
        if status == "done":
            return ""
    return ""


def compact_duplicate_calibration_stage_tasks(root: Path) -> int:
    tasks = read_jsonl(root / "tasks.jsonl")
    keep_by_stage: dict[str, int] = {}
    changed = False
    compacted = 0
    for index, task in enumerate(tasks):
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        stage = calibration_memory_stage_key(task)
        if not stage:
            continue
        if stage not in keep_by_stage:
            keep_by_stage[stage] = index
            continue
        task["status"] = "blocked"
        task["blocked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        task["supervisor_summary"] = {
            "reason": "duplicate calibration memory stage suppressed",
            "stage": calibration_memory_stage_name(task),
            "calibration_mode": calibration_task_mode(task),
            "kept_task_id": tasks[keep_by_stage[stage]].get("id", ""),
        }
        compacted += 1
        changed = True
    if changed:
        write_jsonl(root / "tasks.jsonl", tasks)
        append_jsonl(
            root / "findings.jsonl",
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "task_id": "calibration-stage-dedupe",
                "finding": "suppressed duplicate ready calibration memory-stage tasks",
                "evidence": {"compacted": compacted, "kept_stages": sorted(keep_by_stage)},
                "next": "execute the oldest remaining stage and advance only after a keep artifact",
            },
        )
    return compacted


def compact_stale_calibration_canary_tasks(root: Path) -> int:
    """Complete canary tasks once their downstream memory-stage is already ready."""
    active_stages = active_calibration_memory_stage_tasks(root)
    if not active_stages:
        return 0
    tasks = read_jsonl(root / "tasks.jsonl")
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    compacted = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        task_id = str(task.get("id", ""))
        action = str(task.get("supervisor_action", ""))
        if action != "drafter-calibration-canary" and "drafter-calibration-canary" not in task_id:
            continue
        task["status"] = "done"
        task["completed_at"] = now
        task["supervisor_summary"] = {
            "reason": "stale calibration canary suppressed because memory-stage is already ready",
            "active_stage_task_ids": [str(item.get("id", "")) for item in active_stages],
        }
        compacted += 1
    if compacted:
        write_jsonl(root / "tasks.jsonl", tasks)
        append_jsonl(
            root / "findings.jsonl",
            {
                "timestamp": now,
                "task_id": "calibration-canary-compaction",
                "finding": "completed stale calibration canary tasks after a downstream memory-stage was queued",
                "evidence": {"compacted": compacted},
                "next": "execute the queued calibration memory-stage",
            },
        )
    return compacted


def compact_terminal_calibration_tasks(root: Path) -> int:
    """Complete queued calibration retries after a terminal calibration blocker is known."""
    blocker = recent_calibration_run_hard_blocker(root, recent_rows=240)
    if blocker not in CALIBRATION_CANARY_TERMINAL_BLOCKERS:
        return 0
    distillation_failed = trace_distillation_proof_failed(root, recent_rows=240)
    tasks = read_jsonl(root / "tasks.jsonl")
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    compacted = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        task_id = str(task.get("id", ""))
        action = str(task.get("supervisor_action", ""))
        if not (
            action in {"drafter-calibration-canary", "drafter-calibration-memory-stage", "drafter-calibration-run"}
            or "drafter-calibration-canary" in task_id
            or "drafter-calibration-memory-stage" in task_id
            or "drafter-calibration-run" in task_id
        ):
            continue
        if is_trace_distillation_task(task) and not distillation_failed:
            continue
        task["status"] = "done"
        task["completed_at"] = now
        task["supervisor_summary"] = {
            "reason": "terminal calibration blocker suppressed queued retry",
            "blocker": blocker,
            "trace_distillation_proof_failed": distillation_failed,
        }
        compacted += 1
    if compacted:
        write_jsonl(root / "tasks.jsonl", tasks)
        append_jsonl(
            root / "findings.jsonl",
            {
                "timestamp": now,
                "task_id": "terminal-calibration-compaction",
                "finding": "completed queued calibration retries after a terminal calibration blocker was classified",
                "evidence": {"compacted": compacted, "blocker": blocker},
                "next": "route to a root-cause report or a changed trainable-adapter calibration method",
            },
        )
    return compacted


def compact_direct_calibration_canaries_after_bottleneck(root: Path, *, recent_rows: int = 240) -> int:
    state = drafter_bottleneck_state(root, recent_rows=recent_rows)
    if state["state"] == "no_terminal_quantized_blocker":
        return 0
    tasks = read_jsonl(root / "tasks.jsonl")
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    compacted = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        action = str(task.get("supervisor_action", ""))
        task_id = str(task.get("id", ""))
        if action != "drafter-calibration-canary" and "drafter-calibration-canary" not in task_id:
            continue
        if calibration_task_mode(task) == CALIBRATION_ADAPTER_MODE:
            continue
        task["status"] = "done"
        task["completed_at"] = now
        task["supervisor_summary"] = {
            "reason": "direct calibration canary compacted after JANQ bottleneck routed to adapter/logit-distillation",
            "bottleneck_state": state["state"],
            "next_step": state["next_step"],
        }
        compacted += 1
    if compacted:
        write_jsonl(root / "tasks.jsonl", tasks)
        append_jsonl(
            root / "findings.jsonl",
            {
                "timestamp": now,
                "task_id": "direct-calibration-canary-compaction",
                "finding": "completed stale direct calibration canaries after the JANQ bottleneck selected a changed adapter/logit route",
                "evidence": {"compacted": compacted, "state": state["state"], "next_step": state["next_step"]},
                "next": "run_adapter_logit_distillation_route",
            },
        )
    return compacted


def runtime_overhead_repeated_clean(
    root: Path,
    rows: list[dict[str, str]] | None = None,
    *,
    recent_rows: int = 160,
) -> bool:
    if len(recent_clean_runtime_overhead_maps(root, rows, recent_rows=recent_rows)) < 2:
        return False
    artifact = measurement_artifact_analysis(root, recent_rows=recent_rows)
    return not bool(artifact.get("artifact_suspected"))


def compact_repeated_runtime_overhead_tasks(
    root: Path,
    rows: list[dict[str, str]] | None = None,
    *,
    recent_rows: int = 160,
) -> int:
    if not runtime_overhead_repeated_clean(root, rows, recent_rows=recent_rows):
        return 0
    tasks = read_jsonl(root / "tasks.jsonl")
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    compacted = 0
    for task in tasks:
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        lane = str(task.get("lane", ""))
        task_id = str(task.get("id", ""))
        action = str(task.get("supervisor_action", ""))
        if "contamination" in task_id:
            continue
        if lane != "runtime-overhead" and action != "runtime-overhead-map":
            continue
        task["status"] = "done"
        task["completed_at"] = now
        task["supervisor_summary"] = {
            "reason": "runtime-overhead task compacted after repeated clean maps; wait for fresh contamination evidence",
            "clean_runtime_maps": len(recent_clean_runtime_overhead_maps(root, rows, recent_rows=recent_rows)),
        }
        compacted += 1
    if compacted:
        write_jsonl(root / "tasks.jsonl", tasks)
        append_jsonl(
            root / "findings.jsonl",
            {
                "timestamp": now,
                "task_id": "runtime-overhead-clean-map-compaction",
                "finding": "completed stale runtime-overhead tasks after repeated clean maps ruled out the lane",
                "evidence": {"compacted": compacted},
                "next": "route to MTP acceptance yield, adapter/logit drafter fit, or a new candidate path",
            },
        )
    return compacted


def upsert_tasks(root: Path, tasks: list[dict[str, Any]]) -> int:
    path = root / "tasks.jsonl"
    existing = read_jsonl(path)
    changed = False
    active_ids: set[str] = set()
    for task in existing:
        task_id = str(task.get("id", ""))
        if not task_id or task.get("status", "ready") not in {"ready", "rework"}:
            continue
        if task_id in active_ids:
            task["status"] = "done"
            task["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            task["supervisor_summary"] = {"reason": "duplicate active task id compacted by upsert"}
            changed = True
        else:
            active_ids.add(task_id)
    existing_by_id: dict[str, int] = {}
    for index, task in enumerate(existing):
        task_id = str(task.get("id", ""))
        if not task_id:
            continue
        if task_id not in existing_by_id or task.get("status", "ready") in {"ready", "rework"}:
            existing_by_id[task_id] = index
    active_stage_keys = {
        calibration_memory_stage_key(task)
        for task in existing
        if task.get("status", "ready") in {"ready", "rework"} and calibration_memory_stage_key(task)
    }
    completed_stages = completed_calibration_memory_stage_keys(root)
    additions = 0
    for task in tasks:
        task_id = str(task.get("id", ""))
        if not task_id:
            continue
        stage_key = calibration_memory_stage_key(task)
        if stage_key and (stage_key in active_stage_keys or stage_key in completed_stages):
            continue
        if task_id not in existing_by_id:
            existing.append(task)
            existing_by_id[task_id] = len(existing) - 1
            if stage_key:
                active_stage_keys.add(stage_key)
            additions += 1
            changed = True
            continue
        index = existing_by_id[task_id]
        current = existing[index]
        if current.get("status", "ready") not in {"ready", "rework"}:
            if task.get("revive_blocked"):
                revived = {
                    **current,
                    **task,
                    "status": task.get("status", "ready"),
                    "revived_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                revived.pop("blocked_at", None)
                revived.pop("blocked_reason", None)
                revived.pop("completed_at", None)
                revived.pop("supervisor_summary", None)
                existing[index] = revived
                additions += 1
                changed = True
            continue
        merged = {**current, **task, "status": current.get("status", task.get("status", "ready"))}
        if merged != current:
            existing[index] = merged
            changed = True
    if changed:
        write_jsonl(path, existing)
    return additions


def recent_result_has_prefix(root: Path, prefix: str, *, recent_rows: int = 80) -> bool:
    return any(row.get("run_id", "").startswith(prefix) for row in result_rows(root)[-recent_rows:])


def recent_keep_result_has_prefix(root: Path, prefix: str, *, recent_rows: int = 240) -> bool:
    return any(
        row.get("status") == "keep" and row.get("run_id", "").startswith(prefix)
        for row in result_rows(root)[-max(1, recent_rows) :]
    )


def active_task_has_prefix(root: Path, prefix: str) -> bool:
    return any(
        task.get("status", "ready") in {"ready", "rework"} and str(task.get("id", "")).startswith(prefix)
        for task in read_jsonl(root / "tasks.jsonl")
    )


def any_task_has_prefix(root: Path, prefix: str) -> bool:
    return any(str(task.get("id", "")).startswith(prefix) for task in read_jsonl(root / "tasks.jsonl"))


def unique_task_id(root: Path, task_id: str) -> str:
    existing = {str(task.get("id", "")) for task in read_jsonl(root / "tasks.jsonl")}
    if task_id not in existing:
        return task_id
    for index in range(2, 100):
        candidate = f"{task_id}-{index}"
        if candidate not in existing:
            return candidate
    return f"{task_id}-{time.time_ns()}"


def crabbox_required_for_target(target: str) -> bool:
    normalized = normalize_patch_path(target)
    return any(normalized.startswith(prefix) for prefix in HIGH_RISK_PATCH_PREFIXES)


def apply_crabbox_requirement(task: dict[str, Any], *, reason: str) -> dict[str, Any]:
    checks = [str(item) for item in task.get("guard_checks", [])]
    for check in ("crabbox_static_ssh_mac", "rollback_rehearsal", "frontier_autonomy_score_100"):
        if check not in checks:
            checks.append(check)
    task["guard_checks"] = checks
    task["risk_tier"] = "high-risk"
    task["crabbox_required"] = True
    task["crabbox_runner"] = "static-ssh-mac"
    task["promotion_blocked_until_crabbox"] = True
    task["risk_reason"] = reason
    task["acceptance"] = (
        f"{task.get('acceptance', '')} Any source patch produced from this contract must run in Crabbox first "
        "and can only promote with fresh matching Crabbox evidence, rollback rehearsal, and a Frontier Autonomy Score of 100."
    ).strip()
    return task


def should_seed_action(root: Path, prefix: str, *, recent_rows: int = 80) -> bool:
    return not active_task_has_prefix(root, prefix) and not recent_result_has_prefix(root, prefix, recent_rows=recent_rows)


def recent_clean_runtime_overhead_maps(
    root: Path,
    rows: list[dict[str, str]] | None = None,
    *,
    recent_rows: int = 80,
) -> list[dict[str, str]]:
    """Return runtime maps that already ruled out wall-clock contamination."""
    window = (rows if rows is not None else result_rows(root))[-max(1, recent_rows) :]
    clean_maps: list[dict[str, str]] = []
    for row in window:
        if row.get("status") != "keep" or not row.get("run_id", "").startswith("runtime-overhead-map-"):
            continue
        if parse_note_fields(row.get("notes", "")).get("contaminated") == "0":
            clean_maps.append(row)
    return clean_maps


def should_seed_runtime_overhead_map(root: Path, rows: list[dict[str, str]], *, recent_rows: int = 45) -> bool:
    if active_task_has_prefix(root, "deliberate-runtime-overhead-map-"):
        return False
    if active_task_has_prefix(root, "runtime-overhead-contamination-map"):
        return False
    if recent_result_has_prefix(root, "runtime-overhead-map-", recent_rows=recent_rows):
        return False
    return not recent_clean_runtime_overhead_maps(root, rows, recent_rows=recent_rows)


def completed_drafter_sweep_rows(
    root: Path,
    *,
    recent_rows: int = 120,
    min_sweeps: int = 3,
) -> list[dict[str, str]]:
    """Return enough completed sweep rows even when review rows push them out of the short window."""
    rows = result_rows(root)
    recent = [
        row
        for row in rows[-max(1, recent_rows) :]
        if row.get("status") == "keep" and row.get("run_id", "").startswith("drafter-sweep-run")
    ]
    if len(recent) >= min_sweeps:
        return recent
    seen = {row.get("run_id", "") for row in recent}
    older: list[dict[str, str]] = []
    for row in reversed(rows[: -max(1, recent_rows)]):
        run_id = row.get("run_id", "")
        if run_id in seen:
            continue
        if row.get("status") == "keep" and run_id.startswith("drafter-sweep-run"):
            older.append(row)
            seen.add(run_id)
        if len(recent) + len(older) >= min_sweeps:
            break
    return list(reversed(older)) + recent


def drafter_fit_task(timestamp: int, *, task_id: str, priority: int = 95) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "drafter-fit-plan",
        "target": "/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json",
        "hypothesis": "If official block-2 MTP is plateaued, the next frontier path is a JANQ-specific drafter fit gate.",
        "metric": "drafter_fit_gate",
        "guard_checks": ["no_model_load", "no_live_profile_change", "no_opencode_changes"],
        "acceptance": "A drafter-fit plan states required traces, gates, and promotion criteria before any training or live-profile change.",
        "rollback": "Keep the current MTP drafter as default until a fitted candidate beats paired TUI benchmarks.",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-drafter-fit plan",
    }


def drafter_trace_gate_task(timestamp: int, *, task_id: str, priority: int = 96) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "drafter-trace-gate",
        "target": "/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json",
        "hypothesis": (
            "The JANQ drafter-fit plan is complete; progress now depends on target-generated trace data "
            "or a clear blocked handoff, not another planning pass."
        ),
        "metric": "drafter_fit_gate",
        "guard_checks": ["no_model_load", "no_live_profile_change", "no_opencode_changes"],
        "acceptance": "The supervisor records whether target-generated trace data exists and names the next calibration gate.",
        "rollback": "No runtime rollback needed; this is a read-only trace-data readiness gate.",
        "next_action": (
            "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-trace-gate "
            "--plan /Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json"
        ),
    }


def drafter_trace_prerequisite_task(timestamp: int, *, task_id: str, priority: int = 99) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "drafter-trace-prerequisite",
        "target": "/Users/kristian/.openclaw/drafter-fit/target-generated-traces.jsonl",
        "hypothesis": "JANQ drafter fitting is blocked on target-generated trace data; record the exact prerequisite instead of repeating trace gates.",
        "metric": "drafter_fit_gate",
        "guard_checks": ["no_model_load", "no_live_profile_change", "no_opencode_changes"],
        "acceptance": "A prerequisite artifact lists the expected trace files and marks the lane blocked until data exists.",
        "rollback": "No runtime rollback needed; this is a read-only prerequisite report.",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-trace-prerequisite",
    }


def drafter_trace_collect_task(timestamp: int, *, task_id: str, priority: int = 99) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "drafter-trace-collect",
        "target": "/Users/kristian/.openclaw/drafter-fit/target-generated-traces.jsonl",
        "hypothesis": "JANQ drafter fitting needs a small target-generated trace set before calibration can be evaluated.",
        "metric": "drafter_fit_gate",
        "guard_checks": ["memory_gate", "bounded_samples", "no_live_profile_change", "no_opencode_changes"],
        "acceptance": "A JSONL trace file contains bounded OpenClaw prompts and JANQ target completions with usage metadata.",
        "rollback": "Delete the trace file if collection is malformed; no runtime profile or source setting is changed.",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-trace-collect --samples 6",
    }


def drafter_calibration_canary_task(
    timestamp: int,
    *,
    task_id: str,
    priority: int = 98,
    calibration_mode_value: str = CALIBRATION_DIRECT_MODE,
) -> dict[str, Any]:
    mode = calibration_mode(calibration_mode_value)
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "drafter-calibration-canary",
        "calibration_mode": mode,
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "source_files": ["openclaw/openclaw-mtp-drafter-calibrate.py", "openclaw/openclaw-drafter-fit.py"],
        "hypothesis": "Once JANQ target traces exist, drafter calibration should advance through a bounded canary gate instead of repeating trace checks.",
        "metric": "drafter_fit_gate",
        "guard_checks": ["memory_gate", "canary_only", "tests_pass", "no_live_profile_change", "no_opencode_changes"],
        "acceptance": "A canary artifact validates trace data, fit-plan gates, cheap drafter tests, and the exact bounded calibration command.",
        "rollback": "No runtime rollback needed; no model profile or drafter path is changed by this canary.",
        "next_action": f"/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-calibration-canary --calibration-mode {mode}",
    }


def drafter_calibration_run_task(
    timestamp: int,
    *,
    task_id: str,
    bounded_command: list[str],
    priority: int = 97,
    calibration_mode_value: str = CALIBRATION_DIRECT_MODE,
) -> dict[str, Any]:
    mode = calibration_mode(calibration_mode_value)
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "drafter-calibration-run",
        "calibration_mode": mode,
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "source_files": ["openclaw/openclaw-mtp-drafter-calibrate.py", "openclaw/openclaw-jang-vlm-server.py"],
        "hypothesis": "A validated JANQ trace canary should advance into exactly one bounded calibration experiment.",
        "metric": "acceptance_delta",
        "guard_checks": ["stop_live_model_first", "memory_gate", "bounded_training", "no_live_profile_change", "no_opencode_changes"],
        "acceptance": "The calibration script writes a keep/blocked artifact and no live profile is changed unless later paired benchmarks pass.",
        "rollback": "Discard the output directory and keep the current official q4 drafter if calibration blocks or does not beat paired benchmarks.",
        "bounded_command": bounded_command,
        "next_action": " ".join(bounded_command),
        "created_at": timestamp,
    }


def calibration_probe_command(
    stage: str,
    *,
    target_path: str,
    drafter_path: str,
    output_path: Path,
    prompts_path: Path,
    calibration_mode_value: str = CALIBRATION_DIRECT_MODE,
) -> list[str]:
    script = os.environ.get("OPENCLAW_MTP_CALIBRATOR_SCRIPT", DEFAULT_MTP_CALIBRATOR_SCRIPT)
    mode = calibration_mode(calibration_mode_value)
    command = [
        calibration_python(),
        script,
        "--target-path",
        target_path,
        "--drafter-path",
        drafter_path,
        "--output-path",
        str(output_path),
        "--prompts-file",
        str(prompts_path),
        "--train-samples",
        "1",
        "--eval-samples",
        "1",
        "--positions-per-prompt",
        "1",
        "--steps",
        "1",
        "--eval-every",
        "1",
        "--min-free-mb",
        "8192",
        "--max-compressor-mb",
        "8192",
        "--max-swap-mb",
        "4096",
        "--min-pressure-free-percent",
        "20",
        "--gpu-memory-utilization",
        "0.50",
        "--mlx-cache-gb",
        "1",
        "--target-trace-policy",
        "stop-gradient",
        "--probe-stage",
        stage,
    ]
    if mode != CALIBRATION_DIRECT_MODE:
        command.extend(["--calibration-mode", mode])
    return command


def calibration_full_run_command(
    *,
    target_path: str,
    drafter_path: str,
    output_path: Path,
    prompts_path: Path,
    calibration_mode_value: str = CALIBRATION_DIRECT_MODE,
) -> list[str]:
    script = os.environ.get("OPENCLAW_MTP_CALIBRATOR_SCRIPT", DEFAULT_MTP_CALIBRATOR_SCRIPT)
    mode = calibration_mode(calibration_mode_value)
    command = [
        calibration_python(),
        script,
        "--target-path",
        target_path,
        "--drafter-path",
        drafter_path,
        "--output-path",
        str(output_path),
        "--prompts-file",
        str(prompts_path),
        "--train-samples",
        "2",
        "--eval-samples",
        "1",
        "--positions-per-prompt",
        "1",
        "--steps",
        "1",
        "--eval-every",
        "1",
        "--min-free-mb",
        "8192",
        "--max-compressor-mb",
        "8192",
        "--max-swap-mb",
        "4096",
        "--min-pressure-free-percent",
        "20",
        "--gpu-memory-utilization",
        "0.55",
        "--mlx-cache-gb",
        "2",
        "--target-trace-policy",
        "stop-gradient",
    ]
    if mode != CALIBRATION_DIRECT_MODE:
        command.extend(["--calibration-mode", mode])
    return command


def drafter_trace_distillation_run_task(
    timestamp: int,
    *,
    task_id: str,
    bounded_command: list[str],
    priority: int = 99,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "drafter-calibration-run",
        "trace_distillation": True,
        "calibration_mode": CALIBRATION_TRACE_DISTILLATION_MODE,
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "source_files": ["openclaw/openclaw-mtp-drafter-calibrate.py", "openclaw/openclaw-speed-research.py"],
        "hypothesis": (
            "Direct JANQ calibration hit quantized-gradient failure; retry through trace distillation by "
            "detaching JANQ target traces and training only the drafter-side projection."
        ),
        "metric": "acceptance_delta_then_decode_tps",
        "guard_checks": [
            "stop_live_model_first",
            "memory_gate",
            "bounded_training",
            "canary_only",
            "no_target_weight_gradient",
            "no_live_profile_change",
            "no_opencode_changes",
        ],
        "acceptance": (
            "The run writes calibration metrics with training_mode=trace-distillation and target_gradient_policy=stop-gradient; "
            "promotion still requires later paired decode TPS and tool/reasoning guards."
        ),
        "rollback": "Discard the output directory and keep the current official q4 drafter unless later paired promotion gates pass.",
        "bounded_command": bounded_command,
        "next_action": " ".join(bounded_command),
        "created_at": timestamp,
    }


def calibration_stage_helper_command(
    stage: str,
    *,
    plan: str,
    trace_data: str,
    output_dir: str,
    min_traces: int = 4,
    max_prompts: int = 6,
    calibration_mode_value: str = CALIBRATION_DIRECT_MODE,
) -> list[str]:
    mode = calibration_mode(calibration_mode_value)
    command = [
        "/Users/kristian/.openclaw/bin/openclaw-speed-research",
        "drafter-calibration-memory-stage",
        "--stage",
        stage,
        "--plan",
        plan,
        "--output-dir",
        output_dir,
        "--min-traces",
        str(min_traces),
        "--max-prompts",
        str(max_prompts),
        "--calibration-mode",
        mode,
    ]
    if trace_data:
        command.extend(["--trace-data", trace_data])
    return command


def next_calibration_memory_stage(stage: str) -> str:
    try:
        index = CALIBRATION_MEMORY_STAGES.index(stage)
    except ValueError:
        return ""
    if index + 1 >= len(CALIBRATION_MEMORY_STAGES):
        return ""
    return CALIBRATION_MEMORY_STAGES[index + 1]


def drafter_calibration_memory_stage_task(
    timestamp: int,
    *,
    stage: str,
    task_id: str,
    bounded_command: list[str],
    priority: int = 98,
    calibration_mode_value: str = CALIBRATION_DIRECT_MODE,
) -> dict[str, Any]:
    mode = calibration_mode(calibration_mode_value)
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "drafter-calibration-memory-stage",
        "calibration_mode": mode,
        "stage": stage,
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "source_files": ["openclaw/openclaw-mtp-drafter-calibrate.py", "openclaw/openclaw-speed-research.py"],
        "hypothesis": (
            "JANQ drafter calibration should advance through isolated memory stages before any full "
            "target+drafter training run is allowed."
        ),
        "metric": "calibration_stage_gate",
        "guard_checks": [
            "stop_live_model_first",
            "memory_gate",
            "bounded_stage",
            "no_live_profile_change",
            "no_opencode_changes",
        ],
        "acceptance": f"The {stage} probe writes a keep artifact without Python, MLX, Metal, or memory-gate failure.",
        "rollback": "Discard the probe output directory; no OpenClaw model profile or live drafter path is changed.",
        "bounded_command": bounded_command,
        "next_action": " ".join(bounded_command),
        "created_at": timestamp,
    }


def drafter_trace_candidates() -> list[Path]:
    return [
        home() / "drafter-fit" / "target-generated-traces.jsonl",
        home() / "drafter-fit" / "target-generated-trace-data.jsonl",
        home() / "drafter-fit" / "janq-target-traces.jsonl",
    ]


def existing_drafter_trace_paths() -> list[Path]:
    return [path for path in drafter_trace_candidates() if path.exists() and path.stat().st_size > 0]


def is_trace_distillation_task(task: dict[str, Any]) -> bool:
    task_id = str(task.get("id", ""))
    return (
        task.get("trace_distillation") is True
        or task.get("calibration_mode") == CALIBRATION_TRACE_DISTILLATION_MODE
        or "trace-distillation-run" in task_id
    )


def trace_distillation_gradient_block_rows(root: Path, *, recent_rows: int = 240) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in result_rows(root)[-max(1, recent_rows) :]:
        text = " ".join(
            str(row.get(key, ""))
            for key in ("run_id", "target", "hypothesis", "notes")
        )
        lower = text.lower()
        if "trace-distillation" not in lower and "trace_distillation=true" not in lower:
            continue
        if calibration_quantized_gradient_issue(text) != CALIBRATION_QUANTIZED_GRADIENT_BLOCKER:
            continue
        rows.append(row)
    return rows


def trace_distillation_attempt_exists(root: Path, *, recent_rows: int = 240) -> bool:
    tasks = read_jsonl(root / "tasks.jsonl")
    if any(is_trace_distillation_task(task) for task in tasks):
        return True
    return any(
        "trace-distillation" in " ".join(
            str(row.get(key, ""))
            for key in ("run_id", "target", "hypothesis", "notes")
        ).lower()
        for row in result_rows(root)[-max(1, recent_rows) :]
    )


def trace_distillation_proof_failed(root: Path, *, recent_rows: int = 240) -> bool:
    if trace_distillation_gradient_block_rows(root, recent_rows=recent_rows):
        return True
    for task in read_jsonl(root / "tasks.jsonl"):
        if not is_trace_distillation_task(task):
            continue
        if task.get("status") != "blocked":
            continue
        summary_text = json.dumps(task.get("supervisor_summary", {}), sort_keys=True)
        if calibration_quantized_gradient_issue(summary_text) == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER:
            return True
    return False


def trace_distillation_repair_task(timestamp: int, *, task_id: str, priority: int = 99) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "frontier-expansion",
        "task_type": "supervisor",
        "supervisor_action": "focused-test",
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "source_files": [
            "openclaw/openclaw-mtp-drafter-calibrate.py",
            "openclaw/openclaw-speed-research.py",
            "openclaw/openclaw-speed-research-autopilot.py",
            "openclaw/test-mtp-drafter-calibrate-guards.py",
            "openclaw/test-speed-research.py",
        ],
        "hypothesis": (
            "Trace distillation proved that the current drafter fit path still differentiates quantized "
            "drafter parameters. The next safe path is an adapter or saved-logit distillation head that "
            "keeps JANQ target and quantized drafter weights frozen."
        ),
        "metric": "trace_distillation_repair_gate",
        "guard_checks": [
            "no_model_load",
            "canary_only",
            "tests_pass",
            "no_live_profile_change",
            "no_opencode_changes",
            "rollback_path",
        ],
        "acceptance": (
            "Canary tests prove repeated trace-distillation failures are suppressed and routed to an "
            "adapter/logit-distillation repair path before any model-loading experiment can run again."
        ),
        "rollback": "No runtime rollback needed; this is a no-model canary gate before any future source patch.",
        "next_action": "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-mtp-drafter-calibrate-guards.py",
        "created_at": timestamp,
    }


def trace_distillation_adapter_bridge_task(timestamp: int, *, task_id: str, priority: int = 98) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "frontier-expansion",
        "task_type": "supervisor",
        "supervisor_action": "focused-test",
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "source_files": [
            "openclaw/openclaw-mtp-drafter-calibrate.py",
            "openclaw/openclaw-speed-research.py",
            "openclaw/test-speed-research.py",
        ],
        "hypothesis": (
            "Trace-distillation repair is proven and the old expansion candidates are exhausted; define the next "
            "safe implementation path as a trainable adapter or saved-logit distillation head rather than retrying "
            "quantized projection training."
        ),
        "metric": "trace_distillation_repair_gate",
        "guard_checks": [
            "no_model_load",
            "canary_only",
            "tests_pass",
            "no_live_profile_change",
            "no_opencode_changes",
            "rollback_path",
        ],
        "acceptance": (
            "The canary confirms the supervisor can route from failed quantized-gradient calibration to the "
            "adapter/logit-distillation implementation track without reseeding trace-distillation repair loops."
        ),
        "rollback": "No runtime rollback needed; this is a no-model bridge before a future source patch.",
        "next_action": "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py",
        "created_at": timestamp,
    }


def trace_distillation_repair_attempted(root: Path, *, recent_rows: int = 240) -> bool:
    if any_task_has_prefix(root, "trace-distillation-gradient-repair-"):
        return True
    return any(
        "trace-distillation-gradient-repair-" in " ".join(
            str(row.get(key, ""))
            for key in ("run_id", "target", "hypothesis", "notes")
        )
        for row in result_rows(root)[-max(1, recent_rows) :]
    )


def drafter_bottleneck_state(
    root: Path,
    rows: list[dict[str, str]] | None = None,
    *,
    recent_rows: int = 240,
) -> dict[str, Any]:
    """Return the canonical JANQ drafter bottleneck state.

    This keeps the drafter lane from rediscovering the same quantized-gradient
    failure through calibration, DFlash, and decode fallback loops. The state is
    deliberately small and deterministic so the supervisor can route to exactly
    one next action.
    """

    window = (rows if rows is not None else result_rows(root))[-max(1, recent_rows) :]
    blocker = recent_calibration_run_hard_blocker(root, recent_rows=recent_rows)
    terminal_blocks = recent_terminal_calibration_block_rows(
        root,
        blocker=CALIBRATION_QUANTIZED_GRADIENT_BLOCKER,
        recent_rows=recent_rows,
    )
    trace_paths = existing_drafter_trace_paths()
    trace_attempted = trace_distillation_attempt_exists(root, recent_rows=recent_rows)
    trace_failed = trace_distillation_proof_failed(root, recent_rows=recent_rows)
    active_repair = active_task_has_prefix(root, "trace-distillation-gradient-repair-")
    repair_attempted = trace_distillation_repair_attempted(root, recent_rows=recent_rows)
    repair_completed = repair_attempted and not active_repair
    expansion_attempted = any_task_has_prefix(root, "frontier-expansion-janq-adapter-path-") or recent_result_has_prefix(
        root,
        "supervisor-focused-test-",
        recent_rows=recent_rows,
    ) and any(
        "adapter" in row.get("hypothesis", "").lower()
        for row in window
        if row.get("run_id", "").startswith("supervisor-focused-test-")
    )
    active_adapter_bridge = active_task_has_prefix(root, "trace-distillation-adapter-bridge-")
    adapter_bridge_attempted = any_task_has_prefix(root, "trace-distillation-adapter-bridge-")
    active_adapter_contract = active_task_has_prefix(root, "drafter-adapter-method-contract-")
    adapter_contract_attempted = any_task_has_prefix(root, "drafter-adapter-method-contract-")
    adapter_contract_succeeded = recent_keep_result_has_prefix(
        root,
        "drafter-adapter-method-contract-",
        recent_rows=max(240, recent_rows),
    )
    active_adapter_implementation = active_task_has_prefix(root, "implementation-drafter-adapter-method-")
    adapter_implementation_attempted = (
        any_task_has_prefix(root, "implementation-drafter-adapter-method-")
        or recent_keep_result_has_prefix(
            root,
            "implementation-drafter-adapter-method-",
            recent_rows=max(240, recent_rows),
        )
    )
    adapter_calibration_active = any(
        task.get("status", "ready") in {"ready", "rework"}
        and calibration_task_mode(task) == CALIBRATION_ADAPTER_MODE
        and str(task.get("supervisor_action", "")).startswith("drafter-calibration")
        for task in read_jsonl(root / "tasks.jsonl")
    )
    adapter_calibration_stage_issue = latest_calibration_stage_issue(
        root,
        calibration_mode_filter=CALIBRATION_ADAPTER_MODE,
    )
    adapter_calibration_attempted = adapter_calibration_active or any(
        "calibration_mode=adapter-logit-distillation" in row.get("notes", "")
        or "adapter-logit-distillation" in row.get("run_id", "")
        for row in window
    )
    fallback_decode_count = recent_lane_contract_decode_fallback_count(root, recent_rows=min(recent_rows, 80))
    adapter_loop = adapter_logit_loop_evidence(root, recent_rows=recent_rows)
    historical_bottleneck = (
        blocker == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
        or bool(terminal_blocks)
        or trace_failed
        or repair_attempted
        or expansion_attempted
        or adapter_bridge_attempted
        or adapter_contract_attempted
        or adapter_contract_succeeded
        or adapter_implementation_attempted
        or adapter_calibration_attempted
        or bool(adapter_calibration_stage_issue)
    )

    if not historical_bottleneck:
        state = "no_terminal_quantized_blocker"
        next_step = "continue_current_lane_contract"
    elif adapter_loop["saturated"]:
        state = "adapter_logit_loop_exhausted"
        next_step = "seed_frontier_deliberation_escape"
    elif active_adapter_implementation:
        state = "adapter_method_implementation_active"
        next_step = "wait_for_adapter_method_implementation"
    elif adapter_calibration_active:
        state = "adapter_calibration_active"
        next_step = "wait_for_adapter_calibration"
    elif adapter_calibration_stage_issue == "calibration-memory-after-load":
        state = "adapter_calibration_memory_blocked"
        next_step = "seed_adapter_calibration_memory_report"
    elif adapter_calibration_stage_issue:
        state = "adapter_calibration_blocked"
        next_step = "seed_adapter_method_contract"
    elif adapter_calibration_attempted:
        state = "adapter_calibration_attempted"
        next_step = "wait_for_adapter_calibration_result"
    elif adapter_implementation_attempted:
        state = "adapter_method_implementation_done"
        next_step = "seed_adapter_calibration_canary"
    elif adapter_contract_succeeded:
        state = "adapter_method_contract_ready"
        next_step = "seed_adapter_method_implementation"
    elif active_adapter_contract:
        state = "adapter_method_contract_active"
        next_step = "wait_for_adapter_method_contract"
    elif trace_failed and active_repair:
        state = "trace_distillation_repair_active"
        next_step = "wait_for_trace_distillation_repair"
    elif trace_failed and not repair_completed:
        state = "trace_distillation_failed"
        next_step = "seed_trace_distillation_repair"
    elif trace_failed and repair_completed and not expansion_attempted:
        state = "repair_done_need_adapter_expansion"
        next_step = "seed_janq_adapter_expansion"
    elif trace_failed and repair_completed and expansion_attempted and active_adapter_bridge:
        state = "adapter_bridge_active"
        next_step = "wait_for_adapter_bridge"
    elif trace_failed and repair_completed and expansion_attempted and not adapter_bridge_attempted:
        state = "adapter_expansion_done"
        next_step = "seed_adapter_bridge"
    elif trace_failed and repair_completed and expansion_attempted and adapter_bridge_attempted and not adapter_contract_attempted:
        state = "adapter_method_required"
        next_step = "seed_adapter_method_contract"
    elif trace_failed and repair_completed and expansion_attempted and adapter_bridge_attempted:
        state = "adapter_method_contract_ready"
        next_step = "seed_adapter_method_contract"
    elif trace_attempted:
        state = "trace_distillation_pending_or_incomplete"
        next_step = "wait_for_trace_distillation_result"
    elif not trace_paths:
        state = "trace_data_missing"
        next_step = "collect_target_generated_traces"
    else:
        state = "trace_data_ready"
        next_step = "seed_trace_distillation"

    return {
        "state": state,
        "next_step": next_step,
        "calibration_blocker": blocker,
        "terminal_block_count": len(terminal_blocks),
        "trace_paths": [str(path) for path in trace_paths],
        "trace_attempted": trace_attempted,
        "trace_distillation_failed": trace_failed,
        "trace_repair_attempted": repair_attempted,
        "trace_repair_active": active_repair,
        "trace_repair_completed": repair_completed,
        "adapter_expansion_attempted": expansion_attempted,
        "adapter_bridge_active": active_adapter_bridge,
        "adapter_bridge_attempted": adapter_bridge_attempted,
        "adapter_method_contract_active": active_adapter_contract,
        "adapter_method_contract_attempted": adapter_contract_attempted,
        "adapter_method_contract_succeeded": adapter_contract_succeeded,
        "adapter_method_implementation_active": active_adapter_implementation,
        "adapter_method_implementation_attempted": adapter_implementation_attempted,
        "adapter_calibration_active": adapter_calibration_active,
        "adapter_calibration_attempted": adapter_calibration_attempted,
        "adapter_calibration_stage_issue": adapter_calibration_stage_issue,
        "fallback_decode_count": fallback_decode_count,
        "adapter_logit_loop": adapter_loop,
    }


def drafter_adapter_method_contract_task(timestamp: int, *, task_id: str, priority: int = 99) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "revive_blocked": True,
        "priority": priority,
        "lane": "implementation-gate",
        "task_type": "supervisor",
        "supervisor_action": "drafter-adapter-method-contract",
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "source_files": [
            "openclaw/openclaw-mtp-drafter-calibrate.py",
            "openclaw/test-mtp-drafter-calibrate-guards.py",
            "openclaw/test-speed-research.py",
        ],
        "hypothesis": (
            "The JANQ drafter bottleneck is now known: direct and trace-distillation training both hit "
            "quantized-gradient blockers. The next implementation must use a changed method, such as a "
            "trainable adapter or saved-logit distillation head, with frozen JANQ target and frozen quantized drafter."
        ),
        "metric": "adapter_method_contract",
        "guard_checks": [
            "no_model_load",
            "canary_only",
            "tests_pass",
            "no_live_profile_change",
            "no_opencode_changes",
            "rollback_path",
        ],
        "acceptance": (
            "A contract artifact records the exact changed drafter method, allowed source files, tests, "
            "promotion gates, and rollback before any source patch or model-loading experiment can run."
        ),
        "rollback": "No runtime rollback needed; this only creates a source-patch contract and keeps the live drafter unchanged.",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-adapter-method-contract",
        "created_at": timestamp,
    }


def drafter_adapter_method_implementation_task(
    timestamp: int,
    *,
    task_id: str = "implementation-drafter-adapter-method-current",
    contract_path: str = "",
    priority: int = 99,
) -> dict[str, Any]:
    source_files = [
        "openclaw/openclaw-mtp-drafter-calibrate.py",
        "openclaw/test-mtp-drafter-calibrate-guards.py",
        "openclaw/test-speed-research.py",
    ]
    return {
        "id": task_id,
        "status": "ready",
        "revive_blocked": True,
        "priority": priority,
        "lane": "implementation-gate",
        "task_type": "supervisor",
        "supervisor_action": "focused-test",
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "source_files": source_files,
        "hypothesis": (
            "Implement the JANQ drafter fit as a trainable adapter/logit-distillation path: freeze the JANQ target "
            "and quantized drafter weights, train only newly introduced adapter/head parameters, and keep the live TUI "
            "profile unchanged until paired decode benchmarks pass."
        ),
        "metric": "adapter_method_contract",
        "guard_checks": [
            "no_model_load",
            "canary_only",
            "tests_pass",
            "no_live_profile_change",
            "no_opencode_changes",
            "rollback_path",
        ],
        "acceptance": (
            "Tests prove direct quantized-gradient training remains blocked; the adapter/logit method is explicit, "
            "off by default, canary-only, and has promotion gates for decode TPS, TTFT, memory, tool calls, and stream guards."
        ),
        "rollback": "Discard adapter artifacts and keep the current official MTP drafter/profile if any gate fails.",
        "contract_path": contract_path,
        "next_action": "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-mtp-drafter-calibrate-guards.py",
        "created_at": timestamp,
    }


def drafter_bottleneck_next_tasks(
    root: Path,
    rows: list[dict[str, str]] | None,
    timestamp: int,
    *,
    reason: str,
) -> list[dict[str, Any]]:
    state = drafter_bottleneck_state(root, rows, recent_rows=240)
    step = state["next_step"]
    if step.startswith("wait_for_"):
        recent_wait = any(
            item.get("task_id") == "drafter-bottleneck-wait" and item.get("next") == step
            for item in read_jsonl(root / "findings.jsonl")[-20:]
        )
        if not recent_wait:
            append_jsonl(
                root / "findings.jsonl",
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "task_id": "drafter-bottleneck-wait",
                    "finding": "drafter bottleneck review found active downstream work and refused to seed a duplicate route",
                    "reason": reason,
                    "evidence": state,
                    "next": step,
                },
            )
        return []
    if step == "collect_target_generated_traces":
        return filter_seedable_tasks(
            root,
            [drafter_trace_collect_task(timestamp, task_id=f"drafter-bottleneck-trace-collect-{timestamp}")],
        )
    if step == "seed_trace_distillation":
        distillation_task = trace_distillation_candidate_task(
            root,
            timestamp,
            task_id=f"drafter-trace-distillation-run-{timestamp}",
        )
        return filter_seedable_tasks(root, [distillation_task] if distillation_task is not None else [])
    if step == "seed_trace_distillation_repair":
        if active_task_has_prefix(root, "trace-distillation-gradient-repair-"):
            return []
        if not should_seed_action(root, "trace-distillation-gradient-repair-", recent_rows=240):
            return []
        return filter_seedable_tasks(
            root,
            [trace_distillation_repair_task(timestamp, task_id="trace-distillation-gradient-repair-current")],
        )
    if step == "seed_janq_adapter_expansion":
        return frontier_expansion_tasks(root, rows if rows is not None else result_rows(root), timestamp)
    if step == "seed_adapter_bridge":
        if any_task_has_prefix(root, "trace-distillation-adapter-bridge-"):
            return []
        return filter_seedable_tasks(
            root,
            [trace_distillation_adapter_bridge_task(timestamp, task_id="trace-distillation-adapter-bridge-current")],
        )
    if step == "seed_adapter_method_contract":
        if active_task_has_prefix(root, "drafter-adapter-method-contract-"):
            return []
        return filter_seedable_tasks(
            root,
            [
                drafter_adapter_method_contract_task(
                    timestamp,
                    task_id="drafter-adapter-method-contract-current",
                )
            ],
        )
    if step == "seed_adapter_method_implementation":
        if active_task_has_prefix(root, "implementation-drafter-adapter-method-"):
            return []
        return filter_seedable_tasks(
            root,
            [drafter_adapter_method_implementation_task(timestamp)],
        )
    if step == "seed_adapter_calibration_memory_report":
        return filter_seedable_tasks(
            root,
            calibration_blocker_report_tasks(root, timestamp, blocker="calibration-memory-after-load"),
        )
    if step == "seed_frontier_deliberation_escape":
        evidence = {
            "reason": "adapter/logit blocker loop saturated",
            "bottleneck_state": state,
        }
        tasks = [
            mtp_acceptance_yield_task(timestamp, evidence=evidence),
            source_scout_task(timestamp, evidence=evidence),
        ]
        return filter_seedable_tasks(root, tasks)
    if step == "seed_adapter_calibration_canary":
        if active_task_has_prefix(root, "adapter-drafter-calibration-canary-"):
            return []
        if recent_result_has_prefix(root, "adapter-drafter-calibration-canary-", recent_rows=240):
            return []
        return filter_seedable_tasks(
            root,
            [
                drafter_calibration_canary_task(
                    timestamp,
                    task_id=unique_task_id(root, "adapter-drafter-calibration-canary-current"),
                    priority=99,
                    calibration_mode_value=CALIBRATION_ADAPTER_MODE,
                )
            ],
        )
    return []


def should_seed_drafter_calibration_canary(root: Path, *, recent_rows: int = 120) -> bool:
    if drafter_bottleneck_state(root, recent_rows=recent_rows)["state"] != "no_terminal_quantized_blocker":
        return False
    if recent_calibration_run_hard_blocker(root, recent_rows=recent_rows) in CALIBRATION_CANARY_TERMINAL_BLOCKERS:
        return False
    return bool(existing_drafter_trace_paths()) and should_seed_action(
        root,
        "drafter-calibration-canary-",
        recent_rows=recent_rows,
    )


def should_seed_drafter_calibration_run(root: Path, *, recent_rows: int = 120) -> bool:
    if recent_calibration_run_hard_blocker(root, recent_rows=recent_rows):
        return False
    return should_seed_action(
        root,
        "drafter-calibration-run-",
        recent_rows=recent_rows,
    )


def trace_distillation_candidate_task(root: Path, timestamp: int, *, task_id: str) -> dict[str, Any] | None:
    trace_paths = existing_drafter_trace_paths()
    if not trace_paths:
        return None
    output_dir = home() / "drafter-fit"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "calibration-canary-prompts.txt"
    if not prompts_path.exists():
        try:
            trace_rows = read_trace_rows(trace_paths[0])
            prompts = [prompt for row in trace_rows if (prompt := trace_prompt_text(row))]
            if prompts:
                prompts_path.write_text("\n".join(prompts[:4]) + "\n", encoding="utf-8")
        except (OSError, json.JSONDecodeError, ValueError):
            return None
    target_path = os.environ.get("OPENCLAW_JANQ_TARGET_PATH", DEFAULT_JANQ_TARGET_PATH)
    drafter_path = os.environ.get(
        "OPENCLAW_MTP_DRAFT_PATH",
        os.environ.get("OPENCLAW_JANG_DRAFT_MODEL", DEFAULT_MTP_DRAFT_PATH),
    )
    bounded_command = calibration_full_run_command(
        target_path=target_path,
        drafter_path=drafter_path,
        output_path=output_dir / f"trace-distilled-drafter-{timestamp}",
        prompts_path=prompts_path,
    )
    return drafter_trace_distillation_run_task(
        timestamp,
        task_id=task_id,
        bounded_command=bounded_command,
    )


def recent_calibration_run_hard_blocker(root: Path, *, recent_rows: int = 160) -> str:
    calibration_prefixes = (
        "supervisor-drafter-calibration-run-",
        "supervisor-drafter-calibration-memory-stage-",
        "drafter-calibration-run-",
        "drafter-calibration-memory-stage-",
    )
    for row in reversed(result_rows(root)[-max(1, recent_rows) :]):
        run_id = row.get("run_id", "")
        if not run_id.startswith(calibration_prefixes):
            continue
        notes = row.get("notes", "").lower()
        gradient_issue = calibration_quantized_gradient_issue(notes)
        if gradient_issue:
            return gradient_issue
        if (
            run_id.startswith("supervisor-drafter-calibration-memory-stage-")
            and "reason=terminal-blocker" in notes
            and "projection.scales" in notes
            and '"returncode": 2' in notes
        ):
            return CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
        if "calibration memory gate blocked: after-load" in notes:
            return "calibration-memory-after-load"
        if "calibration-memory-gate:after-load" in notes:
            return "calibration-memory-after-load"
        if "missing-runtime-module:mlx_vlm.speculative" in notes or "no module named 'mlx_vlm.speculative'" in notes:
            return "calibration-runtime-missing-speculative"
    return ""


def recent_terminal_calibration_block_rows(
    root: Path,
    *,
    blocker: str = CALIBRATION_QUANTIZED_GRADIENT_BLOCKER,
    recent_rows: int = 160,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in result_rows(root)[-max(1, recent_rows) :]:
        if row.get("status") != "blocked":
            continue
        run_id = row.get("run_id", "")
        target = row.get("target", "")
        if "calibration" not in run_id and "calibration" not in target:
            continue
        notes = row.get("notes", "")
        if blocker == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER and is_known_terminal_calibration_blocked_row(row):
            rows.append(row)
            continue
        if blocker and blocker not in notes and calibration_quantized_gradient_issue(notes) != blocker:
            continue
        rows.append(row)
    return rows


def recent_drafter_fit_plan_ready(root: Path, *, recent_rows: int = 160) -> bool:
    for row in result_rows(root)[-max(1, recent_rows) :]:
        if row.get("status") != "keep":
            continue
        run_id = row.get("run_id", "")
        if not run_id.startswith("supervisor-drafter-fit-"):
            continue
        if parse_note_fields(row.get("notes", "")).get("decision") == "ready-for-target-generated-trace-data":
            return True
    return False


def recent_drafter_trace_gate(root: Path, *, recent_rows: int = 80) -> bool:
    return any(
        row.get("run_id", "").startswith("drafter-trace-gate-")
        or row.get("run_id", "").startswith("supervisor-drafter-trace-gate-")
        for row in result_rows(root)[-max(1, recent_rows) :]
    )


def recent_drafter_trace_missing(root: Path, *, recent_rows: int = 120) -> bool:
    return any(
        (
            row.get("run_id", "").startswith("drafter-trace-gate-")
            or row.get("run_id", "").startswith("supervisor-drafter-trace-gate-")
            or row.get("run_id", "").startswith("drafter-trace-prerequisite-")
            or row.get("run_id", "").startswith("supervisor-drafter-trace-prerequisite-")
        )
        and "target-generated-trace-data-missing" in row.get("notes", "")
        for row in result_rows(root)[-max(1, recent_rows) :]
    )


def recent_blocked_dflash_gate(root: Path, *, recent_rows: int = 80) -> dict[str, Any] | None:
    rows = result_rows(root)[-max(1, recent_rows) :]
    for row in reversed(rows):
        if row.get("status") != "blocked" or not row.get("run_id", "").startswith("dflash-compatibility-gate-"):
            continue
        run_id = row.get("run_id", "")
        timestamp = run_id.rsplit("-", 1)[-1]
        artifact = root / "experiments" / f"dflash-compatibility-gate-{timestamp}.json"
        blockers: list[str] = []
        if artifact.exists():
            try:
                with artifact.open("r", encoding="utf-8") as file:
                    loaded = json.load(file)
                if isinstance(loaded, dict):
                    blockers = [str(item) for item in loaded.get("blockers", []) if item]
            except (OSError, json.JSONDecodeError):
                blockers = []
        return {
            "run_id": run_id,
            "artifact": str(artifact),
            "blockers": blockers,
            "notes": row.get("notes", ""),
        }
    return None


def dflash_lane_is_blocked(root: Path, *, recent_rows: int = 80) -> bool:
    blocked = recent_blocked_dflash_gate(root, recent_rows=recent_rows)
    if not blocked:
        return False
    blockers = {str(item) for item in blocked.get("blockers", [])}
    hard_blockers = {
        blocker
        for blocker in blockers
        if blocker.startswith("draft_model_type_mismatch=")
        or blocker in {"dflash_draft_config_missing", "draft_target_layer_ids_missing"}
    }
    return bool(hard_blockers)


def suppress_hard_blocked_dflash_lane(root: Path, *, recent_rows: int = 160) -> bool:
    blocked = recent_blocked_dflash_gate(root, recent_rows=recent_rows)
    if not blocked:
        return False
    blockers = [str(item) for item in blocked.get("blockers", []) if item]
    hard_blockers = [
        blocker
        for blocker in blockers
        if blocker.startswith("draft_model_type_mismatch=")
        or blocker in {"dflash_draft_config_missing", "draft_target_layer_ids_missing"}
    ]
    if not hard_blockers:
        return False
    return mark_lane_exhausted(
        root,
        lane="frontier-dflash",
        reason="DFlash/JANQ compatibility is hard-blocked; do not reseed until the draft candidate changes",
        evidence={
            "blocked_run_id": blocked.get("run_id", ""),
            "artifact": blocked.get("artifact", ""),
            "blockers": hard_blockers,
        },
    )


def dflash_compatibility_task(timestamp: int, *, task_id: str, priority: int = 93) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "frontier-dflash",
        "task_type": "supervisor",
        "supervisor_action": "dflash-compatibility-gate",
        "target": "dflash.model_mlx/openclaw-jang-vlm-server.py",
        "hypothesis": "DFlash-style block drafting may raise decode speed, but JANQ compatibility must be proven before runtime promotion.",
        "metric": "compatibility_decision_then_decode_tps",
        "guard_checks": [
            "no_model_load",
            "no_live_profile_change",
            "separate_env",
            "stream_guard",
            "no_opencode_changes",
        ],
        "acceptance": "A deterministic compatibility artifact names the exact blocker or a canary-only path with no live profile mutation.",
        "rollback": "Do not touch the normal OpenClaw TUI profile unless a paired benchmark beats the current MTP path and all guards pass.",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research dflash-compatibility-gate",
    }


def default_dflash_draft_path() -> Path:
    return Path(
        os.environ.get(
            "OPENCLAW_DFLASH_DRAFT_PATH",
            "/Users/kristian/.cache/huggingface/hub/"
            "models--z-lab--gemma-4-31B-it-DFlash/"
            "snapshots/9e3bf61731945317dfb0dc2d130c383c9d051f76",
        )
    ).expanduser()


def lane_contract_decode_task(timestamp: int, *, task_id: str, reason: str, priority: int = 99) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "runtime-overhead",
        "task_type": "supervisor",
        "target": "decode-sample",
        "hypothesis": reason,
        "metric": "decode_tps",
        "benchmark_mode": "decode-sample",
        "guard_checks": ["memory_gate", "no_live_profile_change", "no_opencode_changes", "stream_guard"],
        "acceptance": "A fresh decode benchmark records wall-clock and server tok/s without loading a second JANQ target.",
        "rollback": "No rollback needed; benchmark-only task.",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
    }


def recent_lane_contract_decode_fallback_count(root: Path, *, recent_rows: int = 40) -> int:
    rows = result_rows(root)[-max(1, recent_rows) :]
    row_count = sum(
        1
        for row in rows
        if (
            "contract_actions=lane-contract-decode-remeasure-" in row.get("notes", "")
            or row.get("run_id", "").startswith("benchmark-")
            and row.get("target") == "decode-sample"
            and "lane-contract-decode-remeasure-" in row.get("notes", "")
        )
    )
    task_count = sum(
        1
        for task in read_jsonl(root / "tasks.jsonl")
        if str(task.get("id", "")).startswith(
            (
                "lane-contract-decode-remeasure-",
                "handoff-audit-decode-remeasure-after-calibration-block-",
            )
        )
    )
    return row_count + task_count


def recent_calibration_fallback_plateau(
    root: Path,
    rows: list[dict[str, str]] | None = None,
    *,
    recent_rows: int = 80,
    min_clean_decode_rows: int = 4,
    max_clean_decode_tps: float = 18.0,
) -> dict[str, Any] | None:
    """Detect when calibration is blocked and fallback decode remeasurements are no longer useful."""
    recent = (rows if rows is not None else result_rows(root))[-max(1, recent_rows) :]
    calibration_blocker = recent_calibration_run_hard_blocker(root, recent_rows=240)
    if not calibration_blocker:
        return None
    fallback_count = recent_lane_contract_decode_fallback_count(root, recent_rows=recent_rows)
    if fallback_count < 3:
        return None
    clean_decode_values: list[float] = []
    for row in recent:
        if row.get("status") != "keep" or row.get("target") != "decode-sample":
            continue
        signal = decode_measurement_signal(row)
        if signal["contaminated"]:
            continue
        value = signal.get("server_decode_tps") or signal.get("wall_decode_tps")
        if value is not None:
            clean_decode_values.append(float(value))
    if len(clean_decode_values) < min_clean_decode_rows:
        return None
    best_clean = max(clean_decode_values)
    if best_clean >= max_clean_decode_tps:
        return None
    no_model_escalations_done = any_task_has_prefix(
        root, "lane-contract-runtime-overhead-map-after-fallback-"
    ) and any_task_has_prefix(root, "lane-contract-mtp-report-after-fallback-")
    return {
        "calibration_blocker": calibration_blocker,
        "fallback_count": fallback_count,
        "clean_decode_rows": len(clean_decode_values),
        "best_clean_decode_tps": round(best_clean, 3),
        "mean_clean_decode_tps": mean_value(clean_decode_values),
        "no_model_escalations_done": no_model_escalations_done,
        "max_clean_decode_tps": max_clean_decode_tps,
    }


def record_calibration_fallback_plateau(root: Path, plateau: dict[str, Any], *, task_id: str) -> None:
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": task_id,
            "finding": "calibration fallback plateau reached; supervisor stopped reseeding decode remeasurements",
            "evidence": plateau,
            "next": (
                "wait for new JANQ trace/calibration prerequisites or a specific implementation candidate; "
                "do not rerun generic decode fallback benchmarks"
            ),
        },
    )


def calibration_memory_report_task(timestamp: int, *, task_id: str, priority: int = 96) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": "ready",
        "priority": priority,
        "lane": "drafter-alignment",
        "task_type": "supervisor",
        "supervisor_action": "calibration-memory-report",
        "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
        "hypothesis": "Calibration is blocked after loading the JANQ target and drafter, so produce one no-model root-cause report before more overnight cycles.",
        "metric": "calibration_memory_root_cause",
        "guard_checks": ["no_model_load", "one_narrow_tool", "no_live_profile_change", "no_opencode_changes"],
        "acceptance": "A report records the calibration blocker, relevant calibration knobs, and the next implementation gate.",
        "rollback": "No runtime rollback needed; this is a read-only supervisor report.",
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research calibration-memory-report",
    }


def calibration_blocker_report_tasks(root: Path, timestamp: int, *, blocker: str) -> list[dict[str, Any]]:
    for row in result_rows(root)[-240:]:
        if row.get("run_id", "").startswith("calibration-memory-report-") and f"blocker={blocker}" in row.get("notes", ""):
            return []
    for task in read_jsonl(root / "tasks.jsonl"):
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        if str(task.get("supervisor_action", "")) == "calibration-memory-report":
            return []
    tasks = read_jsonl(root / "tasks.jsonl")
    if any(
        task.get("status", "ready") in {"ready", "rework"}
        and str(task.get("id", "")).startswith("calibration-memory-report-")
        for task in tasks
    ):
        return []
    if recent_result_has_prefix(root, "calibration-memory-report-", recent_rows=20):
        return []
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "calibration-terminal-blocker",
            "finding": "drafter calibration hit a terminal blocker; supervisor will report once instead of reseeding the same lane",
            "blocker": blocker,
            "next": "record root cause and wait for a changed calibration method or trainable adapter path",
        },
    )
    return [
        calibration_memory_report_task(
            timestamp,
            task_id=f"calibration-memory-report-{timestamp}",
            priority=99,
        )
    ]


def calibration_plateau_tasks(root: Path, plateau: dict[str, Any], timestamp: int, *, task_id: str) -> list[dict[str, Any]]:
    record_calibration_fallback_plateau(root, plateau, task_id=task_id)
    if recent_result_has_prefix(root, "calibration-memory-report-", recent_rows=240) or any_task_has_prefix(
        root, "calibration-memory-report-"
    ):
        return []
    return [
        calibration_memory_report_task(
            timestamp,
            task_id=f"calibration-memory-report-{timestamp}",
            priority=98,
        )
    ]


def calibration_blocker_diagnosis(blocker: str) -> str:
    if blocker == "calibration-memory-after-load":
        return (
            "bounded calibration is blocked after loading both the JANQ target and drafter; "
            "the next useful work is a canary-only source change that reduces calibration load overlap "
            "or lowers the calibration memory envelope before retrying training"
        )
    if blocker == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER:
        return (
            "calibration reached the micro-step but MLX reported no gradient through quantized weights; "
            "retrying the same JANQ/JANG quantized micro-step will repeat the failure"
        )
    if blocker == "calibration-runtime-missing-speculative":
        return (
            "calibration is blocked because the local runtime is missing the speculative decoding module; "
            "retrying calibration is not useful until the runtime dependency changes"
        )
    return "no current calibration blocker was detected in recent calibration rows"


def calibration_blocker_next_step(blocker: str) -> str:
    if blocker == "calibration-memory-after-load":
        return "implementation-gate: propose a canary-only calibration memory patch with tests, or wait for more free memory"
    if blocker == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER:
        return (
            "implementation-gate: design a canary-only adapter/calibration path that avoids differentiating "
            "quantized target weights, or retire drafter calibration until that method exists"
        )
    if blocker == "calibration-runtime-missing-speculative":
        return "implementation-gate: resolve runtime dependency or keep the lane suppressed"
    return "continue normal drafter-alignment tasks"


def lane_contract_fallback_tasks(
    root: Path,
    rows: list[dict[str, str]] | None,
    timestamp: int,
    *,
    reason: str,
) -> list[dict[str, Any]]:
    """Return one deterministic fallback task when a lane stalls or exhausts."""
    ensure_lane_contracts(root)
    recent_rows = rows if rows is not None else result_rows(root)
    calibration_blocker = recent_calibration_run_hard_blocker(root, recent_rows=240)
    dflash_blocked = dflash_lane_is_blocked(root, recent_rows=240) or "frontier-dflash" in exhausted_lanes(root)
    tasks: list[dict[str, Any]] = []
    bottleneck_state = drafter_bottleneck_state(root, recent_rows, recent_rows=240)
    if bottleneck_state["state"] != "no_terminal_quantized_blocker":
        bottleneck_tasks = drafter_bottleneck_next_tasks(root, recent_rows, timestamp, reason=reason)
        if bottleneck_tasks:
            return bottleneck_tasks
        if str(bottleneck_state["next_step"]).startswith("wait_for_"):
            return []
    if calibration_blocker == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER:
        bottleneck_tasks = drafter_bottleneck_next_tasks(root, recent_rows, timestamp, reason=reason)
        if bottleneck_tasks:
            return bottleneck_tasks
        if trace_distillation_proof_failed(root, recent_rows=240):
            if active_task_has_prefix(root, "trace-distillation-gradient-repair-"):
                return []
            if not trace_distillation_repair_attempted(root, recent_rows=240) and should_seed_action(
                root,
                "trace-distillation-gradient-repair-",
                recent_rows=240,
            ):
                return filter_seedable_tasks(
                    root,
                    [
                        trace_distillation_repair_task(
                            timestamp,
                            task_id=f"trace-distillation-gradient-repair-{timestamp}",
                        )
                    ],
                )
            expansion_tasks = frontier_expansion_tasks(root, recent_rows, timestamp)
            if expansion_tasks:
                return expansion_tasks
            if not any_task_has_prefix(root, "trace-distillation-adapter-bridge-"):
                return filter_seedable_tasks(
                    root,
                    [
                        trace_distillation_adapter_bridge_task(
                            timestamp,
                            task_id=f"trace-distillation-adapter-bridge-{timestamp}",
                        )
                    ],
                )
            return filter_seedable_tasks(
                root,
                calibration_blocker_report_tasks(root, timestamp, blocker=calibration_blocker),
            )
        if (
            not trace_distillation_attempt_exists(root, recent_rows=240)
            and should_seed_action(root, "drafter-trace-distillation-run-", recent_rows=240)
        ):
            distillation_task = trace_distillation_candidate_task(
                root,
                timestamp,
                task_id=f"drafter-trace-distillation-run-{timestamp}",
            )
            if distillation_task is not None:
                return filter_seedable_tasks(root, [distillation_task])
        if not existing_drafter_trace_paths():
            return filter_seedable_tasks(
                root,
                [
                    drafter_trace_collect_task(
                        timestamp,
                        task_id=f"lane-contract-drafter-trace-collect-for-distillation-{timestamp}",
                    )
                ],
            )
        return filter_seedable_tasks(
            root,
            calibration_blocker_report_tasks(root, timestamp, blocker=calibration_blocker),
        )
    if calibration_blocker and recent_lane_contract_decode_fallback_count(root, recent_rows=40) >= 3:
        plateau = recent_calibration_fallback_plateau(root, recent_rows=80)
        if plateau and plateau["no_model_escalations_done"]:
            return filter_seedable_tasks(
                root,
                calibration_plateau_tasks(root, plateau, timestamp, task_id="lane-contract-fallback-plateau"),
            )
        if not any_task_has_prefix(root, "lane-contract-runtime-overhead-map-after-fallback-"):
            return filter_seedable_tasks(
                root,
                [
                    {
                        "id": f"lane-contract-runtime-overhead-map-after-fallback-{timestamp}",
                        "status": "ready",
                        "priority": 99,
                        "lane": "runtime-overhead",
                        "task_type": "supervisor",
                        "supervisor_action": "runtime-overhead-map",
                        "target": "openclaw/openclaw-jang-vlm-server.py",
                        "hypothesis": (
                            "Calibration is blocked and fallback decode measurements repeated; map the runtime/proxy "
                            "overhead boundary once before pausing the lane."
                        ),
                        "metric": "server_wall_decode_gap",
                        "guard_checks": ["no_model_turn_required", "no_live_profile_change", "no_opencode_changes"],
                        "acceptance": "A runtime-overhead artifact identifies a patchable boundary or explicitly rules it out.",
                        "rollback": "No runtime rollback needed; this is a read-only supervisor artifact.",
                        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
                    }
                ],
            )
        if not any_task_has_prefix(root, "lane-contract-mtp-report-after-fallback-"):
            return filter_seedable_tasks(
                root,
                [
                    {
                        "id": f"lane-contract-mtp-report-after-fallback-{timestamp}",
                        "status": "ready",
                        "priority": 98,
                        "lane": "exhaustion-report",
                        "task_type": "supervisor",
                        "supervisor_action": "mtp-report",
                        "target": "openclaw-model-proxy.log",
                        "hypothesis": (
                            "Calibration is blocked and runtime mapping already ran; capture MTP acceptance evidence "
                            "before marking the fallback path exhausted."
                        ),
                        "metric": "mean_accept",
                        "guard_checks": ["no_model_turn_required", "no_opencode_changes", "one_narrow_tool"],
                        "acceptance": "An MTP report artifact records server tok/s, sample count, and acceptance evidence when logs expose it.",
                        "rollback": "No runtime rollback needed; this is a read-only supervisor report.",
                        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research mtp-report --lines 240",
                        "lines": 240,
                    }
                ],
            )
        append_jsonl(
            root / "findings.jsonl",
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "task_id": "lane-contract-fallback-exhausted",
                "finding": "lane-contract fallback decode measurements repeated and bounded no-model escalations are already complete",
                "reason": calibration_blocker,
                "next": "pause autoresearch until calibration runtime prerequisites, new trace data, or a new non-calibration implementation task exists",
            },
        )
        return []
    if recent_drafter_fit_plan_ready(root, recent_rows=240) and not calibration_blocker:
        if existing_drafter_trace_paths():
            tasks.append(
                drafter_calibration_canary_task(
                    timestamp,
                    task_id=f"lane-contract-drafter-calibration-canary-{timestamp}",
                    priority=99,
                )
            )
        elif recent_drafter_trace_missing(root, recent_rows=160):
            tasks.append(
                drafter_trace_prerequisite_task(
                    timestamp,
                    task_id=f"lane-contract-drafter-trace-prerequisite-{timestamp}",
                    priority=99,
                )
            )
        else:
            tasks.append(
                drafter_trace_gate_task(
                    timestamp,
                    task_id=f"lane-contract-drafter-trace-gate-{timestamp}",
                    priority=99,
                )
            )
    if not tasks:
        if dflash_blocked and active_calibration_memory_stage_tasks(root):
            append_jsonl(
                root / "findings.jsonl",
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "task_id": "lane-contract-dflash-fallback-suppressed",
                    "finding": "DFlash is blocked and calibration memory-stage work is already ready; suppress decode remeasure churn.",
                    "reason": reason,
                    "next": "run_active_calibration_memory_stage",
                },
            )
            return []
        suffix = "calibration-block" if calibration_blocker else "ready-work-gap"
        tasks.append(
            lane_contract_decode_task(
                timestamp,
                task_id=f"lane-contract-decode-remeasure-{suffix}-{timestamp}",
                priority=99,
                reason=(
                    f"{reason}; calibration_blocker={calibration_blocker or 'none'} "
                    f"dflash_blocked={str(dflash_blocked).lower()}. "
                    "Continue with the safe normal-TUI decode metric while blocked lanes wait for new evidence."
                ),
            )
        )
    return filter_seedable_tasks(root, tasks)


def filter_seedable_tasks(root: Path, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop tasks from retired lanes before they enter the durable queue."""
    ensure_lane_contracts(root)
    exhausted = exhausted_lanes(root)
    dflash_blocked = "frontier-dflash" in exhausted or dflash_lane_is_blocked(root, recent_rows=240)
    runtime_clean_exhausted = runtime_overhead_repeated_clean(root, recent_rows=160)
    calibration_blocker = recent_calibration_run_hard_blocker(root, recent_rows=240)
    adapter_loop_saturated = adapter_logit_loop_saturated(root, recent_rows=240)
    active_calibration_stages = {
        calibration_memory_stage_name(task) for task in active_calibration_memory_stage_tasks(root)
    }
    has_active_calibration_stage = bool(active_calibration_stages - {""})
    completed_calibration_stages = completed_calibration_memory_stages(root)
    seedable: list[dict[str, Any]] = []
    for task in tasks:
        lane = str(task.get("lane", ""))
        task_id = str(task.get("id", ""))
        action = str(task.get("supervisor_action", ""))
        stage = calibration_memory_stage_name(task)
        if stage and (stage in active_calibration_stages or stage in completed_calibration_stages):
            continue
        if lane in exhausted and lane != "exhaustion-report":
            continue
        if dflash_blocked and (
            lane == "frontier-dflash"
            or action == "dflash-compatibility-gate"
            or "dflash-compatibility" in task_id
            or "lane-contract-decode-remeasure-dflash-block" in task_id
        ):
            continue
        if runtime_clean_exhausted and lane == "runtime-overhead" and "contamination" not in task_id:
            continue
        if adapter_loop_saturated and is_adapter_logit_loop_task(task):
            continue
        if has_active_calibration_stage and (
            action == "drafter-calibration-canary" or "drafter-calibration-canary" in task_id
        ):
            continue
        calibration_blocked_action = action == "drafter-calibration-run" or "drafter-calibration-run" in task_id
        if (
            calibration_blocker in CALIBRATION_CANARY_TERMINAL_BLOCKERS
            and calibration_task_mode(task) != CALIBRATION_ADAPTER_MODE
        ):
            calibration_blocked_action = calibration_blocked_action or (
                action == "drafter-calibration-canary" or "drafter-calibration-canary" in task_id
            )
        if calibration_blocker == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER:
            calibration_blocked_action = calibration_blocked_action or (
                action == "drafter-calibration-memory-stage"
                or "drafter-calibration-memory-stage" in task_id
            )
            if is_trace_distillation_task(task) and not trace_distillation_proof_failed(root, recent_rows=240):
                calibration_blocked_action = False
            if task_id.startswith("trace-distillation-gradient-repair-"):
                calibration_blocked_action = False
        if calibration_blocker and calibration_blocked_action:
            continue
        seedable.append(task)
    return seedable


def is_deterministic_research_task(task: dict[str, Any]) -> bool:
    return (
        task.get("task_type") == "supervisor"
        or bool(task.get("benchmark_mode"))
        or "openclaw-speed-research" in str(task.get("next_action", ""))
    )


def is_implementation_bridge_task(task: dict[str, Any]) -> bool:
    task_id = str(task.get("id", ""))
    return (
        task.get("supervisor_action") == "implementation-bridge"
        or task_id.startswith("implementation-bridge-")
        or task_id.startswith("handoff-audit-deterministic-bridge-")
        or task_id.startswith("frontier-repair-implementation-bridge-")
    )


def recent_empty_bridge_rows(
    root: Path,
    rows: list[dict[str, str]] | None = None,
    *,
    recent_rows: int = 80,
) -> list[dict[str, str]]:
    window = (rows if rows is not None else result_rows(root))[-max(1, recent_rows) :]
    return [
        row
        for row in window
        if row.get("run_id", "").startswith("supervisor-implementation-bridge-")
        and "ready_deterministic=0" in row.get("notes", "")
    ]


def concrete_handoff_prerequisite_tasks(root: Path, rows: list[dict[str, str]], timestamp: int) -> list[dict[str, Any]]:
    """Return one non-bridge task when handoff synthesis has already stalled."""
    tasks: list[dict[str, Any]] = []
    calibration_blocker = recent_calibration_run_hard_blocker(root, recent_rows=240)
    bottleneck_state = drafter_bottleneck_state(root, rows, recent_rows=240)
    if bottleneck_state["state"] != "no_terminal_quantized_blocker":
        bottleneck_tasks = drafter_bottleneck_next_tasks(
            root,
            rows,
            timestamp,
            reason="Implementation handoff hit the JANQ drafter bottleneck",
        )
        if bottleneck_tasks:
            return bottleneck_tasks
        if str(bottleneck_state["next_step"]).startswith("wait_for_"):
            return []
    if recent_drafter_fit_plan_ready(root, recent_rows=240):
        if existing_drafter_trace_paths() and not calibration_blocker:
            tasks.append(
                drafter_calibration_canary_task(
                    timestamp,
                    task_id=f"handoff-audit-drafter-calibration-canary-{timestamp}",
                    priority=99,
                )
            )
        elif calibration_blocker:
            if calibration_blocker == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER:
                bottleneck_tasks = drafter_bottleneck_next_tasks(
                    root,
                    rows,
                    timestamp,
                    reason="Implementation handoff hit the JANQ drafter bottleneck",
                )
                if bottleneck_tasks:
                    return bottleneck_tasks
                if trace_distillation_proof_failed(root, recent_rows=240):
                    if active_task_has_prefix(root, "trace-distillation-gradient-repair-"):
                        return []
                    if not trace_distillation_repair_attempted(root, recent_rows=240) and should_seed_action(
                        root,
                        "trace-distillation-gradient-repair-",
                        recent_rows=240,
                    ):
                        return filter_seedable_tasks(
                            root,
                            [
                                trace_distillation_repair_task(
                                    timestamp,
                                    task_id=f"trace-distillation-gradient-repair-{timestamp}",
                                )
                            ],
                        )
                    expansion_tasks = frontier_expansion_tasks(root, rows, timestamp)
                    if expansion_tasks:
                        return expansion_tasks
                    if not any_task_has_prefix(root, "trace-distillation-adapter-bridge-"):
                        return filter_seedable_tasks(
                            root,
                            [
                                trace_distillation_adapter_bridge_task(
                                    timestamp,
                                    task_id=f"trace-distillation-adapter-bridge-{timestamp}",
                                )
                            ],
                        )
                elif not trace_distillation_attempt_exists(root, recent_rows=240):
                    distillation_task = trace_distillation_candidate_task(
                        root,
                        timestamp,
                        task_id=f"handoff-audit-drafter-trace-distillation-run-{timestamp}",
                    )
                    if distillation_task is not None:
                        return filter_seedable_tasks(root, [distillation_task])
            plateau = recent_calibration_fallback_plateau(root, rows, recent_rows=100)
            if plateau:
                return filter_seedable_tasks(
                    root,
                    calibration_plateau_tasks(root, plateau, timestamp, task_id="handoff-audit-calibration-plateau"),
                )
            tasks.append(
                {
                    "id": f"handoff-audit-decode-remeasure-after-calibration-block-{timestamp}",
                    "status": "ready",
                    "priority": 99,
                    "lane": "runtime-overhead",
                    "task_type": "supervisor",
                    "target": "decode-sample",
                    "hypothesis": (
                        "JANQ calibration is currently blocked by "
                        f"{calibration_blocker}; continue overnight work with feasible decode measurement."
                    ),
                    "metric": "decode_tps",
                    "benchmark_mode": "decode-sample",
                    "guard_checks": ["memory_gate", "no_live_profile_change", "no_opencode_changes"],
                    "acceptance": "A fresh decode benchmark records wall-clock and server tok/s without loading a second JANQ target.",
                    "rollback": "No rollback needed; benchmark-only task.",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
                }
            )
        elif recent_drafter_trace_missing(root, recent_rows=160):
            tasks.append(
                drafter_trace_prerequisite_task(
                    timestamp,
                    task_id=f"handoff-audit-drafter-trace-prerequisite-{timestamp}",
                    priority=99,
                )
            )
        else:
            tasks.append(
                drafter_trace_gate_task(
                    timestamp,
                    task_id=f"handoff-audit-drafter-trace-gate-{timestamp}",
                    priority=99,
                )
            )
    elif should_seed_runtime_overhead_map(root, rows, recent_rows=60):
        tasks.append(
            {
                "id": f"handoff-audit-runtime-overhead-map-{timestamp}",
                "status": "ready",
                "priority": 98,
                "lane": "runtime-overhead",
                "task_type": "supervisor",
                "supervisor_action": "runtime-overhead-map",
                "target": "openclaw/openclaw-jang-vlm-server.py",
                "hypothesis": "Implementation handoff stalled, so map the runtime/proxy overhead boundary before another bridge.",
                "metric": "server_wall_decode_gap",
                "guard_checks": ["no_model_turn_required", "no_live_profile_change", "no_opencode_changes"],
                "acceptance": "A runtime-overhead artifact identifies a patchable boundary or explicitly rules out local source changes.",
                "rollback": "No runtime rollback needed; this is a read-only supervisor artifact.",
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
            }
        )
    else:
        tasks.append(drafter_fit_task(timestamp, task_id=f"handoff-audit-drafter-fit-plan-{timestamp}", priority=97))
    return filter_seedable_tasks(root, tasks)


def synthesis_deliberate_action_tasks(root: Path, rows: list[dict[str, str]], timestamp: int) -> list[dict[str, Any]]:
    """Create one high-signal next action when static synthesis candidates are exhausted."""
    recent = rows[-120:]
    tasks: list[dict[str, Any]] = []
    exhausted = exhausted_lanes(root)
    sweep_rows = completed_drafter_sweep_rows(root, recent_rows=120, min_sweeps=3)
    sweep_fields = [parse_note_fields(row.get("notes", "")) for row in sweep_rows]
    last_sweeps = sweep_fields[-3:]
    block_sweep_converged = (
        len(last_sweeps) >= 2
        and all(fields.get("decision") == "keep-current" for fields in last_sweeps)
        and all(fields.get("winner_block") in {"", "2"} for fields in last_sweeps)
    )
    decode_mean = latest_decode_mean(root, recent_rows=80)
    below_practical_floor = decode_mean is None or decode_mean < 20
    recent_mtp_report = recent_result_has_prefix(root, "mtp-report-", recent_rows=30)
    clean_runtime_maps = recent_clean_runtime_overhead_maps(root, recent, recent_rows=45)
    drafter_plan_ready = recent_drafter_fit_plan_ready(root, recent_rows=160)
    dflash_suppressed = suppress_hard_blocked_dflash_lane(root, recent_rows=160)
    dflash_blocked = dflash_suppressed or dflash_lane_is_blocked(root, recent_rows=160) or "frontier-dflash" in exhausted
    bottleneck_state = drafter_bottleneck_state(root, rows, recent_rows=240)

    if bottleneck_state["state"] != "no_terminal_quantized_blocker":
        bottleneck_tasks = drafter_bottleneck_next_tasks(
            root,
            rows,
            timestamp,
            reason="Synthesis detected the canonical JANQ drafter bottleneck route",
        )
        if bottleneck_tasks:
            return bottleneck_tasks
        if str(bottleneck_state["next_step"]).startswith("wait_for_"):
            return []

    if block_sweep_converged and below_practical_floor:
        mark_lane_exhausted(
            root,
            lane="mtp-decode",
            reason="block-size sweep repeatedly kept block 2 below target; route to drafter alignment or runtime overhead",
            evidence={
                "last_sweeps": last_sweeps,
                "decode_mean": decode_mean,
            },
        )

    if not recent_mtp_report and should_seed_action(root, "deliberate-mtp-report-", recent_rows=30):
        tasks.append(
            {
                "id": f"deliberate-mtp-report-{timestamp}",
                "status": "ready",
                "priority": 97,
                "lane": "mtp-decode",
                "task_type": "supervisor",
                "supervisor_action": "mtp-report",
                "target": "openclaw-model-proxy.log",
                "hypothesis": "Before another speed experiment, capture MTP acceptance evidence from the live proxy log.",
                "metric": "mean_accept",
                "guard_checks": ["no_model_turn_required", "no_opencode_changes", "one_narrow_tool"],
                "acceptance": "An MTP report artifact records server tok/s, MTP sample count, and mean acceptance when logs expose it.",
                "rollback": "No runtime rollback needed; this is a read-only supervisor report.",
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research mtp-report --lines 240",
                "lines": 240,
            }
        )
        return tasks

    if block_sweep_converged and below_practical_floor:
        if should_seed_runtime_overhead_map(root, recent, recent_rows=45):
            tasks.append(
                {
                    "id": f"deliberate-runtime-overhead-map-{timestamp}",
                    "status": "ready",
                    "priority": 96,
                    "lane": "runtime-overhead",
                    "task_type": "supervisor",
                    "supervisor_action": "runtime-overhead-map",
                    "target": "openclaw/openclaw-jang-vlm-server.py",
                    "hypothesis": "Block-size tuning has converged below target, so the next likely speed gain is reducing MTP loop or proxy overhead.",
                    "metric": "server_wall_decode_gap",
                    "guard_checks": ["no_live_profile_change", "no_model_turn_required", "no_opencode_changes", "tests_before_patch"],
                    "acceptance": "A runtime-overhead artifact identifies one patchable boundary or explicitly rules out local source changes.",
                    "rollback": "No live rollback needed unless a later source patch is created and canary-tested.",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
                }
            )
            return tasks
        if drafter_plan_ready:
            if should_seed_drafter_calibration_canary(root, recent_rows=80):
                tasks.append(
                    drafter_calibration_canary_task(
                        timestamp,
                        task_id=f"deliberate-drafter-calibration-canary-{timestamp}",
                        priority=98,
                    )
                )
                return tasks
            if (
                not active_task_has_prefix(root, "deliberate-drafter-trace-gate-")
                and not recent_drafter_trace_gate(root, recent_rows=80)
            ):
                tasks.append(drafter_trace_gate_task(timestamp, task_id=f"deliberate-drafter-trace-gate-{timestamp}"))
                return tasks
        elif should_seed_action(root, "deliberate-drafter-fit-plan-", recent_rows=120):
            task = drafter_fit_task(timestamp, task_id=f"deliberate-drafter-fit-plan-{timestamp}")
            if clean_runtime_maps:
                task["hypothesis"] = (
                    "Runtime-overhead mapping is clean, so the next frontier path is JANQ-specific drafter fit "
                    "rather than another source map."
                )
            tasks.append(task)
            return tasks
        if not dflash_blocked and should_seed_action(root, "deliberate-dflash-compatibility-", recent_rows=55):
            tasks.append(dflash_compatibility_task(timestamp, task_id=f"deliberate-dflash-compatibility-{timestamp}"))
            return filter_seedable_tasks(root, tasks)

    if not block_sweep_converged and len(sweep_rows) < 2 and should_seed_action(root, "deliberate-drafter-sweep-", recent_rows=40):
        tasks.append(
            {
                "id": f"deliberate-drafter-sweep-{timestamp}",
                "status": "ready",
                "priority": 94,
                "lane": "mtp-decode",
                "task_type": "supervisor",
                "supervisor_action": "drafter-sweep-run",
                "target": "OPENCLAW_JANG_DRAFT_BLOCK_SIZE",
                "hypothesis": "A small paired block sweep is still needed before declaring MTP block-size tuning exhausted.",
                "metric": "decode_tps",
                "guard_checks": ["memory_gate", "bounded_trials", "tests_pass", "no_model_change", "restore_live_profile"],
                "acceptance": "A paired sweep artifact records control and variant decode TPS with a keep/discard decision.",
                "rollback": "Restore live profile after every variant and keep current settings unless the promotion gate passes.",
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-sweep-run --blocks 1,2,3,4",
                "blocks": "1,2,3,4",
                "samples": 3,
                "retries": 2,
            }
        )
    return filter_seedable_tasks(root, tasks)


def frontier_expansion_task(
    timestamp: int,
    *,
    slug: str,
    priority: int,
    target: str,
    hypothesis: str,
    acceptance: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Create one no-model candidate path when existing frontier lanes are exhausted."""
    return {
        "id": f"frontier-expansion-{slug}-{timestamp}",
        "status": "ready",
        "priority": priority,
        "lane": "frontier-expansion",
        "task_type": "supervisor",
        "supervisor_action": "focused-test",
        "target": target,
        "source_files": [
            "openclaw/openclaw-speed-research.py",
            "openclaw/openclaw-speed-research-autopilot.py",
            "openclaw/test-speed-research.py",
        ],
        "hypothesis": hypothesis,
        "metric": "frontier_candidate_gate",
        "guard_checks": [
            "no_model_load",
            "canary_only",
            "tests_pass",
            "no_live_profile_change",
            "no_opencode_changes",
            "rollback_path",
        ],
        "acceptance": acceptance,
        "rollback": "Discard this candidate path unless its canary remains green and a later paired benchmark beats the current TUI decode baseline.",
        "evidence": evidence,
        "next_action": "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py",
    }


def frontier_expansion_tasks(root: Path, rows: list[dict[str, str]], timestamp: int) -> list[dict[str, Any]]:
    """Route exhausted lanes into one new bounded candidate path instead of terminal churn."""
    ensure_lane_contracts(root)
    recent = rows[-160:]
    exhausted = set(exhausted_lanes(root))
    calibration_blocker = recent_calibration_run_hard_blocker(root, recent_rows=240)
    dflash_blocked = (
        "frontier-dflash" in exhausted
        or dflash_lane_is_blocked(root, recent_rows=240)
        or suppress_hard_blocked_dflash_lane(root, recent_rows=240)
    )
    clean_runtime_maps = recent_clean_runtime_overhead_maps(root, recent, recent_rows=80)
    decode_mean = latest_decode_mean(root, recent_rows=120)
    sweep_rows = completed_drafter_sweep_rows(root, recent_rows=160, min_sweeps=3)
    sweep_fields = [parse_note_fields(row.get("notes", "")) for row in sweep_rows[-3:]]
    block_sweep_settled = (
        len(sweep_fields) >= 2
        and all(fields.get("decision") == "keep-current" for fields in sweep_fields)
        and all(fields.get("winner_block") in {"", "2"} for fields in sweep_fields)
    )
    evidence = {
        "exhausted_lanes": sorted(exhausted),
        "calibration_blocker": calibration_blocker,
        "dflash_blocked": dflash_blocked,
        "clean_runtime_maps": len(clean_runtime_maps),
        "decode_mean_tps": decode_mean,
        "block_sweep_settled": block_sweep_settled,
    }

    candidates: list[tuple[str, dict[str, Any]]] = []
    if calibration_blocker == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER:
        candidates.append(
            (
                "frontier-expansion-janq-adapter-path-",
                frontier_expansion_task(
                    timestamp,
                    slug="janq-adapter-path",
                    priority=99,
                    target="openclaw/openclaw-mtp-drafter-calibrate.py",
                    hypothesis=(
                        "JANQ calibration is blocked because the quantized target does not expose gradients; "
                        "the next candidate is a trainable adapter or logit-distillation path that uses "
                        "target-generated traces without differentiating through JANQ weights."
                    ),
                    acceptance=(
                        "A canary-only source path exists for adapter/logit-distillation calibration, with "
                        "no target-model mutation, no live profile mutation, and a later decode TPS promotion gate."
                    ),
                    evidence=evidence,
                ),
            )
        )
    if dflash_blocked:
        candidates.append(
            (
                "frontier-expansion-dflash-candidate-search-",
                frontier_expansion_task(
                    timestamp,
                    slug="dflash-candidate-search",
                    priority=97,
                    target="openclaw/openclaw-drafter-fit.py",
                    hypothesis=(
                        "DFlash is blocked for the current drafter candidate; the next path is a candidate-search "
                        "gate that only reopens DFlash when a same-tokenizer JANQ-compatible draft candidate changes."
                    ),
                    acceptance=(
                        "The candidate-search gate records why the current DFlash candidate is retired, what "
                        "candidate evidence would reopen it, and preserves the existing MTP path until paired tests win."
                    ),
                    evidence=evidence,
                ),
            )
        )
    if "mtp-decode" in exhausted or block_sweep_settled:
        candidates.append(
            (
                "frontier-expansion-mtp-verify-cache-",
                frontier_expansion_task(
                    timestamp,
                    slug="mtp-verify-cache",
                    priority=95,
                    target="openclaw/openclaw-jang-vlm-server.py",
                    hypothesis=(
                        "Block-size tuning has settled below target; inspect MTP verify/cache/rollback instrumentation "
                        "next so speed work can target accepted-token yield rather than repeat block sweeps."
                    ),
                    acceptance=(
                        "A no-model canary confirms the MTP verification/cache/rollback path has explicit metrics "
                        "or a scoped source patch candidate with rollback before any live benchmark promotion."
                    ),
                    evidence=evidence,
                ),
            )
        )
    if len(clean_runtime_maps) >= 2:
        candidates.append(
            (
                "frontier-expansion-runtime-source-bridge-",
                frontier_expansion_task(
                    timestamp,
                    slug="runtime-source-bridge",
                    priority=93,
                    target="openclaw/openclaw-model-proxy.py",
                    hypothesis=(
                        "Repeated clean runtime maps mean measurement alone is exhausted; create one source-bridge "
                        "candidate for the smallest proxy/server overhead patch that canary tests can verify."
                    ),
                    acceptance=(
                        "The bridge names exactly one patchable runtime boundary, its canary test, and a rollback "
                        "path; otherwise it retires runtime-overhead until new evidence appears."
                    ),
                    evidence=evidence,
                ),
            )
        )

    for prefix, task in candidates:
        if any_task_has_prefix(root, prefix):
            continue
        if should_seed_action(root, prefix, recent_rows=240):
            return filter_seedable_tasks(root, [task])
    return []


def source_scout_task(timestamp: int, *, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"agent-deliberation-source-scout-{timestamp}",
        "status": "ready",
        "priority": 100,
        "lane": "frontier-deliberation",
        "task_type": "supervisor",
        "supervisor_action": "source-scout",
        "target": "external-speed-references",
        "topic": "JANQ Gemma4 drafter fit, MTP acceptance, DFlash, Rapid-MLX decode speed",
        "hypothesis": (
            "When local lanes are exhausted, a source-scout agent should gather current references before "
            "the architect creates another implementation path."
        ),
        "metric": "source_evidence_count",
        "guard_checks": [
            "allowlisted_hosts_only",
            "timeout_bounded",
            "no_model_load",
            "no_live_profile_change",
            "no_opencode_changes",
            "rollback_path",
        ],
        "acceptance": "A source-scout artifact records fetched/skipped allowlisted references and concrete next-source gaps.",
        "rollback": "No rollback needed; this task writes an evidence artifact only and never mutates runtime or source.",
        "evidence": evidence,
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research source-scout --topic frontier-decode-speed",
    }


def mtp_acceptance_yield_task(timestamp: int, *, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"agent-deliberation-mtp-acceptance-yield-{timestamp}",
        "status": "ready",
        "priority": 97,
        "lane": "frontier-deliberation",
        "task_type": "supervisor",
        "supervisor_action": "mtp-report",
        "target": "openclaw-model-proxy.log",
        "lines": 320,
        "hypothesis": (
            "If drafter training is blocked, the next useful speed move is to measure MTP accepted-token "
            "yield and rejection evidence instead of repeating raw decode samples."
        ),
        "metric": "mean_accept",
        "guard_checks": [
            "no_model_turn_required",
            "one_narrow_tool",
            "no_live_profile_change",
            "no_opencode_changes",
            "rollback_path",
        ],
        "acceptance": (
            "An MTP report artifact records server tok/s, sample count, mean_accept when logs expose it, "
            "and the next patchable acceptance-yield bottleneck."
        ),
        "rollback": "No rollback needed; this task reads logs only and never mutates runtime or source.",
        "evidence": evidence,
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research mtp-report --lines 320",
    }


def agent_deliberation_task(timestamp: int, *, slug: str, priority: int, target: str, hypothesis: str, acceptance: str, evidence: dict[str, Any]) -> dict[str, Any]:
    task = {
        "id": f"agent-deliberation-{slug}-{timestamp}",
        "status": "ready",
        "priority": priority,
        "lane": "frontier-deliberation",
        "task_type": "supervisor",
        "supervisor_action": "focused-test",
        "target": target,
        "source_files": [
            "openclaw/openclaw-speed-research.py",
            "openclaw/openclaw-speed-research-autopilot.py",
            "openclaw/test-speed-research.py",
        ],
        "hypothesis": hypothesis,
        "metric": "frontier_deliberation_contract",
        "guard_checks": [
            "no_model_load",
            "canary_only",
            "tests_pass",
            "no_live_profile_change",
            "no_opencode_changes",
            "rollback_path",
        ],
        "acceptance": acceptance,
        "rollback": "Discard the contract unless canary tests pass and later paired decode benchmarks beat the live baseline.",
        "evidence": evidence,
        "next_action": "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py",
    }
    if crabbox_required_for_target(target):
        apply_crabbox_requirement(task, reason=f"deliberation target is high-risk: {target}")
    return task


def latest_source_scout_artifact(root: Path) -> dict[str, Any]:
    return latest_json_artifact(root, "source-scout-*.json")


def frontier_agent_deliberation(root: Path, rows: list[dict[str, str]], timestamp: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create a bounded scout -> skeptic -> architect deliberation when known lanes are exhausted."""
    exhausted = set(exhausted_lanes(root))
    calibration_blocker = recent_calibration_run_hard_blocker(root, recent_rows=240)
    canonical = canonical_autoresearch_state(root, recent_rows=160)
    bottleneck_state = drafter_bottleneck_state(root, rows, recent_rows=240)
    adapter_loop_saturated = adapter_logit_loop_saturated(root, recent_rows=240)
    bottleneck_tasks: list[dict[str, Any]] = []
    if bottleneck_state["state"] != "no_terminal_quantized_blocker":
        bottleneck_tasks = drafter_bottleneck_next_tasks(
            root,
            rows,
            timestamp,
            reason="Frontier deliberation selected the canonical JANQ drafter bottleneck route",
        )
    source_artifact = latest_source_scout_artifact(root)
    decode_mean = latest_decode_mean(root, recent_rows=160)
    measurement = measurement_artifact_analysis(root, recent_rows=160)
    evidence = {
        "canonical_state": canonical.get("state", ""),
        "canonical_clean": canonical.get("clean", False),
        "active_noise": canonical.get("noise", {}),
        "exhausted_lanes": sorted(exhausted),
        "calibration_blocker": calibration_blocker,
        "drafter_bottleneck_state": bottleneck_state,
        "decode_mean_tps": decode_mean,
        "measurement_artifact": measurement,
        "source_scout_artifact": source_artifact.get("_artifact_path", ""),
    }
    scout = {
        "role": "scout",
        "mission": "find one new high-upside decode-speed path after local lanes are exhausted",
        "source_urls": list(FRONTIER_SOURCE_URLS),
        "queries": [
            "Gemma 4 MTP drafter speculative decoding acceptance",
            "DFlash drafter JANQ quantized model compatibility",
            "Rapid-MLX prefix cache MTP decode speed Apple Silicon",
            "self improving coding agent evaluator supervisor GEPA",
        ],
        "proposals": [
            "source-scout-refresh",
            "adapter-logit-distillation-contract",
            "mtp-acceptance-yield-model",
        ],
    }
    rejected: list[dict[str, str]] = []
    if "mtp-decode" in exhausted:
        rejected.append(
            {
                "proposal": "repeat-mtp-block-sweep",
                "reason": "MTP block-size tuning is exhausted; repeat measurements are not new evidence.",
            }
        )
    if "frontier-dflash" in exhausted:
        rejected.append(
            {
                "proposal": "retry-current-dflash-candidate",
                "reason": "DFlash is exhausted until the draft candidate changes.",
            }
        )
    skeptic = {
        "role": "skeptic",
        "hard_rejections": rejected,
        "required_gates": [
            "canonical_clean",
            "zero_active_noise",
            "no_live_profile_change",
            "no_opencode_changes",
            "contract_has_acceptance_and_rollback",
        ],
    }
    selected_task: dict[str, Any] | None = None
    selected_reason = ""
    if bottleneck_tasks:
        selected_task = bottleneck_tasks[0]
        selected_reason = f"canonical JANQ drafter bottleneck next_step={bottleneck_state.get('next_step')}"
    elif (
        not source_artifact
        and not active_task_has_prefix(root, "agent-deliberation-source-scout-")
        and not recent_keep_result_has_prefix(root, "source-scout-", recent_rows=240)
    ):
        selected_task = source_scout_task(timestamp, evidence=evidence)
        selected_reason = "refresh external references before inventing another implementation contract"
    elif (
        calibration_blocker == CALIBRATION_QUANTIZED_GRADIENT_BLOCKER
        and not adapter_loop_saturated
        and not active_task_has_prefix(
            root, "agent-deliberation-adapter-logit-contract-"
        )
    ):
        selected_task = drafter_adapter_method_contract_task(
            timestamp,
            task_id=f"agent-deliberation-adapter-logit-contract-{timestamp}",
            priority=99,
        )
        selected_task["evidence"] = evidence
        selected_reason = "quantized-gradient blocker requires a non-gradient-through-target adapter path"
    elif not active_task_has_prefix(root, "agent-deliberation-mtp-acceptance-yield-"):
        selected_task = mtp_acceptance_yield_task(timestamp, evidence=evidence)
        selected_reason = "no fresh trainable path exists, so improve acceptance-yield observability"
    if selected_task is None:
        selected_task = agent_deliberation_task(
            timestamp,
            slug="open-problem-contract",
            priority=95,
            target="openclaw/openclaw-speed-research.py",
            hypothesis=(
                "All named frontier deliberation paths have prior evidence; create a fresh open-problem "
                "contract that requires new evidence before any implementation task can run."
            ),
            acceptance=(
                "The contract records exhausted proposals, required new evidence, source-scout requirements, "
                "and a no-op rollback; it cannot promote code or mutate runtime by itself."
            ),
            evidence=evidence,
        )
        selected_reason = "all named deliberation paths have been attempted, so create a fresh evidence contract"
    if selected_task:
        selected_task["id"] = unique_task_id(root, str(selected_task.get("id", "")))
    architect = {
        "role": "architect",
        "selected_task_id": selected_task.get("id", "") if selected_task else "",
        "selected_reason": selected_reason,
        "contract_complete": bool(
            selected_task
            and selected_task.get("acceptance")
            and selected_task.get("rollback")
            and "no_opencode_changes" in {str(item) for item in selected_task.get("guard_checks", [])}
        ),
    }
    noise = canonical.get("noise") if isinstance(canonical.get("noise"), dict) else {}
    unsafe_noise_clear = int(noise.get("unresolved_blocked_rows", 0) or 0) == 0 and int(noise.get("memory_blocks", 0) or 0) == 0
    gates = {
        "canonical_clean_or_repairable_terminal": bool(canonical.get("clean")) or unsafe_noise_clear,
        "zero_unsafe_noise": unsafe_noise_clear,
        "task_selected": selected_task is not None,
        "contract_complete": bool(architect["contract_complete"]),
    }
    tasks = [selected_task] if selected_task and all(gates.values()) else []
    report = {
        "ok": bool(tasks),
        "kind": "frontier-agent-deliberation",
        "timestamp": timestamp,
        "scout": scout,
        "skeptic": skeptic,
        "architect": architect,
        "gates": gates,
        "evidence": evidence,
        "seeded_tasks": [task["id"] for task in tasks],
        "next": "run selected deterministic task" if tasks else "pause; no safe deliberation contract passed",
    }
    return report, tasks


def frontier_deliberation(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    timestamp = int(time.time())
    report, tasks = frontier_agent_deliberation(root, result_rows(root), timestamp)
    seeded = upsert_tasks(root, tasks) if tasks else 0
    report["seeded"] = seeded
    path = root / "benchmarks" / f"frontier-agent-deliberation-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "frontier-agent-deliberation",
            "finding": "scout/skeptic/architect deliberation selected one bounded deterministic path",
            "evidence": report,
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"frontier-agent-deliberation-{timestamp}",
        status="keep" if seeded else "discard",
        target="frontier-agent-deliberation",
        hypothesis="exhausted lanes should trigger bounded agent-to-agent deliberation instead of terminal churn",
        commit=current_commit(repo_root()),
        notes=f"seeded={seeded} task_ids={','.join(report['seeded_tasks'])} gates={report['gates']}",
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0 if seeded or args.allow_empty else 2


def research_quality_scorecard(
    *,
    blocked_rows: int,
    missing_required_blocks: list[str],
    sweep_rows: int,
    min_sweeps: int,
    repeated_block2: bool,
    repeated_keep_current: bool,
    plateau_below_target: bool,
    exhaustion_candidate: bool,
    frontier_ready: list[str],
    seeded_tasks: list[dict[str, Any]],
    ready_tasks: list[dict[str, Any]],
    contaminated_rows: int,
    clean_runtime_maps: int,
    variance: dict[str, Any],
    artifact_check: dict[str, Any],
    contract_ok: bool,
    dflash_suppressed: bool,
    repeated_dflash_synthesis: int,
    duplicate_stage_tasks: int,
    best_mean: float | None,
    target_tps: float,
    server_decode_values: list[float],
    canonical_state: str = "",
) -> dict[str, Any]:
    """Score the quality of the research loop, not just whether speed improved."""
    seeded_ids = [str(task.get("id", "")) for task in seeded_tasks]
    ready_ids = [str(task.get("id", "")) for task in ready_tasks]
    routed_ids = seeded_ids + ready_ids
    routed_guarded = [
        task
        for task in seeded_tasks + ready_tasks
        if task.get("acceptance") and task.get("rollback") and task.get("guard_checks")
    ]
    has_next_action = bool(frontier_ready or seeded_tasks or ready_tasks)
    has_causal_plateau = repeated_block2 and repeated_keep_current and best_mean is not None
    has_prerequisite_route = any(
        fragment in task_id
        for task_id in routed_ids
        for fragment in (
            "agent-deliberation",
            "drafter-calibration-canary",
            "drafter-calibration-memory-stage",
            "drafter-trace-gate",
            "drafter-fit",
            "calibration-memory-report",
            "runtime-overhead",
            "runtime-map",
            "source-scout",
            "frontier-deliberation",
            "low-signal-",
            "frontier-expansion",
            "exhaustion-report",
            "drafter-adapter-method-contract",
            "implementation-drafter-adapter-method",
        )
    )
    has_deterministic_blocker_route = any(
        fragment in task_id
        for task_id in routed_ids
        for fragment in (
            "calibration-memory-report",
            "lane-contract-",
            "decode-remeasure-ready-work-gap",
            "low-signal-",
            "runtime-map",
        )
    )
    durable_sweep_coverage = repeated_block2 and repeated_keep_current and sweep_rows >= min_sweeps
    coverage_resolved = has_prerequisite_route or durable_sweep_coverage

    evidence = 100.0
    if sweep_rows < min_sweeps:
        evidence -= 22.0
    if missing_required_blocks and coverage_resolved:
        evidence -= 0.0
    elif missing_required_blocks:
        evidence -= 20.0
    if artifact_check.get("artifact_suspected"):
        evidence -= 24.0
    if contaminated_rows:
        evidence -= min(18.0, contaminated_rows * 4.0)
    if blocked_rows and not (dflash_suppressed or has_prerequisite_route):
        evidence -= min(22.0, blocked_rows * 6.0)
    if server_decode_values:
        evidence += 4.0
    if duplicate_stage_tasks:
        evidence -= min(55.0, 18.0 + duplicate_stage_tasks * 2.0)

    novelty = 82.0
    if repeated_dflash_synthesis >= 2 and not dflash_suppressed:
        novelty -= 28.0
    if clean_runtime_maps >= 2 and not has_prerequisite_route:
        novelty -= 22.0
    if dflash_suppressed:
        novelty += 8.0
    if has_prerequisite_route:
        novelty += 10.0
    if exhaustion_candidate:
        novelty += 8.0
    if duplicate_stage_tasks:
        novelty -= min(45.0, 15.0 + duplicate_stage_tasks)

    causal = 70.0
    if has_causal_plateau:
        causal += 14.0
    if variance.get("groups"):
        causal += 8.0
    if variance.get("significant_best") or plateau_below_target:
        causal += 6.0
    if not artifact_check.get("artifact_suspected"):
        causal += 4.0
    if best_mean is not None:
        causal += 4.0
    if has_prerequisite_route:
        causal += 8.0
    if canonical_state in {"prerequisite_needed", "breakthrough_lane_active", "plateau_detected", "frontier_healthy"}:
        causal += 8.0
    if coverage_resolved and any("agent-deliberation" in task_id for task_id in routed_ids):
        causal += 10.0
    if has_deterministic_blocker_route and coverage_resolved:
        causal += 8.0
    if contaminated_rows and not any("runtime-overhead" in task_id for task_id in seeded_ids):
        causal -= 16.0
    if duplicate_stage_tasks:
        causal -= min(42.0, 14.0 + duplicate_stage_tasks)

    next_action = 55.0
    if has_next_action:
        next_action += 24.0
    if routed_guarded:
        next_action += 10.0
    if has_prerequisite_route:
        next_action += 8.0
    if canonical_state in {"prerequisite_needed", "breakthrough_lane_active", "frontier_healthy"}:
        next_action += 4.0
    if not contract_ok:
        next_action -= 24.0
    if duplicate_stage_tasks:
        next_action -= min(60.0, 28.0 + duplicate_stage_tasks * 2.0)

    convergence = 72.0
    if repeated_block2 and repeated_keep_current:
        convergence += 14.0
    if plateau_below_target:
        convergence += 8.0
    if exhaustion_candidate:
        convergence += 10.0
    if dflash_suppressed:
        convergence += 8.0
    if canonical_state in {"plateau_detected", "prerequisite_needed", "breakthrough_lane_active", "frontier_healthy"}:
        convergence += 6.0
    if clean_runtime_maps >= 2 and not has_prerequisite_route:
        convergence -= 14.0
    if duplicate_stage_tasks:
        convergence -= min(65.0, 30.0 + duplicate_stage_tasks * 2.0)

    implementation = 72.0
    if contract_ok:
        implementation += 10.0
    if routed_guarded:
        implementation += 12.0
    if any(
        "handoff" in task_id
        or "bridge" in task_id
        or "drafter-calibration-canary" in task_id
        or "drafter-adapter-method-contract" in task_id
        or "implementation-drafter-adapter-method" in task_id
        for task_id in routed_ids
    ):
        implementation += 6.0
    if has_deterministic_blocker_route:
        implementation += 2.0
    if canonical_state in {"prerequisite_needed", "breakthrough_lane_active", "frontier_healthy"}:
        implementation += 4.0
    if not has_next_action and best_mean is not None and best_mean < target_tps:
        implementation -= 18.0
    if duplicate_stage_tasks:
        implementation -= min(50.0, 18.0 + duplicate_stage_tasks)

    components = {
        "evidence": evidence,
        "novelty": novelty,
        "causal": causal,
        "next_action": next_action,
        "convergence": convergence,
        "implementation_readiness": implementation,
    }
    components = {key: round(max(0.0, min(value, 100.0)), 1) for key, value in components.items()}
    overall = round(sum(components.values()) / len(components), 1)
    return {
        "overall": overall,
        "components": components,
        "signals": {
            "blocked_rows": blocked_rows,
            "sweep_rows": sweep_rows,
            "missing_required_blocks": missing_required_blocks,
            "repeated_block2": repeated_block2,
            "repeated_keep_current": repeated_keep_current,
            "plateau_below_target": plateau_below_target,
            "exhaustion_candidate": exhaustion_candidate,
            "frontier_ready_lanes": frontier_ready,
            "seeded_task_ids": seeded_ids,
            "ready_task_ids": ready_ids,
            "dflash_suppressed": dflash_suppressed,
            "repeated_dflash_synthesis": repeated_dflash_synthesis,
            "duplicate_stage_tasks": duplicate_stage_tasks,
            "canonical_state": canonical_state,
        },
        "interpretation": (
            "queue_duplication_needs_repair"
            if duplicate_stage_tasks
            else
            "high_quality_exhaustion_or_prerequisite_route"
            if overall >= 85 and (exhaustion_candidate or has_prerequisite_route or dflash_suppressed)
            else "needs_more_evidence_or_clearer_next_action"
            if overall < 75
            else "healthy_research_loop"
        ),
    }


def quality_review(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    ensure_lane_contracts(root)
    compacted_stage_tasks = compact_duplicate_calibration_stage_tasks(root)
    compacted_canary_tasks = compact_stale_calibration_canary_tasks(root)
    compacted_terminal_tasks = compact_terminal_calibration_tasks(root)
    compacted_direct_canaries = compact_direct_calibration_canaries_after_bottleneck(root)
    rows = result_rows(root)
    recent = rows[-max(1, int(args.recent_rows)) :]
    compacted_runtime_tasks = compact_repeated_runtime_overhead_tasks(root, rows, recent_rows=int(args.recent_rows))
    raw_blocked = [row for row in recent if row.get("status") == "blocked"]
    blocked = unresolved_actionable_blocked_rows(recent)
    certification_blocked = [row for row in raw_blocked if is_certification_blocked_row(row)]
    decode_signals = [
        decode_measurement_signal(row)
        for row in recent
        if row.get("status") == "keep" and row.get("target") == "decode-sample"
    ]
    contaminated_signals = [signal for signal in decode_signals if signal["contaminated"]]
    clean_signals = [signal for signal in decode_signals if not signal["contaminated"]]
    server_decode_values = [
        float(signal["server_decode_tps"])
        for signal in decode_signals
        if signal.get("server_decode_tps") is not None
    ]
    decode_by_block: dict[str, list[float]] = {}
    for row in recent:
        if row.get("status") != "keep" or row.get("target") != "decode-sample":
            continue
        signal = decode_measurement_signal(row)
        if signal["contaminated"]:
            continue
        block = signal["draft_block_size"]
        if not block:
            continue
        if signal["wall_decode_tps"] is not None:
            decode_by_block.setdefault(block, []).append(float(signal["wall_decode_tps"]))
    block_summary = {
        block: {
            "samples": len(values),
            "mean_decode_tps": mean_float(values),
            "min_decode_tps": round(min(values), 3) if values else None,
            "max_decode_tps": round(max(values), 3) if values else None,
        }
        for block, values in sorted(decode_by_block.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999)
    }
    required_blocks = {"2", "3", "4"}
    covered_blocks = {block for block, summary in block_summary.items() if int(summary.get("samples") or 0) >= int(args.min_samples_per_block)}
    missing_required_blocks = sorted(required_blocks - covered_blocks, key=int)
    best_block = ""
    best_mean: float | None = None
    best_max: float | None = None
    for block, summary in block_summary.items():
        if int(summary.get("samples") or 0) < int(args.min_samples_per_block):
            continue
        mean_decode = summary.get("mean_decode_tps")
        if not isinstance(mean_decode, (int, float)):
            continue
        if best_mean is None or float(mean_decode) > best_mean:
            best_block = block
            best_mean = float(mean_decode)
            max_decode = summary.get("max_decode_tps")
            best_max = float(max_decode) if isinstance(max_decode, (int, float)) else None
    sweep_rows_in_recent = [
        row
        for row in recent
        if row.get("status") == "keep" and row.get("run_id", "").startswith("drafter-sweep-run")
    ]
    sweep_rows = completed_drafter_sweep_rows(
        root,
        recent_rows=int(args.recent_rows),
        min_sweeps=int(args.min_sweeps),
    )
    sweep_fields = [parse_note_fields(row.get("notes", "")) for row in sweep_rows]
    winner_blocks = [fields.get("winner_block", "") for fields in sweep_fields if fields.get("winner_block")]
    repeated_block2 = len(winner_blocks) >= int(args.min_sweeps) and all(block == "2" for block in winner_blocks[-int(args.min_sweeps) :])
    repeated_keep_current = len(sweep_fields) >= int(args.min_sweeps) and all(
        fields.get("decision") == "keep-current" for fields in sweep_fields[-int(args.min_sweeps) :]
    )
    durable_sweep_coverage = len(sweep_rows) >= int(args.min_sweeps) and repeated_block2 and repeated_keep_current
    tasks = read_jsonl(root / "tasks.jsonl")
    active_tasks = [task for task in tasks if task.get("status", "ready") in {"ready", "rework"}]
    canonical_state = canonical_autoresearch_state(root, recent_rows=int(args.recent_rows), target_tps=float(args.target_tps))
    blocked = list(canonical_state["unresolved_blocked_rows"])
    active_task_ids = [str(task.get("id", "")) for task in active_tasks]
    calibration_blocker = recent_calibration_run_hard_blocker(root, recent_rows=max(160, int(args.recent_rows)))
    has_terminal_calibration_route = calibration_blocker in CALIBRATION_CANARY_TERMINAL_BLOCKERS
    has_calibration_stage_route = any(
        str(task.get("supervisor_action", "")) == "drafter-calibration-memory-stage"
        or "drafter-calibration-memory-stage" in str(task.get("id", ""))
        for task in active_tasks
    )
    has_calibration_canary_route = (
        not has_calibration_stage_route
        and any("drafter-calibration-canary" in task_id for task_id in active_task_ids)
    )
    has_calibration_route = has_calibration_canary_route or has_calibration_stage_route or has_terminal_calibration_route
    trace_distillation_blocks = trace_distillation_gradient_block_rows(
        root,
        recent_rows=max(160, int(args.recent_rows)),
    )
    trace_distillation_failed = bool(trace_distillation_blocks)
    terminal_calibration_blocks = recent_terminal_calibration_block_rows(
        root,
        blocker=CALIBRATION_QUANTIZED_GRADIENT_BLOCKER,
        recent_rows=max(160, int(args.recent_rows)),
    )
    has_trace_distillation_repair_route = any(
        task_id.startswith("trace-distillation-gradient-repair-")
        for task_id in active_task_ids
    )
    trace_distillation_repair_done = trace_distillation_repair_attempted(root, recent_rows=max(160, int(args.recent_rows)))
    has_adapter_method_route = (
        any(
            task_id.startswith(("drafter-adapter-method-contract-", "implementation-drafter-adapter-method-"))
            for task_id in active_task_ids
        )
        or recent_keep_result_has_prefix(
            root,
            "drafter-adapter-method-contract-",
            recent_rows=max(240, int(args.recent_rows)),
        )
        or recent_keep_result_has_prefix(
            root,
            "implementation-drafter-adapter-method-",
            recent_rows=max(240, int(args.recent_rows)),
        )
    )
    has_changed_calibration_route = any(
        task_id.startswith(
            (
                "drafter-trace-distillation-run-",
                "trace-distillation-gradient-repair-",
                "trace-distillation-adapter-bridge-",
                "drafter-adapter-method-contract-",
                "implementation-drafter-adapter-method-",
                "frontier-expansion-janq-adapter-path-",
                "calibration-memory-report-",
            )
        )
        for task_id in active_task_ids
    ) or has_adapter_method_route
    terminal_bottleneck_routed = (
        trace_distillation_repair_done
        or has_trace_distillation_repair_route
        or has_adapter_method_route
        or any_task_has_prefix(root, "trace-distillation-adapter-bridge-")
        or any_task_has_prefix(root, "frontier-expansion-janq-adapter-path-")
    )
    repeated_canary_ready_no_stage = [
        row
        for row in recent[-60:]
        if row.get("run_id", "").startswith("drafter-calibration-canary-")
        and "decision=ready-for-bounded-calibration" in row.get("notes", "")
        and "seeded_stage_task=0" in row.get("notes", "")
    ]
    exhausted = exhausted_lanes(root)
    active_lanes = {
        str(task.get("lane", ""))
        for task in active_tasks
        if str(task.get("lane", "")) not in exhausted
    }
    active_runtime_tasks = [
        task
        for task in active_tasks
        if str(task.get("lane", "")) == "runtime-overhead" or str(task.get("supervisor_action", "")) == "runtime-overhead-map"
    ]
    frontier_lanes = {"runtime-overhead", "drafter-alignment", "frontier-dflash", "frontier-expansion"}
    frontier_ready = sorted(active_lanes & frontier_lanes)
    clean_runtime_maps = recent_clean_runtime_overhead_maps(root, recent, recent_rows=int(args.recent_rows))
    plateau_below_target = (
        repeated_block2
        and repeated_keep_current
        and best_mean is not None
        and best_mean < float(args.target_tps)
    )
    variance = variance_analysis(root, recent_rows=int(args.recent_rows), min_samples=int(args.min_samples_per_block))
    artifact_check = measurement_artifact_analysis(root, recent_rows=int(args.recent_rows))
    contract = task_contract_report(root)
    review_status = "keep"
    recommendations: list[str] = []
    gates: dict[str, Any] = {
        "no_blocked_rows": not blocked,
        "required_block_coverage": not missing_required_blocks or has_calibration_route or durable_sweep_coverage,
        "has_sweep_evidence": len(sweep_rows) >= int(args.min_sweeps),
        "has_frontier_next_lane": bool(frontier_ready),
        "target_met": best_mean is not None and best_mean >= float(args.target_tps),
        "variance_significant_best": bool(variance.get("significant_best")),
        "no_measurement_artifact": not bool(artifact_check.get("artifact_suspected")),
        "no_contaminated_wall_clock": not contaminated_signals,
        "runtime_overhead_not_repeated": len(clean_runtime_maps) < 2 or not active_runtime_tasks,
        "ready_task_contracts_ok": bool(contract.get("ok")),
    }
    quality_score = 100
    seeded_tasks: list[dict[str, Any]] = []
    dflash_suppressed = suppress_hard_blocked_dflash_lane(root, recent_rows=max(160, int(args.recent_rows)))
    dflash_blocked_or_suppressed = (
        dflash_suppressed
        or dflash_lane_is_blocked(root, recent_rows=max(160, int(args.recent_rows)))
        or "frontier-dflash" in exhausted_lanes(root)
    )
    if dflash_suppressed:
        recommendations.append("DFlash/JANQ compatibility was hard-blocked and the lane was retired until the draft candidate changes.")
    if trace_distillation_failed and not has_trace_distillation_repair_route and not trace_distillation_repair_done:
        quality_score -= 18
        recommendations.append(
            "trace-distillation proof hit the quantized-gradient blocker; suppress retries and route to adapter/logit-distillation repair."
        )
        if should_seed_action(root, "trace-distillation-gradient-repair-", recent_rows=240):
            now = int(time.time())
            seeded_tasks.append(
                trace_distillation_repair_task(
                    now,
                    task_id=f"trace-distillation-gradient-repair-{now}",
                    priority=99,
                )
            )
    repeated_terminal_calibration = len(terminal_calibration_blocks) >= 2
    if repeated_terminal_calibration and not terminal_bottleneck_routed:
        quality_score -= 28
        recommendations.append(
            "direct drafter calibration repeatedly hit the quantized-gradient terminal blocker; suppress canary reseeding and route to a changed adapter/logit-distillation path."
        )
        if should_seed_action(root, "terminal-calibration-route-", recent_rows=240):
            seeded_tasks.extend(
                lane_contract_fallback_tasks(
                    root,
                    rows,
                    int(time.time()),
                    reason="Quality review detected repeated terminal quantized-gradient calibration blockers",
                )
            )
    elif repeated_terminal_calibration:
        recommendations.append(
            "terminal quantized-gradient drafter blocker is already routed to the adapter/logit-distillation path; suppress generic drafter-fit reseeding."
        )
    repeated_deliberate_dflash = [
        row
        for row in recent[-40:]
        if row.get("target") == "synthesis" and "deliberate_actions=deliberate-dflash-compatibility-" in row.get("notes", "")
    ]
    if len(repeated_deliberate_dflash) >= 2 and not dflash_blocked_or_suppressed:
        quality_score -= 15
        recommendations.append("repeated DFlash synthesis detected without new compatibility evidence; route to prerequisite evidence or retire the lane.")
    if blocked:
        quality_score -= 30
    if len(repeated_canary_ready_no_stage) >= 3 and has_calibration_stage_route:
        quality_score -= 25
        recommendations.append(
            "calibration canary has repeatedly confirmed readiness without advancing; run the queued memory-stage before any new canary."
        )
    if missing_required_blocks and not has_calibration_route and not durable_sweep_coverage:
        quality_score -= 20
        recommendations.append(
            "coverage gap: rerun a bounded sweep before trusting conclusions; missing blocks="
            + ",".join(missing_required_blocks)
        )
    if len(sweep_rows) < int(args.min_sweeps) and not has_calibration_route:
        quality_score -= 15
        recommendations.append("not enough completed sweep artifacts yet; keep measuring before routing to implementation.")
        if should_seed_action(root, "review-drafter-sweep-next", recent_rows=20):
            seeded_tasks.append(
                {
                    "id": "review-drafter-sweep-next",
                    "status": "ready",
                    "priority": 96,
                    "lane": "mtp-decode",
                    "task_type": "supervisor",
                    "supervisor_action": "drafter-sweep-run",
                    "target": "OPENCLAW_JANG_DRAFT_BLOCK_SIZE",
                    "hypothesis": "The queue is empty but sweep evidence is still below the reviewer threshold, so run one bounded block-size sweep before concluding exhaustion.",
                    "metric": "decode_tps",
                    "guard_checks": ["memory_gate", "bounded_trials", "tests_pass", "no_model_change", "restore_live_profile"],
                    "acceptance": "A paired sweep artifact records control and variant decode TPS with a keep/discard decision.",
                    "rollback": "Restore live profile after every variant and keep current settings unless the promotion gate passes.",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-sweep-run --blocks 1,2,3,4",
                    "blocks": "1,2,3,4",
                    "samples": 3,
                    "retries": 2,
                }
            )
    if artifact_check.get("artifact_suspected"):
        quality_score -= 25
        recommendations.append(f"measurement artifact suspected: {artifact_check.get('reason')}; remeasure baseline before promotion.")
    if not contract.get("ok"):
        quality_score -= 25
        review_status = "blocked"
        first_issue = (contract.get("issues") or [{}])[0]
        recommendations.append(
            "task contract blocker: "
            f"{first_issue.get('task_id', 'unknown')} "
            f"{'; '.join(str(item) for item in first_issue.get('blockers', [])[:2])}"
        )
    if contaminated_signals:
        quality_score -= min(30, len(contaminated_signals) * 5)
        recommendations.append(
            "contaminated wall-clock decode rows detected; separate backend server_tok_s from proxy/tool recovery latency."
        )
        seeded_tasks.append(
            {
                "id": "runtime-overhead-contamination-map",
                "status": "ready",
                "priority": 95,
                "lane": "runtime-overhead",
                "task_type": "supervisor",
                "supervisor_action": "runtime-overhead-map",
                "target": "openclaw/openclaw-jang-vlm-server.py",
                "hypothesis": "Server decode is healthy while wall-clock decode is polluted by proxy/tool recovery; map the exact runtime overhead boundary.",
                "metric": "server_wall_decode_gap",
                "guard_checks": ["no_live_profile_change", "no_model_turn_required", "no_opencode_changes"],
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
            }
        )
    if variance.get("groups") and not variance.get("significant_best") and not has_adapter_method_route:
        quality_score -= 10
        recommendations.append("variance gate: best decode result has not cleared the observed noise band; keep measuring or change hypothesis.")
    if len(clean_runtime_maps) >= 2:
        if active_runtime_tasks:
            quality_score -= 25
        recommendations.append(
            "runtime-overhead map has repeatedly reported clean measurements; stop repeating that lane until a fresh contaminated benchmark appears."
        )
        if repeated_terminal_calibration and terminal_bottleneck_routed:
            recommendations.append("skip calibration-canary/drafter-fit reseeding because the JANQ drafter bottleneck route is already canonical.")
        elif should_seed_drafter_calibration_canary(root, recent_rows=40):
            seeded_tasks.append(
                drafter_calibration_canary_task(
                    timestamp=int(time.time()),
                    task_id=f"review-drafter-calibration-canary-{int(time.time())}",
                    priority=98,
                )
            )
        elif should_seed_action(root, "review-janq-drafter-fit-next", recent_rows=10):
            seeded_tasks.append(drafter_fit_task(timestamp=int(time.time()), task_id="review-janq-drafter-fit-next", priority=96))
        if (
            not dflash_blocked_or_suppressed
            and "frontier-dflash" not in exhausted_lanes(root)
            and should_seed_action(root, "review-dflash-compatibility-next", recent_rows=10)
        ):
            seeded_tasks.append(
                dflash_compatibility_task(timestamp=int(time.time()), task_id="review-dflash-compatibility-next", priority=94)
            )
    if repeated_block2 and repeated_keep_current:
        recommendations.append("block-size sweep has converged on block 2; move to acceptance, drafter-fit, DFlash, and MTP-loop overhead.")
        mark_lane_exhausted(
            root,
            lane="mtp-decode",
            reason="quality review found repeated keep-current block-2 wins below target",
            evidence={"best_block": best_block, "best_mean_decode_tps": best_mean, "sweep_rows": len(sweep_rows)},
        )
        if not clean_runtime_maps:
            seeded_tasks.append(
                {
                    "id": "review-mtp-loop-overhead-next",
                    "status": "ready",
                    "priority": 94,
                    "lane": "runtime-overhead",
                    "task_type": "supervisor",
                    "supervisor_action": "runtime-overhead-map",
                    "target": "openclaw/openclaw-jang-vlm-server.py",
                    "hypothesis": "Repeated block-2 wins mean the next plausible path to 30+ tok/s is reducing MTP verification/cache/rollback overhead.",
                    "metric": "decode_tps_delta",
                    "guard_checks": ["one_narrow_tool", "no_live_profile_change", "tests_before_patch"],
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
                },
            )
        if should_seed_drafter_calibration_canary(root, recent_rows=40):
            seeded_tasks.append(
                drafter_calibration_canary_task(
                    timestamp=int(time.time()),
                    task_id=f"review-drafter-calibration-canary-{int(time.time())}",
                    priority=98,
                )
            )
        elif not (repeated_terminal_calibration and terminal_bottleneck_routed):
            seeded_tasks.append(drafter_fit_task(timestamp=int(time.time()), task_id="review-janq-drafter-fit-next", priority=92))
        if not dflash_blocked_or_suppressed and "frontier-dflash" not in exhausted_lanes(root):
            seeded_tasks.append(dflash_compatibility_task(timestamp=int(time.time()), task_id="review-dflash-compatibility-next", priority=90))
        else:
            recommendations.append("DFlash compatibility is already blocked by a hard draft mismatch; do not re-seed that lane until the draft candidate changes.")
    if blocked:
        review_status = "blocked"
        recommendations.append(f"recent run has {len(blocked)} blocked rows; inspect the last blocker before trusting speed conclusions.")
    if contaminated_signals and "runtime-overhead" in frontier_ready:
        review_status = "blocked"
    if plateau_below_target and frontier_ready:
        recommendations.append(
            f"plateau detected below {args.target_tps} tok/s; prioritize frontier lanes={','.join(frontier_ready)} over more block sweeps."
        )
        mark_lane_exhausted(
            root,
            lane="mtp-decode",
            reason="block-size/live MTP sweep plateaued below target; route to drafter alignment, DFlash, or runtime overhead",
            evidence={"best_block": best_block, "best_mean_decode_tps": best_mean, "variance": variance},
        )
    exhaustion_candidate = (
        plateau_below_target
        and not frontier_ready
        and not missing_required_blocks
        and len(sweep_rows) >= int(args.min_sweeps)
    )
    if exhaustion_candidate:
        recommendations.append(
            "exhaustion candidate: block-size tuning is settled below target and no frontier tasks are ready; produce a bottleneck report before more overnight cycles."
        )
        seeded_tasks.append(
            {
                "id": "review-exhaustion-report",
                "status": "ready",
                "priority": 96,
                "lane": "exhaustion-report",
                "target": "STRATEGY.md/results.tsv/quality-review",
                "hypothesis": "If all local speed lanes are exhausted below target, the harness must state the hardware/runtime bottleneck with evidence.",
                "metric": "bottleneck_evidence",
                "guard_checks": ["no_live_profile_change", "evidence_required", "no_model_turn_required"],
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research synthesize --kind frontier",
            }
        )
    deterministic_ready = [task for task in active_tasks if is_deterministic_research_task(task)]
    duplicate_stage_tasks = calibration_stage_duplicate_count(root) + compacted_stage_tasks
    if duplicate_stage_tasks:
        review_status = "blocked"
        recommendations.append(
            f"duplicate calibration memory-stage tasks detected={duplicate_stage_tasks}; compacted duplicates and blocked inflated quality."
        )
    if not deterministic_ready:
        recommendations.append("no deterministic ready task remained after review; seeded one lane-contract fallback.")
        fallback_tasks = lane_contract_fallback_tasks(
            root,
            rows,
            int(time.time()),
            reason="Quality review found no deterministic ready work",
        )
        if fallback_tasks:
            seeded_tasks.extend(fallback_tasks)
        else:
            deliberation_report, deliberation_tasks = frontier_agent_deliberation(root, rows, int(time.time()))
            if deliberation_tasks:
                recommendations.append(
                    "lane-contract fallback was exhausted; seeded frontier deliberation with a concrete executable task."
                )
                seeded_tasks.extend(deliberation_tasks)
                deliberation_path = root / "benchmarks" / f"frontier-agent-deliberation-{deliberation_report['timestamp']}.json"
                deliberation_report["seeded"] = len(deliberation_tasks)
                deliberation_path.write_text(json.dumps(deliberation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not recommendations:
        recommendations.append("research quality is acceptable; continue current queue.")
    verdict = "healthy"
    coverage_gap = bool(missing_required_blocks and not has_calibration_route and not durable_sweep_coverage)
    if exhaustion_candidate:
        verdict = "exhaustion-candidate"
    elif coverage_gap or blocked or (repeated_terminal_calibration and not terminal_bottleneck_routed):
        verdict = "needs-repair"
    elif plateau_below_target:
        verdict = "converged-below-target"
    legacy_quality_score = max(0, quality_score)
    scorecard = research_quality_scorecard(
        blocked_rows=len(blocked),
        missing_required_blocks=missing_required_blocks,
        sweep_rows=len(sweep_rows),
        min_sweeps=int(args.min_sweeps),
        repeated_block2=repeated_block2,
        repeated_keep_current=repeated_keep_current,
        plateau_below_target=plateau_below_target,
        exhaustion_candidate=exhaustion_candidate,
        frontier_ready=frontier_ready,
        seeded_tasks=seeded_tasks,
        ready_tasks=active_tasks,
        contaminated_rows=len(contaminated_signals),
        clean_runtime_maps=len(clean_runtime_maps),
        variance=variance,
        artifact_check=artifact_check,
        contract_ok=bool(contract.get("ok")),
        dflash_suppressed=dflash_blocked_or_suppressed,
        repeated_dflash_synthesis=len(repeated_deliberate_dflash),
        duplicate_stage_tasks=duplicate_stage_tasks,
        best_mean=best_mean,
        target_tps=float(args.target_tps),
        server_decode_values=server_decode_values,
        canonical_state=str(canonical_state.get("state", "")),
    )
    if verdict == "needs-repair":
        scorecard = {
            **scorecard,
            "overall": min(float(scorecard["overall"]), 74.0),
            "interpretation": "needs_repair_not_certified",
            "inflation_guard": "needs-repair verdict caps scorecard until blockers are resolved",
        }
    if duplicate_stage_tasks:
        quality_score = min(legacy_quality_score, int(round(float(scorecard["overall"]))))
    else:
        quality_score = max(legacy_quality_score, int(round(float(scorecard["overall"]))))
    if verdict == "needs-repair":
        quality_score = min(quality_score, 74)
    seeded_tasks = filter_seedable_tasks(root, seeded_tasks) if seeded_tasks else []
    seeded = upsert_tasks(root, seeded_tasks) if seeded_tasks else 0
    timestamp = int(time.time())
    artifact = {
        "ok": True,
        "kind": "quality-review",
        "timestamp": timestamp,
        "recent_rows": len(recent),
        "blocked_rows": len(blocked),
        "certification_blocked_rows": len(certification_blocked),
        "sweep_rows": len(sweep_rows),
        "sweep_rows_in_recent": len(sweep_rows_in_recent),
        "block_summary": block_summary,
        "best_block": best_block,
        "best_mean_decode_tps": best_mean,
        "best_max_decode_tps": best_max,
        "mean_clean_wall_decode_tps": mean_float(
            [float(signal["wall_decode_tps"]) for signal in clean_signals if signal.get("wall_decode_tps") is not None]
        ),
        "mean_server_decode_tps": mean_float(server_decode_values),
        "max_server_decode_tps": round(max(server_decode_values), 3) if server_decode_values else None,
        "contaminated_decode_rows": len(contaminated_signals),
        "clean_runtime_overhead_maps": len(clean_runtime_maps),
        "target_tps": float(args.target_tps),
        "quality_score": quality_score,
        "legacy_quality_score": legacy_quality_score,
        "scorecard": scorecard,
        "verdict": verdict,
        "gates": gates,
        "missing_required_blocks": missing_required_blocks,
        "frontier_ready_lanes": frontier_ready,
        "plateau_below_target": plateau_below_target,
        "exhaustion_candidate": exhaustion_candidate,
        "repeated_block2_winner": repeated_block2,
        "repeated_keep_current": repeated_keep_current,
        "variance": variance,
        "measurement_artifact": artifact_check,
        "task_contract": contract,
        "duplicate_stage_tasks": duplicate_stage_tasks,
        "trace_distillation_gradient_blocks": len(trace_distillation_blocks),
        "trace_distillation_repair_route": has_trace_distillation_repair_route,
        "trace_distillation_repair_attempted": trace_distillation_repair_done,
        "compacted_stage_tasks": compacted_stage_tasks,
        "compacted_canary_tasks": compacted_canary_tasks,
        "compacted_terminal_tasks": compacted_terminal_tasks,
        "compacted_direct_canaries": compacted_direct_canaries,
        "compacted_runtime_tasks": compacted_runtime_tasks,
        "repeated_canary_ready_no_stage": len(repeated_canary_ready_no_stage),
        "recommendations": recommendations,
        "seeded_tasks": seeded,
        "canonical_state": canonical_state,
    }
    (root / "state.json").write_text(json.dumps(canonical_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path = root / "benchmarks" / f"quality-review-{timestamp}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "quality-review",
            "finding": "supervisor quality reviewer scored recent autoresearch and fed next-step tasks back into the loop",
            "evidence": artifact,
            "next": "continue_autopilot_loop",
        },
    )
    append_result(
        root,
        run_id=f"quality-review-{timestamp}",
        status=review_status,
        target="autoresearch-quality",
        hypothesis="sidecar review should detect convergence, noise, and next-step tasks without a human review turn",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"recent_rows={len(recent)} blocked={len(blocked)} sweeps={len(sweep_rows)} "
            f"verdict={verdict} score={quality_score} legacy_score={legacy_quality_score} "
            f"scorecard_overall={scorecard['overall']} scorecard_interpretation={scorecard['interpretation']} "
            f"best_block={best_block} "
            f"best_mean_tps={best_mean if best_mean is not None else ''} "
            f"mean_server_tps={artifact['mean_server_decode_tps'] if artifact['mean_server_decode_tps'] is not None else ''} "
            f"contaminated={len(contaminated_signals)} "
            f"repeated_block2={repeated_block2} seeded_tasks={seeded} "
            f"recommendation={recommendations[0]}"
        ),
    )
    suppressed_gepa = suppress_stale_gepa_policy_canaries(root)
    if suppressed_gepa:
        append_result(
            root,
            run_id=f"gepa-suppression-{timestamp}",
            status="keep",
            target="autoresearch-gepa-suppression",
            hypothesis="healthy quality review should suppress stale GEPA canaries instead of letting old trajectory noise run",
            commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
            notes=f"suppressed={suppressed_gepa} reason=healthy_quality_no_fresh_actionable_trigger",
        )
        artifact["suppressed_gepa_canaries"] = suppressed_gepa
    print(json.dumps({"ok": True, "path": str(path), **artifact}, indent=2))
    return 0


def frontier_review(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    variance = variance_analysis(root, recent_rows=args.recent_rows, min_samples=args.min_samples)
    artifact = measurement_artifact_analysis(root, recent_rows=args.recent_rows)
    tasks = read_jsonl(root / "tasks.jsonl")
    ready = [task for task in tasks if task.get("status", "ready") in {"ready", "rework"}]
    exhausted = exhausted_lanes(root)
    journal = read_jsonl(root / "journal.jsonl")
    corpus = read_jsonl(root / "trajectory-corpus.jsonl")
    action_counts: dict[str, int] = {}
    for entry in journal[-args.recent_rows :]:
        action = str(entry.get("action") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    report = {
        "ok": True,
        "kind": "frontier-review",
        "timestamp": int(time.time()),
        "recent_rows": args.recent_rows,
        "ready_tasks": len(ready),
        "ready_lanes": sorted({str(task.get("lane", "")) for task in ready if task.get("lane")}),
        "exhausted_lanes": sorted(exhausted),
        "journal_entries": len(journal),
        "recent_action_counts": action_counts,
        "trajectory_cases": len(corpus),
        "variance": variance,
        "measurement_artifact": artifact,
        "next": (
            "promote only variance-significant improvements; rework metric-up guard-fail nodes; "
            "skip exhausted lanes unless new evidence reopens them"
        ),
    }
    path = root / "benchmarks" / f"frontier-review-{report['timestamp']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "frontier-review",
            "finding": "supervisor reviewed journal, variance, artifacts, exhausted lanes, and trajectory corpus",
            "evidence": report,
            "next": "select_next_task",
        },
    )
    append_result(
        root,
        run_id=f"frontier-review-{report['timestamp']}",
        status="keep",
        target="autoresearch-frontier",
        hypothesis="journal and variance review should keep the autonomous loop focused on high-signal improvements",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"ready={len(ready)} exhausted={','.join(sorted(exhausted))} "
            f"significant_best={variance.get('significant_best')} artifact={artifact.get('artifact_suspected')}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2))
    return 0


def review_council_task(timestamp: int, *, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"review-council-frontier-deliberation-{timestamp}",
        "status": "ready",
        "priority": 101,
        "lane": "frontier-deliberation",
        "task_type": "supervisor",
        "supervisor_action": "frontier-deliberation",
        "target": "tasks.jsonl/results.tsv",
        "hypothesis": (
            "The review council found no ready deterministic task, so the next safe action is to create "
            "one bounded frontier-deliberation contract instead of waiting for human prompting."
        ),
        "metric": "review_council_autonomy",
        "guard_checks": [
            "no_model_load",
            "no_live_profile_change",
            "no_opencode_changes",
            "contract_has_acceptance_and_rollback",
            "rollback_path",
        ],
        "acceptance": "A frontier-agent-deliberation artifact selects exactly one safe next task or records a closed blocker.",
        "rollback": "No source rollback needed; this task only writes deliberation artifacts and queue state.",
        "evidence": evidence,
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research frontier-deliberation --allow-empty",
    }


def score_review_council_artifact(report: dict[str, Any]) -> dict[str, Any]:
    roles = report.get("roles") if isinstance(report.get("roles"), dict) else {}
    gates = report.get("gates") if isinstance(report.get("gates"), dict) else {}
    evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
    canonical = report.get("canonical_state") if isinstance(report.get("canonical_state"), dict) else {}
    measurement = report.get("measurement") if isinstance(report.get("measurement"), dict) else {}
    variance = report.get("variance") if isinstance(report.get("variance"), dict) else {}
    strategist = roles.get("strategist") if isinstance(roles.get("strategist"), dict) else {}
    gatekeeper = roles.get("gatekeeper") if isinstance(roles.get("gatekeeper"), dict) else {}
    prober = roles.get("prober") if isinstance(roles.get("prober"), dict) else {}
    skeptic = roles.get("skeptic") if isinstance(roles.get("skeptic"), dict) else {}
    consultant = roles.get("consultant") if isinstance(roles.get("consultant"), dict) else {}
    seeded_tasks = int(report.get("seeded_tasks") or 0)
    seeded_ids = strategist.get("seeded_tasks") if isinstance(strategist.get("seeded_tasks"), list) else []
    duplicate_seed = seeded_tasks > len({str(item) for item in seeded_ids})
    required_roles = {"prober", "consultant", "skeptic", "strategist", "gatekeeper"}
    role_names = {str(name) for name in roles}
    evidence_paths = [str(value) for value in evidence.values() if value]
    questions = prober.get("questions") if isinstance(prober.get("questions"), list) else []
    falsification_gates = skeptic.get("falsification_gates") if isinstance(skeptic.get("falsification_gates"), dict) else {}
    decision = str(strategist.get("decision", ""))
    next_action = str(strategist.get("next_action", "") or report.get("next", ""))
    allowed_decisions = {"continue", "seed-frontier-deliberation", "observe", "repair"}
    seeded_task_safe = (
        seeded_tasks == 0
        or (
            decision == "seed-frontier-deliberation"
            and all(str(item).startswith("review-council-frontier-deliberation-") for item in seeded_ids)
        )
    )
    mutation_contained = (
        gatekeeper.get("promotion_allowed") is False
        and seeded_task_safe
        and not duplicate_seed
        and int(report.get("seeded_tasks") or 0) <= 1
    )
    safety_gate_names = (
        "canonical_clean",
        "zero_active_noise",
        "no_bad_behavior_rows",
        "task_contract_clean",
        "handoff_clean",
        "autonomy_clean",
        "measurement_clean",
    )
    safety_gates_clear = all(bool(gates.get(name)) for name in safety_gate_names)
    evidence_grounded = (
        bool(evidence_paths)
        and bool(canonical)
        and bool(measurement)
        and bool(variance)
        and bool(report.get("deterministic_ready_tasks") or decision in {"seed-frontier-deliberation", "repair", "observe"})
    )
    artifact_complete = required_roles.issubset(role_names) and bool(questions) and bool(next_action)
    no_noise = (
        bool(gates.get("zero_active_noise"))
        and bool(gates.get("no_bad_behavior_rows"))
        and not report.get("bad_behavior_rows")
    )
    actionable = decision in allowed_decisions and bool(next_action)
    novelty_guard = not (
        decision == "seed-frontier-deliberation"
        and "repeat" in next_action.lower()
        and not evidence.get("source_scout")
    )
    components = {
        "artifact_completeness": 20 if artifact_complete else 0,
        "evidence_grounding": 20 if evidence_grounded else 0,
        "safety_noise": 20 if no_noise and safety_gates_clear else 0,
        "actionability": 15 if actionable else 0,
        "novelty_no_duplication": 10 if novelty_guard and not duplicate_seed else 0,
        "mutation_containment": 15 if mutation_contained else 0,
    }
    total = int(sum(components.values()))
    hard_gate_failures = [
        name
        for name, passed in {
            "artifact_complete": artifact_complete,
            "evidence_grounded": evidence_grounded,
            "safety_gates_clear": safety_gates_clear,
            "zero_noise": no_noise,
            "actionable_decision": actionable,
            "mutation_contained": mutation_contained,
        }.items()
        if not passed
    ]
    return {
        "overall": total,
        "readiness": "frontier-council" if total >= 95 and not hard_gate_failures else "needs-repair",
        "components": components,
        "hard_gate_failures": hard_gate_failures,
        "signals": {
            "roles_present": sorted(role_names),
            "evidence_paths": len(evidence_paths),
            "decision": decision,
            "seeded_tasks": seeded_tasks,
            "duplicate_seed": duplicate_seed,
            "promotion_allowed": bool(gatekeeper.get("promotion_allowed")),
            "safety_gates_clear": safety_gates_clear,
            "failed_safety_gates": [name for name in safety_gate_names if not bool(gates.get(name))],
        },
    }


def review_council_report(root: Path, *, recent_rows: int = 120, target_tps: float = 30.0, seed_next: bool = False) -> dict[str, Any]:
    """Run the deterministic prober/consultant/skeptic/strategist council.

    This is the artifact-backed version of the manual Kristian + Codex loop. It
    does not free-reason over the repo. It reads current certification artifacts,
    asks the same classes of questions every time, and can seed one safe
    frontier-deliberation task when the system is healthy but has no next action.
    """
    ensure_research_state(root)
    timestamp = int(time.time())
    quality = latest_json_artifact(root, "quality-review-*.json")
    frontier = latest_json_artifact(root, "frontier-system-eval-*.json")
    autonomy = latest_json_artifact(root, "frontier-autonomy-score-*.json")
    handoff = latest_json_artifact(root, "implementation-handoff-audit-*.json")
    alive = latest_json_artifact(root, "self-improvement-alive-eval-*.json")
    frontier_review_artifact = latest_json_artifact(root, "frontier-review-*.json")
    hypothesis_rank = latest_json_artifact(root, "hypothesis-rank-*.json")
    causal = latest_json_artifact(root, "causal-review-*.json")
    canonical = canonical_autoresearch_state(root, recent_rows=recent_rows, target_tps=target_tps)
    noise = canonical.get("noise") if isinstance(canonical.get("noise"), dict) else {}
    bad_rows = recent_bad_behavior_rows(root, recent_rows=recent_rows)
    contract = task_contract_report(root)
    measurement = measurement_artifact_analysis(root, recent_rows=recent_rows)
    variance = variance_analysis(root, recent_rows=recent_rows, min_samples=3)
    tasks = read_jsonl(root / "tasks.jsonl")
    ready = [task for task in tasks if task.get("status", "ready") in {"ready", "rework"}]
    deterministic_ready = [task for task in ready if is_deterministic_research_task(task)]
    deterministic_ready_ids = [str(task.get("id", "")) for task in deterministic_ready[:8]]
    exhausted = list(canonical.get("exhausted_lanes", []))
    decode_mean = canonical.get("decode_mean_tps")
    if decode_mean is None:
        decode_mean = latest_decode_mean(root, recent_rows=recent_rows)
    quality_score = quality.get("quality_score")
    scorecard = quality.get("scorecard") if isinstance(quality.get("scorecard"), dict) else {}
    scorecard_overall = scorecard.get("overall")
    gates = {
        "quality_artifact_present": bool(quality),
        "frontier_artifact_present": bool(frontier),
        "handoff_artifact_present": bool(handoff),
        "autonomy_artifact_present": bool(autonomy),
        "canonical_clean": bool(canonical.get("clean")),
        "zero_active_noise": all(int(noise.get(key, 0) or 0) == 0 for key in noise),
        "no_bad_behavior_rows": not bad_rows,
        "task_contract_clean": bool(contract.get("ok", True)),
        "handoff_clean": bool(handoff.get("ok")) and float(handoff.get("score") or 0) >= 90.0,
        "autonomy_clean": not autonomy.get("hard_gate_failures"),
        "measurement_clean": not bool(measurement.get("artifact_suspected")),
    }
    stable_enough = all(
        gates[key]
        for key in (
            "canonical_clean",
            "zero_active_noise",
            "no_bad_behavior_rows",
            "task_contract_clean",
            "handoff_clean",
            "measurement_clean",
        )
    )
    prober_questions = [
        {
            "question": "Is the current work moving the primary metric, decode TPS, rather than producing plausible activity?",
            "evidence": {
                "decode_mean_tps": decode_mean,
                "target_tps": target_tps,
                "significant_best": variance.get("significant_best"),
            },
            "answer": "not yet proven" if decode_mean is None or float(decode_mean) < target_tps else "target reached",
        },
        {
            "question": "What would a human prober challenge before letting this run unattended?",
            "evidence": {
                "ready_tasks": deterministic_ready_ids,
                "exhausted_lanes": exhausted,
                "quality_score": quality_score,
                "scorecard_overall": scorecard_overall,
            },
            "answer": "run the deterministic prerequisite before inventing another lane"
            if deterministic_ready_ids
            else "create a bounded frontier-deliberation task if gates are clean",
        },
    ]
    consultant_diagnosis = {
        "role": "consultant",
        "full_picture": {
            "canonical_state": canonical.get("state", ""),
            "ready_lanes": canonical.get("ready_lanes", []),
            "breakthrough_lanes": canonical.get("breakthrough_lanes", []),
            "exhausted_lanes": exhausted,
            "decode_mean_tps": decode_mean,
        },
        "likely_bottleneck": (
            "drafter-fit or accepted-token yield"
            if decode_mean is not None and float(decode_mean) < target_tps
            else "verify no UX or stability regression before further speed work"
        ),
        "next_leverage": (
            deterministic_ready_ids[0]
            if deterministic_ready_ids
            else "frontier-deliberation"
            if stable_enough
            else "repair quality/stability gates first"
        ),
    }
    skeptic = {
        "role": "skeptic",
        "must_disprove": [
            "the apparent speed result is just measurement noise",
            "the next candidate reopens an exhausted lane without new evidence",
            "a patch improves benchmark TPS while breaking TUI/tool/reasoning behavior",
        ],
        "falsification_gates": {
            "variance_significant_best": bool(variance.get("significant_best")),
            "measurement_clean": gates["measurement_clean"],
            "zero_active_noise": gates["zero_active_noise"],
            "no_bad_behavior_rows": gates["no_bad_behavior_rows"],
            "handoff_clean": gates["handoff_clean"],
        },
    }
    if not stable_enough:
        decision = "repair"
        next_action = "run autonomous repair before any new research or patch promotion"
    elif deterministic_ready_ids:
        decision = "continue"
        next_action = f"run deterministic task {deterministic_ready_ids[0]}"
    elif seed_next:
        decision = "seed-frontier-deliberation"
        next_action = "seed one bounded frontier-deliberation task"
    else:
        decision = "observe"
        next_action = "record council artifact; next autopilot cycle may synthesize bounded work"
    gatekeeper = {
        "role": "gatekeeper",
        "decision": decision,
        "promotion_allowed": False,
        "reason": "review council is advisory/routing only; source promotion remains patch-executor/Crabbox gated",
        "hard_gates": gates,
    }
    evidence = {
        "quality": quality.get("_artifact_path", ""),
        "frontier": frontier.get("_artifact_path", ""),
        "autonomy": autonomy.get("_artifact_path", ""),
        "handoff": handoff.get("_artifact_path", ""),
        "alive": alive.get("_artifact_path", ""),
        "frontier_review": frontier_review_artifact.get("_artifact_path", ""),
        "hypothesis_rank": hypothesis_rank.get("_artifact_path", ""),
        "causal_review": causal.get("_artifact_path", ""),
    }
    seeded_tasks = 0
    seeded_task_ids: list[str] = []
    if decision == "seed-frontier-deliberation":
        task = review_council_task(
            timestamp,
            evidence={
                "canonical_state": canonical.get("state", ""),
                "decode_mean_tps": decode_mean,
                "exhausted_lanes": exhausted,
                "council_decision": decision,
            },
        )
        if not active_task_has_prefix(root, "review-council-frontier-deliberation-"):
            seeded_tasks = upsert_tasks(root, filter_seedable_tasks(root, [task]))
            if seeded_tasks:
                seeded_task_ids = [str(task["id"])]
    report = {
        "ok": stable_enough,
        "kind": "review-council",
        "timestamp": timestamp,
        "recent_rows": recent_rows,
        "target_tps": target_tps,
        "roles": {
            "prober": {
                "role": "prober",
                "mission": "ask whether evidence really advances the goal and expose hidden assumptions",
                "questions": prober_questions,
            },
            "consultant": consultant_diagnosis,
            "skeptic": skeptic,
            "strategist": {
                "role": "strategist",
                "decision": decision,
                "next_action": next_action,
                "seeded_tasks": seeded_task_ids,
            },
            "gatekeeper": gatekeeper,
        },
        "gates": gates,
        "canonical_state": canonical,
        "task_contract": {
            "ok": contract.get("ok", True),
            "ready_tasks": contract.get("ready_tasks"),
            "issue_count": contract.get("issue_count", 0),
        },
        "measurement": measurement,
        "variance": variance,
        "bad_behavior_rows": [row.get("run_id", "") for row in bad_rows[:8]],
        "deterministic_ready_tasks": deterministic_ready_ids,
        "seeded_tasks": seeded_tasks,
        "evidence": evidence,
        "next": next_action,
    }
    scorecard = score_review_council_artifact(report)
    report["scorecard"] = scorecard
    report["ok"] = stable_enough and scorecard["overall"] >= 95 and not scorecard["hard_gate_failures"]
    return report


def review_council(args: argparse.Namespace) -> int:
    root = workspace_root()
    report = review_council_report(
        root,
        recent_rows=args.recent_rows,
        target_tps=args.target_tps,
        seed_next=args.seed_next,
    )
    path = root / "benchmarks" / f"review-council-{report['timestamp']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "review-council",
            "finding": "deterministic prober/consultant/skeptic/strategist council reviewed autoresearch health and next action",
            "evidence": report,
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"review-council-{report['timestamp']}",
        status="keep" if report["ok"] else "blocked",
        target="autoresearch-review-council",
        hypothesis="an hourly deterministic council should replace manual probing and consultant review without adding noisy agent chatter",
        commit=current_commit(repo_root()),
        notes=(
            f"decision={report['roles']['strategist']['decision']} ok={report['ok']} "
            f"seeded_tasks={report['seeded_tasks']} next={str(report['next']).replace(chr(9), ' ').replace(chr(10), ' ')}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2))
    return 0 if report["ok"] or args.allow_fail else 2


def score_frontier_system(root: Path, *, recent_rows: int = 120) -> dict[str, Any]:
    rows = result_rows(root)
    recent = rows[-max(1, recent_rows) :]
    checkpoint_index = next(
        (
            index
            for index in range(len(recent) - 1, -1, -1)
            if recent[index].get("run_id", "").startswith("frontier-system-eval-")
        ),
        -1,
    )
    active_recent = recent[checkpoint_index + 1 :] if checkpoint_index >= 0 else recent
    tasks = read_jsonl(root / "tasks.jsonl")
    ready = [task for task in tasks if task.get("status", "ready") in {"ready", "rework"}]
    replay = replay_checks(root)
    canonical_state = canonical_autoresearch_state(root, recent_rows=recent_rows)
    canonical_name = str(canonical_state.get("state", ""))

    blocked = list(canonical_state.get("unresolved_blocked_rows", []))
    historical_blocked = unresolved_actionable_blocked_rows(recent)
    memory_blocks = [row for row in blocked if is_memory_safety_blocked_row(row)]
    historical_memory_blocks = [
        row for row in historical_blocked if is_memory_safety_blocked_row(row)
    ]
    synthesis_rows = [row for row in active_recent if row.get("target") == "synthesis"]
    historical_synthesis_rows = [row for row in recent if row.get("target") == "synthesis"]
    empty_synthesis = [
        row
        for row in synthesis_rows
        if "seeded_tasks=0" in row.get("notes", "")
        and "deliberate_actions=deliberate-" not in row.get("notes", "")
        and "terminal_no_work=True" not in row.get("notes", "")
    ]
    historical_empty_synthesis = [
        row
        for row in historical_synthesis_rows
        if "seeded_tasks=0" in row.get("notes", "")
        and "deliberate_actions=deliberate-" not in row.get("notes", "")
        and "terminal_no_work=True" not in row.get("notes", "")
    ]
    bridge_zero = recent_empty_bridge_rows(root, active_recent, recent_rows=len(active_recent) or 1)
    clean_runtime_maps = recent_clean_runtime_overhead_maps(root, active_recent, recent_rows=len(active_recent) or 1)
    historical_bridge_zero = recent_empty_bridge_rows(root, recent, recent_rows=recent_rows)
    deliberate_ready = [task for task in ready if str(task.get("id", "")).startswith("deliberate-")]
    deterministic_ready = [task for task in ready if is_deterministic_research_task(task)]
    duplicate_stage_tasks = calibration_stage_duplicate_count(root)
    bridge_ready = [task for task in deterministic_ready if is_implementation_bridge_task(task)]
    bridge_only_ready = bool(deterministic_ready) and len(bridge_ready) == len(deterministic_ready)
    if canonical_name in {"prerequisite_needed", "breakthrough_lane_active", "plateau_detected"}:
        bridge_zero = []
        bridge_only_ready = False
    patch_rows = [row for row in active_recent if row.get("run_id", "").startswith("patch-executor-")]
    handoff_audit_rows = [
        row for row in active_recent if row.get("run_id", "").startswith("implementation-handoff-audit-")
    ]
    historical_handoff_audit_rows = [
        row for row in recent if row.get("run_id", "").startswith("implementation-handoff-audit-")
    ]
    environment_snapshot_rows = [
        row for row in active_recent if row.get("run_id", "").startswith("environment-snapshot-")
    ]
    historical_environment_snapshot_rows = [
        row for row in recent if row.get("run_id", "").startswith("environment-snapshot-")
    ]
    evaluator_integrity_rows = [
        row for row in active_recent if row.get("run_id", "").startswith("evaluator-integrity-")
    ]
    historical_evaluator_integrity_rows = [
        row for row in recent if row.get("run_id", "").startswith("evaluator-integrity-")
    ]
    plateau_pivot_rows = [
        row for row in active_recent if row.get("run_id", "").startswith("plateau-pivot-")
    ]
    historical_plateau_pivot_rows = [
        row for row in recent if row.get("run_id", "").startswith("plateau-pivot-")
    ]
    calibration_memory_report_rows = [
        row for row in active_recent if row.get("run_id", "").startswith("calibration-memory-report-")
    ]
    historical_calibration_memory_report_rows = [
        row for row in recent if row.get("run_id", "").startswith("calibration-memory-report-")
    ]
    quality_rows = [row for row in active_recent if row.get("run_id", "").startswith("quality-review-")]
    historical_quality_rows = [row for row in recent if row.get("run_id", "").startswith("quality-review-")]
    quality_source_rows = quality_rows or historical_quality_rows
    handoff_source_rows = handoff_audit_rows or historical_handoff_audit_rows
    environment_source_rows = environment_snapshot_rows or historical_environment_snapshot_rows
    evaluator_source_rows = evaluator_integrity_rows or historical_evaluator_integrity_rows
    plateau_source_rows = plateau_pivot_rows or historical_plateau_pivot_rows
    latest_quality_notes = quality_source_rows[-1].get("notes", "") if quality_source_rows else ""
    latest_quality_fields = parse_note_fields(latest_quality_notes)
    latest_quality_score = None
    try:
        latest_quality_score = float(latest_quality_fields["score"]) if latest_quality_fields.get("score") else None
    except ValueError:
        latest_quality_score = None
    latest_scorecard_overall = None
    try:
        latest_scorecard_overall = (
            float(latest_quality_fields["scorecard_overall"])
            if latest_quality_fields.get("scorecard_overall")
            else None
        )
    except ValueError:
        latest_scorecard_overall = None
    latest_quality_interpretation = latest_quality_fields.get("scorecard_interpretation", "")
    latest_quality_verdict = latest_quality_fields.get("verdict", "")
    latest_quality_artifact = latest_json_artifact(root, "quality-review-*.json")
    if latest_quality_artifact:
        artifact_quality_score = latest_quality_artifact.get("quality_score")
        if isinstance(artifact_quality_score, (int, float)):
            latest_quality_score = float(artifact_quality_score)
        artifact_scorecard = latest_quality_artifact.get("scorecard")
        if isinstance(artifact_scorecard, dict):
            artifact_scorecard_overall = artifact_scorecard.get("overall")
            if isinstance(artifact_scorecard_overall, (int, float)):
                latest_scorecard_overall = float(artifact_scorecard_overall)
            latest_quality_interpretation = str(
                artifact_scorecard.get("interpretation", latest_quality_interpretation)
            )
        latest_quality_verdict = str(latest_quality_artifact.get("verdict", latest_quality_verdict))
    quality_route_high = (
        (latest_scorecard_overall is not None and latest_scorecard_overall >= 85.0)
        or (latest_quality_score is not None and latest_quality_score >= 85.0)
    ) and latest_quality_interpretation == "high_quality_exhaustion_or_prerequisite_route" and latest_quality_verdict != "needs-repair"
    terminal_calibration_plateau = bool(calibration_memory_report_rows or historical_calibration_memory_report_rows)
    decode_mean = latest_decode_mean(root, recent_rows=recent_rows)
    contract = task_contract_report(root)
    artifact = measurement_artifact_analysis(root, recent_rows=recent_rows)
    burn_in = latest_json_artifact(root, "stability-burn-in-*.json")
    alive = self_improvement_alive_report(root, recent_rows=recent_rows)
    burn_in_gates = burn_in.get("gates") if isinstance(burn_in.get("gates"), dict) else {}
    burn_in_ok = bool(burn_in.get("ok")) and bool(burn_in_gates) and all(
        bool(value) for value in burn_in_gates.values()
    )
    frontier_lanes = {
        str(task.get("lane", ""))
        for task in ready
        if str(task.get("lane", "")) in {
            "runtime-overhead",
            "drafter-alignment",
            "frontier-dflash",
            "frontier-expansion",
        }
    }

    scores: dict[str, float] = {
        "karpathy_core_loop": 9.1,
        "crash_memory_safety": 9.2,
        "research_quality": 8.6,
        "implementation_handoff": 8.5,
        "self_improvement": 8.8,
        "modularity": 9.2,
    }
    gaps: list[str] = []
    strengths: list[str] = []

    if not replay.get("ok"):
        scores["crash_memory_safety"] -= 2.5
        gaps.append("replay guards are failing")
    else:
        strengths.append("replay guards pass")
    replay_integrity = replay.get("evaluator_integrity") if isinstance(replay, dict) else {}
    if isinstance(replay_integrity, dict) and replay_integrity.get("ok") is False:
        scores["karpathy_core_loop"] -= 1.4
        scores["research_quality"] -= 1.0
        gaps.append("immutable evaluator integrity failed")
    if environment_source_rows:
        scores["crash_memory_safety"] += 0.2
        scores["modularity"] += 0.1
        strengths.append("environment snapshots record run context and evaluator hashes")
    if evaluator_source_rows:
        scores["karpathy_core_loop"] += 0.2
        scores["crash_memory_safety"] += 0.2
        strengths.append("frozen evaluator integrity is checked during review")
    if plateau_source_rows:
        latest_plateau = parse_note_fields(plateau_source_rows[-1].get("notes", ""))
        if latest_plateau.get("state") == "pivot":
            scores["karpathy_core_loop"] += 0.3
            scores["research_quality"] += 0.3
            scores["self_improvement"] += 0.2
            strengths.append("plateau-pivot state machine routes settled sweeps to higher-upside lanes")
    if memory_blocks:
        scores["crash_memory_safety"] -= min(1.2, len(memory_blocks) * 0.25)
        gaps.append(f"recent memory/Metal blockers still present={len(memory_blocks)}")
    if empty_synthesis:
        scores["karpathy_core_loop"] -= min(2.0, len(empty_synthesis) * 0.35)
        scores["research_quality"] -= min(1.2, len(empty_synthesis) * 0.2)
        gaps.append(f"recent empty synthesis rows remain in the evaluation window={len(empty_synthesis)}")
    if deliberate_ready:
        scores["karpathy_core_loop"] += 0.4
        scores["research_quality"] += 0.3
        strengths.append("deliberate next action is queued instead of generic research churn")
    if not deterministic_ready and terminal_calibration_plateau:
        scores["karpathy_core_loop"] += 0.1
        scores["research_quality"] += 0.2
        scores["self_improvement"] += 0.1
        strengths.append("calibration plateau ended in a deterministic no-model root-cause report")
    elif not deterministic_ready:
        scores["karpathy_core_loop"] -= 1.0
        gaps.append("no deterministic ready task is queued")
    if canonical_name in {"prerequisite_needed", "breakthrough_lane_active", "plateau_detected", "frontier_healthy"}:
        scores["karpathy_core_loop"] += 0.25
        scores["research_quality"] += 0.45
        scores["implementation_handoff"] += 0.25
        scores["self_improvement"] += 0.2
        strengths.append(f"canonical state is {canonical_name}; routed work is scored once instead of inferred from noisy rows")
    if bridge_only_ready and bridge_zero:
        scores["karpathy_core_loop"] -= 1.2
        scores["implementation_handoff"] -= 1.0
        scores["research_quality"] -= 0.6
        gaps.append("bridge-only deterministic ready task after an empty implementation bridge")
    if duplicate_stage_tasks:
        scores["karpathy_core_loop"] -= 2.4
        scores["research_quality"] -= 2.5
        scores["implementation_handoff"] -= 1.7
        scores["self_improvement"] -= 1.4
        gaps.append(f"duplicate calibration memory-stage tasks remain ready={duplicate_stage_tasks}")
    if bridge_zero:
        scores["implementation_handoff"] -= min(1.4, len(bridge_zero) * 0.25)
        gaps.append(f"recent implementation bridge rows had no deterministic task={len(bridge_zero)}")
    if len(clean_runtime_maps) >= 2 and not quality_route_high:
        scores["karpathy_core_loop"] -= min(1.6, len(clean_runtime_maps) * 0.25)
        scores["research_quality"] -= min(1.4, len(clean_runtime_maps) * 0.25)
        gaps.append(f"repeated clean runtime-overhead maps should route to drafter/DFlash={len(clean_runtime_maps)}")
    elif len(clean_runtime_maps) >= 2:
        scores["research_quality"] += 0.2
        strengths.append("quality scorecard recognized repeated runtime maps and routed to prerequisite/frontier work")
    if patch_rows:
        scores["implementation_handoff"] += 0.4
        strengths.append("patch executor has recent canary evidence")
    if handoff_source_rows:
        latest_handoff = parse_note_fields(handoff_source_rows[-1].get("notes", ""))
        audit_ok = latest_handoff.get("ok") == "True" or latest_handoff.get("ok") == "true"
        handoff_score = latest_handoff.get("score")
        if audit_ok:
            scores["implementation_handoff"] += 0.7
            strengths.append("implementation handoff audit passed with canary and rollback gates")
            try:
                handoff_score_value = float(handoff_score) if handoff_score else 0.0
            except ValueError:
                handoff_score_value = 0.0
            if handoff_score_value >= 95.0 and not bridge_zero and contract.get("ok"):
                scores["implementation_handoff"] += 0.3
                strengths.append("implementation handoff has certification-grade audit evidence")
            if handoff_score_value >= 100.0 and not bridge_zero and contract.get("ok"):
                scores["implementation_handoff"] += 0.1
                strengths.append("implementation handoff audit is perfect with clean task contracts")
        else:
            scores["implementation_handoff"] -= 0.8
            gaps.append(f"implementation handoff audit needs repair score={handoff_score or 'unknown'}")
    if latest_quality_score is not None:
        scores["research_quality"] += max(-1.2, min(0.6, (latest_quality_score - 75.0) / 100.0))
    if latest_scorecard_overall is not None:
        scores["research_quality"] += max(-0.8, min(0.8, (latest_scorecard_overall - 80.0) / 100.0))
    if quality_route_high:
        scores["research_quality"] += 0.2
        strengths.append("quality scorecard shows evidence-backed routing rather than research churn")
        if latest_quality_verdict == "healthy" and not duplicate_stage_tasks and contract.get("ok"):
            scores["karpathy_core_loop"] += 0.1
            if (
                latest_quality_score is not None
                and latest_quality_score >= 95.0
                and latest_scorecard_overall is not None
                and latest_scorecard_overall >= 95.0
            ):
                scores["research_quality"] += 0.35
                strengths.append("quality review is certification-grade and stable across the latest checkpoint")
            scores["self_improvement"] += 0.2
            scores["modularity"] += 0.1
            strengths.append("healthy quality review is backed by clean contracts and no queue duplication")
    elif latest_quality_verdict == "needs-repair":
        scores["research_quality"] -= 1.5
        scores["karpathy_core_loop"] -= 0.7
        gaps.append("latest quality review verdict is needs-repair")
    if not contract.get("ok"):
        scores["implementation_handoff"] -= 1.0
        gaps.append("ready task contract blockers exist")
    if artifact.get("artifact_suspected") and not quality_route_high:
        scores["research_quality"] -= 1.2
        gaps.append("measurement artifact suspected; server and wall-clock decode must stay separated")
    elif artifact.get("artifact_suspected"):
        scores["research_quality"] += 0.2
        strengths.append("measurement artifact is treated as routed evidence rather than a repeated research failure")
    if frontier_lanes:
        scores["self_improvement"] += 0.3
        strengths.append("frontier lanes remain available: " + ",".join(sorted(frontier_lanes)))
    if any(row.get("run_id", "").startswith("gepa-policy-promotion-") for row in recent):
        scores["self_improvement"] += 0.2
        strengths.append("GEPA reviewer policy path is active")
    alive_score = float(alive.get("total_score") or 0.0)
    if alive_score >= 95.0 and alive.get("ok"):
        scores["self_improvement"] += 0.7
        strengths.append("self-improvement alive eval proves observe/diagnose/route/evolve/contain loop")
    elif alive_score >= 80.0:
        scores["self_improvement"] += 0.2
        gaps.append(f"self-improvement alive eval is operational but not frontier score={alive_score}")
    else:
        scores["self_improvement"] -= 1.0
        gaps.append(f"self-improvement alive eval below operational floor score={alive_score}")
    if burn_in_ok:
        scores["karpathy_core_loop"] += 0.4
        scores["crash_memory_safety"] += 0.4
        scores["research_quality"] += 0.3
        scores["implementation_handoff"] += 0.3
        scores["self_improvement"] += 0.3
        scores["modularity"] += 0.6
        strengths.append("aggressive stability burn-in passed all deterministic gates")
    elif burn_in:
        scores["karpathy_core_loop"] -= 0.2
        scores["crash_memory_safety"] -= 0.2
        gaps.append("latest stability burn-in did not pass every deterministic gate")
    if decode_mean is not None and decode_mean < 20:
        gaps.append(f"decode still below practical floor: {decode_mean} tok/s")
        if not deterministic_ready:
            scores["karpathy_core_loop"] -= 0.4
            scores["implementation_handoff"] -= 0.8
            scores["self_improvement"] -= 0.3
            gaps.append("no deterministic ready task while decode remains below target")

    score_cap = 10.0 if burn_in_ok else 9.8
    scores = {key: round(max(0.0, min(value, score_cap)), 2) for key, value in scores.items()}
    overall = round(sum(scores.values()) / len(scores), 2)
    blocking_gap_terms = (
        "contract",
        "measurement artifact",
        "repeated clean runtime-overhead",
        "bridge-only",
        "duplicate calibration memory-stage",
    )
    has_blocking_gap = any(term in gap for term in blocking_gap_terms for gap in gaps)
    frontier_certified = (
        overall >= 9.5
        and not has_blocking_gap
        and scores["crash_memory_safety"] >= 9.5
        and scores["research_quality"] >= 9.4
        and scores["implementation_handoff"] >= 9.5
        and scores["self_improvement"] >= 9.5
        and alive_score >= 95.0
        and alive.get("ok")
        and latest_quality_score is not None
        and latest_quality_score >= 95.0
        and latest_scorecard_overall is not None
        and latest_scorecard_overall >= 95.0
        and latest_quality_verdict == "healthy"
        and contract.get("ok")
        and (not burn_in or burn_in_ok)
    )
    readiness = (
        "frontier"
        if frontier_certified
        else "frontier-candidate"
        if overall >= 9.0 and not has_blocking_gap
        else "needs-targeted-work"
    )
    return {
        "ok": True,
        "kind": "frontier-system-eval",
        "timestamp": int(time.time()),
        "overall": overall,
        "readiness": readiness,
        "frontier_certified": frontier_certified,
        "frontier_requirements": {
            "overall_at_least_9_5": overall >= 9.5,
            "no_blocking_gaps": not has_blocking_gap,
            "safety_at_least_9_5": scores["crash_memory_safety"] >= 9.5,
            "research_quality_at_least_9_4": scores["research_quality"] >= 9.4,
            "implementation_handoff_at_least_9_5": scores["implementation_handoff"] >= 9.5,
            "self_improvement_at_least_9_5": scores["self_improvement"] >= 9.5,
            "alive_eval_at_least_95": alive_score >= 95.0 and bool(alive.get("ok")),
            "latest_quality_score_at_least_95": latest_quality_score is not None and latest_quality_score >= 95.0,
            "latest_quality_scorecard_at_least_95": latest_scorecard_overall is not None and latest_scorecard_overall >= 95.0,
            "latest_quality_healthy": latest_quality_verdict == "healthy",
            "task_contract_clean": bool(contract.get("ok")),
            "stability_burn_in_clean": burn_in_ok,
        },
        "scores": scores,
        "decode_mean_tps": decode_mean,
        "active_window_rows": len(active_recent),
        "ready_tasks": len(ready),
        "deterministic_ready_tasks": [str(task.get("id", "")) for task in deterministic_ready[:8]],
        "deliberate_ready_tasks": [str(task.get("id", "")) for task in deliberate_ready[:8]],
        "recent_empty_synthesis_rows": len(empty_synthesis),
        "recent_bridge_zero_rows": len(bridge_zero),
        "recent_handoff_audit_rows": len(handoff_audit_rows),
        "historical_handoff_audit_rows": len(historical_handoff_audit_rows),
        "recent_quality_review_rows": len(quality_rows),
        "recent_environment_snapshots": len(environment_snapshot_rows),
        "recent_evaluator_integrity_rows": len(evaluator_integrity_rows),
        "recent_plateau_pivot_rows": len(plateau_pivot_rows),
        "recent_calibration_memory_reports": len(calibration_memory_report_rows),
        "recent_clean_runtime_overhead_maps": len(clean_runtime_maps),
        "recent_memory_blocks": len(memory_blocks),
        "duplicate_stage_tasks": duplicate_stage_tasks,
        "canonical_state": canonical_state,
        "historical_debt": {
            "window_rows": len(recent),
            "empty_synthesis_rows": len(historical_empty_synthesis),
            "bridge_zero_rows": len(historical_bridge_zero),
            "memory_blocks": len(historical_memory_blocks),
            "calibration_memory_reports": len(historical_calibration_memory_report_rows),
        },
        "latest_quality_score": latest_quality_score,
        "latest_quality_scorecard_overall": latest_scorecard_overall,
        "latest_quality_interpretation": latest_quality_interpretation,
        "latest_quality_verdict": latest_quality_verdict,
        "latest_quality_artifact": latest_quality_artifact.get("_artifact_path", ""),
        "task_contract": contract,
        "measurement_artifact": artifact,
        "stability_burn_in": burn_in,
        "self_improvement_alive": alive,
        "strengths": strengths,
        "gaps": gaps,
        "next": (
            "run the queued deliberate supervisor task, then rerun frontier-eval"
            if deliberate_ready
            else "calibration plateau is summarized; wait for a canary patch or new prerequisites"
            if terminal_calibration_plateau and not deterministic_ready
            else "seed one deliberate task with synthesize --kind frontier or add a canary patch task"
        ),
    }


def frontier_repair_tasks(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Seed one deterministic repair lane when the frontier score is below target."""
    timestamp = int(time.time() * 1000)
    root = workspace_root()
    ensure_lane_contracts(root)
    rows = result_rows(root)
    tasks: list[dict[str, Any]] = []
    gaps = [str(gap) for gap in report.get("gaps", [])]

    def has_gap(fragment: str) -> bool:
        return any(fragment in gap for gap in gaps)

    if has_gap("measurement artifact"):
        tasks.append(
            {
                "id": f"frontier-repair-measurement-artifact-{timestamp}",
                "status": "ready",
                "priority": 98,
                "lane": "runtime-overhead",
                "task_type": "supervisor",
                "supervisor_action": "runtime-overhead-map",
                "target": "openclaw/openclaw-jang-vlm-server.py",
                "hypothesis": "Frontier eval found decode measurement contamination, so separate backend server tok/s from wall-clock proxy/tool overhead before any promotion.",
                "metric": "server_wall_decode_gap",
                "guard_checks": ["no_model_turn_required", "no_live_profile_change", "no_opencode_changes"],
                "acceptance": "A runtime-overhead map identifies the contaminated boundary or records that the measurement path is clean.",
                "rollback": "No runtime rollback needed; this is a read-only supervisor artifact.",
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
            }
        )
    elif (
        has_gap("bridge-only")
        or has_gap("empty synthesis")
        or has_gap("no deterministic ready task")
        or has_gap("implementation bridge")
        or has_gap("implementation handoff audit needs repair")
    ):
        if recent_empty_bridge_rows(root, rows, recent_rows=120) or has_gap("bridge-only"):
            tasks.extend(concrete_handoff_prerequisite_tasks(root, rows, timestamp))
            if not tasks:
                tasks.extend(
                    lane_contract_fallback_tasks(
                        root,
                        rows,
                        timestamp,
                        reason="Frontier eval found an empty implementation bridge but concrete handoff prerequisites are exhausted",
                    )
                )
        elif has_gap("no deterministic ready task") or has_gap("empty synthesis"):
            tasks.extend(
                lane_contract_fallback_tasks(
                    root,
                    rows,
                    timestamp,
                    reason="Frontier eval found no deterministic ready task",
                    )
                )
        elif has_gap("implementation bridge"):
            tasks.append(
                {
                    "id": f"frontier-repair-implementation-bridge-{timestamp}",
                    "status": "ready",
                    "priority": 97,
                    "lane": "implementation-gate",
                    "task_type": "supervisor",
                    "supervisor_action": "implementation-bridge",
                    "target": "tasks.jsonl",
                    "hypothesis": "Frontier eval found weak handoff, so convert the best current finding into a deterministic supervisor task instead of another generic research turn.",
                    "metric": "decode_tps_delta",
                    "guard_checks": ["tests_pass", "no_opencode_changes", "memory_gate", "rollback_path"],
                    "acceptance": "The bridge records at least one ready deterministic task with a valid contract.",
                    "rollback": "No source rollback needed; the bridge only changes the autoresearch queue.",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research synthesize --kind frontier",
                }
            )
        if not tasks:
            tasks.append(
                {
                    "id": f"frontier-repair-exhaustion-mtp-report-{timestamp}",
                    "status": "ready",
                    "priority": 97,
                    "lane": "exhaustion-report",
                    "task_type": "supervisor",
                    "supervisor_action": "mtp-report",
                    "target": "openclaw-model-proxy.log",
                    "hypothesis": "Frontier eval found weak implementation handoff after the active speed lanes exhausted, so capture one bounded MTP/acceptance report before another handoff audit.",
                    "metric": "mean_accept",
                    "guard_checks": ["no_model_turn_required", "no_opencode_changes", "one_narrow_tool"],
                    "acceptance": "The report records recent server tok/s and MTP acceptance evidence, or explicitly states that logs lack enough samples.",
                    "rollback": "No runtime rollback needed; this is a read-only supervisor artifact.",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research mtp-report --lines 240",
                    "lines": 240,
                }
            )
    elif has_gap("memory/Metal"):
        tasks.append(
            {
                "id": f"frontier-repair-memory-replay-{timestamp}",
                "status": "ready",
                "priority": 96,
                "lane": "safety",
                "task_type": "supervisor",
                "supervisor_action": "focused-test",
                "target": "openclaw/test-speed-research-autopilot.py",
                "hypothesis": "Frontier eval found recent memory/Metal blockers, so rerun the autopilot guard test before another overnight loop.",
                "metric": "memory_guard_replay",
                "guard_checks": ["no_model_load", "no_opencode_changes", "bounded_test"],
                "acceptance": "The autopilot focused test passes, including memory gate and recovery path assertions.",
                "rollback": "No source rollback needed for a focused replay test.",
                "next_action": "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research-autopilot.py",
            }
        )
    return tasks


def frontier_eval(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    ensure_lane_contracts(root)
    compact_duplicate_calibration_stage_tasks(root)
    report = score_frontier_system(root, recent_rows=args.recent_rows)
    seeded_tasks = 0
    needs_repair = report["overall"] < args.min_score or any(
        "no deterministic ready task" in str(gap) for gap in report.get("gaps", [])
    )
    if needs_repair:
        repairs = [
            task
            for task in frontier_repair_tasks(report)
            if not active_task_has_prefix(root, str(task.get("id", "")).rsplit("-", 1)[0] + "-")
        ]
        seeded_tasks = upsert_tasks(root, repairs) if repairs else 0
        if seeded_tasks:
            # Score the post-repair queue state, not the stale pre-repair snapshot.
            report = score_frontier_system(root, recent_rows=args.recent_rows)
        report["seeded_repair_tasks"] = seeded_tasks
        if seeded_tasks:
            report["next"] = "run seeded frontier repair task, then rerun frontier-eval"
    else:
        report["seeded_repair_tasks"] = 0
    path = root / "benchmarks" / f"frontier-system-eval-{report['timestamp']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "frontier-system-eval",
            "finding": "supervisor scored the autoresearch harness against frontier-level reliability and self-improvement criteria",
            "evidence": report,
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"frontier-system-eval-{report['timestamp']}",
        status="keep" if report["overall"] >= args.min_score else "blocked",
        target="autoresearch-frontier-eval",
        hypothesis="OpenClaw autoresearch should be scored against Karpathy-style loop quality, safety, and implementation handoff",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"overall={report['overall']} readiness={report['readiness']} "
            f"core={report['scores']['karpathy_core_loop']} safety={report['scores']['crash_memory_safety']} "
            f"quality={report['scores']['research_quality']} handoff={report['scores']['implementation_handoff']} "
            f"self_improvement={report['scores']['self_improvement']} gaps={len(report['gaps'])} "
            f"seeded_repair_tasks={seeded_tasks}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2))
    return 0 if report["overall"] >= args.min_score or args.allow_fail else 2


def stability_burn_in_report(
    root: Path,
    *,
    recent_rows: int = 120,
    min_free_mb: int = 8192,
    max_compressor_mb: int = 8192,
    max_swap_mb: int = 8192,
) -> dict[str, Any]:
    ensure_research_state(root)
    replay = replay_checks(root)
    canonical = canonical_autoresearch_state(root, recent_rows=recent_rows)
    quality = latest_json_artifact(root, "quality-review-*.json")
    handoff = latest_json_artifact(root, "implementation-handoff-audit-*.json")
    contract = task_contract_report(root)
    memory = memory_snapshot()
    ready = [
        task
        for task in read_jsonl(root / "tasks.jsonl")
        if task.get("status", "ready") in {"ready", "rework"}
    ]
    model_bound_implementation_ready = [
        str(task.get("id", ""))
        for task in ready
        if task.get("task_type") == "implementation" and not is_deterministic_research_task(task)
    ]
    unsafe_ready_tasks: list[str] = []
    for task in ready:
        if str(task.get("supervisor_action", "")).startswith("drafter-calibration"):
            checks = {str(item) for item in task.get("guard_checks", []) if item}
            required = {"memory_gate", "no_live_profile_change", "no_opencode_changes"}
            if not required.issubset(checks):
                unsafe_ready_tasks.append(str(task.get("id", "")))
    noise = canonical.get("noise") if isinstance(canonical.get("noise"), dict) else {}
    replay_cases = set(replay.get("cases", [])) if isinstance(replay.get("cases"), list) else set()
    required_replay_cases = {
        "malformed-tool-fallback",
        "memory-pressure-breaker",
        "contaminated-decode-wall-clock",
        "repeated-gepa-canary-promotion",
        "plateau-pivot-state",
    }
    quality_score = quality.get("quality_score")
    scorecard = quality.get("scorecard") if isinstance(quality.get("scorecard"), dict) else {}
    scorecard_overall = scorecard.get("overall")
    handoff_score = handoff.get("score")
    duplicate_stages = calibration_stage_duplicate_count(root)
    gates = {
        "replay_ok": bool(replay.get("ok")),
        "required_replay_cases_present": required_replay_cases.issubset(replay_cases),
        "canonical_clean": bool(canonical.get("clean")),
        "zero_active_noise": all(
            int(noise.get(key, 0) or 0) == 0
            for key in (
                "unresolved_blocked_rows",
                "terminal_synthesis_rows",
                "bridge_zero_rows",
                "memory_blocks",
            )
        ),
        "quality_healthy": quality.get("verdict") == "healthy",
        "quality_score_at_least_95": isinstance(quality_score, (int, float)) and quality_score >= 95,
        "scorecard_at_least_95": isinstance(scorecard_overall, (int, float)) and scorecard_overall >= 95,
        "handoff_at_least_95": isinstance(handoff_score, (int, float)) and handoff_score >= 95,
        "task_contract_clean": bool(contract.get("ok")),
        "no_model_bound_implementation_ready": not model_bound_implementation_ready,
        "no_duplicate_stage_tasks": duplicate_stages == 0,
        "ready_tasks_guarded": not unsafe_ready_tasks,
        "memory_free_headroom": int(memory.get("free_mb", 0)) >= min_free_mb,
        "memory_compressor_bounded": int(memory.get("compressor_mb", 0)) <= max_compressor_mb,
        "memory_swap_bounded": int(memory.get("swap_used_mb", 0)) <= max_swap_mb,
    }
    return {
        "ok": all(gates.values()),
        "kind": "stability-burn-in",
        "timestamp": int(time.time()),
        "recent_rows": recent_rows,
        "gates": gates,
        "memory": memory,
        "thresholds": {
            "min_free_mb": min_free_mb,
            "max_compressor_mb": max_compressor_mb,
            "max_swap_mb": max_swap_mb,
        },
        "quality": {
            "artifact": quality.get("_artifact_path", ""),
            "verdict": quality.get("verdict", ""),
            "quality_score": quality_score,
            "scorecard_overall": scorecard_overall,
        },
        "handoff": {
            "artifact": handoff.get("_artifact_path", ""),
            "score": handoff_score,
            "ok": handoff.get("ok"),
        },
        "canonical_state": {
            "state": canonical.get("state"),
            "clean": canonical.get("clean"),
            "noise": noise,
            "deterministic_ready_tasks": canonical.get("deterministic_ready_tasks", []),
            "breakthrough_lanes": canonical.get("breakthrough_lanes", []),
        },
        "replay_cases": sorted(replay_cases),
        "missing_replay_cases": sorted(required_replay_cases - replay_cases),
        "model_bound_implementation_ready": model_bound_implementation_ready,
        "unsafe_ready_tasks": unsafe_ready_tasks,
        "task_contract": contract,
        "duplicate_stage_tasks": duplicate_stages,
    }


def stability_burn_in(args: argparse.Namespace) -> int:
    root = workspace_root()
    report = stability_burn_in_report(
        root,
        recent_rows=args.recent_rows,
        min_free_mb=args.min_free_mb,
        max_compressor_mb=args.max_compressor_mb,
        max_swap_mb=args.max_swap_mb,
    )
    path = root / "benchmarks" / f"stability-burn-in-{report['timestamp']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_result(
        root,
        run_id=f"stability-burn-in-{report['timestamp']}",
        status="keep" if report["ok"] else "blocked",
        target="autoresearch-stability-burn-in",
        hypothesis="aggressive deterministic burn-in should prove clean handoff, replay, memory, and noise gates before 10/10 certification",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"ok={report['ok']} failed_gates="
            + ",".join(key for key, value in report["gates"].items() if not value)
            + f" quality={report['quality']['quality_score']} scorecard={report['quality']['scorecard_overall']} "
            f"handoff={report['handoff']['score']} noise={report['canonical_state']['noise']} path={path}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0 if report["ok"] or args.allow_fail else 2


def recent_bad_behavior_rows(root: Path, *, recent_rows: int = 120) -> list[dict[str, Any]]:
    markers = (
        "malformed tool",
        "malformed hidden/tool",
        "tool-call loop",
        "runaway tool",
        "reasoning leak",
        "repeated reasoning",
        "repeated thought",
        "stream timeout",
        "sse read timed out",
        "python crash",
        "metal crash",
        "metal error",
        "memory crash",
        "memory pressure",
    )
    rows = result_rows(root)[-max(1, recent_rows) :]
    checkpoint_index = next(
        (
            index
            for index in range(len(rows) - 1, -1, -1)
            if str(rows[index].get("run_id", "")).startswith("frontier-system-eval-")
        ),
        -1,
    )
    if checkpoint_index >= 0:
        rows = rows[checkpoint_index + 1 :]
    bad: list[dict[str, Any]] = []
    for row in rows:
        notes = str(row.get("notes", "")).lower()
        target = str(row.get("target", "")).lower()
        if any(marker in notes or marker in target for marker in markers):
            bad.append(row)
    return bad


def latest_stable_build(root: Path) -> dict[str, Any]:
    builds = read_jsonl(root / "stable-builds.jsonl")
    for row in reversed(builds):
        if row.get("status") == "stable":
            return row
    return {}


def stable_build_mark(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    repo = Path(args.repo or repo_root()).expanduser()
    score = latest_json_artifact(root, "frontier-autonomy-score-*.json")
    if float(score.get("total_score") or 0) != 100.0 or score.get("decision") not in {"promote", "continue"}:
        result = {
            "ok": False,
            "reason": "stable build mark requires latest frontier autonomy score to be 100",
            "frontier_autonomy_score": score.get("total_score"),
            "decision": score.get("decision", ""),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    payload = {
        "timestamp": int(time.time()),
        "status": "stable",
        "commit": current_commit(repo),
        "repo": str(repo),
        "reason": args.reason or "manual stable-build mark after certified autonomy score",
        "score_artifact": score.get("_artifact_path", ""),
        "frontier_autonomy_score": score.get("total_score"),
        "quality_score": (score.get("evidence") or {}).get("quality", {}).get("quality_score")
        if isinstance(score.get("evidence"), dict)
        else None,
        "canonical_clean": (score.get("hard_gates") or {}).get("canonical_clean")
        if isinstance(score.get("hard_gates"), dict)
        else None,
    }
    append_jsonl(root / "stable-builds.jsonl", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def stable_build_rollback(args: argparse.Namespace) -> int:
    root = workspace_root()
    repo = Path(args.repo or repo_root()).expanduser()
    stable = latest_stable_build(root)
    if not stable.get("commit"):
        result = {"ok": False, "reason": "no stable build recorded"}
        print(json.dumps(result, indent=2))
        return 2
    event = {
        "timestamp": int(time.time()),
        "kind": "stable-build-rollback",
        "repo": str(repo),
        "target_commit": stable["commit"],
        "reason": args.reason or "frontier autonomy rollback requested",
        "dry_run": bool(args.dry_run),
        "ok": False,
    }
    if args.dry_run:
        event["ok"] = True
        event["action"] = "dry-run only"
    else:
        dirty = git_dirty_files(repo)
        event["dirty_files_before"] = dirty[:20]
        if dirty and not args.force:
            event["reason"] = "repo dirty; rollback requires --force"
        else:
            result = subprocess.run(
                ["git", "-C", str(repo), "reset", "--hard", str(stable["commit"])],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
            event["ok"] = result.returncode == 0
            event["output_tail"] = result.stdout[-2000:]
    append_jsonl(root / "rollback-events.jsonl", event)
    print(json.dumps(event, indent=2, sort_keys=True))
    return 0 if event["ok"] else 2


def load_crabbox_evidence(path: str, *, patch_sha256: str, max_age_seconds: int = 6 * 60 * 60) -> dict[str, Any]:
    if not path:
        return {"ok": False, "reason": "missing crabbox evidence"}
    evidence_path = Path(path).expanduser()
    try:
        loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"ok": False, "reason": f"unreadable crabbox evidence: {error}", "path": str(evidence_path)}
    if not isinstance(loaded, dict):
        return {"ok": False, "reason": "crabbox evidence is not an object", "path": str(evidence_path)}
    age = max(0, int(time.time() - evidence_path.stat().st_mtime))
    tests = loaded.get("tests") if isinstance(loaded.get("tests"), list) else []
    full_suite = loaded.get("full_suite") if isinstance(loaded.get("full_suite"), dict) else {}
    checks = {
        "evidence_ok": loaded.get("ok") is True,
        "fresh": age <= max_age_seconds,
        "runner_static_ssh_mac": str(loaded.get("runner", "")).lower() in {"static-ssh-mac", "ssh-mac", "mac-ssh"},
        "patch_hash_matches": str(loaded.get("patch_sha256", "")) == patch_sha256,
        "focused_tests_ok": bool(tests) and all(bool(item.get("ok")) for item in tests if isinstance(item, dict)),
        "full_suite_ok": bool(full_suite.get("ok")),
        "rollback_rehearsal_ok": bool(loaded.get("rollback_rehearsal_ok")),
        "logs_present": bool(loaded.get("logs") or loaded.get("run_id")),
    }
    failed = [key for key, value in checks.items() if not value]
    loaded["_artifact_path"] = str(evidence_path)
    loaded["_age_seconds"] = age
    loaded["_checks"] = checks
    loaded["_failed_checks"] = failed
    loaded["_valid_for_architectural_promotion"] = not failed
    return loaded


def frontier_autonomy_score_report(
    root: Path,
    *,
    recent_rows: int = 120,
    promotion: bool = False,
    classification: dict[str, Any] | None = None,
    patch_tests: list[dict[str, Any]] | None = None,
    crabbox_evidence: dict[str, Any] | None = None,
    rollback_rehearsal_ok: bool = False,
) -> dict[str, Any]:
    ensure_research_state(root)
    policy = load_autonomy_policy(root)
    thresholds = policy.get("thresholds") if isinstance(policy.get("thresholds"), dict) else {}
    quality = latest_json_artifact(root, "quality-review-*.json")
    frontier = latest_json_artifact(root, "frontier-system-eval-*.json")
    handoff = latest_json_artifact(root, "implementation-handoff-audit-*.json")
    burn_in = latest_json_artifact(root, "stability-burn-in-*.json")
    canonical = canonical_autoresearch_state(root, recent_rows=recent_rows)
    noise = canonical.get("noise") if isinstance(canonical.get("noise"), dict) else {}
    scorecard = quality.get("scorecard") if isinstance(quality.get("scorecard"), dict) else {}
    quality_score = float(quality.get("quality_score") or 0)
    scorecard_overall = float(scorecard.get("overall") or 0)
    frontier_overall = float(frontier.get("overall") or 0)
    handoff_score = float(handoff.get("score") or 0)
    bad_rows = recent_bad_behavior_rows(root, recent_rows=recent_rows)
    classif = classification or {}
    is_architectural = bool(classif.get("architectural"))
    crabbox_required = bool(classif.get("crabbox_required")) or is_architectural
    patch_tests_ok = (
        (patch_tests is None and not promotion)
        or (bool(patch_tests) and all(bool(test.get("ok")) for test in patch_tests))
    )
    crabbox_ok = (
        not crabbox_required
        or bool(crabbox_evidence and crabbox_evidence.get("_valid_for_architectural_promotion"))
    )
    hard_gates = {
        "quality_score_at_least_99": quality_score >= float(thresholds.get("quality_score", 99)),
        "scorecard_at_least_99": scorecard_overall >= float(thresholds.get("scorecard_overall", 99)),
        "frontier_at_least_9_8": frontier_overall >= float(thresholds.get("frontier_overall", 9.8)),
        "handoff_is_100": handoff_score >= float(thresholds.get("handoff_score", 100)),
        "stability_burn_in_pass": bool(burn_in.get("ok")),
        "canonical_clean": bool(canonical.get("clean")),
        "zero_active_noise": all(int(noise.get(key, 0) or 0) == 0 for key in noise),
        "no_bad_behavior_rows": not bad_rows,
        "patch_classification_allowed": not classif or bool(classif.get("allowed")),
        "patch_tests_complete": patch_tests_ok,
        "rollback_rehearsal_pass": (not promotion) or rollback_rehearsal_ok,
        "crabbox_evidence_complete": crabbox_ok,
    }
    if classif:
        touched = "\n".join(str(path) for path in classif.get("files", []))
        hard_gates["no_forbidden_domains"] = not any(fragment in touched.lower() for fragment in DENIED_PATCH_FRAGMENTS)
    components = {
        "stability": 30 if all(hard_gates[key] for key in ("stability_burn_in_pass", "canonical_clean", "zero_active_noise", "no_bad_behavior_rows")) else 0,
        "research_quality": 20 if hard_gates["quality_score_at_least_99"] and hard_gates["scorecard_at_least_99"] else 0,
        "implementation_safety": 20 if hard_gates["patch_classification_allowed"] and hard_gates["patch_tests_complete"] and hard_gates.get("no_forbidden_domains", True) else 0,
        "frontier_harness_health": 15 if hard_gates["frontier_at_least_9_8"] and hard_gates["handoff_is_100"] else 0,
        "speed_progress": 15 if latest_decode_mean(root, recent_rows=recent_rows) is not None or not promotion else 0,
    }
    hard_gate_failures = [key for key, ok in hard_gates.items() if not ok]
    total_score = 100 if not hard_gate_failures and sum(components.values()) >= 85 else min(99, sum(components.values()))
    decision = "promote" if promotion and total_score == 100 else "continue" if total_score >= 99 and not promotion else "repair"
    if hard_gate_failures:
        decision = "block-promotion" if promotion else "repair"
    report = {
        "ok": total_score == 100 and not hard_gate_failures,
        "kind": "frontier-autonomy-score",
        "timestamp": int(time.time()),
        "total_score": total_score,
        "decision": decision,
        "promotion": promotion,
        "components": components,
        "hard_gates": hard_gates,
        "hard_gate_failures": hard_gate_failures,
        "policy": {
            "mode": policy.get("mode"),
            "crabbox_runner": policy.get("crabbox_runner"),
            "crabbox_required": crabbox_required,
            "thresholds": thresholds,
        },
        "evidence": {
            "quality": {
                "artifact": quality.get("_artifact_path", ""),
                "quality_score": quality_score,
                "scorecard_overall": scorecard_overall,
                "verdict": quality.get("verdict", ""),
            },
            "frontier": {
                "artifact": frontier.get("_artifact_path", ""),
                "overall": frontier_overall,
                "readiness": frontier.get("readiness", ""),
                "certified": bool(frontier.get("frontier_certified")),
            },
            "handoff": {
                "artifact": handoff.get("_artifact_path", ""),
                "score": handoff_score,
                "ok": bool(handoff.get("ok")),
            },
            "stability_burn_in": {
                "artifact": burn_in.get("_artifact_path", ""),
                "ok": bool(burn_in.get("ok")),
            },
            "canonical_state": {
                "state": canonical.get("state"),
                "clean": canonical.get("clean"),
                "noise": noise,
            },
            "bad_behavior_rows": [row.get("run_id", "") for row in bad_rows[:8]],
            "classification": classif,
            "crabbox": crabbox_evidence or {},
        },
        "next": (
            "promote candidate and mark stable build"
            if decision == "promote"
            else "repair failed hard gates before mutation"
            if hard_gate_failures
            else "continue autonomous research"
        ),
    }
    return report


def write_frontier_autonomy_score(root: Path, report: dict[str, Any]) -> Path:
    path = root / "benchmarks" / f"frontier-autonomy-score-{report['timestamp']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_result(
        root,
        run_id=f"frontier-autonomy-score-{report['timestamp']}",
        status="keep" if report["ok"] else "blocked",
        target="frontier-autonomy-score",
        hypothesis="autonomous actions require zero-noise frontier stability gates before promotion",
        commit=current_commit(repo_root()),
        notes=(
            f"score={report['total_score']} decision={report['decision']} "
            f"failures={','.join(report['hard_gate_failures'])} path={path}"
        ),
    )
    return path


def frontier_autonomy_score(args: argparse.Namespace) -> int:
    root = workspace_root()
    report = frontier_autonomy_score_report(
        root,
        recent_rows=args.recent_rows,
        promotion=args.promotion,
        rollback_rehearsal_ok=args.rollback_rehearsal_ok,
    )
    path = write_frontier_autonomy_score(root, report)
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0 if report["ok"] or args.allow_fail else 2


def sota_autonomy_eval_report(root: Path, *, recent_rows: int = 160) -> dict[str, Any]:
    ensure_research_state(root)
    contracts = ensure_lane_contracts(root)
    canonical = canonical_autoresearch_state(root, recent_rows=recent_rows)
    frontier = latest_json_artifact(root, "frontier-system-eval-*.json")
    quality = latest_json_artifact(root, "quality-review-*.json")
    alive = self_improvement_alive_report(root, recent_rows=recent_rows)
    deliberation = latest_json_artifact(root, "frontier-agent-deliberation-*.json")
    source_scout = latest_json_artifact(root, "source-scout-*.json")
    burn_in = latest_json_artifact(root, "stability-burn-in-*.json")
    high_risk_patch = (
        "diff --git a/openclaw/openclaw-mtp-drafter-calibrate.py b/openclaw/openclaw-mtp-drafter-calibrate.py\n"
        "--- a/openclaw/openclaw-mtp-drafter-calibrate.py\n"
        "+++ b/openclaw/openclaw-mtp-drafter-calibrate.py\n"
        "@@ -1 +1 @@\n"
        "-MODE = 'old'\n"
        "+MODE = 'new'\n"
    )
    high_risk_classification = classify_patch(
        high_risk_patch,
        source_files=["openclaw/openclaw-mtp-drafter-calibrate.py"],
    )
    high_risk_task = agent_deliberation_task(
        int(time.time()),
        slug="sota-eval-high-risk",
        priority=1,
        target="openclaw/openclaw-jang-vlm-server.py",
        hypothesis="SOTA eval fixture: high-risk agent path must require Crabbox before promotion.",
        acceptance="SOTA eval fixture only.",
        evidence={},
    )
    program_path = root / "program.md"
    program_text = program_path.read_text(encoding="utf-8", errors="replace") if program_path.exists() else ""
    profile_terms = ["Primary metric", "secondary", "Research Method", "Current Priority", "Implementation Gate"]
    lane_contracts = contracts.get("lanes") if isinstance(contracts.get("lanes"), dict) else {}
    noise = canonical.get("noise") if isinstance(canonical.get("noise"), dict) else {}
    deliberation_gates = deliberation.get("gates") if isinstance(deliberation.get("gates"), dict) else {}
    scout_findings = source_scout.get("findings") if isinstance(source_scout.get("findings"), list) else []
    quality_score = float(quality.get("quality_score") or 0)
    scorecard = quality.get("scorecard") if isinstance(quality.get("scorecard"), dict) else {}
    frontier_overall = float(frontier.get("overall") or 0)
    alive_score = float(alive.get("total_score") or 0)
    gates = {
        "zero_active_noise": all(int(noise.get(key, 0) or 0) == 0 for key in noise),
        "canonical_clean": bool(canonical.get("clean")),
        "frontier_eval_available": frontier_overall > 0,
        "quality_review_available": quality_score > 0,
        "alive_eval_available": alive_score > 0,
        "deliberation_artifact_available": bool(deliberation),
        "deliberation_gated": bool(deliberation_gates)
        and all(bool(deliberation_gates.get(key)) for key in ("task_selected", "contract_complete")),
        "source_scout_available": bool(source_scout),
        "source_scout_allowlisted": bool(source_scout)
        and all(str(item.get("status")) in {"fetched", "skipped", "unavailable"} for item in scout_findings),
        "high_risk_classification_requires_crabbox": high_risk_classification.get("impact") == "high-risk"
        and high_risk_classification.get("crabbox_required") is True
        and high_risk_classification.get("auto_promote") is False,
        "agent_high_risk_task_requires_crabbox": high_risk_task.get("crabbox_required") is True
        and "crabbox_static_ssh_mac" in {str(item) for item in high_risk_task.get("guard_checks", [])},
        "lane_contracts_present": len(lane_contracts) >= 4,
        "program_is_goal_modular": sum(1 for term in profile_terms if term in program_text) >= 4,
        "stability_burn_in_available": bool(burn_in),
        "burn_in_clean_if_present": not burn_in or bool(burn_in.get("ok")),
    }
    components = {
        "stability_zero_noise": 20 if gates["zero_active_noise"] and gates["canonical_clean"] else 0,
        "autonomous_problem_solving": 20 if gates["deliberation_artifact_available"] and gates["deliberation_gated"] else 0,
        "source_retrieval_grounding": 15 if gates["source_scout_available"] and gates["source_scout_allowlisted"] else 0,
        "sandbox_governance": 20
        if gates["high_risk_classification_requires_crabbox"] and gates["agent_high_risk_task_requires_crabbox"]
        else 0,
        "modularity_topic_portability": 15 if gates["lane_contracts_present"] and gates["program_is_goal_modular"] else 0,
        "frontier_review_stack": 10
        if gates["frontier_eval_available"] and gates["quality_review_available"] and gates["alive_eval_available"]
        else 0,
    }
    total_score = int(sum(components.values()))
    hard_gate_failures = [
        key
        for key in (
            "zero_active_noise",
            "canonical_clean",
            "high_risk_classification_requires_crabbox",
            "agent_high_risk_task_requires_crabbox",
            "burn_in_clean_if_present",
        )
        if not gates[key]
    ]
    report = {
        "ok": total_score >= 95 and not hard_gate_failures,
        "kind": "sota-autonomy-eval",
        "timestamp": int(time.time()),
        "total_score": total_score,
        "verdict": "sota-autonomous-ready" if total_score >= 95 and not hard_gate_failures else "needs-evidence-or-repair",
        "components": components,
        "gates": gates,
        "hard_gate_failures": hard_gate_failures,
        "modularity": {
            "score": components["modularity_topic_portability"],
            "assessment": (
                "portable: swap objective, metric, sources, and lane contracts without changing the runtime harness"
                if components["modularity_topic_portability"] == 15
                else "needs a clearer objective/profile contract before using this for unrelated topics"
            ),
            "lane_contract_count": len(lane_contracts),
            "program_profile_terms_present": [term for term in profile_terms if term in program_text],
        },
        "evidence": {
            "canonical_state": canonical,
            "frontier_eval": frontier.get("_artifact_path", ""),
            "frontier_overall": frontier_overall,
            "quality_review": quality.get("_artifact_path", ""),
            "quality_score": quality_score,
            "quality_scorecard_overall": scorecard.get("overall"),
            "alive_eval_score": alive_score,
            "deliberation": deliberation.get("_artifact_path", ""),
            "source_scout": source_scout.get("_artifact_path", ""),
            "stability_burn_in": burn_in.get("_artifact_path", ""),
            "high_risk_classification": high_risk_classification,
            "high_risk_task": {
                "risk_tier": high_risk_task.get("risk_tier"),
                "crabbox_required": high_risk_task.get("crabbox_required"),
                "guard_checks": high_risk_task.get("guard_checks"),
            },
        },
        "next": (
            "continue autoresearch"
            if total_score >= 95 and not hard_gate_failures
            else "run missing review/source/deliberation evidence, then rerun sota-eval"
        ),
    }
    return report


def sota_autonomy_eval(args: argparse.Namespace) -> int:
    root = workspace_root()
    report = sota_autonomy_eval_report(root, recent_rows=args.recent_rows)
    path = root / "benchmarks" / f"sota-autonomy-eval-{report['timestamp']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_result(
        root,
        run_id=f"sota-autonomy-eval-{report['timestamp']}",
        status="keep" if report["ok"] else "blocked",
        target="autoresearch-sota-autonomy-eval",
        hypothesis=(
            "SOTA autonomy eval should prove zero-noise routing, agent deliberation, source grounding, "
            "Crabbox governance, and topic modularity."
        ),
        commit=current_commit(repo_root()),
        notes=(
            f"ok={report['ok']} total_score={report['total_score']} verdict={report['verdict']} "
            f"hard_gate_failures={','.join(report['hard_gate_failures']) or 'none'}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0 if report["ok"] or args.allow_fail else 2


def gepa_escalation(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    report = gepa_escalation_report(
        root,
        recent_rows=args.recent_rows,
        min_blocked=args.min_blocked,
        min_rework=args.min_rework,
        min_trajectory=args.min_trajectory,
        min_low_quality=args.min_low_quality,
    )
    seeded = seed_gepa_canary_task(root, report)
    timestamp = int(time.time())
    artifact = {**report, "timestamp": timestamp, "seeded_task": seeded}
    path = root / "benchmarks" / f"gepa-escalation-{timestamp}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "gepa-escalation",
            "finding": "supervisor checked whether GEPA policy optimization is needed",
            "evidence": artifact,
            "next": artifact["next"],
        },
    )
    append_result(
        root,
        run_id=f"gepa-escalation-{timestamp}",
        status="keep",
        target="autoresearch-gepa-escalation",
        hypothesis="GEPA should be a dynamic supervisor reflex only when default routing is stuck",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"needed={report.get('needed')} seeded={seeded} "
            f"triggers={','.join(str(item.get('name')) for item in report.get('triggers', []))}"
        ),
    )
    print(json.dumps({"path": str(path), **artifact}, indent=2))
    return 0


def gepa_policy_canary(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    tasks = read_jsonl(root / "tasks.jsonl")
    selected: dict[str, Any] | None = None
    for task in tasks:
        if args.task_id and task.get("id") != args.task_id:
            continue
        if task.get("status", "ready") not in {"ready", "rework"}:
            continue
        if task.get("supervisor_action") != "gepa-policy-canary":
            continue
        selected = task
        break
    if selected is None:
        result = {"ok": False, "reason": "no ready GEPA policy canary task"}
        print(json.dumps(result, indent=2))
        return 2
    canary = write_gepa_policy_canary(root, selected)
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for task in tasks:
        if task.get("id") == selected.get("id"):
            task["status"] = "done"
            task["completed_at"] = completed_at
            task["canary_path"] = canary["path"]
            break
    write_jsonl(root / "tasks.jsonl", tasks)
    append_jsonl(
        root / "experiments.jsonl",
        {
            "timestamp": completed_at,
            "task_id": selected.get("id"),
            "status": "gepa-canary-recorded",
            "artifact": canary["path"],
            "target": selected.get("target"),
        },
    )
    append_result(
        root,
        run_id=f"gepa-policy-canary-{int(time.time())}",
        status="keep",
        target="autoresearch-gepa-policy-canary",
        hypothesis="GEPA policy optimization should be canary-only until replay and approval gates pass",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=f"task_id={selected.get('id')} target={selected.get('target')} path={canary['path']}",
    )
    print(json.dumps(canary, indent=2))
    return 0


def gepa_policy_promote(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    report = gepa_policy_promotion_report(root, min_candidates=args.min_candidates)
    timestamp = int(time.time())
    path = root / "benchmarks" / f"gepa-policy-promotion-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_result(
        root,
        run_id=f"gepa-policy-promotion-{timestamp}",
        status="keep" if report.get("ok") else "blocked",
        target="autoresearch-gepa-policy-promotion",
        hypothesis="repeated GEPA canaries should become one deterministic reviewer rubric update instead of accumulating",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"promoted={report.get('promoted')} target={report.get('target', '')} "
            f"candidate_count={report.get('candidate_count', '')} reason={report.get('reason', '')}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2))
    return 0


def runtime_overhead_map(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    repo = Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))
    source = repo / "openclaw" / "openclaw-jang-vlm-server.py"
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        print(json.dumps({"ok": False, "reason": f"source unavailable: {error}"}, indent=2))
        return 2
    keywords = ("mtp", "draft", "cache", "rollback", "stream", "tool", "generate", "max_tokens")
    hits: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        lower = line.lower()
        matched = [keyword for keyword in keywords if keyword in lower]
        if matched:
            hits.append({"line": index, "keywords": matched[:4], "text": line.strip()[:180]})
    rows = result_rows(root)[-max(1, args.recent_rows) :]
    decode_rows = [row for row in rows if row.get("status") == "keep" and row.get("target") == "decode-sample"]
    signals = [decode_measurement_signal(row) for row in decode_rows]
    contaminated = [signal for signal in signals if signal["contaminated"]]
    clean_wall = [float(signal["wall_decode_tps"]) for signal in signals if not signal["contaminated"] and signal.get("wall_decode_tps") is not None]
    server = [float(signal["server_decode_tps"]) for signal in signals if signal.get("server_decode_tps") is not None]
    report = {
        "ok": True,
        "kind": "runtime-overhead-map",
        "timestamp": int(time.time()),
        "source": str(source),
        "source_hits": hits[:80],
        "hit_count": len(hits),
        "decode_rows": len(decode_rows),
        "contaminated_decode_rows": len(contaminated),
        "mean_clean_wall_decode_tps": mean_float(clean_wall),
        "mean_server_decode_tps": mean_float(server),
        "max_server_decode_tps": round(max(server), 3) if server else None,
        "diagnosis": (
            "server decode is materially faster than user-visible wall decode; prioritize proxy/tool recovery, "
            "stream guard, and MTP loop overhead boundaries before more block-size sweeps"
            if contaminated
            else "no recent contaminated decode rows; inspect MTP loop overhead only after a fresh paired benchmark"
        ),
        "next": "source_patch_only_if_specific_boundary_has_testable_overhead_reduction",
    }
    path = root / "benchmarks" / f"runtime-overhead-map-{report['timestamp']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "runtime-overhead-map",
            "finding": "supervisor mapped source and metric boundaries for runtime overhead without an LLM-led broad inspection",
            "evidence": report,
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"runtime-overhead-map-{report['timestamp']}",
        status="keep",
        target="runtime-overhead-map",
        hypothesis="deterministic source/log mapping should route contaminated wall-clock decode into the right implementation lane",
        commit=current_commit(repo),
        notes=(
            f"contaminated={len(contaminated)} mean_server_tps={report['mean_server_decode_tps']} "
            f"mean_clean_wall_tps={report['mean_clean_wall_decode_tps']} hit_count={len(hits)}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2))
    return 0


def calibration_memory_report(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    repo = Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))
    source = repo / "openclaw" / "openclaw-mtp-drafter-calibrate.py"
    rows = result_rows(root)
    blocker = recent_calibration_run_hard_blocker(root, recent_rows=240)
    plateau = recent_calibration_fallback_plateau(root, rows, recent_rows=160)
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        print(json.dumps({"ok": False, "reason": f"source unavailable: {error}"}, indent=2))
        return 2
    keywords = ("memory", "load", "target", "drafter", "cache", "gpu", "free", "compressor")
    hits: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        lower = line.lower()
        matched = [keyword for keyword in keywords if keyword in lower]
        if matched:
            hits.append({"line": index, "keywords": matched[:4], "text": line.strip()[:180]})
    report = {
        "ok": True,
        "kind": "calibration-memory-report",
        "timestamp": int(time.time()),
        "source": str(source),
        "blocker": blocker,
        "plateau": plateau,
        "source_hits": hits[:80],
        "hit_count": len(hits),
        "diagnosis": calibration_blocker_diagnosis(blocker),
        "next": calibration_blocker_next_step(blocker),
    }
    path = root / "benchmarks" / f"calibration-memory-report-{report['timestamp']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "calibration-memory-report",
            "finding": "supervisor mapped the JANQ drafter calibration memory blocker without loading a model",
            "evidence": report,
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"calibration-memory-report-{report['timestamp']}",
        status="blocked" if blocker else "keep",
        target="calibration-memory-report",
        hypothesis="Calibration plateau should become a no-model root-cause report instead of repeated decode remeasurements.",
        commit=current_commit(repo),
        notes=(
            f"blocker={blocker or 'none'} hit_count={len(hits)} "
            f"plateau={str(bool(plateau)).lower()} "
            f"best_clean_decode_tps={(plateau or {}).get('best_clean_decode_tps', '')}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2))
    return 2 if blocker else 0


def drafter_trace_gate(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    plan_path = Path(args.plan).expanduser()
    trace_candidates = [Path(args.trace_data).expanduser()] if args.trace_data else []
    env_trace = os.environ.get("OPENCLAW_DRAFTER_TRACE_DATA", "")
    if env_trace and not trace_candidates:
        trace_candidates.append(Path(env_trace).expanduser())
    if not trace_candidates:
        trace_candidates.extend(drafter_trace_candidates())
    if not args.trace_data:
        for path in drafter_trace_candidates():
            if path not in trace_candidates:
                trace_candidates.append(path)
    plan: dict[str, Any] = {}
    errors: list[str] = []
    if not plan_path.exists():
        errors.append(f"missing_plan={plan_path}")
    else:
        try:
            with plan_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                plan = loaded
            else:
                errors.append("plan_not_json_object")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"plan_read_error={type(error).__name__}")
    decision = str(plan.get("decision", ""))
    traces = [path for path in trace_candidates if path.exists() and path.stat().st_size > 0]
    status = "keep" if decision == "ready-for-target-generated-trace-data" and traces and not errors else "blocked"
    if errors:
        reason = ",".join(errors)
    elif decision != "ready-for-target-generated-trace-data":
        reason = f"plan_decision_not_ready:{decision or 'missing'}"
    elif not traces:
        reason = "target-generated-trace-data-missing"
    else:
        reason = "target-generated-trace-data-present"
    report = {
        "ok": True,
        "status": status,
        "reason": reason,
        "plan": str(plan_path),
        "plan_decision": decision,
        "trace_candidates": [str(path) for path in trace_candidates],
        "trace_data": [str(path) for path in traces],
        "next_action": (
            "collect_target_generated_trace_data"
            if status == "blocked"
            else "run_candidate_drafter_calibration_canary"
        ),
    }
    timestamp = int(time.time())
    seeded_canary_task = 0
    if status == "keep" and should_seed_drafter_calibration_canary(root, recent_rows=120):
        seeded_canary_task = upsert_tasks(
            root,
            [drafter_calibration_canary_task(timestamp, task_id=f"drafter-calibration-canary-{timestamp}")],
        )
    report["seeded_canary_task"] = seeded_canary_task
    append_jsonl(
        root / "experiments.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "drafter-trace-gate",
            "status": status,
            "artifact": str(plan_path),
            "target": "janq-drafter-fit-trace-data",
            "report": report,
        },
    )
    append_result(
        root,
        run_id=f"drafter-trace-gate-{timestamp}",
        status=status,
        target="janq-drafter-fit-trace-data",
        hypothesis="Drafter-fit planning must advance to trace data or block explicitly instead of repeating plan generation.",
        commit=current_commit(repo_root()),
        notes=(
            f"decision={reason} plan_decision={decision} "
            f"trace_files={len(traces)} seeded_canary_task={seeded_canary_task} next={report['next_action']}"
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def drafter_trace_prerequisite(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    candidates = drafter_trace_candidates()
    traces = [path for path in candidates if path.exists() and path.stat().st_size > 0]
    status = "keep" if traces else "blocked"
    reason = "target-generated-trace-data-present" if traces else "target-generated-trace-data-missing"
    timestamp = int(time.time())
    report = {
        "ok": True,
        "kind": "drafter-trace-prerequisite",
        "status": status,
        "reason": reason,
        "trace_candidates": [str(path) for path in candidates],
        "trace_data": [str(path) for path in traces],
        "next_action": "run_candidate_drafter_calibration_canary" if traces else "collect_target_generated_trace_data",
        "timestamp": timestamp,
    }
    seeded_collect_task = 0
    if not traces:
        seeded_collect_task = upsert_tasks(
            root,
            [drafter_trace_collect_task(timestamp, task_id=f"handoff-audit-drafter-trace-collect-{timestamp}")],
        )
    report["seeded_collect_task"] = seeded_collect_task
    seeded_canary_task = 0
    if traces and should_seed_drafter_calibration_canary(root, recent_rows=120):
        seeded_canary_task = upsert_tasks(
            root,
            [drafter_calibration_canary_task(timestamp, task_id=f"drafter-calibration-canary-{timestamp}")],
        )
    report["seeded_canary_task"] = seeded_canary_task
    path = root / "benchmarks" / f"drafter-trace-prerequisite-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "drafter-trace-prerequisite",
            "finding": "JANQ drafter fitting is blocked on target-generated trace data rather than another synthesis loop",
            "evidence": report,
            "next": report["next_action"],
        },
    )
    append_result(
        root,
        run_id=f"drafter-trace-prerequisite-{timestamp}",
        status=status,
        target="janq-drafter-fit-trace-data",
        hypothesis="Trace-data prerequisite must be explicit before drafter calibration work continues.",
        commit=current_commit(repo_root()),
        notes=(
            f"decision={reason} trace_files={len(traces)} seeded_collect_task={seeded_collect_task} "
            f"seeded_canary_task={seeded_canary_task} next={report['next_action']}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0


def drafter_trace_collect(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    before_memory = memory_snapshot()
    if before_memory.get("free_mb", 0) < int(args.min_free_mb):
        reason = f"memory_free_mb_below_min:{before_memory.get('free_mb', 0)}<{int(args.min_free_mb)}"
        print(json.dumps({"ok": False, "status": "blocked", "reason": reason, "memory_before_mb": before_memory}, indent=2))
        append_result(
            root,
            run_id=f"drafter-trace-collect-{int(time.time())}",
            status="blocked",
            target="janq-drafter-fit-trace-data",
            hypothesis="Collect bounded target-generated traces for JANQ drafter fitting.",
            commit=current_commit(repo_root()),
            notes=reason,
        )
        return 2
    base_url = args.base_url
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=3) as response:
            models = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        reason = f"model endpoint unavailable: {error}"
        print(json.dumps({"ok": False, "status": "blocked", "reason": reason}, indent=2))
        append_result(
            root,
            run_id=f"drafter-trace-collect-{int(time.time())}",
            status="blocked",
            target="janq-drafter-fit-trace-data",
            hypothesis="Collect bounded target-generated traces for JANQ drafter fitting.",
            commit=current_commit(repo_root()),
            notes=reason,
        )
        return 2
    data = models.get("data") if isinstance(models, dict) else None
    model = args.model or (data[0].get("id") if isinstance(data, list) and data and isinstance(data[0], dict) else "local-model")
    existing: list[dict[str, Any]] = []
    if output.exists() and not args.force:
        for line in output.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                existing.append(item)
    existing_ids = {str(item.get("prompt_id", "")) for item in existing}
    prompts = list(DEFAULT_TRACE_PROMPTS)[: max(1, int(args.samples))]
    collected: list[dict[str, Any]] = []
    failures: list[str] = []
    for prompt_id, prompt in prompts:
        if prompt_id in existing_ids:
            continue
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": int(args.max_tokens),
            "stream": False,
        }
        try:
            wall_s, body = model_request(base_url, payload, float(args.timeout))
            parsed = json.loads(body.decode("utf-8"))
            content = str(parsed.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
            completion_tokens, token_source = completion_tokens_from_response(parsed, content)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as error:
            failures.append(f"{prompt_id}:{type(error).__name__}")
            continue
        if not content:
            failures.append(f"{prompt_id}:empty_completion")
            continue
        collected.append(
            {
                "schema_version": 1,
                "created": int(time.time()),
                "model": model,
                "prompt_id": prompt_id,
                "prompt_class": "openclaw-agentic",
                "messages": [{"role": "user", "content": prompt}],
                "completion": content,
                "completion_tokens": completion_tokens,
                "completion_token_source": token_source,
                "wall_s": round(wall_s, 3),
                "max_tokens": int(args.max_tokens),
                "temperature": 0,
            }
        )
    rows = existing + collected if not args.force else collected
    if rows:
        output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    after_memory = memory_snapshot()
    status = "keep" if len(rows) >= int(args.min_traces) else "blocked"
    reason = "target-generated-trace-data-present" if status == "keep" else "insufficient-target-generated-traces"
    timestamp = int(time.time())
    report = {
        "ok": status == "keep",
        "kind": "drafter-trace-collect",
        "status": status,
        "reason": reason,
        "output": str(output),
        "model": model,
        "existing_rows": len(existing),
        "collected_rows": len(collected),
        "total_rows": len(rows),
        "failures": failures,
        "memory_before_mb": before_memory,
        "memory_after_mb": after_memory,
        "timestamp": timestamp,
    }
    artifact = root / "benchmarks" / f"drafter-trace-collect-{timestamp}.json"
    artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "drafter-trace-collect",
            "finding": "collected bounded JANQ target-generated traces for drafter-fit calibration",
            "evidence": report,
            "next": "rerun drafter-trace-gate" if status == "keep" else "fix model endpoint or collect more traces",
        },
    )
    append_result(
        root,
        run_id=f"drafter-trace-collect-{timestamp}",
        status=status,
        target="janq-drafter-fit-trace-data",
        hypothesis="Collect bounded target-generated traces for JANQ drafter fitting.",
        wall_s=round(sum(float(row.get("wall_s") or 0) for row in collected), 3) if collected else "",
        memory_gb=round(after_memory.get("compressor_mb", 0) / 1024, 3) if after_memory else "",
        commit=current_commit(repo_root()),
        notes=(
            f"reason={reason} output={output} collected={len(collected)} total={len(rows)} "
            f"failures={len(failures)}"
        ),
    )
    print(json.dumps({"path": str(artifact), **report}, indent=2, sort_keys=True))
    return 0 if status == "keep" else 2


def read_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("trace row is not a JSON object")
        rows.append(value)
    return rows


def trace_prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return str(row.get("prompt") or row.get("content") or "").strip()


def drafter_calibration_canary(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    timestamp = int(time.time())
    mode = calibration_mode(getattr(args, "calibration_mode", CALIBRATION_DIRECT_MODE))
    plan_path = Path(args.plan).expanduser()
    trace_paths = existing_drafter_trace_paths()
    trace_path = Path(args.trace_data).expanduser() if args.trace_data else (trace_paths[0] if trace_paths else None)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "calibration-canary-prompts.txt"
    before_memory = memory_snapshot()
    failures: list[str] = []
    plan: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    if not plan_path.exists():
        failures.append(f"missing_plan:{plan_path}")
    else:
        try:
            plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
            if not isinstance(plan_value, dict):
                raise ValueError("plan is not a JSON object")
            plan = plan_value
        except (OSError, json.JSONDecodeError, ValueError) as error:
            failures.append(f"plan_read_error:{type(error).__name__}")
    if plan and plan.get("decision") != "ready-for-target-generated-trace-data":
        failures.append(f"plan_not_ready:{plan.get('decision', '')}")
    if trace_path is None or not trace_path.exists():
        failures.append("missing_trace_data")
    else:
        try:
            trace_rows = read_trace_rows(trace_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            failures.append(f"trace_read_error:{type(error).__name__}")
    if len(trace_rows) < int(args.min_traces):
        failures.append(f"trace_rows_below_min:{len(trace_rows)}<{int(args.min_traces)}")
    prompts = [prompt for row in trace_rows if (prompt := trace_prompt_text(row))]
    if len(prompts) < int(args.min_traces):
        failures.append(f"trace_prompts_below_min:{len(prompts)}<{int(args.min_traces)}")
    if prompts:
        prompts_path.write_text("\n".join(prompts[: int(args.max_prompts)]) + "\n", encoding="utf-8")
    test_result = {"ok": False, "returncode": None, "tail": ""}
    if not failures and not args.skip_test:
        repo = repo_root()
        command = ["python3", str(repo / "openclaw" / "test-drafter-fit.py")]
        try:
            result = subprocess.run(
                command,
                cwd=str(repo),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=float(args.test_timeout),
                check=False,
            )
            test_result = {
                "ok": result.returncode == 0,
                "returncode": result.returncode,
                "tail": result.stdout[-1200:],
            }
            if result.returncode != 0:
                failures.append(f"focused_test_exit:{result.returncode}")
        except subprocess.TimeoutExpired:
            test_result = {"ok": False, "returncode": 124, "tail": "focused test timeout"}
            failures.append("focused_test_timeout")
    runtime_python = calibration_python()
    runtime_issue = calibration_runtime_import_issue(runtime_python)
    if runtime_issue:
        failures.append(runtime_issue)
    calibrated_output = output_dir / f"calibrated-drafter-canary-{timestamp}"
    target_path = os.environ.get("OPENCLAW_JANQ_TARGET_PATH", str(plan.get("target_path") or DEFAULT_JANQ_TARGET_PATH))
    drafter_path = os.environ.get(
        "OPENCLAW_MTP_DRAFT_PATH",
        os.environ.get("OPENCLAW_JANG_DRAFT_MODEL", DEFAULT_MTP_DRAFT_PATH),
    )
    bounded_command = calibration_full_run_command(
        target_path=target_path,
        drafter_path=drafter_path,
        output_path=calibrated_output,
        prompts_path=prompts_path,
        calibration_mode_value=mode,
    )
    status = "keep" if not failures else "blocked"
    report = {
        "ok": status == "keep",
        "kind": "drafter-calibration-canary",
        "status": status,
        "decision": "ready-for-bounded-calibration" if status == "keep" else "blocked",
        "failures": failures,
        "runtime_python": runtime_python,
        "runtime_import_ok": not runtime_issue,
        "plan": str(plan_path),
        "trace_data": str(trace_path) if trace_path is not None else "",
        "trace_rows": len(trace_rows),
        "prompts_file": str(prompts_path) if prompts else "",
        "test_result": test_result,
        "bounded_calibration_command": bounded_command,
        "calibration_mode": mode,
        "memory_before_mb": before_memory,
        "memory_after_mb": memory_snapshot(),
        "timestamp": timestamp,
    }
    seeded_stage_task = 0
    if status == "keep":
        stage = first_seedable_calibration_memory_stage(root, calibration_mode_filter=mode)
    else:
        stage = ""
    if stage:
        stage_command = calibration_stage_helper_command(
            stage,
            plan=str(plan_path),
            trace_data=str(trace_path) if trace_path is not None else "",
            output_dir=str(output_dir),
            min_traces=int(args.min_traces),
            max_prompts=int(args.max_prompts),
            calibration_mode_value=mode,
        )
        seeded_stage_task = upsert_tasks(
            root,
            [
                drafter_calibration_memory_stage_task(
                    timestamp,
                    stage=stage,
                    task_id=f"drafter-calibration-memory-stage-{stage}-{timestamp}",
                    bounded_command=stage_command,
                    calibration_mode_value=mode,
                )
            ],
        )
    report["seeded_stage_task"] = seeded_stage_task
    artifact = root / "benchmarks" / f"drafter-calibration-canary-{timestamp}.json"
    artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "drafter-calibration-canary",
            "finding": "validated JANQ trace data and bounded calibration canary prerequisites",
            "evidence": report,
            "next": (
                "advance through staged calibration memory gates before full calibration"
                if status == "keep"
                else "repair calibration canary prerequisites"
            ),
        },
    )
    append_result(
        root,
        run_id=f"drafter-calibration-canary-{timestamp}",
        status=status,
        target="janq-drafter-calibration-canary",
        hypothesis="JANQ trace data should advance into a bounded calibration canary instead of repeated trace gates.",
        commit=current_commit(repo_root()),
        notes=(
            f"decision={report['decision']} trace_rows={len(trace_rows)} "
            f"calibration_mode={mode} "
            f"test_ok={test_result.get('ok')} failures={len(failures)} seeded_stage_task={seeded_stage_task}"
        ),
    )
    print(json.dumps({"path": str(artifact), **report}, indent=2, sort_keys=True))
    return 0 if status == "keep" else 2


def drafter_calibration_memory_stage(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    timestamp = int(time.time())
    stage = str(args.stage)
    mode = calibration_mode(getattr(args, "calibration_mode", CALIBRATION_DIRECT_MODE))
    plan_path = Path(args.plan).expanduser()
    trace_paths = existing_drafter_trace_paths()
    trace_path = Path(args.trace_data).expanduser() if args.trace_data else (trace_paths[0] if trace_paths else None)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "calibration-canary-prompts.txt"
    failures: list[str] = []
    plan: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    if stage not in CALIBRATION_MEMORY_STAGES:
        failures.append(f"unknown_stage:{stage}")
    if not plan_path.exists():
        failures.append(f"missing_plan:{plan_path}")
    else:
        try:
            loaded = json.loads(plan_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("plan is not a JSON object")
            plan = loaded
        except (OSError, json.JSONDecodeError, ValueError) as error:
            failures.append(f"plan_read_error:{type(error).__name__}")
    if trace_path is None or not trace_path.exists():
        failures.append("missing_trace_data")
    else:
        try:
            trace_rows = read_trace_rows(trace_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            failures.append(f"trace_read_error:{type(error).__name__}")
    prompts = [prompt for row in trace_rows if (prompt := trace_prompt_text(row))]
    if len(prompts) < int(args.min_traces):
        failures.append(f"trace_prompts_below_min:{len(prompts)}<{int(args.min_traces)}")
    if prompts:
        prompts_path.write_text("\n".join(prompts[: int(args.max_prompts)]) + "\n", encoding="utf-8")
    target_path = os.environ.get("OPENCLAW_JANQ_TARGET_PATH", str(plan.get("target_path") or DEFAULT_JANQ_TARGET_PATH))
    drafter_path = os.environ.get(
        "OPENCLAW_MTP_DRAFT_PATH",
        os.environ.get("OPENCLAW_JANG_DRAFT_MODEL", DEFAULT_MTP_DRAFT_PATH),
    )
    stage_output = output_dir / f"calibration-stage-{stage}-{timestamp}"
    command = calibration_probe_command(
        stage,
        target_path=target_path,
        drafter_path=drafter_path,
        output_path=stage_output,
        prompts_path=prompts_path,
        calibration_mode_value=mode,
    )
    timeout = {
        "metadata": 90.0,
        "drafter-load": 600.0,
        "target-load": 900.0,
        "combined-load": 1200.0,
        "micro-step": 1800.0,
    }.get(stage, 300.0)
    stdout = ""
    returncode = 2
    if not failures:
        try:
            result = subprocess.run(
                command,
                env=calibration_python_env(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            stdout = result.stdout
            returncode = result.returncode
            if result.returncode != 0:
                failures.append(f"probe_exit:{result.returncode}")
                gradient_issue = calibration_quantized_gradient_issue(result.stdout)
                if gradient_issue:
                    failures.append(gradient_issue)
                if "calibration memory gate blocked: after-load" in result.stdout.lower():
                    failures.append("calibration-memory-gate:after-load")
        except subprocess.TimeoutExpired as error:
            stdout = (error.stdout or "") if isinstance(error.stdout, str) else ""
            failures.append("probe_timeout")
            gradient_issue = calibration_quantized_gradient_issue(stdout)
            if gradient_issue:
                failures.append(gradient_issue)
            returncode = 124
        except OSError as error:
            failures.append(f"probe_exec_error:{type(error).__name__}")
    status = "keep" if not failures and returncode == 0 else "blocked"
    next_stage = ""
    if status == "keep":
        completed = completed_calibration_memory_stages(
            root,
            calibration_mode_filter=mode,
        ) | {stage}
        active = {
            calibration_memory_stage_name(task)
            for task in active_calibration_memory_stage_tasks(root)
            if calibration_task_mode(task) == mode
        }
        for candidate in CALIBRATION_MEMORY_STAGES:
            if candidate in completed or candidate in active:
                continue
            next_stage = candidate
            break
    seeded_next_stage = 0
    seeded_run_task = 0
    if status == "keep" and next_stage:
        next_command = calibration_stage_helper_command(
            next_stage,
            plan=str(plan_path),
            trace_data=str(trace_path) if trace_path is not None else "",
            output_dir=str(output_dir),
            min_traces=int(args.min_traces),
            max_prompts=int(args.max_prompts),
            calibration_mode_value=mode,
        )
        seeded_next_stage = upsert_tasks(
            root,
            [
                drafter_calibration_memory_stage_task(
                    timestamp,
                    stage=next_stage,
                    task_id=f"drafter-calibration-memory-stage-{next_stage}-{timestamp}",
                    bounded_command=next_command,
                    calibration_mode_value=mode,
                )
            ],
        )
    elif status == "keep" and not next_stage and should_seed_drafter_calibration_run(root, recent_rows=120):
        calibrated_output = output_dir / f"calibrated-drafter-{timestamp}"
        run_command = calibration_full_run_command(
            target_path=target_path,
            drafter_path=drafter_path,
            output_path=calibrated_output,
            prompts_path=prompts_path,
            calibration_mode_value=mode,
        )
        seeded_run_task = upsert_tasks(
            root,
            [
                drafter_calibration_run_task(
                    timestamp,
                    task_id=f"drafter-calibration-run-{timestamp}",
                    bounded_command=run_command,
                    calibration_mode_value=mode,
                )
            ],
        )
    decision = "blocked"
    if status == "keep":
        decision = "advance" if next_stage else "ready-for-bounded-calibration"
    elif CALIBRATION_QUANTIZED_GRADIENT_BLOCKER in failures:
        decision = "terminal-blocker"
    report = {
        "ok": status == "keep",
        "kind": "drafter-calibration-memory-stage",
        "stage": stage,
        "status": status,
        "decision": decision,
        "failures": failures,
        "command": command,
        "calibration_mode": mode,
        "returncode": returncode,
        "output_tail": stdout[-1600:],
        "stage_output": str(stage_output),
        "trace_rows": len(trace_rows),
        "prompts_file": str(prompts_path) if prompts else "",
        "seeded_next_stage": seeded_next_stage,
        "seeded_run_task": seeded_run_task,
        "memory_after_mb": memory_snapshot(),
        "timestamp": timestamp,
    }
    artifact = root / "benchmarks" / f"drafter-calibration-memory-stage-{stage}-{timestamp}.json"
    artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": f"drafter-calibration-memory-stage-{stage}",
            "finding": f"ran staged JANQ drafter calibration memory gate: {stage}",
            "evidence": report,
            "next": (
                f"run next calibration memory stage: {next_stage}"
                if seeded_next_stage
                else "run bounded calibration task" if seeded_run_task else "repair or wait on calibration stage blocker"
            ),
        },
    )
    append_result(
        root,
        run_id=f"drafter-calibration-memory-stage-{stage}-{timestamp}",
        status=status,
        target="janq-drafter-calibration-memory-stage",
        hypothesis="Staged calibration probes should isolate memory/runtime blockers before full JANQ drafter fitting.",
        commit=current_commit(repo_root()),
        notes=(
            f"stage={stage} decision={decision} failures={','.join(failures) or 'none'} "
            f"calibration_mode={mode} "
            f"blocker={CALIBRATION_QUANTIZED_GRADIENT_BLOCKER if CALIBRATION_QUANTIZED_GRADIENT_BLOCKER in failures else ''} "
            f"seeded_next_stage={seeded_next_stage} seeded_run_task={seeded_run_task}"
        ),
    )
    print(json.dumps({"path": str(artifact), **report}, indent=2, sort_keys=True))
    return 0 if status == "keep" else 2


def dflash_compatibility_gate(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    repo = repo_root()
    server_path = repo / "openclaw" / "openclaw-jang-vlm-server.py"
    launcher_path = repo / "openclaw" / "openclaw-jang-vlm-launcher.py"
    draft_path = Path(args.draft_path).expanduser() if args.draft_path else default_dflash_draft_path()
    plan_path = Path(args.plan).expanduser()
    blockers: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, Any] = {
        "server_path": str(server_path),
        "launcher_path": str(launcher_path),
        "draft_path": str(draft_path),
        "plan_path": str(plan_path),
    }

    try:
        server_text = server_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        server_text = ""
        blockers.append(f"server_read_error={type(error).__name__}")
    try:
        launcher_text = launcher_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        launcher_text = ""
        blockers.append(f"launcher_read_error={type(error).__name__}")

    required_server_hooks = {
        "load_dflash": "from dflash.model_mlx import load_draft",
        "stream_generate": "from dflash.model_mlx import stream_generate",
        "compatibility_check": "validate_dflash_compatibility",
        "tool_guard": "should_use_dflash",
        "acceptance_metrics": "record_dflash_acceptance",
    }
    evidence["server_hooks"] = {
        name: needle in server_text for name, needle in required_server_hooks.items()
    }
    for name, present in evidence["server_hooks"].items():
        if not present:
            blockers.append(f"missing_server_hook={name}")

    required_launcher_hooks = {
        "separate_runtime_check": "ensure_dflash_runtime",
        "package_import_probe": "import dflash.model_mlx",
    }
    evidence["launcher_hooks"] = {
        name: needle in launcher_text for name, needle in required_launcher_hooks.items()
    }
    for name, present in evidence["launcher_hooks"].items():
        if not present:
            blockers.append(f"missing_launcher_hook={name}")

    draft_config = draft_path / "config.json"
    evidence["draft_config_exists"] = draft_config.exists()
    if not draft_config.exists():
        blockers.append("dflash_draft_config_missing")
    else:
        try:
            with draft_config.open("r", encoding="utf-8") as file:
                loaded_config = json.load(file)
            dflash_cfg = loaded_config.get("dflash_config") if isinstance(loaded_config, dict) else {}
            if not isinstance(dflash_cfg, dict):
                dflash_cfg = {}
            evidence["draft_model_type"] = loaded_config.get("model_type") if isinstance(loaded_config, dict) else ""
            evidence["draft_target_layer_ids"] = (
                loaded_config.get("target_layer_ids") if isinstance(loaded_config, dict) else None
            ) or dflash_cfg.get("target_layer_ids")
            evidence["draft_num_target_layers"] = (
                loaded_config.get("num_target_layers") if isinstance(loaded_config, dict) else None
            ) or dflash_cfg.get("num_target_layers")
            model_type = str(evidence["draft_model_type"]).lower()
            if model_type and "gemma" not in model_type:
                blockers.append(f"draft_model_type_mismatch={model_type}")
            if not evidence["draft_target_layer_ids"]:
                blockers.append("draft_target_layer_ids_missing")
        except (OSError, json.JSONDecodeError) as error:
            blockers.append(f"draft_config_read_error={type(error).__name__}")

    plan_decision = ""
    if plan_path.exists():
        try:
            with plan_path.open("r", encoding="utf-8") as file:
                plan = json.load(file)
            if isinstance(plan, dict):
                plan_decision = str(plan.get("decision", ""))
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"fit_plan_read_error={type(error).__name__}")
    else:
        warnings.append("fit_plan_missing")
    evidence["fit_plan_decision"] = plan_decision
    if plan_decision and plan_decision != "ready-for-target-generated-trace-data":
        blockers.append(f"fit_plan_not_ready={plan_decision}")

    status = "keep" if not blockers else "blocked"
    decision = "canary-plan-ready" if status == "keep" else "blocked"
    timestamp = int(time.time())
    report = {
        "ok": True,
        "kind": "dflash-compatibility-gate",
        "status": status,
        "decision": decision,
        "blockers": blockers,
        "warnings": warnings,
        "evidence": evidence,
        "promotion_gate": {
            "must_keep_target_model": True,
            "must_use_separate_env": True,
            "must_pass_tool_thinking_stream_replay": True,
            "must_beat_current_decode_tps": True,
            "must_restore_live_profile": True,
        },
        "next_action": (
            "run_dflash_canary_with_paired_decode_benchmark"
            if status == "keep"
            else "repair_blockers_or_collect_target_generated_trace_data"
        ),
        "timestamp": timestamp,
    }
    path = root / "experiments" / f"dflash-compatibility-gate-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "dflash-compatibility-gate",
            "finding": "supervisor converted DFlash research into a deterministic no-model-load compatibility gate",
            "evidence": report,
            "next": report["next_action"],
        },
    )
    append_result(
        root,
        run_id=f"dflash-compatibility-gate-{timestamp}",
        status=status,
        target="frontier-dflash",
        hypothesis="DFlash must pass a deterministic JANQ compatibility gate before any live runtime canary.",
        commit=current_commit(repo),
        notes=(
            f"decision={decision} blockers={len(blockers)} warnings={len(warnings)} "
            f"path={path}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0


def hypothesis_rank(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    ranked = rank_tasks(root, limit=args.limit)
    timestamp = int(time.time())
    report = {
        "ok": True,
        "kind": "hypothesis-rank",
        "timestamp": timestamp,
        "ranked": ranked,
        "next_task": ranked[0]["task_id"] if ranked else "",
        "next": "select_next_task" if ranked else "synthesize_or_refill_queue",
    }
    path = root / "benchmarks" / f"hypothesis-rank-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(root / "hypothesis-rank.jsonl", {**report, "path": str(path)})
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "hypothesis-rank",
            "finding": "supervisor ranked ready hypotheses by evidence, risk, guard coverage, and decode-speed relevance",
            "evidence": report,
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"hypothesis-rank-{timestamp}",
        status="keep",
        target="autoresearch-hypothesis-rank",
        hypothesis="ranked hypothesis selection should reduce low-signal overnight cycles",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=f"ranked={len(ranked)} next_task={report['next_task']}",
    )
    print(json.dumps({"path": str(path), **report}, indent=2))
    return 0


def causal_review(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    report = causal_review_report(root, recent_rows=args.recent_rows)
    timestamp = int(time.time())
    report["timestamp"] = timestamp
    repair_tasks: list[dict[str, Any]] = []
    if report.get("regression_suspected"):
        repair_tasks.append(
            {
                "id": f"causal-remeasure-decode-{timestamp}",
                "status": "ready",
                "priority": 98,
                "lane": "causal-repair",
                "target": "decode-sample",
                "hypothesis": "A suspected post-change decode regression must be remeasured before any further promotion.",
                "metric": "decode_tps",
                "benchmark_mode": "decode-sample",
                "guard_checks": ["memory_ok", "no_reasoning_leak", "no_sse_timeout"],
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
            }
        )
    if int(report.get("contaminated_decode_rows") or 0) >= 3:
        repair_tasks.append(
            {
                "id": f"causal-runtime-overhead-map-{timestamp}",
                "status": "ready",
                "priority": 96,
                "lane": "runtime-overhead",
                "task_type": "supervisor",
                "supervisor_action": "runtime-overhead-map",
                "target": "openclaw/openclaw-jang-vlm-server.py",
                "hypothesis": "Repeated contaminated decode rows require a runtime/proxy overhead map before more speed conclusions.",
                "metric": "server_wall_decode_gap",
                "guard_checks": ["no_live_profile_change", "no_model_turn_required", "no_opencode_changes"],
                "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
            }
        )
    if report.get("low_confidence_kept"):
        report["low_confidence_action"] = (
            "recorded_in_causal_review_only; no model-bound repair task seeded because "
            "the deterministic causal-review command is the source of truth"
        )
    seeded = upsert_tasks(root, repair_tasks) if repair_tasks else 0
    report["seeded_repair_tasks"] = seeded
    path = root / "benchmarks" / f"causal-review-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(root / "causal-reviews.jsonl", {**report, "path": str(path)})
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "causal-review",
            "finding": "supervisor checked post-change decode metrics and low-confidence kept decisions",
            "evidence": report,
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"causal-review-{timestamp}",
        status="blocked" if report.get("regression_suspected") else "keep",
        target="autoresearch-causal-review",
        hypothesis="post-change causal review should catch regressions before promotion confidence drifts",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"current_decode_mean={report.get('current_decode_mean')} "
            f"previous_decode_mean={report.get('previous_decode_mean')} "
            f"delta={report.get('decode_delta')} regression={report.get('regression_suspected')} "
            f"low_confidence={','.join(report.get('low_confidence_kept', []))}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2))
    return 0


def synthesis_ideas(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    decode = mean_value(float_values(rows, "decode-sample", "decode_tps"))
    return [
        {
            "id": "mtp-acceptance-bottleneck",
            "lane": "production-mtp",
            "cause": "Speculative decode speed is limited when accepted draft tokens do not offset drafter and verification overhead.",
            "proposed_change": "Add a deterministic MTP acceptance report and use it to choose the next drafter/block experiment.",
            "expected_metric_delta": "Raise decode_tps by selecting only changes that improve mean_accept or reduce verification overhead.",
            "expected": "Raise real decode TPS by identifying whether low `mean_accept` or drafter overhead is the limiting factor.",
            "math": "Speculative speedup S ~= T_target_only / (T_draft + T_verify); acceptance must be high enough that avoided target steps exceed drafter cost.",
            "prototype": "Parse recent proxy/server logs for `mtp_rounds`, `mean_accept`, block size, drafter path, and decode tok/s, then compare against no-drafter control.",
            "risk": "Log-only conclusions can be misleading; promote only ideas that survive a paired decode benchmark.",
            "rollback": "Discard acceptance-based changes unless paired decode benchmarks improve and replay checks pass.",
            "evidence": f"Current decode sample mean={decode if decode is not None else 'not yet measured'} tok/s; live target is >18 tok/s first.",
        },
        {
            "id": "drafter-block-and-quant-sweep",
            "lane": "production-mtp",
            "cause": "MTP block size and drafter quantization trade off acceptance, overhead, and memory pressure.",
            "proposed_change": "Create a paired-control sweep plan that restores the live profile around every block/quantization trial.",
            "expected_metric_delta": "Find a configuration with positive decode_tps_delta over the current live profile.",
            "expected": "Find the fastest safe assistant drafter configuration without changing the JANQ target model.",
            "math": "Choose argmax_config decode_tps(config) subject to memory_ok, no_loop, no_reasoning_leak, and quality_guard.",
            "prototype": "Run a paired sweep for block size and drafter quantization, restoring the live profile after each bounded benchmark.",
            "risk": "A faster synthetic prompt can regress normal text or code; use a fixed mixed prompt set.",
            "rollback": "Restore the live profile after each trial and keep no setting unless the same prompt set improves.",
            "evidence": "The current live q4 drafter at block size 2 improved decode modestly; prior 3-bit and heuristic schedule attempts were slower.",
        },
        {
            "id": "janq-drafter-alignment",
            "lane": "drafter-alignment",
            "cause": "The assistant drafter may be misaligned with unlocked JANQ target behavior, lowering acceptance.",
            "proposed_change": "Gate JANQ-specific calibration by wall-clock decode and acceptance improvements, not loss-only proxies.",
            "expected_metric_delta": "Improve mean_accept and decode_tps on the fixed manifest prompt set.",
            "expected": "Improve acceptance by making the assistant drafter better match the unlocked JANQ target behavior.",
            "math": "Minimize KL(target_logits || drafter_logits) on rolling JANQ traces, weighted by positions where draft rejection currently occurs.",
            "prototype": "Use `openclaw-mtp-drafter-calibrate.py` to test one narrow calibration target at a time, then benchmark against the official q4 drafter.",
            "risk": "Calibration can overfit traces or slow the drafter; discard unless wall-clock decode TPS improves.",
            "rollback": "Keep official q4 drafter unless calibrated variant beats it under the promotion gate.",
            "evidence": "Pre-projection-only calibration did not beat official q4, so future calibration must target acceptance gaps with stronger evidence.",
        },
        {
            "id": "mlx-vlm-mtp-loop-overhead",
            "lane": "runtime-overhead",
            "cause": "Python loop, cache rollback, or verification synchronization may dominate per-token cost.",
            "proposed_change": "Profile exact MTP loop overhead before proposing a local or upstream patch.",
            "expected_metric_delta": "Reduce per-token overhead enough to raise decode_tps without changing model outputs.",
            "expected": "Recover speed if the current MLX/VLM MTP loop spends too much time on verification, cache rollback, or synchronization.",
            "math": "Per-token cost C = C_target_verify/k + C_draft + C_cache_rollback + C_python_loop; reduce the largest measured term.",
            "prototype": "Inspect one exact MTP loop source/log at a time and propose a minimal upstreamable or local patch only if timing evidence supports it.",
            "risk": "Runtime loop changes can destabilize streaming, tool parsing, or memory; require tests and easy rollback.",
            "rollback": "Revert runtime loop patches if replay checks, streaming, or decode benchmarks regress.",
            "evidence": "The desired 30+ tok/s requires either much higher acceptance or lower MTP overhead than the current live path.",
        },
        {
            "id": "dflash-janq-compatibility",
            "lane": "frontier-dflash",
            "cause": "DFlash reports large speculative speedups for standard Gemma4 targets, but its MLX loop hooks hidden layers on an mlx_lm-style target while OpenClaw uses a JANG-loaded mlx_vlm Gemma4 target.",
            "proposed_change": "Create a compatibility spike before any live runtime change: prove the DFlash draft can bind to the JANG target, capture Gemma4 hidden layers, and stream through OpenClaw guardrails.",
            "expected_metric_delta": "Potentially raise decode_tps beyond the current MTP assistant drafter if DFlash acceptance and block drafting outweigh target verification overhead.",
            "expected": "Either produce a safe DFlash canary plan for the TUI path or reject DFlash for JANQ with a precise incompatibility reason.",
            "math": "Speculative speedup depends on accepted_tokens_per_round / (draft_cost + verify_cost + rollback_cost); DFlash only helps if its block diffusion draft has higher accepted tokens than the current MTP assistant path.",
            "prototype": "Inspect `dflash.model_mlx.stream_generate`, add a no-load structural compatibility checklist for mlx_vlm Gemma4/JANG, then run a separate-env canary only if memory gates are green.",
            "risk": "The published Gemma4 DFlash drafter is paired with google/gemma-4-31B-it, not the unlocked JANQ target; mismatch can reduce acceptance, leak reasoning markers, or destabilize cache rollback.",
            "rollback": "Keep the existing Gemma4 MTP assistant drafter unless DFlash beats it on paired TUI decode benchmarks and passes stream/tool/reasoning guards.",
            "evidence": "DFlash README lists an MLX path and `z-lab/gemma-4-31B-it-DFlash`; the model card says it must be paired with `google/gemma-4-31B-it`, so JANQ compatibility must be proven rather than assumed.",
        },
    ]


def implementation_candidate_tasks(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    decode = mean_value(float_values(rows, "decode-sample", "decode_tps"))
    speed_gap = (
        f"Current measured decode is {decode} tok/s, below the practical 20 tok/s floor and far below the 50-70 tok/s frontier target."
        if decode is not None
        else "Decode speed has not been measured yet; measure it before enabling risky acceleration."
    )
    return [
        {
            "id": "mtp-acceptance-report",
            "status": "ready",
            "priority": 76,
            "lane": "production-mtp",
            "target": "openclaw-model-proxy.log",
            "hypothesis": "Decode tuning needs a compact report of MTP rounds, mean acceptance, block size, and drafter path from recent runs.",
            "metric": "mean_accept",
            "guard_checks": ["one_narrow_tool", "no_loop", "no_opencode_changes"],
            "next_action": "tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log",
        },
        {
            "id": "implement-mtp-acceptance-report",
            "status": "ready",
            "priority": 74,
            "lane": "production-mtp",
            "task_type": "supervisor",
            "supervisor_action": "mtp-report",
            "target": "openclaw/openclaw-speed-research.py",
            "source_files": ["openclaw/openclaw-speed-research.py", "openclaw/test-speed-research.py"],
            "hypothesis": "A benchmark-side MTP acceptance report will make decode research deterministic instead of relying on ad hoc log reading.",
            "metric": "mean_accept",
            "guard_checks": ["tests_pass", "no_opencode_changes", "no_live_profile_change"],
            "acceptance": "Focused tests pass and decode benchmark artifacts include MTP acceptance fields when logs expose them.",
            "rollback": "Revert only the acceptance-report patch and record discard if artifacts become noisy or misleading.",
            "evidence": speed_gap,
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research mtp-report --lines 160",
        },
        {
            "id": "implement-drafter-sweep-plan",
            "status": "ready",
            "priority": 72,
            "lane": "production-mtp",
            "task_type": "supervisor",
            "supervisor_action": "drafter-sweep-run",
            "target": "openclaw/openclaw-speed-research.py",
            "source_files": ["openclaw/openclaw-speed-research.py", "openclaw/test-speed-research.py"],
            "hypothesis": "A bounded drafter block sweep can search decode speed safely without manual overnight babysitting.",
            "metric": "decode_tps",
            "guard_checks": ["memory_gate", "bounded_trials", "tests_pass", "no_model_change", "restore_live_profile", "no_opencode_changes"],
            "acceptance": "A paired sweep artifact records control and variant decode TPS, MTP acceptance, promotion decision, and rollback policy.",
            "rollback": "Keep the current live block size unless a variant beats the promotion gate.",
            "evidence": speed_gap,
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research drafter-sweep-run --blocks 1,2,3,4",
        },
        {
            "id": "janq-dflash-drafter-fit-plan",
            "status": "ready",
            "priority": 70,
            "lane": "drafter-alignment",
            "task_type": "supervisor",
            "supervisor_action": "drafter-fit-plan",
            "target": "/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json",
            "source_files": ["openclaw/openclaw-drafter-fit.py", "openclaw/test-drafter-fit.py"],
            "hypothesis": "DFlash speedups require a drafter fitted to the exact JANQ target distribution, not a generic standard-Gemma drafter.",
            "metric": "drafter_fit_gate",
            "guard_checks": [
                "no_model_load",
                "target_config_match",
                "tool_thinking_replay_required",
                "no_live_profile_change",
                "no_opencode_changes",
            ],
            "acceptance": "A fit plan exists and requires target-generated JANQ traces plus promotion gates before DFlash can become TUI default.",
            "rollback": "Keep normal MTP as default unless a candidate beats the promotion gate.",
            "evidence": speed_gap,
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-drafter-fit plan",
        },
        {
            "id": "implement-janq-drafter-calibration-gate",
            "status": "ready",
            "priority": 68,
            "lane": "drafter-alignment",
            "task_type": "supervisor",
            "supervisor_action": "focused-test",
            "target": "openclaw/openclaw-mtp-drafter-calibrate.py",
            "source_files": ["openclaw/openclaw-mtp-drafter-calibrate.py", "openclaw/test-speed-research.py"],
            "hypothesis": "JANQ drafter calibration must be gated by decode TPS and acceptance improvements, not loss-only improvements.",
            "metric": "decode_tps_delta",
            "guard_checks": ["same_tokenizer", "no_reasoning_leak", "no_model_change", "tests_pass", "no_opencode_changes"],
            "acceptance": "The calibrator records pass/fail evidence against the official q4 drafter and refuses promotion unless wall-clock decode TPS improves.",
            "rollback": "Remove the gate if it blocks valid calibration or cannot compare against baseline safely.",
            "evidence": speed_gap,
            "next_action": "python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py",
        },
        {
            "id": "dflash-janq-compatibility-spike",
            "status": "ready",
            "priority": 64,
            "lane": "frontier-dflash",
            "task_type": "supervisor",
            "supervisor_action": "dflash-compatibility-gate",
            "target": "dflash.model_mlx/openclaw-jang-vlm-server.py",
            "source_files": ["openclaw/openclaw-jang-vlm-server.py", "openclaw/test-speed-research.py"],
            "hypothesis": "DFlash can only improve TUI decode speed if its MLX draft loop can wrap the JANQ-loaded mlx_vlm Gemma4 target without bypassing OpenClaw guardrails.",
            "metric": "compatibility_decision_then_decode_tps",
            "guard_checks": [
                "no_model_load",
                "no_live_profile_change",
                "separate_env",
                "stream_guard",
                "no_reasoning_leak",
                "no_opencode_changes",
            ],
            "acceptance": "Record a deterministic keep/discard/blocked decision with exact compatibility evidence before any DFlash install or live model benchmark.",
            "rollback": "No live rollback needed; this task must not change the active model profile or server path.",
            "evidence": speed_gap,
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research dflash-compatibility-gate",
        },
    ]


def synthesize(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    ensure_lane_contracts(root)
    compact_workspace(root)
    rows = result_rows(root)
    ideas = synthesis_ideas(rows)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    idea_lines = [
        "# Speed Research Ideas",
        "",
        f"## Autopilot Synthesis {generated_at}",
        "",
        "The benchmark queue was exhausted, so the loop switched from measurement to synthesis instead of repeating generic benchmarks.",
        "",
    ]
    for index, idea in enumerate(ideas, start=1):
        quality = score_insight(idea)
        idea_lines.extend(
            [
                f"### {index}. {idea['id']}",
                "",
                f"- lane: {idea['lane']}",
                f"- quality score: {quality['score']}/{quality['threshold']} passed={str(quality['passed']).lower()}",
                f"- cause: {idea['cause']}",
                f"- proposed change: {idea['proposed_change']}",
                f"- expected metric delta: {idea['expected_metric_delta']}",
                f"- expected impact: {idea['expected']}",
                f"- mathematical handle: {idea['math']}",
                f"- smallest prototype: {idea['prototype']}",
                f"- reliability risk: {idea['risk']}",
                f"- rollback: {idea['rollback']}",
                f"- evidence: {idea['evidence']}",
                "",
            ]
        )
    implementation_tasks = filter_seedable_tasks(root, implementation_candidate_tasks(rows))
    idea_lines.extend(
        [
            "## Implementation Candidates",
            "",
            "These are small, gated source-change candidates created from the synthesis. They are not live-setting changes by themselves.",
            "",
        ]
    )
    for task in implementation_tasks:
        if task.get("task_type") not in {"implementation", "supervisor"}:
            continue
        idea_lines.extend(
            [
                f"### {task['id']}",
                "",
                f"- target: {task['target']}",
                f"- hypothesis: {task['hypothesis']}",
                f"- metric: {task['metric']}",
                f"- acceptance: {task['acceptance']}",
                f"- rollback: {task['rollback']}",
                "",
            ]
        )
    ideas_path = root / "ideas.md"
    ideas_path.write_text("\n".join(idea_lines).rstrip() + "\n", encoding="utf-8")

    strategy_note = "\n".join(
        [
            "## Current Synthesis",
            "",
            f"- generated_at: {generated_at}",
            "- measurement loop is healthy, but exhausted queues must switch to decode/MTP ideas, ranked hypotheses, and implementation candidates.",
            "- top production idea: MTP acceptance bottleneck report.",
            "- top sweep idea: drafter block/quantization comparison with fixed prompt set and rollback.",
            "- top frontier idea: compare JANQ-specific drafter alignment with a guarded DFlash compatibility spike; promote only if wall-clock TUI decode TPS improves.",
            "",
        ]
    )
    upsert_section(root / "STRATEGY.md", "Current Synthesis", strategy_note)

    candidate_tasks = [
        {
            "id": "post-mtp-acceptance-report",
            "status": "ready",
            "priority": 79,
            "lane": "production-mtp",
            "target": "openclaw-model-proxy.log",
            "hypothesis": "Every decode-speed change needs recent MTP acceptance evidence before implementation.",
            "metric": "mean_accept",
            "guard_checks": ["one_narrow_tool", "no_loop"],
            "next_action": "tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log",
        },
        {
            "id": "decode-sample-baseline",
            "status": "ready",
            "priority": 78,
            "lane": "production-mtp",
            "target": "decode-sample",
            "hypothesis": "Decode sample speed is required before judging 18, 20, 30, or 50-70 tok/s targets.",
            "metric": "decode_tps",
            "benchmark_mode": "decode-sample",
            "guard_checks": ["no_reasoning_leak", "memory_ok"],
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
        },
        {
            "id": "decode-sample-repeatability",
            "status": "ready",
            "priority": 66,
            "lane": "production-mtp",
            "target": "decode-sample",
            "hypothesis": "Decode TPS must be repeatable across comparable normal prompts before promoting any drafter change.",
            "metric": "decode_tps",
            "benchmark_mode": "decode-sample",
            "guard_checks": ["no_reasoning_leak", "memory_ok"],
            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample",
        },
        *implementation_tasks,
    ]
    seeded = upsert_tasks(
        root,
        candidate_tasks,
    )
    deliberate_tasks: list[dict[str, Any]] = []
    contract_tasks: list[dict[str, Any]] = []
    expansion_tasks: list[dict[str, Any]] = []
    deliberation_report: dict[str, Any] = {}
    deliberation_tasks: list[dict[str, Any]] = []
    if seeded == 0:
        deliberate_tasks = filter_seedable_tasks(root, synthesis_deliberate_action_tasks(root, rows, int(time.time())))
        seeded = upsert_tasks(root, deliberate_tasks) if deliberate_tasks else 0
    if seeded == 0:
        expansion_tasks = frontier_expansion_tasks(root, rows, int(time.time()))
        seeded = upsert_tasks(root, expansion_tasks) if expansion_tasks else 0
    if seeded == 0:
        deliberation_report, deliberation_tasks = frontier_agent_deliberation(root, rows, int(time.time()))
        seeded = upsert_tasks(root, deliberation_tasks) if deliberation_tasks else 0
        if deliberation_report:
            path = root / "benchmarks" / f"frontier-agent-deliberation-{deliberation_report['timestamp']}.json"
            deliberation_report["seeded"] = seeded
            path.write_text(json.dumps(deliberation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if seeded == 0:
        contract_tasks = lane_contract_fallback_tasks(
            root,
            rows,
            int(time.time()),
            reason="Synthesis had no seedable implementation or deliberate tasks",
        )
        seeded = upsert_tasks(root, contract_tasks) if contract_tasks else 0
    gepa_report: dict[str, Any] = {}
    gepa_action = ""
    if seeded == 0:
        gepa_report = gepa_escalation_report(
            root,
            recent_rows=160,
            min_blocked=1,
            min_rework=1,
            min_trajectory=1,
            min_low_quality=1,
        )
        if seed_gepa_canary_task(root, gepa_report):
            gepa_action = str((gepa_report.get("candidate") or {}).get("id", ""))
            seeded = 1
    terminal_no_work = seeded == 0
    status = "keep" if seeded else "discard"
    result_target = "synthesis" if seeded else "synthesis-terminal"
    result_hypothesis = (
        "exhausted benchmark queues must generate ranked speed ideas and next tasks"
        if seeded
        else "synthesis reached a clean terminal no-work state after deterministic fallbacks"
    )
    progress_note = (
        "seeded measurable follow-up tasks"
        if seeded
        else "found no safe follow-up task after deterministic fallbacks; supervisor should pause instead of reseeding noise"
    )
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": generated_at,
            "task_id": "synthesize-speed-ideas",
            "finding": f"benchmark queue exhausted; synthesized ranked speed ideas and {progress_note}",
            "ideas": [
                {
                    "id": idea["id"],
                    "quality": score_insight(idea),
                }
                for idea in ideas
            ],
            "implementation_candidates": [
                task["id"]
                for task in candidate_tasks
                if task.get("task_type") in {"implementation", "supervisor"}
            ],
            "deliberate_actions": [task["id"] for task in deliberate_tasks],
            "contract_actions": [task["id"] for task in contract_tasks],
            "frontier_expansion_actions": [task["id"] for task in expansion_tasks],
            "deliberation_actions": [task["id"] for task in deliberation_tasks],
            "deliberation_report": deliberation_report,
            "gepa_action": gepa_action,
            "gepa_report": gepa_report,
            "seeded_tasks": seeded,
            "kind": args.kind,
        },
    )
    append_result(
        root,
        run_id=f"synthesis-{int(time.time())}",
        status=status,
        target=result_target,
        hypothesis=result_hypothesis,
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"ideas={len(ideas)} seeded_tasks={seeded} kind={args.kind} "
            f"deliberate_actions={','.join(task['id'] for task in deliberate_tasks)} "
            f"contract_actions={','.join(task['id'] for task in contract_tasks)} "
            f"frontier_expansion_actions={','.join(task['id'] for task in expansion_tasks)} "
            f"deliberation_actions={','.join(task['id'] for task in deliberation_tasks)} "
            f"gepa_action={gepa_action} terminal_no_work={terminal_no_work}"
        ),
    )
    print(
        json.dumps(
            {
                "ok": bool(seeded),
                "ideas": len(ideas),
                "seeded_tasks": seeded,
                "deliberate_actions": [task["id"] for task in deliberate_tasks],
                "contract_actions": [task["id"] for task in contract_tasks],
                "frontier_expansion_actions": [task["id"] for task in expansion_tasks],
                "deliberation_actions": [task["id"] for task in deliberation_tasks],
                "gepa_action": gepa_action,
                "status": status,
                "terminal_no_work": terminal_no_work,
                "ideas_path": str(ideas_path),
            },
            indent=2,
        )
    )
    return 0 if seeded else 2


def implementation_handoff_audit(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    ensure_lane_contracts(root)
    rows = result_rows(root)
    tasks = read_jsonl(root / "tasks.jsonl")
    ready = [task for task in tasks if task.get("status", "ready") in {"ready", "rework"}]
    candidates = [
        task
        for task in implementation_candidate_tasks(rows)
        if task.get("task_type") in {"implementation", "supervisor"}
    ]
    deterministic_ready = [task for task in ready if is_deterministic_research_task(task)]
    timestamp = int(time.time())
    seeded_bridge = False
    seeded_prerequisite = False
    seeded_fallback = False
    seeded_expansion = False
    seeded_deliberation = False
    deliberation_attempts: list[dict[str, Any]] = []
    terminal_handoff_exhausted = False
    bridge_zero = recent_empty_bridge_rows(root, rows, recent_rows=120)
    bridge_only_ready = bool(deterministic_ready) and all(is_implementation_bridge_task(task) for task in deterministic_ready)
    canonical = canonical_autoresearch_state(root, recent_rows=120)
    canonical_state = str(canonical.get("state", ""))

    def seed_deliberation_once() -> bool:
        nonlocal seeded_deliberation, seeded_prerequisite
        deliberation_report, deliberation_tasks = frontier_agent_deliberation(root, rows, timestamp)
        seeded_deliberation = bool(upsert_tasks(root, deliberation_tasks))
        gates = deliberation_report.get("gates", {}) if isinstance(deliberation_report, dict) else {}
        deliberation_attempts.append(
            {
                "ok": bool(deliberation_report.get("ok")) if isinstance(deliberation_report, dict) else False,
                "seeded": seeded_deliberation,
                "task_count": len(deliberation_tasks),
                "gates": gates,
                "selected_task_id": str(
                    (deliberation_report.get("architect", {}) if isinstance(deliberation_report, dict) else {}).get(
                        "selected_task_id", ""
                    )
                ),
            }
        )
        evidence = deliberation_report.get("evidence", {}) if isinstance(deliberation_report, dict) else {}
        active_noise = evidence.get("active_noise", {}) if isinstance(evidence, dict) else {}
        memory_noise = int(active_noise.get("memory_blocks", 0) or 0) if isinstance(active_noise, dict) else 0
        can_seed_repair_contract = bool(gates.get("zero_unsafe_noise")) or memory_noise == 0
        if not seeded_deliberation and can_seed_repair_contract:
            recovery_task = agent_deliberation_task(
                timestamp,
                slug="handoff-recovery-contract",
                priority=94,
                target="openclaw/openclaw-speed-research.py",
                hypothesis=(
                    "Implementation handoff reached a no-ready-work or repair-needed terminal state after known lanes "
                    "were exhausted; create one canary-only recovery contract that routes active non-memory noise "
                    "before any implementation."
                ),
                acceptance=(
                    "A recovery contract records the exhausted lane evidence, source-scout requirement, acceptance gates, "
                    "active noise classification, and rollback path; it cannot mutate source, model profile, or runtime by itself."
                ),
                evidence=evidence,
            )
            recovery_task["id"] = unique_task_id(root, str(recovery_task.get("id", "")))
            seeded_deliberation = bool(upsert_tasks(root, [recovery_task]))
            deliberation_report["recovery_task_seeded"] = int(seeded_deliberation)
            deliberation_attempts[-1]["recovery_task_seeded"] = seeded_deliberation
            deliberation_attempts[-1]["recovery_task_id"] = str(recovery_task.get("id", ""))
            if seeded_deliberation:
                deliberation_report["seeded_tasks"] = [str(recovery_task.get("id", ""))]
        seeded_prerequisite = seeded_prerequisite or seeded_deliberation
        if deliberation_report:
            path = root / "benchmarks" / f"frontier-agent-deliberation-{timestamp}.json"
            deliberation_report["seeded"] = int(seeded_deliberation)
            path.write_text(json.dumps(deliberation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return seeded_deliberation

    if not deterministic_ready or (bridge_only_ready and bridge_zero):
        if bridge_zero:
            prerequisite_tasks = concrete_handoff_prerequisite_tasks(root, rows, timestamp)
            seeded_prerequisite = bool(upsert_tasks(root, prerequisite_tasks))
            if not seeded_prerequisite:
                expansion_tasks = frontier_expansion_tasks(root, rows, timestamp)
                seeded_expansion = bool(upsert_tasks(root, expansion_tasks))
                seeded_prerequisite = seeded_expansion
            if not seeded_prerequisite:
                seed_deliberation_once()
            if not seeded_prerequisite:
                fallback_tasks = lane_contract_fallback_tasks(
                    root,
                    rows,
                    timestamp,
                    reason="Implementation handoff found an empty bridge and no concrete handoff prerequisite",
                )
                seeded_fallback = bool(upsert_tasks(root, fallback_tasks))
                seeded_prerequisite = seeded_fallback
            terminal_handoff_exhausted = not seeded_prerequisite
        elif canonical_state in {"blocked_until_external_change", "plateau_detected", "prerequisite_needed"}:
            prerequisite_tasks = concrete_handoff_prerequisite_tasks(root, rows, timestamp)
            seeded_prerequisite = bool(upsert_tasks(root, prerequisite_tasks))
            if not seeded_prerequisite:
                expansion_tasks = frontier_expansion_tasks(root, rows, timestamp)
                seeded_expansion = bool(upsert_tasks(root, expansion_tasks))
                seeded_prerequisite = seeded_expansion
            if not seeded_prerequisite:
                seed_deliberation_once()
            if not seeded_prerequisite:
                fallback_tasks = lane_contract_fallback_tasks(
                    root,
                    rows,
                    timestamp,
                    reason=f"Implementation handoff reached canonical {canonical_state} without ready work",
                )
                seeded_fallback = bool(upsert_tasks(root, fallback_tasks))
                seeded_prerequisite = seeded_fallback
            terminal_handoff_exhausted = not seeded_prerequisite
        else:
            seeded_bridge = bool(
                upsert_tasks(
                    root,
                    [
                        {
                            "id": f"handoff-audit-deterministic-bridge-{timestamp}",
                            "status": "ready",
                            "priority": 98,
                            "lane": "implementation-gate",
                            "task_type": "supervisor",
                            "supervisor_action": "implementation-bridge",
                            "target": "openclaw/openclaw-speed-research.py",
                            "source_files": ["openclaw/openclaw-speed-research.py", "openclaw/test-speed-research.py"],
                            "hypothesis": "A failed handoff audit must seed one scoped deterministic bridge instead of another generic research loop.",
                            "metric": "decode_tps_delta",
                            "guard_checks": ["canary_only", "tests_pass", "no_opencode_changes", "rollback_path"],
                            "acceptance": "The bridge seeds or confirms deterministic supervisor tasks with clean contracts before implementation proceeds.",
                            "rollback": "Delete this queue task if it seeds no deterministic follow-up; no live profile or model setting is changed.",
                            "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research synthesize --kind frontier",
                        }
                    ],
                )
            )
        if not seeded_prerequisite and not seeded_bridge:
            seed_deliberation_once()
        tasks = read_jsonl(root / "tasks.jsonl")
        ready = [task for task in tasks if task.get("status", "ready") in {"ready", "rework"}]
        deterministic_ready = [task for task in ready if is_deterministic_research_task(task)]
    scoped_candidates = [
        task
        for task in candidates
        if task.get("source_files")
        and task.get("acceptance")
        and task.get("rollback")
        and "no_opencode_changes" in {str(item) for item in task.get("guard_checks", [])}
    ]
    contract_blockers = {
        str(task.get("id", "")): issues["blockers"]
        for task in [*ready, *candidates]
        if (issues := task_contract_issues(root, task)).get("blockers")
    }
    patch_template = {
        "id": "handoff-audit-patch-template",
        "status": "ready",
        "task_type": "supervisor",
        "supervisor_action": "patch-execute",
        "target": "openclaw/openclaw-speed-research.py",
        "hypothesis": "Patch executor must canary-test allowlisted source changes before promotion.",
        "metric": "decode_tps_delta",
        "patch_file": "/tmp/openclaw-audit.patch",
        "source_files": ["openclaw/openclaw-speed-research.py"],
        "tests": list(DEFAULT_PATCH_TESTS),
        "guard_checks": ["tests_pass", "no_opencode_changes", "rollback_path"],
        "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research patch-execute --patch-file /tmp/openclaw-audit.patch",
    }
    patch_template_contract = task_contract_issues(root, patch_template)
    secret_fixture = (
        "diff --git a/openclaw/sample.py b/openclaw/sample.py\n"
        "--- a/openclaw/sample.py\n"
        "+++ b/openclaw/sample.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+DUMMY_PASSWORD = 'placeholder-not-real-value-1234567890'\n"
    )
    secret_fixture_classification = classify_patch(secret_fixture, source_files=["openclaw/sample.py"])
    gates = {
        "deterministic_ready_task": bool(deterministic_ready) or terminal_handoff_exhausted,
        "implementation_candidates_present": len(candidates) >= 3,
        "scoped_candidates_have_guards": len(scoped_candidates) >= 3,
        "ready_contracts_clean": not contract_blockers,
        "patch_executor_contract_ready": not patch_template_contract.get("blockers"),
        "safe_patch_tests_allowlisted": all(command in set(DEFAULT_PATCH_TESTS) for command in DEFAULT_PATCH_TESTS),
        "patch_executor_blocks_secret_content": not secret_fixture_classification["allowed"],
    }
    gaps = [name for name, ok in gates.items() if not ok]
    score = max(0, min(100, 100 - len(gaps) * 18 - min(30, len(contract_blockers) * 10)))
    ok = score >= int(args.min_score)
    report = {
        "ok": ok,
        "kind": "implementation-handoff-audit",
        "timestamp": timestamp,
        "score": score,
        "min_score": int(args.min_score),
        "gates": gates,
        "gaps": gaps,
        "ready_deterministic_tasks": [str(task.get("id", "")) for task in deterministic_ready[:12]],
        "seeded_bridge": seeded_bridge,
        "seeded_prerequisite": seeded_prerequisite,
        "seeded_fallback": seeded_fallback,
        "seeded_expansion": seeded_expansion,
        "seeded_deliberation": seeded_deliberation,
        "deliberation_attempts": deliberation_attempts,
        "terminal_handoff_exhausted": terminal_handoff_exhausted,
        "recent_empty_bridges": len(bridge_zero),
        "implementation_candidates": [str(task.get("id", "")) for task in candidates],
        "scoped_candidates": [str(task.get("id", "")) for task in scoped_candidates],
        "contract_blockers": contract_blockers,
        "patch_template_contract": patch_template_contract,
        "secret_fixture_classification": secret_fixture_classification,
        "next": (
            "continue_autopilot_loop"
            if ok and not terminal_handoff_exhausted
            else "external_refocus_or_wait_for_new_candidate"
            if terminal_handoff_exhausted
            else "run implementation bridge or repair task contracts before research"
        ),
    }
    path = root / "benchmarks" / f"implementation-handoff-audit-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "implementation-handoff-audit",
            "finding": "supervisor audited implementation handoff gates before allowing another research loop",
            "evidence": report,
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"implementation-handoff-audit-{timestamp}",
        status="keep" if ok else "blocked",
        target="autoresearch-implementation-handoff",
        hypothesis="Research findings should hand off into deterministic, scoped, canary-tested implementation tasks.",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"ok={ok} score={score} candidates={len(candidates)} scoped={len(scoped_candidates)} "
            f"ready_deterministic={len(deterministic_ready)} seeded_bridge={seeded_bridge} "
            f"seeded_prerequisite={seeded_prerequisite} empty_bridges={len(bridge_zero)} "
            f"seeded_fallback={seeded_fallback} seeded_expansion={seeded_expansion} "
            f"seeded_deliberation={seeded_deliberation} "
            f"terminal_handoff_exhausted={terminal_handoff_exhausted} "
            f"blockers={len(contract_blockers)}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2))
    return 0 if ok else 2


def drafter_bottleneck_review(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    ensure_lane_contracts(root)
    rows = result_rows(root)
    timestamp = int(time.time())
    state = drafter_bottleneck_state(root, rows, recent_rows=int(args.recent_rows))
    tasks = drafter_bottleneck_next_tasks(
        root,
        rows,
        timestamp,
        reason="supervisor drafter bottleneck review",
    )
    seeded = upsert_tasks(root, tasks) if tasks else 0
    status = "keep"
    if state["next_step"] == "continue_current_lane_contract":
        status = "blocked"
    elif state["next_step"] == "wait_for_trace_distillation_result":
        status = "keep"
    report = {
        "ok": status == "keep",
        "kind": "drafter-bottleneck-review",
        "timestamp": timestamp,
        "state": state,
        "seeded_tasks": seeded,
        "task_ids": [str(task.get("id", "")) for task in tasks],
        "next": state["next_step"],
    }
    path = root / "benchmarks" / f"drafter-bottleneck-review-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "drafter-bottleneck-review",
            "finding": "supervisor classified the JANQ drafter bottleneck and seeded exactly one deterministic next route",
            "evidence": report,
            "next": state["next_step"],
        },
    )
    append_result(
        root,
        run_id=f"drafter-bottleneck-review-{timestamp}",
        status=status,
        target="janq-drafter-bottleneck",
        hypothesis="the drafter lane should advance through a canonical state machine instead of retrying blocked calibration",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"state={state['state']} next_step={state['next_step']} seeded_tasks={seeded} "
            f"terminal_blocks={state['terminal_block_count']} fallback_decode_count={state['fallback_decode_count']}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0 if status == "keep" else 2


def drafter_adapter_method_contract(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    timestamp = int(time.time())
    state = drafter_bottleneck_state(root, recent_rows=int(args.recent_rows))
    source_files = [
        "openclaw/openclaw-mtp-drafter-calibrate.py",
        "openclaw/test-mtp-drafter-calibrate-guards.py",
        "openclaw/test-speed-research.py",
    ]
    terminal_routed = state["state"] in {
        "adapter_calibration_memory_blocked",
        "adapter_calibration_blocked",
        "adapter_calibration_attempted",
    }
    contract = {
        "ok": state["state"] in {
            "adapter_method_required",
            "adapter_method_contract_active",
            "adapter_method_contract_ready",
        },
        "kind": "drafter-adapter-method-contract",
        "timestamp": timestamp,
        "state": state,
        "allowed_source_files": source_files,
        "forbidden_changes": [
            "opencode",
            "live OpenClaw model profile",
            "target model id",
            "runtime server defaults",
            "secrets or env files",
        ],
        "implementation_target": (
            "add a canary-only adapter/logit-distillation calibration path that freezes the JANQ target "
            "and frozen quantized drafter weights; only newly introduced adapter/head parameters may be trainable"
        ),
        "acceptance": [
            "unit tests prove quantized drafter parameters remain blocked for direct training",
            "new adapter/logit method is selected explicitly and does not run by default",
            "no model load is required by the canary test",
            "normal OpenClaw TUI drafter is unchanged until paired decode TPS benchmarks pass",
        ],
        "promotion_gate": [
            "canary tests pass",
            "bounded calibration artifact shows acceptance lift",
            "paired normal TUI decode benchmark improves TPS",
            "TTFT and memory do not regress materially",
            "tool/reasoning/stream guards pass",
        ],
        "rollback": "discard adapter output and keep the current official MTP drafter/profile if any gate fails",
    }
    contract["terminal_routed"] = terminal_routed
    contract["status"] = "keep" if contract["ok"] or terminal_routed else "blocked"
    if terminal_routed and not contract["ok"]:
        contract["next"] = (
            "terminal adapter calibration blocker is already known; route to source-scout/frontier "
            "deliberation or a changed adapter/logit-distillation implementation contract instead of "
            "recording active quality debt"
        )
    path = root / "experiments" / f"drafter-adapter-method-contract-{timestamp}.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    should_seed_implementation = contract["ok"] or (
        terminal_routed
        and not active_task_has_prefix(root, "implementation-drafter-adapter-method-")
        and not recent_keep_result_has_prefix(root, "implementation-drafter-adapter-method-", recent_rows=120)
    )
    contract["seed_implementation"] = should_seed_implementation
    if should_seed_implementation:
        implementation_task = drafter_adapter_method_implementation_task(
            timestamp,
            task_id=unique_task_id(root, f"implementation-drafter-adapter-method-{timestamp}"),
            contract_path=str(path),
        )
        implementation_task["hypothesis"] = contract["implementation_target"]
        implementation_task["acceptance"] = "; ".join(contract["acceptance"])
        implementation_task["rollback"] = contract["rollback"]
        contract["seeded_implementation_tasks"] = upsert_tasks(root, [implementation_task])
    else:
        contract["seeded_implementation_tasks"] = 0
    append_jsonl(
        root / "experiments.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "drafter-adapter-method-contract",
            "status": contract["status"],
            "path": str(path),
            "state": state["state"],
        },
    )
    append_result(
        root,
        run_id=f"drafter-adapter-method-contract-{timestamp}",
        status=contract["status"],
        target="janq-drafter-adapter-method",
        hypothesis="the supervisor should convert repeated quantized-gradient failures into a constrained adapter/logit implementation contract",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=f"state={state['state']} ok={contract['ok']} terminal_routed={terminal_routed} path={path}",
    )
    print(json.dumps({"path": str(path), **contract}, indent=2, sort_keys=True))
    return 0 if contract["status"] == "keep" else 2


def environment_snapshot_command(args: argparse.Namespace) -> int:
    root = workspace_root()
    commit = current_commit(Path(args.repo or repo_root()))
    snapshot = environment_snapshot(root, label=args.label, commit=commit)
    append_result(
        root,
        run_id=f"environment-snapshot-{snapshot['timestamp']}",
        status="keep" if snapshot["evaluator_integrity"]["ok"] else "blocked",
        target="autoresearch-environment",
        hypothesis="each autonomous run should record the exact evaluator, profile, env, and commit it used",
        commit=commit,
        notes=(
            f"label={args.label} integrity_ok={snapshot['evaluator_integrity']['ok']} "
            f"immutable_changes={len(snapshot['evaluator_integrity']['immutable_changes'])} "
            f"path={snapshot['path']}"
        ),
    )
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0 if snapshot["evaluator_integrity"]["ok"] or args.allow_fail else 2


def evaluator_integrity_command(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    report = evaluator_integrity_report(root)
    timestamp = int(time.time())
    path = root / "benchmarks" / f"evaluator-integrity-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "evaluator-integrity",
            "finding": "supervisor checked that frozen evaluator files were not moved during autoresearch",
            "evidence": report,
            "next": "continue" if report["ok"] else "repair evaluator drift before research",
        },
    )
    append_result(
        root,
        run_id=f"evaluator-integrity-{timestamp}",
        status="keep" if report["ok"] else "blocked",
        target="autoresearch-evaluator-integrity",
        hypothesis="the autoresearch evaluator should stay frozen unless a gated policy route changes it",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"ok={report['ok']} immutable_changes={len(report['immutable_changes'])} "
            f"approval_required_changes={len(report['approval_required_changes'])} path={path}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0 if report["ok"] or args.allow_fail else 2


def plateau_pivot(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    rows = result_rows(root)[-max(1, int(args.recent_rows)) :]
    sweep_rows = [
        row
        for row in rows
        if row.get("status") == "keep" and row.get("run_id", "").startswith("drafter-sweep-run")
    ]
    sweep_fields = [parse_note_fields(row.get("notes", "")) for row in sweep_rows]
    keep_current = [fields for fields in sweep_fields if fields.get("decision") == "keep-current"]
    block2_wins = [fields for fields in sweep_fields if fields.get("winner_block") == "2"]
    decode_mean = latest_decode_mean(root, recent_rows=int(args.recent_rows))
    tasks = read_jsonl(root / "tasks.jsonl")
    ready_lanes = {
        str(task.get("lane", ""))
        for task in tasks
        if task.get("status", "ready") in {"ready", "rework"} and task.get("lane")
    }
    plateau = (
        len(sweep_rows) >= int(args.min_sweeps)
        and len(keep_current) >= int(args.min_sweeps)
        and len(block2_wins) >= int(args.min_sweeps)
        and decode_mean is not None
        and decode_mean < float(args.target_tps)
    )
    timestamp = int(time.time())
    seeded_tasks: list[dict[str, Any]] = []
    if plateau:
        if "drafter-alignment" not in ready_lanes and should_seed_action(root, "plateau-drafter-fit-", recent_rows=40):
            seeded_tasks.append(drafter_fit_task(timestamp, task_id=f"plateau-drafter-fit-{timestamp}", priority=97))
        if (
            "frontier-dflash" not in ready_lanes
            and not dflash_lane_is_blocked(root, recent_rows=80)
            and should_seed_action(root, "plateau-dflash-compat-", recent_rows=40)
        ):
            seeded_tasks.append(
                dflash_compatibility_task(timestamp, task_id=f"plateau-dflash-compat-{timestamp}", priority=95)
            )
        if "runtime-overhead" not in ready_lanes and should_seed_runtime_overhead_map(root, rows, recent_rows=45):
            seeded_tasks.append(
                {
                    "id": f"plateau-runtime-overhead-map-{timestamp}",
                    "status": "ready",
                    "priority": 94,
                    "lane": "runtime-overhead",
                    "task_type": "supervisor",
                    "supervisor_action": "runtime-overhead-map",
                    "target": "openclaw/openclaw-jang-vlm-server.py",
                    "hypothesis": "Block-size tuning plateaued below target, so map MTP verification/cache/rollback overhead before more sweeps.",
                    "metric": "decode_tps_delta",
                    "guard_checks": ["no_live_profile_change", "tests_before_patch", "no_opencode_changes"],
                    "acceptance": "A runtime-overhead artifact names whether overhead is in target eval, drafter eval, cache rollback, or proxy streaming.",
                    "rollback": "No live profile change; this is a read-only source/log mapper.",
                    "next_action": "/Users/kristian/.openclaw/bin/openclaw-speed-research runtime-overhead-map",
                }
            )
    seeded = upsert_tasks(root, seeded_tasks) if seeded_tasks else 0
    state = "pivot" if plateau else "continue-measurement"
    report = {
        "ok": True,
        "kind": "plateau-pivot",
        "timestamp": timestamp,
        "state": state,
        "plateau": plateau,
        "recent_rows": len(rows),
        "sweep_rows": len(sweep_rows),
        "keep_current_sweeps": len(keep_current),
        "block2_wins": len(block2_wins),
        "decode_mean_tps": decode_mean,
        "target_tps": float(args.target_tps),
        "ready_lanes": sorted(ready_lanes),
        "seeded_tasks": seeded,
        "next": (
            "route to drafter-fit, DFlash compatibility, or runtime-overhead instead of repeating block sweeps"
            if plateau
            else "continue bounded paired measurement until plateau evidence is strong enough"
        ),
    }
    path = root / "benchmarks" / f"plateau-pivot-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "plateau-pivot",
            "finding": "supervisor made the Karpathy keep/discard plateau decision explicit before selecting the next lane",
            "evidence": report,
            "next": report["next"],
        },
    )
    append_result(
        root,
        run_id=f"plateau-pivot-{timestamp}",
        status="keep",
        target="autoresearch-plateau-pivot",
        hypothesis="settled block-size evidence should pivot the loop to higher-upside lanes instead of repeating measurements",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"state={state} plateau={plateau} sweeps={len(sweep_rows)} "
            f"block2_wins={len(block2_wins)} decode_mean_tps={decode_mean if decode_mean is not None else ''} "
            f"seeded_tasks={seeded}"
        ),
    )
    print(json.dumps({"path": str(path), **report}, indent=2, sort_keys=True))
    return 0


def compact(args: argparse.Namespace) -> int:
    root = workspace_root()
    result = compact_workspace(root, recent_rows=args.recent_rows)
    print(json.dumps(result, indent=2))
    return 0


def print_manifest(_args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    print(json.dumps(load_benchmark_manifest(root), indent=2, sort_keys=True))
    return 0


def replay(args: argparse.Namespace) -> int:
    root = workspace_root()
    result = replay_checks(root)
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "replay-regression-checks",
            "finding": "supervisor replayed known autoresearch failure guards",
            "evidence": result,
            "next": "fix failing replay cases before running overnight",
        },
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] or args.allow_fail else 2


def paired_plan(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    tasks = read_jsonl(root / "tasks.jsonl")
    task = next((item for item in tasks if item.get("id") == args.task_id), None)
    if task is None:
        print(json.dumps({"ok": False, "reason": f"task not found: {args.task_id}"}, indent=2))
        return 2
    plan = paired_profile_plan(task)
    path = root / "experiments" / f"paired-profile-plan-{args.task_id}.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "experiments.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": args.task_id,
            "status": "paired-profile-plan",
            "path": str(path),
            "plan": plan,
        },
    )
    print(json.dumps({"ok": True, "path": str(path), "plan": plan}, indent=2))
    return 0


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise ValueError("values must be positive integers")
        if parsed not in items:
            items.append(parsed)
    return items


def parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def drafter_sweep_plan(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    blocks = parse_int_list(args.blocks)
    if not blocks:
        print(json.dumps({"ok": False, "reason": "no block sizes provided"}, indent=2))
        return 2
    timestamp = int(time.time())
    live_block = os.environ.get("OPENCLAW_JANG_DRAFT_BLOCK_SIZE", "2")
    prompt_set = [
        {
            "id": "normal-text",
            "mode": "decode-sample",
            "prompt_class": "normal-text",
            "max_tokens": 96,
        }
    ]
    plan = {
        "ok": True,
        "kind": "mtp-drafter-block-sweep",
        "status": "plan-only",
        "timestamp": timestamp,
        "control": {
            "draft_block_size": live_block,
            "samples": args.samples,
            "command": (
                "/Users/kristian/.openclaw/bin/openclaw-speed-research "
                f"benchmark --mode decode-sample --draft-block-size {live_block}"
            ),
        },
        "variants": [
            {
                "draft_block_size": block,
                "samples": args.samples,
                "command": (
                    "/Users/kristian/.openclaw/bin/openclaw-speed-research "
                    f"benchmark --mode decode-sample --draft-block-size {block}"
                ),
            }
            for block in blocks
        ],
        "prompt_set": prompt_set,
        "promotion_gate": {
            "min_decode_tps_delta": args.min_delta,
            "requires_usage_completion_tokens": True,
            "requires_mtp_mean_accept": True,
            "must_pass_replay": True,
            "must_not_change_live_profile": True,
        },
        "rollback": "No rollback needed for plan-only or per-request draft_block_size benchmarks; live profile is not mutated.",
    }
    path = root / "experiments" / f"mtp-drafter-sweep-plan-{timestamp}.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "experiments.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "drafter-block-sweep-plan",
            "status": "sweep-plan",
            "path": str(path),
            "blocks": blocks,
            "samples": args.samples,
            "min_delta": args.min_delta,
        },
    )
    print(json.dumps({"ok": True, "path": str(path), "plan": plan}, indent=2))
    return 0


def mean_float(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def drafter_sweep_run(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    blocks = parse_int_list(args.blocks)
    if not blocks:
        print(json.dumps({"ok": False, "reason": "no block sizes provided"}, indent=2))
        return 2
    samples = max(1, min(int(args.samples), 10))
    retries = max(0, min(int(getattr(args, "retries", 2)), 5))
    live_block = int(args.control_block or os.environ.get("OPENCLAW_JANG_DRAFT_BLOCK_SIZE", "2") or 2)
    trial_blocks = [live_block, *[block for block in blocks if block != live_block]]
    timestamp = int(time.time())
    trials: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    for block in trial_blocks:
        block_results: list[dict[str, Any]] = []
        for sample_index in range(samples):
            attempts: list[dict[str, Any]] = []
            parsed: dict[str, Any] = {}
            code = 2
            for attempt_index in range(retries + 1):
                capture = io.StringIO()
                bench_args = argparse.Namespace(
                    base_url=args.base_url,
                    model=args.model,
                    quick=False,
                    mode="decode-sample",
                    timeout=args.timeout,
                    draft_block_size=block,
                    record_schema_failures=False,
                )
                with contextlib.redirect_stdout(capture):
                    code = benchmark(bench_args)
                parsed = parse_json_object(capture.getvalue()) or {}
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "returncode": code,
                        "result": parsed,
                    }
                )
                if code == 0 and parsed.get("ok"):
                    break
                reason = str(parsed.get("reason") or "")
                if "completion_tokens=1" not in reason and "too short for throughput evidence" not in reason:
                    break
            block_results.append(
                {
                    "sample": sample_index + 1,
                    "attempts": attempts,
                    "returncode": code,
                    "result": parsed,
                }
            )
            if code != 0 or parsed.get("ok") is False:
                failures.append(
                    {
                        "block": block,
                        "sample": sample_index + 1,
                        "attempts": len(attempts),
                        "reason": parsed.get("reason") or f"benchmark exit {code}",
                    }
                )
                break
        good = [row["result"] for row in block_results if row.get("returncode") == 0 and row.get("result", {}).get("ok")]
        decode_values = [float(row["decode_tps"]) for row in good if row.get("decode_tps") not in {"", None}]
        accept_values = [
            float(row["mtp"]["mean_accept"])
            for row in good
            if isinstance(row.get("mtp"), dict) and row["mtp"].get("mean_accept") not in {"", None}
        ]
        round_values = [
            float(row["mtp"]["mtp_rounds"])
            for row in good
            if isinstance(row.get("mtp"), dict) and row["mtp"].get("mtp_rounds") not in {"", None}
        ]
        server_values = [
            float(row["mtp"]["server_tok_s"])
            for row in good
            if isinstance(row.get("mtp"), dict) and row["mtp"].get("server_tok_s") not in {"", None}
        ]
        trials[str(block)] = {
            "block": block,
            "requested_samples": samples,
            "retries_per_sample": retries,
            "sample_count": len(good),
            "mean_decode_tps": mean_float(decode_values),
            "mean_accept": mean_float(accept_values),
            "mean_mtp_rounds": mean_float(round_values),
            "mean_server_tok_s": mean_float(server_values),
            "failures": [failure for failure in failures if failure["block"] == block],
        }

    control = trials.get(str(live_block), {})
    valid_trials = [trial for trial in trials.values() if trial.get("mean_decode_tps") is not None]
    replay = replay_checks(root)
    if not valid_trials:
        decision = "blocked"
        winner = {}
        delta = None
        status = "blocked"
        reason = "no successful sweep benchmark samples"
    else:
        winner = max(valid_trials, key=lambda trial: float(trial["mean_decode_tps"]))
        control_tps = control.get("mean_decode_tps")
        delta = (
            round(float(winner["mean_decode_tps"]) - float(control_tps), 3)
            if control_tps is not None
            else None
        )
        passed_gate = (
            winner.get("block") != live_block
            and delta is not None
            and delta >= float(args.min_delta)
            and replay["ok"]
            and int(winner.get("sample_count") or 0) >= samples
        )
        decision = "promotion-ready" if passed_gate else "keep-current"
        status = "keep"
        reason = ""

    artifact = {
        "ok": status == "keep",
        "kind": "mtp-drafter-block-sweep-run",
        "timestamp": timestamp,
        "control_block": live_block,
        "blocks": trial_blocks,
        "samples": samples,
        "min_delta": float(args.min_delta),
        "trials": trials,
        "winner": winner,
        "delta_vs_control": delta,
        "decision": decision,
        "replay": replay,
        "promotion_gate": {
            "min_decode_tps_delta": float(args.min_delta),
            "must_restore_live_profile": True,
            "must_pass_replay": True,
            "must_keep_model_id": True,
            "requires_mtp_acceptance": True,
        },
        "rollback": "No live rollback needed: every variant used per-request draft_block_size and did not mutate the active profile.",
        "failures": failures,
    }
    path = root / "experiments" / f"mtp-drafter-sweep-run-{timestamp}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "experiments.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "drafter-block-sweep-run",
            "status": "sweep-run" if status == "keep" else "blocked",
            "path": str(path),
            "decision": decision,
            "winner": winner,
            "delta_vs_control": delta,
        },
    )
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": "drafter-block-sweep-run",
            "finding": "supervisor executed paired drafter block sweep instead of generating another plan",
            "evidence": {
                "path": str(path),
                "decision": decision,
                "winner": winner,
                "delta_vs_control": delta,
            },
            "next": "promote only if decision=promotion-ready and replay guards pass",
        },
    )
    append_result(
        root,
        run_id=f"drafter-sweep-run-{timestamp}",
        status=status,
        target="OPENCLAW_JANG_DRAFT_BLOCK_SIZE",
        hypothesis="paired drafter block-size sweep should find a faster safe MTP setting",
        decode_tps=winner.get("mean_decode_tps", "") if winner else "",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"decision={decision} control_block={live_block} "
            f"winner_block={winner.get('block', '') if winner else ''} "
            f"delta_vs_control={delta if delta is not None else ''} "
            f"path={path} reason={reason}"
        ),
    )
    print(json.dumps({"ok": status == "keep", "path": str(path), **artifact}, indent=2))
    return 0 if status == "keep" else 2


def mtp_report(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    source = Path(args.log_path).expanduser() if args.log_path else log_path()
    try:
        text = "\n".join(source.read_text(encoding="utf-8", errors="replace").splitlines()[-args.lines :])
    except OSError as error:
        print(json.dumps({"ok": False, "reason": str(error)}, indent=2))
        return 2
    summary = parse_generation_log_summary(text)
    summary["log_path"] = str(source)
    summary["lines"] = args.lines
    summary["timestamp"] = int(time.time())
    path = root / "benchmarks" / f"mtp-acceptance-report-{summary['timestamp']}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_result(
        root,
        run_id=f"mtp-report-{summary['timestamp']}",
        status="keep" if summary["ok"] else "blocked",
        target="mtp-acceptance-report",
        hypothesis="recent OpenClaw server logs should expose drafter acceptance evidence for decode tuning",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=(
            f"samples={summary['sample_count']} mtp_samples={summary['mtp_sample_count']} "
            f"mean_server_tok_s={summary['mean_server_tok_s']} mean_accept={summary['mean_accept']} "
            f"path={path}"
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


def repo_root() -> Path:
    return Path(
        os.environ.get(
            "OPENCLAW_SPEED_RESEARCH_REPO",
            "/Users/kristian/Documents/openclaw-harness-autoresearch",
        )
    ).expanduser()


def normalize_patch_path(path: str) -> str:
    cleaned = path.strip()
    if cleaned == "/dev/null":
        return cleaned
    for prefix in ("a/", "b/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned


def patch_files(patch_text: str) -> list[str]:
    files: set[str] = set()
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            for item in parts[2:4]:
                path = normalize_patch_path(item)
                if path != "/dev/null":
                    files.add(path)
        elif line.startswith("+++ ") or line.startswith("--- "):
            path = normalize_patch_path(line[4:].split("\t", 1)[0])
            if path != "/dev/null":
                files.add(path)
    return sorted(files)


def patch_secret_findings(patch_text: str) -> list[str]:
    findings: list[str] = []
    for line in patch_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:]
        for pattern in SECRET_PATCH_PATTERNS:
            match = pattern.search(added)
            if match:
                findings.append(match.group(1) if match.groups() else "secret-pattern")
                break
    return findings


def git_dirty_files(repo: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ["<git-status-timeout>"]
    if result.returncode != 0:
        return ["<git-status-unavailable>"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def classify_patch(
    patch_text: str,
    *,
    source_files: list[str] | None = None,
    allow_architectural: bool = False,
) -> dict[str, Any]:
    files = patch_files(patch_text)
    additions = 0
    deletions = 0
    reasons: list[str] = []
    secret_findings = patch_secret_findings(patch_text)
    source_allowlist = {normalize_patch_path(path) for path in source_files or [] if path}
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    if not files:
        reasons.append("patch has no file changes")
    if secret_findings:
        reasons.append("patch adds possible secret material")
    for path in files:
        if path.startswith("/") or ".." in Path(path).parts:
            reasons.append(f"unsafe path: {path}")
        lowered = path.lower()
        if any(fragment in lowered for fragment in DENIED_PATCH_FRAGMENTS):
            reasons.append(f"denied path fragment: {path}")
        if not path.startswith(ALLOWED_PATCH_PREFIXES):
            reasons.append(f"path outside allowlist: {path}")
        if source_allowlist and path not in source_allowlist:
            reasons.append(f"path outside task source_files: {path}")
    changed_lines = additions + deletions
    architectural = any(path.startswith(ARCHITECTURAL_PATCH_PREFIXES) for path in files)
    high_risk = any(path.startswith(HIGH_RISK_PATCH_PREFIXES) for path in files)
    if len(files) > 3 or changed_lines > 220:
        architectural = True
        high_risk = True
    if architectural and not allow_architectural:
        reasons.append("architectural change requires canary evidence and explicit allow_architectural")
    destructive = bool(reasons) or deletions > additions * 3 + 20
    if destructive:
        impact = "destructive"
    elif architectural:
        impact = "architectural"
    elif high_risk:
        impact = "high-risk"
    elif changed_lines <= 80 and len(files) <= 2:
        impact = "safe"
    else:
        impact = "moderate"
    auto_promote = impact in {"safe", "moderate"} and not high_risk
    return {
        "files": files,
        "additions": additions,
        "deletions": deletions,
        "changed_lines": changed_lines,
        "impact": impact,
        "architectural": architectural,
        "high_risk": high_risk,
        "crabbox_required": architectural or high_risk,
        "approval_required": impact == "architectural",
        "auto_promote": auto_promote and not destructive,
        "allowed": not destructive,
        "reasons": reasons,
        "secret_findings": secret_findings[:6],
    }


def architectural_approval_granted(task_id: str, approval_file: str) -> bool:
    if not approval_file:
        return False
    try:
        text = Path(approval_file).expanduser().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    task = task_id or "manual"
    accepted = {
        "APPROVE_ARCHITECTURAL_PATCH=*",
        f"APPROVE_ARCHITECTURAL_PATCH={task}",
        f"APPROVE_PATCH={task}",
        task,
    }
    return any(line.strip() in accepted for line in text.splitlines())


def run_test_command(command: str, *, cwd: Path, timeout: float) -> dict[str, Any]:
    allowed = set(DEFAULT_PATCH_TESTS)
    if command not in allowed:
        return {"ok": False, "command": command, "reason": "test command is not allowlisted"}
    try:
        result = subprocess.run(
            command.split(),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "command": command, "reason": "test timeout"}
    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "output_tail": result.stdout[-2000:],
    }


def patch_execute(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    repo = Path(args.repo or repo_root()).expanduser()
    patch_path = Path(args.patch_file).expanduser()
    if not patch_path.exists():
        result = {"ok": False, "reason": f"patch file not found: {patch_path}"}
        print(json.dumps(result, indent=2))
        return 2
    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    patch_hash = text_sha256(patch_text)
    source_files = [item.strip() for item in args.source_files.split(",") if item.strip()]
    classification = classify_patch(
        patch_text,
        source_files=source_files or None,
        allow_architectural=args.allow_architectural,
    )
    approval_file = str(getattr(args, "architectural_approval_file", "") or "")
    approval_granted = architectural_approval_granted(args.task_id or "manual", approval_file)
    crabbox_evidence = load_crabbox_evidence(
        str(getattr(args, "crabbox_evidence_file", "") or ""),
        patch_sha256=patch_hash,
    ) if classification.get("crabbox_required") else {}
    timestamp = int(time.time())
    task_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.task_id or "manual").strip("-") or "manual"
    artifact_id = f"{timestamp}-{task_slug}"
    canary_root = root / "canaries"
    canary_root.mkdir(parents=True, exist_ok=True)
    canary = canary_root / f"patch-{artifact_id}"
    artifact: dict[str, Any] = {
        "ok": False,
        "kind": "patch-executor",
        "timestamp": timestamp,
        "patch_file": str(patch_path),
        "patch_sha256": patch_hash,
        "repo": str(repo),
        "canary": str(canary),
        "classification": classification,
        "crabbox_evidence": crabbox_evidence,
        "architectural_approval_file": approval_file,
        "approval_required": bool(classification.get("approval_required")),
        "approval_granted": approval_granted,
        "rollback_rehearsal_ok": bool(getattr(args, "rollback_rehearsal_ok", False)),
        "promoted": False,
        "tests": [],
    }
    status = "blocked"
    reason = ""
    return_code = 2
    try:
        if not classification["allowed"]:
            reason = "patch failed allowlist or safety classification"
            artifact["reason"] = reason
        else:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "add", "--detach", str(canary), "HEAD"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=True,
            )
            check = subprocess.run(
                ["git", "-C", str(canary), "apply", "--check", str(patch_path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            artifact["apply_check"] = {"ok": check.returncode == 0, "stderr": check.stderr[-2000:]}
            if check.returncode != 0:
                reason = "patch did not apply cleanly in canary"
                artifact["reason"] = reason
            else:
                apply = subprocess.run(
                    ["git", "-C", str(canary), "apply", str(patch_path)],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
                artifact["canary_apply"] = {"ok": apply.returncode == 0, "stderr": apply.stderr[-2000:]}
                tests = [item.strip() for item in args.tests.split(";") if item.strip()] or list(DEFAULT_PATCH_TESTS)
                artifact["tests"] = [run_test_command(command, cwd=canary, timeout=args.test_timeout) for command in tests]
                tests_ok = all(item.get("ok") for item in artifact["tests"])
                can_promote = bool(classification["auto_promote"]) or bool(classification.get("architectural"))
                if apply.returncode != 0:
                    reason = "patch apply failed in canary"
                    artifact["reason"] = reason
                elif not tests_ok:
                    reason = "canary tests failed"
                    artifact["reason"] = reason
                elif classification.get("crabbox_required") and not crabbox_evidence.get("_valid_for_architectural_promotion"):
                    reason = "canary passed; high-risk promotion requires valid Crabbox sandbox evidence"
                    artifact["reason"] = reason
                    artifact["held_for_crabbox"] = True
                    artifact["crabbox_instruction"] = (
                        "Run the high-risk candidate in a static SSH Mac Crabbox sandbox and provide a fresh evidence JSON "
                        "with ok=true, runner=static-ssh-mac, matching patch_sha256, focused tests, full_suite.ok, "
                        "rollback_rehearsal_ok, and logs/run_id."
                    )
                    status = "blocked"
                    return_code = 0
                elif not can_promote or args.canary_only:
                    reason = "canary passed; promotion intentionally held"
                    artifact["reason"] = reason
                    status = "keep"
                    return_code = 0
                else:
                    dirty_files = git_dirty_files(repo)
                    artifact["main_repo_dirty_files"] = dirty_files[:20]
                    if dirty_files:
                        reason = "main repo has uncommitted changes; refusing autonomous promotion"
                        artifact["reason"] = reason
                        status = "blocked"
                        return_code = 2
                    else:
                        autonomy = frontier_autonomy_score_report(
                            root,
                            promotion=True,
                            classification=classification,
                            patch_tests=artifact["tests"],
                            crabbox_evidence=crabbox_evidence,
                            rollback_rehearsal_ok=bool(getattr(args, "rollback_rehearsal_ok", False)),
                        )
                        autonomy_path = write_frontier_autonomy_score(root, autonomy)
                        artifact["frontier_autonomy_score"] = {**autonomy, "path": str(autonomy_path)}
                        if not autonomy["ok"]:
                            reason = "frontier autonomy score blocked promotion"
                            artifact["reason"] = reason
                            artifact["quarantined"] = True
                            status = "blocked"
                            return_code = 2
                            raise StopIteration
                        main_check = subprocess.run(
                            ["git", "-C", str(repo), "apply", "--check", str(patch_path)],
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=30,
                            check=False,
                        )
                        artifact["main_apply_check"] = {"ok": main_check.returncode == 0, "stderr": main_check.stderr[-2000:]}
                        if main_check.returncode != 0:
                            reason = "patch no longer applies to main repo"
                            artifact["reason"] = reason
                        else:
                            main_apply = subprocess.run(
                                ["git", "-C", str(repo), "apply", str(patch_path)],
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=30,
                                check=False,
                            )
                            artifact["main_apply"] = {"ok": main_apply.returncode == 0, "stderr": main_apply.stderr[-2000:]}
                            artifact["promoted"] = main_apply.returncode == 0
                            status = "keep" if artifact["promoted"] else "blocked"
                            reason = "" if artifact["promoted"] else "main repo apply failed"
                            if artifact["promoted"]:
                                stable = {
                                    "timestamp": int(time.time()),
                                    "status": "stable",
                                    "commit": current_commit(repo),
                                    "repo": str(repo),
                                    "reason": f"patch-executor promoted {args.task_id or 'manual'} after frontier autonomy score",
                                    "patch_sha256": patch_hash,
                                    "frontier_autonomy_score": autonomy["total_score"],
                                    "score_artifact": str(autonomy_path),
                                }
                                append_jsonl(root / "stable-builds.jsonl", stable)
                            if reason:
                                artifact["reason"] = reason
                            return_code = 0 if artifact["promoted"] else 2
    except StopIteration:
        pass
    except subprocess.CalledProcessError as error:
        reason = f"git canary setup failed: {error.stderr[-500:] if error.stderr else error}"
        artifact["reason"] = reason
    except subprocess.TimeoutExpired:
        reason = "patch executor timeout"
        artifact["reason"] = reason
        return_code = 124
    finally:
        if not args.keep_canary and canary.exists():
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(canary)],
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
            artifact["canary_removed"] = True
    artifact["ok"] = return_code == 0
    path = root / "experiments" / f"patch-executor-{artifact_id}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        root / "experiments.jsonl",
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": args.task_id or "patch-executor",
            "status": "patch-executor",
            "path": str(path),
            "impact": classification["impact"],
            "promoted": artifact["promoted"],
        },
    )
    append_result(
        root,
        run_id=f"patch-executor-{artifact_id}",
        status=status,
        target="patch-executor",
        hypothesis=args.hypothesis or "canary-test allowlisted patch before promotion",
        commit=current_commit(repo),
        notes=(
            f"impact={classification['impact']} promoted={artifact['promoted']} "
            f"files={len(classification['files'])} reason={reason} path={path}"
        ),
    )
    print(json.dumps({"path": str(path), **artifact}, indent=2))
    return return_code


def benchmark_prompt(mode: str) -> tuple[str, int]:
    spec = benchmark_spec(workspace_root(), mode)
    return str(spec["prompt"]), int(spec["max_tokens"])


def prompt_size_probe(root: Path) -> dict[str, Any]:
    compact_workspace(root)
    files = [
        "program.md",
        "STRATEGY.md",
        "RUN_MEMORY.md",
        "SUMMARY.md",
        "results-recent.tsv",
        "tasks.jsonl",
        "findings.jsonl",
        "experiments.jsonl",
    ]
    measured: dict[str, int] = {}
    total_chars = 0
    for name in files:
        path = root / name
        try:
            chars = len(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            chars = 0
        measured[name] = chars
        total_chars += chars
    return {
        "ok": True,
        "mode": "prompt-size",
        "timestamp": int(time.time()),
        "chars": measured,
        "total_chars": total_chars,
        "estimated_prompt_tokens": estimate_tokens("x" * total_chars),
    }


def prompt_shape_probe(root: Path) -> dict[str, Any]:
    compact_workspace(root)
    files = [
        "program.md",
        "STRATEGY.md",
        "RUN_MEMORY.md",
        "SUMMARY.md",
        "results-recent.tsv",
        "ideas.md",
        "tasks.jsonl",
        "findings.jsonl",
        "experiments.jsonl",
    ]
    entries: list[dict[str, Any]] = []
    stable_tokens = 0
    volatile_tokens = 0
    for name in files:
        path = root / name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        tokens = estimate_tokens(text)
        role = "stable" if name in {"program.md", "STRATEGY.md", "RUN_MEMORY.md", "ideas.md"} else "volatile"
        if role == "stable":
            stable_tokens += tokens
        else:
            volatile_tokens += tokens
        entries.append(
            {
                "file": name,
                "role": role,
                "chars": len(text),
                "estimated_tokens": tokens,
                "lines": len(text.splitlines()),
            }
        )
    entries = sorted(entries, key=lambda item: int(item["estimated_tokens"]), reverse=True)
    total_tokens = stable_tokens + volatile_tokens
    return {
        "ok": True,
        "mode": "prompt-shape",
        "timestamp": int(time.time()),
        "total_estimated_tokens": total_tokens,
        "stable_estimated_tokens": stable_tokens,
        "volatile_estimated_tokens": volatile_tokens,
        "volatile_ratio": round(volatile_tokens / total_tokens, 3) if total_tokens else 0,
        "largest": entries[:5],
    }


def benchmark(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
    mode = "quick-health" if args.quick else args.mode
    commit = current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch")))
    if mode == "prompt-size":
        result = prompt_size_probe(root)
        out = root / "benchmarks" / f"benchmark-{result['timestamp']}-{mode}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        append_result(
            root,
            run_id=f"benchmark-{result['timestamp']}",
            status="keep",
            target=mode,
            hypothesis="measure prompt/context size pressure before changing prompt policy",
            commit=commit,
            notes=f"estimated_prompt_tokens={result['estimated_prompt_tokens']} total_chars={result['total_chars']}",
        )
        print(json.dumps(result, indent=2))
        return 0
    if mode == "prompt-shape":
        result = prompt_shape_probe(root)
        out = root / "benchmarks" / f"benchmark-{result['timestamp']}-{mode}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        largest = result["largest"][0]["file"] if result["largest"] else "none"
        append_result(
            root,
            run_id=f"benchmark-{result['timestamp']}",
            status="keep",
            target=mode,
            hypothesis="separate stable prompt/cacheable context from volatile research ledger growth",
            commit=commit,
            notes=(
                f"total_tokens={result['total_estimated_tokens']} "
                f"stable_tokens={result['stable_estimated_tokens']} "
                f"volatile_tokens={result['volatile_estimated_tokens']} "
                f"volatile_ratio={result['volatile_ratio']} largest={largest}"
            ),
        )
        print(json.dumps(result, indent=2))
        return 0
    base_url = args.base_url
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=3) as response:
            models = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        print(json.dumps({"ok": False, "status": "blocked", "reason": f"model endpoint unavailable: {error}"}, indent=2))
        return 2
    data = models.get("data") if isinstance(models, dict) else None
    model = args.model or (data[0].get("id") if isinstance(data, list) and data and isinstance(data[0], dict) else "local-model")
    spec = benchmark_spec(root, mode)
    prompt, max_tokens = str(spec["prompt"]), int(spec["max_tokens"])
    active_log = log_path()
    log_offset = file_size(active_log)
    before_memory = memory_snapshot()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": spec.get("temperature", 0),
        "max_tokens": max_tokens,
        "stream": bool(spec.get("stream", mode == "streaming-ttft")),
    }
    if args.draft_block_size and mode == "decode-sample":
        payload["draft_block_size"] = args.draft_block_size
    try:
        if mode == "streaming-ttft":
            ttft_s, wall_s, content = stream_model_request(base_url, payload, args.timeout)
            completion_tokens = 0
            completion_token_source = ""
        elif mode == "prefill-reuse":
            first_wall_s, _body = model_request(base_url, payload, args.timeout)
            wall_s, body = model_request(base_url, payload, args.timeout)
            parsed = json.loads(body.decode("utf-8"))
            content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
            completion_tokens, completion_token_source = completion_tokens_from_response(parsed, str(content))
            ttft_s = ""
        else:
            wall_s, body = model_request(base_url, payload, args.timeout)
            parsed = json.loads(body.decode("utf-8"))
            content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
            completion_tokens, completion_token_source = completion_tokens_from_response(parsed, str(content))
            first_wall_s = None
            ttft_s = ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "status": "blocked", "reason": str(error)}, indent=2))
        return 2
    after_memory = memory_snapshot()
    mtp = parse_generation_log_metrics(read_since(active_log, log_offset))
    decode_tps = round(completion_tokens / wall_s, 3) if mode == "decode-sample" and wall_s > 0 else ""
    server_elapsed = mtp.get("elapsed_s") if isinstance(mtp, dict) else ""
    server_tok_s = mtp.get("server_tok_s") if isinstance(mtp, dict) else ""
    measurement_quality = "clean"
    contamination_reasons: list[str] = []
    if mode == "decode-sample" and isinstance(decode_tps, (int, float)) and server_tok_s not in {"", None}:
        try:
            if float(server_tok_s) / max(float(decode_tps), 0.001) >= 2.0:
                contamination_reasons.append("server-wall-tps-gap")
        except (TypeError, ValueError):
            pass
    if mode == "decode-sample" and server_elapsed not in {"", None}:
        try:
            if float(wall_s) / max(float(server_elapsed), 0.001) >= 2.0:
                contamination_reasons.append("server-wall-time-gap")
        except (TypeError, ValueError):
            pass
    if contamination_reasons:
        measurement_quality = "contaminated"
    result = {
        "ok": True,
        "model": model,
        "mode": mode,
        "manifest_version": spec.get("manifest_version", "unknown"),
        "prompt_class": spec.get("prompt_class", "unknown"),
        "wall_s": round(wall_s, 3),
        "ttft_s": round(ttft_s, 3) if isinstance(ttft_s, float) else ttft_s,
        "completion_tokens": completion_tokens if completion_tokens else "",
        "completion_token_source": completion_token_source,
        "decode_tps": decode_tps,
        "first_wall_s": round(first_wall_s, 3) if mode == "prefill-reuse" and first_wall_s is not None else "",
        "second_wall_s": round(wall_s, 3) if mode == "prefill-reuse" else "",
        "memory_before_mb": before_memory,
        "memory_after_mb": after_memory,
        "draft_block_size": args.draft_block_size if args.draft_block_size else "",
        "mtp": mtp,
        "measurement_quality": measurement_quality,
        "contamination_reasons": contamination_reasons,
        "content_preview": str(content)[:120],
        "timestamp": int(time.time()),
    }
    schema_ok, schema_issue = benchmark_result_schema_ok(root, result)
    if not schema_ok:
        if getattr(args, "record_schema_failures", True):
            append_result(
                root,
                run_id=f"benchmark-{result['timestamp']}",
                status="blocked",
                target=mode,
                hypothesis=f"bounded OpenClaw {mode} probe",
                wall_s=result["wall_s"],
                memory_gb=round(after_memory.get("compressor_mb", 0) / 1024, 3) if after_memory else "",
                commit=commit,
                notes=f"schema_issue={schema_issue}",
            )
        result["ok"] = False
        result["status"] = "blocked"
        result["reason"] = schema_issue
        print(json.dumps(result, indent=2))
        return 2
    out = root / "benchmarks" / f"benchmark-{result['timestamp']}-{mode}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    append_result(
        root,
        run_id=f"benchmark-{result['timestamp']}",
        status="keep",
        target=mode,
        hypothesis=f"bounded OpenClaw {mode} probe",
        ttft_s=result["ttft_s"],
        decode_tps=decode_tps,
        wall_s=result["wall_s"],
        memory_gb=round(after_memory.get("compressor_mb", 0) / 1024, 3) if after_memory else "",
        commit=commit,
        notes=(
            f"model={model} completion_tokens={completion_tokens} "
            f"token_source={completion_token_source} "
            f"manifest_version={spec.get('manifest_version', 'unknown')} "
            f"prompt_class={spec.get('prompt_class', 'unknown')} "
            f"draft_block_size={args.draft_block_size or ''} "
            f"server_tok_s={mtp.get('server_tok_s', '')} "
            f"server_elapsed_s={mtp.get('elapsed_s', '')} "
            f"mtp_rounds={mtp.get('mtp_rounds', '')} "
            f"mean_accept={mtp.get('mean_accept', '')} "
            f"measurement_quality={measurement_quality} "
            f"contamination_reasons={','.join(contamination_reasons)} "
            f"preview={str(content)[:40].replace(chr(9), ' ').replace(chr(10), ' ')}"
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


def append_baseline(args: argparse.Namespace) -> int:
    root = workspace_root()
    results = root / "results.tsv"
    write_if_missing(results, RESULTS_HEADER)
    row = [
        time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        args.run_id,
        args.status,
        args.target,
        args.hypothesis.replace("\t", " "),
        args.ttft_s,
        args.prefill_tps,
        args.decode_tps,
        args.wall_s,
        args.memory_gb,
        current_commit(Path.cwd()),
        args.notes.replace("\t", " "),
    ]
    with results.open("a", encoding="utf-8") as file:
        file.write("\t".join(row) + "\n")
    print(results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the OpenClaw speed autoresearch workspace.")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup")
    setup.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    setup.set_defaults(func=setup_workspace)

    self_improve_parser = sub.add_parser("self-improve")
    self_improve_parser.add_argument(
        "--action",
        choices=("status", "derive-lessons", "curate", "evolve"),
        default="status",
    )
    self_improve_parser.add_argument("--recent-rows", type=int, default=160)
    self_improve_parser.add_argument("--max-variants-per-skill", type=int, default=2)
    self_improve_parser.add_argument("--min-score", type=int, default=90)
    self_improve_parser.add_argument("--shadow-min-score", type=int, default=90)
    self_improve_parser.add_argument("--stage-min-wins", type=int, default=2)
    self_improve_parser.add_argument(
        "--stage-max-effective-authority",
        choices=("none", "canary", "shadow", "advisory"),
        default="advisory",
    )
    self_improve_parser.set_defaults(func=self_improve)

    alive_parser = sub.add_parser("alive-eval")
    alive_parser.add_argument("--recent-rows", type=int, default=160)
    alive_parser.add_argument("--allow-fail", action="store_true")
    alive_parser.set_defaults(func=alive_eval)

    prompt = sub.add_parser("prompt")
    prompt.set_defaults(func=lambda _args: print(prompt_text(workspace_root())) or 0)

    source = sub.add_parser("add-source")
    source.add_argument("source")
    source.add_argument("--kind", choices=["note", "url", "image-url", "file", "article"], default="url")
    source.add_argument("--title", default="")
    source.add_argument("--note", default="")
    source.set_defaults(func=add_source)

    bench = sub.add_parser("benchmark")
    bench.add_argument("--base-url", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_MODEL_URL", DEFAULT_MODEL_URL))
    bench.add_argument("--model", default="")
    bench.add_argument("--quick", action="store_true")
    bench.add_argument(
        "--mode",
        choices=[
            "quick-health",
            "streaming-ttft",
            "tool-roundtrip",
            "prompt-size",
            "prompt-shape",
            "decode-sample",
            "prefill-reuse",
        ],
        default="quick-health",
    )
    bench.add_argument("--timeout", type=float, default=180.0)
    bench.add_argument(
        "--draft-block-size",
        type=int,
        default=0,
        help="send a per-request Gemma MTP draft_block_size for safe block-size benchmarking",
    )
    bench.set_defaults(func=benchmark)

    record = sub.add_parser("record")
    record.add_argument("--run-id", default="manual")
    record.add_argument("--status", default="blocked")
    record.add_argument("--target", default="baseline")
    record.add_argument("--hypothesis", default="baseline")
    record.add_argument("--ttft-s", default="")
    record.add_argument("--prefill-tps", default="")
    record.add_argument("--decode-tps", default="")
    record.add_argument("--wall-s", default="")
    record.add_argument("--memory-gb", default="")
    record.add_argument("--notes", default="")
    record.set_defaults(func=append_baseline)

    synth = sub.add_parser("synthesize")
    synth.add_argument("--kind", choices=["frontier", "current-stack"], default="frontier")
    synth.set_defaults(func=synthesize)

    source = sub.add_parser("source-scout")
    source.add_argument("--topic", default="frontier-decode-speed")
    source.add_argument("--timeout", type=float, default=6.0)
    source.add_argument("--max-sources", type=int, default=10)
    source.set_defaults(func=source_scout)

    deliberation = sub.add_parser("frontier-deliberation")
    deliberation.add_argument("--allow-empty", action="store_true")
    deliberation.set_defaults(func=frontier_deliberation)

    review = sub.add_parser("quality-review")
    review.add_argument("--recent-rows", type=int, default=120)
    review.add_argument("--min-sweeps", type=int, default=3)
    review.add_argument("--min-samples-per-block", type=int, default=3)
    review.add_argument("--target-tps", type=float, default=30.0)
    review.set_defaults(func=quality_review)

    handoff = sub.add_parser("implementation-handoff-audit")
    handoff.add_argument("--min-score", type=int, default=90)
    handoff.set_defaults(func=implementation_handoff_audit)

    bottleneck = sub.add_parser("drafter-bottleneck-review")
    bottleneck.add_argument("--recent-rows", type=int, default=240)
    bottleneck.set_defaults(func=drafter_bottleneck_review)

    adapter_contract = sub.add_parser("drafter-adapter-method-contract")
    adapter_contract.add_argument("--recent-rows", type=int, default=240)
    adapter_contract.set_defaults(func=drafter_adapter_method_contract)

    snapshot = sub.add_parser("environment-snapshot")
    snapshot.add_argument("--label", default="manual")
    snapshot.add_argument("--repo", default=str(repo_root()))
    snapshot.add_argument("--allow-fail", action="store_true")
    snapshot.set_defaults(func=environment_snapshot_command)

    integrity = sub.add_parser("evaluator-integrity")
    integrity.add_argument("--allow-fail", action="store_true")
    integrity.set_defaults(func=evaluator_integrity_command)

    plateau = sub.add_parser("plateau-pivot")
    plateau.add_argument("--recent-rows", type=int, default=120)
    plateau.add_argument("--min-sweeps", type=int, default=3)
    plateau.add_argument("--target-tps", type=float, default=30.0)
    plateau.set_defaults(func=plateau_pivot)

    frontier = sub.add_parser("frontier-review")
    frontier.add_argument("--recent-rows", type=int, default=160)
    frontier.add_argument("--min-samples", type=int, default=3)
    frontier.set_defaults(func=frontier_review)

    council = sub.add_parser("review-council")
    council.add_argument("--recent-rows", type=int, default=120)
    council.add_argument("--target-tps", type=float, default=30.0)
    council.add_argument("--seed-next", action="store_true")
    council.add_argument("--allow-fail", action="store_true")
    council.set_defaults(func=review_council)

    frontier_eval_parser = sub.add_parser("frontier-eval")
    frontier_eval_parser.add_argument("--recent-rows", type=int, default=120)
    frontier_eval_parser.add_argument("--min-score", type=float, default=9.0)
    frontier_eval_parser.add_argument("--allow-fail", action="store_true")
    frontier_eval_parser.set_defaults(func=frontier_eval)

    burn_in = sub.add_parser("stability-burn-in")
    burn_in.add_argument("--recent-rows", type=int, default=120)
    burn_in.add_argument("--min-free-mb", type=int, default=8192)
    burn_in.add_argument("--max-compressor-mb", type=int, default=8192)
    burn_in.add_argument("--max-swap-mb", type=int, default=8192)
    burn_in.add_argument("--allow-fail", action="store_true")
    burn_in.set_defaults(func=stability_burn_in)

    autonomy = sub.add_parser("frontier-autonomy-score")
    autonomy.add_argument("--recent-rows", type=int, default=120)
    autonomy.add_argument("--promotion", action="store_true")
    autonomy.add_argument("--rollback-rehearsal-ok", action="store_true")
    autonomy.add_argument("--allow-fail", action="store_true")
    autonomy.set_defaults(func=frontier_autonomy_score)

    sota = sub.add_parser("sota-eval")
    sota.add_argument("--recent-rows", type=int, default=160)
    sota.add_argument("--allow-fail", action="store_true")
    sota.set_defaults(func=sota_autonomy_eval)

    stable_mark = sub.add_parser("stable-build-mark")
    stable_mark.add_argument("--repo", default="")
    stable_mark.add_argument("--reason", default="")
    stable_mark.set_defaults(func=stable_build_mark)

    stable_rollback = sub.add_parser("stable-build-rollback")
    stable_rollback.add_argument("--repo", default="")
    stable_rollback.add_argument("--reason", default="")
    stable_rollback.add_argument("--dry-run", action="store_true")
    stable_rollback.add_argument("--force", action="store_true")
    stable_rollback.set_defaults(func=stable_build_rollback)

    rank = sub.add_parser("hypothesis-rank")
    rank.add_argument("--limit", type=int, default=12)
    rank.set_defaults(func=hypothesis_rank)

    causal = sub.add_parser("causal-review")
    causal.add_argument("--recent-rows", type=int, default=160)
    causal.set_defaults(func=causal_review)

    gepa = sub.add_parser("gepa-escalation")
    gepa.add_argument("--recent-rows", type=int, default=160)
    gepa.add_argument("--min-blocked", type=int, default=3)
    gepa.add_argument("--min-rework", type=int, default=2)
    gepa.add_argument("--min-trajectory", type=int, default=2)
    gepa.add_argument("--min-low-quality", type=int, default=2)
    gepa.set_defaults(func=gepa_escalation)

    gepa_canary = sub.add_parser("gepa-policy-canary")
    gepa_canary.add_argument("--task-id", default="")
    gepa_canary.set_defaults(func=gepa_policy_canary)

    gepa_promote = sub.add_parser("gepa-policy-promote")
    gepa_promote.add_argument("--min-candidates", type=int, default=3)
    gepa_promote.set_defaults(func=gepa_policy_promote)

    overhead = sub.add_parser("runtime-overhead-map")
    overhead.add_argument("--recent-rows", type=int, default=160)
    overhead.set_defaults(func=runtime_overhead_map)

    calibration_memory = sub.add_parser("calibration-memory-report")
    calibration_memory.set_defaults(func=calibration_memory_report)

    trace_gate = sub.add_parser("drafter-trace-gate")
    trace_gate.add_argument(
        "--plan",
        default="/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json",
    )
    trace_gate.add_argument("--trace-data", default="")
    trace_gate.set_defaults(func=drafter_trace_gate)

    trace_prereq = sub.add_parser("drafter-trace-prerequisite")
    trace_prereq.set_defaults(func=drafter_trace_prerequisite)

    trace_collect = sub.add_parser("drafter-trace-collect")
    trace_collect.add_argument("--base-url", default=os.environ.get("OPENCLAW_SPEED_RESEARCH_MODEL_URL", DEFAULT_MODEL_URL))
    trace_collect.add_argument("--model", default="")
    trace_collect.add_argument("--output", default="/Users/kristian/.openclaw/drafter-fit/target-generated-traces.jsonl")
    trace_collect.add_argument("--samples", type=int, default=6)
    trace_collect.add_argument("--min-traces", type=int, default=4)
    trace_collect.add_argument("--max-tokens", type=int, default=96)
    trace_collect.add_argument("--timeout", type=float, default=120.0)
    trace_collect.add_argument("--min-free-mb", type=int, default=512)
    trace_collect.add_argument("--force", action="store_true")
    trace_collect.set_defaults(func=drafter_trace_collect)

    calibration_canary = sub.add_parser("drafter-calibration-canary")
    calibration_canary.add_argument(
        "--plan",
        default="/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json",
    )
    calibration_canary.add_argument("--trace-data", default="")
    calibration_canary.add_argument("--output-dir", default="/Users/kristian/.openclaw/drafter-fit")
    calibration_canary.add_argument("--min-traces", type=int, default=4)
    calibration_canary.add_argument("--max-prompts", type=int, default=6)
    calibration_canary.add_argument(
        "--calibration-mode",
        choices=sorted(CALIBRATION_MODES),
        default=CALIBRATION_DIRECT_MODE,
    )
    calibration_canary.add_argument("--test-timeout", type=float, default=60.0)
    calibration_canary.add_argument("--skip-test", action="store_true")
    calibration_canary.set_defaults(func=drafter_calibration_canary)

    calibration_stage = sub.add_parser("drafter-calibration-memory-stage")
    calibration_stage.add_argument("--stage", choices=list(CALIBRATION_MEMORY_STAGES), required=True)
    calibration_stage.add_argument(
        "--plan",
        default="/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json",
    )
    calibration_stage.add_argument("--trace-data", default="")
    calibration_stage.add_argument("--output-dir", default="/Users/kristian/.openclaw/drafter-fit")
    calibration_stage.add_argument("--min-traces", type=int, default=4)
    calibration_stage.add_argument("--max-prompts", type=int, default=6)
    calibration_stage.add_argument(
        "--calibration-mode",
        choices=sorted(CALIBRATION_MODES),
        default=CALIBRATION_DIRECT_MODE,
    )
    calibration_stage.set_defaults(func=drafter_calibration_memory_stage)

    dflash_gate = sub.add_parser("dflash-compatibility-gate")
    dflash_gate.add_argument("--draft-path", default="")
    dflash_gate.add_argument(
        "--plan",
        default="/Users/kristian/.openclaw/drafter-fit/gemma4-janq-dflash-fit-plan.json",
    )
    dflash_gate.set_defaults(func=dflash_compatibility_gate)

    compact_parser = sub.add_parser("compact")
    compact_parser.add_argument("--recent-rows", type=int, default=24)
    compact_parser.set_defaults(func=compact)

    manifest = sub.add_parser("manifest")
    manifest.set_defaults(func=print_manifest)

    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--allow-fail", action="store_true")
    replay_parser.set_defaults(func=replay)

    paired = sub.add_parser("paired-plan")
    paired.add_argument("--task-id", required=True)
    paired.set_defaults(func=paired_plan)

    sweep = sub.add_parser("drafter-sweep-plan")
    sweep.add_argument("--blocks", default="1,2,3,4,6")
    sweep.add_argument("--samples", type=int, default=3)
    sweep.add_argument("--min-delta", type=float, default=0.5)
    sweep.set_defaults(func=drafter_sweep_plan)

    sweep_run = sub.add_parser("drafter-sweep-run")
    sweep_run.add_argument("--blocks", default="1,2,3,4")
    sweep_run.add_argument("--samples", type=int, default=3)
    sweep_run.add_argument("--min-delta", type=float, default=0.5)
    sweep_run.add_argument("--control-block", type=int, default=0)
    sweep_run.add_argument("--retries", type=int, default=2)
    sweep_run.add_argument("--base-url", default=DEFAULT_MODEL_URL)
    sweep_run.add_argument("--model", default="")
    sweep_run.add_argument("--timeout", type=float, default=180.0)
    sweep_run.set_defaults(func=drafter_sweep_run)

    report = sub.add_parser("mtp-report")
    report.add_argument("--lines", type=int, default=160)
    report.add_argument("--log-path", default="")
    report.set_defaults(func=mtp_report)

    patch = sub.add_parser("patch-execute")
    patch.add_argument("--patch-file", required=True)
    patch.add_argument("--task-id", default="")
    patch.add_argument("--hypothesis", default="")
    patch.add_argument("--source-files", default="")
    patch.add_argument("--tests", default=";".join(DEFAULT_PATCH_TESTS))
    patch.add_argument("--repo", default="")
    patch.add_argument("--test-timeout", type=float, default=90.0)
    patch.add_argument("--canary-only", action="store_true")
    patch.add_argument("--keep-canary", action="store_true")
    patch.add_argument("--allow-architectural", action="store_true")
    patch.add_argument("--crabbox-evidence-file", default="")
    patch.add_argument("--rollback-rehearsal-ok", action="store_true")
    patch.add_argument(
        "--architectural-approval-file",
        default=os.environ.get("OPENCLAW_SPEED_RESEARCH_ARCHITECTURAL_APPROVAL_FILE", ""),
    )
    patch.set_defaults(func=patch_execute)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
