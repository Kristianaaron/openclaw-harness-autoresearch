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
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from openclaw_speed_research_core import (
    RESULTS_HEADER,
    append_result,
    append_jsonl,
    benchmark_result_schema_ok,
    benchmark_spec,
    ensure_research_state,
    load_benchmark_manifest,
    paired_profile_plan,
    read_jsonl,
    replay_checks,
    score_insight,
    write_jsonl,
)

DEFAULT_REPO_URL = "https://github.com/karpathy/autoresearch.git"
DEFAULT_MODEL_URL = "http://127.0.0.1:8091/v1"
DEFAULT_PROXY_LOG = "/Users/kristian/.openclaw/logs/openclaw-model-proxy.log"
DEFAULT_PATCH_TESTS = (
    "python3 openclaw/test-speed-research.py",
    "python3 openclaw/test-speed-research-autopilot.py",
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


def home() -> Path:
    return Path(os.environ.get("OPENCLAW_HOME", Path.home() / ".openclaw")).expanduser()


def workspace_root() -> Path:
    return Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_DIR", home() / "research" / "speed")).expanduser()


def run(argv: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def git_available() -> bool:
    try:
        run(["git", "--version"])
        return True
    except Exception:
        return False


def write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


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
- Architectural patches are held after canary unless `allow_architectural` is explicitly set by the supervisor.
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


def prompt_text(root: Path) -> str:
    compact_workspace(root)
    return f"""OpenClaw Speed Autoresearch bootstrap.

Workspace: {root}

Primary scope: improve normal `openclaw tui` decode speed and visible response smoothness first. Improve autoresearch itself only when it helps produce safer, better TUI decode-speed changes.

First assistant action: run exactly this narrow benchmark command:
`/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --mode decode-sample`

Then read exactly:
`{root / 'SUMMARY.md'}`

Do not read the full `program.md` unless a human explicitly asks. It is installed policy, not first-turn context.

Hard constraints:
- OpenClaw only. Do not touch opencode.
- Do not change the model unless the user explicitly asks.
- Continue the loop without asking me to manually continue.
- Use one narrow tool call per assistant turn.
- Never use broad local search commands.
- Do not run setup commands during bootstrap.
"""


def setup_workspace(args: argparse.Namespace) -> int:
    root = workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    ensure_research_state(root)
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
            f"- full ledger: {root / 'results.tsv'}",
            f"- tasks: {root / 'tasks.jsonl'}",
        ]
    )
    write_if_changed(root / "SUMMARY.md", "\n".join(summary_lines).rstrip() + "\n")
    return {"ok": True, "recent_rows": len(recent), "ready_tasks": len(ready_tasks), "blocked_tasks": len(blocked_tasks)}


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


def upsert_tasks(root: Path, tasks: list[dict[str, Any]]) -> int:
    path = root / "tasks.jsonl"
    existing = read_jsonl(path)
    existing_by_id = {str(task.get("id", "")): index for index, task in enumerate(existing)}
    additions = 0
    changed = False
    for task in tasks:
        task_id = str(task.get("id", ""))
        if task_id not in existing_by_id:
            existing.append(task)
            additions += 1
            changed = True
            continue
        index = existing_by_id[task_id]
        current = existing[index]
        if current.get("status", "ready") not in {"ready", "rework"}:
            continue
        merged = {**current, **task, "status": current.get("status", task.get("status", "ready"))}
        if merged != current:
            existing[index] = merged
            changed = True
    if changed:
        write_jsonl(path, existing)
    return additions


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
            "guard_checks": ["one_narrow_tool", "no_loop"],
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
            "guard_checks": ["memory_gate", "bounded_trials", "tests_pass", "no_model_change", "restore_live_profile"],
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
            "guard_checks": ["no_model_load", "target_config_match", "tool_thinking_replay_required", "no_live_profile_change"],
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
            "guard_checks": ["same_tokenizer", "no_reasoning_leak", "no_model_change", "tests_pass"],
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
            "task_type": "analysis",
            "target": "dflash.model_mlx/openclaw-jang-vlm-server.py",
            "source_files": ["openclaw/openclaw-jang-vlm-server.py", "openclaw/test-speed-research.py"],
            "hypothesis": "DFlash can only improve TUI decode speed if its MLX draft loop can wrap the JANQ-loaded mlx_vlm Gemma4 target without bypassing OpenClaw guardrails.",
            "metric": "compatibility_decision_then_decode_tps",
            "guard_checks": ["no_live_profile_change", "separate_env", "memory_gate", "stream_guard", "no_reasoning_leak"],
            "acceptance": "Record a keep/discard/blocked decision with exact compatibility evidence before any DFlash install or live model benchmark.",
            "rollback": "No live rollback needed; this task must not change the active model profile or server path.",
            "evidence": speed_gap,
            "next_action": (
                "First tool call: read exactly /Users/kristian/.openclaw/research/speed/implementation-skill.md. "
                "Then inspect exactly https://github.com/z-lab/dflash or /tmp/dflash-openclaw-inspect/dflash/model_mlx.py if already cloned. "
                "Do not install DFlash into the live OpenClaw runtime and do not change the active model profile."
            ),
        },
    ]


def synthesize(args: argparse.Namespace) -> int:
    root = workspace_root()
    ensure_research_state(root)
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
    implementation_tasks = implementation_candidate_tasks(rows)
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
    append_jsonl(
        root / "findings.jsonl",
        {
            "timestamp": generated_at,
            "task_id": "synthesize-speed-ideas",
            "finding": "benchmark queue exhausted; synthesized ranked speed ideas and seeded measurable follow-up tasks",
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
            "seeded_tasks": seeded,
            "kind": args.kind,
        },
    )
    append_result(
        root,
        run_id=f"synthesis-{int(time.time())}",
        status="keep",
        target="synthesis",
        hypothesis="exhausted benchmark queues must generate ranked speed ideas and next tasks",
        commit=current_commit(Path(os.environ.get("OPENCLAW_SPEED_RESEARCH_REPO", "/Users/kristian/Documents/openclaw-harness-autoresearch"))),
        notes=f"ideas={len(ideas)} seeded_tasks={seeded} kind={args.kind}",
    )
    print(json.dumps({"ok": True, "ideas": len(ideas), "seeded_tasks": seeded, "ideas_path": str(ideas_path)}, indent=2))
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
    source_allowlist = {normalize_patch_path(path) for path in source_files or [] if path}
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    if not files:
        reasons.append("patch has no file changes")
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
    if len(files) > 3 or changed_lines > 220:
        architectural = True
    destructive = bool(reasons) or deletions > additions * 3 + 20
    if destructive:
        impact = "destructive"
    elif architectural:
        impact = "architectural"
    elif changed_lines <= 80 and len(files) <= 2:
        impact = "safe"
    else:
        impact = "moderate"
    auto_promote = impact in {"safe", "moderate"} or (impact == "architectural" and allow_architectural)
    if impact == "architectural" and not allow_architectural:
        reasons.append("architectural change requires canary evidence and explicit allow_architectural")
    return {
        "files": files,
        "additions": additions,
        "deletions": deletions,
        "changed_lines": changed_lines,
        "impact": impact,
        "architectural": architectural,
        "auto_promote": auto_promote and not destructive,
        "allowed": not destructive,
        "reasons": reasons,
    }


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
    source_files = [item.strip() for item in args.source_files.split(",") if item.strip()]
    classification = classify_patch(
        patch_text,
        source_files=source_files or None,
        allow_architectural=args.allow_architectural,
    )
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
        "repo": str(repo),
        "canary": str(canary),
        "classification": classification,
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
                if apply.returncode != 0:
                    reason = "patch apply failed in canary"
                    artifact["reason"] = reason
                elif not tests_ok:
                    reason = "canary tests failed"
                    artifact["reason"] = reason
                elif not classification["auto_promote"] or args.canary_only:
                    reason = "canary passed; promotion intentionally held"
                    artifact["reason"] = reason
                    status = "keep"
                    return_code = 0
                else:
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
                        if reason:
                            artifact["reason"] = reason
                        return_code = 0 if artifact["promoted"] else 2
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
    files = ["program.md", "STRATEGY.md", "SUMMARY.md", "results-recent.tsv", "tasks.jsonl", "findings.jsonl", "experiments.jsonl"]
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
    files = ["program.md", "STRATEGY.md", "SUMMARY.md", "results-recent.tsv", "ideas.md", "tasks.jsonl", "findings.jsonl", "experiments.jsonl"]
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
        role = "stable" if name in {"program.md", "STRATEGY.md", "ideas.md"} else "volatile"
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
            f"mtp_rounds={mtp.get('mtp_rounds', '')} "
            f"mean_accept={mtp.get('mean_accept', '')} "
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
    patch.set_defaults(func=patch_execute)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
