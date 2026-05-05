#!/usr/bin/env python3
"""OpenClaw speed autoresearch workspace helper.

This adapts the karpathy/autoresearch loop to OpenClaw runtime speed work:
small scoped experiments, bounded benchmarks, TSV results, and keep/discard
discipline. It intentionally does not start a model by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO_URL = "https://github.com/karpathy/autoresearch.git"
DEFAULT_MODEL_URL = "http://127.0.0.1:8091/v1"
RESULTS_HEADER = (
    "timestamp\trun_id\tstatus\ttarget\thypothesis\tttft_s\tprefill_tps\tdecode_tps\t"
    "wall_s\tmemory_gb\tcommit\tnotes\n"
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

This workspace adapts the `karpathy/autoresearch` method to OpenClaw runtime speed and reliability research. The goal is not to train a model. The goal is to improve OpenClaw's perceived speed, prefill behavior, decode behavior, memory stability, tool-call reliability, and loop resistance while preserving the native OpenClaw experience.

## Scope

You are working on OpenClaw only. Do not touch opencode. Do not change the user's model choice unless the user explicitly asks. The active model target is Gemma 4 31B JANG through the OpenClaw model profile layer.

In-scope files are the OpenClaw harness files in the setup repository, especially:

- `openclaw/openclaw-wrapper.zsh`
- `openclaw/openclaw-model-proxy.py`
- `openclaw/openclaw-rapid-launcher.py`
- `openclaw/openclaw-prefix-warmer.py`
- `openclaw/openclaw-local-health`
- `openclaw/rapid-overlay/openclaw_rapid_jang.py`
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

- Rapid-MLX serving behavior for Gemma 4 31B JANG/JANQ.
- JANG bridge behavior in `openclaw/rapid-overlay/openclaw_rapid_jang.py`.
- OpenClaw model proxy behavior in `openclaw/openclaw-model-proxy.py`.
- Rapid launcher/profile settings in `openclaw/openclaw-rapid-launcher.py` and `openclaw/openclaw-wrapper.zsh`.
- Prefix/prompt reuse, context compaction, tool-call shaping, and realistic TTFT.

Think broadly, but every useful idea must become one of: a source patch, a benchmark result, a rejected experiment with evidence, or a source note for later.

Do not spend rounds on toy prompts such as repeated single words except as a one-time health check. Do not optimize for synthetic decode-only numbers if the change does not improve OpenClaw agent UX.

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

Use `openclaw speed-research benchmark --quick` for a bounded probe. If the model is stopped and memory pressure is high, record the blocker instead of starting a 31B model.

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

In a fresh run, perform these steps in order. Each step is exactly one assistant turn and exactly one narrow tool call.

1. Read exactly `/Users/kristian/.openclaw/research/speed/program.md`.
2. Read exactly `/Users/kristian/.openclaw/research/speed/README-openclaw-speed.md`.
3. Read exactly `/Users/kristian/.openclaw/research/speed/results.tsv`.
4. Run exactly `/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --quick`.
5. Record a TSV row only if the benchmark helper did not already record the result.

Do not run setup commands during bootstrap. The workspace already exists. Do not use `find`, recursive `ls`, recursive grep, or broad local search. Do not read `reference/autoresearch/program.md` during bootstrap; it is method reference only, not required for the first experiment.

## Narrow Tool Catalog

Allowed narrow actions are:

- `read` a single explicit file path from Scope, Bootstrap Ladder, or `sources/queue.md`.
- `exec` one exact command against the OpenClaw source repo, such as `git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`.
- `exec` one exact test file, such as `python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py`.
- `exec` one exact benchmark helper, such as `/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --quick`.
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
openclaw speed-research-benchmark --quick
```

The research agent should improve OpenClaw speed and reliability without touching opencode.
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
"""


def bootstrap_ladder_section() -> str:
    return """## Bootstrap Ladder

In a fresh run, perform these steps in order. Each step is exactly one assistant turn and exactly one narrow tool call.

1. Read exactly `/Users/kristian/.openclaw/research/speed/program.md`.
2. Read exactly `/Users/kristian/.openclaw/research/speed/README-openclaw-speed.md`.
3. Read exactly `/Users/kristian/.openclaw/research/speed/results.tsv`.
4. Run exactly `/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --quick`.
5. Record a TSV row only if the benchmark helper did not already record the result.

Do not run setup commands during bootstrap. The workspace already exists. Do not use `find`, recursive `ls`, recursive grep, or broad local search. Do not read `reference/autoresearch/program.md` during bootstrap; it is method reference only, not required for the first experiment.
"""


def narrow_tool_catalog_section() -> str:
    return """## Narrow Tool Catalog

Allowed narrow actions are:

- `read` a single explicit file path from Scope, Bootstrap Ladder, or `sources/queue.md`.
- `exec` one exact command against the OpenClaw source repo, such as `git -C /Users/kristian/Documents/openclaw-harness-autoresearch status --short --branch`.
- `exec` one exact test file, such as `python3 /Users/kristian/Documents/openclaw-harness-autoresearch/openclaw/test-speed-research.py`.
- `exec` one exact benchmark helper, such as `/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --quick`.
- `exec` one exact log tail, such as `tail -n 80 /Users/kristian/.openclaw/logs/openclaw-model-proxy.log`.

Forbidden actions include `find ~`, `find /`, `find /Users`, `ls -R`, `grep -R`, recursive `rg` over home, `mdfind`, and any broad command intended to discover files. If you need a file, use the explicit paths in this program.
"""


def current_priority_section() -> str:
    return """## Current Priority

Focus on improvements that make this exact OpenClaw setup faster and more reliable:

- Rapid-MLX serving behavior for Gemma 4 31B JANG/JANQ.
- JANG bridge behavior in `openclaw/rapid-overlay/openclaw_rapid_jang.py`.
- OpenClaw model proxy behavior in `openclaw/openclaw-model-proxy.py`.
- Rapid launcher/profile settings in `openclaw/openclaw-rapid-launcher.py` and `openclaw/openclaw-wrapper.zsh`.
- Prefix/prompt reuse, context compaction, tool-call shaping, and realistic TTFT.

Think broadly, but every useful idea must become one of: a source patch, a benchmark result, a rejected experiment with evidence, or a source note for later.

Do not spend rounds on toy prompts such as repeated single words except as a one-time health check. Do not optimize for synthetic decode-only numbers if the change does not improve OpenClaw agent UX.
"""


def frontier_speed_track_section() -> str:
    return """## Frontier Speed Track

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
"""


def realistic_experiment_backlog_section() -> str:
    return """## Realistic Experiment Backlog

Prioritize these experiment types over generic LLM speed prompts:

- OpenClaw bootstrap turn: first tool call latency when asked to read `program.md`.
- Tool-call round trip: read one explicit OpenClaw source file, summarize it, then write one result row.
- Prompt-size impact: compare shaped prompt tokens and TTFT before/after context compaction or history limits.
- Rapid-MLX backend settings: test one setting at a time, such as prefill step size, batch size, KV cache quantization, prefix cache policy, stream interval, or max tool tokens.
- JANG/JANQ bridge behavior: inspect whether the bridge prevents repeated tokens, reasoning leakage, and expensive unnecessary full-prompt paths.
- Perceived latency: verify heartbeat/status behavior during long prefill and time to first useful TUI status.
- Memory safety: record Metal active/peak memory, RSS, and whether compressor/swap rises.

Each experiment must name the exact file or runtime knob under test and the exact command used to measure it.
"""


def speed_targets_section() -> str:
    return """## Speed Targets

Treat these as directional targets, not promises:

- Immediate usability target: first useful status in under 5s and follow-up agent turns under 15s when prefix/cache is warm.
- Current-stack stretch target: realistic OpenClaw decode above 20 tok/s and much lower repeated prefill.
- Frontier target: investigate paths that could reach 50-70 tok/s perceived or measured decode on suitable workloads.

Prefer perceived speed wins that help agentic work: faster first tool call, less repeated prefill, better streaming status, fewer unnecessary model turns, and fewer wasted tokens.
"""


def implementation_gate_section() -> str:
    return """## Implementation Gate

Before keeping an idea, prove it can be implemented in this setup:

1. Identify the specific source file or model profile knob.
2. Make the smallest source/config change.
3. Run focused tests.
4. Deploy only if tests pass.
5. Run a realistic benchmark.
6. Record `keep`, `discard`, or `blocked` in `results.tsv` with the reason.

If a change requires an upstream Rapid-MLX feature that is not present locally, record the gap clearly and move to the next implementable improvement.
"""


def prompt_text(root: Path) -> str:
    return f"""OpenClaw Speed Autoresearch bootstrap.

Workspace: {root}

First assistant action: read exactly this file and nothing else:
`{root / 'program.md'}`

After that, follow the Bootstrap Ladder in `program.md` exactly. The next action after reading `program.md` is to read exactly:
`{root / 'README-openclaw-speed.md'}`

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
    (root / "experiments").mkdir(exist_ok=True)
    (root / "benchmarks").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "sources").mkdir(exist_ok=True)
    clone_status = clone_or_update_reference(root, args.repo_url)
    write_if_missing(root / "program.md", program_md())
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
    write_if_missing(root / "README-openclaw-speed.md", readme_md())
    write_if_missing(root / "ideas.md", "# Speed Research Ideas\n\n")
    write_if_missing(
        root / "sources" / "queue.md",
        "# Research Source Queue\n\n"
        "Add notes, URLs, article excerpts, image URLs, and local file paths here. "
        "The OpenClaw speed researcher should read this file at the start of each loop "
        "and incorporate relevant sources without broad speculative searches.\n\n",
    )
    normalize_source_queue(root / "sources" / "queue.md")
    write_if_missing(root / "results.tsv", RESULTS_HEADER)
    write_if_missing(root / ".gitignore", "logs/\nbenchmarks/*.json\n*.tmp\n")
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


def benchmark(args: argparse.Namespace) -> int:
    root = workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=3) as response:
            models = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        print(json.dumps({"ok": False, "status": "blocked", "reason": f"model endpoint unavailable: {error}"}, indent=2))
        return 2
    data = models.get("data") if isinstance(models, dict) else None
    model = args.model or (data[0].get("id") if isinstance(data, list) and data and isinstance(data[0], dict) else "local-model")
    prompt = "Reply with exactly: OK" if args.quick else "Briefly explain one way to reduce perceived latency in a local agent harness."
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 12 if args.quick else 96,
        "stream": False,
    }
    try:
        wall_s, body = model_request(base_url, payload, args.timeout)
        parsed = json.loads(body.decode("utf-8"))
        content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "status": "blocked", "reason": str(error)}, indent=2))
        return 2
    result = {
        "ok": True,
        "model": model,
        "quick": args.quick,
        "wall_s": round(wall_s, 3),
        "content_preview": str(content)[:120],
        "timestamp": int(time.time()),
    }
    out = root / "benchmarks" / f"benchmark-{result['timestamp']}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
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
    bench.add_argument("--timeout", type=float, default=180.0)
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

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
