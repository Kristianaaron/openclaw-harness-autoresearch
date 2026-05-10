# OpenClaw Harness Autoresearch

OpenClaw-only local harness hardening for Gemma 4 31B JANG/JANQ on Apple Silicon.

This repository contains the OpenClaw model profile layer, local MLX/JANG launchers,
model proxy guardrails, prefix warming, memory gates, deterministic benchmarking,
and an autonomous speed autoresearch supervisor. The primary objective is to make
normal `openclaw tui` use faster and more reliable without touching opencode.

## Scope

- OpenClaw only. No opencode configuration, runtime files, or behavior changes.
- Model/profile-driven backend selection, so backend details stay outside the
  gateway and TUI.
- Local Apple Silicon MLX/JANG reliability and speed experiments.
- Current optimization focus: Gemma 4 31B JANG/JANQ decode speed, MTP/drafter
  acceptance, DFlash compatibility, and perceived latency.
- Sensitive runtime state stays outside the repo under the user's OpenClaw home
  and is ignored by Git.

## Key Features

### Model And Runtime Layer

- OpenClaw model profiles describe backend type, ports, health URLs, server
  commands, logs, and memory class.
- Gateway remains model-agnostic; the active model profile decides which local
  backend should run.
- Launch wrappers start only the OpenClaw-owned backend that is needed.
- Memory gates check macOS free memory, compressor, swap, and pressure before
  launching or continuing large local MLX work.
- Interrupt handling writes a checkpoint and stops OpenClaw-owned model
  processes so memory can be released cleanly after `Ctrl+C`.

### Proxy Guardrails

- SSE deadlines and idle watchdogs prevent silent stream hangs.
- Tool-call and reasoning streams are separated and bounded.
- Repeated `thought`, reasoning marker, and malformed tool-call patterns are
  detected before they can flood the TUI.
- Broad local tool commands are blocked or routed toward narrower inspections.
- Large prompt/tool contexts are preflighted before local MLX execution to avoid
  Metal or memory pressure crashes.

### Autoresearch Supervisor

The autoresearch loop is a deterministic supervisor around OpenClaw research
turns. The LLM can propose hypotheses and synthesize findings, but the supervisor
owns benchmark execution, task routing, review, memory protection, and promotion
gates.

Core loop:

1. Measure baseline health and speed.
2. Select the next runnable task from `tasks.jsonl`.
3. Prefer deterministic supervisor tasks before model-bound research turns.
4. Run bounded benchmarks or analyses.
5. Record evidence in `results.tsv`, `findings.jsonl`, `experiments.jsonl`, and
   benchmark artifacts.
6. Review quality, task contracts, implementation handoff, replay guards, and
   frontier eval.
7. Keep, discard, suppress, or refocus lanes based on evidence.
8. Seed the next deterministic tranche automatically.

The loop is designed to continue overnight until `--max-hours` is reached or the
user presses `Ctrl+C`. Clean external blockers are converted into an autonomous
refocus tranche by default instead of stopping and waiting for manual prompting.

### Frontier Expansion

When the active speed lanes exhaust, the supervisor now opens one new bounded
candidate path instead of repeating the same synthesis or stopping:

- JANQ quantized-gradient calibration blockers route to a trace-distillation
  candidate that detaches target traces and trains only the drafter-side
  projection.
- DFlash/JANQ blocker loops route to a candidate-search gate that only reopens
  DFlash when the drafter candidate changes.
- Settled MTP block-size sweeps route to verify/cache/rollback instrumentation.
- Repeated clean runtime maps route to one scoped source-bridge candidate.

Every frontier-expansion task is no-model-load, canary-only, scoped to OpenClaw,
and carries explicit acceptance and rollback gates before any later promotion.

### Deterministic Lane Contracts

Each research lane declares the work it is allowed to do:

- prerequisites before it can run;
- known hard blockers;
- fallback or refocus route;
- memory/cost class;
- promotion gates;
- rollback requirements.

Examples:

- DFlash/JANQ compatibility is blocked until the exact drafter candidate and
  target-generated trace requirements are satisfied.
- Settled block-size sweeps stop reseeding repeated work once evidence shows the
  current block is the winner.
- Drafter calibration fails closed when memory constraints make local training
  unsafe, and uses stop-gradient target traces so quantized JANQ weights are not
  differentiated through.
- Runtime-overhead work is routed separately from raw decode-speed work so
  measurement contamination does not masquerade as model speed.

### Implementation Handoff

Research does not directly mutate the live setup. Implementation goes through a
gated bridge:

- implementation candidates must be scoped to OpenClaw-owned files;
- patch paths are allowlisted;
- `.env`, token, key, password, secret, opencode, and private config paths are
  denied;
- safe patches run in a canary workspace first;
- tests must pass before promotion;
- architectural changes require explicit approval;
- rollback information must be present.

### Self-Improvement Sidecar

The sidecar records trajectories, lessons, proposed skill updates, and evaluator
feedback. It is advisory by default: generated variants are held for review and
canary evaluation rather than mutating the active system automatically.

The goal is a stable evolutionary layer: the system can learn from repeated
failure modes and improve its own research instructions without destabilizing the
main OpenClaw runtime.

### Autoresearch Watchdog

`openclaw-autoresearch-watchdog` is the independent reviewer layer. It is
deliberately deterministic: it reads the active research workspace, checks the
autopilot lock, latest log freshness, quality review, frontier eval,
implementation handoff, stability burn-in, active noise, and decode metrics, then
writes a review artifact under `~/.openclaw/research/speed/watchdog/`.

The watchdog does not edit `tasks.jsonl` while autopilot owns the workspace. That
keeps the main loop stable and avoids hidden races. Its job is to do the
Codex-style health/quality review automatically and state the next deterministic
move: continue, repair routing, investigate a stall, or seed the next candidate.

The watchdog also writes an explicit architecture contract and advisory candidate
list into each report. These candidates are evidence-linked and marked
`allowed_for_live_queue=false`: the sidecar may propose, but only the autopilot
and patch-executor can mutate live tasks after their normal quality, canary, and
rollback gates pass. This keeps new ideas from becoming duplicate task noise.

Run once:

```bash
openclaw speed-research-watchdog --allow-degraded
```

Install the provided LaunchAgent template if you want it to run every 10 minutes:

```bash
cp launchagents/local.openclaw-autoresearch-watchdog.plist ~/Library/LaunchAgents/
launchctl bootstrap "gui/$UID" ~/Library/LaunchAgents/local.openclaw-autoresearch-watchdog.plist
launchctl kickstart -k "gui/$UID/local.openclaw-autoresearch-watchdog"
```

## Referenced Ideas And Repositories

This project is custom OpenClaw harness code, but several external projects and
methods inform the design:

- Karpathy `autoresearch`: small experiments, evidence-first notes,
  keep/discard decisions, and durable research loops.
  <https://github.com/karpathy/autoresearch>
- DSPy GEPA: reflective prompt/program optimization ideas used as inspiration
  for reviewer and policy refinement, while keeping active promotion gated.
  <https://github.com/stanfordnlp/dspy>
- Nous Research Hermes Agent: self-improvement and skill-memory concepts used
  as inspiration for the sidecar trajectory/lesson/proposal layer.
  <https://github.com/NousResearch/hermes-agent>
- Rapid-MLX: speed-oriented MLX serving ideas, prefix cache, batching, and
  backend performance direction.
  <https://github.com/raullenchai/Rapid-MLX>
- vMLX behavior: reference-only guidance for Gemma/JANG handling, streaming
  stability, thinking/tool separation, and loop avoidance.
- dFlash: speculative decoding and drafter-fit research direction for future
  decode-speed work.
  <https://github.com/z-lab/dflash>
- Google Gemma MTP documentation: official multi-token prediction/drafter
  concepts for Gemma-family decode speedups.
  <https://ai.google.dev/gemma/docs/mtp/mtp>
- Hugging Face model card guidance for the active Gemma 4 JANG/JANQ model family.

External code is not vendored here unless explicitly present in this repository.
These references are design inputs, not a claim that this repo is a fork of any
of them.

## Running Autoresearch

Short run:

```bash
openclaw speed-research --max-hours 1 --cycles 20
```

Overnight run:

```bash
openclaw speed-research --max-hours 12 --cycles 80
```

The cycle count is a tranche size. The supervisor auto-extends while useful work
remains and the max-hours budget has not expired. Stop with `Ctrl+C`; the
autopilot writes a neutral interrupt checkpoint and stops OpenClaw-owned model
processes when configured to do so.

Runtime evidence is written under the OpenClaw research workspace:

```text
~/.openclaw/research/speed
```

That runtime workspace is intentionally not committed to this repo.

## Health And Quality Checks

Common checks:

```bash
python3 openclaw/test-speed-research.py
python3 openclaw/test-speed-research-autopilot.py
python3 openclaw/test-self-improvement.py
python3 openclaw/test-autoresearch-watchdog.py
python3 -m compileall -q openclaw
zsh -n openclaw/openclaw-wrapper.zsh
```

Live no-model review checks:

```bash
~/.openclaw/bin/openclaw-speed-research quality-review --recent-rows 120 --min-sweeps 3 --min-samples-per-block 3 --target-tps 30
~/.openclaw/bin/openclaw-speed-research implementation-handoff-audit --min-score 90
~/.openclaw/bin/openclaw-speed-research frontier-eval --recent-rows 120 --min-score 9 --allow-fail
```

The frontier eval tracks:

- Karpathy-style core loop quality;
- crash and memory safety;
- research quality;
- implementation handoff;
- self-improvement;
- modularity.

## Repository Layout

```text
openclaw/
  openclaw-wrapper.zsh                 # user-facing OpenClaw wrapper
  openclaw-model-proxy.py              # OpenAI-compatible proxy guardrails
  openclaw-speed-research.py           # deterministic research helper commands
  openclaw-speed-research-autopilot.py # overnight supervisor loop
  openclaw-autoresearch-watchdog.py    # independent deterministic reviewer
  openclaw-drafter-fit.py              # JANQ drafter-fit promotion gates
  openclaw-mtp-drafter-calibrate.py    # bounded calibration helpers
  test-*.py                            # focused safety and behavior tests

launchagents/
  local.openclaw-model-server.plist    # optional macOS LaunchAgent template
  local.openclaw-autoresearch-watchdog.plist

docs/case-studies/
  *.md                                 # portfolio-readable case study logs
```

## Safety And Secret Hygiene

The repo is intended to contain source, tests, docs, and launch templates only.
Do not commit:

- `.env` or `.env.*`;
- API keys, passwords, OAuth tokens, or private certificates;
- private SSH keys;
- model cache files;
- runtime logs;
- local OpenClaw research artifacts;
- opencode config or runtime files.

The patch executor and implementation handoff also scan candidate patches for
secret-like content and deny unsafe paths before promotion.

Before pushing, run at minimum:

```bash
git status --short
find . -maxdepth 3 \( -name '.env*' -o -name '*secret*' -o -name '*token*' -o -name '*key*' -o -name '*.pem' -o -name '*.p12' -o -name 'id_rsa*' -o -name 'id_ed25519*' \) -not -path './.git/*' -print
rg -n "(API_KEY|SECRET|TOKEN|PASSWORD|BEGIN (RSA|OPENSSH|PRIVATE)|sk-[A-Za-z0-9])" --glob '!*.log' --glob '!**/.git/**'
```

Expected matches should be placeholder strings, denied-path tests, environment
variable names, or documentation examples only. Investigate anything that looks
like a real credential before committing.
