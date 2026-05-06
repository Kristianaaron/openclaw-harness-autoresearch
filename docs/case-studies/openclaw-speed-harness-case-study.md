# OpenClaw Speed Harness Case Study Log

## Problem Statement

OpenClaw is being hardened into a local frontier-style agent harness for a
31B-class Gemma 4 JANQ model on Apple Silicon. The original user experience had
three blocking failures:

- long-horizon tasks could stall, loop, or stop after a partial result;
- local model runs could trigger memory, Python, or Metal instability;
- speed work lacked a deterministic loop that converted measurements into
  implementation candidates.

The goal is a usable build harness: normal `openclaw tui` should feel stable,
recover from gateway/model/runtime issues, measure speed honestly, and improve
itself without requiring the user to babysit every cycle.

## System Under Study

- Target experience: normal OpenClaw TUI chat and agentic work.
- Current model: `Gemma-4-31B-JANG_4M-CRACK`.
- Runtime path: OpenClaw model proxy plus local Gemma/JANG MLX server.
- Research loop: `openclaw-speed-research-autopilot`.
- Main metric: real TUI-relevant decode tok/s on normal prompts.
- Secondary metrics: TTFT, wall time, memory pressure, MTP acceptance,
  tool-call safety, gateway stability, and number of autonomous useful cycles.

## Bridge Work

### Model Bridge

OpenClaw now keeps the gateway model-agnostic and lets model profiles select the
backend. This made it possible to keep the JANQ model stable while iterating on
serving/runtime choices.

### Reliability Bridge

The harness now treats gateway, memory, stream, and tool-call failures as
recoverable system states instead of letting the model improvise recovery.
Important guardrails added during this phase:

- memory preflight gates before large local runs;
- active memory circuit breakers during long turns;
- bounded tool-result caps;
- malformed hidden/tool output recovery;
- broad local command blocking;
- repeated implementation stall blocking;
- gateway health checks before agent turns;
- gateway restart/recovery after embedded fallback or WebSocket failure.

### Research Bridge

The research loop moved away from fully open-ended LLM research and toward a
supervisor-owned loop:

1. deterministic benchmark or log-review task;
2. structured evidence row;
3. synthesis of ranked ideas;
4. gated implementation candidate;
5. focused tests and rollback path.

This reduced hallucinated progress and made stalled work visible as a blocked
artifact rather than a silent hang.

## Evidence Timeline

| Date | Change | Evidence | Result |
| --- | --- | --- | --- |
| 2026-05-06 | Gateway recovery added to autoresearch autopilot | Reproduced state where model endpoint was live but gateway was down; recovery started gateway and `/health` returned live | Fixed the cycle 22-80 `gateway embedded fallback` failure mode |
| 2026-05-06 | MTP calibration memory guards added | Unsafe calibrator runs now exit with code `2` and write `openclaw-calibration-blocked.json` | Python/Metal-heavy drafter calibration fails closed instead of crashing or hanging |
| 2026-05-06 | Supervisor-driven drafter/research tasks added | Canary ran supervisor drafter sweep, fit plan, and focused test without starting unnecessary model work | Reduced LLM tool-loop risk for repeated research tasks |
| 2026-05-06 | Current autoresearch run after gateway fix | Decode repeatability rows around `14.2-14.7 tok/s`; gateway and model endpoints live | Run is healthy enough to continue, but implementation bridge is still weak |

## Current Speed Snapshot

Observed after gateway recovery hardening:

- decode sample range: about `14.2-14.7 tok/s`;
- repeatability mean examples: `14.549`, `14.554`, `14.500`, `14.482`, `14.471 tok/s`;
- wall time for 96-token decode sample: about `6.5-6.8s`;
- MTP acceptance review: about `0.56-0.57`;
- MTP rounds: about `126-128` in recent log reviews.

Interpretation: speed is stable enough to benchmark, but the MTP path still
looks inefficient. Moderate acceptance with high MTP round counts suggests
drafting overhead may be eating a meaningful part of the theoretical speedup.

## Completed Autoresearch Run - 2026-05-06

The overnight/autonomous run completed all 80 requested cycles after the gateway
recovery hardening.

- result rows after start: 80 `keep`, 10 `blocked`;
- decode benchmark samples: 30;
- mean decode speed: `14.499 tok/s`;
- min/max decode speed: `14.160-14.677 tok/s`;
- final 5 decode samples: `14.585`, `14.659`, `14.280`, `14.586`,
  `14.583 tok/s`;
- final MTP review: `mean_accept=0.56`, `mean_mtp_rounds=128.0`;
- gateway embedded fallback loop did not recur.

Quality judgment: the run was healthy as a supervisor/benchmark loop and useful
as measurement evidence. It was not yet strong enough as an implementation loop:
every implementation bridge task blocked with `no durable artifact`, and the
drafter block-size work produced repeatable sweep plans rather than executing a
paired sweep.

Next engineering step: make the drafter block-size sweep supervisor-owned and
executable, with paired measurements for block sizes 1, 2, 3, and 4, plus a
no-drafter control if the backend can expose one safely. Do not rely on the LLM
to turn the sweep plan into benchmark execution.

## Open Questions

- Can the supervisor execute the full drafter block-size sweep rather than only
  producing sweep plans?
- Can MTP acceptance logging be made complete for every benchmark row?
- Can implementation tasks become supervisor-owned enough to produce patches
  rather than `no durable artifact` blocks?
- What is the no-drafter control speed, measured with the same prompt set and
  rollback guarantee?
- Which part of the current 14-15 tok/s limit is model decode, MTP overhead,
  Python/MLX loop overhead, or memory pressure?

## Portfolio Angle

This case study is about building a local agent harness that behaves like a
frontier coding tool under resource constraints. The core story is not only
speed; it is turning fragile local inference into a measured, recoverable,
self-improving system.
