# Gemma 4 JANQ Drafter Training Case Study Log

## Problem Statement

Gemma 4 official multi-token prediction drafters promise large decode speedups,
but the active OpenClaw model is a JANQ/JANG variant. The target model should
remain unchanged, so any drafter or DFlash path must prove it fits the JANQ
target distribution before it is promoted to normal TUI chat.

The core question: can a drafter be calibrated or selected so the JANQ model
gets meaningful speculative decode speedup without breaking tool calls,
reasoning separation, streaming, or memory stability?

## Training Objective

Train or fit only the drafter/speculative helper path. Do not train or replace
the target JANQ model.

The desired drafter must:

- share the tokenizer and Gemma 4 structural assumptions;
- improve real wall-clock decode tok/s in OpenClaw TUI benchmarks;
- improve or preserve mean MTP acceptance;
- avoid repeated reasoning markers;
- avoid malformed tool JSON;
- pass stream and tool-call replay guards;
- be rejected automatically if it does not beat the current default.

## Current Fit Plan

OpenClaw has a JANQ DFlash drafter-fit plan that records structural compatibility
and promotion gates before any live promotion:

- target: `Gemma-4-31B-JANG_4M-CRACK`;
- candidate family: Gemma 4 DFlash/MTP drafter path;
- status: ready for target-generated trace data;
- default runtime policy: do not promote DFlash by default yet;
- promotion gate: speedup, acceptance, stream safety, tool safety, reasoning-loop
  safety, and rollback.

## Promotion Gates

A drafter is not promoted unless it clears all gates:

- wall-clock decode TPS improves over the current default;
- mean acceptance improves enough to justify verification overhead;
- tool-bearing requests stay on a safe path unless the drafter is proven safe;
- thinking/reasoning requests do not leak repeated markers;
- stream watchdog and malformed-output repair still pass;
- live OpenClaw profile can be restored immediately.

## Current Evidence

Recent autoresearch evidence after hardening:

- normal decode sample: about `14.2-14.7 tok/s`;
- MTP acceptance review: about `0.56-0.57`;
- MTP rounds: about `126-128`;
- earlier better samples reached about `15.7 tok/s`;
- current implementation bridge is not yet producing drafter patches.

Interpretation: the current drafter path is helping enough to produce stable
14-15 tok/s results, but acceptance/round-count evidence suggests the drafter is
not yet fit tightly enough to deliver the desired larger speedup. The next
training work should focus on increasing acceptance per verification round, not
just enabling a bigger drafter.

## Completed Autoresearch Run - 2026-05-06

The completed 80-cycle autoresearch run produced stable drafter-path evidence:

- 30 normal decode benchmark samples;
- mean decode speed: `14.499 tok/s`;
- min/max decode speed: `14.160-14.677 tok/s`;
- final MTP review: `mean_accept=0.56`, `mean_mtp_rounds=128.0`,
  `mean_tok_s=11.386`;
- repeated focused tests passed during the run;
- no gateway fallback loop was observed in the final run.

The bottleneck is now more specific: the current drafter path is stable but not
accepting enough tokens per verification round to reach the desired 30+ tok/s
range. The autoresearch loop repeatedly identified `mtp-acceptance-bottleneck`,
`drafter-block-and-quant-sweep`, and `mlx-vlm-mtp-loop-overhead` as top-ranked
ideas, but it did not implement them.

Next training step: execute the drafter sweep deterministically instead of
generating another plan. The promotion gate should compare each variant against
the current live default and reject every change unless wall-clock decode speed,
acceptance, stream safety, tool safety, and reasoning-loop guards all pass.

## Deterministic Drafter Sweep - 2026-05-06

The block-size search is now an executable supervisor task, not an instruction
for the model to remember. `drafter-sweep-run` runs the current control block
and candidate block sizes on the same decode benchmark, records decode TPS,
MTP acceptance, MTP rounds, replay status, and a promotion decision.

Promotion remains gated: no live TUI default changes unless the candidate beats
the current control by the configured decode TPS delta and replay guards pass.
This keeps overnight research useful without letting a faster-but-unsafe drafter
silently enter normal OpenClaw chat.

## Training Loop Design

The drafter training/calibration loop should be staged:

1. collect target-generated traces from JANQ prompts;
2. evaluate baseline drafter acceptance and decode speed;
3. tune only the drafter or adapter components;
4. re-evaluate acceptance on held-out JANQ traces;
5. benchmark normal TUI decode speed;
6. replay tool/thinking/stream guards;
7. promote only if speed and safety both improve.

## Crash And Memory Guardrails

The MTP calibrator now fails closed:

- memory is checked before loading the target model;
- MLX memory/cache limits are set conservatively;
- memory is checked after load, after trace building, and during training;
- unsafe runs write `openclaw-calibration-blocked.json`;
- Python dependency/config failures are converted into structured blocked
  artifacts instead of terminal tracebacks.

## Open Questions

- What no-drafter control speed should be used as the true lower bound?
- What is the exact acceptance threshold where MTP becomes faster than target-only
  decode for this runtime?
- Which drafter block size gives the best acceptance/overhead tradeoff?
- Does DFlash need JANQ-generated trace data, adapter tuning, or selective layer
  tuning to fit the abliterated target distribution?
- Can calibration improve acceptance without changing the model's tool/reasoning
  behavior?

## Portfolio Angle

This case study is about adapting speculative decoding to a customized local
model. The story is the difference between "turn on a drafter" and building a
measured promotion gate that proves the drafter is faster, safer, and aligned
with the exact target model.
