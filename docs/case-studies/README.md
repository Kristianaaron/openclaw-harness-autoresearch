# OpenClaw Case Study Logs

These notes are a side log for portfolio case studies. They are separate from
the live autoresearch workspace under `~/.openclaw/research/speed`, which is
runtime evidence and may be noisy during overnight runs.

## Tracks

- `openclaw-speed-harness-case-study.md` tracks the harness, runtime, memory,
  gateway, streaming, tool-calling, and autoresearch bridges.
- `gemma-janq-drafter-training-case-study.md` tracks Gemma 4 JANQ drafter fit,
  MTP acceptance, DFlash compatibility, calibration, and promotion gates.

## Update Rule

Append dated entries when a change creates useful evidence:

- baseline or improved tok/s
- prefill or TTFT change
- memory/crash prevention
- gateway or tool-loop recovery
- benchmark methodology improvement
- drafter fit or training result
- rejected experiment with a clear reason

Keep these logs portfolio-readable: problem, action, evidence, result, and next
question. Avoid raw secrets, local tokens, or private environment details.

## GitHub Attribution Check

Future commits should use the GitHub-linked noreply author identity so pushed
OpenClaw harness work appears in the account contribution graph.
