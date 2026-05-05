# OpenClaw Harness Autoresearch

OpenClaw-only local harness hardening for Gemma 4 31B JANG/JANQ on Apple Silicon.

This repo contains the OpenClaw model profile layer, Rapid-MLX/JANG launchers, proxy guardrails, prefix warming, memory gates, and the speed autoresearch runner.

## Scope

- OpenClaw only.
- No opencode configuration or runtime files.
- Model/profile-driven backend selection.
- Local MLX/Rapid-MLX focused reliability and speed experiments.

## Autoresearch

Run the autonomous speed research loop:

```bash
openclaw speed-research-auto
```

Longer overnight run:

```bash
openclaw speed-research-auto --max-hours 10 --cycles 80
```

Results are written under:

```text
~/.openclaw/research/speed
```
