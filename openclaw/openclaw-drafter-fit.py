#!/usr/bin/env python3
"""JANQ drafter-fit gates for OpenClaw speculative decoding.

This utility does not train or serve a model by itself. It creates the
deterministic evidence we need before allowing a drafter change into the normal
OpenClaw TUI path: exact target pairing, target-generated data requirements,
paired benchmark promotion, and loop/tool safety gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_TARGET_LAYER_IDS = [1, 12, 23, 35, 46, 57]
DEFAULT_DFLASH_DRAFT = "z-lab/gemma-4-31B-it-DFlash"
DEFAULT_MIN_SPEEDUP = 1.35
DEFAULT_MIN_ACCEPT = 2.25


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_config_hash(model_path: Path) -> str:
    config = model_path / "config.json"
    if not config.exists():
        raise FileNotFoundError(f"missing target config: {config}")
    return file_sha256(config)


def nested_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config


def load_model_config(model_path: Path) -> dict[str, Any]:
    config = read_json(model_path / "config.json")
    text = nested_text_config(config)
    return {
        "model_type": text.get("model_type") or config.get("model_type"),
        "hidden_size": text.get("hidden_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
        "vocab_size": text.get("vocab_size"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "final_logit_softcapping": text.get("final_logit_softcapping"),
        "config_sha256": model_config_hash(model_path),
    }


def load_drafter_config(drafter_path: Path | None) -> dict[str, Any]:
    if drafter_path is None:
        return {"source": DEFAULT_DFLASH_DRAFT, "local_config": False}
    config_path = drafter_path / "config.json"
    if not config_path.exists():
        return {"source": str(drafter_path), "local_config": False}
    config = read_json(config_path)
    dflash_config = config.get("dflash_config") if isinstance(config.get("dflash_config"), dict) else {}
    return {
        "source": str(drafter_path),
        "local_config": True,
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "final_logit_softcapping": config.get("final_logit_softcapping"),
        "target_layer_ids": config.get("target_layer_ids") or dflash_config.get("target_layer_ids"),
        "num_target_layers": config.get("num_target_layers"),
        "block_size": config.get("block_size"),
        "config_sha256": file_sha256(config_path),
    }


def structural_mismatches(target: dict[str, Any], drafter: dict[str, Any]) -> list[str]:
    if not drafter.get("local_config"):
        return ["drafter config is not local; download/inspect before compatibility can be trusted"]
    mismatches: list[str] = []
    for key in ("hidden_size", "vocab_size", "max_position_embeddings", "final_logit_softcapping"):
        if target.get(key) != drafter.get(key):
            mismatches.append(f"{key}: target={target.get(key)!r} drafter={drafter.get(key)!r}")
    target_layers = int(target.get("num_hidden_layers") or 0)
    if target_layers <= 0:
        mismatches.append("target num_hidden_layers is unavailable")
    if drafter.get("num_target_layers") not in (None, target_layers):
        mismatches.append(
            f"num_target_layers: target={target_layers} drafter={drafter.get('num_target_layers')!r}"
        )
    target_layer_ids = drafter.get("target_layer_ids") or []
    if not target_layer_ids:
        mismatches.append("target_layer_ids are unavailable")
    for layer in target_layer_ids:
        if int(layer) < 0 or int(layer) >= target_layers:
            mismatches.append(f"target_layer_id out of range: {layer}")
    return mismatches


def plan(args: argparse.Namespace) -> int:
    target_path = Path(args.target_path).expanduser()
    drafter_path = Path(args.drafter_path).expanduser() if args.drafter_path else None
    target = load_model_config(target_path)
    drafter = load_drafter_config(drafter_path)
    mismatches = structural_mismatches(target, drafter) if drafter.get("local_config") else []
    layer_ids = drafter.get("target_layer_ids") or DEFAULT_TARGET_LAYER_IDS
    output = {
        "ok": True,
        "kind": "janq-drafter-fit-plan",
        "created": int(time.time()),
        "target_path": str(target_path),
        "target": target,
        "drafter": drafter,
        "structural_mismatches": mismatches,
        "decision": "blocked" if mismatches else "ready-for-target-generated-trace-data",
        "training_requirements": {
            "target_model_must_generate_data": True,
            "reason": "Speculative acceptance depends on matching the exact JANQ target distribution.",
            "recommended_target_layer_ids": layer_ids,
            "speculator_type": "dflash",
            "block_size": int(args.block_size),
            "draft_layers": int(args.draft_layers),
            "seed_data": "OpenClaw coding/tool/chat prompts regenerated by the JANQ target, not copied from generic datasets.",
        },
        "promotion_gate": {
            "minimum_speedup_vs_current": float(args.min_speedup),
            "minimum_mean_accept": float(args.min_accept),
            "must_pass_tool_and_thinking_replay": True,
            "must_show_no_repeated_reasoning_markers": True,
            "must_restore_default_mtp_on_failure": True,
            "normal_tui_default_remains_mtp_until_promoted": True,
        },
        "runtime_policy": {
            "dflash_default": False,
            "requires_OPENCLAW_JANG_DFLASH_EXPERIMENTAL_ACK": True,
            "tools_and_thinking_bypassed_until_promoted": True,
        },
    }
    if args.output:
        write_json(Path(args.output).expanduser(), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not mismatches else 2


def metric_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return default
    return default


def benchmark_decode_tps(result: dict[str, Any]) -> float:
    return metric_number(result.get("decode_tps") or result.get("decode_tps_estimate"))


def benchmark_mean_accept(result: dict[str, Any]) -> float:
    mtp = result.get("mtp")
    if isinstance(mtp, dict):
        return metric_number(mtp.get("mean_accept"))
    return metric_number(result.get("mean_accept"))


def guard_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in (
        "repeated_reasoning_markers",
        "repeated_token_loop",
        "malformed_tool_calls",
        "tool_call_loop",
        "stream_errors",
        "python_crashes",
        "metal_crashes",
        "memory_pressure_blocks",
    ):
        value = report.get(key, 0)
        if isinstance(value, bool):
            if value:
                failures.append(key)
        elif metric_number(value) > 0:
            failures.append(key)
    status = str(report.get("status") or "").lower()
    if status and status not in {"ok", "pass", "passed"}:
        failures.append(f"status={status}")
    return failures


def decide(args: argparse.Namespace) -> int:
    baseline = read_json(Path(args.baseline).expanduser())
    candidate = read_json(Path(args.candidate).expanduser())
    guard = read_json(Path(args.guard_report).expanduser()) if args.guard_report else {}
    baseline_tps = benchmark_decode_tps(baseline)
    candidate_tps = benchmark_decode_tps(candidate)
    mean_accept = benchmark_mean_accept(candidate)
    required_tps = max(baseline_tps + float(args.min_delta), baseline_tps * float(args.min_speedup))
    reasons: list[str] = []
    if baseline_tps <= 0:
        reasons.append("baseline decode_tps is unavailable")
    if candidate_tps < required_tps:
        reasons.append(f"candidate decode_tps {candidate_tps:.3f} < required {required_tps:.3f}")
    if mean_accept < float(args.min_accept):
        reasons.append(f"candidate mean_accept {mean_accept:.3f} < required {float(args.min_accept):.3f}")
    if baseline.get("model") and candidate.get("model") and baseline.get("model") != candidate.get("model"):
        reasons.append(f"model changed: baseline={baseline.get('model')!r} candidate={candidate.get('model')!r}")
    for failure in guard_failures(guard):
        reasons.append(f"guard failure: {failure}")
    decision = "promote" if not reasons else "reject"
    output = {
        "ok": decision == "promote",
        "decision": decision,
        "created": int(time.time()),
        "baseline_decode_tps": baseline_tps,
        "candidate_decode_tps": candidate_tps,
        "candidate_mean_accept": mean_accept,
        "required_decode_tps": required_tps,
        "speedup": round(candidate_tps / baseline_tps, 3) if baseline_tps > 0 else 0,
        "reasons": reasons,
        "runtime_policy": {
            "promote_to_default_tui": decision == "promote",
            "keep_dflash_ack_gated": decision != "promote",
            "keep_tools_thinking_bypassed": decision != "promote",
        },
    }
    if args.output:
        write_json(Path(args.output).expanduser(), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if decision == "promote" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether a drafter fits OpenClaw's JANQ target.")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--target-path", required=True)
    plan_parser.add_argument("--drafter-path")
    plan_parser.add_argument("--output")
    plan_parser.add_argument("--block-size", type=int, default=16)
    plan_parser.add_argument("--draft-layers", type=int, default=5)
    plan_parser.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    plan_parser.add_argument("--min-accept", type=float, default=DEFAULT_MIN_ACCEPT)
    plan_parser.set_defaults(func=plan)

    decide_parser = sub.add_parser("decide")
    decide_parser.add_argument("--baseline", required=True)
    decide_parser.add_argument("--candidate", required=True)
    decide_parser.add_argument("--guard-report")
    decide_parser.add_argument("--output")
    decide_parser.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    decide_parser.add_argument("--min-delta", type=float, default=0.5)
    decide_parser.add_argument("--min-accept", type=float, default=DEFAULT_MIN_ACCEPT)
    decide_parser.set_defaults(func=decide)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
