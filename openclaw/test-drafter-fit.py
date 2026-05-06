#!/usr/bin/env python3
"""Checks for JANQ drafter-fit promotion gates."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from argparse import Namespace
from pathlib import Path


HELPER_PATH = Path(__file__).with_name("openclaw-drafter-fit.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("openclaw_drafter_fit", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def main() -> int:
    helper = load_helper()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "target"
        draft = root / "draft"
        write_json(
            target / "config.json",
            {
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 5376,
                    "num_hidden_layers": 60,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 16,
                    "head_dim": 256,
                    "vocab_size": 262144,
                    "max_position_embeddings": 262144,
                    "final_logit_softcapping": 30.0,
                }
            },
        )
        write_json(
            draft / "config.json",
            {
                "model_type": "qwen3",
                "hidden_size": 5376,
                "num_hidden_layers": 5,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "final_logit_softcapping": 30.0,
                "dflash_config": {"target_layer_ids": [1, 12, 23, 35, 46, 57]},
                "num_target_layers": 60,
                "block_size": 16,
            },
        )
        plan_path = root / "plan.json"
        assert helper.plan(
            Namespace(
                target_path=str(target),
                drafter_path=str(draft),
                output=str(plan_path),
                block_size=16,
                draft_layers=5,
                min_speedup=1.35,
                min_accept=2.25,
            )
        ) == 0
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        assert plan["decision"] == "ready-for-target-generated-trace-data"
        assert plan["training_requirements"]["target_model_must_generate_data"] is True
        assert plan["runtime_policy"]["dflash_default"] is False
        assert plan["promotion_gate"]["must_pass_tool_and_thinking_replay"] is True

        baseline = root / "baseline.json"
        candidate = root / "candidate.json"
        guard = root / "guard.json"
        write_json(baseline, {"ok": True, "model": "Gemma-4-31B-JANG_4M-CRACK", "decode_tps": 15.0})
        write_json(
            candidate,
            {
                "ok": True,
                "model": "Gemma-4-31B-JANG_4M-CRACK",
                "decode_tps": 45.5,
                "mtp": {"mean_accept": 3.4},
            },
        )
        write_json(guard, {"status": "ok", "repeated_token_loop": 0, "tool_call_loop": 0, "stream_errors": 0})
        decision_path = root / "decision.json"
        assert helper.decide(
            Namespace(
                baseline=str(baseline),
                candidate=str(candidate),
                guard_report=str(guard),
                output=str(decision_path),
                min_speedup=2.5,
                min_delta=0.5,
                min_accept=2.25,
            )
        ) == 0
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        assert decision["decision"] == "promote"
        assert decision["runtime_policy"]["promote_to_default_tui"] is True

        write_json(candidate, {"ok": True, "model": "Gemma-4-31B-JANG_4M-CRACK", "decode_tps": 20.0, "mtp": {"mean_accept": 1.1}})
        write_json(guard, {"status": "ok", "repeated_reasoning_markers": True})
        assert helper.decide(
            Namespace(
                baseline=str(baseline),
                candidate=str(candidate),
                guard_report=str(guard),
                output=None,
                min_speedup=2.5,
                min_delta=0.5,
                min_accept=2.25,
            )
        ) == 2
    print("ok openclaw drafter fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
