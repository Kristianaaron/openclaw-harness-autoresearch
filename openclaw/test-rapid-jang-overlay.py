#!/usr/bin/env python3
"""Static smoke checks for the OpenClaw Rapid/JANG bridge."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import patch


BRIDGE_PATH = Path(__file__).parent / "rapid-overlay" / "openclaw_rapid_jang.py"


def load_bridge():
    spec = importlib.util.spec_from_file_location("openclaw_rapid_jang", BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BRIDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def main() -> int:
    bridge = load_bridge()
    source = BRIDGE_PATH.read_text(encoding="utf-8")
    assert "generation_config.json" in source
    assert "_openclaw_model_path" in source
    assert "len(per_request_caches) == 1" in source
    assert "chunking long Gemma4 prefill" in source
    assert "exceeds safe limit" not in source
    assert 'OPENCLAW_RAPID_MLLM_REPETITION_PENALTY", 1.0' in source
    assert "original_step_impl" in source
    assert "OpenClawGemma4TextWrapper" in source
    assert "get_input_embeddings" in source
    assert "_patch_gemma4_text_prompt_boundary" in source
    assert "generate_with_gemma4_full_prompt" in source
    assert "next_with_gemma4_full_prompt" in source
    assert "BatchGenerator._next = next_with_gemma4_full_prompt" in source
    assert "GenerationBatch.__new__(GenerationBatch)" in source
    assert "OpenClawSingleRequestCacheAdapter" in source
    assert "prompt_batch.prompt_cache = _wrap_single_request_caches(caches)" in source
    assert "init_with_gemma4_stop_tokens" in source
    assert "openclaw_stop_token_ids" in source

    assert bridge._looks_repeated_text("OK\n{OKOKOKOKOKOKOKOK")
    assert bridge._clean_repeated_text("OK\n{OKOKOKOKOKOKOKOK") == "OK"
    assert bridge._looks_repeated_text("thought thought thought thought thought")
    assert not bridge._looks_repeated_text("OpenClaw is ready to help with a project.")

    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp)
        write_json(model / "config.json", {"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}})
        write_json(model / "jang_config.json", {"format": "jang", "format_version": "2.0"})
        assert bridge.is_jang_model(model)
        assert bridge._is_gemma4(model)
        assert not bridge._is_jangtq(bridge._jang_config(model))

        calls: list[str] = []

        def fake_load_jang_for_llm(path, tokenizer_config=None):
            calls.append(f"llm:{Path(path).name}")
            return "model", "tokenizer"

        vllm_mlx = types.ModuleType("vllm_mlx")
        utils = types.ModuleType("vllm_mlx.utils")
        tokenizer_module = types.ModuleType("vllm_mlx.utils.tokenizer")
        tokenizer_module.load_model_with_fallback = lambda model_name, tokenizer_config=None: ("orig-model", "orig-tokenizer")
        vllm_mlx.utils = utils
        utils.tokenizer = tokenizer_module
        modules = {
            "vllm_mlx": vllm_mlx,
            "vllm_mlx.utils": utils,
            "vllm_mlx.utils.tokenizer": tokenizer_module,
        }
        with patch.dict("sys.modules", modules):
            with patch.object(bridge, "load_jang_for_llm", fake_load_jang_for_llm):
                bridge._patch_tokenizer_loader()
                assert tokenizer_module.load_model_with_fallback(str(model)) == ("model", "tokenizer")
                assert calls == [f"llm:{model.name}"]

    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp)
        write_json(model / "config.json", {"model_type": "gemma4"})
        write_json(model / "jang_config.json", {"weight_format": "mxtq"})
        assert bridge._is_jangtq(bridge._jang_config(model))

    print("ok openclaw Rapid JANG overlay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
