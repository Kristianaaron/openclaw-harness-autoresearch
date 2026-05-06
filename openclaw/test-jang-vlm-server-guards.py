#!/usr/bin/env python3
"""Unit checks for OpenClaw's JANG/VLM server safety shims."""

from __future__ import annotations

import importlib.util
import json
import os
from types import SimpleNamespace
from pathlib import Path


SERVER_PATH = Path(__file__).with_name("openclaw-jang-vlm-server.py")
spec = importlib.util.spec_from_file_location("openclaw_jang_vlm_server", SERVER_PATH)
assert spec is not None and spec.loader is not None
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_native_gemma_tool_call() -> None:
    text = '<|tool_call>call:lookup_obsidian_memory{limit:3,topic:<|"|>obsidian memory setup<|"|>}<tool_call|>'
    content, calls = server.parse_tool_call(text)
    assert content == ""
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "lookup_obsidian_memory"
    args = json.loads(call["function"]["arguments"])
    assert args == {"limit": 3, "topic": "obsidian memory setup"}


def test_json_tool_call() -> None:
    text = '<tool_call>{"name":"lookup","arguments":{"topic":"memory"}}</tool_call>'
    content, calls = server.parse_tool_call(text)
    assert content == ""
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"topic": "memory"}


def test_reasoning_and_loop_guards() -> None:
    clean, reasoning = server.strip_reasoning_markers("hello <think>hidden</think> world")
    assert clean == "hello  world"
    assert reasoning == "hidden"
    assert server.has_repeated_token_loop("thoughtthought")
    assert server.has_repeated_token_loop("thoughtthoughtthoughtthought")


def test_dflash_adapter_and_kwargs() -> None:
    class FakeLanguageModel:
        def __init__(self) -> None:
            self.layers = [object()]
            self.model = SimpleNamespace(embed_tokens=SimpleNamespace(as_linear="lm-head"))

        def make_cache(self):
            return ["cache"]

    class FakeTarget:
        def __init__(self) -> None:
            self.language_model = FakeLanguageModel()
            self.calls = []

        def __call__(self, input_ids, cache=None, **kwargs):
            self.calls.append((input_ids, cache, kwargs))
            return SimpleNamespace(logits="logits")

    target = FakeTarget()
    adapter = server.DFlashVLMTargetAdapter(target)
    assert adapter.make_cache() == ["cache"]
    assert adapter("tokens", cache=["cache"]) == "logits"
    assert target.calls == [("tokens", ["cache"], {})]

    old_draft = server.DRAFT_MODEL
    old_backend = server.DRAFT_BACKEND
    old_env = os.environ.get("OPENCLAW_JANG_DFLASH_BLOCK_SIZE")
    old_allow_tools = os.environ.get("OPENCLAW_JANG_DFLASH_ALLOW_TOOLS")
    old_allow_thinking = os.environ.get("OPENCLAW_JANG_DFLASH_ALLOW_THINKING")
    try:
        server.DRAFT_MODEL = object()
        server.DRAFT_BACKEND = "dflash"
        os.environ["OPENCLAW_JANG_DFLASH_BLOCK_SIZE"] = "16"
        assert server.should_use_dflash({"messages": [{"role": "user", "content": "hi"}]})
        assert not server.should_use_dflash({"tools": [{"type": "function"}]})
        assert not server.should_use_dflash({"enable_thinking": True})
        os.environ["OPENCLAW_JANG_DFLASH_ALLOW_TOOLS"] = "1"
        os.environ["OPENCLAW_JANG_DFLASH_ALLOW_THINKING"] = "1"
        assert server.should_use_dflash({"tools": [{"type": "function"}], "enable_thinking": True})
        kwargs = server.generation_kwargs({"max_tokens": 32})
        assert "draft_model" not in kwargs
        assert "draft_kind" not in kwargs
        assert server.dflash_generation_kwargs({"max_tokens": 32}) == {
            "max_tokens": 32,
            "temperature": 0.0,
            "block_size": 16,
        }
        server.DFLASH_ACCEPT_LENS[:] = [2, 8, 4]
        assert server.current_speculative_stat_index() == 3
        assert server.speculative_stats_since(1) == " mtp_rounds=2 mean_accept=6.00"
        server.DFLASH_ACCEPT_LENS[:] = []
        server.record_dflash_acceptance(SimpleNamespace(accepted=16), first_response=True)
        server.record_dflash_acceptance(SimpleNamespace(accepted=8), first_response=False)
        assert server.DFLASH_ACCEPT_LENS == [8]
        target_cfg = SimpleNamespace(
            model_type="gemma4_text",
            hidden_size=5376,
            vocab_size=262144,
            max_position_embeddings=262144,
            final_logit_softcapping=30.0,
            num_hidden_layers=60,
        )
        draft_cfg = SimpleNamespace(
            hidden_size=5376,
            vocab_size=262144,
            max_position_embeddings=262144,
            final_logit_softcapping=30.0,
            num_target_layers=60,
            dflash_config={"target_layer_ids": (1, 12, 23, 35, 46, 57)},
        )
        server.validate_dflash_compatibility(
            SimpleNamespace(language_model=SimpleNamespace(config=target_cfg)),
            SimpleNamespace(config=draft_cfg),
        )
        draft_cfg.hidden_size = 4096
        try:
            server.validate_dflash_compatibility(
                SimpleNamespace(language_model=SimpleNamespace(config=target_cfg)),
                SimpleNamespace(config=draft_cfg),
            )
        except RuntimeError as error:
            assert "hidden_size" in str(error)
        else:
            raise AssertionError("DFlash structural mismatch should fail")
    finally:
        server.DRAFT_MODEL = old_draft
        server.DRAFT_BACKEND = old_backend
        server.DFLASH_ACCEPT_LENS[:] = []
        if old_env is None:
            os.environ.pop("OPENCLAW_JANG_DFLASH_BLOCK_SIZE", None)
        else:
            os.environ["OPENCLAW_JANG_DFLASH_BLOCK_SIZE"] = old_env
        for key, value in {
            "OPENCLAW_JANG_DFLASH_ALLOW_TOOLS": old_allow_tools,
            "OPENCLAW_JANG_DFLASH_ALLOW_THINKING": old_allow_thinking,
        }.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    test_native_gemma_tool_call()
    test_json_tool_call()
    test_reasoning_and_loop_guards()
    test_dflash_adapter_and_kwargs()
    print("ok")
