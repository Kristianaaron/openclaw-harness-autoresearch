#!/usr/bin/env python3
"""Unit checks for OpenClaw's JANG/VLM server safety shims."""

from __future__ import annotations

import importlib.util
import json
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


if __name__ == "__main__":
    test_native_gemma_tool_call()
    test_json_tool_call()
    test_reasoning_and_loop_guards()
    print("ok")
