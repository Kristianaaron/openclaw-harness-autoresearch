#!/usr/bin/env python3
"""Lightweight checks for OpenClaw's local model proxy guardrails."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


PROXY_PATH = Path(__file__).with_name("openclaw-model-proxy.py")


def load_proxy():
    spec = importlib.util.spec_from_file_location("openclaw_vmlx_proxy", PROXY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PROXY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    mode = "timeout"
    post_count = 0
    received_payloads: list[dict] = []

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        body = b'{"data":[{"id":"fake"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        type(self).post_count += 1
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        type(self).received_payloads.append(payload)
        if self.mode == "timeout":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            time.sleep(1.2)
            return
        if self.mode == "tool_runaway" and payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for index in range(8):
                event = {
                    "id": "fake-tool-runaway",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fake",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": f"call_{index}",
                                        "type": "function",
                                        "function": {
                                            "name": "exec",
                                            "arguments": f'{{"command":"pwd # probe {index}"}}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                try:
                    self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            time.sleep(0.2)
            return
        if self.mode == "broad_tool" and payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            event = {
                "id": "fake-broad-tool",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_broad",
                                    "type": "function",
                                    "function": {
                                        "name": "exec",
                                        "arguments": '{"command":"ls -R /Users/kristian/.openclaw"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
            self.wfile.flush()
            time.sleep(0.2)
            return
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            event = {
                "id": "fake-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fake",
                "choices": [{"index": 0, "delta": {"content": "thought " * 40}, "finish_reason": None}],
            }
            self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        body = json.dumps(
            {
                "id": "fake-nonstream",
                "object": "chat.completion",
                "created": 1,
                "model": "fake",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "RECOVERED"}, "finish_reason": "stop"}],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_server(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def post_json(url: str, payload: dict) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.read().decode("utf-8")


def test_proxy_stream_timeout(proxy) -> None:
    FakeUpstreamHandler.mode = "timeout"
    FakeUpstreamHandler.post_count = 0
    FakeUpstreamHandler.received_payloads = []
    upstream = start_server(FakeUpstreamHandler)
    proxy_server = start_server(proxy.ProxyHandler)
    proxy_server.upstream_base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    try:
        env = {
            "OPENCLAW_VMLX_STREAM_IDLE_SECONDS": "0.1",
            "OPENCLAW_VMLX_FIRST_EVENT_TIMEOUT_SECONDS": "0.3",
            "OPENCLAW_VMLX_STREAM_TOTAL_TIMEOUT_SECONDS": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            body = post_json(
                f"http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions",
                {"model": "fake", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
            )
        assert "OpenClaw model proxy timeout" in body
        assert "no upstream event within" in body
    finally:
        proxy_server.shutdown()
        upstream.shutdown()
        proxy_server.server_close()
        upstream.server_close()


def test_proxy_stream_recovery(proxy) -> None:
    FakeUpstreamHandler.mode = "malformed"
    FakeUpstreamHandler.post_count = 0
    FakeUpstreamHandler.received_payloads = []
    upstream = start_server(FakeUpstreamHandler)
    proxy_server = start_server(proxy.ProxyHandler)
    proxy_server.upstream_base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    try:
        body = post_json(
            f"http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions",
            {"model": "fake", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert "RECOVERED" in body
        assert "thought thought" not in body
        assert FakeUpstreamHandler.post_count == 2
    finally:
        proxy_server.shutdown()
        upstream.shutdown()
        proxy_server.server_close()
        upstream.server_close()


def test_proxy_tool_runaway_guard(proxy) -> None:
    FakeUpstreamHandler.mode = "tool_runaway"
    FakeUpstreamHandler.post_count = 0
    FakeUpstreamHandler.received_payloads = []
    upstream = start_server(FakeUpstreamHandler)
    proxy_server = start_server(proxy.ProxyHandler)
    proxy_server.upstream_base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    try:
        env = {
            "OPENCLAW_MODEL_MAX_STREAM_TOOL_CALLS": "2",
            "OPENCLAW_MODEL_MAX_STREAM_TOOL_CHUNKS": "6",
            "OPENCLAW_VMLX_STREAM_TOTAL_TIMEOUT_SECONDS": "5",
        }
        with patch.dict(os.environ, env, clear=False):
            body = post_json(
                f"http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions",
                {
                    "model": "fake",
                    "stream": True,
                    "messages": [{"role": "user", "content": "search my notes"}],
                    "tools": [{"type": "function", "function": {"name": "exec", "parameters": {"type": "object"}}}],
                },
            )
        assert "OpenClaw model proxy timeout" not in body
        assert '"finish_reason":"tool_calls"' in body
        assert body.count('"delta":{"tool_calls"') == 2
        assert FakeUpstreamHandler.received_payloads[0]["parallel_tool_calls"] is False
        assert FakeUpstreamHandler.received_payloads[0]["max_tokens"] == proxy.DEFAULT_TOOL_MAX_TOKENS
    finally:
        proxy_server.shutdown()
        upstream.shutdown()
        proxy_server.server_close()
        upstream.server_close()


def test_proxy_finishes_after_first_complete_tool_call(proxy) -> None:
    FakeUpstreamHandler.mode = "tool_runaway"
    FakeUpstreamHandler.post_count = 0
    FakeUpstreamHandler.received_payloads = []
    upstream = start_server(FakeUpstreamHandler)
    proxy_server = start_server(proxy.ProxyHandler)
    proxy_server.upstream_base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    try:
        env = {
            "OPENCLAW_MODEL_FINISH_AFTER_FIRST_COMPLETE_TOOL_CALL": "1",
            "OPENCLAW_MODEL_MAX_STREAM_TOOL_CALLS": "8",
            "OPENCLAW_VMLX_STREAM_TOTAL_TIMEOUT_SECONDS": "5",
        }
        with patch.dict(os.environ, env, clear=False):
            body = post_json(
                f"http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions",
                {
                    "model": "fake",
                    "stream": True,
                    "messages": [{"role": "user", "content": "read one file"}],
                    "tools": [{"type": "function", "function": {"name": "exec", "parameters": {"type": "object"}}}],
                },
            )
        assert '"finish_reason":"tool_calls"' in body
        assert body.count('"delta":{"tool_calls"') == 1
    finally:
        proxy_server.shutdown()
        upstream.shutdown()
        proxy_server.server_close()
        upstream.server_close()


def test_proxy_blocks_broad_tool_before_forwarding(proxy) -> None:
    FakeUpstreamHandler.mode = "broad_tool"
    FakeUpstreamHandler.post_count = 0
    FakeUpstreamHandler.received_payloads = []
    upstream = start_server(FakeUpstreamHandler)
    proxy_server = start_server(proxy.ProxyHandler)
    proxy_server.upstream_base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    try:
        body = post_json(
            f"http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions",
            {
                "model": "fake",
                "stream": True,
                "messages": [{"role": "user", "content": "inspect openclaw"}],
                "tools": [{"type": "function", "function": {"name": "exec", "parameters": {"type": "object"}}}],
            },
        )
        assert "OpenClaw blocked a broad local tool command" in body
        assert '"delta":{"tool_calls"' not in body
        assert '"finish_reason":"stop"' in body
    finally:
        proxy_server.shutdown()
        upstream.shutdown()
        proxy_server.server_close()
        upstream.server_close()


def test_proxy_preflight_blocks_large_prompt_before_upstream(proxy) -> None:
    FakeUpstreamHandler.mode = "malformed"
    FakeUpstreamHandler.post_count = 0
    FakeUpstreamHandler.received_payloads = []
    upstream = start_server(FakeUpstreamHandler)
    proxy_server = start_server(proxy.ProxyHandler)
    proxy_server.upstream_base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    try:
        env = {"OPENCLAW_MODEL_MAX_SAFE_PROMPT_TOKENS": "16"}
        with patch.dict(os.environ, env, clear=False):
            body = post_json(
                f"http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions",
                {
                    "model": "fake",
                    "stream": True,
                    "messages": [{"role": "user", "content": "large prompt " * 80}],
                },
            )
        assert "OpenClaw blocked this request before model execution" in body
        assert '"finish_reason":"stop"' in body
        assert FakeUpstreamHandler.post_count == 0
    finally:
        proxy_server.shutdown()
        upstream.shutdown()
        proxy_server.server_close()
        upstream.server_close()


def test_proxy_memory_pressure_blocks_before_upstream(proxy) -> None:
    FakeUpstreamHandler.mode = "malformed"
    FakeUpstreamHandler.post_count = 0
    FakeUpstreamHandler.received_payloads = []
    upstream = start_server(FakeUpstreamHandler)
    proxy_server = start_server(proxy.ProxyHandler)
    proxy_server.upstream_base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    try:
        with patch.object(proxy, "macos_memory_snapshot", return_value={"free_mb": 64, "compressor_mb": 9000, "swap_used_mb": 0}):
            body = post_json(
                f"http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions",
                {
                    "model": "fake",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert "OpenClaw paused this model request because macOS memory pressure is too high" in body
        assert '"finish_reason":"stop"' in body
        assert FakeUpstreamHandler.post_count == 0
    finally:
        proxy_server.shutdown()
        upstream.shutdown()
        proxy_server.server_close()
        upstream.server_close()


def main() -> int:
    proxy = load_proxy()

    cleaned, reason = proxy.sanitize_visible_content("thought\n" * 32 + "call:exec{command: ls -la")
    assert cleaned == ""
    assert reason == "reasoning_only"

    cleaned, reason = proxy.sanitize_visible_content("<|channel>thought\nhidden\n<channel|>Visible<turn|>")
    assert cleaned == "Visible"
    assert reason is None

    cleaned, reason = proxy.sanitize_visible_content("thought " * 40)
    assert cleaned == ""
    assert reason == "repeated_output"

    cleaned, reason = proxy.sanitize_visible_content("step by step " * 24)
    assert cleaned == ""
    assert reason == "repeated_output"

    shaped = proxy.shape_upstream_payload({"stream": True, "reasoning_effort": "low", "messages": []})
    assert shaped["stream"] is False
    assert shaped["enable_thinking"] is False
    assert "reasoning_effort" not in shaped
    streamed = proxy.shape_upstream_payload({"stream": False, "messages": []}, stream=True)
    assert streamed["stream"] is True
    assert streamed["enable_thinking"] is False
    large = {"messages": [{"role": "user", "content": "x" * 20000}]}
    assert proxy.prompt_preflight_block_reason(large) == "estimated_prompt_tokens=6258>4500"
    assert proxy.memory_pressure_block_reason({"free_mb": 64, "compressor_mb": 0, "swap_used_mb": 0}) is None
    assert proxy.memory_pressure_block_reason({"free_mb": 460, "compressor_mb": 1024, "swap_used_mb": 2048}) is None
    assert proxy.memory_pressure_block_reason({"free_mb": 3003, "compressor_mb": 1024, "swap_used_mb": 2048}) is None
    assert proxy.memory_pressure_block_reason({"free_mb": 8192, "compressor_mb": 9000, "swap_used_mb": 0}) == "compressor_mb=9000>=8192"
    recovered_snapshot = {"free_mb": 24000, "compressor_mb": 21000, "swap_used_mb": 10500, "pressure_free_pct": 62}
    assert proxy.memory_pressure_block_reason(recovered_snapshot) is None
    tool_shaped = proxy.shape_upstream_payload(
        {
            "stream": True,
            "parallel_tool_calls": True,
            "messages": [{"role": "system", "content": "base"}],
            "tools": [{"type": "function", "function": {"name": "exec", "parameters": {"type": "object"}}}],
        },
        stream=True,
    )
    assert tool_shaped["parallel_tool_calls"] is False
    assert tool_shaped["max_tokens"] == proxy.DEFAULT_TOOL_MAX_TOKENS
    assert "OpenClaw local backend tool discipline" in tool_shaped["messages"][0]["content"]
    assert "ls -R" in tool_shaped["messages"][0]["content"]
    capped = proxy.shape_upstream_payload({"stream": True, "messages": [{"role": "user", "content": "hello"}]}, stream=True)
    assert capped["max_tokens"] == proxy.DEFAULT_MODEL_MAX_TOKENS
    pruned = proxy.prune_runaway_tool_history(
        [{"role": "system", "content": "base"}]
        + [{"role": "tool", "content": f"tool {index}"} for index in range(12)]
        + [{"role": "user", "content": "continue"}]
    )
    assert len([message for message in pruned if message.get("role") == "tool"]) == proxy.DEFAULT_MAX_TOOL_HISTORY_MESSAGES
    assert "pruned 8 earlier tool-result messages" in pruned[0]["content"]
    with patch.dict(os.environ, {"OPENCLAW_MODEL_MAX_ASSISTANT_HISTORY_MESSAGES": "3"}, clear=False):
        assistant_pruned = proxy.shape_upstream_payload(
            {
                "messages": [{"role": "system", "content": "base"}]
                + [{"role": "assistant", "content": f"older assistant {index}"} for index in range(9)]
                + [{"role": "user", "content": "continue"}],
            },
            stream=True,
        )
    assert len([message for message in assistant_pruned["messages"] if message.get("role") == "assistant"]) == 3
    assert "pruned 6 older assistant messages" in assistant_pruned["messages"][0]["content"]
    compacted = proxy.shape_upstream_payload(
        {
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "exec", "arguments": "x" * 2000}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "y" * 5000},
            ],
            "tools": [{"type": "function", "function": {"name": "exec", "parameters": {"type": "object"}}}],
        },
        stream=True,
    )
    compacted_tool = next(message for message in compacted["messages"] if message.get("role") == "tool")
    compacted_assistant = next(message for message in compacted["messages"] if message.get("role") == "assistant")
    assert len(compacted_tool["content"]) < 1800
    assert "OpenClaw compacted tool result" in compacted_tool["content"]
    compacted_args = compacted_assistant["tool_calls"][0]["function"]["arguments"]
    assert len(compacted_args) < 1200
    assert "openclaw_compacted" in compacted_args
    with patch.dict(os.environ, {"OPENCLAW_MODEL_EXACT_REQUESTS_NONSTREAMING": "1"}):
        assert proxy.prefer_nonstreaming({"messages": [{"role": "user", "content": "Reply with exactly: OK"}]})
    assert not proxy.prefer_nonstreaming({"messages": [{"role": "user", "content": "Reply with exactly: OK"}]})
    assert not proxy.prefer_nonstreaming({"messages": [{"role": "user", "content": "Write a paragraph"}], "max_tokens": 180})

    guard = proxy.StreamGuard(hold_chars=8)
    text = guard.feed("Concise engineering notes") + guard.feed(" accelerate development")
    tail, suppressed = guard.finish()
    assert text + tail == "Concise engineering notes accelerate development"
    assert suppressed is False

    guard = proxy.StreamGuard(hold_chars=8)
    assert guard.feed("thought " * 40) == ""
    tail, suppressed = guard.finish()
    assert tail == ""
    assert suppressed is True

    tool_calls = proxy.normalize_tool_calls(
        [{"id": "abc", "type": "function", "function": {"name": "exec", "arguments": {"command": "pwd"}}}]
    )
    assert tool_calls == [
        {
            "index": 0,
            "id": "abc",
            "type": "function",
            "function": {"name": "exec", "arguments": '{"command":"pwd"}'},
        }
    ]

    tool_calls = proxy.normalize_tool_calls(
        [{"id": "bad", "type": "function", "function": {"name": "exec", "arguments": "command: pwd"}}]
    )
    assert tool_calls[0]["function"]["arguments"] == '{"raw":"command: pwd"}'

    with patch.dict(os.environ, {"OPENCLAW_MODEL_DISABLE_MEMORY_PRESSURE_GUARD": "1"}, clear=False):
        test_proxy_stream_timeout(proxy)
        test_proxy_stream_recovery(proxy)
        test_proxy_tool_runaway_guard(proxy)
        test_proxy_finishes_after_first_complete_tool_call(proxy)
        test_proxy_blocks_broad_tool_before_forwarding(proxy)
        test_proxy_preflight_blocks_large_prompt_before_upstream(proxy)
    test_proxy_memory_pressure_blocks_before_upstream(proxy)

    print("ok openclaw local model proxy guards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
