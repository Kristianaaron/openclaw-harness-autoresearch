#!/usr/bin/env python3
"""Unit checks for OpenClaw prefix warmer."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = Path(__file__).with_name("openclaw-prefix-warmer.py")
spec = importlib.util.spec_from_file_location("openclaw_prefix_warmer", SCRIPT_PATH)
warmer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(warmer)


class FakeChatHandler(BaseHTTPRequestHandler):
    request_payload: dict | None = None

    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or "0")
        FakeChatHandler.request_payload = json.loads(self.rfile.read(length).decode("utf-8"))
        body = json.dumps(
            {
                "id": "fake",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_latest_compiled_context() -> None:
    with TemporaryDirectory() as tmp:
        sessions = Path(tmp)
        first = sessions / "old.trajectory.jsonl"
        latest = sessions / "new.trajectory.jsonl"
        first.write_text(
            json.dumps({"type": "context.compiled", "data": {"systemPrompt": "old system", "prompt": "old prompt"}}) + "\n",
            encoding="utf-8",
        )
        latest.write_text(
            json.dumps({"type": "context.compiled", "data": {"systemPrompt": "new system", "prompt": "new prompt"}}) + "\n",
            encoding="utf-8",
        )
        latest.touch()
        context = warmer.latest_compiled_context(sessions)
        assert context == ("new system", "new prompt", latest.name)


def test_latest_compiled_context_skips_large_sources() -> None:
    with TemporaryDirectory() as tmp:
        sessions = Path(tmp)
        large = sessions / "large.trajectory.jsonl"
        small = sessions / "small.trajectory.jsonl"
        small.write_text(
            json.dumps({"type": "context.compiled", "data": {"systemPrompt": "small system", "prompt": "small prompt"}}) + "\n",
            encoding="utf-8",
        )
        large.write_text(
            json.dumps({"type": "context.compiled", "data": {"systemPrompt": "x" * 7000, "prompt": "large prompt"}}) + "\n",
            encoding="utf-8",
        )
        large.touch()
        context = warmer.latest_compiled_context(sessions)
        assert context == ("small system", "small prompt", small.name)


def test_skip_is_process_scoped() -> None:
    state = {"fingerprint": "abc", "listenerPids": "123", "warmedAt": time.time()}
    assert warmer.should_skip_warmup(state=state, fingerprint="abc", pids="123", max_age_seconds=999_999)
    assert not warmer.should_skip_warmup(state=state, fingerprint="abc", pids="456", max_age_seconds=999_999)
    assert not warmer.should_skip_warmup(state=state, fingerprint="def", pids="123", max_age_seconds=999_999)


def test_warm_prefix_posts_minimal_completion() -> None:
    server = start_server()
    try:
        profile = {"baseUrl": f"http://127.0.0.1:{server.server_port}/v1", "model": "local-model"}
        elapsed, status, body = warmer.warm_prefix(
            profile,
            "system prompt",
            "warm prompt",
            timeout=5,
            max_tokens=1,
        )
        assert elapsed >= 0
        assert status == 200
        assert b"OK" in body
        assert FakeChatHandler.request_payload == {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "warm prompt"},
            ],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "enable_thinking": False,
        }
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    test_latest_compiled_context()
    test_latest_compiled_context_skips_large_sources()
    test_skip_is_process_scoped()
    test_warm_prefix_posts_minimal_completion()
    print("ok openclaw prefix warmer")
