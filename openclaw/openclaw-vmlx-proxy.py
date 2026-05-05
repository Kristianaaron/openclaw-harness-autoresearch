#!/usr/bin/env python3
"""OpenAI-compatible vMLX proxy for OpenClaw.

vMLX is kept as the model backend, but this proxy shields OpenClaw from
backend-specific streaming quirks. For streaming chat requests, it asks vMLX for
one non-streaming completion and emits a valid OpenAI SSE stream back to
OpenClaw. Non-streaming requests and model listing are forwarded unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


UPSTREAM_PROCESS: subprocess.Popen[bytes] | None = None
GEMMA_THOUGHT_START = "<|channel>thought\n"
GEMMA_THOUGHT_END = "<channel|>"
GEMMA_TURN_END = "<turn|>"
MALFORMED_TOOL_PATTERNS = (
    re.compile(r"(?im)^\s*call:[a-zA-Z_][\w.-]*\s*\{"),
    re.compile(r"(?is)<tool_call\b"),
    re.compile(r"(?is)<\|tool_call\>"),
    re.compile(r"(?is)<tool_call\|>"),
    re.compile(r"(?is)<function=[^>]+>"),
    re.compile(r"(?im)^\s*to=[a-zA-Z_][\w.-]*\s+code\s*\{"),
)
RECOVERY_INSTRUCTION = (
    "OpenClaw local backend guard: do not write reasoning channel markers, repeated "
    "'thought' lines, or raw tool-call text in assistant-visible content. If a tool is "
    "needed, use the provided structured tool-call channel only. If no tool is needed, "
    "answer directly in plain text."
)
DEFAULT_STREAM_IDLE_SECONDS = 5.0
DEFAULT_FIRST_EVENT_TIMEOUT_SECONDS = 180.0
DEFAULT_FIRST_VISIBLE_TIMEOUT_SECONDS = 240.0
DEFAULT_STREAM_TOTAL_TIMEOUT_SECONDS = 900.0
DEFAULT_UPSTREAM_READ_TIMEOUT_SECONDS = 180.0


def log(message: str) -> None:
    print(f"[openclaw-vmlx-proxy] {message}", file=sys.stderr, flush=True)


def http_request(method: str, url: str, body: bytes | None = None, timeout: float = 900.0) -> tuple[int, dict[str, str], bytes]:
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()


def upstream_ready(base_url: str, timeout: float = 2.0) -> bool:
    try:
        status, _, _ = http_request("GET", f"{base_url.rstrip('/')}/models", timeout=timeout)
        return 200 <= status < 300
    except Exception:
        return False


def start_upstream_if_needed(base_url: str, command_json: str | None, wait_seconds: int) -> None:
    global UPSTREAM_PROCESS
    if upstream_ready(base_url):
        log(f"upstream already ready at {base_url}")
        return
    if not command_json:
        raise RuntimeError(f"upstream is not ready at {base_url} and no upstream command was provided")
    argv = json.loads(command_json)
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise RuntimeError("OPENCLAW_VMLX_UPSTREAM_ARGV_JSON must be a JSON array of strings")
    log("starting vMLX upstream")
    UPSTREAM_PROCESS = subprocess.Popen(argv, stdout=sys.stdout.buffer, stderr=sys.stderr.buffer)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if upstream_ready(base_url):
            log(f"upstream ready at {base_url}")
            return
        if UPSTREAM_PROCESS.poll() is not None:
            raise RuntimeError(f"upstream exited early with code {UPSTREAM_PROCESS.returncode}")
        time.sleep(1)
    raise RuntimeError(f"upstream did not become ready within {wait_seconds}s")


def terminate_upstream() -> None:
    global UPSTREAM_PROCESS
    process = UPSTREAM_PROCESS
    if process is None or process.poll() is not None:
        return
    log("stopping vMLX upstream child")
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        log(f"ignoring invalid {name}={raw!r}")
        return default
    return value if value > 0 else default


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "on", "yes", "1"}:
            return True
        if lowered in {"false", "off", "no", "0", "none"}:
            return False
    return None


def extract_message(completion: dict[str, Any]) -> tuple[str, str | None, list[dict[str, Any]], str]:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None, [], "stop"
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    tool_calls = message.get("tool_calls")
    finish_reason = first.get("finish_reason") or "stop"
    return (
        content if isinstance(content, str) else "",
        reasoning if isinstance(reasoning, str) else None,
        tool_calls if isinstance(tool_calls, list) else [],
        str(finish_reason),
    )


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = normalize_tool_arguments(function.get("arguments", "{}"))
        normalized.append(
            {
                "index": index,
                "id": str(raw_call.get("id") or f"call_openclaw_{index}"),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return normalized


def normalize_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, dict):
        return json.dumps(arguments, separators=(",", ":"))
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return json.dumps(parsed, separators=(",", ":"))
        except json.JSONDecodeError:
            pass
        # OpenAI-compatible tool calls require a JSON object string. Preserve
        # the model's raw text in a structured field so OpenClaw never receives
        # invalid JSON arguments.
        return json.dumps({"raw": stripped}, separators=(",", ":"))
    return json.dumps({"raw": str(arguments)}, separators=(",", ":"))


def strip_reasoning_markers(text: str) -> tuple[str, bool]:
    changed = False
    cleaned = text.replace(GEMMA_TURN_END, "")
    if cleaned != text:
        changed = True

    while GEMMA_THOUGHT_START in cleaned and GEMMA_THOUGHT_END in cleaned:
        start = cleaned.find(GEMMA_THOUGHT_START)
        end = cleaned.find(GEMMA_THOUGHT_END, start + len(GEMMA_THOUGHT_START))
        if end < 0:
            break
        cleaned = cleaned[:start] + cleaned[end + len(GEMMA_THOUGHT_END):]
        changed = True

    degraded = cleaned.lstrip()
    if degraded.startswith("thought\n"):
        lead = len(cleaned) - len(degraded)
        after = degraded[len("thought\n"):]
        if GEMMA_THOUGHT_END in after:
            cleaned = cleaned[:lead] + after.split(GEMMA_THOUGHT_END, 1)[1]
        else:
            cleaned = ""
        changed = True

    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned)
    thought_line_pattern = re.compile(r"(?im)^(?:\s*thought\s*\n){3,}")
    cleaned, count = thought_line_pattern.subn("", cleaned)
    changed = changed or count > 0
    return cleaned.strip(), changed


def repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 12:
        return 0.0
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    return max(counts.values()) / len(lines)


def has_repeated_token_loop(text: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]+", "", text)
    if len(compact) >= 12:
        lowered = compact.lower()
        for marker in ("thought", "think"):
            repeats, remainder = divmod(len(lowered), len(marker))
            if repeats >= 2 and remainder == 0 and marker * repeats == lowered:
                return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())
    if len(words) < 24:
        return False
    run_word = ""
    run_len = 0
    for word in words:
        if word == run_word:
            run_len += 1
        else:
            run_word = word
            run_len = 1
        if run_len >= 12:
            return True
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    if max(counts.values()) / len(words) >= 0.7:
        return True
    tail = words[-48:]
    for size in range(2, 7):
        if len(tail) < size * 4:
            continue
        grams = [tuple(tail[index : index + size]) for index in range(0, len(tail) - size + 1, size)]
        if len(grams) >= 4 and len(set(grams[-4:])) == 1:
            return True
    return False


def has_malformed_tool_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in MALFORMED_TOOL_PATTERNS)


def sanitize_visible_content(content: str) -> tuple[str, str | None]:
    cleaned, stripped_reasoning = strip_reasoning_markers(content)
    if not cleaned and stripped_reasoning:
        return "", "reasoning_only"
    if cleaned and repeated_line_ratio(cleaned) >= 0.7:
        return "", "repeated_output"
    if cleaned and has_repeated_token_loop(cleaned):
        return "", "repeated_output"
    if cleaned and has_malformed_tool_text(cleaned):
        return "", "malformed_tool_text"
    return cleaned, None


def shape_upstream_payload(payload: dict[str, Any], *, stream: bool = False, recovery: bool = False) -> dict[str, Any]:
    upstream_payload = dict(payload)
    upstream_payload["stream"] = stream

    allow_thinking = bool_env("OPENCLAW_VMLX_ALLOW_THINKING", False)
    requested_thinking = as_bool(upstream_payload.get("enable_thinking"))
    if not allow_thinking and requested_thinking is not True:
        upstream_payload["enable_thinking"] = False
        upstream_payload.pop("reasoning_effort", None)

    if recovery:
        upstream_payload["enable_thinking"] = False
        upstream_payload.pop("reasoning_effort", None)
        upstream_payload["temperature"] = min(float(upstream_payload.get("temperature") or 0.3), 0.2)
        upstream_payload["max_tokens"] = min(int(upstream_payload.get("max_tokens") or 1024), 1024)
        messages = upstream_payload.get("messages")
        if isinstance(messages, list):
            guard = {"role": "system", "content": RECOVERY_INSTRUCTION}
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                first = dict(messages[0])
                first["content"] = f"{first.get('content') or ''}\n\n{RECOVERY_INSTRUCTION}"
                upstream_payload["messages"] = [first, *messages[1:]]
            else:
                upstream_payload["messages"] = [guard, *messages]

    return upstream_payload


def sanitize_completion_response(response_body: bytes) -> bytes:
    try:
        completion = json.loads(response_body.decode("utf-8"))
        if not isinstance(completion, dict):
            return response_body
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            return response_body
        first = choices[0] if isinstance(choices[0], dict) else None
        if not isinstance(first, dict):
            return response_body
        message = first.get("message")
        if not isinstance(message, dict):
            return response_body
        content = message.get("content")
        if isinstance(content, str):
            cleaned, reason = sanitize_visible_content(content)
            message["content"] = cleaned
            if reason and not message.get("tool_calls"):
                message["content"] = "[OpenClaw model emitted malformed hidden/tool output and the proxy suppressed it. Please retry the request.]"
        if not bool_env("OPENCLAW_VMLX_FORWARD_REASONING", False):
            message.pop("reasoning_content", None)
        return json.dumps(completion, separators=(",", ":")).encode("utf-8")
    except Exception as error:
        log(f"non-stream sanitization skipped: {error}")
        return response_body


def last_user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
    return ""


def prefer_nonstreaming(payload: dict[str, Any]) -> bool:
    """Avoid vMLX's empty streaming edge case for tiny exact-answer requests."""
    max_tokens = payload.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens <= 32:
        return True
    text = last_user_text(payload).lower()
    exact_patterns = (
        r"\breply with exactly\b",
        r"\brespond with exactly\b",
        r"\banswer with exactly\b",
        r"\boutput exactly\b",
        r"\bsay exactly\b",
    )
    return any(re.search(pattern, text) for pattern in exact_patterns)


def sse_payload(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n\n"


def payload_text_chars(payload: dict[str, Any]) -> int:
    total = 0
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        total += len(item["text"])
    return total


def payload_tools_chars(payload: dict[str, Any]) -> int:
    tools = payload.get("tools")
    if not tools:
        return 0
    try:
        return len(json.dumps(tools, separators=(",", ":")))
    except Exception:
        return 0


def payload_summary(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    message_count = len(messages) if isinstance(messages, list) else 0
    roles: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                roles.append(str(message.get("role") or "?"))
    tools = payload.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0
    return (
        f"messages={message_count} roles={roles} text_chars={payload_text_chars(payload)} "
        f"tools={tool_count} tool_schema_chars={payload_tools_chars(payload)} "
        f"max_tokens={payload.get('max_tokens')} stream={payload.get('stream')}"
    )


class StreamGuard:
    """Holds a small content tail so hidden channel markers cannot leak live."""

    def __init__(self, hold_chars: int = 48) -> None:
        self.hold_chars = hold_chars
        self.buffer = ""
        self.suppressed = False

    def feed(self, text: str) -> str:
        if self.suppressed or not text:
            return ""
        self.buffer += text
        cleaned, stripped_reasoning = strip_reasoning_markers(self.buffer)
        if stripped_reasoning:
            if not cleaned:
                self.buffer = ""
                self.suppressed = True
                return ""
            self.buffer = cleaned
        if repeated_line_ratio(self.buffer) >= 0.7 or has_malformed_tool_text(self.buffer):
            self.buffer = ""
            self.suppressed = True
            return ""
        if has_repeated_token_loop(self.buffer):
            self.buffer = ""
            self.suppressed = True
            return ""
        if len(self.buffer) <= self.hold_chars:
            return ""
        emit = self.buffer[:-self.hold_chars]
        self.buffer = self.buffer[-self.hold_chars:]
        return emit

    def finish(self) -> tuple[str, bool]:
        if self.suppressed:
            return "", True
        cleaned, stripped_reasoning = strip_reasoning_markers(self.buffer)
        if stripped_reasoning:
            self.buffer = ""
            return cleaned, not bool(cleaned)
        if repeated_line_ratio(self.buffer) >= 0.7 or has_malformed_tool_text(self.buffer):
            self.buffer = ""
            return "", True
        if has_repeated_token_loop(self.buffer):
            self.buffer = ""
            return "", True
        cleaned = self.buffer
        self.buffer = ""
        return cleaned, False


class ClientDisconnected(Exception):
    pass


class UpstreamProgressTimeout(Exception):
    pass


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "OpenClawVMLXProxy/1.0"

    @property
    def upstream_base_url(self) -> str:
        return self.server.upstream_base_url  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        log(fmt % args)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def forward(self, method: str) -> None:
        body = None
        if method in {"POST", "PUT", "PATCH"}:
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length else b"{}"
        target = f"{self.upstream_base_url.rstrip('/')}{self.path.removeprefix('/v1')}"
        status, headers, response_body = http_request(method, target, body=body)
        self.send_response(status)
        self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def do_GET(self) -> None:
        if self.path == "/health":
            status = 200 if upstream_ready(self.upstream_base_url) else 503
            payload = json.dumps({"ok": status == 200}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.forward("GET")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.forward("POST")
            return
        try:
            payload = self.read_json_body()
        except Exception as error:
            self.send_json_error(400, f"invalid JSON body: {error}")
            return
        request_start = time.monotonic()
        log(f"request start: {payload_summary(payload)}")
        if not payload.get("stream"):
            body = json.dumps(shape_upstream_payload(payload)).encode("utf-8")
            status, headers, response_body = http_request(
                "POST",
                f"{self.upstream_base_url.rstrip('/')}/chat/completions",
                body=body,
                timeout=float_env("OPENCLAW_VMLX_NONSTREAM_TIMEOUT_SECONDS", DEFAULT_STREAM_TOTAL_TIMEOUT_SECONDS),
            )
            if 200 <= status < 300:
                response_body = sanitize_completion_response(response_body)
            log(f"request done: mode=nonstream status={status} elapsed={time.monotonic() - request_start:.2f}s bytes={len(response_body)}")
            self.send_response(status)
            self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            return
        self.stream_completion(payload, request_start)

    def send_json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_sse(self, payload: dict[str, Any]) -> None:
        try:
            self.wfile.write(sse_payload(payload))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as error:
            self.close_connection = True
            raise ClientDisconnected(str(error)) from error

    def write_sse_comment(self, message: str) -> None:
        try:
            self.wfile.write(f": {message}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as error:
            self.close_connection = True
            raise ClientDisconnected(str(error)) from error

    def stream_completion(self, payload: dict[str, Any], request_start: float | None = None) -> None:
        request_id = f"chatcmpl-openclaw-{int(time.time() * 1000)}"
        model = str(payload.get("model") or "local-model")
        created = int(time.time())
        request_start = request_start if request_start is not None else time.monotonic()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.write_sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        try:
            if prefer_nonstreaming(payload):
                self.emit_nonstream_completion(payload, request_id, model, created)
            else:
                streamed = self.stream_upstream_completion(payload, request_id, model, created)
                if not streamed:
                    log("upstream stream returned no visible content; retrying once non-streaming")
                    self.emit_nonstream_completion(payload, request_id, model, created)
        except ClientDisconnected:
            log(f"client disconnected during stream after {time.monotonic() - request_start:.2f}s")
            return
        except UpstreamProgressTimeout as error:
            log(f"stream progress timeout: {error}")
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": f"[OpenClaw model proxy timeout: {error}]"}, "finish_reason": None}],
                }
            )
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
        except Exception as error:
            log(f"stream conversion failed: {error}")
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": f"[OpenClaw model proxy error: {error}]"}, "finish_reason": None}],
                }
            )
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True
        log(f"request done: mode=stream elapsed={time.monotonic() - request_start:.2f}s")

    def stream_upstream_completion(self, payload: dict[str, Any], request_id: str, model: str, created: int) -> bool:
        upstream_payload = shape_upstream_payload(payload, stream=True)
        body = json.dumps(upstream_payload).encode("utf-8")
        events: queue.Queue[str | BaseException | None] = queue.Queue()
        upstream_start = time.monotonic()
        first_event_at: float | None = None
        first_visible_at: float | None = None
        last_event_at = upstream_start
        idle_seconds = float_env("OPENCLAW_VMLX_STREAM_IDLE_SECONDS", DEFAULT_STREAM_IDLE_SECONDS)
        first_event_timeout = float_env(
            "OPENCLAW_VMLX_FIRST_EVENT_TIMEOUT_SECONDS",
            DEFAULT_FIRST_EVENT_TIMEOUT_SECONDS,
        )
        first_visible_timeout = float_env(
            "OPENCLAW_VMLX_FIRST_VISIBLE_TIMEOUT_SECONDS",
            DEFAULT_FIRST_VISIBLE_TIMEOUT_SECONDS,
        )
        total_timeout = float_env("OPENCLAW_VMLX_STREAM_TOTAL_TIMEOUT_SECONDS", DEFAULT_STREAM_TOTAL_TIMEOUT_SECONDS)
        read_timeout = float_env("OPENCLAW_VMLX_UPSTREAM_READ_TIMEOUT_SECONDS", DEFAULT_UPSTREAM_READ_TIMEOUT_SECONDS)

        def read_upstream() -> None:
            request = urllib.request.Request(
                f"{self.upstream_base_url.rstrip('/')}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=read_timeout) as response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if line:
                            events.put(line)
                events.put(None)
            except BaseException as error:
                events.put(error)

        worker = threading.Thread(target=read_upstream, daemon=True)
        worker.start()
        guard = StreamGuard()
        emitted = False
        finish_reason = "stop"
        tool_finish = False
        upstream_done = False

        while not upstream_done:
            try:
                item = events.get(timeout=idle_seconds)
            except queue.Empty:
                elapsed = time.monotonic() - upstream_start
                no_event_elapsed = time.monotonic() - last_event_at
                if elapsed >= total_timeout:
                    raise UpstreamProgressTimeout(f"total stream deadline exceeded after {elapsed:.0f}s")
                if first_event_at is None and elapsed >= first_event_timeout:
                    raise UpstreamProgressTimeout(f"no upstream event within {first_event_timeout:.1f}s")
                if first_visible_at is None and elapsed >= first_visible_timeout:
                    raise UpstreamProgressTimeout(f"no visible model output within {first_visible_timeout:.1f}s")
                self.write_sse_comment("openclaw-vmlx-proxy: upstream generation still running")
                log(f"stream heartbeat: elapsed={elapsed:.1f}s idle={no_event_elapsed:.1f}s")
                continue
            if item is None:
                upstream_done = True
                break
            if isinstance(item, BaseException):
                raise item
            if item == "data: [DONE]":
                upstream_done = True
                break
            if item.startswith(":"):
                last_event_at = time.monotonic()
                elapsed = last_event_at - upstream_start
                if first_visible_at is None and elapsed >= first_visible_timeout:
                    raise UpstreamProgressTimeout(f"no visible model output within {first_visible_timeout:.1f}s")
                self.write_sse_comment(item[1:].strip() or "openclaw-vmlx-proxy: upstream generation still running")
                continue
            if not item.startswith("data:"):
                continue
            if first_event_at is None:
                first_event_at = time.monotonic()
            last_event_at = time.monotonic()
            try:
                event = json.loads(item.removeprefix("data:").strip())
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                request_id = str(event.get("id") or request_id)
                created = int(event.get("created") or created)
            choices = event.get("choices") if isinstance(event, dict) else None
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                finish = choice.get("finish_reason")
                if finish:
                    finish_reason = str(finish)
                delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                if not isinstance(delta, dict):
                    continue
                if delta.get("reasoning_content") and bool_env("OPENCLAW_VMLX_FORWARD_REASONING", False):
                    self.write_sse(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"reasoning_content": delta["reasoning_content"]}, "finish_reason": None}],
                        }
                    )
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    tool_finish = True
                    self.write_sse(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}],
                        }
                    )
                    emitted = True
                content = delta.get("content")
                if isinstance(content, str) and content:
                    chunk = guard.feed(content)
                    if chunk:
                        if first_visible_at is None:
                            first_visible_at = time.monotonic()
                            log(f"first visible stream chunk after {first_visible_at - upstream_start:.2f}s")
                        self.write_sse(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                            }
                        )
                        emitted = True

        tail, suppressed = guard.finish()
        if tail:
            if first_visible_at is None:
                first_visible_at = time.monotonic()
                log(f"first visible stream chunk after {first_visible_at - upstream_start:.2f}s")
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
                }
            )
            emitted = True
        if suppressed and not emitted:
            log("stream guard suppressed malformed hidden/tool output; retrying non-stream recovery")
            return False
        event_latency = first_event_at - upstream_start if first_event_at is not None else None
        visible_latency = first_visible_at - upstream_start if first_visible_at is not None else None
        elapsed = time.monotonic() - upstream_start
        first_event_label = f"{event_latency:.2f}s" if event_latency is not None else "none"
        log(f"upstream stream done: elapsed={elapsed:.2f}s first_event={first_event_label}")
        if visible_latency is not None:
            log(f"upstream stream first_visible={visible_latency:.2f}s emitted={emitted} finish={finish_reason}")
        else:
            log(f"upstream stream first_visible=none emitted={emitted} finish={finish_reason}")
        self.write_sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_finish else finish_reason}],
            }
        )
        return emitted

    def emit_nonstream_completion(self, payload: dict[str, Any], request_id: str, model: str, created: int) -> bool:
        upstream_start = time.monotonic()
        body = json.dumps(shape_upstream_payload(payload)).encode("utf-8")
        status, _, response_body = http_request(
            "POST",
            f"{self.upstream_base_url.rstrip('/')}/chat/completions",
            body=body,
            timeout=float_env("OPENCLAW_VMLX_NONSTREAM_TIMEOUT_SECONDS", DEFAULT_STREAM_TOTAL_TIMEOUT_SECONDS),
        )
        if not 200 <= status < 300:
            raise RuntimeError(response_body.decode("utf-8", errors="replace")[:800])
        log(f"upstream nonstream done: elapsed={time.monotonic() - upstream_start:.2f}s bytes={len(response_body)}")
        completion = json.loads(response_body.decode("utf-8"))
        content, reasoning, tool_calls, finish_reason = extract_message(completion)
        content, sanitize_reason = sanitize_visible_content(content)
        if sanitize_reason and not tool_calls:
            log(f"sanitized malformed completion ({sanitize_reason}); retrying once")
            retry_body = json.dumps(shape_upstream_payload(payload, recovery=True)).encode("utf-8")
            status, _, response_body = http_request(
                "POST",
                f"{self.upstream_base_url.rstrip('/')}/chat/completions",
                body=retry_body,
                timeout=float_env("OPENCLAW_VMLX_NONSTREAM_TIMEOUT_SECONDS", DEFAULT_STREAM_TOTAL_TIMEOUT_SECONDS),
            )
            if not 200 <= status < 300:
                raise RuntimeError(response_body.decode("utf-8", errors="replace")[:800])
            completion = json.loads(response_body.decode("utf-8"))
            content, reasoning, tool_calls, finish_reason = extract_message(completion)
            content, sanitize_reason = sanitize_visible_content(content)
            if sanitize_reason:
                log(f"retry still malformed ({sanitize_reason}); suppressing visible output")
                content = "[OpenClaw model emitted malformed hidden/tool output and the proxy suppressed it. Please retry the request.]"
                reasoning = None
        request_id = str(completion.get("id") or request_id) if isinstance(completion, dict) else request_id
        created = int(completion.get("created") or created) if isinstance(completion, dict) else created
        normalized_tool_calls = normalize_tool_calls(tool_calls)
        emitted = False
        if normalized_tool_calls:
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"tool_calls": normalized_tool_calls}, "finish_reason": None}],
                }
            )
            emitted = True
        if reasoning and bool_env("OPENCLAW_VMLX_FORWARD_REASONING", False):
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}],
                }
            )
        if content:
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
            )
            emitted = True
        if not emitted:
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": "[OpenClaw model returned an empty response]"}, "finish_reason": None}],
                }
            )
        self.write_sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if normalized_tool_calls else finish_reason}],
            }
        )
        return emitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenClaw vMLX compatibility proxy.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8091)
    parser.add_argument("--upstream-base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--upstream-command-json", default=os.environ.get("OPENCLAW_VMLX_UPSTREAM_ARGV_JSON"))
    parser.add_argument("--upstream-wait-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_upstream_if_needed(args.upstream_base_url, args.upstream_command_json, args.upstream_wait_seconds)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    server.upstream_base_url = args.upstream_base_url  # type: ignore[attr-defined]

    def shutdown(_signum: int, _frame: Any) -> None:
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log(f"proxy listening on http://{args.listen_host}:{args.listen_port}/v1 -> {args.upstream_base_url}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        terminate_upstream()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
