#!/usr/bin/env python3
"""OpenAI-compatible local model proxy for OpenClaw.

The backend can be Rapid-MLX, vMLX, or another OpenAI-compatible local server.
This proxy shields OpenClaw from backend-specific streaming quirks while keeping
the model runtime itself backend-owned.
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
TOOL_DISCIPLINE_INSTRUCTION = (
    "OpenClaw local backend tool discipline: when tools are available, call at most one "
    "tool per assistant turn unless the user explicitly asks for broad search. Prefer "
    "bounded paths and commands. Never emit a long sequence of speculative search, find, "
    "grep, or mdfind calls in one response. Avoid broad commands such as `ls -R`, "
    "`find ~`, recursive grep over home directories, or whole-disk mdfind. Use targeted "
    "`rg --files`, bounded `find -maxdepth`, and small previews instead."
)
DEFAULT_STREAM_IDLE_SECONDS = 5.0
DEFAULT_FIRST_EVENT_TIMEOUT_SECONDS = 180.0
DEFAULT_FIRST_VISIBLE_TIMEOUT_SECONDS = 240.0
DEFAULT_STREAM_TOTAL_TIMEOUT_SECONDS = 900.0
DEFAULT_UPSTREAM_READ_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_STREAM_TOOL_CALLS = 2
DEFAULT_MAX_STREAM_TOOL_CHUNKS = 6
DEFAULT_MAX_STREAM_TOOL_ARG_CHARS = 2048
DEFAULT_MODEL_MAX_TOKENS = 512
DEFAULT_TOOL_MAX_TOKENS = 384
DEFAULT_MAX_TOOL_HISTORY_MESSAGES = 4
DEFAULT_MAX_ASSISTANT_HISTORY_MESSAGES = 12
DEFAULT_MAX_TOOL_RESULT_CHARS = 1400
DEFAULT_MAX_ASSISTANT_TOOL_ARG_CHARS = 700
DEFAULT_PROMPT_CHARS_PER_TOKEN = 3.2
DEFAULT_MAX_SAFE_PROMPT_TOKENS = 4500
DEFAULT_MAX_SAFE_TOOL_PROMPT_TOKENS = 3500
DEFAULT_MIN_SAFE_FREE_MB = 0
DEFAULT_MAX_SAFE_COMPRESSOR_MB = 8192
DEFAULT_MAX_SAFE_SWAP_MB = 8192
PREFLIGHT_BLOCK_MESSAGE = (
    "[OpenClaw blocked this request before model execution because the prompt/tool context "
    "is large enough to risk a local MLX/Metal memory crash. Start a fresh session, reduce "
    "history, or inspect one narrower source at a time.]"
)
MEMORY_PRESSURE_BLOCK_MESSAGE = (
    "[OpenClaw paused this model request because macOS memory pressure is too high for a "
    "31B local MLX run. Let memory settle or stop other local model/runtime processes, then retry.]"
)
BROAD_TOOL_COMMAND_PATTERNS = (
    re.compile(r"(?is)\bls\s+[^\"'\n]*-R\b"),
    re.compile(r"(?is)\bls\s+-R\b"),
    re.compile(r"(?is)\bfind\s+(?:~|/|/Users(?:/kristian)?)\b(?![^\"'\n]*\s-maxdepth\s+[0-3]\b)"),
    re.compile(r"(?is)\bgrep\s+[^\"'\n]*-[^\s\"']*R"),
    re.compile(r"(?is)\brg\b[^\"'\n]*(?:\s/|\s~|\s/Users(?:/kristian)?\b)(?![^\"'\n]*\s--max-count\b)"),
    re.compile(r"(?is)\bmdfind\b"),
)
BROAD_TOOL_BLOCK_MESSAGE = (
    "[OpenClaw blocked a broad local tool command before execution. "
    "Retry with one exact file or command. For speed research, use the Bootstrap Ladder: "
    "read /Users/kristian/.openclaw/research/speed/README-openclaw-speed.md, "
    "read /Users/kristian/.openclaw/research/speed/results.tsv, or run "
    "/Users/kristian/.openclaw/bin/openclaw-speed-research benchmark --quick.]"
)


def tool_call_arguments_complete(tool_calls: Any) -> bool:
    if not isinstance(tool_calls, list) or not tool_calls:
        return False
    for raw_call in tool_calls:
        if not isinstance(raw_call, dict):
            return False
        function = raw_call.get("function")
        if not isinstance(function, dict):
            return False
        arguments = function.get("arguments")
        if not isinstance(arguments, str) or not arguments.strip():
            return False
        stripped = arguments.strip()
        try:
            json.loads(stripped)
            continue
        except json.JSONDecodeError:
            pass
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return False
        depth = 0
        in_string = False
        escape = False
        for char in stripped:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    return False
        if depth != 0 or in_string:
            return False
    return True


def log(message: str) -> None:
    print(f"[openclaw-model-proxy] {message}", file=sys.stderr, flush=True)


def tool_call_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    parts: list[str] = []
    for raw_call in tool_calls:
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(name, str):
                parts.append(name)
            if isinstance(arguments, str):
                parts.append(arguments)
            elif isinstance(arguments, dict):
                try:
                    parts.append(json.dumps(arguments, separators=(",", ":")))
                except TypeError:
                    pass
        else:
            try:
                parts.append(json.dumps(raw_call, separators=(",", ":")))
            except TypeError:
                pass
    return "\n".join(parts)


def broad_tool_command_reason(tool_calls: Any) -> str | None:
    text = tool_call_text(tool_calls)
    if not text:
        return None
    for pattern in BROAD_TOOL_COMMAND_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


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


def upstream_model_id(base_url: str) -> str:
    status, _, body = http_request("GET", f"{base_url.rstrip('/')}/models", timeout=5)
    if not 200 <= status < 300:
        raise RuntimeError(f"upstream /models returned HTTP {status}")
    models = json.loads(body.decode("utf-8"))
    data = models.get("data") if isinstance(models, dict) else None
    first = data[0] if isinstance(data, list) and data else {}
    model_id = first.get("id") if isinstance(first, dict) else None
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("upstream /models did not return a usable model id")
    return model_id


def upstream_compatibility_ready(base_url: str) -> bool:
    if bool_env("OPENCLAW_MODEL_SKIP_COMPAT_SMOKE", False):
        log("upstream compatibility smoke skipped by OPENCLAW_MODEL_SKIP_COMPAT_SMOKE")
        return True
    model = os.environ.get("OPENCLAW_MODEL_COMPAT_SMOKE_MODEL") or upstream_model_id(base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 12,
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": False,
        "stream": False,
    }
    status, _, body = http_request(
        "POST",
        f"{base_url.rstrip('/')}/chat/completions",
        body=json.dumps(payload).encode("utf-8"),
        timeout=float_env("OPENCLAW_MODEL_COMPAT_SMOKE_TIMEOUT_SECONDS", 90),
    )
    if not 200 <= status < 300:
        log(f"upstream compatibility smoke returned HTTP {status}")
        return False
    try:
        completion = json.loads(body.decode("utf-8"))
        choices = completion.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else {}
        message = first.get("message") if isinstance(first, dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        finish = first.get("finish_reason") if isinstance(first, dict) else None
    except Exception as error:
        log(f"upstream compatibility smoke returned invalid JSON: {error}")
        return False
    if not isinstance(content, str) or not content.strip():
        log(f"upstream compatibility smoke returned no content; finish={finish!r}")
        return False
    if has_repeated_token_loop(content) or repeated_line_ratio(content) >= 0.7:
        log(f"upstream compatibility smoke detected repeated-token loop: {content[:120]!r}")
        return False
    log(f"upstream compatibility smoke passed: {content[:80]!r}")
    return True


def start_upstream_if_needed(base_url: str, command_json: str | None, wait_seconds: int) -> None:
    global UPSTREAM_PROCESS
    if upstream_ready(base_url):
        if not upstream_compatibility_ready(base_url):
            raise RuntimeError("upstream is ready but failed OpenClaw compatibility smoke")
        log(f"upstream already ready at {base_url}")
        return
    if not command_json:
        raise RuntimeError(f"upstream is not ready at {base_url} and no upstream command was provided")
    argv = json.loads(command_json)
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise RuntimeError("OPENCLAW_VMLX_UPSTREAM_ARGV_JSON must be a JSON array of strings")
    log("starting local model upstream")
    UPSTREAM_PROCESS = subprocess.Popen(argv, stdout=sys.stdout.buffer, stderr=sys.stderr.buffer)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if upstream_ready(base_url):
            if not upstream_compatibility_ready(base_url):
                terminate_upstream()
                raise RuntimeError("upstream failed OpenClaw compatibility smoke")
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
    log("stopping local model upstream child")
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def bool_env(name: str, default: bool = False) -> bool:
    generic_name = name.replace("OPENCLAW_VMLX_", "OPENCLAW_MODEL_", 1)
    raw = os.environ.get(generic_name) if generic_name != name else None
    if raw is None:
        raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def float_env(name: str, default: float) -> float:
    generic_name = name.replace("OPENCLAW_VMLX_", "OPENCLAW_MODEL_", 1)
    raw = os.environ.get(generic_name) if generic_name != name else None
    if raw is None:
        raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        log(f"ignoring invalid {name}={raw!r}")
        return default
    return value if value > 0 else default


def int_env(name: str, default: int) -> int:
    generic_name = name.replace("OPENCLAW_VMLX_", "OPENCLAW_MODEL_", 1)
    raw = os.environ.get(generic_name) if generic_name != name else None
    if raw is None:
        raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        log(f"ignoring invalid {name}={raw!r}")
        return default
    return value if value > 0 else default


def macos_memory_snapshot() -> dict[str, int]:
    snapshot = {"free_mb": 0, "compressor_mb": 0, "swap_used_mb": 0}
    try:
        vm_stat = subprocess.check_output(["/usr/bin/vm_stat"], text=True, stderr=subprocess.DEVNULL, timeout=3)
        page_size = 16384
        free_pages = 0
        speculative_pages = 0
        compressor_pages = 0
        for line in vm_stat.splitlines():
            if "page size of" in line:
                match = re.search(r"page size of\s+(\d+)", line)
                if match:
                    page_size = int(match.group(1))
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":", 1)[1].strip().rstrip("."))
            elif line.startswith("Pages speculative:"):
                speculative_pages = int(line.split(":", 1)[1].strip().rstrip("."))
            elif line.startswith("Pages occupied by compressor:"):
                compressor_pages = int(line.split(":", 1)[1].strip().rstrip("."))
        snapshot["free_mb"] = int((free_pages + speculative_pages) * page_size / 1048576)
        snapshot["compressor_mb"] = int(compressor_pages * page_size / 1048576)
    except Exception as error:
        log(f"memory snapshot vm_stat unavailable: {error}")
    try:
        swap = subprocess.check_output(["/usr/sbin/sysctl", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL, timeout=3)
        match = re.search(r"used = ([0-9.]+)M", swap)
        if match:
            snapshot["swap_used_mb"] = int(float(match.group(1)))
    except Exception as error:
        log(f"memory snapshot swap unavailable: {error}")
    return snapshot


def memory_pressure_block_reason(snapshot: dict[str, int] | None = None) -> str | None:
    if bool_env("OPENCLAW_MODEL_DISABLE_MEMORY_PRESSURE_GUARD", False):
        return None
    snapshot = snapshot or macos_memory_snapshot()
    free_mb = snapshot.get("free_mb", 0)
    compressor_mb = snapshot.get("compressor_mb", 0)
    swap_used_mb = snapshot.get("swap_used_mb", 0)
    min_free_mb = int_env("OPENCLAW_MODEL_MIN_SAFE_FREE_MB", DEFAULT_MIN_SAFE_FREE_MB)
    max_compressor_mb = int_env("OPENCLAW_MODEL_MAX_SAFE_COMPRESSOR_MB", DEFAULT_MAX_SAFE_COMPRESSOR_MB)
    max_swap_mb = int_env("OPENCLAW_MODEL_MAX_SAFE_SWAP_MB", DEFAULT_MAX_SAFE_SWAP_MB)
    if compressor_mb >= max_compressor_mb:
        return f"compressor_mb={compressor_mb}>={max_compressor_mb}"
    if swap_used_mb >= max_swap_mb:
        return f"swap_used_mb={swap_used_mb}>={max_swap_mb}"
    if free_mb and free_mb < min_free_mb:
        return f"free_mb={free_mb}<{min_free_mb}"
    return None


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
        for size in range(1, min(16, len(lowered) // 4) + 1):
            unit = lowered[:size]
            if not unit.strip("0123456789"):
                continue
            repeats, remainder = divmod(len(lowered), size)
            if repeats >= 4 and remainder == 0 and unit * repeats == lowered:
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


def has_tools(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    return isinstance(tools, list) and bool(tools)


def append_system_instruction(messages: Any, instruction: str) -> Any:
    if not isinstance(messages, list):
        return messages
    copied = [dict(message) if isinstance(message, dict) else message for message in messages]
    if copied and isinstance(copied[0], dict) and copied[0].get("role") == "system":
        content = copied[0].get("content")
        if isinstance(content, str) and instruction not in content:
            copied[0]["content"] = f"{content}\n\n{instruction}"
        return copied
    return [{"role": "system", "content": instruction}, *copied]


def compact_text(value: str, max_chars: int, label: str) -> tuple[str, bool]:
    if max_chars <= 0 or len(value) <= max_chars:
        return value, False
    head_chars = max(120, int(max_chars * 0.65))
    tail_chars = max(80, max_chars - head_chars - 160)
    omitted = len(value) - head_chars - tail_chars
    return (
        f"{value[:head_chars]}\n\n"
        f"[OpenClaw compacted {label}: omitted {omitted} chars to keep the local MLX prompt safe.]\n\n"
        f"{value[-tail_chars:]}",
        True,
    )


def compact_message_content(content: Any, max_chars: int, label: str) -> tuple[Any, bool]:
    if isinstance(content, str):
        return compact_text(content, max_chars, label)
    if isinstance(content, list):
        changed = False
        next_items: list[Any] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                next_item = dict(item)
                next_text, did_change = compact_text(next_item["text"], max_chars, label)
                next_item["text"] = next_text
                changed = changed or did_change
                next_items.append(next_item)
            else:
                next_items.append(item)
        return next_items, changed
    return content, False


def compact_tool_result_message(message: dict[str, Any], max_chars: int) -> tuple[dict[str, Any], bool]:
    next_message = dict(message)
    compacted, changed = compact_message_content(next_message.get("content"), max_chars, "tool result")
    next_message["content"] = compacted
    return next_message, changed


def compact_assistant_tool_calls(message: dict[str, Any], max_arg_chars: int) -> tuple[dict[str, Any], bool]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message, False
    changed = False
    next_calls: list[Any] = []
    for raw_call in tool_calls:
        if not isinstance(raw_call, dict):
            next_calls.append(raw_call)
            continue
        next_call = dict(raw_call)
        function = next_call.get("function")
        if isinstance(function, dict) and isinstance(function.get("arguments"), str):
            next_function = dict(function)
            compacted, did_change = compact_text(next_function["arguments"], max_arg_chars, "tool arguments")
            if did_change:
                next_function["arguments"] = json.dumps({"openclaw_compacted": True, "preview": compacted}, separators=(",", ":"))
                next_call["function"] = next_function
                changed = True
        next_calls.append(next_call)
    if not changed:
        return message, False
    next_message = dict(message)
    next_message["tool_calls"] = next_calls
    return next_message, True


def prune_runaway_tool_history(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    max_tool_messages = int_env("OPENCLAW_MODEL_MAX_TOOL_HISTORY_MESSAGES", DEFAULT_MAX_TOOL_HISTORY_MESSAGES)
    tool_positions = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    if len(tool_positions) <= max_tool_messages:
        return messages

    keep_positions = set(tool_positions[-max_tool_messages:])
    pruned = 0
    next_messages: list[Any] = []
    for index, message in enumerate(messages):
        if not (isinstance(message, dict) and message.get("role") == "tool"):
            next_messages.append(message)
            continue
        if index in keep_positions:
            next_messages.append(message)
            continue
        pruned += 1

    note = {
        "role": "system",
        "content": (
            f"OpenClaw proxy pruned {pruned} earlier tool-result messages from a previous "
            "runaway tool sequence. Use the remaining recent context and avoid repeating "
            "the same broad search."
        ),
    }
    log(f"pruned runaway tool history: removed={pruned} kept={len(keep_positions)}")
    if next_messages and isinstance(next_messages[0], dict) and next_messages[0].get("role") == "system":
        first = dict(next_messages[0])
        first["content"] = f"{first.get('content') or ''}\n\n{note['content']}"
        return [first, *next_messages[1:]]
    return [note, *next_messages]


def prune_assistant_history(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    max_assistant_messages = int_env("OPENCLAW_MODEL_MAX_ASSISTANT_HISTORY_MESSAGES", DEFAULT_MAX_ASSISTANT_HISTORY_MESSAGES)
    if max_assistant_messages <= 0:
        return messages
    assistant_positions = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if len(assistant_positions) <= max_assistant_messages:
        return messages

    keep_positions = set(assistant_positions[-max_assistant_messages:])
    pruned = 0
    next_messages: list[Any] = []
    for index, message in enumerate(messages):
        if not (isinstance(message, dict) and message.get("role") == "assistant"):
            next_messages.append(message)
            continue
        if index in keep_positions:
            next_messages.append(message)
            continue
        pruned += 1

    note = {
        "role": "system",
        "content": (
            f"OpenClaw proxy pruned {pruned} older assistant messages to keep the local "
            "MLX agent prompt fast. Use the remaining recent context and continue the task."
        ),
    }
    log(f"pruned assistant history: removed={pruned} kept={len(keep_positions)}")
    if next_messages and isinstance(next_messages[0], dict) and next_messages[0].get("role") == "system":
        first = dict(next_messages[0])
        first["content"] = f"{first.get('content') or ''}\n\n{note['content']}"
        return [first, *next_messages[1:]]
    return [note, *next_messages]


def compact_tool_context(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    max_tool_result_chars = int_env("OPENCLAW_MODEL_MAX_TOOL_RESULT_CHARS", DEFAULT_MAX_TOOL_RESULT_CHARS)
    max_tool_arg_chars = int_env("OPENCLAW_MODEL_MAX_ASSISTANT_TOOL_ARG_CHARS", DEFAULT_MAX_ASSISTANT_TOOL_ARG_CHARS)
    next_messages: list[Any] = []
    compacted_results = 0
    compacted_args = 0
    for message in messages:
        if not isinstance(message, dict):
            next_messages.append(message)
            continue
        if message.get("role") == "tool":
            next_message, changed = compact_tool_result_message(message, max_tool_result_chars)
            compacted_results += int(changed)
            next_messages.append(next_message)
            continue
        if message.get("role") == "assistant":
            next_message, changed = compact_assistant_tool_calls(message, max_tool_arg_chars)
            compacted_args += int(changed)
            next_messages.append(next_message)
            continue
        next_messages.append(message)
    if compacted_results or compacted_args:
        log(f"compacted tool context: tool_results={compacted_results} assistant_tool_args={compacted_args}")
    return next_messages


def shape_upstream_payload(payload: dict[str, Any], *, stream: bool = False, recovery: bool = False) -> dict[str, Any]:
    upstream_payload = dict(payload)
    upstream_payload["stream"] = stream
    upstream_payload["messages"] = compact_tool_context(
        prune_assistant_history(prune_runaway_tool_history(upstream_payload.get("messages")))
    )

    if has_tools(upstream_payload):
        upstream_payload["messages"] = append_system_instruction(upstream_payload.get("messages"), TOOL_DISCIPLINE_INSTRUCTION)
        upstream_payload["parallel_tool_calls"] = False
        if upstream_payload.get("max_tokens") is None:
            upstream_payload["max_tokens"] = int_env("OPENCLAW_MODEL_TOOL_MAX_TOKENS", DEFAULT_TOOL_MAX_TOKENS)
    elif upstream_payload.get("max_tokens") is None:
        upstream_payload["max_tokens"] = int_env("OPENCLAW_MODEL_MAX_TOKENS", DEFAULT_MODEL_MAX_TOKENS)

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


def estimated_prompt_tokens(payload: dict[str, Any]) -> int:
    chars_per_token = float_env("OPENCLAW_MODEL_PROMPT_CHARS_PER_TOKEN", DEFAULT_PROMPT_CHARS_PER_TOKEN)
    text_chars = payload_text_chars(payload)
    tool_chars = payload_tools_chars(payload)
    messages = payload.get("messages")
    message_count = len(messages) if isinstance(messages, list) else 0
    return max(1, int(((text_chars + tool_chars) / chars_per_token) + (message_count * 8)))


def prompt_preflight_block_reason(payload: dict[str, Any]) -> str | None:
    if bool_env("OPENCLAW_MODEL_ALLOW_LARGE_PREFILL", False):
        return None
    estimate = estimated_prompt_tokens(payload)
    has_tool_schema = has_tools(payload)
    limit = int_env(
        "OPENCLAW_MODEL_MAX_SAFE_TOOL_PROMPT_TOKENS" if has_tool_schema else "OPENCLAW_MODEL_MAX_SAFE_PROMPT_TOKENS",
        DEFAULT_MAX_SAFE_TOOL_PROMPT_TOKENS if has_tool_schema else DEFAULT_MAX_SAFE_PROMPT_TOKENS,
    )
    if estimate > limit:
        return f"estimated_prompt_tokens={estimate}>{limit}"
    return None


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
        f"estimated_prompt_tokens={estimated_prompt_tokens(payload)} "
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


class UpstreamMemoryPressure(Exception):
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
        try:
            status, headers, response_body = http_request(method, target, body=body)
        except Exception as error:
            log(f"upstream forward failed: method={method} path={self.path} error={error}")
            response_body = json.dumps(
                {
                    "error": {
                        "message": (
                            "OpenClaw model upstream is unavailable. The local model process may have "
                            "crashed or stopped; run `openclaw model-stop` then `openclaw model-start`."
                        )
                    }
                },
                separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            return
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
        memory_block_reason = memory_pressure_block_reason()
        if memory_block_reason:
            log(f"request start: {payload_summary(payload)}")
            log(f"request memory preflight blocked: {memory_block_reason}")
            if payload.get("stream"):
                self.stream_preflight_block(payload, request_start, memory_block_reason, MEMORY_PRESSURE_BLOCK_MESSAGE)
            else:
                self.send_json_error(503, f"{MEMORY_PRESSURE_BLOCK_MESSAGE} ({memory_block_reason})")
            return
        payload = shape_upstream_payload(payload, stream=bool(payload.get("stream")))
        log(f"request start: {payload_summary(payload)}")
        block_reason = prompt_preflight_block_reason(payload)
        if block_reason:
            log(f"request preflight blocked: {block_reason}")
            if payload.get("stream"):
                self.stream_preflight_block(payload, request_start, block_reason, PREFLIGHT_BLOCK_MESSAGE)
            else:
                self.send_json_error(413, f"{PREFLIGHT_BLOCK_MESSAGE} ({block_reason})")
            return
        if not payload.get("stream"):
            body = json.dumps(shape_upstream_payload(payload)).encode("utf-8")
            try:
                status, headers, response_body = http_request(
                    "POST",
                    f"{self.upstream_base_url.rstrip('/')}/chat/completions",
                    body=body,
                    timeout=float_env("OPENCLAW_VMLX_NONSTREAM_TIMEOUT_SECONDS", DEFAULT_STREAM_TOTAL_TIMEOUT_SECONDS),
                )
            except Exception as error:
                log(f"nonstream upstream unavailable: {error}")
                self.send_json_error(503, f"OpenClaw model upstream unavailable: {error}")
                return
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

    def stream_preflight_block(self, payload: dict[str, Any], request_start: float, reason: str, message: str) -> None:
        request_id = f"chatcmpl-openclaw-{int(time.time() * 1000)}"
        model = str(payload.get("model") or "local-model")
        created = int(time.time())
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
        self.write_sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": f"{message} ({reason})"}, "finish_reason": None}],
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
        log(f"request done: mode=stream-preflight-block reason={reason} elapsed={time.monotonic() - request_start:.2f}s")

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
        except UpstreamMemoryPressure as error:
            log(f"stream memory pressure abort: {error}")
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": f"{MEMORY_PRESSURE_BLOCK_MESSAGE} ({error})"}, "finish_reason": None}],
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
        max_tool_calls = int_env("OPENCLAW_MODEL_MAX_STREAM_TOOL_CALLS", DEFAULT_MAX_STREAM_TOOL_CALLS)
        max_tool_chunks = int_env("OPENCLAW_MODEL_MAX_STREAM_TOOL_CHUNKS", DEFAULT_MAX_STREAM_TOOL_CHUNKS)
        max_tool_arg_chars = int_env("OPENCLAW_MODEL_MAX_STREAM_TOOL_ARG_CHARS", DEFAULT_MAX_STREAM_TOOL_ARG_CHARS)
        finish_after_first_complete_tool = bool_env("OPENCLAW_MODEL_FINISH_AFTER_FIRST_COMPLETE_TOOL_CALL", False)
        memory_check_interval = float_env("OPENCLAW_MODEL_STREAM_MEMORY_CHECK_INTERVAL_SECONDS", 5.0)
        last_memory_check_at = upstream_start
        stop_event = threading.Event()
        response_holder: dict[str, Any] = {}

        def stop_upstream_reader(*, close_response: bool = True) -> None:
            stop_event.set()
            if not close_response:
                return
            response = response_holder.get("response")
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

        def read_upstream() -> None:
            request = urllib.request.Request(
                f"{self.upstream_base_url.rstrip('/')}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=read_timeout) as response:
                    response_holder["response"] = response
                    for raw_line in response:
                        if stop_event.is_set():
                            break
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if line:
                            events.put(line)
                events.put(None)
            except BaseException as error:
                events.put(None if stop_event.is_set() else error)

        worker = threading.Thread(target=read_upstream, daemon=True)
        worker.start()
        guard = StreamGuard()
        emitted = False
        finish_reason = "stop"
        tool_finish = False
        upstream_done = False
        tool_call_chunks = 0
        tool_arg_chars = 0
        tool_call_keys: set[str] = set()
        tool_guard_reason: str | None = None

        def abort_if_memory_pressure() -> None:
            nonlocal last_memory_check_at
            now = time.monotonic()
            if now - last_memory_check_at < memory_check_interval:
                return
            last_memory_check_at = now
            reason = memory_pressure_block_reason()
            if reason:
                log(f"stream memory guard closing upstream: {reason}")
                stop_upstream_reader()
                raise UpstreamMemoryPressure(reason)

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
                if not emitted and first_visible_at is None and elapsed >= first_visible_timeout:
                    raise UpstreamProgressTimeout(f"no visible model output within {first_visible_timeout:.1f}s")
                abort_if_memory_pressure()
                self.write_sse_comment("openclaw-model-proxy: upstream generation still running")
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
                self.write_sse_comment(item[1:].strip() or "openclaw-model-proxy: upstream generation still running")
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
            abort_if_memory_pressure()
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
                    tool_call_chunks += 1
                    broad_command_reason = broad_tool_command_reason(tool_calls)
                    for fallback_index, raw_call in enumerate(tool_calls):
                        if not isinstance(raw_call, dict):
                            continue
                        key = raw_call.get("id")
                        if not isinstance(key, str) or not key:
                            raw_index = raw_call.get("index")
                            key = f"index:{raw_index}" if raw_index is not None else f"chunk:{tool_call_chunks}:{fallback_index}"
                        tool_call_keys.add(key)
                        function = raw_call.get("function")
                        if isinstance(function, dict):
                            arguments = function.get("arguments")
                            if isinstance(arguments, str):
                                tool_arg_chars += len(arguments)

                    if broad_command_reason:
                        tool_guard_reason = f"broad_tool_command={broad_command_reason}"
                    elif len(tool_call_keys) > max_tool_calls:
                        tool_guard_reason = f"unique_tool_calls={len(tool_call_keys)}>{max_tool_calls}"
                    elif tool_call_chunks > max_tool_chunks:
                        tool_guard_reason = f"tool_chunks={tool_call_chunks}>{max_tool_chunks}"
                    elif tool_arg_chars > max_tool_arg_chars:
                        tool_guard_reason = f"tool_arg_chars={tool_arg_chars}>{max_tool_arg_chars}"
                    if tool_guard_reason:
                        log(f"stream tool-call guard stopped runaway: {tool_guard_reason}")
                        stop_upstream_reader()
                        if broad_command_reason:
                            if first_visible_at is None:
                                first_visible_at = time.monotonic()
                                log(f"blocked broad tool command after {first_visible_at - upstream_start:.2f}s")
                            self.write_sse(
                                {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": BROAD_TOOL_BLOCK_MESSAGE}, "finish_reason": None}],
                                }
                            )
                        self.write_sse(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if broad_command_reason else "tool_calls"}],
                            }
                        )
                        return True
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
                    if finish_after_first_complete_tool and len(tool_call_keys) == 1 and tool_call_arguments_complete(tool_calls):
                        log("stream tool-call guard finished after first complete tool call")
                        stop_upstream_reader(close_response=False)
                        self.write_sse(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                            }
                        )
                        return True
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
    parser = argparse.ArgumentParser(description="Run the OpenClaw local model compatibility proxy.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8091)
    parser.add_argument("--upstream-base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument(
        "--upstream-command-json",
        default=os.environ.get("OPENCLAW_MODEL_UPSTREAM_ARGV_JSON")
        or os.environ.get("OPENCLAW_VMLX_UPSTREAM_ARGV_JSON"),
    )
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
