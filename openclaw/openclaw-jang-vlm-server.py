#!/usr/bin/env python3
"""OpenAI-compatible JANG/VLM server for OpenClaw.

This is the production-safe backend for Gemma4 JANG models when Rapid-MLX's
JANG scheduler path is not compatibility-clean. It uses the same known-good
JANG + mlx_vlm generation path that vMLX relies on for Gemma4 behavior, while
OpenClaw's outer proxy still owns UX guardrails such as SSE deadlines, recovery,
reasoning separation, tool-call normalization, and repeated-output suppression.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL = None
PROCESSOR = None
DRAFT_MODEL = None
DRAFT_BACKEND = ""
DFLASH_ACCEPT_LENS: list[int] = []
MODEL_PATH = ""
MODEL_ID = ""
GENERATION_LOCK = threading.Lock()
MODEL_READY = threading.Event()
MODEL_LOAD_ERROR: Exception | None = None
MODEL_TASKS: "queue.Queue[tuple[Any, queue.Queue[tuple[bool, Any]]]]" = queue.Queue()
GEMMA_THOUGHT_START = "<|channel>thought\n"
GEMMA_THOUGHT_END = "<channel|>"
GEMMA_TURN_END = "<turn|>"


def log(message: str) -> None:
    print(f"[openclaw-jang-vlm-server] {message}", file=sys.stderr, flush=True)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        log(f"ignoring invalid {name}={raw!r}")
        return default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        log(f"ignoring invalid {name}={raw!r}")
        return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def configure_mlx_memory() -> None:
    try:
        import mlx.core as mx

        if mx.metal.is_available():
            device_info = mx.device_info()
            recommended = device_info.get(
                "max_recommended_working_set_size",
                device_info.get("memory_size", 0),
            )
            utilization = env_float("OPENCLAW_JANG_GPU_MEMORY_UTILIZATION", 0.82)
            if recommended:
                mx.set_memory_limit(int(recommended * utilization))
            cache_gb = env_float("OPENCLAW_JANG_MLX_CACHE_GB", 16)
            mx.set_cache_limit(int(cache_gb * 1024**3))
            log(
                "Metal memory configured "
                f"allocation={utilization:.2f} cache={cache_gb:.1f}GB"
            )
    except Exception as error:
        log(f"MLX memory configuration skipped: {error}")


def load_model(model_path: str) -> None:
    global MODEL, PROCESSOR, DRAFT_MODEL, DRAFT_BACKEND, MODEL_PATH
    configure_mlx_memory()
    from jang_tools.loader import load_jang_vlm_model

    start = time.monotonic()
    MODEL, PROCESSOR = load_jang_vlm_model(model_path)
    MODEL_PATH = model_path
    log(f"loaded JANG VLM model in {time.monotonic() - start:.2f}s: {model_path}")
    draft_kind = os.environ.get("OPENCLAW_JANG_DRAFT_KIND", "mtp").strip().lower()
    dflash_path = os.environ.get("OPENCLAW_JANG_DFLASH_DRAFT_MODEL")
    draft_path = dflash_path or os.environ.get("OPENCLAW_JANG_DRAFT_MODEL")
    if draft_path:
        draft_start = time.monotonic()
        if draft_kind == "dflash" or dflash_path:
            if not env_bool("OPENCLAW_JANG_DFLASH_EXPERIMENTAL_ACK"):
                raise RuntimeError(
                    "DFlash is experimental for the JANQ target; set "
                    "OPENCLAW_JANG_DFLASH_EXPERIMENTAL_ACK=1 for isolated canaries only"
                )
            from dflash.model_mlx import load_draft

            DRAFT_MODEL = load_draft(draft_path)
            validate_dflash_compatibility(MODEL, DRAFT_MODEL)
            DRAFT_BACKEND = "dflash"
        else:
            from mlx_vlm.speculative.drafters import load_drafter

            DRAFT_MODEL = load_drafter(draft_path, kind=draft_kind)
            DRAFT_BACKEND = "mtp"
        block = getattr(getattr(DRAFT_MODEL, "config", None), "block_size", "?")
        log(
            "loaded Gemma drafter "
            f"backend={DRAFT_BACKEND or draft_kind} block={block} in {time.monotonic() - draft_start:.2f}s: {draft_path}"
        )


def model_worker(model_path: str) -> None:
    global MODEL_LOAD_ERROR
    try:
        load_model(model_path)
    except Exception as error:
        MODEL_LOAD_ERROR = error
        MODEL_READY.set()
        log(f"model load failed: {type(error).__name__}: {error}")
        return
    MODEL_READY.set()
    while True:
        fn, result_queue = MODEL_TASKS.get()
        try:
            result_queue.put((True, fn()))
        except Exception as error:
            result_queue.put((False, error))


def submit_model_task(fn: Any) -> "queue.Queue[tuple[bool, Any]]":
    result_queue: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)
    MODEL_TASKS.put((fn, result_queue))
    return result_queue


def run_model_task(fn: Any) -> Any:
    ok, value = submit_model_task(fn).get()
    if ok:
        return value
    raise value


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized: list[dict[str, str]] = []
    system_parts: list[str] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "").strip()
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        content = raw.get("content")
        text = content_to_text(content)
        if not text and role != "assistant":
            continue
        if role == "system":
            system_parts.append(text)
        else:
            normalized.append({"role": role, "content": text})
    if system_parts:
        normalized.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError("No user query found in messages")
    return normalized


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    return str(content)


def inject_tools(messages: list[dict[str, str]], tools: Any) -> list[dict[str, str]]:
    if not isinstance(tools, list) or not tools:
        return messages
    instruction = (
        "Available tools are provided as JSON schemas. If a tool is needed, "
        "emit exactly one tool call using the model's tool-call format. "
        "Do not expose private reasoning.\n\n"
        + json.dumps(tools, separators=(",", ":"))
    )
    if messages and messages[0]["role"] == "system":
        first = dict(messages[0])
        first["content"] = f"{first['content']}\n\n{instruction}"
        return [first, *messages[1:]]
    return [{"role": "system", "content": instruction}, *messages]


def build_prompt(payload: dict[str, Any]) -> str:
    assert PROCESSOR is not None
    messages = normalize_messages(payload.get("messages"))
    tools = payload.get("tools")
    enable_thinking = payload.get("enable_thinking")
    if enable_thinking is None:
        enable_thinking = False
    attempts = [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": bool(enable_thinking), "tools": tools},
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": bool(enable_thinking)},
        {"tokenize": False, "add_generation_prompt": True},
    ]
    if tools:
        attempts.append({"tokenize": False, "add_generation_prompt": True, "enable_thinking": bool(enable_thinking)})
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            local_messages = inject_tools(messages, tools) if tools and "tools" not in kwargs else messages
            clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
            return PROCESSOR.apply_chat_template(local_messages, **clean_kwargs)
        except TypeError as error:
            last_error = error
    raise RuntimeError(f"could not render chat template: {last_error}")


def generation_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    max_tokens = int(payload.get("max_tokens") or env_int("OPENCLAW_JANG_MAX_TOKENS", 4096))
    max_tokens = max(1, min(max_tokens, env_int("OPENCLAW_JANG_HARD_MAX_TOKENS", 4096)))
    default_temperature = env_float(
        "OPENCLAW_JANG_DEFAULT_TEMPERATURE",
        0.0 if DRAFT_MODEL is not None else 0.3,
    )
    default_repetition_penalty = env_float(
        "OPENCLAW_JANG_REPETITION_PENALTY",
        1.0 if DRAFT_MODEL is not None else 1.05,
    )
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": float(payload.get("temperature") if payload.get("temperature") is not None else default_temperature),
        "top_p": float(payload.get("top_p") if payload.get("top_p") is not None else 0.9),
        "repetition_penalty": float(
            payload.get("repetition_penalty")
            if payload.get("repetition_penalty") is not None
            else default_repetition_penalty
        ),
    }
    for name in ("top_k", "min_p"):
        if payload.get(name) is not None:
            kwargs[name] = payload[name]
    prefill_step_size = payload.get("prefill_step_size")
    if prefill_step_size is None:
        prefill_step_size = env_int("OPENCLAW_JANG_PREFILL_STEP_SIZE", 2048)
    if prefill_step_size:
        kwargs["prefill_step_size"] = max(1, int(prefill_step_size))
    max_kv_size = payload.get("max_kv_size")
    if max_kv_size is None:
        max_kv_size = env_int("OPENCLAW_JANG_MAX_KV_SIZE", 0)
    if max_kv_size:
        kwargs["max_kv_size"] = max(1, int(max_kv_size))
    if DRAFT_MODEL is not None and DRAFT_BACKEND != "dflash":
        kwargs["draft_model"] = DRAFT_MODEL
        kwargs["draft_kind"] = os.environ.get("OPENCLAW_JANG_DRAFT_KIND", "mtp")
        draft_block_size = payload.get("draft_block_size")
        if draft_block_size is None:
            draft_block_size = env_int("OPENCLAW_JANG_DRAFT_BLOCK_SIZE", 0)
        if draft_block_size:
            kwargs["draft_block_size"] = max(1, int(draft_block_size))
    return kwargs


def has_request_tools(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    return isinstance(tools, list) and bool(tools)


def request_enables_thinking(payload: dict[str, Any]) -> bool:
    return bool(payload.get("enable_thinking"))


def should_use_dflash(payload: dict[str, Any]) -> bool:
    if DRAFT_BACKEND != "dflash" or DRAFT_MODEL is None:
        return False
    if has_request_tools(payload) and not env_bool("OPENCLAW_JANG_DFLASH_ALLOW_TOOLS"):
        log("DFlash bypassed for tool-bearing request; using target-only decode")
        return False
    if request_enables_thinking(payload) and not env_bool("OPENCLAW_JANG_DFLASH_ALLOW_THINKING"):
        log("DFlash bypassed for thinking request; using target-only decode")
        return False
    return True


def dflash_generation_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    kwargs = generation_kwargs(payload)
    allowed = {
        "max_tokens": kwargs["max_tokens"],
        "temperature": kwargs["temperature"],
    }
    draft_block_size = payload.get("draft_block_size")
    if draft_block_size is None:
        draft_block_size = env_int("OPENCLAW_JANG_DFLASH_BLOCK_SIZE", 0) or env_int("OPENCLAW_JANG_DRAFT_BLOCK_SIZE", 0)
    if draft_block_size:
        allowed["block_size"] = max(1, int(draft_block_size))
    return allowed


def target_text_config(model: Any) -> Any:
    return getattr(getattr(model, "config", None), "text_config", None) or getattr(
        getattr(model, "language_model", None), "config", None
    )


def validate_dflash_compatibility(model: Any, draft: Any) -> None:
    target_cfg = target_text_config(model)
    draft_cfg = getattr(draft, "config", None)
    if target_cfg is None or draft_cfg is None:
        raise RuntimeError("DFlash compatibility check requires target and draft configs")
    target_layers = int(getattr(target_cfg, "num_hidden_layers", 0) or 0)
    target_layer_ids = tuple(getattr(draft_cfg, "target_layer_ids", ()) or ())
    checks = {
        "hidden_size": (
            getattr(target_cfg, "hidden_size", None),
            getattr(draft_cfg, "hidden_size", None),
        ),
        "vocab_size": (
            getattr(target_cfg, "vocab_size", None),
            getattr(draft_cfg, "vocab_size", None),
        ),
        "max_position_embeddings": (
            getattr(target_cfg, "max_position_embeddings", None),
            getattr(draft_cfg, "max_position_embeddings", None),
        ),
        "final_logit_softcapping": (
            getattr(target_cfg, "final_logit_softcapping", None),
            getattr(draft_cfg, "final_logit_softcapping", None),
        ),
    }
    mismatches = [
        f"{name}: target={target!r} draft={draft_value!r}"
        for name, (target, draft_value) in checks.items()
        if target != draft_value
    ]
    if target_layers <= 0:
        mismatches.append("target num_hidden_layers is unavailable")
    bad_layers = [layer for layer in target_layer_ids if int(layer) < 0 or int(layer) >= target_layers]
    if bad_layers:
        mismatches.append(f"target_layer_ids out of range for target layers={target_layers}: {bad_layers}")
    if int(getattr(draft_cfg, "num_target_layers", target_layers) or 0) != target_layers:
        mismatches.append(
            f"num_target_layers: target={target_layers} draft={getattr(draft_cfg, 'num_target_layers', None)!r}"
        )
    if mismatches:
        raise RuntimeError("DFlash/JANQ structural mismatch: " + "; ".join(mismatches))
    log(
        "DFlash structural compatibility passed "
        f"target_layers={target_layers} target_layer_ids={list(target_layer_ids)} "
        "but JANQ behavioral acceptance still requires canary benchmarks"
    )


class DFlashVLMTargetAdapter:
    """Expose a JANG-loaded mlx_vlm Gemma4 target with mlx_lm-like semantics."""

    def __init__(self, model: Any):
        self._model = model
        self.language_model = model.language_model

    def make_cache(self) -> Any:
        return self.language_model.make_cache()

    def __call__(self, input_ids: Any, cache: Any = None, **kwargs: Any) -> Any:
        output = self._model(input_ids, cache=cache, **kwargs)
        return getattr(output, "logits", output)


def dflash_tokenizer() -> Any:
    tokenizer = getattr(PROCESSOR, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(PROCESSOR, "_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("DFlash requires a tokenizer on the Gemma processor")
    return tokenizer


def record_dflash_acceptance(response: Any, *, first_response: bool) -> None:
    if first_response:
        return
    accepted = int(getattr(response, "accepted", 0) or 0)
    if accepted:
        DFLASH_ACCEPT_LENS.append(accepted)


def strip_reasoning_markers(text: str) -> tuple[str, str | None]:
    reasoning_parts: list[str] = []
    cleaned = text
    while GEMMA_THOUGHT_START in cleaned and GEMMA_THOUGHT_END in cleaned:
        before, rest = cleaned.split(GEMMA_THOUGHT_START, 1)
        thought, after = rest.split(GEMMA_THOUGHT_END, 1)
        reasoning_parts.append(thought.strip())
        cleaned = before + after
    cleaned = re.sub(r"(?is)<think>(.*?)</think>", lambda match: reasoning_parts.append(match.group(1).strip()) or "", cleaned)
    cleaned = cleaned.replace(GEMMA_TURN_END, "")
    cleaned = re.sub(r"(?im)^(?:\\s*thought\\s*\\n){2,}", "", cleaned)
    return cleaned.strip(), "\n\n".join(part for part in reasoning_parts if part) or None


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
            repeats, remainder = divmod(len(lowered), size)
            if repeats >= 4 and remainder == 0 and unit * repeats == lowered:
                return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())
    if len(words) >= 12 and max(words.count(word) for word in set(words)) / len(words) >= 0.7:
        return True
    return False


def parse_native_gemma_value(raw: str) -> Any:
    value = raw.strip()
    if value.startswith('<|"|>') and value.endswith('<|"|>'):
        return value[len('<|"|>') : -len('<|"|>')]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.replace('<|"|>', '"')


def split_native_gemma_args(raw: str) -> dict[str, Any]:
    args = raw.strip()
    if args.startswith("{") and args.endswith("}"):
        args = args[1:-1]
    parsed: dict[str, Any] = {}
    index = 0
    while index < len(args):
        while index < len(args) and args[index] in " ,\n\t":
            index += 1
        key_start = index
        while index < len(args) and re.match(r"[\w.-]", args[index]):
            index += 1
        key = args[key_start:index]
        while index < len(args) and args[index] in " \n\t":
            index += 1
        if not key or index >= len(args) or args[index] != ":":
            break
        index += 1
        while index < len(args) and args[index] in " \n\t":
            index += 1
        if args.startswith('<|"|>', index):
            value_start = index
            index += len('<|"|>')
            end = args.find('<|"|>', index)
            if end == -1:
                value = args[value_start:]
                index = len(args)
            else:
                index = end + len('<|"|>')
                value = args[value_start:index]
        else:
            value_start = index
            brace_depth = 0
            while index < len(args):
                char = args[index]
                if char == "{":
                    brace_depth += 1
                elif char == "}":
                    brace_depth = max(0, brace_depth - 1)
                elif char == "," and brace_depth == 0:
                    break
                index += 1
            value = args[value_start:index]
        parsed[key] = parse_native_gemma_value(value)
        if index < len(args) and args[index] == ",":
            index += 1
    return parsed


def make_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, separators=(",", ":"))
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def parse_tool_call(text: str) -> tuple[str, list[dict[str, Any]]]:
    native = re.search(
        r"(?is)<\|tool_call\>\s*call:([a-zA-Z_][\w.-]*)\s*(\{.*?\})\s*(?:<tool_call\|>|$)",
        text,
    )
    if native:
        cleaned = (text[: native.start()] + text[native.end() :]).strip()
        return cleaned, [make_tool_call(native.group(1), split_native_gemma_args(native.group(2)))]
    match = re.search(r"(?is)<tool_call>\s*(\{.*?\})\s*</tool_call>", text)
    if not match:
        match = re.search(r"(?is)<tool_call\s*>\s*(\{.*)", text)
    if not match:
        return text, []
    try:
        call = json.loads(match.group(1))
    except Exception:
        return text, []
    name = call.get("name") or call.get("function", {}).get("name")
    arguments = call.get("arguments") or call.get("function", {}).get("arguments") or {}
    if not isinstance(name, str) or not name:
        return text, []
    cleaned = (text[: match.start()] + text[match.end() :]).strip()
    return cleaned, [make_tool_call(name, arguments)]


def speculative_stats_since(start_index: int) -> str:
    if DRAFT_MODEL is None:
        return ""
    accept_lens = DFLASH_ACCEPT_LENS if DRAFT_BACKEND == "dflash" else (getattr(DRAFT_MODEL, "accept_lens", None) or [])
    if start_index > len(accept_lens):
        start_index = 0
    recent = accept_lens[start_index:]
    if not recent:
        return ""
    mean_accept = sum(recent) / len(recent)
    return f" mtp_rounds={len(recent)} mean_accept={mean_accept:.2f}"


def current_speculative_stat_index() -> int:
    if DRAFT_MODEL is None:
        return 0
    if DRAFT_BACKEND == "dflash":
        return len(DFLASH_ACCEPT_LENS)
    return len(getattr(DRAFT_MODEL, "accept_lens", None) or [])


def stream_visible_text(raw_text: str) -> str:
    """Return only content that is safe to stream before final parsing."""
    first_tool = len(raw_text)
    for marker in ("<|tool_call>", "<tool_call"):
        index = raw_text.find(marker)
        if index >= 0:
            first_tool = min(first_tool, index)
    visible = raw_text[:first_tool]
    while GEMMA_THOUGHT_START in visible:
        before, rest = visible.split(GEMMA_THOUGHT_START, 1)
        if GEMMA_THOUGHT_END not in rest:
            visible = before
            break
        _thought, after = rest.split(GEMMA_THOUGHT_END, 1)
        visible = before + after
    visible = re.sub(r"(?is)<think>.*?</think>", "", visible)
    for partial in ("<|channel>thought", "<think", "<|tool", "<tool"):
        index = visible.rfind(partial)
        if index >= 0 and index > len(visible) - 64:
            visible = visible[:index]
    visible = visible.replace(GEMMA_TURN_END, "")
    visible = re.sub(r"(?im)^(?:\\s*thought\\s*\\n){2,}", "", visible)
    return visible


def chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    return run_model_task(lambda: _chat_completion_on_worker(payload))


def dflash_stream(prompt: str, payload: dict[str, Any]) -> Any:
    if DRAFT_MODEL is None:
        raise RuntimeError("DFlash draft model is not loaded")
    from dflash.model_mlx import stream_generate as dflash_stream_generate

    target = DFlashVLMTargetAdapter(MODEL)
    return dflash_stream_generate(
        target,
        DRAFT_MODEL,
        dflash_tokenizer(),
        prompt,
        **dflash_generation_kwargs(payload),
    )


def _chat_completion_on_worker(payload: dict[str, Any]) -> dict[str, Any]:
    from mlx_vlm import generate

    prompt = build_prompt(payload)
    start = time.monotonic()
    speculative_start = current_speculative_stat_index()
    if should_use_dflash(payload):
        raw_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        first_dflash_response = True
        for response in dflash_stream(prompt, payload):
            raw_text += str(getattr(response, "text", "") or "")
            prompt_tokens = int(getattr(response, "prompt_tokens", prompt_tokens) or prompt_tokens)
            completion_tokens = int(getattr(response, "generation_tokens", completion_tokens) or completion_tokens)
            record_dflash_acceptance(response, first_response=first_dflash_response)
            first_dflash_response = False
        result = None
    else:
        kwargs = generation_kwargs(payload)
        result = generate(MODEL, PROCESSOR, prompt, verbose=False, **kwargs)
        raw_text = str(getattr(result, "text", result) or "")
        prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(result, "generation_tokens", 0) or 0)
    content, reasoning = strip_reasoning_markers(raw_text)
    content, tool_calls = parse_tool_call(content)
    finish_reason = "tool_calls" if tool_calls else "stop"
    if has_repeated_token_loop(content):
        log(f"suppressed repeated-token loop in nonstream output: {content[:120]!r}")
        content = ""
        finish_reason = "content_filter"
    elapsed = time.monotonic() - start
    tok_s = completion_tokens / elapsed if elapsed > 0 else 0
    speculative = speculative_stats_since(speculative_start)
    log(
        f"chat completion: prompt={prompt_tokens} completion={completion_tokens} "
        f"elapsed={elapsed:.2f}s tok_s={tok_s:.1f}{speculative}"
    )
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning and os.environ.get("OPENCLAW_JANG_FORWARD_REASONING", "0").lower() in {"1", "true", "yes", "on"}:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _stream_chat_completion_on_worker(
    payload: dict[str, Any],
    event_queue: "queue.Queue[tuple[str, Any]]",
) -> dict[str, Any]:
    prompt = build_prompt(payload)
    start = time.monotonic()
    speculative_start = current_speculative_stat_index()
    raw_text = ""
    emitted = ""
    last_response = None
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = "stop"
    try:
        use_dflash = should_use_dflash(payload)
        if use_dflash:
            response_iter = dflash_stream(prompt, payload)
        else:
            from mlx_vlm import stream_generate

            response_iter = stream_generate(MODEL, PROCESSOR, prompt, verbose=False, **generation_kwargs(payload))
        first_dflash_response = True
        for response in response_iter:
            last_response = response
            segment = str(getattr(response, "text", "") or "")
            if segment:
                raw_text += segment
            prompt_tokens = int(getattr(response, "prompt_tokens", prompt_tokens) or prompt_tokens)
            completion_tokens = int(getattr(response, "generation_tokens", completion_tokens) or completion_tokens)
            if use_dflash:
                record_dflash_acceptance(response, first_response=first_dflash_response)
                first_dflash_response = False
            visible = stream_visible_text(raw_text)
            if has_repeated_token_loop(visible or raw_text):
                log(f"suppressed repeated-token loop in stream output: {(visible or raw_text)[:120]!r}")
                raw_text = ""
                emitted = ""
                finish_reason = "content_filter"
                break
            if visible.startswith(emitted) and len(visible) > len(emitted):
                delta = visible[len(emitted) :]
                emitted = visible
                event_queue.put(("content", delta))
        if last_response is not None:
            prompt_tokens = int(getattr(last_response, "prompt_tokens", prompt_tokens) or prompt_tokens)
            completion_tokens = int(getattr(last_response, "generation_tokens", completion_tokens) or completion_tokens)
        content, reasoning = strip_reasoning_markers(raw_text)
        content, tool_calls = parse_tool_call(content)
        if finish_reason != "content_filter":
            finish_reason = "tool_calls" if tool_calls else "stop"
        if has_repeated_token_loop(content):
            log(f"suppressed repeated-token loop in final stream output: {content[:120]!r}")
            content = ""
            tool_calls = []
            finish_reason = "content_filter"
        if tool_calls:
            event_queue.put(("tool_calls", tool_calls))
        elif content.startswith(emitted) and len(content) > len(emitted):
            event_queue.put(("content", content[len(emitted) :]))
        elif content and not emitted.startswith(content):
            event_queue.put(("content", content))
        elapsed = time.monotonic() - start
        tok_s = completion_tokens / elapsed if elapsed > 0 else 0
        speculative = speculative_stats_since(speculative_start)
        log(
            f"stream chat completion: prompt={prompt_tokens} completion={completion_tokens} "
            f"elapsed={elapsed:.2f}s tok_s={tok_s:.1f}{speculative}"
        )
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning and os.environ.get("OPENCLAW_JANG_FORWARD_REASONING", "0").lower() in {"1", "true", "yes", "on"}:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        result = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        event_queue.put(("done", result))
        return result
    except Exception as error:
        event_queue.put(("error", error))
        raise


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenClawJangVLM/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        log(fmt % args)

    def send_json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"/v1/models", "/models"}:
            self.send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "openclaw-jang-vlm",
                        }
                    ],
                }
            )
            return
        self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as error:
            self.send_json({"error": {"message": f"invalid JSON: {error}"}}, status=400)
            return
        if self.path.rstrip("/") not in {"/v1/chat/completions", "/chat/completions"}:
            self.send_json({"error": "not found"}, status=404)
            return
        try:
            if payload.get("stream"):
                self.stream_chat(payload)
            else:
                self.send_json(chat_completion(payload))
        except ValueError as error:
            self.send_json({"error": {"message": str(error), "type": "invalid_request_error"}}, status=400)
        except Exception as error:
            log(f"request failed: {type(error).__name__}: {error}")
            self.send_json({"error": {"message": str(error), "type": "server_error"}}, status=500)

    def write_sse(self, value: dict[str, Any]) -> None:
        self.wfile.write(b"data: ")
        self.wfile.write(json.dumps(value, separators=(",", ":")).encode("utf-8"))
        self.wfile.write(b"\n\n")
        self.wfile.flush()

    def stream_chat(self, payload: dict[str, Any]) -> None:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.write_sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        completion_payload = dict(payload)
        completion_payload["stream"] = False
        event_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        result_queue = submit_model_task(
            lambda: _stream_chat_completion_on_worker(completion_payload, event_queue)
        )
        final_result: dict[str, Any] | None = None
        while True:
            try:
                event, value = event_queue.get(timeout=0.25)
            except queue.Empty:
                try:
                    self.wfile.write(b": openclaw-jang-vlm-server: generation still running\n\n")
                    self.wfile.flush()
                except BrokenPipeError:
                    return
                continue
            if event == "content":
                if isinstance(value, str) and value:
                    self.write_sse(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {"content": value}, "finish_reason": None}],
                        }
                    )
                continue
            if event == "tool_calls":
                self.write_sse(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {"tool_calls": value}, "finish_reason": None}],
                    }
                )
                continue
            if event == "error":
                raise value
            if event == "done":
                final_result = value
                break
        if not result_queue.empty():
            ok, queued_result = result_queue.get()
            if not ok:
                raise queued_result
            if final_result is None:
                final_result = queued_result
        if final_result is None:
            raise RuntimeError("stream finished without a final result")
        choice = final_result.get("choices", [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            message = {}
        tool_calls = message.get("tool_calls")
        content = message.get("content")
        if not tool_calls and not content:
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": "[OpenClaw model returned an empty response]"}, "finish_reason": None}],
                }
            )
        finish_reason = "tool_calls" if isinstance(tool_calls, list) and tool_calls else choice.get("finish_reason", "stop")
        self.write_sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": final_result.get("usage"),
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenClaw JANG/VLM server.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    return parser.parse_args()


def main() -> int:
    global MODEL_ID
    args = parse_args()
    MODEL_ID = args.served_model_name
    threading.Thread(target=model_worker, args=(args.model_path,), name="openclaw-jang-vlm-model", daemon=True).start()
    if not MODEL_READY.wait(timeout=env_int("OPENCLAW_JANG_LOAD_TIMEOUT_SECONDS", 180)):
        raise RuntimeError("model load timed out")
    if MODEL_LOAD_ERROR is not None:
        raise MODEL_LOAD_ERROR
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    def shutdown(_signum: int, _frame: Any) -> None:
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log(f"listening on http://{args.host}:{args.port}/v1")
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
