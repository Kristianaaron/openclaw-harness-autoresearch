"""Rapid-MLX JANG loader bridge for OpenClaw.

This module patches Rapid-MLX at the model-loading boundary only. Rapid still
owns the server, scheduler, prefix cache, streaming, parsers, and request API.
The bridge makes JANG/JANGTQ model directories load through ``jang_tools``
before Rapid falls back to its normal Gemma4/MLX loaders.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
import inspect
from pathlib import Path
from typing import Any

LOG = logging.getLogger("openclaw.rapid_jang")
INSTALLED = False


class OpenClawGemma4TextWrapper:
    """Run Gemma4 JANG text through the VLM embedding path on Rapid's LLM engine.

    Rapid's stock Gemma4 text wrapper calls the language model directly with
    token IDs. Gemma4's mlx-vlm implementation first builds embeddings and
    per-layer inputs via ``Model.get_input_embeddings(...)`` before calling the
    language model. Keeping that call shape is what made the MLLM path stable;
    this wrapper brings the same behavior to Rapid's text engine so prefix
    cache can be used without corrupting Gemma4/JANG logits.
    """

    def __init__(self, vlm_model):
        self.vlm_model = vlm_model
        self.language_model = vlm_model.language_model
        self.config = self.language_model.config
        self.model = self.language_model.model
        self.model_type = getattr(self.language_model, "model_type", "gemma4")
        self.openclaw_stop_token_ids = []

    def __call__(self, input_ids, cache=None, **kwargs):
        embedding_output = self.vlm_model.get_input_embeddings(
            input_ids=input_ids,
            **kwargs,
        )
        language_kwargs = {
            key: value
            for key, value in embedding_output.to_dict().items()
            if key != "inputs_embeds" and value is not None
        }
        output = self.language_model(
            input_ids=None,
            cache=cache,
            inputs_embeds=embedding_output.inputs_embeds,
            **language_kwargs,
        )
        return output.logits if hasattr(output, "logits") else output

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def head_dim(self):
        return self.language_model.head_dim

    @property
    def n_kv_heads(self):
        return self.language_model.n_kv_heads


class OpenClawSingleRequestCacheAdapter:
    """Expose a single-request KV cache through Rapid's batch cache surface."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def state(self):
        return self.inner.state

    @state.setter
    def state(self, value):
        self.inner.state = value

    @property
    def nbytes(self):
        return self.inner.nbytes

    def extract(self, idx: int):
        if idx != 0:
            raise IndexError("OpenClaw Gemma4 text path supports one active request per cache")
        return self.inner

    def filter(self, keep):
        if keep not in ([0], (0,), range(1)):
            if keep:
                raise IndexError("OpenClaw Gemma4 text path supports one active request per cache")


def _wrap_single_request_caches(caches):
    return [
        cache_entry
        if hasattr(cache_entry, "extract")
        else OpenClawSingleRequestCacheAdapter(cache_entry)
        for cache_entry in caches
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _resolve_path(model_name: str | Path) -> Path:
    path = Path(model_name).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_name)))


def _jang_config(path: Path) -> dict[str, Any]:
    for name in ("jang_config.json", "jjqf_config.json", "jang_cfg.json", "mxq_config.json"):
        candidate = path / name
        if candidate.exists():
            data = _read_json(candidate)
            data["_path"] = str(candidate)
            return data
    return {}


def is_jang_model(model_name: str | Path) -> bool:
    try:
        return bool(_jang_config(_resolve_path(model_name)))
    except Exception:
        return False


def _is_jangtq(jang_cfg: dict[str, Any]) -> bool:
    return jang_cfg.get("weight_format") == "mxtq" or jang_cfg.get("format") == "mxtq"


def _is_gemma4(path: Path) -> bool:
    config = _read_json(path / "config.json")
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    return config.get("model_type") == "gemma4" or text.get("model_type") == "gemma4_text"


def load_jang_for_llm(model_name: str | Path, tokenizer_config: dict[str, Any] | None = None):
    path = _resolve_path(model_name)
    jang_cfg = _jang_config(path)
    if not jang_cfg:
        raise FileNotFoundError(f"No JANG config found in {path}")
    if _is_gemma4(path):
        model, processor = load_jang_for_mllm(path)
        language_model = getattr(model, "language_model", None)
        if language_model is None:
            raise RuntimeError(
                "Gemma4 JANG loaded without a language_model; cannot attach to Rapid LLM scheduler"
            )
        tokenizer = getattr(processor, "tokenizer", processor)
        wrapper = OpenClawGemma4TextWrapper(model)
        stop_ids = []
        for token in ("<turn|>", "<end_of_turn>"):
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if isinstance(token_id, int) and token_id >= 0:
                stop_ids.append(token_id)
        wrapper.openclaw_stop_token_ids = sorted(set(stop_ids))
        LOG.info(
            "OpenClaw Rapid bridge wrapping Gemma4 JANG VLM embedding path for Rapid LLM scheduler: %s",
            path,
        )
        return wrapper, tokenizer
    if _is_jangtq(jang_cfg):
        from jang_tools.load_jangtq import load_jangtq_model

        LOG.info("OpenClaw Rapid bridge loading JANGTQ LLM: %s", path)
        return load_jangtq_model(str(path))
    from jang_tools.loader import load_jang_model

    LOG.info("OpenClaw Rapid bridge loading JANG LLM: %s", path)
    return load_jang_model(str(path))


def load_jang_for_mllm(model_name: str | Path):
    path = _resolve_path(model_name)
    jang_cfg = _jang_config(path)
    if not jang_cfg:
        raise FileNotFoundError(f"No JANG config found in {path}")
    if _is_jangtq(jang_cfg):
        from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model

        LOG.info("OpenClaw Rapid bridge loading JANGTQ VLM: %s", path)
        return load_jangtq_vlm_model(str(path))
    from jang_tools.loader import load_jang_vlm_model

    LOG.info("OpenClaw Rapid bridge loading JANG VLM: %s", path)
    return load_jang_vlm_model(str(path))


def _patch_tokenizer_loader() -> None:
    import vllm_mlx.utils.tokenizer as tokenizer_mod

    original = tokenizer_mod.load_model_with_fallback

    def load_model_with_fallback(model_name: str, tokenizer_config: dict[str, Any] | None = None):
        path = _resolve_path(model_name)
        if _jang_config(path):
            if _is_gemma4(path):
                LOG.info(
                    "Gemma4 JANG detected on Rapid LLM path; loading through jang_tools. "
                    "Rapid still owns scheduling after load."
                )
            return load_jang_for_llm(path, tokenizer_config)
        return original(model_name, tokenizer_config=tokenizer_config)

    tokenizer_mod.load_model_with_fallback = load_model_with_fallback


def _patch_mllm_loader() -> None:
    from vllm_mlx.models.mllm import MLXMultimodalLM
    from mlx_vlm.utils import load_config

    original = MLXMultimodalLM.load

    def load(self) -> None:
        if getattr(self, "_loaded", False):
            return
        path = _resolve_path(self.model_name)
        if not _jang_config(path):
            return original(self)
        self.model, self.processor = load_jang_for_mllm(path)
        try:
            setattr(self.model, "_openclaw_model_path", str(path))
        except Exception:
            pass
        self.config = load_config(str(path))
        self._loaded = True
        self._video_native = hasattr(self.model.config, "video_token_id") or hasattr(
            self.model.config,
            "video_token_index",
        )
        LOG.info("OpenClaw Rapid bridge loaded MLLM JANG model successfully: %s", path)

    MLXMultimodalLM.load = load


def _patch_mllm_scheduler_executor() -> None:
    """Keep every MLLM generation step on one executor thread.

    Rapid-MLX 0.6.1 creates an MLX stream for ``MLLMBatchGenerator`` during the
    first prefill step. If later decode steps run inline on the asyncio thread,
    MLX raises "There is no Stream(gpu, N) in current thread." Running all MLLM
    steps on the same single-worker executor preserves Rapid's scheduler while
    keeping the stream/thread ownership valid.
    """

    import asyncio
    import concurrent.futures

    import mlx.core as mx

    from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchGenerator
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    original_close = MLLMBatchGenerator.close
    original_next_impl = MLLMBatchGenerator._next
    original_step_impl = MLLMBatchGenerator._step
    original_stream = mx.stream
    rapid_has_dedicated_step_executor = "_injected_step_executor" in inspect.getsource(
        MLLMScheduler._process_loop
    )
    original_get_stop_tokens = MLLMScheduler._get_stop_tokens

    def get_stop_tokens_with_model_config(self):
        stop_tokens = set(original_get_stop_tokens(self) or set())

        def add(value):
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add(item)
                return
            try:
                stop_tokens.add(int(value))
            except (TypeError, ValueError):
                pass

        for source in (
            getattr(self, "model_config", None),
            getattr(getattr(self, "model_config", None), "text_config", None),
            getattr(getattr(self, "model", None), "config", None),
            getattr(getattr(getattr(self, "model", None), "language_model", None), "config", None),
        ):
            add(getattr(source, "eos_token_id", None))
            add(getattr(source, "eos_token_ids", None))

        model_path = getattr(getattr(self, "model", None), "_openclaw_model_path", None)
        if model_path:
            generation_config = Path(model_path) / "generation_config.json"
            if generation_config.exists():
                add(_read_json(generation_config).get("eos_token_id"))

        LOG.info("OpenClaw Rapid Gemma4 stop tokens: %s", sorted(stop_tokens))
        return stop_tokens

    MLLMScheduler._get_stop_tokens = get_stop_tokens_with_model_config

    class SafeStream:
        def __init__(self, stream):
            self.stream = stream
            self.context = None
            self.fallback = False

        def __enter__(self):
            self.context = original_stream(self.stream)
            try:
                return self.context.__enter__()
            except RuntimeError as error:
                if "There is no Stream" not in str(error):
                    raise
                LOG.warning("Falling back to current MLX stream for stale Rapid MLLM stream: %s", error)
                self.fallback = True
                self.context = None
                return None

        def __exit__(self, exc_type, exc, tb):
            if self.fallback or self.context is None:
                return False
            return self.context.__exit__(exc_type, exc, tb)

    def safe_stream(stream):
        return SafeStream(stream)

    mx.stream = safe_stream

    def close(self):
        try:
            return original_close(self)
        except RuntimeError as error:
            if "There is no Stream" in str(error):
                LOG.warning("Ignoring stale MLX stream during MLLM generator close: %s", error)
                old_limit = getattr(self, "_old_wired_limit", None)
                if old_limit is not None:
                    try:
                        mx.set_wired_limit(old_limit)
                    except Exception:
                        pass
                    self._old_wired_limit = None
                return None
            raise

    MLLMBatchGenerator.close = close

    def next_on_current_thread_stream(self):
        """Run the whole MLLM step on a stream created by this thread.

        Rapid creates ``MLLMBatchGenerator._stream`` during generator
        construction, which can happen on the asyncio/event-loop thread. MLX
        streams are thread-local, so reusing that object inside the executor
        thread can fail with "There is no Stream(gpu, N) in current thread".
        Create and install the stream inside the actual step thread, then call
        the implementation directly.
        """

        MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())
        try:
            with original_stream(MLLMBatchGenerator._stream):
                return original_next_impl(self)
        except Exception:
            LOG.error("Rapid MLLM step failed with traceback:\n%s", traceback.format_exc())
            raise

    if not rapid_has_dedicated_step_executor:
        MLLMBatchGenerator.next = next_on_current_thread_stream

    def process_prompts_with_mixed_cache(self, requests):
        """Process MLLM prompts while preserving Gemma4 mixed cache classes.

        Rapid-MLX 0.6.1 rejects ``RotatingKVCache`` before merge even though
        mlx-lm provides ``RotatingKVCache.merge()`` and
        ``BatchRotatingKVCache``. Gemma4 uses mixed full/sliding attention, so
        the cache list naturally contains both ``KVCache`` and
        ``RotatingKVCache`` layers. Let each layer merge through its own cache
        implementation instead of forcing all layers to be plain ``KVCache``.
        """

        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        tic = time.perf_counter()
        for req in requests:
            self._preprocess_request(req)

        total_prompt_tokens = sum(
            req.input_ids.size if req.input_ids is not None else 1 for req in requests
        )
        self._stats.prompt_tokens += total_prompt_tokens

        max_step_tokens = self.prefill_step_size * len(requests)
        if total_prompt_tokens > max_step_tokens:
            LOG.info(
                "OpenClaw Rapid bridge chunking long Gemma4 prefill: prompt_tokens=%s "
                "step_budget=%s requests=%s",
                total_prompt_tokens,
                max_step_tokens,
                len(requests),
            )

        first_tokens = []
        all_logprobs = []
        per_request_caches = []

        for req in requests:
            request_cache = make_prompt_cache(self.language_model)
            logits = _run_mlx_vlm_style_prefill(self, req, request_cache)
            last_logits = logits[:, -1, :]
            logprobs = last_logits - mx.logsumexp(
                last_logits, axis=-1, keepdims=True
            )
            req_sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
            sampled = req_sampler(logprobs)
            mx.eval(sampled, logprobs)
            first_tokens.append(sampled.item())
            all_logprobs.append(logprobs.squeeze(0))
            per_request_caches.append(request_cache)

        if len(per_request_caches) == 1:
            batch_cache = per_request_caches[0]
        else:
            try:
                batch_cache = [
                    per_request_caches[0][layer_idx].merge(
                        [c[layer_idx] for c in per_request_caches]
                    )
                    for layer_idx in range(len(per_request_caches[0]))
                ]
            except Exception:
                LOG.error("Failed to merge Gemma4 mixed KV caches:\n%s", traceback.format_exc())
                raise

        y = mx.array(first_tokens)
        self._stats.prompt_time += time.perf_counter() - tic
        return MLLMBatch(
            uids=[req.uid for req in requests],
            request_ids=[req.request_id for req in requests],
            y=y,
            logprobs=all_logprobs,
            max_tokens=[req.max_tokens for req in requests],
            num_tokens=[0] * len(requests),
            cache=batch_cache,
            requests=requests,
        )

    MLLMBatchGenerator._process_prompts = process_prompts_with_mixed_cache

    def _run_mlx_vlm_style_prefill(self, request, prompt_cache):
        """Prefill Gemma4/JANG exactly like ``mlx_vlm.generate_step``.

        The stable OpenClaw backend uses ``mlx_vlm.generate``. Its Gemma4 path
        first calls ``model.get_input_embeddings(...)`` and then feeds those
        embeddings into ``model.language_model(...)`` while carrying
        ``per_layer_inputs`` and chunked-prefill cache state. Rapid's generic
        MLLM scheduler called the full VLM forward method directly, which is
        not identical for Gemma4/JANG and produced repeated-token collapse.
        """

        kwargs = dict(getattr(request, "extra_kwargs", {}) or {})
        input_ids = request.input_ids
        if input_ids is None:
            raise ValueError("MLX-VLM style prefill requires prepared input_ids")
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        pixel_values = getattr(request, "pixel_values", None)
        attention_mask = getattr(request, "attention_mask", None)

        embedding_output = self.model.get_input_embeddings(
            input_ids,
            pixel_values,
            mask=attention_mask,
            **kwargs,
        )
        inputs_embeds = embedding_output.inputs_embeds
        language_kwargs = {
            key: value
            for key, value in embedding_output.to_dict().items()
            if key != "inputs_embeds" and value is not None
        }

        prefill_step_size = self.prefill_step_size
        if getattr(self.model, "no_chunked_prefill", False):
            prefill_step_size = None

        if (
            prefill_step_size is not None
            and inputs_embeds.shape[1] > prefill_step_size
        ):
            while inputs_embeds.shape[1] > 1:
                n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                self.language_model(
                    inputs=input_ids[:, :n_to_process],
                    inputs_embeds=inputs_embeds[:, :n_to_process],
                    cache=prompt_cache,
                    n_to_process=n_to_process,
                    **language_kwargs,
                )
                mx.eval([c.state for c in prompt_cache])
                inputs_embeds = inputs_embeds[:, n_to_process:]
                input_ids = input_ids[:, n_to_process:]
                mx.clear_cache()
            input_ids = input_ids[:, -1:]

        output = self.language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            cache=prompt_cache,
            **language_kwargs,
        )
        request.vision_encoded = True
        return output.logits if hasattr(output, "logits") else output

    def step_with_repetition_guard(self, input_tokens, cache, requests=None):
        """Rapid MLLM decode step with optional OpenClaw repetition penalties."""

        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        penalty = _float_env("OPENCLAW_RAPID_MLLM_REPETITION_PENALTY", 1.0)
        context = _int_env("OPENCLAW_RAPID_MLLM_REPETITION_CONTEXT_SIZE", 96)
        presence = _float_env("OPENCLAW_RAPID_MLLM_PRESENCE_PENALTY", 0.0)
        frequency = _float_env("OPENCLAW_RAPID_MLLM_FREQUENCY_PENALTY", 0.0)
        if (
            (not penalty or penalty == 1.0)
            and not presence
            and not frequency
        ):
            return original_step_impl(self, input_tokens, cache, requests)

        if input_tokens.ndim == 1:
            input_tokens = input_tokens[:, None]

        output = self.language_model(input_tokens, cache=cache)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        processors = make_logits_processors(
            repetition_penalty=penalty if penalty and penalty != 1.0 else None,
            repetition_context_size=context,
            presence_penalty=presence or None,
            presence_context_size=context,
            frequency_penalty=frequency or None,
            frequency_context_size=context,
        )

        input_history = input_tokens[:, 0].tolist()
        adjusted_rows = []
        for index in range(logits.shape[0]):
            row = logits[index : index + 1]
            history = []
            if requests and index < len(requests):
                history.extend(int(token) for token in getattr(requests[index], "output_tokens", []))
            token = input_history[index]
            if isinstance(token, list):
                token = token[0]
            history.append(int(token))
            for processor in processors:
                row = processor(history, row)
            adjusted_rows.append(row)
        if adjusted_rows:
            logits = mx.concatenate(adjusted_rows, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if requests and len(requests) == logprobs.shape[0]:
            sampled_tokens = []
            for i, req in enumerate(requests):
                sampler_key = (req.temperature, req.top_p)
                cached = getattr(req, "_cached_sampler", None)
                if cached is None or cached[0] != sampler_key:
                    req_sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
                    req._cached_sampler = (sampler_key, req_sampler)
                else:
                    req_sampler = cached[1]
                sampled_tokens.append(req_sampler(logprobs[i : i + 1]))
            sampled = mx.concatenate(sampled_tokens, axis=0)
        else:
            sampled = self.sampler(logprobs)

        return sampled, list(logprobs)

    MLLMBatchGenerator._step = step_with_repetition_guard

    original_process_batch_responses = MLLMScheduler._process_batch_responses

    def process_batch_responses_with_loop_guard(self, responses):
        from vllm_mlx.request import RequestStatus

        outputs, finished_ids = original_process_batch_responses(self, responses)
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        for output in outputs:
            request = self.running.get(output.request_id)
            token_ids = (
                list(getattr(request, "output_tokens", []) or [])
                if request is not None
                else list(getattr(output, "output_token_ids", []) or [])
            )
            if len(token_ids) < 4:
                continue
            decoded = output.output_text or tokenizer.decode(token_ids)
            if not _looks_repeated_text(decoded):
                continue
            clean = _clean_repeated_text(decoded)
            LOG.warning(
                "OpenClaw Rapid guard stopped repeated MLLM output for %s: %r",
                output.request_id,
                decoded[:160],
            )
            if request is not None:
                request.status = RequestStatus.FINISHED_STOPPED
                request.output_text = clean
                request.finish_reason = "stop"
            output.finished = True
            output.finish_reason = "stop"
            output.output_text = clean
            output.new_text = clean
            finished_ids.add(output.request_id)
        return outputs, finished_ids

    MLLMScheduler._process_batch_responses = process_batch_responses_with_loop_guard

    async def _process_loop(self):
        LOG.info("OpenClaw patched Rapid MLLM process loop active")
        if os.environ.get("OPENCLAW_RAPID_MLLM_INLINE_STEPS", "1").lower() in {"1", "true", "yes", "on"}:
            try:
                while self._running:
                    try:
                        if self.has_requests():
                            output = self._step_no_queue()
                            if output is not None:
                                self._distribute_outputs(output)
                            await asyncio.sleep(0)
                        else:
                            await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        break
                    except Exception as error:
                        LOG.error("Error in inline patched MLLM process loop: %s", error)
                        await asyncio.sleep(0.1)
            finally:
                self._step_executor = None
            return

        self._step_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="openclaw-mllm-step",
        )
        loop = asyncio.get_running_loop()

        def step_on_executor():
            MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())
            LOG.debug("OpenClaw reset MLLM stream on executor thread")
            return self._step_no_queue()

        try:
            while self._running:
                try:
                    if self.has_requests():
                        output = await loop.run_in_executor(
                            self._step_executor,
                            step_on_executor,
                        )
                        if output is not None:
                            self._distribute_outputs(output)
                        await asyncio.sleep(0)
                    else:
                        await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    break
                except Exception as error:
                    LOG.error("Error in patched MLLM process loop: %s", error)
                    await asyncio.sleep(0.1)
        finally:
            if self._step_executor is not None:
                self._step_executor.shutdown(wait=False)
                self._step_executor = None

    if not rapid_has_dedicated_step_executor:
        MLLMScheduler._process_loop = _process_loop


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        LOG.warning("Ignoring invalid %s=%r", name, raw)
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        LOG.warning("Ignoring invalid %s=%r", name, raw)
        return default


def _looks_repeated_text(text: str) -> bool:
    compact = "".join(ch for ch in text.lower() if ch.isalnum())
    if len(compact) >= 8:
        for marker in ("thought", "think", "ok"):
            repeats, remainder = divmod(len(compact), len(marker))
            if repeats >= 3 and remainder == 0 and marker * repeats == compact:
                return True
        for size in range(1, min(16, len(compact) // 3) + 1):
            unit = compact[:size]
            repeats, remainder = divmod(len(compact), size)
            if repeats >= 4 and remainder == 0 and unit * repeats == compact:
                return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())
    if len(words) >= 8 and max(words.count(word) for word in set(words)) / len(words) >= 0.7:
        return True
    return False


def _clean_repeated_text(text: str) -> str:
    stripped = text.strip()
    if re.match(r"(?is)^ok(?:\W*ok)+", stripped) or re.match(r"(?is)^ok\s*\\?\{?\s*ok", stripped):
        return "OK"
    match = re.match(r"(?is)^(.{1,240}?)(?:\s*\1){2,}", stripped)
    if match:
        return match.group(1).strip()
    return "[OpenClaw Rapid guard stopped repeated output]"


def _patch_batch_generator_repetition_penalty() -> None:
    """Apply OpenClaw's default repetition guard inside Rapid's LLM scheduler.

    Rapid-MLX 0.6.1 exposes ``SamplingParams.repetition_penalty`` but does not
    thread it into ``BatchGenerator.insert``. The OpenClaw overlay keeps this as
    a runtime patch so Homebrew's Rapid package remains untouched.
    """

    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_logits_processors

    original_insert = BatchGenerator.insert

    def insert(
        self,
        prompts,
        max_tokens=None,
        caches=None,
        all_tokens=None,
        samplers=None,
        logits_processors=None,
        state_machines=None,
    ):
        penalty = _float_env("OPENCLAW_RAPID_REPETITION_PENALTY", 1.25)
        if penalty and penalty != 1.0:
            context = _int_env("OPENCLAW_RAPID_REPETITION_CONTEXT_SIZE", 64)
            presence = _float_env("OPENCLAW_RAPID_PRESENCE_PENALTY", 0.1)
            frequency = _float_env("OPENCLAW_RAPID_FREQUENCY_PENALTY", 0.05)
            extra_processors = make_logits_processors(
                repetition_penalty=penalty,
                repetition_context_size=context,
                presence_penalty=presence or None,
                presence_context_size=context,
                frequency_penalty=frequency or None,
                frequency_context_size=context,
            )
            if extra_processors:
                prompt_count = len(prompts)
                if logits_processors is None:
                    logits_processors = [list(extra_processors) for _ in range(prompt_count)]
                else:
                    normalized = []
                    for index in range(prompt_count):
                        existing = logits_processors[index] if index < len(logits_processors) else []
                        normalized.append(list(existing or []) + list(extra_processors))
                    logits_processors = normalized
        return original_insert(
            self,
            prompts,
            max_tokens=max_tokens,
            caches=caches,
            all_tokens=all_tokens,
            samplers=samplers,
            logits_processors=logits_processors,
            state_machines=state_machines,
        )

    BatchGenerator.insert = insert


def _patch_gemma4_text_prompt_boundary() -> None:
    """Keep Rapid's LLM prompt boundary compatible with Gemma4 VLM prefill.

    mlx-lm's generic ``PromptProcessingBatch.generate`` processes all but the
    final prompt token, then lets ``GenerationBatch`` run that final token as a
    decode step. Gemma4's stable mlx-vlm path feeds the full prompt through
    ``get_input_embeddings`` and samples from the resulting last-position
    logits. The split boundary is enough to destabilize this JANG model on
    Rapid's LLM engine, so OpenClaw handles the first-token handoff explicitly
    for its Gemma4 text wrapper.
    """

    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator, GenerationBatch, PromptProcessingBatch
    from mlx_lm.models.cache import TokenBuffer

    original_init = BatchGenerator.__init__
    original_next = BatchGenerator._next
    original_generate = PromptProcessingBatch.generate

    def init_with_gemma4_stop_tokens(self, model, *args, stop_tokens=None, **kwargs):
        if isinstance(model, OpenClawGemma4TextWrapper):
            extra_stops = [[token_id] for token_id in getattr(model, "openclaw_stop_token_ids", [])]
            existing = list(stop_tokens or [])
            for stop in extra_stops:
                if stop not in existing:
                    existing.append(stop)
            stop_tokens = existing
        return original_init(self, model, *args, stop_tokens=stop_tokens, **kwargs)

    def generate_with_gemma4_full_prompt(self, tokens):
        if not isinstance(getattr(self, "model", None), OpenClawGemma4TextWrapper):
            return original_generate(self, tokens)
        if len(tokens) != 1:
            return original_generate(self, tokens)
        if not tokens or not tokens[0]:
            return original_generate(self, tokens)

        prompt_tokens = tokens[0]
        self.tokens[0] += prompt_tokens
        prompt_array = mx.array([prompt_tokens])
        logits = None

        while prompt_array.shape[1] > 0:
            n_to_process = min(self.prefill_step_size, prompt_array.shape[1])
            logits = self.model(
                prompt_array[:, :n_to_process],
                cache=self.prompt_cache,
            )
            mx.eval([c.state for c in self.prompt_cache], logits)
            prompt_array = prompt_array[:, n_to_process:]
            mx.clear_cache()

        if logits is None:
            return original_generate(self, tokens)

        last_logits = logits[:, -1, :]
        if any(self.logits_processors):
            processors = self.logits_processors[0] or []
            if processors:
                if self.tokens[0]:
                    token_context = TokenBuffer(self.tokens[0][:-1]).update_and_fetch(
                        mx.array([self.tokens[0][-1]])
                    )
                else:
                    token_context = mx.array([], dtype=mx.uint32)
                for processor in processors:
                    last_logits = processor(token_context, last_logits)
        logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
        sampler = self.samplers[0] if self.samplers and self.samplers[0] else self.fallback_sampler
        sampled = sampler(logprobs)
        mx.eval(sampled, logprobs)
        LOG.info(
            "OpenClaw Gemma4 text full-prompt handoff: prompt_tokens=%s first_token=%s",
            len(prompt_tokens),
            sampled.tolist(),
        )

        generation = GenerationBatch.__new__(GenerationBatch)
        generation.model = self.model
        generation.uids = list(self.uids)
        generation.prompt_cache = self.prompt_cache
        generation.tokens = self.tokens
        generation.samplers = self.samplers
        generation.fallback_sampler = self.fallback_sampler
        generation.logits_processors = self.logits_processors
        generation.state_machines = self.state_machines
        generation.max_tokens = self.max_tokens
        generation._current_tokens = None
        generation._current_logprobs = []
        generation._next_tokens = sampled
        generation._next_logprobs = list(logprobs)
        generation._token_context = [TokenBuffer(t) for t in self.tokens]
        generation._num_tokens = [0] * len(self.uids)
        generation._matcher_states = [m.make_state() for m in self.state_machines]

        self.uids = []
        self.prompt_cache = []
        self.tokens = []
        self.samplers = []
        self.logits_processors = []
        self.max_tokens = []

        return generation

    def next_with_gemma4_full_prompt(self):
        if not isinstance(getattr(self, "model", None), OpenClawGemma4TextWrapper):
            return original_next(self)

        generation_responses = []
        prompt_responses = []

        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        if len(self._prompt_batch) > 0 or self._currently_processing:
            return original_next(self)

        if self._unprocessed_sequences:
            uid, segments, max_tokens, caches, all_tokens, sampler, logits_processors, state_machine = (
                self._unprocessed_sequences.popleft()
            )
            full_prompt = [token for segment in segments for token in segment]
            if not full_prompt:
                return prompt_responses, generation_responses

            prompt_batch = PromptProcessingBatch(
                model=self.model,
                uids=[uid],
                caches=[caches],
                tokens=[all_tokens],
                prefill_step_size=self.prefill_step_size,
                samplers=[sampler],
                fallback_sampler=self.sampler,
                logits_processors=[logits_processors],
                state_machines=[state_machine],
                max_tokens=[max_tokens],
            )
            prompt_batch.prompt_cache = _wrap_single_request_caches(caches)
            self._prompt_tokens_counter += len(full_prompt)
            tic = time.perf_counter()
            gen_batch = prompt_batch.generate([full_prompt])
            self._prompt_time_counter += time.perf_counter() - tic
            self._generation_batch.extend(gen_batch)
            prompt_responses.append(
                PromptProcessingBatch.Response(
                    uid=uid,
                    progress=(len(full_prompt), len(full_prompt)),
                    end_of_segment=True,
                    end_of_prompt=True,
                )
            )

        return prompt_responses, generation_responses

    BatchGenerator.__init__ = init_with_gemma4_stop_tokens
    PromptProcessingBatch.generate = generate_with_gemma4_full_prompt
    BatchGenerator._next = next_with_gemma4_full_prompt


def _patch_batched_engine_output_guard() -> None:
    """Guard Rapid's public GenerationOutput surface.

    Some MLLM paths collect the final text above the scheduler layer, so a
    repeated-token failure can escape even when per-step scheduler output was
    patched. Keep the safety invariant at Rapid's engine boundary too: callers
    should never receive raw ``thought thought`` or ``OKOK`` style loops.
    """

    from vllm_mlx.engine.batched import BatchedEngine

    original_generate = BatchedEngine.generate
    original_stream_generate = BatchedEngine.stream_generate

    async def generate_with_output_guard(self, *args, **kwargs):
        output = await original_generate(self, *args, **kwargs)
        if _looks_repeated_text(getattr(output, "text", "") or ""):
            clean = _clean_repeated_text(output.text)
            LOG.warning("OpenClaw Rapid guard sanitized final output: %r", output.text[:160])
            output.text = clean
            output.new_text = clean
            output.finish_reason = "stop"
            output.finished = True
        return output

    async def stream_generate_with_output_guard(self, *args, **kwargs):
        async for output in original_stream_generate(self, *args, **kwargs):
            current = getattr(output, "text", "") or getattr(output, "new_text", "") or ""
            if _looks_repeated_text(current):
                clean = _clean_repeated_text(current)
                LOG.warning("OpenClaw Rapid guard sanitized streaming output: %r", current[:160])
                output.text = clean
                output.new_text = clean
                output.finish_reason = "stop"
                output.finished = True
                yield output
                break
            yield output

    BatchedEngine.generate = generate_with_output_guard
    BatchedEngine.stream_generate = stream_generate_with_output_guard


def install() -> None:
    global INSTALLED
    if INSTALLED:
        return
    try:
        import jang_tools  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "jang_tools is not importable in the Rapid backend environment. "
            "Run the OpenClaw Rapid launcher so it can install the OpenClaw-managed "
            "JANG dependency target."
        ) from error
    _patch_tokenizer_loader()
    _patch_mllm_loader()
    _patch_mllm_scheduler_executor()
    _patch_batch_generator_repetition_penalty()
    _patch_gemma4_text_prompt_boundary()
    _patch_batched_engine_output_guard()
    INSTALLED = True
    LOG.info("OpenClaw Rapid JANG bridge installed")
