#!/usr/bin/env python3
"""Calibrate a Gemma 4 MTP assistant drafter against OpenClaw's JANQ target.

The first production-safe calibration target is block-size 2: improve the
assistant's first drafted token acceptance while leaving the JANQ target frozen.
This script tunes only the drafter pre-projection by default, then writes a
normal MLX drafter directory that can be quantized and benchmarked separately.
"""

from __future__ import annotations

import argparse
import os
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


OPENCLAW_RUNTIME_SITE = Path(
    os.environ.get("OPENCLAW_JANG_TARGET", Path.home() / ".openclaw" / "runtime" / "rapid-mlx" / "site")
).expanduser()
if OPENCLAW_RUNTIME_SITE.exists():
    sys.path.insert(0, str(OPENCLAW_RUNTIME_SITE))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


DEFAULT_PROMPTS = [
    "Explain speculative decoding in one detailed paragraph.",
    "Write a simple Python function that adds two numbers and returns the result.",
    "List ten common Unix shell commands with one short use each.",
    "Summarize the purpose of an agent harness in practical engineering terms.",
    "Write a concise troubleshooting checklist for a local model server.",
    "Explain why prefill latency can dominate the first response from an LLM.",
    "Draft a short shell script that prints disk usage and memory pressure.",
    "Describe how a tool-calling agent should decide whether to use a shell command.",
    "Give a practical note on avoiding repeated reasoning-marker loops.",
    "Explain how a prefix cache can improve perceived latency in an agent.",
    "Write a brief status update for an engineering task that is halfway complete.",
    "List five ways to make a CLI tool safer for long-running local tasks.",
    "Explain why deterministic decoding helps speculative decoding acceptance.",
    "Write a small Python function that validates JSON and returns a dictionary.",
    "Describe the difference between decode throughput and prompt prefill throughput.",
    "Give a short guide to interpreting tokens-per-second benchmark results.",
]


def log(message: str) -> None:
    print(f"[openclaw-mtp-calibrate] {message}", flush=True)


def memory_snapshot() -> dict[str, int]:
    snapshot = {"free_mb": 0, "compressor_mb": 0, "swap_used_mb": 0, "pressure_free_percent": -1}
    try:
        vm_stat = subprocess.check_output(["/usr/bin/vm_stat"], text=True, stderr=subprocess.DEVNULL)
        page_size = 16384
        free_pages = speculative_pages = compressor_pages = 0
        for line in vm_stat.splitlines():
            if "page size of" in line:
                digits = "".join(ch for ch in line if ch.isdigit())
                if digits:
                    page_size = int(digits)
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":", 1)[1].strip().rstrip("."))
            elif line.startswith("Pages speculative:"):
                speculative_pages = int(line.split(":", 1)[1].strip().rstrip("."))
            elif line.startswith("Pages occupied by compressor:"):
                compressor_pages = int(line.split(":", 1)[1].strip().rstrip("."))
        snapshot["free_mb"] = int((free_pages + speculative_pages) * page_size / 1048576)
        snapshot["compressor_mb"] = int(compressor_pages * page_size / 1048576)
    except Exception:
        pass
    try:
        swap = subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL)
        # Format: total = 2048.00M  used = 481.31M  free = ...
        parts = swap.replace("M", "").split()
        if "used" in parts:
            used_index = parts.index("used")
            if used_index + 2 < len(parts):
                snapshot["swap_used_mb"] = int(float(parts[used_index + 2]))
    except Exception:
        pass
    try:
        pressure = subprocess.check_output(["/usr/bin/memory_pressure"], text=True, stderr=subprocess.DEVNULL)
        for line in pressure.splitlines():
            if "System-wide memory free percentage:" in line:
                snapshot["pressure_free_percent"] = int(line.rsplit(":", 1)[1].strip().rstrip("%"))
                break
    except Exception:
        pass
    return snapshot


def memory_block_reason(args: argparse.Namespace, *, phase: str) -> str:
    snap = memory_snapshot()
    pressure_free = int(snap.get("pressure_free_percent", -1))
    if snap["free_mb"] and snap["free_mb"] < args.min_free_mb:
        if pressure_free < 0 or pressure_free < args.min_pressure_free_percent:
            return (
                f"{phase}: free={snap['free_mb']}MB<{args.min_free_mb}MB "
                f"pressureFree={pressure_free if pressure_free >= 0 else '?'}%<{args.min_pressure_free_percent}%"
            )
    if snap["compressor_mb"] >= args.max_compressor_mb:
        return f"{phase}: compressor={snap['compressor_mb']}MB>={args.max_compressor_mb}MB"
    if snap["swap_used_mb"] >= args.max_swap_mb:
        return f"{phase}: swap={snap['swap_used_mb']}MB>={args.max_swap_mb}MB"
    return ""


def require_memory_safe(args: argparse.Namespace, *, phase: str) -> None:
    try:
        mx.clear_cache()
        if hasattr(mx, "metal"):
            mx.metal.clear_cache()
    except Exception:
        pass
    reason = memory_block_reason(args, phase=phase)
    if reason:
        raise RuntimeError(f"calibration memory gate blocked: {reason}")


def configure_mlx_limits(args: argparse.Namespace) -> None:
    try:
        if mx.metal.is_available():
            info = mx.device_info()
            recommended = info.get("max_recommended_working_set_size", info.get("memory_size", 0))
            if recommended:
                mx.set_memory_limit(int(recommended * args.gpu_memory_utilization))
            mx.set_cache_limit(int(args.mlx_cache_gb * 1024**3))
            log(
                "MLX limits configured "
                f"gpu_memory_utilization={args.gpu_memory_utilization:.2f} "
                f"cache={args.mlx_cache_gb:.1f}GB"
            )
    except Exception as error:
        log(f"MLX limit configuration skipped: {error}")


def copy_metadata(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "README.md"):
        src = source / name
        if src.exists():
            shutil.copy2(src, destination / name)


def load_target(target_path: str) -> tuple[Any, Any]:
    from jang_tools.loader import load_jang_vlm_model

    return load_jang_vlm_model(target_path)


def load_mtp_drafter(drafter_path: str) -> Any:
    from mlx_vlm.speculative.drafters import load_drafter

    return load_drafter(drafter_path, kind="mtp")


def load_target_and_drafter(target_path: str, drafter_path: str) -> tuple[Any, Any, Any]:
    model, processor = load_target(target_path)
    drafter = load_mtp_drafter(drafter_path)
    drafter.bind(model)
    return model, processor, drafter


def make_prompt(processor: Any, text: str) -> str:
    messages = [
        {"role": "system", "content": "You are concise and deterministic."},
        {"role": "user", "content": text},
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prepare_text_inputs(model: Any, processor: Any, prompt: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    import importlib

    generate_module = importlib.import_module("mlx_vlm.generate")
    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model.config.model_type in ["gemma3", "gemma3n", "gemma4"]
        else True
    )
    inputs = generate_module.prepare_inputs(
        processor,
        images=None,
        audio=None,
        videos=None,
        prompts=prompt,
        image_token_index=getattr(model.config, "image_token_index", None),
        resize_shape=None,
        add_special_tokens=add_special_tokens,
    )
    input_ids = inputs.get("input_ids")
    pixel_values = inputs.get("pixel_values")
    mask = inputs.get("attention_mask")
    data_kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in ["input_ids", "pixel_values", "attention_mask"]
    }
    return input_ids, pixel_values, mask, data_kwargs


def make_prompt_cache(model: Any) -> list[Any]:
    import importlib

    generate_module = importlib.import_module("mlx_vlm.generate")
    return generate_module.cache.make_prompt_cache(model.language_model)


def target_traces(model: Any, processor: Any, prompt_text: str, count: int) -> list[dict[str, Any]]:
    prompt = make_prompt(processor, prompt_text)
    input_ids, pixel_values, mask, kwargs = prepare_text_inputs(model, processor, prompt)
    prompt_cache = make_prompt_cache(model)
    kwargs = dict(kwargs)
    kwargs["return_hidden"] = True
    kwargs["return_shared_kv"] = True

    embedding_output = model.get_input_embeddings(input_ids, pixel_values, mask=mask, **kwargs)
    inputs_embeds = embedding_output.inputs_embeds
    kwargs.update(
        {
            key: value
            for key, value in embedding_output.to_dict().items()
            if key != "inputs_embeds" and value is not None
        }
    )
    outputs = model.language_model(
        input_ids,
        inputs_embeds=inputs_embeds,
        cache=prompt_cache,
        **kwargs,
    )
    first_bonus = mx.argmax(outputs.logits[:, -1, :], axis=-1)
    hidden = outputs.hidden_states[-1][:, -1:, :]
    shared_kv = outputs.shared_kv_states
    mx.eval(first_bonus, hidden)

    traces: list[dict[str, Any]] = []
    bonus = first_bonus
    for _ in range(max(1, count)):
        trace = {
            "first_bonus": bonus,
            "hidden": hidden,
            "shared_kv": shared_kv,
            "kv_offset": int(prompt_cache[0].offset),
        }
        verify_out = model.language_model(
            bonus[:, None],
            cache=prompt_cache,
            return_hidden=True,
            return_shared_kv=True,
        )
        label = mx.argmax(verify_out.logits[:, -1, :], axis=-1)
        hidden = verify_out.hidden_states[-1][:, -1:, :]
        shared_kv = verify_out.shared_kv_states
        bonus = label
        mx.eval(label, hidden, bonus)
        trace["label"] = label
        traces.append(trace)
    return traces


def build_traces(model: Any, processor: Any, prompts: list[str], positions_per_prompt: int) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for prompt in prompts:
        traces.extend(target_traces(model, processor, prompt, positions_per_prompt))
    return traces


def drafter_logits(drafter: Any, trace: dict[str, Any]) -> Any:
    tok = trace["first_bonus"][:, None]
    tok_embed = drafter._input_embed(tok) * drafter._input_embed_scale
    inputs_embeds = mx.concatenate([tok_embed, trace["hidden"]], axis=-1)
    position_ids = mx.array([[trace["kv_offset"]]])
    _hidden, logits = drafter(inputs_embeds, trace["shared_kv"], position_ids)
    return logits[:, -1, :]


def acceptance(drafter: Any, traces: list[dict[str, Any]]) -> float:
    if not traces:
        return 0.0
    correct = 0
    for trace in traces:
        logits = drafter_logits(drafter, trace)
        pred = mx.argmax(logits, axis=-1)
        mx.eval(pred)
        correct += int(pred.item() == int(trace["label"].item()))
    return correct / len(traces)


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    source = Path(args.drafter_path).expanduser()
    output = Path(args.output_path).expanduser()
    require_memory_safe(args, phase="preflight")
    configure_mlx_limits(args)
    log("loading JANQ target and BF16 drafter")
    model, processor, drafter = load_target_and_drafter(args.target_path, args.drafter_path)
    require_memory_safe(args, phase="after-load")

    drafter.freeze()
    drafter.pre_projection.unfreeze()
    trainable = dict(tree_flatten(drafter.trainable_parameters()))
    log("trainable parameters: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in trainable.items()))

    prompts = list(DEFAULT_PROMPTS)
    if args.prompts_file:
        prompts.extend(
            line.strip()
            for line in Path(args.prompts_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    random.shuffle(prompts)
    train_prompts = prompts[: args.train_samples]
    eval_prompts = prompts[args.train_samples : args.train_samples + args.eval_samples]
    if not eval_prompts:
        eval_prompts = prompts[: min(args.eval_samples, len(prompts))]

    log(
        "building traces "
        f"train_prompts={len(train_prompts)} eval_prompts={len(eval_prompts)} "
        f"positions_per_prompt={args.positions_per_prompt}"
    )
    require_memory_safe(args, phase="before-trace-build")
    train_traces = build_traces(model, processor, train_prompts, args.positions_per_prompt)
    require_memory_safe(args, phase="after-train-traces")
    eval_traces = build_traces(model, processor, eval_prompts, args.positions_per_prompt)
    require_memory_safe(args, phase="after-eval-traces")
    baseline = acceptance(drafter, eval_traces)
    log(f"baseline first-draft acceptance={baseline:.3f}")

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    def loss_fn(draft_model: Any, trace: dict[str, Any]) -> Any:
        logits = drafter_logits(draft_model, trace)
        return nn.losses.cross_entropy(logits, trace["label"], reduction="mean")

    loss_and_grad = nn.value_and_grad(drafter, loss_fn)
    best_acceptance = baseline
    best_params = drafter.parameters()
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        if step == 1 or step % max(1, args.memory_check_every) == 0:
            require_memory_safe(args, phase=f"train-step-{step}")
        trace = train_traces[(step - 1) % len(train_traces)]
        loss, grads = loss_and_grad(drafter, trace)
        optimizer.update(drafter, grads)
        mx.eval(drafter.parameters(), optimizer.state)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            current = acceptance(drafter, eval_traces)
            log(f"step={step} loss={float(loss.item()):.4f} eval_acceptance={current:.3f}")
            if current >= best_acceptance:
                best_acceptance = current
                best_params = drafter.parameters()
    drafter.update(best_params)
    mx.eval(drafter.parameters())

    copy_metadata(source, output)
    from mlx_vlm.utils import save_weights

    save_weights(output, drafter)
    metrics = {
        "baseline_first_draft_acceptance": baseline,
        "best_first_draft_acceptance": best_acceptance,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "train_samples": len(train_traces),
        "eval_samples": len(eval_traces),
        "positions_per_prompt": args.positions_per_prompt,
        "elapsed_seconds": time.monotonic() - start,
        "trainable": list(trainable.keys()),
    }
    (output / "openclaw-calibration.json").write_text(json.dumps(metrics, indent=2) + "\n")
    log(f"saved calibrated drafter to {output}")
    log(json.dumps(metrics, indent=2))
    return 0


def write_probe_artifact(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    output = Path(args.output_path).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    stage = str(payload.get("stage", "unknown")).replace("/", "-")
    (output / f"openclaw-calibration-probe-{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")


def probe_stage(args: argparse.Namespace) -> int:
    """Run one bounded calibration readiness stage.

    This keeps the overnight supervisor from jumping straight into a full
    target+drafter training load when the previous blocker was after-load
    memory pressure.
    """
    stage = args.probe_stage
    started = time.monotonic()
    payload: dict[str, Any] = {
        "ok": False,
        "stage": stage,
        "timestamp": int(time.time()),
        "target_path": str(Path(args.target_path).expanduser()),
        "drafter_path": str(Path(args.drafter_path).expanduser()),
        "memory_before_mb": memory_snapshot(),
    }
    require_memory_safe(args, phase=f"{stage}:preflight")
    configure_mlx_limits(args)

    if stage == "metadata":
        target = Path(args.target_path).expanduser()
        drafter = Path(args.drafter_path).expanduser()
        missing = [
            str(path)
            for path in (
                target / "config.json",
                target / "tokenizer_config.json",
                drafter / "config.json",
            )
            if not path.exists()
        ]
        if missing:
            raise RuntimeError("calibration metadata gate blocked: missing=" + ",".join(missing))
    elif stage == "drafter-load":
        log("probe drafter-only load")
        drafter = load_mtp_drafter(args.drafter_path)
        mx.eval(drafter.parameters())
        require_memory_safe(args, phase="drafter-load:after-drafter")
        del drafter
    elif stage == "target-load":
        log("probe target-only load")
        model, processor = load_target(args.target_path)
        _ = processor
        require_memory_safe(args, phase="target-load:after-target")
        del model, processor
    elif stage == "combined-load":
        log("probe target+drafter load")
        model, processor, drafter = load_target_and_drafter(args.target_path, args.drafter_path)
        _ = processor
        require_memory_safe(args, phase="combined-load:after-load")
        del model, processor, drafter
    elif stage == "micro-step":
        args.train_samples = min(args.train_samples, 1)
        args.eval_samples = min(args.eval_samples, 1)
        args.positions_per_prompt = min(args.positions_per_prompt, 1)
        args.steps = min(args.steps, 1)
        args.eval_every = 1
        code = train(args)
        payload["train_returncode"] = code
        if code != 0:
            payload["memory_after_mb"] = memory_snapshot()
            payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
            write_probe_artifact(args, payload)
            return code
    else:
        raise RuntimeError(f"unknown calibration probe stage: {stage}")

    try:
        mx.clear_cache()
        if hasattr(mx, "metal"):
            mx.metal.clear_cache()
    except Exception:
        pass
    payload["ok"] = True
    payload["memory_after_mb"] = memory_snapshot()
    payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
    write_probe_artifact(args, payload)
    log(json.dumps(payload, indent=2))
    return 0


def write_blocked(output_path: str, reason: str) -> None:
    try:
        output = Path(output_path).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "blocked",
            "reason": reason,
            "timestamp": int(time.time()),
        }
        (output / "openclaw-calibration-blocked.json").write_text(json.dumps(payload, indent=2) + "\n")
    except Exception as error:
        log(f"could not write blocked calibration artifact: {error}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a Gemma MTP drafter against OpenClaw JANQ traces.")
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--drafter-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompts-file")
    parser.add_argument("--train-samples", type=int, default=12)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--positions-per-prompt", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-free-mb", type=int, default=12288)
    parser.add_argument("--max-compressor-mb", type=int, default=4096)
    parser.add_argument("--max-swap-mb", type=int, default=1024)
    parser.add_argument("--min-pressure-free-percent", type=int, default=20)
    parser.add_argument("--memory-check-every", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--mlx-cache-gb", type=float, default=8.0)
    parser.add_argument(
        "--probe-stage",
        choices=["metadata", "drafter-load", "target-load", "combined-load", "micro-step"],
        default="",
    )
    return parser.parse_args(argv)


def run_with_args(args: argparse.Namespace) -> int:
    try:
        if args.probe_stage:
            return probe_stage(args)
        return train(args)
    except RuntimeError as error:
        reason = str(error)
        log(reason)
        write_blocked(args.output_path, reason)
        return 2
    except MemoryError:
        reason = "calibration aborted before memory exhaustion could crash Metal/Python"
        log(reason)
        write_blocked(args.output_path, reason)
        return 2
    except Exception as error:
        reason = f"calibration failed safely: {type(error).__name__}: {error}"
        log(reason)
        write_blocked(args.output_path, reason)
        return 2


def main_with_args_for_test(argv: list[str]) -> int:
    return run_with_args(parse_args(argv))


def main() -> int:
    return run_with_args(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
