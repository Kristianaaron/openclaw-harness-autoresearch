#!/usr/bin/env python3
"""Calibrate a Gemma 4 MTP assistant drafter against OpenClaw's JANQ target.

The first production-safe calibration target is block-size 2: improve the
assistant's first drafted token acceptance while leaving the JANQ target frozen.
This script tunes only the drafter pre-projection by default, then writes a
normal MLX drafter directory that can be quantized and benchmarked separately.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

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


def copy_metadata(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "README.md"):
        src = source / name
        if src.exists():
            shutil.copy2(src, destination / name)


def load_target_and_drafter(target_path: str, drafter_path: str) -> tuple[Any, Any, Any]:
    from jang_tools.loader import load_jang_vlm_model
    from mlx_vlm.speculative.drafters import load_drafter

    model, processor = load_jang_vlm_model(target_path)
    drafter = load_drafter(drafter_path, kind="mtp")
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
    log("loading JANQ target and BF16 drafter")
    model, processor, drafter = load_target_and_drafter(args.target_path, args.drafter_path)

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
    train_traces = build_traces(model, processor, train_prompts, args.positions_per_prompt)
    eval_traces = build_traces(model, processor, eval_prompts, args.positions_per_prompt)
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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> int:
    return train(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
