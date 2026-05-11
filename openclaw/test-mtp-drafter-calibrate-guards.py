#!/usr/bin/env python3
"""Memory/crash guard checks for JANQ MTP drafter calibration."""

from __future__ import annotations

import importlib.util
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


HELPER_PATH = Path(__file__).with_name("openclaw-mtp-drafter-calibrate.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("openclaw_mtp_drafter_calibrate", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    defaults = {
        "min_free_mb": 12288,
        "max_compressor_mb": 4096,
        "max_swap_mb": 1024,
        "gpu_memory_utilization": 0.72,
        "mlx_cache_gb": 8.0,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def main() -> int:
    helper = load_helper()
    safe = {"free_mb": 24000, "compressor_mb": 512, "swap_used_mb": 0}
    low_free = {"free_mb": 8000, "compressor_mb": 512, "swap_used_mb": 0}
    compressed = {"free_mb": 24000, "compressor_mb": 5000, "swap_used_mb": 0}
    swapped = {"free_mb": 24000, "compressor_mb": 512, "swap_used_mb": 2048}
    with patch.object(helper, "memory_snapshot", return_value=safe):
        assert helper.memory_block_reason(args(), phase="preflight") == ""
        helper.require_memory_safe(args(), phase="preflight")
    with patch.object(helper, "memory_snapshot", return_value=low_free):
        reason = helper.memory_block_reason(args(), phase="preflight")
        assert "free=8000MB<12288MB" in reason
        try:
            helper.require_memory_safe(args(), phase="preflight")
        except RuntimeError as error:
            assert "calibration memory gate blocked" in str(error)
        else:
            raise AssertionError("low free memory should block calibration")
    with patch.object(helper, "memory_snapshot", return_value=compressed):
        assert "compressor=5000MB>=4096MB" in helper.memory_block_reason(args(), phase="after-load")
    with patch.object(helper, "memory_snapshot", return_value=swapped):
        assert "swap=2048MB>=1024MB" in helper.memory_block_reason(args(), phase="train-step-2")
    x = helper.mx.array([1.0])
    grad = helper.mx.grad(lambda value: helper.mx.sum(helper.detach_target_trace({"x": value})["x"]))(x)
    helper.mx.eval(grad)
    assert float(grad.item()) == 0.0
    assert helper.quantized_trainable_parameter_names(
        {
            "pre_projection.weight": object(),
            "pre_projection.scales": object(),
            "pre_projection.biases": object(),
        }
    ) == ["pre_projection.biases", "pre_projection.scales"]
    quantized = {
        "pre_projection.weight": object(),
        "pre_projection.scales": object(),
        "pre_projection.biases": object(),
    }
    direct_args = args(calibration_mode="direct-pre-projection", allow_quantized_drafter_training=False)
    adapter_args = args(calibration_mode="adapter-logit-distillation", allow_quantized_drafter_training=False)
    assert helper.blocks_quantized_drafter_training(direct_args, quantized) is True
    assert helper.blocks_quantized_drafter_training(adapter_args, quantized) is False
    adapter = helper.LogitBiasAdapter(8)
    logits = helper.mx.zeros((1, 8))
    shifted = adapter(logits)
    helper.mx.eval(shifted)
    assert tuple(shifted.shape) == (1, 8)
    assert helper.parse_args(
        [
            "--target-path",
            "target",
            "--drafter-path",
            "draft",
            "--output-path",
            "out",
            "--min-free-mb",
            "16000",
            "--calibration-mode",
            "adapter-logit-distillation",
        ]
    ).calibration_mode == "adapter-logit-distillation"
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(helper, "memory_snapshot", return_value=low_free):
            code = helper.main_with_args_for_test(
                [
                    "--target-path",
                    "target",
                    "--drafter-path",
                    "draft",
                    "--output-path",
                    tmp,
                ]
            )
        assert code == 2
        assert (Path(tmp) / "openclaw-calibration-blocked.json").exists()
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(helper, "train", side_effect=ImportError("missing jang_tools")):
            code = helper.main_with_args_for_test(
                [
                    "--target-path",
                    "target",
                    "--drafter-path",
                    "draft",
                    "--output-path",
                    tmp,
                ]
            )
        assert code == 2
        blocked = (Path(tmp) / "openclaw-calibration-blocked.json").read_text()
        assert "calibration failed safely: ImportError: missing jang_tools" in blocked
    print("ok openclaw MTP drafter calibration guards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
