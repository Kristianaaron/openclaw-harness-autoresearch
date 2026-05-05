#!/usr/bin/env python3
"""Static guard checks for the OpenClaw Rapid launcher."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


LAUNCHER_PATH = Path(__file__).parent / "openclaw-rapid-launcher.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("openclaw_rapid_launcher", LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    launcher = load_launcher()
    assert launcher._is_exact_ok_smoke_response("OK")
    assert launcher._is_exact_ok_smoke_response("OK.")
    assert not launcher._is_exact_ok_smoke_response("okay")
    assert not launcher._is_exact_ok_smoke_response("eB1L deenedyto own que")
    assert launcher._looks_repeated("thought thought thought thought thought")
    with patch.dict(
        os.environ,
        {
            "OPENCLAW_RAPID_MIN_FREE_MB": "4096",
            "OPENCLAW_RAPID_MAX_COMPRESSOR_MB": "4096",
            "OPENCLAW_RAPID_MAX_SWAP_MB": "2048",
        },
        clear=False,
    ):
        assert launcher.memory_block_reason(
            "test", {"free_mb": 3000, "compressor_mb": 1, "swap_used_mb": 0, "pressure_free_pct": 0}
        ).startswith("test: free_mb=3000<4096")
        assert launcher.memory_block_reason(
            "test", {"free_mb": 8000, "compressor_mb": 5000, "swap_used_mb": 0, "pressure_free_pct": 0}
        ).startswith("test: compressor_mb=5000>=4096")
        assert launcher.memory_block_reason(
            "test", {"free_mb": 8000, "compressor_mb": 1, "swap_used_mb": 3000, "pressure_free_pct": 0}
        ).startswith("test: swap_mb=3000>=2048")
        assert launcher.memory_block_reason(
            "test", {"free_mb": 8000, "compressor_mb": 1, "swap_used_mb": 0, "pressure_free_pct": 0}
        ) is None
        assert launcher.memory_block_reason(
            "startup", {"free_mb": 3000, "compressor_mb": 1, "swap_used_mb": 0, "pressure_free_pct": 0}
        ) is None
        assert launcher.memory_block_reason(
            "runtime", {"free_mb": 64, "compressor_mb": 1, "swap_used_mb": 0, "pressure_free_pct": 0}
        ) is None
        assert launcher.memory_block_reason(
            "runtime", {"free_mb": 8000, "compressor_mb": 5000, "swap_used_mb": 0, "pressure_free_pct": 0}
        ).startswith("runtime: compressor_mb=5000>=4096")
    with TemporaryDirectory() as tmp:
        model = Path(tmp)
        (model / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
        with patch.dict(os.environ, {"OPENCLAW_RAPID_ENABLE_MTP": "auto"}, clear=False):
            assert launcher.should_enable_mtp(str(model)) == (False, "model has no built-in MTP head files")
        with patch.dict(os.environ, {"OPENCLAW_RAPID_ENABLE_MTP": "1"}, clear=False):
            try:
                launcher.should_enable_mtp(str(model))
            except RuntimeError as error:
                assert "separate models" in str(error)
            else:
                raise AssertionError("explicit MTP should fail for checkpoints without MTP heads")
        (model / "config.json").write_text(
            json.dumps({"model_type": "gemma4", "text_config": {"num_nextn_predict_layers": 1}}),
            encoding="utf-8",
        )
        assert not launcher.model_has_mtp_head(str(model))
        (model / "model-mtp.safetensors").write_text("placeholder", encoding="utf-8")
        assert launcher.model_has_mtp_head(str(model))
        with patch.dict(os.environ, {"OPENCLAW_RAPID_ENABLE_MTP": "auto"}, clear=False):
            enabled, reason = launcher.should_enable_mtp(str(model))
            assert enabled
            assert "model-mtp.safetensors" in reason
    assert "SIGABRT" in launcher.child_exit_summary(-6)
    assert "SIGSEGV" in launcher.child_exit_summary(-11)
    print("ok openclaw Rapid launcher guards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
