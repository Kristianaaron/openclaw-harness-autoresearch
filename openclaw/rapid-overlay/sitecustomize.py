"""OpenClaw startup hook for the Rapid-MLX backend.

Python imports ``sitecustomize`` automatically when this directory is present
on ``PYTHONPATH``. Keep this file tiny: it installs the Rapid/JANG compatibility
patch and then gets out of Rapid-MLX's way.
"""

from __future__ import annotations

try:
    from openclaw_rapid_jang import install

    install()
except Exception as error:  # pragma: no cover - startup safety net
    import sys

    print(f"[openclaw-rapid] JANG bootstrap failed: {error}", file=sys.stderr, flush=True)
