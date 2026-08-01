"""Shared real-engine test model resolution.

Not a test module itself (no test_ prefix) -- a small helper imported by
the tests that need to load a real model (test_gladius_scheduler_cpu.py,
test_gladius_contract_e2e.py, test_gladius_rocm_smoke.py).
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_TEST_MODEL = "Qwen/Qwen3-1.7B"  # already present in the local HF cache


def resolve_test_model(default: str = DEFAULT_TEST_MODEL) -> str:
    """Resolve the model id/path used by real-engine tests.

    `GLADIUS_TEST_MODEL` overrides the default -- needed on offline hosts
    with local model directories instead of HF Hub access (see
    docs/design/gladius_next_steps_h100.md P0-C, e.g.
    `/data/user/yge269/models/Phi-4-mini-instruct`). If the override looks
    like a local filesystem path, it must exist: no network fallback is
    attempted for a missing local path, this fails early with a clear error
    instead of silently trying the Hub.
    """
    override = os.environ.get("GLADIUS_TEST_MODEL")
    if not override:
        return default
    looks_like_path = override.startswith(("/", "./", "../")) or os.path.isabs(override)
    if looks_like_path and not Path(override).exists():
        raise FileNotFoundError(
            f"GLADIUS_TEST_MODEL={override!r} does not exist on this host; "
            "no network fallback is attempted for local model paths."
        )
    return override
