"""Build a real `GladiusScheduler` on CPU for lifecycle tests.

Extracted from `test_gladius_scheduler_cpu.py` so that tests which need to
drive the *production* `schedule()` path -- rather than call a writer method
directly -- can do so without duplicating the config plumbing. The scheduler
built here is the same class the deployed server loads via
`--scheduler-cls`; only the executor is absent.
"""

from __future__ import annotations

from pathlib import Path

from gladius_vllm.scheduler import GladiusScheduler
from tests.gladius._test_model import resolve_test_model
from tests.v1.core.utils import create_requests, create_scheduler

MODEL = resolve_test_model()


def build_cpu_gladius_scheduler(
    policy_dir: Path,
    monkeypatch,
    *,
    engine_id: str = "test-engine",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_requests: int = 8,
) -> GladiusScheduler:
    """A live GladiusScheduler bound to `policy_dir`, with work queued."""
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(policy_dir))
    monkeypatch.setenv("GLADIUS_ENGINE_ID", engine_id)
    # Production forbids a 0ms poll interval; tests inject it directly so a
    # freshly written control file is observed on the very next step.
    monkeypatch.setattr("gladius_vllm.scheduler._resolve_poll_interval_ms", lambda: 0)

    vanilla = create_scheduler(
        model=MODEL,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    scheduler = GladiusScheduler(
        vllm_config=vanilla.vllm_config,
        kv_cache_config=vanilla.kv_cache_config,
        structured_output_manager=vanilla.structured_output_manager,
        block_size=16,
        log_stats=True,
    )
    for request in create_requests(
        num_requests=num_requests, num_tokens=50, max_tokens=8
    ):
        scheduler.add_request(request)
    return scheduler
