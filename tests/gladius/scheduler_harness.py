"""Build a real `GladiusScheduler` on CPU for lifecycle tests.

Extracted from `test_gladius_scheduler_cpu.py` so that tests which need to
drive the *production* `schedule()` path -- rather than call a writer method
directly -- can do so without duplicating the config plumbing. The scheduler
built here is the same class the deployed server loads via
`--scheduler-cls`; only the executor is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gladius_vllm.receipt import (
    IDENTITY_SOURCE_CORROBORATED,
    PhysicalGpuIdentity,
    assemble_server_start_receipt,
)
from gladius_vllm.scheduler import GladiusScheduler
from tests.gladius._test_model import resolve_test_model
from tests.v1.core.utils import create_requests, create_scheduler

MODEL = resolve_test_model()

# The receipt digests the model and tokenizer *trees*, so attestation needs a
# real local directory rather than a Hub repo id.
_HUB_SNAPSHOTS = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots"
)


def local_model_path() -> str:
    """A local model directory, or skip: attestation cannot digest a repo id."""
    override = os.environ.get("GLADIUS_TEST_MODEL_PATH")
    if override:
        return override
    if _HUB_SNAPSHOTS.is_dir():
        for snapshot in sorted(_HUB_SNAPSHOTS.iterdir()):
            if (snapshot / "config.json").is_file():
                return str(snapshot)
    pytest.skip("no local model directory for attestation; set GLADIUS_TEST_MODEL_PATH")


def build_cpu_gladius_scheduler(
    policy_dir: Path,
    monkeypatch,
    *,
    engine_id: str = "test-engine",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_requests: int = 8,
    attest: bool = False,
    nonce: str | None = None,
    gpu_uuid: str = "GPU-11112222-3333-4444-5555-666677778888",
) -> GladiusScheduler:
    """A live GladiusScheduler bound to `policy_dir`, with work queued.

    With `attest=True` the scheduler completes the real two-process
    attestation handshake against this test process, so it carries a genuine
    `server_instance_id` and can be asked to seal. The physical GPU probe is
    the one thing stubbed -- there is no CUDA device in a CPU test -- and
    that fact is established instead by the four-process hardware smoke.
    """
    model = local_model_path() if attest else MODEL
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(policy_dir))
    monkeypatch.setenv("GLADIUS_ENGINE_ID", engine_id)
    # Production forbids a 0ms poll interval; tests inject it directly so a
    # freshly written control file is observed on the very next step.
    monkeypatch.setattr("gladius_vllm.scheduler._resolve_poll_interval_ms", lambda: 0)

    if attest:
        nonce = nonce or "a" * 64
        monkeypatch.setenv("GLADIUS_ATTESTATION_NONCE", nonce)
        monkeypatch.setattr(
            "gladius_vllm.receipt._resolve_physical_gpu",
            lambda: PhysicalGpuIdentity(
                physical_gpu_uuid=gpu_uuid,
                physical_gpu_name="stub device",
                physical_gpu_identity_source=IDENTITY_SOURCE_CORROBORATED,
                mig_uuid=None,
                mig_profile=None,
                mig_parent_gpu_uuid=None,
            ),
        )
    else:
        monkeypatch.delenv("GLADIUS_ATTESTATION_NONCE", raising=False)

    vanilla = create_scheduler(
        model=model,
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
    if attest:
        # The API-side half. In production this is `attest publish` running
        # in the API process; here the test process plays that role, and the
        # PID and start identity it contributes are its own real ones.
        assemble_server_start_receipt(
            policy_dir,
            api_pid=os.getpid(),
            listen_host="127.0.0.1",
            listen_port=8000,
            expected_nonce=nonce,
        )
    for request in create_requests(
        num_requests=num_requests, num_tokens=50, max_tokens=8
    ):
        scheduler.add_request(request)
    return scheduler
