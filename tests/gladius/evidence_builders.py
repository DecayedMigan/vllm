"""Builders for a coherent, *valid* execution-evidence 2.0.0 policy directory.

These exist so an adversarial test can start from evidence that genuinely
passes, mutate exactly one property, coherently rebuild every local hash, and
then assert the mutation is still refused. A test that started from
hand-written near-valid JSON would prove only that the parser rejects
garbage, which is not the claim under review.

Nothing here fabricates a `server_instance_id`: it is always derived through
`gladius_vllm.digest.derive_server_instance_id` from this test process's own
real `/proc` identity, so the identity half of the evidence is measured
rather than invented.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from gladius_vllm.digest import derive_server_instance_id, process_start_identity
from gladius_vllm.schema import EXECUTION_EVIDENCE_SCHEMA_VERSION

NONCE = "b6f4c1d0e9a8b7c6d5e4f30219283746b6f4c1d0e9a8b7c6d5e4f30219283746"
GPU_UUID = "GPU-11112222-3333-4444-5555-666677778888"
PCI_BUS_ID = "00000000:17:00.0"
ENGINE_ID = "gladius-h100-gpu0"
MODEL_ID = "/models/Qwen3-8B"
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8000
TREE_HASH_ALGORITHM_VERSION = "gladius-tree-sha256-v1"


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def digest_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instance_id(
    *, api_pid: int | None = None, engine_core_pid: int | None = None
) -> str:
    api_pid = os.getpid() if api_pid is None else api_pid
    engine_core_pid = os.getpid() if engine_core_pid is None else engine_core_pid
    return derive_server_instance_id(
        attestation_nonce=NONCE,
        api_pid=api_pid,
        api_process_start_identity=process_start_identity(api_pid),
        engine_core_pid=engine_core_pid,
        engine_core_process_start_identity=process_start_identity(engine_core_pid),
        engine_id=ENGINE_ID,
        model_id=MODEL_ID,
        physical_gpu_uuid=GPU_UUID,
    )


def receipt_payload(**overrides: Any) -> dict[str, Any]:
    pid = os.getpid()
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "attestation_nonce": NONCE,
        "server_instance_id": instance_id(),
        "created_at": "2026-08-02T09:00:00Z",
        "api_pid": pid,
        "api_process_start_identity": process_start_identity(pid),
        "engine_core_pid": pid,
        "engine_core_process_start_identity": process_start_identity(pid),
        "engine_id": ENGINE_ID,
        "model_id": MODEL_ID,
        "listen_host": LISTEN_HOST,
        "listen_port": LISTEN_PORT,
        "cuda_visible_devices": "0",
        "physical_gpu_uuid": GPU_UUID,
        "physical_gpu_name": "NVIDIA H100 80GB HBM3",
        "model_path": MODEL_ID,
        "model_tree_sha256": sha("model"),
        "tokenizer_tree_sha256": sha("tokenizer"),
        "vllm_version": "0.25.1",
        "vllm_module_path": "/opt/venv/lib/python3.12/site-packages/vllm",
        "vllm_package_tree_sha256": sha("vllm"),
        "vllm_native_binary_sha256": sha("native"),
        "gladius_overlay_path": "/opt/gladius/gladius_vllm",
        "gladius_overlay_tree_sha256": sha("overlay"),
        "tree_hash_algorithm_version": TREE_HASH_ALGORITHM_VERSION,
        "startup_max_model_len": 8192,
        "startup_max_num_seqs": 32,
        "startup_max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.75,
        "prefix_caching_enabled": True,
        "chunked_prefill_enabled": True,
        "enforce_eager": False,
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
    }
    payload.update(overrides)
    return payload


def deployment_manifest_payload(**overrides: Any) -> dict[str, Any]:
    """The complete expectation vLLM plan §3 requires.

    Written from the *receipt's* own values so the baseline manifest agrees
    with the baseline receipt; a test that wants disagreement overrides one
    field explicitly.
    """
    receipt = receipt_payload()
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "attestation_nonce": NONCE,
        "engine_id": ENGINE_ID,
        "model_id": MODEL_ID,
        "model_path": receipt["model_path"],
        "listen_host": LISTEN_HOST,
        "listen_port": LISTEN_PORT,
        "physical_gpu_uuid": GPU_UUID,
        "physical_gpu_identity_source": "cuda-nvml-corroborated",
        "mig_uuid": None,
        "mig_profile": None,
        "mig_parent_gpu_uuid": None,
        "model_tree_sha256": receipt["model_tree_sha256"],
        "tokenizer_tree_sha256": receipt["tokenizer_tree_sha256"],
        "vllm_package_tree_sha256": receipt["vllm_package_tree_sha256"],
        "vllm_native_binary_sha256": receipt["vllm_native_binary_sha256"],
        "gladius_overlay_tree_sha256": receipt["gladius_overlay_tree_sha256"],
        "tree_hash_algorithm_version": TREE_HASH_ALGORITHM_VERSION,
        "vllm_version": receipt["vllm_version"],
        "vllm_module_path": receipt["vllm_module_path"],
        "gladius_overlay_path": receipt["gladius_overlay_path"],
        "startup_max_model_len": 8192,
        "startup_max_num_seqs": 32,
        "startup_max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.75,
        "prefix_caching_enabled": True,
        "chunked_prefill_enabled": True,
        "enforce_eager": False,
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
    }
    payload.update(overrides)
    return payload


def telemetry_record(step: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "server_instance_id": instance_id(),
        "generation_high_watermark": 7,
        "generation": 7,
        "policy_id": "policy-a",
        "decision_id": "policy-a",
        "window_id": None,
        "model_id": MODEL_ID,
        "engine_id": ENGINE_ID,
        "created_at": "2026-08-02T09:00:00Z",
        "expires_at": None,
        "step": step,
        "num_running_reqs": 3,
        "num_waiting_reqs": 1,
        "num_skipped_waiting_reqs": 0,
        "num_scheduled_reqs": 2,
        "num_scheduled_tokens": 512,
        "num_prefill_reqs": 1,
        "num_decode_reqs": 1,
        "kv_cache_usage": 0.25,
        "policy_status": "active",
        "policy_source": "file",
        "requested_admission": {"max_num_seqs": 32, "max_num_batched_tokens": 8192},
        "effective_admission": {"max_num_seqs": 32, "max_num_batched_tokens": 8192},
        "clamped": {"max_num_seqs": False, "max_num_batched_tokens": False},
        "policy_poll_ns": 1000,
        "policy_apply_ns": 500,
        "telemetry_write_ns": 250,
    }
    payload.update(overrides)
    return payload


def application_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "server_instance_id": instance_id(),
        "generation_high_watermark": 7,
        "engine_id": ENGINE_ID,
        "model_id": MODEL_ID,
        "generation": 7,
        "policy_id": "policy-a",
        "decision_id": "policy-a",
        "observed_at": "2026-08-02T09:00:00Z",
        "scheduler_step": 3,
        "state": "active",
        "requested_admission": {"max_num_seqs": 32, "max_num_batched_tokens": 8192},
        "effective_admission": {"max_num_seqs": 32, "max_num_batched_tokens": 8192},
        "clamped": {"max_num_seqs": False, "max_num_batched_tokens": False},
    }
    payload.update(overrides)
    return payload


def seal_payload(policy_dir: Path, *, steps: tuple[int, ...], **overrides: Any) -> dict:
    telemetry = policy_dir / "telemetry.jsonl"
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "engine_id": ENGINE_ID,
        "model_id": MODEL_ID,
        "sealed_at": "2026-08-02T09:05:00Z",
        "server_instance_id": instance_id(),
        "attestation_receipt_sha256": digest_of(
            policy_dir / "server_start_receipt.json"
        ),
        "policy_application_sha256": digest_of(policy_dir / "policy_application.json"),
        "first_scheduler_step": steps[0],
        "final_scheduler_step": steps[-1],
        "record_count": len(steps),
        "generation_high_watermark": 7,
        "files": [
            {
                "name": telemetry.name,
                "size": telemetry.stat().st_size,
                "sha256": digest_of(telemetry),
            }
        ],
    }
    payload.update(overrides)
    return payload


def build_sealed_policy_dir(
    policy_dir: Path, *, steps: tuple[int, ...] = (1, 2, 3)
) -> Path:
    """A coherent, fully valid sealed evidence directory."""
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "telemetry.jsonl").write_text(
        "".join(json.dumps(telemetry_record(step)) + "\n" for step in steps)
    )
    (policy_dir / "server_start_receipt.json").write_text(json.dumps(receipt_payload()))
    (policy_dir / "policy_application.json").write_text(
        json.dumps(application_payload(scheduler_step=steps[-1]))
    )
    (policy_dir / "telemetry_seal.json").write_text(
        json.dumps(seal_payload(policy_dir, steps=steps))
    )
    return policy_dir


def reseal_after_mutation(policy_dir: Path, *, steps: tuple[int, ...]) -> None:
    """Rebuild every hash inside the directory so it is self-consistent again.

    This is the attacker's move the reviewer described: a rewrite is only
    interesting if all the local checksums are made to agree with it.
    """
    (policy_dir / "telemetry_seal.json").write_text(
        json.dumps(seal_payload(policy_dir, steps=steps))
    )
