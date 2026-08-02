"""P0-A: nonce-bound server-start receipt (required tests 1 and 2).

Pure Python: no GPU, no engine, no live server. The EngineCore contribution
is fabricated with *this* test process's real PID so the reuse-proof process
identity is a genuine `/proc` reading rather than a mock -- that is the one
part of the receipt whose whole purpose is to be un-fakeable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gladius_vllm.digest import (
    DigestError,
    derive_server_instance_id,
    process_start_identity,
    tree_sha256,
)
from gladius_vllm.receipt import (
    ENGINE_CONTRIBUTION_FILENAME,
    RECEIPT_FILENAME,
    ReceiptError,
    ServerInstanceBinding,
    assemble_server_start_receipt,
    parse_engine_contribution,
    parse_server_start_receipt,
    read_server_start_receipt,
    verify_server_start_receipt,
)

NONCE = "b6f4c1d0e9a8b7c6d5e4f30219283746b6f4c1d0e9a8b7c6d5e4f30219283746"
GPU_UUID = "GPU-11112222-3333-4444-5555-666677778888"


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _contribution(**overrides) -> dict:
    payload = {
        "schema_version": "2.0.0",
        "attestation_nonce": NONCE,
        "observed_at": "2026-08-02T09:00:00Z",
        "engine_core_pid": os.getpid(),
        "engine_core_process_start_identity": process_start_identity(os.getpid()),
        "engine_id": "gladius-h100-gpu0",
        "model_id": "/models/Qwen3-8B",
        "cuda_visible_devices": "0",
        "physical_gpu_uuid": GPU_UUID,
        "physical_gpu_name": "NVIDIA H100 80GB HBM3",
        "model_path": "/models/Qwen3-8B",
        "model_tree_sha256": _sha("model"),
        "tokenizer_tree_sha256": _sha("tokenizer"),
        "vllm_version": "0.25.1",
        "vllm_module_path": "/opt/venv/lib/python3.12/site-packages/vllm",
        "vllm_package_tree_sha256": _sha("vllm"),
        "vllm_native_binary_sha256": _sha("native"),
        "gladius_overlay_path": "/opt/gladius/gladius_vllm",
        "gladius_overlay_tree_sha256": _sha("overlay"),
        "tree_hash_algorithm_version": "gladius-tree-sha256-v1",
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


def _publish_contribution(policy_dir: Path, **overrides) -> dict:
    payload = _contribution(**overrides)
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / ENGINE_CONTRIBUTION_FILENAME).write_text(json.dumps(payload))
    return payload


# --- required test 1: two-process assembly -------------------------------


def test_receipt_assembly_is_atomic_directory_synced_and_nonce_bound(
    tmp_path, monkeypatch
):
    _publish_contribution(tmp_path)
    synced_directories: list[int] = []
    real_fsync = os.fsync

    def _recording_fsync(fd: int) -> None:
        try:
            if os.path.isdir(f"/proc/self/fd/{fd}"):
                synced_directories.append(fd)
        except OSError:
            pass
        real_fsync(fd)

    monkeypatch.setattr("gladius_vllm.atomic.os.fsync", _recording_fsync)

    receipt = assemble_server_start_receipt(
        tmp_path,
        api_pid=os.getpid(),
        listen_host="127.0.0.1",
        listen_port=8000,
        expected_nonce=NONCE,
    )

    assert receipt.attestation_nonce == NONCE
    assert receipt.server_instance_id.startswith("srv-")
    # No temporary file survives, and the parent directory entry was fsynced
    # so the rename itself is durable, not just the file contents.
    assert list(tmp_path.glob("*.tmp")) == []
    assert synced_directories
    assert read_server_start_receipt(tmp_path / RECEIPT_FILENAME) == receipt


def test_receipt_binds_both_processes_and_the_nonce():
    identity = process_start_identity(os.getpid())
    first = derive_server_instance_id(
        attestation_nonce=NONCE,
        api_pid=1,
        api_process_start_identity=identity,
        engine_core_pid=2,
        engine_core_process_start_identity=identity,
        engine_id="e",
        model_id="m",
        physical_gpu_uuid=GPU_UUID,
    )
    for changed in (
        {"attestation_nonce": "0" * 64},
        {"api_pid": 3},
        {"engine_core_pid": 4},
        {"physical_gpu_uuid": "GPU-other"},
        {"engine_id": "other"},
    ):
        arguments = {
            "attestation_nonce": NONCE,
            "api_pid": 1,
            "api_process_start_identity": identity,
            "engine_core_pid": 2,
            "engine_core_process_start_identity": identity,
            "engine_id": "e",
            "model_id": "m",
            "physical_gpu_uuid": GPU_UUID,
        }
        arguments.update(changed)
        assert derive_server_instance_id(**arguments) != first


def test_assembly_refuses_a_contribution_from_another_launch(tmp_path):
    _publish_contribution(tmp_path, attestation_nonce="a" * 64)

    with pytest.raises(ReceiptError, match="different attestation nonce"):
        assemble_server_start_receipt(
            tmp_path,
            api_pid=os.getpid(),
            listen_host="127.0.0.1",
            listen_port=8000,
            expected_nonce=NONCE,
        )


def test_assembly_refuses_to_overwrite_another_instances_receipt(tmp_path):
    _publish_contribution(tmp_path)
    first = assemble_server_start_receipt(
        tmp_path,
        api_pid=os.getpid(),
        listen_host="127.0.0.1",
        listen_port=8000,
        expected_nonce=NONCE,
    )

    # Re-attesting the same pair is idempotent...
    again = assemble_server_start_receipt(
        tmp_path,
        api_pid=os.getpid(),
        listen_host="127.0.0.1",
        listen_port=8000,
        expected_nonce=NONCE,
    )
    assert again.server_instance_id == first.server_instance_id

    # ...but a different EngineCore in the same directory is refused outright
    # rather than silently inheriting the running campaign's evidence.
    _publish_contribution(tmp_path, engine_core_pid=os.getpid() + 1)
    with pytest.raises(ReceiptError, match="must use a new policy directory"):
        assemble_server_start_receipt(
            tmp_path,
            api_pid=os.getpid(),
            listen_host="127.0.0.1",
            listen_port=8000,
            expected_nonce=NONCE,
        )
    assert (
        read_server_start_receipt(tmp_path / RECEIPT_FILENAME).server_instance_id
        == first.server_instance_id
    )


def test_assembly_requires_the_engine_core_contribution(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ReceiptError, match="not ready to be attested"):
        assemble_server_start_receipt(
            tmp_path,
            api_pid=os.getpid(),
            listen_host="127.0.0.1",
            listen_port=8000,
            expected_nonce=NONCE,
        )


# --- required test 2: receipt parsing rejects every mismatch -------------


def _published(tmp_path: Path, **overrides):
    _publish_contribution(tmp_path, **overrides)
    return assemble_server_start_receipt(
        tmp_path,
        api_pid=os.getpid(),
        listen_host="127.0.0.1",
        listen_port=8000,
        expected_nonce=NONCE,
    )


def test_verify_rejects_wrong_nonce_gpu_engine_and_endpoint(tmp_path):
    receipt = _published(tmp_path)

    assert (
        verify_server_start_receipt(
            receipt,
            expected_nonce=NONCE,
            expected_engine_id="gladius-h100-gpu0",
            expected_model_id="/models/Qwen3-8B",
            expected_listen_host="127.0.0.1",
            expected_listen_port=8000,
            expected_gpu_uuid=GPU_UUID,
        )
        == []
    )

    errors = verify_server_start_receipt(
        receipt,
        expected_nonce="f" * 64,
        expected_engine_id="other-engine",
        expected_listen_port=8001,
        expected_gpu_uuid="GPU-not-this-one",
    )
    assert len(errors) == 4
    assert any("attestation_nonce" in error for error in errors)
    assert any("engine_id" in error for error in errors)
    assert any("listen_port" in error for error in errors)
    assert any("physical_gpu_uuid" in error for error in errors)


def test_verify_rejects_an_altered_digest(tmp_path):
    receipt = _published(tmp_path)

    errors = verify_server_start_receipt(
        receipt, expected_digests={"model_tree_sha256": _sha("a different model")}
    )

    assert len(errors) == 1
    assert "model_tree_sha256" in errors[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enforce_eager", True),
        ("prefix_caching_enabled", False),
        ("chunked_prefill_enabled", False),
        ("startup_max_model_len", 4096),
        ("startup_max_num_seqs", 16),
        ("startup_max_num_batched_tokens", 4096),
        ("gpu_memory_utilization", 0.9),
    ],
)
def test_formal_verification_rejects_a_startup_configuration_mismatch(
    tmp_path, field, value
):
    receipt = _published(tmp_path, **{field: value})

    errors = verify_server_start_receipt(receipt, require_formal_startup=True)

    assert [error for error in errors if field in error], errors


def test_verify_detects_a_replaced_process(tmp_path):
    receipt = _published(tmp_path)
    payload = json.loads((tmp_path / RECEIPT_FILENAME).read_text())
    # A PID that exists but was started at a different time is exactly the
    # PID-reuse case a bare PID comparison would miss.
    payload["engine_core_process_start_identity"] = "boot-id:1"
    payload["server_instance_id"] = derive_server_instance_id(
        attestation_nonce=payload["attestation_nonce"],
        api_pid=payload["api_pid"],
        api_process_start_identity=payload["api_process_start_identity"],
        engine_core_pid=payload["engine_core_pid"],
        engine_core_process_start_identity="boot-id:1",
        engine_id=payload["engine_id"],
        model_id=payload["model_id"],
        physical_gpu_uuid=payload["physical_gpu_uuid"],
    )

    errors = verify_server_start_receipt(parse_server_start_receipt(payload))

    assert any("was replaced" in error for error in errors)
    assert receipt.engine_core_pid == os.getpid()


def test_verify_reports_a_dead_process(tmp_path):
    _publish_contribution(tmp_path)
    receipt = assemble_server_start_receipt(
        tmp_path,
        api_pid=os.getpid(),
        listen_host="127.0.0.1",
        listen_port=8000,
        expected_nonce=NONCE,
    )
    payload = json.loads((tmp_path / RECEIPT_FILENAME).read_text())
    dead_pid = 4194303  # above the default pid_max, so it cannot be live
    payload["engine_core_pid"] = dead_pid
    payload["server_instance_id"] = derive_server_instance_id(
        attestation_nonce=payload["attestation_nonce"],
        api_pid=payload["api_pid"],
        api_process_start_identity=payload["api_process_start_identity"],
        engine_core_pid=dead_pid,
        engine_core_process_start_identity=(
            payload["engine_core_process_start_identity"]
        ),
        engine_id=payload["engine_id"],
        model_id=payload["model_id"],
        physical_gpu_uuid=payload["physical_gpu_uuid"],
    )

    errors = verify_server_start_receipt(parse_server_start_receipt(payload))

    assert any("no longer running" in error for error in errors)
    assert receipt.server_instance_id != payload["server_instance_id"]


def test_parser_rejects_a_forged_server_instance_id(tmp_path):
    _published(tmp_path)
    payload = json.loads((tmp_path / RECEIPT_FILENAME).read_text())
    payload["server_instance_id"] = "srv-" + "0" * 32

    with pytest.raises(ReceiptError, match="not bound to the receipt"):
        parse_server_start_receipt(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": "1.0.0"}, "execution-evidence schema"),
        ({"model_tree_sha256": "not-a-digest"}, "sha256 hex digest"),
        ({"gpu_memory_utilization": 1.5}, "must be in"),
        ({"prefix_caching_enabled": "yes"}, "must be a boolean"),
        ({"listen_port": 0}, "positive integer"),
        ({"engine_id": ""}, "non-empty string"),
    ],
)
def test_parser_rejects_structurally_invalid_receipts(tmp_path, mutation, message):
    _published(tmp_path)
    payload = json.loads((tmp_path / RECEIPT_FILENAME).read_text())
    payload.update(mutation)

    with pytest.raises(ReceiptError, match=message):
        parse_server_start_receipt(payload)


def test_parser_rejects_unknown_and_missing_receipt_fields(tmp_path):
    _published(tmp_path)
    payload = json.loads((tmp_path / RECEIPT_FILENAME).read_text())

    with pytest.raises(ReceiptError, match="do not match the contract"):
        parse_server_start_receipt({**payload, "surprise": True})
    with pytest.raises(ReceiptError, match="do not match the contract"):
        parse_server_start_receipt(
            {key: value for key, value in payload.items() if key != "listen_port"}
        )


def test_engine_contribution_parser_is_strict():
    payload = _contribution()

    assert parse_engine_contribution(payload) == payload
    with pytest.raises(ReceiptError, match="do not match the contract"):
        parse_engine_contribution({**payload, "api_pid": 1})


# --- instance binding: the EngineCore side adopting the joined receipt ---


def test_binding_adopts_only_a_receipt_for_this_process(tmp_path):
    _published(tmp_path)
    receipt = read_server_start_receipt(tmp_path / RECEIPT_FILENAME)

    matching = ServerInstanceBinding(
        tmp_path, expected_nonce=NONCE, engine_core_pid=os.getpid()
    )
    assert matching.refresh() == receipt.server_instance_id

    other_process = ServerInstanceBinding(
        tmp_path, expected_nonce=NONCE, engine_core_pid=os.getpid() + 1
    )
    assert other_process.refresh() is None

    other_launch = ServerInstanceBinding(
        tmp_path, expected_nonce="c" * 64, engine_core_pid=os.getpid()
    )
    assert other_launch.refresh() is None


def test_binding_is_silent_before_attestation(tmp_path):
    binding = ServerInstanceBinding(
        tmp_path, expected_nonce=NONCE, engine_core_pid=os.getpid()
    )

    # No receipt yet: honest `None` rather than a guessed identity, and never
    # an exception into the scheduling path.
    assert binding.refresh() is None
    (tmp_path / RECEIPT_FILENAME).write_text("{ this is not json")
    assert binding.refresh() is None


# --- digest primitives ---------------------------------------------------


def test_tree_digest_is_content_addressed_and_order_independent(tmp_path):
    for name, body in (("b.py", "second"), ("a.py", "first")):
        (tmp_path / name).write_text(body)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "c.py").write_text("third")
    baseline = tree_sha256(tmp_path)

    # Re-hashing is stable, and a __pycache__ dropping does not change it.
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00")
    assert tree_sha256(tmp_path) == baseline

    (tmp_path / "pkg" / "c.py").write_text("third!")
    assert tree_sha256(tmp_path) != baseline


def test_tree_digest_refuses_an_unenumerable_path(tmp_path):
    with pytest.raises(DigestError):
        tree_sha256(tmp_path / "missing")
    with pytest.raises(DigestError):
        tree_sha256(tmp_path)


def test_process_start_identity_rejects_a_dead_pid():
    assert process_start_identity(os.getpid()) == process_start_identity(os.getpid())
    with pytest.raises(DigestError):
        process_start_identity(4194303)
