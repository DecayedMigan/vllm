"""Negative probes for the second review's P0-A through P0-H.

Each test reproduces the exact failure the review demonstrated against
`c9248aa0`, then asserts the fixed behaviour. They are grouped by finding so
a reviewer can map a test back to the paragraph that motivated it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gladius_vllm.digest import DigestError, process_start_identity
from gladius_vllm.receipt import (
    ENGINE_CONTRIBUTION_FILENAME,
    RECEIPT_FILENAME,
    DeploymentExpectation,
    ReceiptError,
    _resolve_nvml_handle,
    assemble_server_start_receipt,
    parse_server_start_receipt,
    verify_receipt_against_deployment,
    verify_server_start_receipt,
)
from gladius_vllm.telemetry import (
    POLICY_APPLICATION_FILENAME,
    RETIRED_MARKER_FILENAME,
    SERVER_START_RECEIPT_FILENAME,
    TelemetrySealError,
    TelemetryWriter,
    parse_telemetry_record_v2,
    parse_telemetry_seal,
    verify_telemetry_seal,
)

FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "gladius-execution-evidence-v2.json"
    ).read_text()
)
RECEIPT = FIXTURE["valid_server_start_receipt"]
APPLICATION = FIXTURE["valid_applications"]["active"]
INSTANCE = RECEIPT["server_instance_id"]


def _record(step: int, **overrides) -> dict:
    record = dict(FIXTURE["valid_telemetry"]["file_backed"])
    record["step"] = step
    record.update(overrides)
    return record


def _write_siblings(policy_dir: Path) -> None:
    (policy_dir / SERVER_START_RECEIPT_FILENAME).write_text(json.dumps(RECEIPT))
    (policy_dir / POLICY_APPLICATION_FILENAME).write_text(json.dumps(APPLICATION))


def _seal(tmp_path: Path, records: list[dict], *, siblings: bool = True) -> bool:
    path = tmp_path / "telemetry.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    if siblings:
        _write_siblings(tmp_path)
    writer = TelemetryWriter(
        path=path, engine_id=RECEIPT["engine_id"], model_id=RECEIPT["model_id"]
    )
    return writer.seal(tmp_path / "telemetry_seal.json")


# --- P0-A: logical CUDA ordinal is not a physical NVML ordinal ------------


class _FakeNvml:
    """NVML with four devices, indexed physically.

    Physical GPU 0 is deliberately *not* the device this process bound, so a
    lookup that passes a logical ordinal straight through resolves the wrong
    card -- which is what the review reproduced.
    """

    def __init__(self):
        self.devices = {
            "00000000:0A:00.0": "GPU-aaaa0000-0000-0000-0000-000000000000",
            "00000000:0B:00.0": "GPU-bbbb1111-1111-1111-1111-111111111111",
            "00000000:0C:00.0": "GPU-cccc2222-2222-2222-2222-222222222222",
            "00000000:0D:00.0": "GPU-dddd3333-3333-3333-3333-333333333333",
        }
        self.by_uuid = {uuid: uuid for uuid in self.devices.values()}

    def nvmlDeviceGetHandleByIndex(self, index):  # noqa: N802 - NVML naming
        return list(self.devices.values())[index]

    def nvmlDeviceGetHandleByPciBusId(self, bus_id):  # noqa: N802
        key = bus_id.decode() if isinstance(bus_id, bytes) else bus_id
        if key not in self.devices:
            raise RuntimeError(f"unknown pci bus id {key}")
        return self.devices[key]

    def nvmlDeviceGetHandleByUUID(self, uuid):  # noqa: N802
        key = uuid.decode() if isinstance(uuid, bytes) else uuid
        if key not in self.by_uuid:
            raise RuntimeError(f"unknown uuid {key}")
        return self.by_uuid[key]


@pytest.mark.parametrize(
    ("bus_id", "expected"),
    [
        ("00000000:0A:00.0", "GPU-aaaa0000-0000-0000-0000-000000000000"),
        ("00000000:0B:00.0", "GPU-bbbb1111-1111-1111-1111-111111111111"),
        ("00000000:0C:00.0", "GPU-cccc2222-2222-2222-2222-222222222222"),
        ("00000000:0D:00.0", "GPU-dddd3333-3333-3333-3333-333333333333"),
    ],
)
def test_the_handle_is_resolved_by_pci_bus_id_not_logical_ordinal(bus_id, expected):
    """Every replica sees logical device 0 under one-process-per-GPU."""
    nvml = _FakeNvml()

    handle = _resolve_nvml_handle(nvml, pci_bus_id=bus_id, cuda_uuid=None)

    assert handle == expected
    # The reviewed code would have returned physical GPU 0 for all four.
    assert nvml.nvmlDeviceGetHandleByIndex(0) == (
        "GPU-aaaa0000-0000-0000-0000-000000000000"
    )


def test_a_uuid_form_mask_still_resolves_the_right_device():
    nvml = _FakeNvml()

    bare = _resolve_nvml_handle(
        nvml, pci_bus_id=None, cuda_uuid="cccc2222-2222-2222-2222-222222222222"
    )
    prefixed = _resolve_nvml_handle(
        nvml, pci_bus_id=None, cuda_uuid="GPU-cccc2222-2222-2222-2222-222222222222"
    )

    assert bare == prefixed == "GPU-cccc2222-2222-2222-2222-222222222222"


def test_a_reordered_mask_does_not_shift_the_resolved_device():
    """`CUDA_VISIBLE_DEVICES=3,1` renumbers CUDA but not NVML."""
    nvml = _FakeNvml()

    # Logical 0 under the mask "3,1" is physical GPU 3.
    handle = _resolve_nvml_handle(nvml, pci_bus_id="00000000:0D:00.0", cuda_uuid=None)

    assert handle == "GPU-dddd3333-3333-3333-3333-333333333333"


def test_an_unresolvable_device_fails_rather_than_guessing():
    nvml = _FakeNvml()

    with pytest.raises(DigestError, match="neither a PCI bus id nor a CUDA device"):
        _resolve_nvml_handle(nvml, pci_bus_id=None, cuda_uuid=None)

    with pytest.raises(DigestError, match="cannot map the bound CUDA device"):
        _resolve_nvml_handle(
            nvml, pci_bus_id="00000000:FF:00.0", cuda_uuid="GPU-not-present"
        )


def test_a_stale_pci_id_falls_through_to_the_uuid(caplog):
    nvml = _FakeNvml()

    handle = _resolve_nvml_handle(
        nvml,
        pci_bus_id="00000000:FF:00.0",
        cuda_uuid="GPU-bbbb1111-1111-1111-1111-111111111111",
    )

    assert handle == "GPU-bbbb1111-1111-1111-1111-111111111111"


# --- P0-B: no unverified API-PID override --------------------------------


def _contribution(**overrides) -> dict:
    payload = {
        key: value
        for key, value in RECEIPT.items()
        if key
        not in {
            "server_instance_id",
            "created_at",
            "api_pid",
            "api_process_start_identity",
            "listen_host",
            "listen_port",
        }
    }
    payload["observed_at"] = "2026-08-02T09:00:00Z"
    payload["engine_core_pid"] = os.getpid()
    payload["engine_core_process_start_identity"] = process_start_identity(os.getpid())
    payload.update(overrides)
    return payload


def test_the_publish_cli_has_no_api_pid_override():
    from gladius_vllm.attest import _build_parser

    args = _build_parser().parse_args(
        [
            "publish",
            "--policy-dir",
            "/tmp",
            "--nonce",
            "n",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )

    assert not hasattr(args, "api_pid")
    # Only a cross-check remains, and it can never *supply* the PID.
    assert args.expect_api_pid is None


def test_assembly_rejects_an_engine_core_that_was_replaced(tmp_path):
    contribution = _contribution(engine_core_process_start_identity="boot-id:1")
    (tmp_path / ENGINE_CONTRIBUTION_FILENAME).write_text(json.dumps(contribution))

    with pytest.raises(ReceiptError, match="was replaced between"):
        assemble_server_start_receipt(
            tmp_path,
            api_pid=os.getpid(),
            listen_host="127.0.0.1",
            listen_port=8000,
            expected_nonce=RECEIPT["attestation_nonce"],
        )


def test_assembly_rejects_a_dead_engine_core(tmp_path):
    contribution = _contribution(engine_core_pid=4194303)
    (tmp_path / ENGINE_CONTRIBUTION_FILENAME).write_text(json.dumps(contribution))

    with pytest.raises(ReceiptError, match="no longer running"):
        assemble_server_start_receipt(
            tmp_path,
            api_pid=os.getpid(),
            listen_host="127.0.0.1",
            listen_port=8000,
            expected_nonce=RECEIPT["attestation_nonce"],
        )


# --- P0-C: expected digests are mandatory --------------------------------


def _deployment(**overrides) -> dict:
    payload = {
        "schema_version": "2.0.0",
        "attestation_nonce": RECEIPT["attestation_nonce"],
        "engine_id": RECEIPT["engine_id"],
        "model_id": RECEIPT["model_id"],
        "listen_host": RECEIPT["listen_host"],
        "listen_port": RECEIPT["listen_port"],
        "physical_gpu_uuid": RECEIPT["physical_gpu_uuid"],
        "model_tree_sha256": RECEIPT["model_tree_sha256"],
        "tokenizer_tree_sha256": RECEIPT["tokenizer_tree_sha256"],
        "vllm_package_tree_sha256": RECEIPT["vllm_package_tree_sha256"],
        "vllm_native_binary_sha256": RECEIPT["vllm_native_binary_sha256"],
        "gladius_overlay_tree_sha256": RECEIPT["gladius_overlay_tree_sha256"],
        "tree_hash_algorithm_version": RECEIPT["tree_hash_algorithm_version"],
    }
    payload.update(overrides)
    return payload


def test_formal_verification_without_expectations_is_a_failure():
    receipt = parse_server_start_receipt(RECEIPT)

    # The reviewed behaviour: no expectations supplied, every digest check
    # silently skipped, and only the startup values compared.
    errors = verify_server_start_receipt(receipt, require_formal_startup=True)

    assert any("skipped check is not a pass" in error for error in errors)


def test_the_deployment_manifest_drives_every_check():
    receipt = parse_server_start_receipt(RECEIPT)
    expectation = DeploymentExpectation.from_dict(_deployment())

    errors = verify_receipt_against_deployment(receipt, expectation)

    # The only remaining complaints are about the fixture's synthetic PIDs
    # not being live processes -- every identity and digest check passed.
    assert all("process" in error for error in errors), errors


@pytest.mark.parametrize(
    "field",
    [
        "model_tree_sha256",
        "tokenizer_tree_sha256",
        "vllm_package_tree_sha256",
        "vllm_native_binary_sha256",
        "gladius_overlay_tree_sha256",
    ],
)
def test_an_altered_deployment_digest_fails_verification(field):
    receipt = parse_server_start_receipt(RECEIPT)
    expectation = DeploymentExpectation.from_dict(_deployment(**{field: "9" * 64}))

    errors = verify_receipt_against_deployment(receipt, expectation)

    assert any(field in error for error in errors)


def test_an_altered_tree_hash_algorithm_version_fails_verification():
    receipt = parse_server_start_receipt(RECEIPT)
    expectation = DeploymentExpectation.from_dict(
        _deployment(tree_hash_algorithm_version="gladius-tree-sha256-v2")
    )

    errors = verify_receipt_against_deployment(receipt, expectation)

    assert any("tree_hash_algorithm_version" in error for error in errors)


def test_the_deployment_manifest_parser_is_strict():
    with pytest.raises(ReceiptError, match="fields mismatch"):
        DeploymentExpectation.from_dict({**_deployment(), "surprise": 1})
    incomplete = _deployment()
    del incomplete["model_tree_sha256"]
    with pytest.raises(ReceiptError, match="fields mismatch"):
        DeploymentExpectation.from_dict(incomplete)


def test_formal_attest_verify_requires_a_deployment_manifest(tmp_path):
    from gladius_vllm.attest import main

    (tmp_path / RECEIPT_FILENAME).write_text(json.dumps(RECEIPT))

    exit_code = main(
        [
            "verify",
            "--policy-dir",
            str(tmp_path),
            "--nonce",
            RECEIPT["attestation_nonce"],
            "--host",
            RECEIPT["listen_host"],
            "--port",
            str(RECEIPT["listen_port"]),
            "--formal",
        ]
    )

    assert exit_code == 1


# --- P0-D: unattested or sibling-less seals ------------------------------


def test_sealing_refuses_a_wholly_unattested_stream(tmp_path):
    """The review's probe: every record null, seal returned true."""
    assert _seal(tmp_path, [_record(1, server_instance_id=None)]) is False
    assert not (tmp_path / "telemetry_seal.json").exists()


def test_sealing_refuses_a_partially_unattested_stream(tmp_path):
    assert _seal(tmp_path, [_record(1, server_instance_id=None), _record(2)]) is False


@pytest.mark.parametrize(
    "missing", [SERVER_START_RECEIPT_FILENAME, POLICY_APPLICATION_FILENAME]
)
def test_sealing_refuses_a_missing_sibling(tmp_path, missing):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(json.dumps(_record(1)) + "\n")
    _write_siblings(tmp_path)
    (tmp_path / missing).unlink()
    writer = TelemetryWriter(
        path=path, engine_id=RECEIPT["engine_id"], model_id=RECEIPT["model_id"]
    )

    assert writer.seal(tmp_path / "telemetry_seal.json") is False


def test_sealing_refuses_a_sibling_from_another_instance(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(json.dumps(_record(1)) + "\n")
    _write_siblings(tmp_path)
    (tmp_path / POLICY_APPLICATION_FILENAME).write_text(
        json.dumps(FIXTURE["invalid_cases"]["receipt_mismatch_application"])
    )
    writer = TelemetryWriter(
        path=path, engine_id=RECEIPT["engine_id"], model_id=RECEIPT["model_id"]
    )

    assert writer.seal(tmp_path / "telemetry_seal.json") is False


# --- P0-E: verification re-derives every manifest field ------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_high_watermark", 999),
        ("record_count", 99),
        ("first_scheduler_step", 77),
        ("final_scheduler_step", 77),
        ("server_instance_id", "srv-somewhere-else"),
        ("engine_id", "engine-other"),
        ("model_id", "/models/other"),
    ],
)
def test_every_seal_manifest_field_is_re_derived(tmp_path, field, value):
    """The review changed the watermark from 7 to 999 and saw no error."""
    assert _seal(tmp_path, [_record(1), _record(2)]) is True
    manifest_path = tmp_path / "telemetry_seal.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    errors = verify_telemetry_seal(manifest_path, tmp_path)

    assert errors, f"mutating {field} was not detected"


def test_a_replaced_sibling_with_an_updated_digest_is_still_rejected(tmp_path):
    """Digest agreement alone is not enough: the content must parse and match."""
    assert _seal(tmp_path, [_record(1)]) is True
    manifest_path = tmp_path / "telemetry_seal.json"
    manifest = json.loads(manifest_path.read_text())

    import hashlib

    foreign = json.dumps(FIXTURE["invalid_cases"]["receipt_mismatch_application"])
    (tmp_path / POLICY_APPLICATION_FILENAME).write_text(foreign)
    manifest["policy_application_sha256"] = hashlib.sha256(foreign.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    errors = verify_telemetry_seal(manifest_path, tmp_path)

    assert any("different server instance" in error for error in errors), errors


def test_the_seal_manifest_parser_rejects_missing_and_unknown_fields(tmp_path):
    assert _seal(tmp_path, [_record(1)]) is True
    manifest = json.loads((tmp_path / "telemetry_seal.json").read_text())

    with pytest.raises(TelemetrySealError, match="fields mismatch"):
        parse_telemetry_seal({**manifest, "surprise": 1})
    incomplete = dict(manifest)
    del incomplete["record_count"]
    with pytest.raises(TelemetrySealError, match="fields mismatch"):
        parse_telemetry_seal(incomplete)


# --- P0-F: the writer enforces the native invariant ----------------------


def _fake_scheduler():
    stats = SimpleNamespace(
        num_running_reqs=0,
        num_waiting_reqs=0,
        num_skipped_waiting_reqs=0,
        kv_cache_usage=0.0,
    )
    return SimpleNamespace(
        requests={},
        running=[],
        waiting=[],
        skipped_waiting=[],
        max_num_running_reqs=8,
        max_num_scheduled_tokens=2048,
        make_stats=lambda: stats,
    )


def _fake_output():
    return SimpleNamespace(
        scheduled_new_reqs=[], num_scheduled_tokens={}, total_num_scheduled_tokens=0
    )


@pytest.mark.parametrize(("generation", "policy_id"), [(7, None), (None, "policy-7")])
def test_the_writer_refuses_to_publish_a_half_native_decision(
    tmp_path, generation, policy_id
):
    from gladius_vllm.policy import PolicyDecision

    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(
        path=path, engine_id=RECEIPT["engine_id"], model_id=RECEIPT["model_id"]
    )
    decision = PolicyDecision(
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        policy_id=policy_id,
        generation=generation,
        status="active",
        source="file",
    )

    writer.record(
        _fake_scheduler(), _fake_output(), decision, server_instance_id=INSTANCE
    )

    # The reviewed writer rewrote this into an all-null "native" record.
    assert path.read_text() == ""
    assert writer.seal(tmp_path / "telemetry_seal.json") is False


# --- P0-G: schema-strict record parsing ----------------------------------


def test_an_unknown_field_is_rejected():
    with pytest.raises(TelemetrySealError, match="fields mismatch"):
        parse_telemetry_record_v2({**_record(1), "surprise": 1})


def test_a_missing_field_is_rejected():
    record = _record(1)
    del record["kv_cache_usage"]
    with pytest.raises(TelemetrySealError, match="fields mismatch"):
        parse_telemetry_record_v2(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_running_reqs", -1, "integer >= 0"),
        ("kv_cache_usage", 1.5, r"\[0, 1\]"),
        ("created_at", "not-a-timestamp", "invalid created_at"),
        ("policy_status", "invented", "not a known status"),
        ("policy_source", "invented", "not a known source"),
        ("window_id", 7, "string or null"),
        ("generation_high_watermark", 1, "must be >="),
    ],
)
def test_out_of_contract_values_are_rejected(field, value, message):
    with pytest.raises(TelemetrySealError, match=message):
        parse_telemetry_record_v2(_record(1, **{field: value}))


def test_a_clamp_flag_that_contradicts_the_admission_is_rejected():
    with pytest.raises(TelemetrySealError, match="clamp flags do not describe"):
        parse_telemetry_record_v2(
            _record(1, clamped={"max_num_seqs": True, "max_num_batched_tokens": False})
        )


def test_a_native_record_claiming_a_file_policy_source_is_rejected():
    with pytest.raises(TelemetrySealError, match="policy_source and the identity"):
        parse_telemetry_record_v2(
            _record(1, generation=None, policy_id=None, decision_id=None)
        )


def test_prefill_plus_decode_must_equal_scheduled():
    with pytest.raises(TelemetrySealError, match="must equal num_scheduled_reqs"):
        parse_telemetry_record_v2(_record(1, num_prefill_reqs=5))


# --- P0-H: a sealed policy directory is immutable ------------------------


def test_a_second_writer_cannot_reopen_a_sealed_directory(tmp_path):
    assert _seal(tmp_path, [_record(1), _record(2)]) is True
    certified = (tmp_path / "telemetry.jsonl").read_bytes()
    assert (tmp_path / RETIRED_MARKER_FILENAME).is_file()

    # A brand new scheduler process pointed at the same policy directory.
    second = TelemetryWriter(
        path=tmp_path / "telemetry.jsonl",
        engine_id=RECEIPT["engine_id"],
        model_id=RECEIPT["model_id"],
    )
    from gladius_vllm.policy import PolicyDecision

    for _ in range(5):
        second.record(
            _fake_scheduler(),
            _fake_output(),
            PolicyDecision(
                max_num_seqs=8,
                max_num_batched_tokens=2048,
                policy_id="policy-99",
                generation=99,
                status="active",
                source="file",
            ),
            server_instance_id=INSTANCE,
        )

    assert (tmp_path / "telemetry.jsonl").read_bytes() == certified
    assert verify_telemetry_seal(tmp_path / "telemetry_seal.json", tmp_path) == []


def test_a_retired_directory_refuses_a_writer_even_without_a_seal(tmp_path):
    from gladius_vllm.telemetry import retire_policy_directory

    retire_policy_directory(tmp_path)

    writer = TelemetryWriter(
        path=tmp_path / "telemetry.jsonl",
        engine_id=RECEIPT["engine_id"],
        model_id=RECEIPT["model_id"],
    )

    assert writer._file is None
    assert not (tmp_path / "telemetry.jsonl").exists()
