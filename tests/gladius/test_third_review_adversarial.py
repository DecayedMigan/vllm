"""Third-review adversarial acceptance suite (vLLM plan §2).

These are the reviewer-owned red tests. Each one starts from evidence that
genuinely passes, changes exactly one property, coherently rebuilds every
local checksum, and then requires a *classified* refusal naming the violated
invariant. The point is that a checksum match must never substitute for
semantics.

Error codes are pinned here as literal strings rather than imported from the
implementation. Importing them would let a rename silently redefine the
contract these tests exist to hold, which is the exact failure mode the third
review identified: tests shaped to confirm the implementation instead of to
falsify its claim.

Every test in this file must fail on the rejected revision c311bc0 and pass
unchanged on the candidate. Weakening an assertion to accommodate an
implementation is out of contract.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from gladius_vllm.digest import DigestError
from tests.gladius.evidence_builders import (
    ENGINE_ID,
    GPU_UUID,
    LISTEN_HOST,
    MODEL_ID,
    PCI_BUS_ID,
    application_payload,
    build_sealed_policy_dir,
    deployment_manifest_payload,
    receipt_payload,
    reseal_after_mutation,
    telemetry_record,
)

# --- the pinned refusal vocabulary ---------------------------------------
#
# A classified code is what lets a caller distinguish "this evidence is
# forged" from "this file is missing". Free text cannot be acted on.
SEAL_APPLICATION_SEMANTIC_MISMATCH = "SEAL_APPLICATION_SEMANTIC_MISMATCH"
SEAL_RECEIPT_EXPECTATION_MISMATCH = "SEAL_RECEIPT_EXPECTATION_MISMATCH"
SEAL_SEGMENT_SET_MISMATCH = "SEAL_SEGMENT_SET_MISMATCH"
SEAL_APPLICATION_STEP_UNSEALED = "SEAL_APPLICATION_STEP_UNSEALED"
RECEIPT_EXPECTATION_INCOMPLETE = "RECEIPT_EXPECTATION_INCOMPLETE"
RECEIPT_SOCKET_OWNER_MISMATCH = "RECEIPT_SOCKET_OWNER_MISMATCH"
GPU_IDENTITY_DISAGREEMENT = "GPU_IDENTITY_DISAGREEMENT"
GPU_IDENTITY_AMBIGUOUS = "GPU_IDENTITY_AMBIGUOUS"
TELEMETRY_DIRECTORY_RETIRED = "TELEMETRY_DIRECTORY_RETIRED"


def _codes(errors: list[str]) -> set[str]:
    """The leading classified code of each reported error."""
    return {error.split(":", 1)[0].strip() for error in errors}


def _expectation(tmp_path: Path, **overrides) -> object:
    from gladius_vllm.receipt import DeploymentExpectation

    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(deployment_manifest_payload(**overrides)))
    return DeploymentExpectation.from_file(path)


def _verify(policy_dir: Path, expectation: object) -> list[str]:
    """The one formal seal-verification entry point under review."""
    from gladius_vllm.telemetry import verify_telemetry_seal

    return verify_telemetry_seal(
        policy_dir / "telemetry_seal.json",
        policy_dir,
        expectation=expectation,
    )


# --- 1: a rehashed application action rewrite ----------------------------


def test_seal_rejects_rehashed_application_action_rewrite(tmp_path):
    """Rewriting the acknowledged action and rebuilding its digest must fail.

    The sealed telemetry says the effective admission was 32/8192. An
    attacker who edits the acknowledgement to claim a different action, then
    updates `policy_application_sha256` to match, produces a directory in
    which every hash agrees. Only a semantic cross-check catches it.
    """
    policy_dir = build_sealed_policy_dir(tmp_path / "gpu0")
    expectation = _expectation(tmp_path)
    assert _verify(policy_dir, expectation) == []

    (policy_dir / "policy_application.json").write_text(
        json.dumps(
            application_payload(
                scheduler_step=3,
                requested_admission={
                    "max_num_seqs": 8,
                    "max_num_batched_tokens": 2048,
                },
                effective_admission={
                    "max_num_seqs": 8,
                    "max_num_batched_tokens": 2048,
                },
            )
        )
    )
    reseal_after_mutation(policy_dir, steps=(1, 2, 3))

    errors = _verify(policy_dir, expectation)
    assert SEAL_APPLICATION_SEMANTIC_MISMATCH in _codes(errors), errors


# --- 2: a rehashed receipt startup rewrite -------------------------------


def test_seal_rejects_rehashed_receipt_startup_rewrite(tmp_path):
    """Changing a frozen deployment field must fail even when rehashed."""
    policy_dir = build_sealed_policy_dir(tmp_path / "gpu0")
    expectation = _expectation(tmp_path)
    assert _verify(policy_dir, expectation) == []

    (policy_dir / "server_start_receipt.json").write_text(
        json.dumps(receipt_payload(startup_max_model_len=4096))
    )
    reseal_after_mutation(policy_dir, steps=(1, 2, 3))

    errors = _verify(policy_dir, expectation)
    assert SEAL_RECEIPT_EXPECTATION_MISMATCH in _codes(errors), errors


# --- 3: unlisted, missing, reordered, duplicated segments ----------------


@pytest.mark.parametrize(
    "attack",
    ["unlisted_extra_segment", "omitted_listed_segment", "duplicated_filename"],
)
def test_seal_rejects_unlisted_and_missing_segments(tmp_path, attack):
    """The manifest's file list must equal the segments actually present.

    Listing a subset lets an attacker drop an inconvenient segment from
    certification while leaving it on disk; the reviewed verifier only ever
    walked the manifest, so an unlisted segment was invisible to it.
    """
    # Two certified segments, so "omit one" is expressible. An empty file
    # list would be caught by schema validation before the set comparison.
    policy_dir = build_sealed_policy_dir(
        tmp_path / "gpu0", steps=(3, 4), rotated_steps=(1, 2)
    )
    expectation = _expectation(tmp_path)
    assert _verify(policy_dir, expectation) == []

    manifest_path = policy_dir / "telemetry_seal.json"
    manifest = json.loads(manifest_path.read_text())
    if attack == "unlisted_extra_segment":
        extra = policy_dir / "telemetry.jsonl.1754140000-2"
        extra.write_text(json.dumps(telemetry_record(99)) + "\n")
    elif attack == "omitted_listed_segment":
        manifest["files"] = manifest["files"][:1]
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest["files"] = manifest["files"] + list(manifest["files"])
        manifest_path.write_text(json.dumps(manifest))

    errors = _verify(policy_dir, expectation)
    assert SEAL_SEGMENT_SET_MISMATCH in _codes(errors), errors


# --- 4: the application must be bound to the sealed telemetry ------------


@pytest.mark.parametrize(
    "mutation",
    [
        {"scheduler_step": 999},
        {"generation": 3},
        {"generation_high_watermark": 999},
        {"engine_id": "gladius-h100-gpu3"},
    ],
)
def test_seal_binds_application_to_telemetry_step(tmp_path, mutation):
    """An acknowledgement must describe a step the sealed telemetry contains.

    Each mutation is coherently rehashed, so only re-derivation from the
    telemetry stream itself can reject it.
    """
    policy_dir = build_sealed_policy_dir(tmp_path / "gpu0")
    expectation = _expectation(tmp_path)
    assert _verify(policy_dir, expectation) == []

    payload = application_payload(scheduler_step=3)
    payload.update(mutation)
    (policy_dir / "policy_application.json").write_text(json.dumps(payload))
    reseal_after_mutation(policy_dir, steps=(1, 2, 3))

    errors = _verify(policy_dir, expectation)
    assert _codes(errors) & {
        SEAL_APPLICATION_STEP_UNSEALED,
        SEAL_APPLICATION_SEMANTIC_MISMATCH,
    }, errors


# --- 5: formal verification requires a complete expectation --------------


@pytest.mark.parametrize(
    "absent_field",
    [
        "model_path",
        "cuda_graph_mode",
        "vllm_version",
        "vllm_module_path",
        "startup_max_model_len",
        "gpu_memory_utilization",
        "physical_gpu_identity_source",
        "physical_gpu_uuid",
        "attestation_nonce",
        "listen_port",
        "model_tree_sha256",
    ],
)
def test_formal_verify_requires_complete_deployment_expectation(tmp_path, absent_field):
    """A manifest missing any required field must be refused, not tolerated.

    An expectation that can be partially supplied is an expectation that can
    be partially skipped, and a skipped check was reported as a pass by the
    reviewed revision.
    """
    from gladius_vllm.receipt import DeploymentExpectation, ReceiptError

    payload = deployment_manifest_payload()
    payload.pop(absent_field)
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ReceiptError) as caught:
        DeploymentExpectation.from_file(path)
    assert RECEIPT_EXPECTATION_INCOMPLETE in str(caught.value)


def test_no_formal_api_verifies_a_receipt_without_an_expectation():
    """There must be no overload that skips the deployment expectation.

    `verify_server_start_receipt(receipt)` with every expectation defaulted
    to `None` is precisely the permissive path the plan forbids.
    """
    import gladius_vllm.receipt as receipt_module

    assert not hasattr(receipt_module, "verify_server_start_receipt"), (
        "the optional-expectation verifier must be removed, not merely "
        "discouraged: while it exists a caller can verify nothing and be "
        "told the receipt is valid"
    )


# --- 6: the socket owner must be rechecked, not remembered ---------------


def test_receipt_recheck_detects_socket_owner_replacement(tmp_path):
    """A live recorded PID is not proof that it still owns the endpoint.

    The receipt names this process. A *different* process then binds the
    port. The recorded PID is still alive and its start identity still
    matches, so only an actual re-read of the socket owner can notice that
    traffic would now reach someone else.
    """
    from gladius_vllm.receipt import recheck_receipt_liveness

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((LISTEN_HOST, 0))
    port = listener.getsockname()[1]
    listener.listen(4)
    listener.close()

    # A separate process owns the port; this test process does not.
    holder_source = "\n".join(
        (
            "import socket, sys, time",
            "s = socket.socket()",
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)",
            f"s.bind(({LISTEN_HOST!r}, {port}))",
            "s.listen(4)",
            "sys.stdout.write('up\\n')",
            "sys.stdout.flush()",
            "time.sleep(30)",
        )
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_source],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "up"

        policy_dir = tmp_path / "gpu0"
        policy_dir.mkdir()
        (policy_dir / "server_start_receipt.json").write_text(
            json.dumps(receipt_payload(listen_port=port))
        )

        errors = recheck_receipt_liveness(policy_dir / "server_start_receipt.json")
        assert RECEIPT_SOCKET_OWNER_MISMATCH in _codes(errors), errors
    finally:
        holder.kill()
        holder.wait(timeout=10)


# --- 7: CUDA and NVML must corroborate one physical device ---------------


class _FakeNvmlHandle:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeNvml:
    """Minimal NVML stand-in whose PCI and UUID lookups can disagree."""

    def __init__(self, *, by_pci: str | None, by_uuid: str | None) -> None:
        self._by_pci = by_pci
        self._by_uuid = by_uuid
        self.uuids = {
            "handle-pci": by_pci,
            "handle-uuid": by_uuid,
            "handle-parent": "GPU-aaaabbbb-cccc-dddd-eeee-ffff00001111",
        }

    def nvmlInit(self) -> None:  # noqa: N802 - NVML's own spelling
        return None

    def nvmlShutdown(self) -> None:  # noqa: N802
        return None

    def nvmlDeviceGetHandleByPciBusId(self, value):  # noqa: N802
        if self._by_pci is None:
            raise RuntimeError("no such PCI device")
        return _FakeNvmlHandle("handle-pci")

    def nvmlDeviceGetHandleByUUID(self, value):  # noqa: N802
        if self._by_uuid is None:
            raise RuntimeError("no such UUID")
        return _FakeNvmlHandle("handle-uuid")

    def nvmlDeviceGetUUID(self, handle):  # noqa: N802
        return self.uuids[handle.name]

    def nvmlDeviceGetName(self, handle):  # noqa: N802
        return "NVIDIA H100 80GB HBM3"

    def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(self, handle):  # noqa: N802
        return _FakeNvmlHandle("handle-parent")


def test_gpu_resolution_rejects_cuda_nvml_disagreement():
    """Two physical identities that resolve to different devices must fail."""
    from gladius_vllm.receipt import resolve_physical_gpu_identity

    nvml = _FakeNvml(
        by_pci=GPU_UUID,
        by_uuid="GPU-99998888-7777-6666-5555-444433332222",
    )
    with pytest.raises(DigestError) as caught:
        resolve_physical_gpu_identity(
            nvml=nvml, pci_bus_id=PCI_BUS_ID, cuda_uuid=GPU_UUID
        )
    assert GPU_IDENTITY_DISAGREEMENT in str(caught.value)


def test_gpu_resolution_rejects_uncorroborated_identity():
    """One resolvable identity is not corroboration."""
    from gladius_vllm.receipt import resolve_physical_gpu_identity

    nvml = _FakeNvml(by_pci=None, by_uuid=GPU_UUID)
    with pytest.raises(DigestError) as caught:
        resolve_physical_gpu_identity(
            nvml=nvml, pci_bus_id=PCI_BUS_ID, cuda_uuid=GPU_UUID
        )
    assert GPU_IDENTITY_AMBIGUOUS in str(caught.value)


def test_gpu_resolution_never_manufactures_a_gpu_prefix():
    """A MIG device is `MIG-...`; forcing `GPU-` invents an identity.

    The reviewed code unconditionally prepended `GPU-` to any UUID that did
    not already start with it, which turns a MIG instance UUID into a
    physical-GPU UUID that names a different device.
    """
    from gladius_vllm.receipt import resolve_physical_gpu_identity

    mig_uuid = "MIG-11112222-3333-4444-5555-666677778888"
    nvml = _FakeNvml(by_pci=mig_uuid, by_uuid=mig_uuid)
    identity = resolve_physical_gpu_identity(
        nvml=nvml, pci_bus_id=PCI_BUS_ID, cuda_uuid=mig_uuid
    )
    assert not identity.physical_gpu_uuid.startswith("GPU-MIG-")


# --- 8: the seal must be reachable from the live scheduler ---------------


def test_live_server_seal_request_creates_terminal_artifacts(tmp_path, monkeypatch):
    """The scheduler's own loop must produce the seal, not a direct call.

    `seal_telemetry()` having no caller anywhere in the repository is what
    made the reviewed seal unreachable in production. This drives the real
    `GladiusScheduler.schedule()` path with a seal request present and
    requires the terminal artifacts to appear without the test ever touching
    the writer.
    """
    from gladius_vllm.seal_lifecycle import (
        SEAL_ACK_FILENAME,
        SEAL_REQUEST_FILENAME,
        write_seal_request,
    )
    from tests.gladius.scheduler_harness import build_cpu_gladius_scheduler

    policy_dir = tmp_path / "gpu0"
    policy_dir.mkdir()
    nonce = "a" * 64
    scheduler = build_cpu_gladius_scheduler(
        policy_dir, monkeypatch, attest=True, nonce=nonce
    )

    # The scheduler must have written telemetry before anything can be
    # sealed; drive a few real scheduling steps first. The first step is
    # also what lets the binding adopt the published receipt.
    for _ in range(3):
        scheduler.schedule()
    assert scheduler.server_instance_id is not None

    write_seal_request(
        policy_dir,
        server_instance_id=scheduler.server_instance_id,
        deployment_manifest_sha256="0" * 64,
        expected_final_generation=0,
        attestation_nonce=nonce,
    )
    assert (policy_dir / SEAL_REQUEST_FILENAME).is_file()

    for _ in range(5):
        scheduler.schedule()

    assert (policy_dir / "telemetry_seal.json").is_file(), (
        "the live scheduler observed a seal request and did not seal"
    )
    assert (policy_dir / "RETIRED").is_file()
    ack = json.loads((policy_dir / SEAL_ACK_FILENAME).read_text())
    assert ack["server_instance_id"] == scheduler.server_instance_id
    assert ack["status"] == "sealed"


# --- 9: retirement is atomic against a writer opened beforehand ----------


def test_writer_open_before_seal_cannot_append_after_retirement(tmp_path):
    """A writer that predates the seal must not extend certified evidence.

    The reviewed writer only checked for retirement in its constructor, so
    a handle opened before sealing kept appending to a directory that had
    already been certified as complete.
    """
    from gladius_vllm.telemetry import TelemetryWriter, retire_policy_directory

    policy_dir = tmp_path / "gpu0"
    policy_dir.mkdir()
    telemetry = policy_dir / "telemetry.jsonl"

    early = TelemetryWriter(path=telemetry, engine_id=ENGINE_ID, model_id=MODEL_ID)
    early.append_record(telemetry_record(1))
    retire_policy_directory(policy_dir)

    with pytest.raises(ValueError) as caught:
        early.append_record(telemetry_record(2))
    assert TELEMETRY_DIRECTORY_RETIRED in str(caught.value)
    assert telemetry.read_text().count("\n") == 1


def test_concurrent_writers_cannot_interleave_across_a_seal(tmp_path):
    """Two writers racing a seal must leave exactly one certified stream."""
    from gladius_vllm.telemetry import TelemetryWriter, retire_policy_directory

    policy_dir = tmp_path / "gpu0"
    policy_dir.mkdir()
    telemetry = policy_dir / "telemetry.jsonl"

    writers = [
        TelemetryWriter(path=telemetry, engine_id=ENGINE_ID, model_id=MODEL_ID)
        for _ in range(2)
    ]
    refusals: list[Exception] = []
    barrier = threading.Barrier(3)

    def append(writer, step):
        barrier.wait(timeout=10)
        try:
            writer.append_record(telemetry_record(step))
        except ValueError as error:  # noqa: PERF203 - one per thread
            refusals.append(error)

    threads = [
        threading.Thread(target=append, args=(writer, index + 10))
        for index, writer in enumerate(writers)
    ]
    for thread in threads:
        thread.start()
    retire_policy_directory(policy_dir)
    barrier.wait(timeout=10)
    for thread in threads:
        thread.join(timeout=10)

    for writer in writers:
        with pytest.raises(ValueError):
            writer.append_record(telemetry_record(99))


@pytest.mark.parametrize(
    ("server_digest", "observed_generation"),
    [
        (None, 42),
        ("a" * 64, None),
        ("a" * 64, 43),
    ],
)
def test_seal_request_never_skips_deployment_or_exact_generation_binding(
    server_digest, observed_generation
):
    from gladius_vllm.evidence_codes import (
        SEAL_REQUEST_DEPLOYMENT_CHANGED,
        SEAL_REQUEST_STALE_GENERATION,
    )
    from gladius_vllm.seal_lifecycle import SealRequest, classify_refusal

    request = SealRequest(
        request_id="seal-a",
        server_instance_id="srv-a",
        deployment_manifest_sha256="a" * 64,
        expected_final_generation=42,
        attestation_nonce="nonce-a",
        requested_at="2026-08-03T00:00:00Z",
    )

    refusal = classify_refusal(
        request,
        server_instance_id="srv-a",
        deployment_manifest_sha256=server_digest,
        attestation_nonce="nonce-a",
        observed_generation=observed_generation,
        already_retired=False,
    )

    expected = (
        SEAL_REQUEST_DEPLOYMENT_CHANGED
        if server_digest is None
        else SEAL_REQUEST_STALE_GENERATION
    )
    assert refusal is not None and expected in refusal


def test_seal_request_refuses_a_server_without_an_attestation_nonce():
    from gladius_vllm.evidence_codes import SEAL_REQUEST_FOREIGN_INSTANCE
    from gladius_vllm.seal_lifecycle import SealRequest, classify_refusal

    refusal = classify_refusal(
        SealRequest(
            request_id="seal-a",
            server_instance_id="srv-a",
            deployment_manifest_sha256="a" * 64,
            expected_final_generation=42,
            attestation_nonce="nonce-a",
            requested_at="2026-08-03T00:00:00Z",
        ),
        server_instance_id="srv-a",
        deployment_manifest_sha256="a" * 64,
        attestation_nonce=None,
        observed_generation=42,
        already_retired=False,
    )

    assert refusal is not None and SEAL_REQUEST_FOREIGN_INSTANCE in refusal


def test_seal_request_parser_requires_a_sha256_deployment_digest():
    from gladius_vllm.evidence_codes import SEAL_REQUEST_SCHEMA_INVALID
    from gladius_vllm.seal_lifecycle import SealRequestError, parse_seal_request

    with pytest.raises(SealRequestError) as caught:
        parse_seal_request(
            {
                "schema_version": "2.0.0",
                "request_id": "seal-a",
                "server_instance_id": "srv-a",
                "deployment_manifest_sha256": "not-a-digest",
                "expected_final_generation": 42,
                "attestation_nonce": "nonce-a",
                "requested_at": "2026-08-03T00:00:00Z",
            }
        )

    assert SEAL_REQUEST_SCHEMA_INVALID in str(caught.value)


def test_seal_request_parser_requires_an_exact_final_generation():
    from gladius_vllm.evidence_codes import SEAL_REQUEST_SCHEMA_INVALID
    from gladius_vllm.seal_lifecycle import SealRequestError, parse_seal_request

    with pytest.raises(SealRequestError) as caught:
        parse_seal_request(
            {
                "schema_version": "2.0.0",
                "request_id": "seal-a",
                "server_instance_id": "srv-a",
                "deployment_manifest_sha256": "a" * 64,
                "expected_final_generation": None,
                "attestation_nonce": "nonce-a",
                "requested_at": "2026-08-03T00:00:00Z",
            }
        )

    assert SEAL_REQUEST_SCHEMA_INVALID in str(caught.value)


# --- 10: both repositories refuse the same invalid corpus ----------------


def _smig_root() -> Path:
    value = os.environ.get("GLADIUS_SMIG_ROOT")
    if not value:
        pytest.skip("set GLADIUS_SMIG_ROOT for the paired-corpus contract")
    return Path(value)


def test_vllm_and_smig_reject_identical_invalid_corpus():
    """One corpus, two consumers, identical classified verdicts.

    A private "compatible" fixture in each repository proves only that each
    parser agrees with itself.
    """
    corpus_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "gladius-third-review-corpus.json"
    )
    corpus = json.loads(corpus_path.read_text())
    smig_copy = _smig_root() / "tests" / "fixtures" / "gladius-third-review-corpus.json"
    assert smig_copy.read_bytes() == corpus_path.read_bytes(), (
        "the two repositories are pinned to different corpora"
    )

    from gladius_vllm.corpus import evaluate_corpus_case

    for case in corpus["cases"]:
        verdict = evaluate_corpus_case(case)
        assert verdict.accepted == case["expected_accepted"], case["id"]
        if not case["expected_accepted"]:
            assert verdict.error_code == case["expected_error_code"], case["id"]
