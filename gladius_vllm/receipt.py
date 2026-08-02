"""Nonce-bound server-start receipt (execution-evidence schema 2.0.0).

`engine_id` and `model_id` survive a process restart, so on their own they
cannot tell an external campaign *which live serving process* applied an
action. The receipt closes that gap by joining two independently measured
contributions:

* the **EngineCore** process -- the one that actually loaded the model onto a
  GPU and owns `GladiusScheduler` -- writes what only it can observe: its own
  reuse-proof process identity, the physical GPU UUID seen from inside the
  model process, the model/tokenizer/vLLM/overlay digests, and the frozen
  startup execution mode.
* the **API** process -- the one that bound the listen socket -- contributes
  its own reuse-proof identity and the bound host/port.

Neither side can fabricate the other's facts, and `server_instance_id` is a
digest over both plus the launcher's unpredictable nonce, so a restart always
produces a new instance id even when PIDs are recycled.

Publication is a two-step handshake through `GLADIUS_POLICY_DIR`:

    EngineCore  -> server_start_receipt.engine.json   (contribution)
    attestor    -> server_start_receipt.json          (final receipt)

The attestor (`python -m gladius_vllm.attest`) runs after the API socket is
bound; it joins the two contributions but measures neither the model nor
itself.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gladius_vllm.atomic import atomic_write_json
from gladius_vllm.digest import (
    TREE_HASH_ALGORITHM_VERSION,
    DigestError,
    derive_server_instance_id,
    loaded_native_extension_manifest,
    process_start_identity,
    tree_sha256,
)
from gladius_vllm.evidence_codes import (
    GPU_IDENTITY_AMBIGUOUS,
    GPU_IDENTITY_DISAGREEMENT,
    RECEIPT_DIGEST_MISMATCH,
    RECEIPT_EXPECTATION_INCOMPLETE,
    RECEIPT_IDENTITY_MISMATCH,
    RECEIPT_PROCESS_DEAD,
    RECEIPT_PROCESS_REPLACED,
    RECEIPT_SCHEMA_INVALID,
    RECEIPT_SOCKET_OWNER_MISMATCH,
    RECEIPT_STARTUP_MISMATCH,
    classify,
)
from gladius_vllm.netowner import SocketOwnerError, listen_socket_owner_pid
from gladius_vllm.schema import (
    EXECUTION_EVIDENCE_SCHEMA_VERSION,
    format_iso8601,
    parse_iso8601,
    resolve_engine_id,
    resolve_model_id,
)

logger = logging.getLogger(__name__)

RECEIPT_FILENAME = "server_start_receipt.json"
ENGINE_CONTRIBUTION_FILENAME = "server_start_receipt.engine.json"

ATTESTATION_NONCE_ENV = "GLADIUS_ATTESTATION_NONCE"

# Everything the EngineCore process measures about itself and its model.
ENGINE_CONTRIBUTION_FIELDS = frozenset(
    {
        "schema_version",
        "attestation_nonce",
        "observed_at",
        "engine_core_pid",
        "engine_core_process_start_identity",
        "engine_id",
        "model_id",
        "cuda_visible_devices",
        "physical_gpu_uuid",
        "physical_gpu_name",
        "physical_gpu_identity_source",
        "mig_uuid",
        "mig_profile",
        "mig_parent_gpu_uuid",
        "model_path",
        "model_tree_sha256",
        "tokenizer_tree_sha256",
        "vllm_version",
        "vllm_module_path",
        "vllm_package_tree_sha256",
        "vllm_native_binary_sha256",
        "gladius_overlay_path",
        "gladius_overlay_tree_sha256",
        "tree_hash_algorithm_version",
        "startup_max_model_len",
        "startup_max_num_seqs",
        "startup_max_num_batched_tokens",
        "gpu_memory_utilization",
        "prefix_caching_enabled",
        "chunked_prefill_enabled",
        "enforce_eager",
        "cuda_graph_mode",
    }
)

# The published receipt is the contribution plus the API-side facts, with
# `observed_at` replaced by the joint `created_at`.
_API_ONLY_FIELDS = frozenset(
    {
        "server_instance_id",
        "created_at",
        "api_pid",
        "api_process_start_identity",
        "listen_host",
        "listen_port",
    }
)
RECEIPT_FIELDS = (ENGINE_CONTRIBUTION_FIELDS - {"observed_at"}) | _API_ONLY_FIELDS

_TEXT_FIELDS = (
    "attestation_nonce",
    "server_instance_id",
    "engine_core_process_start_identity",
    "api_process_start_identity",
    "engine_id",
    "model_id",
    "physical_gpu_uuid",
    "physical_gpu_name",
    "physical_gpu_identity_source",
    "model_path",
    "vllm_version",
    "vllm_module_path",
    "gladius_overlay_path",
    "tree_hash_algorithm_version",
    "listen_host",
    "cuda_graph_mode",
)
_DIGEST_FIELDS = (
    "model_tree_sha256",
    "tokenizer_tree_sha256",
    "vllm_package_tree_sha256",
    "vllm_native_binary_sha256",
    "gladius_overlay_tree_sha256",
)
_POSITIVE_INT_FIELDS = (
    "engine_core_pid",
    "api_pid",
    "listen_port",
    "startup_max_model_len",
    "startup_max_num_seqs",
    "startup_max_num_batched_tokens",
)
_BOOL_FIELDS = ("prefix_caching_enabled", "chunked_prefill_enabled", "enforce_eager")

_EXPECTATION_MIG_FIELDS = ("mig_uuid", "mig_profile", "mig_parent_gpu_uuid")

IDENTITY_SOURCE_CORROBORATED = "cuda-nvml-corroborated"
IDENTITY_SOURCE_ROCM = "rocm-torch"

# The only source a formal H100 campaign may accept. A development host that
# can publish evidence at all (ROCm, where NVML does not exist) is therefore
# unable to satisfy a formal deployment manifest, which is the point: the
# weaker source can be exercised end to end without ever being mistaken for
# the stronger one.
FORMAL_IDENTITY_SOURCES = frozenset({IDENTITY_SOURCE_CORROBORATED})


# The parent specification freezes the formal Qwen3-8B serving configuration.
# A receipt that does not prove these values cannot certify a formal cell.
FORMAL_STARTUP_EXPECTATIONS: dict[str, object] = {
    "startup_max_model_len": 8192,
    "startup_max_num_seqs": 32,
    "startup_max_num_batched_tokens": 8192,
    "gpu_memory_utilization": 0.75,
    "prefix_caching_enabled": True,
    "chunked_prefill_enabled": True,
    "enforce_eager": False,
}


class ReceiptError(ValueError):
    """A server-start receipt or contribution failed strict validation."""


@dataclass(frozen=True)
class ServerStartReceipt:
    """A strictly parsed `server_start_receipt.json`."""

    schema_version: str
    attestation_nonce: str
    server_instance_id: str
    created_at: str
    api_pid: int
    api_process_start_identity: str
    engine_core_pid: int
    engine_core_process_start_identity: str
    engine_id: str
    model_id: str
    listen_host: str
    listen_port: int
    cuda_visible_devices: str
    physical_gpu_uuid: str
    physical_gpu_name: str
    physical_gpu_identity_source: str
    mig_uuid: str | None
    mig_profile: str | None
    mig_parent_gpu_uuid: str | None
    model_path: str
    model_tree_sha256: str
    tokenizer_tree_sha256: str
    vllm_version: str
    vllm_module_path: str
    vllm_package_tree_sha256: str
    vllm_native_binary_sha256: str
    gladius_overlay_path: str
    gladius_overlay_tree_sha256: str
    tree_hash_algorithm_version: str
    startup_max_model_len: int
    startup_max_num_seqs: int
    startup_max_num_batched_tokens: int
    gpu_memory_utilization: float
    prefix_caching_enabled: bool
    chunked_prefill_enabled: bool
    enforce_eager: bool
    cuda_graph_mode: str


def _require_text(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"{field} must be a non-empty string")
    return value


def _require_digest(payload: dict, field: str) -> str:
    value = _require_text(payload, field)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ReceiptError(f"{field} must be a lowercase sha256 hex digest")
    return value


def _require_positive_int(payload: dict, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReceiptError(f"{field} must be a positive integer")
    return value


def _require_bool(payload: dict, field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise ReceiptError(f"{field} must be a boolean")
    return value


def _require_ratio(payload: dict, field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptError(f"{field} must be a number")
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ReceiptError(f"{field} must be in (0, 1]")
    return value


def _validate_common(payload: dict) -> None:
    if payload.get("schema_version") != EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise ReceiptError(
            "receipt requires execution-evidence schema "
            f"{EXECUTION_EVIDENCE_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("cuda_visible_devices"), str):
        raise ReceiptError("cuda_visible_devices must be a string")
    for field in _DIGEST_FIELDS:
        _require_digest(payload, field)
    for field in _BOOL_FIELDS:
        _require_bool(payload, field)
    _require_ratio(payload, "gpu_memory_utilization")
    # MIG identity is absent on a whole card and present on a slice, so the
    # fields are nullable -- but all three move together: a MIG UUID with no
    # parent names a device whose physical identity is unknown.
    mig_present = {
        field: payload.get(field) is not None for field in _EXPECTATION_MIG_FIELDS
    }
    if len(set(mig_present.values())) != 1:
        raise ReceiptError(
            "mig_uuid, mig_profile, and mig_parent_gpu_uuid must be all "
            f"present or all null; got {mig_present}"
        )
    for field in _EXPECTATION_MIG_FIELDS:
        value = payload.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ReceiptError(f"{field} must be a non-empty string or null")
    if payload.get("physical_gpu_identity_source") not in (
        IDENTITY_SOURCE_CORROBORATED,
        IDENTITY_SOURCE_ROCM,
    ):
        raise ReceiptError(
            "physical_gpu_identity_source must name how the device was "
            f"identified; got {payload.get('physical_gpu_identity_source')!r}"
        )


def parse_engine_contribution(payload: object) -> dict[str, Any]:
    """Strictly validate the EngineCore half of the receipt."""
    if not isinstance(payload, dict) or set(payload) != ENGINE_CONTRIBUTION_FIELDS:
        raise ReceiptError("engine contribution fields do not match the contract")
    _validate_common(payload)
    parse_iso8601(payload["observed_at"])
    for field in _TEXT_FIELDS:
        if field in payload:
            _require_text(payload, field)
    for field in _POSITIVE_INT_FIELDS:
        if field in payload:
            _require_positive_int(payload, field)
    return dict(payload)


def parse_server_start_receipt(payload: object) -> ServerStartReceipt:
    """Strictly validate a published `server_start_receipt.json`."""
    if not isinstance(payload, dict) or set(payload) != RECEIPT_FIELDS:
        raise ReceiptError("server start receipt fields do not match the contract")
    _validate_common(payload)
    parse_iso8601(payload["created_at"])
    for field in _TEXT_FIELDS:
        _require_text(payload, field)
    for field in _POSITIVE_INT_FIELDS:
        _require_positive_int(payload, field)
    if payload["listen_port"] > 65535:
        raise ReceiptError("listen_port must be a valid TCP port")

    expected_instance_id = derive_server_instance_id(
        attestation_nonce=payload["attestation_nonce"],
        api_pid=payload["api_pid"],
        api_process_start_identity=payload["api_process_start_identity"],
        engine_core_pid=payload["engine_core_pid"],
        engine_core_process_start_identity=(
            payload["engine_core_process_start_identity"]
        ),
        engine_id=payload["engine_id"],
        model_id=payload["model_id"],
        physical_gpu_uuid=payload["physical_gpu_uuid"],
    )
    if payload["server_instance_id"] != expected_instance_id:
        raise ReceiptError(
            "server_instance_id is not bound to the receipt's own identity fields"
        )
    return ServerStartReceipt(**payload)


def read_server_start_receipt(path: Path) -> ServerStartReceipt:
    return parse_server_start_receipt(json.loads(Path(path).read_text()))


# Every field a formal verification consumes. There is no partial form: an
# expectation that can be omitted is a check that can be skipped, and the
# reviewed revision reported skipped checks as passes.
_EXPECTATION_IDENTITY_FIELDS = (
    "attestation_nonce",
    "engine_id",
    "model_id",
    "model_path",
    "listen_host",
    "physical_gpu_uuid",
    "physical_gpu_identity_source",
    "tree_hash_algorithm_version",
    "vllm_version",
    "vllm_module_path",
    "gladius_overlay_path",
    "cuda_graph_mode",
)
_EXPECTATION_STARTUP_FIELDS = (
    "startup_max_model_len",
    "startup_max_num_seqs",
    "startup_max_num_batched_tokens",
    "gpu_memory_utilization",
    "prefix_caching_enabled",
    "chunked_prefill_enabled",
    "enforce_eager",
)

DEPLOYMENT_MANIFEST_FIELDS = frozenset(
    {"schema_version", "listen_port"}
    | set(_EXPECTATION_IDENTITY_FIELDS)
    | set(_EXPECTATION_MIG_FIELDS)
    | set(_EXPECTATION_STARTUP_FIELDS)
    | set(_DIGEST_FIELDS)
)


@dataclass(frozen=True)
class DeploymentExpectation:
    """What the campaign says one lane's server must be, in full.

    Formal verification consumes exactly this and nothing less. Every field
    below is compared against the receipt, so adding one here without
    comparing it is a bug the tests catch rather than a silent no-op.
    """

    attestation_nonce: str
    engine_id: str
    model_id: str
    model_path: str
    listen_host: str
    listen_port: int
    physical_gpu_uuid: str
    physical_gpu_identity_source: str
    mig_uuid: str | None
    mig_profile: str | None
    mig_parent_gpu_uuid: str | None
    model_tree_sha256: str
    tokenizer_tree_sha256: str
    vllm_package_tree_sha256: str
    vllm_native_binary_sha256: str
    gladius_overlay_tree_sha256: str
    tree_hash_algorithm_version: str
    vllm_version: str
    vllm_module_path: str
    gladius_overlay_path: str
    startup_max_model_len: int
    startup_max_num_seqs: int
    startup_max_num_batched_tokens: int
    gpu_memory_utilization: float
    prefix_caching_enabled: bool
    chunked_prefill_enabled: bool
    enforce_eager: bool
    cuda_graph_mode: str

    @property
    def digests(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _DIGEST_FIELDS}

    @classmethod
    def from_dict(cls, payload: object) -> DeploymentExpectation:
        if not isinstance(payload, dict):
            raise ReceiptError(
                classify(
                    RECEIPT_EXPECTATION_INCOMPLETE,
                    "deployment manifest must be a JSON object",
                )
            )
        fields = set(payload)
        if fields != set(DEPLOYMENT_MANIFEST_FIELDS):
            missing = sorted(DEPLOYMENT_MANIFEST_FIELDS - fields)
            unknown = sorted(fields - DEPLOYMENT_MANIFEST_FIELDS)
            raise ReceiptError(
                classify(
                    RECEIPT_EXPECTATION_INCOMPLETE,
                    f"deployment manifest fields mismatch: missing={missing}, "
                    f"unknown={unknown}",
                )
            )
        if payload["schema_version"] != EXECUTION_EVIDENCE_SCHEMA_VERSION:
            raise ReceiptError(
                classify(
                    RECEIPT_EXPECTATION_INCOMPLETE,
                    "deployment manifest requires execution-evidence schema "
                    f"{EXECUTION_EVIDENCE_SCHEMA_VERSION}",
                )
            )
        try:
            for field in _DIGEST_FIELDS:
                _require_digest(payload, field)
            _require_positive_int(payload, "listen_port")
            for field in _EXPECTATION_IDENTITY_FIELDS:
                _require_text(payload, field)
            for field in (
                "startup_max_model_len",
                "startup_max_num_seqs",
                "startup_max_num_batched_tokens",
            ):
                _require_positive_int(payload, field)
            _require_ratio(payload, "gpu_memory_utilization")
            for field in _BOOL_FIELDS:
                _require_bool(payload, field)
            for field in _EXPECTATION_MIG_FIELDS:
                value = payload[field]
                if value is not None and (not isinstance(value, str) or not value):
                    raise ReceiptError(f"{field} must be a non-empty string or null")
        except ReceiptError as error:
            raise ReceiptError(
                classify(RECEIPT_EXPECTATION_INCOMPLETE, str(error))
            ) from error
        return cls(
            **{key: value for key, value in payload.items() if key != "schema_version"}
        )

    @classmethod
    def from_file(cls, path: Path) -> DeploymentExpectation:
        return cls.from_dict(json.loads(Path(path).read_text()))


def verify_receipt_against_deployment(
    receipt: ServerStartReceipt,
    expectation: DeploymentExpectation,
    *,
    recheck_socket_owner: bool = True,
    recheck_live_processes: bool = True,
) -> list[str]:
    """Every way `receipt` disagrees with the immutable deployment manifest.

    Returns a list rather than raising on the first problem, so one pass
    gives an operator the complete diagnosis of a mis-launched server. There
    is deliberately no variant of this function that accepts a partial
    expectation.
    """
    errors: list[str] = []

    def check(code: str, name: str, expected: object, actual: object) -> None:
        if expected != actual:
            errors.append(
                classify(code, f"{name}: expected {expected!r}, receipt has {actual!r}")
            )

    for field in _EXPECTATION_IDENTITY_FIELDS + _EXPECTATION_MIG_FIELDS:
        check(
            RECEIPT_IDENTITY_MISMATCH,
            field,
            getattr(expectation, field),
            getattr(receipt, field),
        )
    check(
        RECEIPT_IDENTITY_MISMATCH,
        "listen_port",
        expectation.listen_port,
        receipt.listen_port,
    )
    for field in _DIGEST_FIELDS:
        check(
            RECEIPT_DIGEST_MISMATCH,
            field,
            getattr(expectation, field),
            getattr(receipt, field),
        )
    for field in _EXPECTATION_STARTUP_FIELDS:
        check(
            RECEIPT_STARTUP_MISMATCH,
            field,
            getattr(expectation, field),
            getattr(receipt, field),
        )

    # The frozen parent specification, checked independently of the manifest
    # so a manifest that itself drifts from the specification is caught.
    for field, expected_value in FORMAL_STARTUP_EXPECTATIONS.items():
        check(
            RECEIPT_STARTUP_MISMATCH,
            f"frozen {field}",
            expected_value,
            getattr(receipt, field),
        )

    if receipt.physical_gpu_identity_source not in FORMAL_IDENTITY_SOURCES:
        errors.append(
            classify(
                RECEIPT_IDENTITY_MISMATCH,
                f"physical GPU identity source "
                f"{receipt.physical_gpu_identity_source!r} is not corroborated "
                f"across CUDA and NVML; formal evidence requires one of "
                f"{sorted(FORMAL_IDENTITY_SOURCES)}",
            )
        )

    # Both live checks are on by default -- a pre-traffic verification must
    # make them -- but off for archival verification: a sealed attempt is
    # verified after the server has exited, often on another machine, so
    # requiring its processes to still be running would make every genuine
    # seal unverifiable.
    if recheck_live_processes or recheck_socket_owner:
        errors.extend(
            _verify_live_identity(
                receipt,
                check_processes=recheck_live_processes,
                check_socket=recheck_socket_owner,
            )
        )
    return errors


def _verify_live_identity(
    receipt: ServerStartReceipt, *, check_processes: bool = True, check_socket: bool
) -> list[str]:
    """Re-read what the receipt asserts about the world as it is now.

    A receipt is a statement about the past. Everything here is measured at
    call time, immediately before a campaign would start sending traffic.

    Both checks are skippable *only* for archival verification, where the
    server has long exited by design.
    """
    errors: list[str] = []
    processes = (
        (receipt.api_pid, receipt.api_process_start_identity, "api"),
        (
            receipt.engine_core_pid,
            receipt.engine_core_process_start_identity,
            "engine_core",
        ),
    )
    for pid, identity, label in processes if check_processes else ():
        try:
            live_identity = process_start_identity(pid)
        except DigestError:
            errors.append(
                classify(
                    RECEIPT_PROCESS_DEAD, f"{label} process {pid} is no longer running"
                )
            )
            continue
        if live_identity != identity:
            errors.append(
                classify(
                    RECEIPT_PROCESS_REPLACED,
                    f"{label} process {pid} was replaced since the receipt was written",
                )
            )

    if not check_socket:
        return errors

    # The decisive check the reviewed code never made: the recorded API PID
    # can be alive, with its original start identity, while a different
    # process holds the port the campaign will actually call.
    try:
        owner = listen_socket_owner_pid(receipt.listen_host, receipt.listen_port)
    except SocketOwnerError as error:
        errors.append(classify(RECEIPT_SOCKET_OWNER_MISMATCH, str(error)))
        return errors
    if owner != receipt.api_pid:
        errors.append(
            classify(
                RECEIPT_SOCKET_OWNER_MISMATCH,
                f"{receipt.listen_host}:{receipt.listen_port} is owned by PID "
                f"{owner}, but the receipt names API PID {receipt.api_pid}",
            )
        )
    return errors


def recheck_receipt_liveness(receipt_path: Path) -> list[str]:
    """Re-measure a published receipt's live facts, with no expectation.

    Used immediately before traffic by a caller that has already verified
    the receipt against its deployment manifest and only needs to know
    whether the world still matches it.
    """
    try:
        receipt = read_server_start_receipt(receipt_path)
    except (OSError, ValueError) as error:
        return [classify(RECEIPT_SCHEMA_INVALID, str(error))]
    return _verify_live_identity(receipt, check_socket=True)


@dataclass(frozen=True)
class PhysicalGpuIdentity:
    """One device, corroborated across every namespace that can name it."""

    physical_gpu_uuid: str
    physical_gpu_name: str
    physical_gpu_identity_source: str
    mig_uuid: str | None
    mig_profile: str | None
    mig_parent_gpu_uuid: str | None


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _nvml_uuid_candidate(uuid: str) -> str:
    """NVML's lookup form for a CUDA-reported UUID.

    Never manufactures a `GPU-` prefix onto something already namespaced.
    The reviewed code prepended it unconditionally, which turns a MIG
    instance UUID into a physical-GPU UUID naming a *different* device.
    """
    if uuid.startswith(("GPU-", "MIG-")):
        return uuid
    return f"GPU-{uuid}"


def resolve_physical_gpu_identity(
    *, nvml: Any, pci_bus_id: str | None, cuda_uuid: str | None
) -> PhysicalGpuIdentity:
    """Resolve the bound device through PCI *and* UUID, and require agreement.

    CUDA device indices are logical -- renumbered inside
    `CUDA_VISIBLE_DEVICES` -- while NVML indices are physical machine
    ordinals the variable does not remap. Under one-process-per-GPU every
    replica sees logical device 0, so an ordinal lookup reports physical GPU
    0 for all four replicas.

    Resolving by one physical identity fixes that but proves nothing about
    itself: a single lookup that silently returns the wrong handle is
    indistinguishable from a correct one. Two independent namespaces must
    name the same device, or this refuses.
    """
    if not pci_bus_id or not cuda_uuid:
        raise DigestError(
            classify(
                GPU_IDENTITY_AMBIGUOUS,
                "physical GPU identity needs both a PCI bus id and a CUDA "
                f"device UUID to corroborate (pci_bus_id={pci_bus_id!r}, "
                f"cuda_uuid={cuda_uuid!r}); one unverified lookup is not "
                "corroboration and an ordinal is not a physical identity",
            )
        )

    try:
        by_pci = nvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id.encode())
    except Exception as error:  # noqa: BLE001 - reported as ambiguity
        raise DigestError(
            classify(
                GPU_IDENTITY_AMBIGUOUS,
                f"NVML could not resolve PCI bus id {pci_bus_id!r}: {error}",
            )
        ) from error

    candidate = _nvml_uuid_candidate(cuda_uuid)
    try:
        by_uuid = nvml.nvmlDeviceGetHandleByUUID(candidate.encode())
    except Exception as error:  # noqa: BLE001 - reported as ambiguity
        raise DigestError(
            classify(
                GPU_IDENTITY_AMBIGUOUS,
                f"NVML could not resolve device UUID {candidate!r}: {error}",
            )
        ) from error

    uuid_from_pci = _text(nvml.nvmlDeviceGetUUID(by_pci))
    uuid_from_uuid = _text(nvml.nvmlDeviceGetUUID(by_uuid))
    if uuid_from_pci != uuid_from_uuid:
        raise DigestError(
            classify(
                GPU_IDENTITY_DISAGREEMENT,
                f"PCI bus id {pci_bus_id!r} resolves to {uuid_from_pci!r} but "
                f"CUDA UUID {candidate!r} resolves to {uuid_from_uuid!r}; the "
                "two namespaces do not name the same physical device",
            )
        )

    authoritative = uuid_from_pci
    name = _text(nvml.nvmlDeviceGetName(by_pci))
    mig_uuid: str | None = None
    mig_profile: str | None = None
    mig_parent: str | None = None
    if authoritative.startswith("MIG-"):
        mig_uuid = authoritative
        mig_profile = name
        parent_getter = getattr(
            nvml, "nvmlDeviceGetDeviceHandleFromMigDeviceHandle", None
        )
        if not callable(parent_getter):
            raise DigestError(
                classify(
                    GPU_IDENTITY_AMBIGUOUS,
                    f"{authoritative!r} is a MIG instance but this NVML build "
                    "cannot resolve its parent GPU; a MIG identity that cannot "
                    "name its parent is not a physical identity",
                )
            )
        try:
            mig_parent = _text(nvml.nvmlDeviceGetUUID(parent_getter(by_pci)))
        except Exception as error:  # noqa: BLE001 - reported as ambiguity
            raise DigestError(
                classify(
                    GPU_IDENTITY_AMBIGUOUS,
                    f"cannot resolve the parent GPU of {authoritative!r}: {error}",
                )
            ) from error

    return PhysicalGpuIdentity(
        physical_gpu_uuid=authoritative,
        physical_gpu_name=name,
        physical_gpu_identity_source=IDENTITY_SOURCE_CORROBORATED,
        mig_uuid=mig_uuid,
        mig_profile=mig_profile,
        mig_parent_gpu_uuid=mig_parent,
    )


def _probe_cuda_device() -> tuple[str | None, str | None, str]:
    """PCI bus id, CUDA UUID, and device name as CUDA itself reports them."""
    pci_bus_id: str | None = None
    cuda_uuid: str | None = None
    name = "unknown"
    try:
        import torch

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            name = _text(getattr(properties, "name", "unknown"))
            raw_uuid = getattr(properties, "uuid", None)
            if raw_uuid is not None:
                cuda_uuid = _text(raw_uuid)
            # Not exposed on every build, so fall back to the CUDA runtime's
            # own formatted string when it is available.
            getter = getattr(torch.cuda, "get_device_pci_bus_id", None)
            if callable(getter):
                pci_bus_id = _text(getter(device_index))
            else:
                domain = getattr(properties, "pci_domain_id", None)
                bus = getattr(properties, "pci_bus_id", None)
                device = getattr(properties, "pci_device_id", None)
                if None not in (domain, bus, device):
                    pci_bus_id = f"{domain:08X}:{bus:02X}:{device:02X}.0"
    except Exception:  # noqa: BLE001 - a torch probe must not break startup
        logger.debug("GLADIUS: torch device probe unavailable", exc_info=True)
    return pci_bus_id, cuda_uuid, name


def _resolve_physical_gpu() -> PhysicalGpuIdentity:
    """Measure the bound device from inside the process that owns it.

    Deliberately not inferred from `CUDA_VISIBLE_DEVICES`: that records what
    the launcher *asked* for, while the campaign needs to know where the
    model actually landed.
    """
    pci_bus_id, cuda_uuid, cuda_name = _probe_cuda_device()

    try:
        from vllm.third_party import pynvml
    except Exception:  # noqa: BLE001 - NVML is absent on ROCm/CPU hosts
        pynvml = None

    if pynvml is not None:
        pynvml.nvmlInit()
        try:
            return resolve_physical_gpu_identity(
                nvml=pynvml, pci_bus_id=pci_bus_id, cuda_uuid=cuda_uuid
            )
        finally:
            pynvml.nvmlShutdown()

    # ROCm development hosts have no NVML at all, so corroboration is
    # impossible rather than merely skipped. The identity is still measured
    # from inside the model process, but it is labelled honestly and
    # `FORMAL_IDENTITY_SOURCES` excludes it, so it can never satisfy a formal
    # deployment manifest.
    if cuda_uuid:
        return PhysicalGpuIdentity(
            physical_gpu_uuid=cuda_uuid,
            physical_gpu_name=cuda_name,
            physical_gpu_identity_source=IDENTITY_SOURCE_ROCM,
            mig_uuid=None,
            mig_profile=None,
            mig_parent_gpu_uuid=None,
        )
    raise DigestError(
        classify(
            GPU_IDENTITY_AMBIGUOUS,
            "no physical GPU identity is observable: NVML is unavailable and "
            "the CUDA runtime reported no device UUID",
        )
    )


def collect_engine_contribution(
    vllm_config: Any,
    *,
    startup_max_num_seqs: int,
    startup_max_num_batched_tokens: int,
    attestation_nonce: str,
    gpu_probe: Any = None,
) -> dict[str, Any]:
    """Measure every EngineCore-side receipt fact. Raises on any gap."""
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    compilation_config = getattr(vllm_config, "compilation_config", None)
    scheduler_config = vllm_config.scheduler_config

    import vllm

    if not vllm.__file__:
        raise DigestError("the imported vllm package has no file location to digest")
    vllm_module_path = str(Path(vllm.__file__).resolve().parent)
    overlay_path = str(Path(__file__).resolve().parent)

    model_path = Path(model_config.model)
    tokenizer_path = Path(
        getattr(model_config, "tokenizer", None) or model_config.model
    )
    if not model_path.exists() or not tokenizer_path.exists():
        raise DigestError(
            "model and tokenizer must be local paths so their trees can be "
            f"digested; got model={model_config.model!r} "
            f"tokenizer={getattr(model_config, 'tokenizer', None)!r}"
        )

    probe = gpu_probe or _resolve_physical_gpu
    gpu = probe()

    return {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "attestation_nonce": attestation_nonce,
        "observed_at": format_iso8601(),
        "engine_core_pid": os.getpid(),
        "engine_core_process_start_identity": process_start_identity(os.getpid()),
        "engine_id": resolve_engine_id(vllm_config),
        "model_id": resolve_model_id(vllm_config),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu_uuid": gpu.physical_gpu_uuid,
        "physical_gpu_name": gpu.physical_gpu_name,
        "physical_gpu_identity_source": gpu.physical_gpu_identity_source,
        "mig_uuid": gpu.mig_uuid,
        "mig_profile": gpu.mig_profile,
        "mig_parent_gpu_uuid": gpu.mig_parent_gpu_uuid,
        "model_path": str(model_path.resolve()),
        "model_tree_sha256": tree_sha256(model_path),
        "tokenizer_tree_sha256": tree_sha256(tokenizer_path),
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "vllm_module_path": vllm_module_path,
        "vllm_package_tree_sha256": tree_sha256(Path(vllm_module_path)),
        "vllm_native_binary_sha256": loaded_native_extension_manifest(
            Path(vllm_module_path)
        ),
        "gladius_overlay_path": overlay_path,
        "gladius_overlay_tree_sha256": tree_sha256(Path(overlay_path)),
        "tree_hash_algorithm_version": TREE_HASH_ALGORITHM_VERSION,
        "startup_max_model_len": int(model_config.max_model_len),
        "startup_max_num_seqs": int(startup_max_num_seqs),
        "startup_max_num_batched_tokens": int(startup_max_num_batched_tokens),
        "gpu_memory_utilization": float(cache_config.gpu_memory_utilization),
        "prefix_caching_enabled": bool(cache_config.enable_prefix_caching),
        "chunked_prefill_enabled": bool(
            getattr(scheduler_config, "enable_chunked_prefill", False)
        ),
        "enforce_eager": bool(getattr(model_config, "enforce_eager", False)),
        "cuda_graph_mode": str(
            getattr(compilation_config, "cudagraph_mode", None) or "unknown"
        ),
    }


def publish_engine_contribution(policy_dir: Path, contribution: dict[str, Any]) -> Path:
    """Atomically publish the EngineCore half. Raises on validation failure."""
    parse_engine_contribution(contribution)
    path = Path(policy_dir) / ENGINE_CONTRIBUTION_FILENAME
    atomic_write_json(path, contribution)
    return path


def assemble_server_start_receipt(
    policy_dir: Path,
    *,
    api_pid: int,
    listen_host: str,
    listen_port: int,
    expected_nonce: str,
    now: str | None = None,
) -> ServerStartReceipt:
    """Join the EngineCore contribution with this API process's identity.

    Refuses to overwrite a receipt belonging to a *different* server
    instance: one policy directory describes exactly one serving process
    pair, and silently replacing that receipt would let a restarted server
    inherit a still-running campaign's evidence directory.
    """
    policy_dir = Path(policy_dir)
    contribution_path = policy_dir / ENGINE_CONTRIBUTION_FILENAME
    if not contribution_path.is_file():
        raise ReceiptError(
            f"EngineCore has not published {ENGINE_CONTRIBUTION_FILENAME}; "
            "the model process is not ready to be attested"
        )
    contribution = parse_engine_contribution(json.loads(contribution_path.read_text()))
    if contribution["attestation_nonce"] != expected_nonce:
        raise ReceiptError(
            "EngineCore contribution carries a different attestation nonce; "
            "it belongs to another launch"
        )

    # The EngineCore's own identity is re-read here, immediately before the
    # receipt is minted. A process that died or was replaced between
    # publishing its contribution and being attested must not be certified.
    engine_core_pid = int(contribution["engine_core_pid"])
    try:
        live_engine_identity = process_start_identity(engine_core_pid)
    except DigestError as error:
        raise ReceiptError(
            f"EngineCore process {engine_core_pid} is no longer running: {error}"
        ) from error
    if live_engine_identity != contribution["engine_core_process_start_identity"]:
        raise ReceiptError(
            f"EngineCore process {engine_core_pid} was replaced between "
            "publishing its contribution and attestation"
        )

    api_start_identity = process_start_identity(int(api_pid))
    payload = {
        key: value for key, value in contribution.items() if key != "observed_at"
    }
    payload.update(
        {
            "created_at": now or format_iso8601(),
            "api_pid": int(api_pid),
            "api_process_start_identity": api_start_identity,
            "listen_host": listen_host,
            "listen_port": int(listen_port),
        }
    )
    payload["server_instance_id"] = derive_server_instance_id(
        attestation_nonce=payload["attestation_nonce"],
        api_pid=payload["api_pid"],
        api_process_start_identity=payload["api_process_start_identity"],
        engine_core_pid=payload["engine_core_pid"],
        engine_core_process_start_identity=(
            payload["engine_core_process_start_identity"]
        ),
        engine_id=payload["engine_id"],
        model_id=payload["model_id"],
        physical_gpu_uuid=payload["physical_gpu_uuid"],
    )
    receipt = parse_server_start_receipt(payload)
    # Re-read the API identity too: if it changed while the receipt was being
    # assembled, the socket the campaign will call is no longer owned by the
    # process this receipt names.
    if process_start_identity(int(api_pid)) != api_start_identity:
        raise ReceiptError(
            f"API process {api_pid} was replaced during receipt assembly"
        )

    receipt_path = policy_dir / RECEIPT_FILENAME
    if receipt_path.is_file():
        existing = parse_server_start_receipt(json.loads(receipt_path.read_text()))
        if existing.server_instance_id != receipt.server_instance_id:
            raise ReceiptError(
                f"{policy_dir} already holds receipt "
                f"{existing.server_instance_id}; a second server instance must "
                "use a new policy directory"
            )
        return existing

    atomic_write_json(receipt_path, payload)
    return receipt


class ServerInstanceBinding:
    """Lets the scheduler adopt its `server_instance_id` once attested.

    The EngineCore process cannot compute the id alone -- it does not know
    the API process's identity -- so it publishes its contribution and then
    watches for the joined receipt. Until the attestor publishes, evidence
    records honestly carry `server_instance_id: null` rather than a guess.
    """

    def __init__(
        self,
        policy_dir: Path | None,
        *,
        expected_nonce: str | None,
        engine_core_pid: int,
    ) -> None:
        self._path = Path(policy_dir) / RECEIPT_FILENAME if policy_dir else None
        self._expected_nonce = expected_nonce
        self._engine_core_pid = engine_core_pid
        self._server_instance_id: str | None = None

    @property
    def server_instance_id(self) -> str | None:
        return self._server_instance_id

    def refresh(self) -> str | None:
        """Adopt the receipt's instance id if it binds *this* process.

        Never raises: attestation is evidence for an external campaign, and
        a missing or malformed receipt must not interrupt serving.
        """
        if self._server_instance_id is not None or self._path is None:
            return self._server_instance_id
        try:
            receipt = read_server_start_receipt(self._path)
        except (OSError, ValueError):
            return None
        if receipt.engine_core_pid != self._engine_core_pid:
            return None
        if self._expected_nonce and receipt.attestation_nonce != self._expected_nonce:
            return None
        self._server_instance_id = receipt.server_instance_id
        return self._server_instance_id


def resolve_attestation_nonce() -> str | None:
    value = os.environ.get(ATTESTATION_NONCE_ENV)
    return value or None


def publish_startup_attestation(
    vllm_config: Any,
    *,
    policy_dir: Path | None,
    startup_max_num_seqs: int,
    startup_max_num_batched_tokens: int,
    gpu_probe: Any = None,
) -> ServerInstanceBinding:
    """EngineCore-side startup hook. Fail-open: serving always continues.

    A receipt failure leaves the server usable but leaves formal readiness
    false, because the campaign cannot bind a cell to an instance it was
    never shown.
    """
    nonce = resolve_attestation_nonce()
    binding = ServerInstanceBinding(
        policy_dir, expected_nonce=nonce, engine_core_pid=os.getpid()
    )
    if policy_dir is None or nonce is None:
        return binding
    try:
        contribution = collect_engine_contribution(
            vllm_config,
            startup_max_num_seqs=startup_max_num_seqs,
            startup_max_num_batched_tokens=startup_max_num_batched_tokens,
            attestation_nonce=nonce,
            gpu_probe=gpu_probe,
        )
        publish_engine_contribution(Path(policy_dir), contribution)
    except Exception:  # noqa: BLE001 - attestation never blocks serving
        logger.warning(
            "GLADIUS server-start attestation unavailable; serving continues "
            "but this instance cannot certify a formal discovery cell",
            exc_info=True,
        )
    return binding
