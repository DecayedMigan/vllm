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


def verify_server_start_receipt(
    receipt: ServerStartReceipt,
    *,
    expected_nonce: str | None = None,
    expected_engine_id: str | None = None,
    expected_model_id: str | None = None,
    expected_listen_host: str | None = None,
    expected_listen_port: int | None = None,
    expected_gpu_uuid: str | None = None,
    expected_api_pid: int | None = None,
    expected_engine_core_pid: int | None = None,
    expected_digests: dict[str, str] | None = None,
    require_formal_startup: bool = False,
) -> list[str]:
    """Return every way `receipt` disagrees with what the campaign expected.

    Returns a list rather than raising on the first problem so an operator
    sees the complete diagnosis of a mis-launched server in one pass.
    """
    errors: list[str] = []

    def check(name: str, expected: object, actual: object) -> None:
        if expected is not None and expected != actual:
            errors.append(f"{name}: expected {expected!r}, receipt has {actual!r}")

    check("attestation_nonce", expected_nonce, receipt.attestation_nonce)
    check("engine_id", expected_engine_id, receipt.engine_id)
    check("model_id", expected_model_id, receipt.model_id)
    check("listen_host", expected_listen_host, receipt.listen_host)
    check("listen_port", expected_listen_port, receipt.listen_port)
    check("physical_gpu_uuid", expected_gpu_uuid, receipt.physical_gpu_uuid)
    check("api_pid", expected_api_pid, receipt.api_pid)
    check("engine_core_pid", expected_engine_core_pid, receipt.engine_core_pid)

    for field, expected_digest in (expected_digests or {}).items():
        if field not in _DIGEST_FIELDS:
            errors.append(f"{field} is not a receipt digest field")
            continue
        check(field, expected_digest, getattr(receipt, field))

    for pid, identity, label in (
        (receipt.api_pid, receipt.api_process_start_identity, "api"),
        (
            receipt.engine_core_pid,
            receipt.engine_core_process_start_identity,
            "engine_core",
        ),
    ):
        try:
            live_identity = process_start_identity(pid)
        except DigestError:
            errors.append(f"{label} process {pid} is no longer running")
            continue
        if live_identity != identity:
            errors.append(
                f"{label} process {pid} was replaced since the receipt was written"
            )

    if require_formal_startup:
        for field, expected_value in FORMAL_STARTUP_EXPECTATIONS.items():
            actual = getattr(receipt, field)
            if actual != expected_value:
                errors.append(
                    f"formal startup {field}: expected {expected_value!r}, "
                    f"receipt has {actual!r}"
                )
    return errors


def _resolve_physical_gpu(device_index: int = 0) -> tuple[str, str]:
    """Read the GPU UUID/name from inside the process that owns the device.

    Deliberately not inferred from the launcher's `CUDA_VISIBLE_DEVICES`
    string: that only records what the launcher *asked* for, while the
    campaign needs to know which physical device the model actually landed
    on.
    """

    def _text(value: object) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    torch_properties = None
    try:
        import torch

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            torch_properties = torch.cuda.get_device_properties(device_index)
    except Exception:  # noqa: BLE001 - torch import/probe must not break startup
        logger.debug("GLADIUS: torch device probe unavailable", exc_info=True)

    # NVML is authoritative on the formal NVIDIA target and reports the same
    # UUID string an operator sees from `nvidia-smi`, so it is tried first.
    nvml_error: Exception | None = None
    try:
        from vllm.third_party import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            return (
                _text(pynvml.nvmlDeviceGetUUID(handle)),
                _text(pynvml.nvmlDeviceGetName(handle)),
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 - NVML is absent on ROCm/CPU hosts
        nvml_error = exc

    # Fallback for non-NVIDIA development hosts (ROCm): torch reports the
    # device it actually bound, which is still a measurement from inside the
    # model process rather than an inference from CUDA_VISIBLE_DEVICES.
    uuid_value = getattr(torch_properties, "uuid", None)
    if uuid_value is not None:
        return _text(uuid_value), _text(getattr(torch_properties, "name", "unknown"))
    raise DigestError(f"cannot read physical GPU identity: {nvml_error}")


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
    physical_gpu_uuid, physical_gpu_name = probe()

    return {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "attestation_nonce": attestation_nonce,
        "observed_at": format_iso8601(),
        "engine_core_pid": os.getpid(),
        "engine_core_process_start_identity": process_start_identity(os.getpid()),
        "engine_id": resolve_engine_id(vllm_config),
        "model_id": resolve_model_id(vllm_config),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu_uuid": physical_gpu_uuid,
        "physical_gpu_name": physical_gpu_name,
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

    payload = {
        key: value for key, value in contribution.items() if key != "observed_at"
    }
    payload.update(
        {
            "created_at": now or format_iso8601(),
            "api_pid": int(api_pid),
            "api_process_start_identity": process_start_identity(int(api_pid)),
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
