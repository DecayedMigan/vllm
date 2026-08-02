"""The seal request/acknowledgement protocol between an operator and a server.

`seal_telemetry()` previously had no caller anywhere in the repository --
the seal was written by nothing, so every claim about certified evidence
described a code path production never reached. A test that calls the method
directly does not fix that; it only proves the method works when called.

The transition therefore has a real trigger. The operator writes an
instance-bound `seal_request.json`; the *live scheduler* observes it at a
safe scheduling boundary, writes terminal state, seals, retires the
directory, and answers in `seal_ack.json`. Nothing outside the serving
process ever writes the seal.

Being instance-bound matters: a policy directory outlives the process that
filled it, so a request naming a different `server_instance_id` -- a stale
one left behind by an earlier launch -- must be refused rather than sealing
whatever happens to be running now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gladius_vllm.atomic import atomic_write_json
from gladius_vllm.evidence_codes import (
    SEAL_REQUEST_AFTER_RETIREMENT,
    SEAL_REQUEST_DEPLOYMENT_CHANGED,
    SEAL_REQUEST_FOREIGN_INSTANCE,
    SEAL_REQUEST_SCHEMA_INVALID,
    SEAL_REQUEST_STALE_GENERATION,
    classify,
)
from gladius_vllm.schema import EXECUTION_EVIDENCE_SCHEMA_VERSION, format_iso8601

SEAL_REQUEST_FILENAME = "seal_request.json"
SEAL_ACK_FILENAME = "seal_ack.json"

SEAL_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "server_instance_id",
        "deployment_manifest_sha256",
        "expected_final_generation",
        "attestation_nonce",
        "requested_at",
    }
)

SEAL_ACK_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "server_instance_id",
        "status",
        "telemetry_seal_sha256",
        "error_code",
        "error_detail",
        "acknowledged_at",
    }
)

ACK_SEALED = "sealed"
ACK_REFUSED = "refused"


class SealRequestError(ValueError):
    """A seal request could not be honoured by this server instance."""


@dataclass(frozen=True)
class SealRequest:
    request_id: str
    server_instance_id: str
    deployment_manifest_sha256: str
    expected_final_generation: int | None
    attestation_nonce: str
    requested_at: str


def parse_seal_request(payload: object) -> SealRequest:
    """Exact-schema parse. An unknown field was not written by this contract."""
    if not isinstance(payload, dict):
        raise SealRequestError(
            classify(SEAL_REQUEST_SCHEMA_INVALID, "seal request must be a JSON object")
        )
    fields = set(payload)
    if fields != set(SEAL_REQUEST_FIELDS):
        missing = sorted(SEAL_REQUEST_FIELDS - fields)
        unknown = sorted(fields - SEAL_REQUEST_FIELDS)
        raise SealRequestError(
            classify(
                SEAL_REQUEST_SCHEMA_INVALID,
                f"seal request fields mismatch: missing={missing}, unknown={unknown}",
            )
        )
    if payload["schema_version"] != EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise SealRequestError(
            classify(
                SEAL_REQUEST_SCHEMA_INVALID,
                "seal request requires execution-evidence schema "
                f"{EXECUTION_EVIDENCE_SCHEMA_VERSION}",
            )
        )
    for field in (
        "request_id",
        "server_instance_id",
        "deployment_manifest_sha256",
        "attestation_nonce",
        "requested_at",
    ):
        value = payload[field]
        if not isinstance(value, str) or not value:
            raise SealRequestError(
                classify(
                    SEAL_REQUEST_SCHEMA_INVALID, f"{field} must be a non-empty string"
                )
            )
    generation = payload["expected_final_generation"]
    if generation is not None and (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise SealRequestError(
            classify(
                SEAL_REQUEST_SCHEMA_INVALID,
                "expected_final_generation must be a non-negative integer or null",
            )
        )
    return SealRequest(
        request_id=payload["request_id"],
        server_instance_id=payload["server_instance_id"],
        deployment_manifest_sha256=payload["deployment_manifest_sha256"],
        expected_final_generation=generation,
        attestation_nonce=payload["attestation_nonce"],
        requested_at=payload["requested_at"],
    )


def write_seal_request(
    policy_dir: Path,
    *,
    server_instance_id: str,
    deployment_manifest_sha256: str,
    expected_final_generation: int | None,
    attestation_nonce: str,
    request_id: str | None = None,
) -> Path:
    """Atomically publish a seal request for one specific server instance."""
    policy_dir = Path(policy_dir)
    if not server_instance_id:
        raise SealRequestError(
            classify(
                SEAL_REQUEST_SCHEMA_INVALID,
                "a seal request must name the server instance it is for; an "
                "unattested server cannot be asked to certify anything",
            )
        )
    payload = {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "request_id": request_id or f"seal-{server_instance_id}",
        "server_instance_id": server_instance_id,
        "deployment_manifest_sha256": deployment_manifest_sha256,
        "expected_final_generation": expected_final_generation,
        "attestation_nonce": attestation_nonce,
        "requested_at": format_iso8601(),
    }
    parse_seal_request(payload)
    path = policy_dir / SEAL_REQUEST_FILENAME
    atomic_write_json(path, payload)
    return path


def read_seal_ack(policy_dir: Path) -> dict[str, Any] | None:
    """The acknowledgement, if the server has answered yet."""
    path = Path(policy_dir) / SEAL_ACK_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != set(SEAL_ACK_FIELDS):
        raise SealRequestError(
            classify(
                SEAL_REQUEST_SCHEMA_INVALID,
                "seal acknowledgement fields do not match the contract",
            )
        )
    return payload


def write_seal_ack(
    policy_dir: Path,
    *,
    request: SealRequest,
    status: str,
    telemetry_seal_sha256: str | None = None,
    error_code: str | None = None,
    error_detail: str | None = None,
) -> Path:
    payload = {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "request_id": request.request_id,
        "server_instance_id": request.server_instance_id,
        "status": status,
        "telemetry_seal_sha256": telemetry_seal_sha256,
        "error_code": error_code,
        "error_detail": error_detail,
        "acknowledged_at": format_iso8601(),
    }
    path = Path(policy_dir) / SEAL_ACK_FILENAME
    atomic_write_json(path, payload)
    return path


def classify_refusal(
    request: SealRequest,
    *,
    server_instance_id: str | None,
    deployment_manifest_sha256: str | None,
    attestation_nonce: str | None,
    observed_generation: int | None,
    already_retired: bool,
) -> str | None:
    """Why this instance must not honour `request`, or None if it may.

    Kept separate from the scheduler so the decision is testable without a
    live engine, and so every refusal reason is enumerated in one place.
    """
    if server_instance_id is None:
        return classify(
            SEAL_REQUEST_FOREIGN_INSTANCE,
            "this server has no attested instance id and cannot certify evidence",
        )
    if request.server_instance_id != server_instance_id:
        return classify(
            SEAL_REQUEST_FOREIGN_INSTANCE,
            f"seal request names instance {request.server_instance_id!r}, but "
            f"this server is {server_instance_id!r}",
        )
    if attestation_nonce is not None and request.attestation_nonce != attestation_nonce:
        return classify(
            SEAL_REQUEST_FOREIGN_INSTANCE,
            "seal request carries a different attestation nonce; it belongs to "
            "another launch",
        )
    if (
        deployment_manifest_sha256 is not None
        and request.deployment_manifest_sha256 != deployment_manifest_sha256
    ):
        return classify(
            SEAL_REQUEST_DEPLOYMENT_CHANGED,
            "seal request was written against a different deployment manifest",
        )
    if (
        request.expected_final_generation is not None
        and observed_generation is not None
        and request.expected_final_generation > observed_generation
    ):
        return classify(
            SEAL_REQUEST_STALE_GENERATION,
            f"seal request expects final generation "
            f"{request.expected_final_generation}, but this instance has only "
            f"reached {observed_generation}",
        )
    if already_retired:
        return classify(
            SEAL_REQUEST_AFTER_RETIREMENT,
            "this policy directory is already retired; a different request "
            "cannot reopen certified evidence",
        )
    return None
