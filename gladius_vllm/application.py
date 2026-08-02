"""Atomic, fail-open policy-application acknowledgement writer.

Execution-evidence schema 2.0.0 adds two fields that turn this file from a
convenience into evidence:

* `server_instance_id` binds the acknowledgement to one attested serving
  process pair. `engine_id`/`model_id` survive a restart; this does not.
* `generation_high_watermark` publishes the greatest generation this
  scheduler has ever accepted. Without it, a client that reconnects to a
  still-running server after a fallback cannot tell which generations the
  scheduler will now reject, and could reuse one.

The historical 1.x parser is kept alongside the 2.0.0 parser. A 1.x document
must never satisfy a 2.0.0 requirement, because it cannot prove which process
produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gladius_vllm.atomic import try_atomic_write_json
from gladius_vllm.policy import PolicyDecision
from gladius_vllm.schema import (
    EXECUTION_EVIDENCE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    format_iso8601,
    parse_iso8601,
)

logger = logging.getLogger(__name__)

ApplicationState = Literal["active", "attriting", "fallback"]

_APPLICATION_FIELDS_V1 = {
    "schema_version",
    "engine_id",
    "model_id",
    "generation",
    "policy_id",
    "decision_id",
    "observed_at",
    "scheduler_step",
    "state",
    "requested_admission",
    "effective_admission",
    "clamped",
}
_APPLICATION_FIELDS_V2 = _APPLICATION_FIELDS_V1 | {
    "server_instance_id",
    "generation_high_watermark",
}
_ADMISSION_FIELDS = {"max_num_seqs", "max_num_batched_tokens"}
_APPLICATION_STATES = {"active", "attriting", "fallback"}


@dataclass(frozen=True)
class PolicyApplication:
    """A strictly parsed acknowledgement (1.x fields only)."""

    engine_id: str
    model_id: str
    generation: int
    policy_id: str
    decision_id: str
    scheduler_step: int
    state: ApplicationState


@dataclass(frozen=True)
class PolicyApplicationV2:
    """A strictly parsed execution-evidence 2.0.0 acknowledgement."""

    server_instance_id: str | None
    engine_id: str
    model_id: str
    generation: int | None
    policy_id: str | None
    decision_id: str | None
    generation_high_watermark: int | None
    scheduler_step: int
    state: ApplicationState

    @property
    def is_native(self) -> bool:
        return self.generation is None


def _validate_admission_block(payload: dict) -> tuple[dict, dict, dict]:
    for field in ("requested_admission", "effective_admission", "clamped"):
        value = payload[field]
        if not isinstance(value, dict) or set(value) != _ADMISSION_FIELDS:
            raise ValueError(f"{field} does not match the contract")
    requested = payload["requested_admission"]
    effective = payload["effective_admission"]
    clamped = payload["clamped"]
    for admission in (requested, effective):
        for field in _ADMISSION_FIELDS:
            value = admission[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
    if not all(isinstance(clamped[field], bool) for field in _ADMISSION_FIELDS):
        raise ValueError("clamped values must be boolean")
    return requested, effective, clamped


def _validate_state_semantics(
    state: str, requested: dict, effective: dict, clamped: dict
) -> None:
    if state not in _APPLICATION_STATES:
        raise ValueError("invalid policy application state")
    if state == "active" and (requested != effective or any(clamped.values())):
        raise ValueError("active application must be effective and unclamped")
    if state == "attriting" and not (
        clamped["max_num_seqs"]
        and effective["max_num_seqs"] > requested["max_num_seqs"]
        and not clamped["max_num_batched_tokens"]
    ):
        raise ValueError("attriting application must wait for sequence attrition")


def parse_policy_application(payload: object) -> PolicyApplication:
    """Strictly validate a historical schema-1.0.0 acknowledgement.

    Retained so 1.x artifacts remain readable. It cannot certify a formal
    discovery cell: a 1.x document carries no `server_instance_id`.
    """
    if not isinstance(payload, dict) or set(payload) != _APPLICATION_FIELDS_V1:
        raise ValueError("policy application fields do not match the contract")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported policy application schema_version")
    for field in ("engine_id", "model_id", "policy_id", "decision_id"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"{field} must be a non-empty string")
    for field in ("generation", "scheduler_step"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    parse_iso8601(payload["observed_at"])
    requested, effective, clamped = _validate_admission_block(payload)
    _validate_state_semantics(payload["state"], requested, effective, clamped)
    return PolicyApplication(
        engine_id=payload["engine_id"],
        model_id=payload["model_id"],
        generation=payload["generation"],
        policy_id=payload["policy_id"],
        decision_id=payload["decision_id"],
        scheduler_step=payload["scheduler_step"],
        state=payload["state"],
    )


def parse_policy_application_v2(
    payload: object,
    *,
    allow_unattested: bool = False,
) -> PolicyApplicationV2:
    """Strictly validate an execution-evidence 2.0.0 acknowledgement.

    `allow_unattested` exists only for the writer's own pre-publication
    self-check while a server is still waiting for its receipt to be
    assembled. A formal campaign always parses with it off, so an
    unattested acknowledgement fails closed.
    """
    if not isinstance(payload, dict) or set(payload) != _APPLICATION_FIELDS_V2:
        raise ValueError("policy application fields do not match the contract")
    if payload["schema_version"] != EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            "policy application requires execution-evidence schema "
            f"{EXECUTION_EVIDENCE_SCHEMA_VERSION}"
        )
    for field in ("engine_id", "model_id"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"{field} must be a non-empty string")

    server_instance_id = payload["server_instance_id"]
    if server_instance_id is None:
        if not allow_unattested:
            raise ValueError(
                "server_instance_id is required: an unattested acknowledgement "
                "cannot prove which serving process applied the policy"
            )
    elif not isinstance(server_instance_id, str) or not server_instance_id:
        raise ValueError("server_instance_id must be a non-empty string or null")

    step = payload["scheduler_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("scheduler_step must be a non-negative integer")

    generation = payload["generation"]
    policy_id = payload["policy_id"]
    decision_id = payload["decision_id"]
    identity_present = (
        generation is not None,
        policy_id is not None,
        decision_id is not None,
    )
    if len(set(identity_present)) != 1:
        raise ValueError(
            "half-native application: generation, policy_id, and decision_id "
            "must be all present or all null"
        )
    native = not identity_present[0]
    if not native:
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("generation must be a non-negative integer")
        if generation < 0:
            raise ValueError("generation must be a non-negative integer")
        for field, value in (("policy_id", policy_id), ("decision_id", decision_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        if policy_id != decision_id:
            raise ValueError("policy_id and decision_id must be identical")

    watermark = payload["generation_high_watermark"]
    if watermark is not None:
        if isinstance(watermark, bool) or not isinstance(watermark, int):
            raise ValueError("generation_high_watermark must be an integer or null")
        if watermark < 0:
            raise ValueError("generation_high_watermark must be non-negative")
        if not native and watermark < generation:
            raise ValueError(
                "generation_high_watermark must be >= the applied generation"
            )

    parse_iso8601(payload["observed_at"])
    requested, effective, clamped = _validate_admission_block(payload)
    state = payload["state"]
    _validate_state_semantics(state, requested, effective, clamped)
    if native and state != "fallback":
        raise ValueError("a native/default decision can only be in fallback state")

    return PolicyApplicationV2(
        server_instance_id=server_instance_id,
        engine_id=payload["engine_id"],
        model_id=payload["model_id"],
        generation=generation,
        policy_id=policy_id,
        decision_id=decision_id,
        generation_high_watermark=watermark,
        scheduler_step=step,
        state=state,
    )


def read_policy_application_v2(
    path: Path, *, allow_unattested: bool = False
) -> PolicyApplicationV2:
    import json

    return parse_policy_application_v2(
        json.loads(Path(path).read_text()), allow_unattested=allow_unattested
    )


def verify_application_binds_receipt(
    application: PolicyApplicationV2, receipt: object
) -> list[str]:
    """Check an acknowledgement against the server-start receipt.

    An acknowledgement whose instance/engine/model triple disagrees with the
    receipt describes a different server than the one the campaign attested.
    """
    errors: list[str] = []
    for field in ("server_instance_id", "engine_id", "model_id"):
        expected = getattr(receipt, field, None)
        actual = getattr(application, field)
        if expected != actual:
            errors.append(
                f"application {field} {actual!r} does not match receipt {expected!r}"
            )
    return errors


class PolicyApplicationWriter:
    """Publish the scheduler's latest effective policy state.

    Rewrites occur only when policy identity, state, effective admission, or
    the published high-watermark changes. All filesystem failures are
    contained so acknowledgement I/O can never interrupt scheduling.
    """

    def __init__(self, path: Path | None, engine_id: str, model_id: str) -> None:
        self._path = path
        self._engine_id = engine_id
        self._model_id = model_id
        self._last_fingerprint: tuple[object, ...] | None = None

    def record(
        self,
        decision: PolicyDecision,
        *,
        scheduler_step: int,
        effective_max_num_seqs: int,
        effective_max_num_batched_tokens: int,
        server_instance_id: str | None = None,
        generation_high_watermark: int | None = None,
    ) -> None:
        if self._path is None:
            return
        try:
            payload, fingerprint = self._build_payload(
                decision,
                scheduler_step=scheduler_step,
                effective_max_num_seqs=effective_max_num_seqs,
                effective_max_num_batched_tokens=(effective_max_num_batched_tokens),
                server_instance_id=server_instance_id,
                generation_high_watermark=generation_high_watermark,
            )
            if fingerprint == self._last_fingerprint:
                return
            parse_policy_application_v2(payload, allow_unattested=True)
            if try_atomic_write_json(
                self._path, payload, what="policy application acknowledgement"
            ):
                self._last_fingerprint = fingerprint
        except Exception:
            logger.warning(
                "GLADIUS policy application record failed",
                exc_info=True,
            )

    def _build_payload(
        self,
        decision: PolicyDecision,
        *,
        scheduler_step: int,
        effective_max_num_seqs: int,
        effective_max_num_batched_tokens: int,
        server_instance_id: str | None,
        generation_high_watermark: int | None,
    ) -> tuple[dict[str, object], tuple[object, ...]]:
        native = decision.source == "default"
        clamped_seqs = effective_max_num_seqs != decision.max_num_seqs
        clamped_tokens = (
            effective_max_num_batched_tokens != decision.max_num_batched_tokens
        )
        if native:
            # A native/default decision is never "the policy took effect".
            # Reporting it as generation 0 / "startup-default" (schema 1.x
            # behaviour) invented an identity the scheduler never accepted
            # and hid the real high-watermark from a resuming client.
            state: ApplicationState = "fallback"
            generation: int | None = None
            policy_id: str | None = None
        else:
            generation = decision.generation
            policy_id = decision.policy_id
            if clamped_seqs and effective_max_num_seqs > decision.max_num_seqs:
                state = "attriting" if not clamped_tokens else "fallback"
            elif not clamped_seqs and not clamped_tokens:
                state = "active"
            else:
                state = "fallback"
        fingerprint = (
            server_instance_id,
            generation,
            policy_id,
            generation_high_watermark,
            state,
            decision.max_num_seqs,
            decision.max_num_batched_tokens,
            effective_max_num_seqs,
            effective_max_num_batched_tokens,
        )
        payload = {
            "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
            "server_instance_id": server_instance_id,
            "engine_id": self._engine_id,
            "model_id": self._model_id,
            "generation": generation,
            "policy_id": policy_id,
            "decision_id": policy_id,
            "generation_high_watermark": generation_high_watermark,
            "observed_at": format_iso8601(),
            "scheduler_step": scheduler_step,
            "state": state,
            "requested_admission": {
                "max_num_seqs": decision.max_num_seqs,
                "max_num_batched_tokens": decision.max_num_batched_tokens,
            },
            "effective_admission": {
                "max_num_seqs": effective_max_num_seqs,
                "max_num_batched_tokens": effective_max_num_batched_tokens,
            },
            "clamped": {
                "max_num_seqs": clamped_seqs,
                "max_num_batched_tokens": clamped_tokens,
            },
        }
        return payload, fingerprint
