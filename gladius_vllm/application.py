"""Atomic, fail-open policy-application acknowledgement writer."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gladius_vllm.policy import PolicyDecision
from gladius_vllm.schema import SCHEMA_VERSION, format_iso8601, parse_iso8601

logger = logging.getLogger(__name__)

ApplicationState = Literal["active", "attriting", "fallback"]
_APPLICATION_FIELDS = {
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
_ADMISSION_FIELDS = {"max_num_seqs", "max_num_batched_tokens"}


@dataclass(frozen=True)
class PolicyApplication:
    engine_id: str
    model_id: str
    generation: int
    policy_id: str
    decision_id: str
    scheduler_step: int
    state: ApplicationState


def parse_policy_application(payload: object) -> PolicyApplication:
    """Strictly validate a canonical policy-application acknowledgement."""
    if not isinstance(payload, dict) or set(payload) != _APPLICATION_FIELDS:
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
    state = payload["state"]
    if state not in {"active", "attriting", "fallback"}:
        raise ValueError("invalid policy application state")
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
    if state == "active" and (requested != effective or any(clamped.values())):
        raise ValueError("active application must be effective and unclamped")
    if state == "attriting" and not (
        clamped["max_num_seqs"]
        and effective["max_num_seqs"] > requested["max_num_seqs"]
    ):
        raise ValueError("attriting application must wait for sequence attrition")
    return PolicyApplication(
        engine_id=payload["engine_id"],
        model_id=payload["model_id"],
        generation=payload["generation"],
        policy_id=payload["policy_id"],
        decision_id=payload["decision_id"],
        scheduler_step=payload["scheduler_step"],
        state=state,
    )


class PolicyApplicationWriter:
    """Publish the scheduler's latest effective policy state.

    Rewrites occur only when policy identity, state, or effective admission
    changes. All filesystem failures are contained so acknowledgement I/O can
    never interrupt scheduling.
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
    ) -> None:
        if self._path is None:
            return
        try:
            payload, fingerprint = self._build_payload(
                decision,
                scheduler_step=scheduler_step,
                effective_max_num_seqs=effective_max_num_seqs,
                effective_max_num_batched_tokens=(
                    effective_max_num_batched_tokens
                ),
            )
            if fingerprint == self._last_fingerprint:
                return
            parse_policy_application(payload)
            if self._write(payload):
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
    ) -> tuple[dict[str, object], tuple[object, ...]]:
        policy_id = decision.policy_id or "startup-default"
        generation = decision.generation if decision.generation is not None else 0
        clamped_seqs = effective_max_num_seqs != decision.max_num_seqs
        clamped_tokens = (
            effective_max_num_batched_tokens != decision.max_num_batched_tokens
        )
        if decision.source == "default":
            state = "fallback"
        elif clamped_seqs and effective_max_num_seqs > decision.max_num_seqs:
            state = "attriting"
        elif not clamped_seqs and not clamped_tokens:
            state = "active"
        else:
            state = "fallback"
        fingerprint = (
            generation,
            policy_id,
            state,
            decision.max_num_seqs,
            decision.max_num_batched_tokens,
            effective_max_num_seqs,
            effective_max_num_batched_tokens,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "engine_id": self._engine_id,
            "model_id": self._model_id,
            "generation": generation,
            "policy_id": policy_id,
            "decision_id": policy_id,
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

    def _write(self, payload: dict[str, object]) -> bool:
        assert self._path is not None
        temporary_path: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path)
            temporary_path = None
            directory_fd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return True
        except Exception:
            logger.warning(
                "GLADIUS policy application acknowledgement failed for %s",
                self._path,
                exc_info=True,
            )
            return False
        finally:
            if temporary_path is not None:
                with contextlib.suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
