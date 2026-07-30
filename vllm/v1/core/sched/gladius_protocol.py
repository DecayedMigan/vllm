"""Standard-library implementation of the GLADIUS control protocol v1."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = 1
SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "generation",
        "policy_id",
        "model_id",
        "admission_limit",
        "source_experience_id",
        "created_at",
        "expires_at",
    }
)


class ProtocolError(ValueError):
    """Raised when a GLADIUS protocol payload violates the v1 schema."""


class PolicySnapshot(NamedTuple):
    """Validated policy snapshot received from the GLADIUS control plane."""

    generation: int
    policy_id: str
    model_id: str
    admission_limit: int
    source_experience_id: str | None
    created_at: datetime
    expires_at: datetime


class PolicyState(NamedTuple):
    """Policy state actually used by a scheduler step."""

    admission_limit: int
    generation: int
    policy_id: str
    error: str | None
    expires_at: datetime | None


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def _require_int(payload: dict[str, Any], field: str, minimum: int) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{field} must be an integer >= {minimum}")
    return value


def _require_string(payload: dict[str, Any], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _parse_timestamp(payload: dict[str, Any], field: str) -> datetime:
    value = payload[field]
    if not isinstance(value, str):
        raise ProtocolError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProtocolError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def parse_policy_snapshot(payload: object) -> PolicySnapshot:
    """Parse and strictly validate a GLADIUS policy snapshot."""

    if not isinstance(payload, dict):
        raise ProtocolError("snapshot must be a JSON object")
    fields = set(payload)
    if fields != SNAPSHOT_FIELDS:
        missing = sorted(SNAPSHOT_FIELDS - fields)
        unknown = sorted(fields - SNAPSHOT_FIELDS)
        raise ProtocolError(f"fields mismatch: missing={missing}, unknown={unknown}")
    schema_version = _require_int(payload, "schema_version", 0)
    if schema_version != SCHEMA_VERSION:
        raise ProtocolError(
            f"schema_version must be {SCHEMA_VERSION}, got {schema_version}"
        )
    generation = _require_int(payload, "generation", 0)
    admission_limit = _require_int(payload, "admission_limit", 1)
    policy_id = _require_string(payload, "policy_id")
    model_id = _require_string(payload, "model_id")
    source_experience_id = payload["source_experience_id"]
    if source_experience_id is not None and (
        not isinstance(source_experience_id, str) or not source_experience_id.strip()
    ):
        raise ProtocolError("source_experience_id must be null or a non-empty string")
    created_at = _parse_timestamp(payload, "created_at")
    expires_at = _parse_timestamp(payload, "expires_at")
    if expires_at <= created_at:
        raise ProtocolError("expires_at must be later than created_at")
    return PolicySnapshot(
        generation=generation,
        policy_id=policy_id,
        model_id=model_id,
        admission_limit=admission_limit,
        source_experience_id=source_experience_id,
        created_at=created_at,
        expires_at=expires_at,
    )


def read_policy_snapshot(path: Path) -> PolicySnapshot:
    """Read and validate one atomically published policy snapshot."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolError(f"policy file missing: {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"policy file unreadable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg}") from exc
    return parse_policy_snapshot(payload)


class PolicyController:
    """Maintain the last valid unexpired policy for one vLLM instance."""

    def __init__(
        self,
        *,
        policy_path: Path | None,
        model_id: str,
        startup_admission_limit: int,
    ) -> None:
        if startup_admission_limit < 1:
            raise ValueError("startup_admission_limit must be >= 1")
        self._policy_path = policy_path
        self._model_id = model_id
        self._startup_admission_limit = startup_admission_limit
        self._current: PolicySnapshot | None = None
        self._highest_generation = -1
        self._observed_fingerprint: tuple[int, int, int] | None = None
        self._has_observed_file = False
        self._last_error: str | None = None

    def _state(self, error: str | None) -> PolicyState:
        if self._current is None:
            return PolicyState(
                admission_limit=self._startup_admission_limit,
                generation=0,
                policy_id="startup",
                error=error,
                expires_at=None,
            )
        snapshot = self._current
        return PolicyState(
            admission_limit=min(
                snapshot.admission_limit, self._startup_admission_limit
            ),
            generation=snapshot.generation,
            policy_id=snapshot.policy_id,
            error=error,
            expires_at=snapshot.expires_at,
        )

    def refresh(self, now: datetime) -> PolicyState:
        """Reload a changed snapshot and return the policy valid at ``now``."""

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must include a timezone")
        now = now.astimezone(UTC)
        expiry_error = None
        if self._current is not None and now >= self._current.expires_at:
            expiry_error = f"policy expired at {self._current.expires_at.isoformat()}"
            self._current = None
            self._last_error = expiry_error
        if self._policy_path is None:
            return self._state(expiry_error or self._last_error)

        try:
            stat = self._policy_path.stat()
            fingerprint = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            fingerprint = None
        except OSError as exc:
            self._last_error = f"policy file unreadable: {exc}"
            return self._state(self._last_error)

        if self._has_observed_file and fingerprint == self._observed_fingerprint:
            return self._state(expiry_error or self._last_error)
        self._has_observed_file = True
        self._observed_fingerprint = fingerprint
        if fingerprint is None:
            self._last_error = f"policy file missing: {self._policy_path}"
            return self._state(self._last_error)

        try:
            candidate = read_policy_snapshot(self._policy_path)
            if candidate.model_id != self._model_id:
                raise ProtocolError(
                    "model_id mismatch: "
                    f"expected {self._model_id!r}, got {candidate.model_id!r}"
                )
            if now < candidate.created_at:
                raise ProtocolError(
                    f"policy not active before {candidate.created_at.isoformat()}"
                )
            if now >= candidate.expires_at:
                raise ProtocolError(
                    f"policy expired at {candidate.expires_at.isoformat()}"
                )
            if candidate.generation < self._highest_generation:
                raise ProtocolError(
                    "stale generation: "
                    f"{candidate.generation} < {self._highest_generation}"
                )
        except ProtocolError as exc:
            self._last_error = str(exc)
            return self._state(self._last_error)

        self._current = candidate
        self._highest_generation = max(self._highest_generation, candidate.generation)
        self._last_error = None
        return self._state(None)


def _format_timestamp(timestamp: datetime) -> str:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def append_telemetry(
    path: Path,
    *,
    engine_id: str,
    model_id: str,
    step: int,
    state: PolicyState,
    running: int,
    waiting: int,
    scheduled_requests: int,
    scheduled_tokens: int,
    timestamp: datetime,
) -> None:
    """Append one scheduler telemetry record as a complete JSON line."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "engine_id": engine_id,
        "model_id": model_id,
        "step": step,
        "generation": state.generation,
        "policy_id": state.policy_id,
        "running": running,
        "waiting": waiting,
        "scheduled_requests": scheduled_requests,
        "scheduled_tokens": scheduled_tokens,
        "timestamp": _format_timestamp(timestamp),
        "error": state.error,
    }
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    with path.open("a", encoding="utf-8") as telemetry_file:
        telemetry_file.write(line)
        telemetry_file.write("\n")
        telemetry_file.flush()
