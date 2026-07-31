"""Atomic policy-snapshot hot-reload for GladiusScheduler.

`parse_policy_snapshot()` is the public, canonical-contract parser (pure
structural validation, no notion of "this engine's identity"). `PolicyLoader`
layers engine/model-identity and generation-monotonicity checks on top of it
and is the single entry point used by the scheduler: `PolicyLoader.poll()`
never raises, and always returns a PolicyDecision usable directly.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from gladius_vllm.errors import (
    PolicyCorruptError,
    PolicyEngineMismatchError,
    PolicyStaleError,
)
from gladius_vllm.schema import (
    DEFAULT_POLICY_POLL_INTERVAL_MS,
    SUPPORTED_SCHEMA_MAJOR,
    parse_iso8601,
)

PolicyStatus = Literal[
    "active",
    "no_policy",
    "expired",
    "corrupt",
    "rejected_regression",
    "rejected_engine_mismatch",
]

PolicySource = Literal["file", "default"]

_CANONICAL_SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "generation",
        "policy_id",
        "model_id",
        "engine_id",
        "created_at",
        "expires_at",
        "admission",
    }
)
_ADMISSION_FIELDS = frozenset({"max_num_seqs", "max_num_batched_tokens"})

# Strict MAJOR.MINOR.PATCH only -- no leading zeros, no pre-release/build
# metadata suffixes, no truncated forms ("1", "1.0"). Publishers write
# exactly "1.0.0"; this reader tolerates any MINOR.PATCH as long as MAJOR
# matches SUPPORTED_SCHEMA_MAJOR.
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True)
class PolicySnapshot:
    """A structurally-valid canonical policy_snapshot.json payload."""

    schema_version: str
    generation: int
    policy_id: str
    model_id: str
    engine_id: str
    created_at: datetime
    expires_at: datetime
    max_num_seqs: int | None
    max_num_batched_tokens: int | None


@dataclass(frozen=True)
class PolicyDecision:
    """The scheduler's per-step admission ceilings and their provenance."""

    max_num_seqs: int
    max_num_batched_tokens: int
    policy_id: str | None
    generation: int | None
    status: PolicyStatus
    source: PolicySource


def _default_decision(
    startup_max_num_seqs: int,
    startup_max_num_batched_tokens: int,
    status: PolicyStatus,
) -> PolicyDecision:
    return PolicyDecision(
        max_num_seqs=startup_max_num_seqs,
        max_num_batched_tokens=startup_max_num_batched_tokens,
        policy_id=None,
        generation=None,
        status=status,
        source="default",
    )


def _require_nonempty_str(payload: dict, field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value:
        raise PolicyCorruptError(f"{field} must be a non-empty string, got {value!r}")
    return value


def _require_positive_int_or_none(admission: dict, field: str) -> int | None:
    value = admission[field]
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PolicyCorruptError(
            f"admission.{field} must be a positive int or null, got {value!r}"
        )
    return value


def parse_policy_snapshot(payload: object) -> PolicySnapshot:
    """Parse and strictly validate a canonical `policy_snapshot.json` payload.

    Pure structural validation only: top-level fields must exactly match the
    canonical set (unknown or missing fields are corrupt), IDs must be
    non-empty strings, `admission` must have both keys present (value a
    positive int or `null`), `schema_version` must be a semver string whose
    major component is the currently supported one, and `expires_at` must be
    strictly after `created_at` (both timezone-aware).

    Does NOT check engine/model identity against a running instance, or
    generation monotonicity -- those require caller context and are
    `PolicyLoader`-level concerns layered on top of this parse.

    Raises PolicyCorruptError on any violation.
    """
    if not isinstance(payload, dict):
        raise PolicyCorruptError(
            f"snapshot must be a JSON object, got {type(payload).__name__}"
        )

    fields = set(payload)
    if fields != _CANONICAL_SNAPSHOT_FIELDS:
        missing = sorted(_CANONICAL_SNAPSHOT_FIELDS - fields)
        unknown = sorted(fields - _CANONICAL_SNAPSHOT_FIELDS)
        raise PolicyCorruptError(
            f"top-level fields mismatch: missing={missing}, unknown={unknown}"
        )

    schema_version = payload["schema_version"]
    if not isinstance(schema_version, str):
        raise PolicyCorruptError(
            f"schema_version must be a string, got {schema_version!r}"
        )
    match = _SEMVER_RE.match(schema_version)
    if match is None:
        raise PolicyCorruptError(
            f"schema_version must be strict MAJOR.MINOR.PATCH, got {schema_version!r}"
        )
    if match.group(1) != SUPPORTED_SCHEMA_MAJOR:
        raise PolicyCorruptError(
            f"unsupported schema_version major: {schema_version!r} "
            f"(expected major {SUPPORTED_SCHEMA_MAJOR!r})"
        )

    generation = payload["generation"]
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
    ):
        raise PolicyCorruptError(
            f"generation must be a non-negative int, got {generation!r}"
        )

    policy_id = _require_nonempty_str(payload, "policy_id")
    model_id = _require_nonempty_str(payload, "model_id")
    engine_id = _require_nonempty_str(payload, "engine_id")

    try:
        created_at = parse_iso8601(payload["created_at"])
        expires_at = parse_iso8601(payload["expires_at"])
    except (TypeError, ValueError) as exc:
        raise PolicyCorruptError(f"invalid timestamp: {exc}") from exc
    if expires_at <= created_at:
        raise PolicyCorruptError("expires_at must be strictly after created_at")

    admission = payload["admission"]
    if not isinstance(admission, dict):
        raise PolicyCorruptError("admission must be an object")
    if set(admission) != _ADMISSION_FIELDS:
        missing = sorted(_ADMISSION_FIELDS - set(admission))
        unknown = sorted(set(admission) - _ADMISSION_FIELDS)
        raise PolicyCorruptError(
            f"admission fields mismatch: missing={missing}, unknown={unknown}"
        )

    max_num_seqs = _require_positive_int_or_none(admission, "max_num_seqs")
    max_num_batched_tokens = _require_positive_int_or_none(
        admission, "max_num_batched_tokens"
    )

    return PolicySnapshot(
        schema_version=schema_version,
        generation=generation,
        policy_id=policy_id,
        model_id=model_id,
        engine_id=engine_id,
        created_at=created_at,
        expires_at=expires_at,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )


class PolicyLoader:
    """Polls `policy_snapshot.json` and returns hot-reloadable admission ceilings.

    Startup ceilings act as both the "no policy" default and the hard upper
    clamp — callers pass them in and get them back verbatim whenever no
    policy is active, expired, or rejected.
    """

    def __init__(
        self,
        snapshot_path: Path | None,
        engine_id: str,
        model_id: str,
        startup_max_num_seqs: int,
        startup_max_num_batched_tokens: int,
        poll_interval_ms: int = DEFAULT_POLICY_POLL_INTERVAL_MS,
    ) -> None:
        self._snapshot_path = snapshot_path
        self._engine_id = engine_id
        self._model_id = model_id
        self._startup_max_num_seqs = startup_max_num_seqs
        self._startup_max_num_batched_tokens = startup_max_num_batched_tokens
        self._poll_interval_s = poll_interval_ms / 1000.0

        self._last_stat: tuple[int, int] | None = None  # (mtime_ns, size)
        self._last_poll_monotonic: float | None = None
        self._last_accepted_generation: int | None = None
        self._last_accepted_policy_id: str | None = None
        self._last_accepted_expires_at: datetime | None = None
        self._last_accepted_max_num_seqs: int | None = None
        self._last_accepted_max_num_batched_tokens: int | None = None
        self._last_decision: PolicyDecision = _default_decision(
            startup_max_num_seqs, startup_max_num_batched_tokens, "no_policy"
        )

    def poll(self) -> PolicyDecision:
        if self._snapshot_path is None:
            return self._default("no_policy")

        now_monotonic = time.monotonic()
        if (
            self._last_poll_monotonic is not None
            and now_monotonic - self._last_poll_monotonic < self._poll_interval_s
        ):
            return self._check_expiry_only()
        self._last_poll_monotonic = now_monotonic

        try:
            stat = self._snapshot_path.stat()
        except FileNotFoundError:
            self._last_stat = None
            return self._default("no_policy")
        except OSError:
            # PermissionError, transient I/O errors, etc. -- distinct from
            # "no policy configured": something is wrong reading a path that
            # otherwise exists, so treat it like a corrupt read (keep an
            # unexpired last-good if there is one) rather than silently
            # reverting to native/default as if no policy were intended.
            return self._reject_keep_last_or_default("corrupt")

        current_stat = (stat.st_mtime_ns, stat.st_size)
        if current_stat == self._last_stat:
            return self._check_expiry_only()

        self._last_stat = current_stat
        try:
            raw_text = self._snapshot_path.read_text()
            raw = json.loads(raw_text)
            snapshot = parse_policy_snapshot(raw)
            if (
                snapshot.model_id != self._model_id
                or snapshot.engine_id != self._engine_id
            ):
                raise PolicyEngineMismatchError(
                    f"snapshot identity {snapshot.engine_id!r}/{snapshot.model_id!r} "
                    f"does not match this scheduler "
                    f"{self._engine_id!r}/{self._model_id!r}"
                )
            if (
                self._last_accepted_generation is not None
                and snapshot.generation <= self._last_accepted_generation
            ):
                raise PolicyStaleError(
                    f"generation {snapshot.generation} <= "
                    f"last accepted {self._last_accepted_generation}"
                )
        except (OSError, json.JSONDecodeError, PolicyCorruptError):
            return self._reject_keep_last_or_default("corrupt")
        except PolicyEngineMismatchError:
            return self._reject_keep_last_or_default("rejected_engine_mismatch")
        except PolicyStaleError:
            return self._reject_keep_last_or_default("rejected_regression")

        self._last_accepted_generation = snapshot.generation
        self._last_accepted_policy_id = snapshot.policy_id
        self._last_accepted_expires_at = snapshot.expires_at
        self._last_accepted_max_num_seqs = (
            snapshot.max_num_seqs
            if snapshot.max_num_seqs is not None
            else self._startup_max_num_seqs
        )
        self._last_accepted_max_num_batched_tokens = (
            snapshot.max_num_batched_tokens
            if snapshot.max_num_batched_tokens is not None
            else self._startup_max_num_batched_tokens
        )
        self._last_decision = PolicyDecision(
            max_num_seqs=self._last_accepted_max_num_seqs,
            max_num_batched_tokens=self._last_accepted_max_num_batched_tokens,
            policy_id=self._last_accepted_policy_id,
            generation=self._last_accepted_generation,
            status="active",
            source="file",
        )
        return self._check_expiry_only()

    def _check_expiry_only(self) -> PolicyDecision:
        """Re-check expiry every call, independent of whether the file changed."""
        if self._last_accepted_expires_at is None:
            return self._last_decision
        if self._is_expired():
            return self._default("expired")
        return self._last_decision

    def _is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self._last_accepted_expires_at

    def _reject_keep_last_or_default(self, status: PolicyStatus) -> PolicyDecision:
        # A last-good policy is only worth keeping if it hasn't itself
        # expired -- otherwise a corrupt/mismatched/stale write arriving
        # after TTL would incorrectly resurrect an already-expired ceiling.
        # `_is_expired()` is only called once `_last_accepted_generation` is
        # known non-None, so `_last_accepted_expires_at` is guaranteed set.
        if self._last_accepted_generation is None or self._is_expired():
            decision = self._default(status)
        else:
            decision = PolicyDecision(
                max_num_seqs=self._last_accepted_max_num_seqs,
                max_num_batched_tokens=self._last_accepted_max_num_batched_tokens,
                policy_id=self._last_accepted_policy_id,
                generation=self._last_accepted_generation,
                status=status,
                source="file",
            )
        self._last_decision = decision
        return decision

    def _default(self, status: PolicyStatus) -> PolicyDecision:
        decision = _default_decision(
            self._startup_max_num_seqs, self._startup_max_num_batched_tokens, status
        )
        self._last_decision = decision
        return decision
