# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only receipts for prefix-retention allocation decisions."""

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION = "amd-kv-retention-observer-v3"


class PrefixRetentionBlockCategory(str, Enum):
    """Stable eviction-selection category captured before selection."""

    UNHASHED = "UNHASHED"
    UNPROTECTED_CACHED = "UNPROTECTED_CACHED"
    PROTECTED_CACHED = "PROTECTED_CACHED"


@dataclass(frozen=True, slots=True)
class PrefixRetentionBlockPreimage:
    """Immutable identity and queue position before allocator mutation."""

    queue_ordinal: int
    block_id: int
    block_hash_hex: str | None
    group_id: int | None
    category: PrefixRetentionBlockCategory


@dataclass(frozen=True, slots=True)
class PrefixRetentionHashPreimage:
    """Full group-aware hash plus an explicit verifier-friendly group id."""

    block_hash_hex: str
    group_id: int


@dataclass(frozen=True, slots=True)
class PrefixRetentionTrackerMetadataPreimage:
    """Recomputable tracker fact captured before allocation selection."""

    block_hash_hex: str
    group_id: int
    parent_hash_hex: str | None
    parent_group_id: int | None
    completed_count: int
    last_access_ordinal: int
    last_four_gaps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixRetentionARCPreimage:
    """Complete bounded ARC state before one allocation decision."""

    target_t1: int
    t1_hashes_hex: tuple[str, ...]
    t2_hashes_hex: tuple[str, ...]
    b1_hashes_hex: tuple[str, ...]
    b2_hashes_hex: tuple[str, ...]


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def digest_hash_preimage(hashes_hex: tuple[str, ...]) -> str:
    """Digest an already canonical tuple of full group-aware hashes."""
    return _digest(hashes_hex)


def digest_block_preimage(
    blocks: tuple[PrefixRetentionBlockPreimage, ...],
) -> str:
    """Digest immutable block preimages without using mutable block objects."""
    return _digest(tuple(asdict(block) for block in blocks))


def digest_tracker_metadata_preimage(
    metadata: tuple[PrefixRetentionTrackerMetadataPreimage, ...],
) -> str:
    return _digest(tuple(asdict(item) for item in metadata))


@dataclass(frozen=True, slots=True)
class PrefixRetentionDecisionReceipt:
    """Post-mutation observation of one real block allocation decision."""

    schema_version: str
    request_id: str | None
    allocation_ordinal: int
    policy: str
    budget_blocks: int
    tracker_generation: int
    completed_ordinal: int
    tracker_metadata_preimage: tuple[PrefixRetentionTrackerMetadataPreimage, ...]
    arc_preimage: PrefixRetentionARCPreimage
    protected_hashes_hex: tuple[str, ...]
    resident_hashes_hex: tuple[str, ...]
    candidates: tuple[PrefixRetentionBlockPreimage, ...]
    selected: tuple[PrefixRetentionBlockPreimage, ...]
    victims: tuple[PrefixRetentionBlockPreimage, ...]
    free_blocks_before: int
    free_blocks_after: int
    non_null_used_blocks_after: int


@dataclass(frozen=True, slots=True)
class PrefixRetentionCompletedAccessReceipt:
    """One successful completed-access update and its ordered hash chain."""

    schema_version: str
    request_id: str
    policy: str
    budget_blocks: int
    tracker_generation: int
    completed_ordinal: int
    ordered_hash_chain: tuple[PrefixRetentionHashPreimage, ...]


@dataclass(frozen=True, slots=True)
class PrefixRetentionRegisteredChainReceipt:
    """One successful tracker chain registration after cache publication."""

    schema_version: str
    request_id: str
    registration_ordinal: int
    policy: str
    budget_blocks: int
    tracker_generation: int
    completed_ordinal: int
    ordered_hash_chain: tuple[PrefixRetentionHashPreimage, ...]


@dataclass(frozen=True, slots=True)
class PrefixRetentionRuntimeConfigReceipt:
    schema_version: str
    policy: str
    budget_blocks: int
    prefix_caching_enabled: bool
    scheduler_block_size: int
    hash_block_size: int
    num_gpu_blocks: int
    num_kv_groups: int
    capability_mode: str
    tracker_binding_verified: bool
    observer_enabled: bool
    observer_capacity: int


@dataclass(frozen=True, slots=True)
class PrefixRetentionLocalState:
    non_null_used_blocks: int
    resident_key_count: int
    hashed_physical_block_count: int
    tracker_generation: int
    tracker_completed_ordinal: int
    tracker_metadata_count: int


@dataclass(frozen=True, slots=True)
class PrefixRetentionResetReceipt:
    """Outcome of one cache reset without changing the legacy bool contract."""

    schema_version: str
    attempt_ordinal: int
    reset_running_requests_requested: bool
    reset_connector_requested: bool
    running_requests_before: int
    preempted_requests: int
    running_requests_after: int
    local_reset_attempted: bool
    local_reset_succeeded: bool
    connector_configured: bool
    connector_reset_attempted: bool
    connector_reset_succeeded: bool | None
    overall_succeeded: bool
    reason_tokens: tuple[str, ...]
    local_state_before: PrefixRetentionLocalState | None
    local_state_after: PrefixRetentionLocalState | None
    all_blocks_cleared_emitted: bool


PrefixRetentionReceipt = (
    PrefixRetentionRuntimeConfigReceipt
    | PrefixRetentionDecisionReceipt
    | PrefixRetentionCompletedAccessReceipt
    | PrefixRetentionRegisteredChainReceipt
    | PrefixRetentionResetReceipt
)


@dataclass(frozen=True, slots=True)
class PrefixRetentionReceiptBatch:
    """Atomic drain result; failure state and counters remain sticky."""

    schema_version: str
    observer_enabled: bool
    receipts: tuple[PrefixRetentionReceipt, ...]
    observation_failed: bool
    capacity: int
    overflow_count: int
    append_failure_count: int
    dropped_count: int


class PrefixRetentionReceiptBuffer:
    """Append-only internal receipt buffer with fail-closed observation state.

    The buffer has no callback and its append return value is always ignored by
    the allocator. Observation failure is sticky across drains so a consumer
    cannot mistake an incomplete stream for valid evidence.
    """

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("prefix retention receipt capacity must be positive")
        self._capacity = capacity
        self._receipts: list[PrefixRetentionReceipt] = []
        self._observation_failed = False
        self._overflow_count = 0
        self._append_failure_count = 0
        self._dropped_count = 0

    @property
    def observation_failed(self) -> bool:
        return self._observation_failed

    @property
    def capacity(self) -> int:
        return self._capacity

    def mark_failed(self) -> None:
        self._observation_failed = True
        self._dropped_count += 1

    def append(self, receipt: PrefixRetentionReceipt) -> None:
        if len(self._receipts) >= self._capacity:
            self._overflow_count += 1
            self.mark_failed()
            return
        try:
            self._append_unchecked(receipt)
        except Exception:
            self._append_failure_count += 1
            self.mark_failed()

    def _append_unchecked(self, receipt: PrefixRetentionReceipt) -> None:
        self._receipts.append(receipt)

    def drain(self) -> PrefixRetentionReceiptBatch:
        receipts = tuple(self._receipts)
        self._receipts.clear()
        return PrefixRetentionReceiptBatch(
            schema_version=PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION,
            observer_enabled=True,
            receipts=receipts,
            observation_failed=self._observation_failed,
            capacity=self._capacity,
            overflow_count=self._overflow_count,
            append_failure_count=self._append_failure_count,
            dropped_count=self._dropped_count,
        )


def disabled_prefix_retention_receipt_batch() -> PrefixRetentionReceiptBatch:
    return PrefixRetentionReceiptBatch(
        schema_version=PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION,
        observer_enabled=False,
        receipts=(),
        observation_failed=False,
        capacity=0,
        overflow_count=0,
        append_failure_count=0,
        dropped_count=0,
    )
