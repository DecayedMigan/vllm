# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure policy state for advisory prefix-cache retention."""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    get_block_hash,
    get_group_id,
)


class PrefixRetentionPolicy(str, Enum):
    LRU = "lru"
    PREFIX_RECENCY = "prefix_recency"
    LFU = "lfu"
    ARC = "arc"
    RECURPLAN = "recurplan"


@dataclass(frozen=True)
class _PrefixMetadata:
    block_hash: BlockHashWithGroupId
    parent: BlockHashWithGroupId | None
    completed_count: int
    last_access_ordinal: int
    gaps: tuple[int, ...]


@dataclass(frozen=True)
class PrefixRetentionSnapshot:
    resident_hashes: tuple[BlockHashWithGroupId, ...]
    completed_ordinal: int
    hash_block_size: int
    metadata: tuple[_PrefixMetadata, ...]
    _generation: int = 0


@dataclass(frozen=True)
class PrefixRetentionMetadataState:
    """Read-only tracker metadata for an independent observer."""

    block_hash: BlockHashWithGroupId
    parent: BlockHashWithGroupId | None
    completed_count: int
    last_access_ordinal: int
    gaps: tuple[int, ...]


@dataclass(frozen=True)
class PrefixRetentionObserverState:
    """Immutable tracker state without invoking the policy snapshot API."""

    generation: int
    completed_ordinal: int
    metadata: tuple[PrefixRetentionMetadataState, ...]
    arc_state: "PrefixRetentionARCState"


@dataclass(frozen=True)
class PrefixRetentionARCState:
    """Bounded adaptive-replacement metadata, ordered least to most recent."""

    target_t1: int
    t1_lru_to_mru: tuple[BlockHashWithGroupId, ...]
    t2_lru_to_mru: tuple[BlockHashWithGroupId, ...]
    b1_lru_to_mru: tuple[BlockHashWithGroupId, ...]
    b2_lru_to_mru: tuple[BlockHashWithGroupId, ...]


@dataclass
class _MutableMetadata:
    parent: BlockHashWithGroupId | None
    completed_count: int = 0
    last_access_ordinal: int = 0
    gaps: tuple[int, ...] = ()


def _canonical_key(key: BlockHashWithGroupId) -> tuple[bytes, int]:
    return bytes(get_block_hash(key)), get_group_id(key)


class PrefixRetentionTracker:
    """Track stable prefix identities without owning physical cache blocks."""

    def __init__(
        self,
        *,
        policy: PrefixRetentionPolicy,
        budget_blocks: int,
        hash_block_size: int,
    ) -> None:
        if budget_blocks < 0:
            raise ValueError("budget_blocks must be non-negative")
        if hash_block_size <= 0:
            raise ValueError("hash_block_size must be positive")
        self.policy = policy
        self.budget_blocks = budget_blocks
        self.hash_block_size = hash_block_size
        self.completed_ordinal = 0
        self._generation = 0
        self._metadata: dict[BlockHashWithGroupId, _MutableMetadata] = {}
        self._arc_target_t1 = 0
        self._arc_t1: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
        self._arc_t2: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
        self._arc_b1: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
        self._arc_b2: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()

    def register_chain(self, block_hashes: Sequence[BlockHashWithGroupId]) -> bool:
        if not block_hashes:
            return False

        staged: dict[BlockHashWithGroupId, _MutableMetadata] = {}
        parent: BlockHashWithGroupId | None = None
        for key in block_hashes:
            existing = self._metadata.get(key)
            if existing is None:
                existing = staged.get(key)
            if existing is not None:
                if existing.parent != parent:
                    return False
            else:
                staged[key] = _MutableMetadata(parent=parent)
            parent = key

        self._metadata.update(staged)
        return True

    def record_completed_access(
        self, block_hashes: Sequence[BlockHashWithGroupId]
    ) -> bool:
        if not block_hashes:
            return False

        distinct_hashes = tuple(dict.fromkeys(block_hashes))
        parent: BlockHashWithGroupId | None = None
        for key in distinct_hashes:
            metadata = self._metadata.get(key)
            if metadata is None or metadata.parent != parent:
                return False
            parent = key

        ordinal = self.completed_ordinal + 1
        for key in distinct_hashes:
            metadata = self._metadata[key]
            if metadata.last_access_ordinal:
                gap = ordinal - metadata.last_access_ordinal
                if gap > 0:
                    metadata.gaps = (*metadata.gaps, gap)[-4:]
            metadata.completed_count += 1
            metadata.last_access_ordinal = ordinal
            if self.policy in (
                PrefixRetentionPolicy.ARC,
                PrefixRetentionPolicy.RECURPLAN,
            ):
                self._arc_access(key)
        self.completed_ordinal = ordinal
        return True

    def arc_state(self) -> PrefixRetentionARCState:
        return PrefixRetentionARCState(
            target_t1=self._arc_target_t1,
            t1_lru_to_mru=tuple(self._arc_t1),
            t2_lru_to_mru=tuple(self._arc_t2),
            b1_lru_to_mru=tuple(self._arc_b1),
            b2_lru_to_mru=tuple(self._arc_b2),
        )

    def _arc_access(self, key: BlockHashWithGroupId) -> None:
        capacity = self.budget_blocks
        if capacity <= 0:
            return

        if key in self._arc_t1:
            del self._arc_t1[key]
            self._arc_t2[key] = None
        elif key in self._arc_t2:
            self._arc_t2.move_to_end(key)
        elif key in self._arc_b1:
            delta = max(1, len(self._arc_b2) // max(len(self._arc_b1), 1))
            self._arc_target_t1 = min(capacity, self._arc_target_t1 + delta)
            self._arc_replace(key)
            del self._arc_b1[key]
            self._arc_t2[key] = None
        elif key in self._arc_b2:
            delta = max(1, len(self._arc_b1) // max(len(self._arc_b2), 1))
            self._arc_target_t1 = max(0, self._arc_target_t1 - delta)
            self._arc_replace(key)
            del self._arc_b2[key]
            self._arc_t2[key] = None
        else:
            if len(self._arc_t1) + len(self._arc_b1) == capacity:
                if len(self._arc_t1) < capacity:
                    self._arc_b1.popitem(last=False)
                    self._arc_replace(key)
                else:
                    victim, _ = self._arc_t1.popitem(last=False)
                    self._arc_b1[victim] = None
            elif (
                len(self._arc_t1)
                + len(self._arc_t2)
                + len(self._arc_b1)
                + len(self._arc_b2)
                >= capacity
            ):
                total = (
                    len(self._arc_t1)
                    + len(self._arc_t2)
                    + len(self._arc_b1)
                    + len(self._arc_b2)
                )
                if total >= 2 * capacity and self._arc_b2:
                    self._arc_b2.popitem(last=False)
                self._arc_replace(key)
            self._arc_t1[key] = None
        self._trim_arc_state()

    def _arc_replace(self, incoming: BlockHashWithGroupId) -> None:
        choose_t1 = bool(self._arc_t1) and (
            len(self._arc_t1) > self._arc_target_t1
            or (incoming in self._arc_b2 and len(self._arc_t1) == self._arc_target_t1)
        )
        if choose_t1:
            victim, _ = self._arc_t1.popitem(last=False)
            self._arc_b1[victim] = None
        elif self._arc_t2:
            victim, _ = self._arc_t2.popitem(last=False)
            self._arc_b2[victim] = None

    def _trim_arc_state(self) -> None:
        capacity = self.budget_blocks
        while len(self._arc_t1) + len(self._arc_t2) > capacity:
            choose_t1 = bool(self._arc_t1) and (
                len(self._arc_t1) > self._arc_target_t1 or not self._arc_t2
            )
            if choose_t1:
                victim, _ = self._arc_t1.popitem(last=False)
                self._arc_b1[victim] = None
            elif self._arc_t2:
                victim, _ = self._arc_t2.popitem(last=False)
                self._arc_b2[victim] = None
            else:
                break
        self._trim_arc_ghosts()

    def _trim_arc_ghosts(self) -> None:
        capacity = self.budget_blocks
        while len(self._arc_b1) + len(self._arc_b2) > capacity:
            if len(self._arc_b1) > self._arc_target_t1:
                self._arc_b1.popitem(last=False)
            elif self._arc_b2:
                self._arc_b2.popitem(last=False)
            else:
                self._arc_b1.popitem(last=False)

    def snapshot(
        self, resident_hashes: Sequence[BlockHashWithGroupId]
    ) -> PrefixRetentionSnapshot:
        resident = tuple(sorted(set(resident_hashes), key=_canonical_key))
        metadata = tuple(
            _PrefixMetadata(
                block_hash=key,
                parent=value.parent,
                completed_count=value.completed_count,
                last_access_ordinal=value.last_access_ordinal,
                gaps=value.gaps,
            )
            for key, value in sorted(
                self._metadata.items(), key=lambda item: _canonical_key(item[0])
            )
        )
        return PrefixRetentionSnapshot(
            resident_hashes=resident,
            completed_ordinal=self.completed_ordinal,
            hash_block_size=self.hash_block_size,
            metadata=metadata,
            _generation=self._generation,
        )

    def observation_state(self) -> PrefixRetentionObserverState:
        """Freeze policy metadata without changing policy or allocation state."""
        metadata = tuple(
            PrefixRetentionMetadataState(
                block_hash=key,
                parent=value.parent,
                completed_count=value.completed_count,
                last_access_ordinal=value.last_access_ordinal,
                gaps=value.gaps,
            )
            for key, value in sorted(
                self._metadata.items(), key=lambda item: _canonical_key(item[0])
            )
        )
        return PrefixRetentionObserverState(
            generation=self._generation,
            completed_ordinal=self.completed_ordinal,
            metadata=metadata,
            arc_state=self.arc_state(),
        )

    def protected_hashes(
        self, snapshot: PrefixRetentionSnapshot
    ) -> frozenset[BlockHashWithGroupId]:
        if (
            snapshot._generation != self._generation
            or self.policy is PrefixRetentionPolicy.LRU
            or self.budget_blocks == 0
        ):
            return frozenset()

        metadata = {item.block_hash: item for item in snapshot.metadata}
        resident = frozenset(snapshot.resident_hashes)
        closures: dict[BlockHashWithGroupId, frozenset[BlockHashWithGroupId]] = {}
        for terminal in snapshot.resident_hashes:
            closure = self._resident_closure(terminal, metadata, resident)
            if closure is not None:
                closures[terminal] = closure

        protected: set[BlockHashWithGroupId] = set()
        considered: set[BlockHashWithGroupId] = set()
        while len(protected) < self.budget_blocks:
            remaining = self.budget_blocks - len(protected)
            ranked: list[
                tuple[
                    tuple[object, ...],
                    BlockHashWithGroupId,
                    frozenset[BlockHashWithGroupId],
                ]
            ] = []
            for terminal, closure in closures.items():
                if terminal in considered:
                    continue
                marginal = closure.difference(protected)
                cost = len(marginal)
                if cost == 0:
                    considered.add(terminal)
                    continue
                if cost > remaining:
                    continue
                ranking = self._ranking(
                    terminal,
                    metadata[terminal],
                    cost,
                    len(closure),
                    snapshot,
                )
                if ranking is not None:
                    ranked.append((ranking, terminal, marginal))

            if not ranked:
                break
            _, terminal, marginal = min(ranked, key=lambda item: item[0])
            considered.add(terminal)
            protected.update(marginal)

        return frozenset(protected)

    @staticmethod
    def _resident_closure(
        terminal: BlockHashWithGroupId,
        metadata: dict[BlockHashWithGroupId, _PrefixMetadata],
        resident: frozenset[BlockHashWithGroupId],
    ) -> frozenset[BlockHashWithGroupId] | None:
        closure: set[BlockHashWithGroupId] = set()
        key: BlockHashWithGroupId | None = terminal
        while key is not None:
            if key in closure or key not in resident:
                return None
            item = metadata.get(key)
            if item is None:
                return None
            closure.add(key)
            key = item.parent
        return frozenset(closure)

    def _ranking(
        self,
        terminal: BlockHashWithGroupId,
        metadata: _PrefixMetadata,
        marginal_cost: int,
        terminal_depth: int,
        snapshot: PrefixRetentionSnapshot,
    ) -> tuple[object, ...] | None:
        canonical = _canonical_key(terminal)
        if self.policy is PrefixRetentionPolicy.PREFIX_RECENCY:
            return (-metadata.last_access_ordinal, marginal_cost, *canonical)
        if self.policy is PrefixRetentionPolicy.LFU:
            return (
                -metadata.completed_count,
                -metadata.last_access_ordinal,
                marginal_cost,
                *canonical,
            )

        if self.policy is PrefixRetentionPolicy.ARC:
            return self._arc_ranking(terminal, marginal_cost)

        timing_score = self._timing_score(metadata, snapshot.completed_ordinal)
        if timing_score is None:
            arc_ranking = self._arc_ranking(terminal, marginal_cost)
            return None if arc_ranking is None else (1, *arc_ranking)
        value = timing_score * terminal_depth * snapshot.hash_block_size / marginal_cost
        return (0, -value, marginal_cost, *canonical)

    def _arc_ranking(
        self, terminal: BlockHashWithGroupId, marginal_cost: int
    ) -> tuple[object, ...] | None:
        canonical = _canonical_key(terminal)
        if terminal in self._arc_t2:
            recency = tuple(reversed(self._arc_t2)).index(terminal)
            within_target = recency < max(self.budget_blocks - self._arc_target_t1, 0)
            return (0 if within_target else 2, recency, marginal_cost, *canonical)
        if terminal in self._arc_t1:
            recency = tuple(reversed(self._arc_t1)).index(terminal)
            within_target = recency < self._arc_target_t1
            return (0 if within_target else 1, recency, marginal_cost, *canonical)
        return None

    @staticmethod
    def _timing_score(
        metadata: _PrefixMetadata, completed_ordinal: int
    ) -> Fraction | None:
        if len(metadata.gaps) < 3:
            return None
        gaps = sorted(metadata.gaps)
        middle = len(gaps) // 2
        if len(gaps) % 2:
            median = Fraction(gaps[middle])
        else:
            median = Fraction(gaps[middle - 1] + gaps[middle], 2)
        dispersion = max(abs(Fraction(gap) - median) for gap in gaps)
        if dispersion > max(Fraction(1), median):
            return None
        age = completed_ordinal - metadata.last_access_ordinal
        uncertainty_radius = max(Fraction(1), dispersion)
        if age > median + uncertainty_radius:
            return None
        score = Fraction(1) - abs(Fraction(age) - median) / max(median, Fraction(1))
        return score if score > 0 else None

    def reset(self) -> None:
        self._metadata.clear()
        self.completed_ordinal = 0
        self._arc_target_t1 = 0
        self._arc_t1.clear()
        self._arc_t2.clear()
        self._arc_b1.clear()
        self._arc_b2.clear()
        self._generation += 1
