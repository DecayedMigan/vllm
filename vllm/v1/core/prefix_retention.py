# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure policy state for advisory prefix-cache retention."""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from heapq import heappop, heappush

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
class _DescendingRatio:
    numerator: int
    denominator: int

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _DescendingRatio):
            return NotImplemented
        return self.numerator * other.denominator > (
            other.numerator * self.denominator
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _DescendingRatio):
            return NotImplemented
        return self.numerator * other.denominator == (
            other.numerator * self.denominator
        )


@dataclass(frozen=True)
class _PrefixMetadata:
    block_hash: BlockHashWithGroupId
    parent: BlockHashWithGroupId | None
    completed_count: int
    last_access_ordinal: int
    gaps: tuple[int, ...]


@dataclass
class _PrefixRetentionSelectionStats:
    """Test and benchmark counters for one shared selection decision."""

    closure_constructions: int = 0
    candidate_evaluations: int = 0
    mask_operations: int = 0


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
        return self._protected_hashes(snapshot)

    def _protected_hashes_with_stats(
        self, snapshot: PrefixRetentionSnapshot
    ) -> tuple[
        frozenset[BlockHashWithGroupId], _PrefixRetentionSelectionStats
    ]:
        """Run the public selection path with deterministic work counters."""
        stats = _PrefixRetentionSelectionStats()
        return self._protected_hashes(snapshot, stats=stats), stats

    def _protected_hashes(
        self,
        snapshot: PrefixRetentionSnapshot,
        *,
        stats: _PrefixRetentionSelectionStats | None = None,
    ) -> frozenset[BlockHashWithGroupId]:
        if (
            snapshot._generation != self._generation
            or self.policy is PrefixRetentionPolicy.LRU
            or self.budget_blocks == 0
        ):
            return frozenset()

        metadata = {item.block_hash: item for item in snapshot.metadata}
        resident = frozenset(snapshot.resident_hashes)
        bit_for = {
            key: 1 << index
            for index, key in enumerate(snapshot.resident_hashes)
        }
        closure_masks: dict[BlockHashWithGroupId, int] = {}
        known_masks: dict[BlockHashWithGroupId, int] = {}
        invalid_closures: set[BlockHashWithGroupId] = set()
        for terminal in snapshot.resident_hashes:
            if stats is not None:
                stats.closure_constructions += 1
            closure_mask = self._resident_closure_mask(
                terminal,
                metadata,
                resident,
                bit_for,
                known_masks,
                invalid_closures,
                stats,
            )
            if closure_mask is not None:
                closure_masks[terminal] = closure_mask

        return self._select_protected_masks(
            snapshot, metadata, bit_for, closure_masks, stats=stats
        )

    def _select_protected_masks(
        self,
        snapshot: PrefixRetentionSnapshot,
        metadata: dict[BlockHashWithGroupId, _PrefixMetadata],
        bit_for: dict[BlockHashWithGroupId, int],
        closure_masks: dict[BlockHashWithGroupId, int],
        *,
        stats: _PrefixRetentionSelectionStats | None = None,
    ) -> frozenset[BlockHashWithGroupId]:
        """Greedily protect resident closures with bitmask marginal costs."""
        timing_scores = (
            {
                terminal: self._timing_score(
                    metadata[terminal], snapshot.completed_ordinal
                )
                for terminal in closure_masks
            }
            if self.policy is PrefixRetentionPolicy.RECURPLAN
            else None
        )
        static_priority: dict[
            BlockHashWithGroupId, tuple[object, ...] | None
        ] = {}
        for terminal in closure_masks:
            terminal_metadata = metadata[terminal]
            if self.policy is PrefixRetentionPolicy.PREFIX_RECENCY:
                static_priority[terminal] = (
                    -terminal_metadata.last_access_ordinal,
                )
            elif self.policy is PrefixRetentionPolicy.LFU:
                static_priority[terminal] = (
                    -terminal_metadata.completed_count,
                    -terminal_metadata.last_access_ordinal,
                )
            elif self.policy is PrefixRetentionPolicy.ARC:
                arc_ranking = self._arc_ranking(terminal, 1)
                static_priority[terminal] = (
                    arc_ranking[:2] if arc_ranking is not None else None
                )
            else:
                static_priority[terminal] = None
        static_priority_counts: dict[tuple[object, ...], int] = {}
        for priority in static_priority.values():
            if priority is not None:
                static_priority_counts[priority] = (
                    static_priority_counts.get(priority, 0) + 1
                )
        rerank_on_cost_change = {
            terminal: (
                priority is None or static_priority_counts[priority] > 1
            )
            for terminal, priority in static_priority.items()
        }
        terminal_depths = {
            terminal: closure_mask.bit_count()
            for terminal, closure_mask in closure_masks.items()
        }
        marginal_masks = dict(closure_masks)
        marginal_costs = dict(terminal_depths)
        terminals = tuple(closure_masks)
        dependent_terminal_masks = [0] * len(snapshot.resident_hashes)
        for terminal_index, terminal in enumerate(terminals):
            closure_mask = closure_masks[terminal]
            terminal_bit = 1 << terminal_index
            unindexed_mask = closure_mask
            while unindexed_mask:
                bit = unindexed_mask & -unindexed_mask
                resident_index = bit.bit_length() - 1
                dependent_terminal_masks[resident_index] |= terminal_bit
                unindexed_mask ^= bit
        if stats is not None:
            stats.mask_operations += len(closure_masks)

        def ranking_for(
            terminal: BlockHashWithGroupId,
        ) -> tuple[object, ...] | None:
            if stats is not None:
                stats.candidate_evaluations += 1
            return self._ranking(
                terminal,
                metadata[terminal],
                marginal_costs[terminal],
                terminal_depths[terminal],
                snapshot,
                timing_score=(
                    timing_scores[terminal]
                    if timing_scores is not None
                    else None
                ),
            )

        versions = {terminal: 0 for terminal in closure_masks}
        ranked: list[
            tuple[tuple[object, ...], int, BlockHashWithGroupId]
        ] = []
        for terminal in closure_masks:
            ranking = ranking_for(terminal)
            if ranking is not None:
                heappush(ranked, (ranking, versions[terminal], terminal))

        protected_mask = 0
        protected_count = 0
        considered: set[BlockHashWithGroupId] = set()
        while protected_count < self.budget_blocks:
            remaining = self.budget_blocks - protected_count
            best_terminal: BlockHashWithGroupId | None = None
            deferred: list[
                tuple[tuple[object, ...], int, BlockHashWithGroupId]
            ] = []
            while ranked:
                _, version, terminal = heappop(ranked)
                if version != versions[terminal] or terminal in considered:
                    continue
                if marginal_costs[terminal] == 0:
                    considered.add(terminal)
                    continue
                if marginal_costs[terminal] > remaining:
                    ranking = ranking_for(terminal)
                    if ranking is not None:
                        deferred.append((ranking, version, terminal))
                    continue
                best_terminal = terminal
                break
            for candidate in deferred:
                heappush(ranked, candidate)

            if best_terminal is None:
                break
            considered.add(best_terminal)
            best_marginal = marginal_masks[best_terminal]
            protected_mask |= best_marginal
            if stats is not None:
                stats.mask_operations += 1
            protected_count = protected_mask.bit_count()
            unprotected_bits = best_marginal
            affected_terminal_mask = 0
            while unprotected_bits:
                bit = unprotected_bits & -unprotected_bits
                resident_index = bit.bit_length() - 1
                affected_terminal_mask |= dependent_terminal_masks[
                    resident_index
                ]
                unprotected_bits ^= bit
            while affected_terminal_mask:
                terminal_bit = affected_terminal_mask & -affected_terminal_mask
                terminal = terminals[terminal_bit.bit_length() - 1]
                affected_terminal_mask ^= terminal_bit
                if terminal in considered:
                    continue
                newly_protected = marginal_masks[terminal] & best_marginal
                if not newly_protected:
                    continue
                marginal_masks[terminal] &= ~best_marginal
                marginal_costs[terminal] -= newly_protected.bit_count()
                if stats is not None:
                    stats.mask_operations += 1
                if marginal_costs[terminal] == 0:
                    versions[terminal] += 1
                    considered.add(terminal)
                    continue
                if rerank_on_cost_change[terminal]:
                    versions[terminal] += 1
                    ranking = ranking_for(terminal)
                    if ranking is not None:
                        heappush(ranked, (ranking, versions[terminal], terminal))

        return frozenset(
            key for key, bit in bit_for.items() if protected_mask & bit
        )

    @staticmethod
    def _resident_closure_mask(
        terminal: BlockHashWithGroupId,
        metadata: dict[BlockHashWithGroupId, _PrefixMetadata],
        resident: frozenset[BlockHashWithGroupId],
        bit_for: dict[BlockHashWithGroupId, int],
        known_masks: dict[BlockHashWithGroupId, int],
        invalid_closures: set[BlockHashWithGroupId],
        stats: _PrefixRetentionSelectionStats | None,
    ) -> int | None:
        path: list[BlockHashWithGroupId] = []
        path_keys: set[BlockHashWithGroupId] = set()
        key: BlockHashWithGroupId | None = terminal
        closure_mask = 0
        while key is not None:
            if key in invalid_closures:
                invalid_closures.update(path)
                return None
            known_mask = known_masks.get(key)
            if known_mask is not None:
                closure_mask = known_mask
                break
            if key in path_keys or key not in resident:
                invalid_closures.update(path)
                return None
            item = metadata.get(key)
            if item is None:
                invalid_closures.update(path)
                return None
            path.append(key)
            path_keys.add(key)
            key = item.parent

        for path_key in reversed(path):
            closure_mask |= bit_for[path_key]
            known_masks[path_key] = closure_mask
            if stats is not None:
                stats.mask_operations += 1
        return closure_mask

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
        *,
        timing_score: Fraction | None = None,
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

        if timing_score is None:
            arc_ranking = self._arc_ranking(terminal, marginal_cost)
            return None if arc_ranking is None else (1, *arc_ranking)
        descending_value = _DescendingRatio(
            numerator=(
                timing_score.numerator
                * terminal_depth
                * snapshot.hash_block_size
            ),
            denominator=timing_score.denominator * marginal_cost,
        )
        return (0, descending_value, marginal_cost, *canonical)

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
            twice_median = 2 * gaps[middle]
        else:
            twice_median = gaps[middle - 1] + gaps[middle]
        twice_dispersion = max(abs(2 * gap - twice_median) for gap in gaps)
        if twice_dispersion > max(2, twice_median):
            return None
        age = completed_ordinal - metadata.last_access_ordinal
        twice_uncertainty_radius = max(2, twice_dispersion)
        if 2 * age > twice_median + twice_uncertainty_radius:
            return None
        twice_denominator = max(twice_median, 2)
        score = Fraction(
            twice_denominator - abs(2 * age - twice_median),
            twice_denominator,
        )
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
