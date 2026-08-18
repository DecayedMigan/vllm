# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from fractions import Fraction

import pytest

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    make_block_hash_with_group_id,
)
from vllm.v1.core.prefix_retention import (
    PrefixRetentionPolicy,
    PrefixRetentionTracker,
    _canonical_key,
)

pytestmark = pytest.mark.cpu_test


def _hash(raw: bytes, group_id: int = 0) -> BlockHashWithGroupId:
    return make_block_hash_with_group_id(BlockHash(raw), group_id)


def _tracker(
    policy: PrefixRetentionPolicy = PrefixRetentionPolicy.PREFIX_RECENCY,
    budget: int = 8,
) -> PrefixRetentionTracker:
    return PrefixRetentionTracker(
        policy=policy,
        budget_blocks=budget,
        hash_block_size=4,
    )


def _legacy_protected_hashes(
    tracker: PrefixRetentionTracker,
    snapshot,
) -> frozenset[BlockHashWithGroupId]:
    """Minimal test-only oracle for the pre-shared set selection loop."""
    if (
        snapshot._generation != tracker._generation
        or tracker.policy is PrefixRetentionPolicy.LRU
        or tracker.budget_blocks == 0
    ):
        return frozenset()

    metadata = {item.block_hash: item for item in snapshot.metadata}
    resident = frozenset(snapshot.resident_hashes)
    closures = {
        terminal: closure
        for terminal in snapshot.resident_hashes
        if (
            closure := tracker._resident_closure(terminal, metadata, resident)
        )
        is not None
    }
    if tracker.policy is PrefixRetentionPolicy.RECURPLAN:
        return _legacy_recurplan_protected_hashes(
            tracker, snapshot, metadata, closures
        )
    ranking_cache: dict[
        tuple[BlockHashWithGroupId, int], tuple[object, ...] | None
    ] = {}
    protected: set[BlockHashWithGroupId] = set()
    considered: set[BlockHashWithGroupId] = set()
    while len(protected) < tracker.budget_blocks:
        remaining = tracker.budget_blocks - len(protected)
        ranked = []
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
            cache_key = (terminal, cost)
            if cache_key not in ranking_cache:
                ranking_cache[cache_key] = tracker._ranking(
                    terminal,
                    metadata[terminal],
                    cost,
                    len(closure),
                    snapshot,
                )
            ranking = ranking_cache[cache_key]
            if ranking is not None:
                ranked.append((ranking, terminal, marginal))

        if not ranked:
            break
        _, terminal, marginal = min(ranked, key=lambda item: item[0])
        considered.add(terminal)
        protected.update(marginal)

    return frozenset(protected)


def _legacy_recurplan_protected_hashes(
    tracker: PrefixRetentionTracker,
    snapshot,
    metadata,
    closures,
) -> frozenset[BlockHashWithGroupId]:
    """Test-only reference for the frozen RecurPlan bitmask selector."""
    bit_for = {
        key: 1 << index for index, key in enumerate(snapshot.resident_hashes)
    }
    closure_masks = {
        terminal: sum(bit_for[key] for key in closure)
        for terminal, closure in closures.items()
    }
    timing_scores = {
        terminal: _legacy_timing_score(
            metadata[terminal], snapshot.completed_ordinal
        )
        for terminal in closures
    }
    canonical = {terminal: _canonical_key(terminal) for terminal in closures}
    t1_recency = {
        key: index for index, key in enumerate(reversed(tracker._arc_t1))
    }
    t2_recency = {
        key: index for index, key in enumerate(reversed(tracker._arc_t2))
    }
    t1_target = tracker._arc_target_t1
    t2_target = max(tracker.budget_blocks - t1_target, 0)

    protected_mask = 0
    protected_count = 0
    considered: set[BlockHashWithGroupId] = set()
    while protected_count < tracker.budget_blocks:
        remaining = tracker.budget_blocks - protected_count
        best_terminal: BlockHashWithGroupId | None = None
        best_marginal = 0
        best_kind = 2
        best_numerator = 0
        best_denominator = 1
        best_tail: tuple[object, ...] = ()

        for terminal, closure_mask in closure_masks.items():
            if terminal in considered:
                continue
            marginal = closure_mask & ~protected_mask
            cost = marginal.bit_count()
            if cost == 0:
                considered.add(terminal)
                continue
            if cost > remaining:
                continue

            score = timing_scores[terminal]
            terminal_canonical = canonical[terminal]
            if score is not None:
                numerator = (
                    score.numerator
                    * len(closures[terminal])
                    * snapshot.hash_block_size
                )
                denominator = score.denominator * cost
                tail = (cost, *terminal_canonical)
                better = best_terminal is None or best_kind != 0
                if best_terminal is not None and best_kind == 0:
                    cross = numerator * best_denominator
                    best_cross = best_numerator * denominator
                    better = cross > best_cross or (
                        cross == best_cross and tail < best_tail
                    )
                if better:
                    best_terminal = terminal
                    best_marginal = marginal
                    best_kind = 0
                    best_numerator = numerator
                    best_denominator = denominator
                    best_tail = tail
                continue

            if terminal in t2_recency:
                recency = t2_recency[terminal]
                tier = 0 if recency < t2_target else 2
            elif terminal in t1_recency:
                recency = t1_recency[terminal]
                tier = 0 if recency < t1_target else 1
            else:
                continue
            tail = (tier, recency, cost, *terminal_canonical)
            if best_terminal is None or (best_kind == 1 and tail < best_tail):
                best_terminal = terminal
                best_marginal = marginal
                best_kind = 1
                best_tail = tail

        if best_terminal is None:
            break
        considered.add(best_terminal)
        protected_mask |= best_marginal
        protected_count = protected_mask.bit_count()

    return frozenset(
        key for key, bit in bit_for.items() if protected_mask & bit
    )


def _legacy_timing_score(metadata, completed_ordinal: int) -> Fraction | None:
    """Reference the frozen Fraction-based RecurPlan timing arithmetic."""
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
    score = Fraction(1) - abs(Fraction(age) - median) / max(
        median, Fraction(1)
    )
    return score if score > 0 else None


@pytest.mark.parametrize(
    ("gaps", "last_access_ordinal", "completed_ordinal"),
    (
        ((1, 1, 1), 3, 4),
        ((1, 2, 3, 4), 20, 22),
        ((1, 1, 7), 7, 8),
        ((2, 2, 2, 2), 8, 12),
        ((2, 2, 2), 4, 9),
    ),
)
def test_recurplan_timing_score_matches_fraction_reference(
    gaps: tuple[int, ...],
    last_access_ordinal: int,
    completed_ordinal: int,
):
    key = _hash(b"timing")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=1)
    assert tracker.register_chain([key])
    metadata = replace(
        tracker.snapshot([key]).metadata[0],
        gaps=gaps,
        last_access_ordinal=last_access_ordinal,
    )

    assert tracker._timing_score(
        metadata, completed_ordinal
    ) == _legacy_timing_score(metadata, completed_ordinal)


def test_register_chain_builds_ancestor_closure_and_rejects_conflicts_atomically():
    root, child, leaf = _hash(b"a"), _hash(b"b"), _hash(b"c")
    other_root, uncommitted = _hash(b"x"), _hash(b"y")
    tracker = _tracker(budget=3)

    assert tracker.register_chain([root, child, leaf])
    assert not tracker.register_chain([other_root, child, uncommitted])
    assert tracker.protected_hashes(tracker.snapshot([leaf, child, root])) == {
        root,
        child,
        leaf,
    }

    # The rejected call must not retain even its otherwise valid new root.
    assert not tracker.record_completed_access([other_root])
    assert tracker.snapshot([other_root, uncommitted]).completed_ordinal == 0


def test_shared_ancestor_uses_marginal_cost_and_never_protects_orphan_suffixes():
    root, left, right = _hash(b"a"), _hash(b"b"), _hash(b"c")
    tracker = _tracker(PrefixRetentionPolicy.LFU, budget=2)
    assert tracker.register_chain([root, left])
    assert tracker.register_chain([root, right])
    assert tracker.record_completed_access([root, left])
    assert tracker.record_completed_access([root, right])
    assert tracker.record_completed_access([root, right])

    assert tracker.protected_hashes(tracker.snapshot([root, left, right])) == {
        root,
        right,
    }
    assert tracker.protected_hashes(tracker.snapshot([left, right])) == frozenset()


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (0, frozenset()),
        (1, frozenset({_hash(b"a")})),
        (2, frozenset({_hash(b"a"), _hash(b"b")})),
    ],
)
def test_budget_counts_unique_hashes_and_skips_over_budget_candidates(
    budget: int, expected: frozenset[BlockHashWithGroupId]
):
    root, leaf = _hash(b"a"), _hash(b"b")
    tracker = _tracker(budget=budget)
    assert tracker.register_chain([root, leaf])

    assert tracker.protected_hashes(tracker.snapshot([root, leaf])) == expected


def test_lru_always_returns_empty_even_with_budget_and_history():
    key = _hash(b"a")
    tracker = _tracker(PrefixRetentionPolicy.LRU, budget=8)
    assert tracker.register_chain([key])
    assert tracker.record_completed_access([key])

    assert tracker.protected_hashes(tracker.snapshot([key])) == frozenset()


def test_lru_bypasses_the_shared_selector_even_with_a_positive_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    key = _hash(b"lru")
    tracker = _tracker(PrefixRetentionPolicy.LRU, budget=1)
    assert tracker.register_chain([key])
    assert tracker.record_completed_access([key])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("lru must bypass the shared selector")

    monkeypatch.setattr(
        tracker, "_select_protected_masks", fail_if_called
    )
    assert tracker.protected_hashes(tracker.snapshot([key])) == frozenset()


@pytest.mark.parametrize(
    "policy",
    (
        PrefixRetentionPolicy.PREFIX_RECENCY,
        PrefixRetentionPolicy.LFU,
        PrefixRetentionPolicy.ARC,
        PrefixRetentionPolicy.RECURPLAN,
    ),
)
def test_non_lru_policies_call_the_shared_bitmask_selector(
    monkeypatch: pytest.MonkeyPatch,
    policy: PrefixRetentionPolicy,
):
    """A policy bypassing the shared selector would reintroduce an unfair path."""
    key = _hash(b"shared-selector")
    tracker = _tracker(policy, budget=1)
    assert tracker.register_chain([key])
    assert tracker.record_completed_access([key])

    calls: list[PrefixRetentionPolicy] = []
    original = tracker._select_protected_masks

    def counted(*args, **kwargs):
        calls.append(policy)
        return original(*args, **kwargs)

    monkeypatch.setattr(tracker, "_select_protected_masks", counted)

    assert tracker.protected_hashes(tracker.snapshot([key])) == {key}
    assert calls == [policy]


@pytest.mark.parametrize(
    "policy",
    (
        PrefixRetentionPolicy.PREFIX_RECENCY,
        PrefixRetentionPolicy.LFU,
        PrefixRetentionPolicy.ARC,
        PrefixRetentionPolicy.RECURPLAN,
    ),
)
@pytest.mark.parametrize("budget", (0, 1, 2, 3, 4, 5))
def test_shared_selector_matches_legacy_set_oracle_for_shared_ancestors(
    policy: PrefixRetentionPolicy,
    budget: int,
):
    root = _hash(b"root", group_id=1)
    left = _hash(b"same-child", group_id=1)
    right = _hash(b"same-child", group_id=2)
    standalone = _hash(b"standalone", group_id=0)
    tracker = _tracker(policy, budget=budget)
    assert tracker.register_chain([root, left])
    assert tracker.register_chain([root, right])
    assert tracker.register_chain([standalone])

    for chain in (
        [root, left],
        [root, right],
        [standalone],
        [root, left],
        [root, right],
        [standalone],
        [root, left],
        [root, right],
        [standalone],
        [root, left],
    ):
        assert tracker.record_completed_access(chain)

    snapshot = tracker.snapshot([left, standalone, right, root, left])
    assert tracker.protected_hashes(snapshot) == _legacy_protected_hashes(
        tracker, snapshot
    )


@pytest.mark.parametrize(
    "policy",
    (
        PrefixRetentionPolicy.PREFIX_RECENCY,
        PrefixRetentionPolicy.LFU,
        PrefixRetentionPolicy.ARC,
        PrefixRetentionPolicy.RECURPLAN,
    ),
)
def test_shared_selector_matches_legacy_oracle_for_invalid_closures(
    policy: PrefixRetentionPolicy,
):
    root, leaf = _hash(b"root"), _hash(b"leaf")
    tracker = _tracker(policy, budget=2)
    assert tracker.register_chain([root, leaf])
    assert tracker.record_completed_access([root, leaf])

    non_resident_ancestor = tracker.snapshot([leaf])
    assert tracker.protected_hashes(
        non_resident_ancestor
    ) == _legacy_protected_hashes(tracker, non_resident_ancestor)
    missing_metadata = replace(
        tracker.snapshot([root, leaf]),
        metadata=tuple(
            item
            for item in tracker.snapshot([root, leaf]).metadata
            if item.block_hash != root
        ),
    )
    assert tracker.protected_hashes(
        missing_metadata
    ) == _legacy_protected_hashes(tracker, missing_metadata)


@pytest.mark.parametrize(
    "policy",
    (
        PrefixRetentionPolicy.PREFIX_RECENCY,
        PrefixRetentionPolicy.LFU,
    ),
)
def test_shared_selector_uses_canonical_ties_for_equally_ranked_terminals(
    policy: PrefixRetentionPolicy,
):
    earlier = _hash(b"a")
    later = _hash(b"b")
    tracker = _tracker(policy, budget=1)
    assert tracker.register_chain([later])
    assert tracker.register_chain([earlier])

    snapshot = tracker.snapshot([later, earlier])
    assert tracker.protected_hashes(snapshot) == {earlier}
    assert tracker.protected_hashes(snapshot) == _legacy_protected_hashes(
        tracker, snapshot
    )


def test_shared_selector_operation_stats_pin_mask_work():
    root, leaf = _hash(b"root"), _hash(b"leaf")
    tracker = _tracker(PrefixRetentionPolicy.PREFIX_RECENCY, budget=1)
    assert tracker.register_chain([root, leaf])

    protected, stats = tracker._protected_hashes_with_stats(
        tracker.snapshot([root, leaf])
    )

    assert protected == {root}
    assert stats.closure_constructions == 2
    assert stats.candidate_evaluations == 3
    assert stats.mask_operations == 6


def test_shared_selector_only_reranks_candidates_with_equal_fixed_priority():
    root, older, newer = _hash(b"root"), _hash(b"older"), _hash(b"newer")
    tracker = _tracker(PrefixRetentionPolicy.PREFIX_RECENCY, budget=2)
    assert tracker.register_chain([root, older])
    assert tracker.register_chain([root, newer])
    assert tracker.record_completed_access([root, older])
    assert tracker.record_completed_access([root, newer])

    protected, stats = tracker._protected_hashes_with_stats(
        tracker.snapshot([root, older, newer])
    )

    assert protected == {root, newer}
    assert stats.candidate_evaluations == 4


def test_shared_selector_batches_each_selected_marginal_before_reranking():
    root, shared = _hash(b"root"), _hash(b"shared")
    earlier, later = _hash(b"earlier"), _hash(b"later")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=3)
    assert tracker.register_chain([root, shared, earlier])
    assert tracker.register_chain([root, shared, later])

    # Only the leaves receive valid RecurPlan scores. Selecting `earlier`
    # protects two shared ancestors at once, so `later` must be reranked once
    # after the whole marginal update—not once for every changed bit.
    snapshot = tracker.snapshot([root, shared, earlier, later])
    snapshot = replace(
        snapshot,
        completed_ordinal=4,
        metadata=tuple(
            replace(
                item,
                completed_count=4,
                last_access_ordinal=3,
                gaps=(1, 1, 1),
            )
            if item.block_hash in {earlier, later}
            else item
            for item in snapshot.metadata
        ),
    )

    protected, stats = tracker._protected_hashes_with_stats(snapshot)

    assert protected == {root, shared, earlier}
    assert stats.candidate_evaluations == 5
    assert stats.mask_operations == 12


def test_selector_benchmark_fixture_preserves_the_required_scale():
    from benchmarks.benchmark_prefix_retention import (
        build_prefix_retention_fixture,
    )

    fixture = build_prefix_retention_fixture()

    assert len(fixture.snapshot.resident_hashes) == 4696
    assert fixture.valid_terminal_count == 4665
    assert fixture.tracker.budget_blocks == 1174


def test_candidate_snapshots_are_identical_for_every_policy():
    root, leaf = _hash(b"a"), _hash(b"b")
    snapshots = []
    for policy in PrefixRetentionPolicy:
        tracker = _tracker(policy)
        assert tracker.register_chain([root, leaf])
        assert tracker.record_completed_access([root, leaf])
        snapshots.append(tracker.snapshot([leaf, root, leaf]))

    assert all(snapshot == snapshots[0] for snapshot in snapshots[1:])
    assert snapshots[0].resident_hashes == (root, leaf)


def test_prefix_recency_prefers_the_latest_completed_access():
    older, newer = _hash(b"a"), _hash(b"b")
    tracker = _tracker(PrefixRetentionPolicy.PREFIX_RECENCY, budget=1)
    assert tracker.register_chain([older])
    assert tracker.register_chain([newer])
    assert tracker.record_completed_access([older])
    assert tracker.record_completed_access([newer])

    assert tracker.protected_hashes(tracker.snapshot([older, newer])) == {newer}


def test_prefix_recency_is_closure_aware_last_seen_without_period_features():
    old_root, old_leaf = _hash(b"old-root"), _hash(b"old-leaf")
    new_root, new_leaf = _hash(b"new-root"), _hash(b"new-leaf")
    tracker = _tracker(PrefixRetentionPolicy.PREFIX_RECENCY, budget=2)
    assert tracker.register_chain([old_root, old_leaf])
    assert tracker.register_chain([new_root, new_leaf])
    assert tracker.record_completed_access([old_root, old_leaf])
    assert tracker.record_completed_access([new_root, new_leaf])

    snapshot = tracker.snapshot([old_root, old_leaf, new_root, new_leaf])
    expected = frozenset({new_root, new_leaf})
    assert tracker.protected_hashes(snapshot) == expected

    altered = replace(
        snapshot,
        metadata=tuple(
            replace(item, gaps=(1, 1, 1, 1000)) for item in snapshot.metadata
        ),
    )
    assert tracker.protected_hashes(altered) == expected

    old_chain = frozenset({old_root, old_leaf})
    adversarial = replace(
        snapshot,
        metadata=tuple(
            replace(
                item,
                gaps=(1, 1, 1, 1)
                if item.block_hash in old_chain
                else (1, 1000, 1, 1000),
            )
            for item in snapshot.metadata
        ),
    )
    protected = tracker.protected_hashes(adversarial)
    assert protected == expected
    assert len(protected) == tracker.budget_blocks


def test_lfu_prefers_count_before_recency():
    frequent, recent = _hash(b"a"), _hash(b"b")
    tracker = _tracker(PrefixRetentionPolicy.LFU, budget=1)
    assert tracker.register_chain([frequent])
    assert tracker.register_chain([recent])
    assert tracker.record_completed_access([frequent])
    assert tracker.record_completed_access([frequent])
    assert tracker.record_completed_access([recent])

    assert tracker.protected_hashes(tracker.snapshot([frequent, recent])) == {frequent}


def test_arc_promotes_repeated_access_and_bounds_ghost_history():
    first, second = _hash(b"a"), _hash(b"b")
    tracker = _tracker(PrefixRetentionPolicy.ARC, budget=2)
    for key in (first, second):
        assert tracker.register_chain([key])

    assert tracker.record_completed_access([first])
    assert tracker.record_completed_access([second])
    assert tracker.record_completed_access([first])

    state = tracker.arc_state()
    assert state.t1_lru_to_mru == (second,)
    assert state.t2_lru_to_mru == (first,)
    assert state.b1_lru_to_mru == ()
    assert state.b2_lru_to_mru == ()
    assert len(state.t1_lru_to_mru) + len(state.t2_lru_to_mru) <= 2


def test_arc_ghost_hits_move_the_adaptive_partition_both_directions():
    a, b, c, d = (_hash(raw) for raw in (b"a", b"b", b"c", b"d"))
    tracker = _tracker(PrefixRetentionPolicy.ARC, budget=2)
    for key in (a, b, c, d):
        assert tracker.register_chain([key])

    for key in (a, b, c):
        assert tracker.record_completed_access([key])
    before_b1_hit = tracker.arc_state()
    assert a in before_b1_hit.b1_lru_to_mru
    assert before_b1_hit.target_t1 == 0

    assert tracker.record_completed_access([a])
    after_b1_hit = tracker.arc_state()
    assert after_b1_hit.target_t1 == 1
    assert a in after_b1_hit.t2_lru_to_mru

    assert tracker.record_completed_access([d])
    before_b2_hit = tracker.arc_state()
    assert a in before_b2_hit.b2_lru_to_mru
    assert tracker.record_completed_access([a])
    after_b2_hit = tracker.arc_state()
    assert after_b2_hit.target_t1 == 0
    assert a in after_b2_hit.t2_lru_to_mru

    resident = set(after_b2_hit.t1_lru_to_mru + after_b2_hit.t2_lru_to_mru)
    ghosts = set(after_b2_hit.b1_lru_to_mru + after_b2_hit.b2_lru_to_mru)
    assert resident.isdisjoint(ghosts)
    assert len(resident) <= 2
    assert len(ghosts) <= 2


def test_arc_resident_history_never_exceeds_budget_under_repeated_churn():
    tracker = _tracker(PrefixRetentionPolicy.ARC, budget=31)
    keys = tuple(_hash(f"key-{index:d}".encode()) for index in range(64))
    for key in keys:
        assert tracker.register_chain([key])

    sequence = keys[:31] + keys[31:62] + keys[:31] + keys[62:] + keys[31:62] + keys[:31]
    for key in sequence:
        assert tracker.record_completed_access([key])
        state = tracker.arc_state()
        resident = len(state.t1_lru_to_mru) + len(state.t2_lru_to_mru)
        ghosts = len(state.b1_lru_to_mru) + len(state.b2_lru_to_mru)
        assert resident <= 31
        assert ghosts <= 31


def test_arc_reset_and_victim_order_are_deterministic():
    a, b, c = (_hash(raw) for raw in (b"a", b"b", b"c"))
    tracker = _tracker(PrefixRetentionPolicy.ARC, budget=2)
    for key in (c, b, a):
        assert tracker.register_chain([key])
        assert tracker.record_completed_access([key])
    assert tracker.record_completed_access([a])

    expected = tracker.protected_hashes(tracker.snapshot([c, b, a]))
    assert expected == {a, b}
    assert tracker.protected_hashes(tracker.snapshot([a, b, c])) == expected

    tracker.reset()
    state = tracker.arc_state()
    assert state.t1_lru_to_mru == state.t2_lru_to_mru == ()
    assert state.b1_lru_to_mru == state.b2_lru_to_mru == ()
    assert state.target_t1 == 0


def test_completed_access_deduplicates_identities_before_atomic_validation():
    root, child, unknown = _hash(b"a"), _hash(b"b"), _hash(b"x")
    tracker = _tracker(PrefixRetentionPolicy.LFU, budget=2)
    assert tracker.register_chain([root, child])

    assert tracker.record_completed_access([root, child, child])
    snapshot = tracker.snapshot([root, child])
    metadata = {item.block_hash: item for item in snapshot.metadata}
    assert snapshot.completed_ordinal == 1
    assert metadata[root].completed_count == 1
    assert metadata[child].completed_count == 1

    assert not tracker.record_completed_access([root, unknown, child, child])
    assert tracker.snapshot([root, child]) == snapshot


def test_recurplan_uses_frozen_ordinal_gap_score():
    periodic, early = _hash(b"a"), _hash(b"b")
    filler = _hash(b"z")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=1)
    for key in (periodic, early, filler):
        assert tracker.register_chain([key])

    # periodic: ordinals 1, 3, 5, 7 => median gap 2, then age 2.
    # early: ordinals 2, 4, 6, 8 => median gap 2, then age 1.
    for key in (
        periodic,
        early,
        periodic,
        early,
        periodic,
        early,
        periodic,
        early,
        filler,
    ):
        assert tracker.record_completed_access([key])

    assert tracker.protected_hashes(tracker.snapshot([periodic, early])) == {periodic}


def test_recurplan_scores_each_terminal_once_per_decision(
    monkeypatch: pytest.MonkeyPatch,
):
    recurring = _hash(b"recurring")
    fillers = [_hash(f"filler-{index}".encode()) for index in range(20)]
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=8)
    for key in (recurring, *fillers):
        assert tracker.register_chain([key])
    for ordinal in range(1, 14):
        key = recurring if ordinal in {1, 5, 9, 13} else fillers[ordinal - 1]
        assert tracker.record_completed_access([key])

    calls = 0
    original = tracker._timing_score

    def counted(metadata, completed_ordinal):
        nonlocal calls
        calls += 1
        return original(metadata, completed_ordinal)

    monkeypatch.setattr(tracker, "_timing_score", counted)
    snapshot = tracker.snapshot([recurring, *fillers])
    tracker.protected_hashes(snapshot)

    assert calls == len(snapshot.resident_hashes)


def test_recurplan_protects_a_reliable_recurrence_before_it_is_due():
    recurring, filler = _hash(b"a"), _hash(b"z")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=1)
    for key in (recurring, filler):
        assert tracker.register_chain([key])
    for ordinal in range(1, 15):
        key = recurring if ordinal in {1, 5, 9, 13} else filler
        assert tracker.record_completed_access([key])

    assert tracker.protected_hashes(tracker.snapshot([recurring, filler])) == {
        recurring
    }


def test_recurplan_falls_back_to_arc_with_fewer_than_three_completed_gaps():
    key = _hash(b"a")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=1)
    assert tracker.register_chain([key])
    for _ in range(3):
        assert tracker.record_completed_access([key])

    assert tracker.protected_hashes(tracker.snapshot([key])) == {key}


def test_recurplan_rejects_drifting_and_stale_interval_predictions():
    changed, stale, filler = _hash(b"a"), _hash(b"b"), _hash(b"z")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=2)
    for key in (changed, stale, filler):
        assert tracker.register_chain([key])

    changed_ordinals = {1, 2, 11, 12}
    stale_ordinals = {3, 5, 7, 9}
    for ordinal in range(1, 20):
        if ordinal in changed_ordinals:
            key = changed
        elif ordinal in stale_ordinals:
            key = stale
        else:
            key = filler
        assert tracker.record_completed_access([key])

    snapshot = tracker.snapshot([changed, stale])
    metadata = {item.block_hash: item for item in snapshot.metadata}
    assert tracker._timing_score(metadata[changed], snapshot.completed_ordinal) is None
    assert tracker._timing_score(metadata[stale], snapshot.completed_ordinal) is None
    assert tracker.protected_hashes(snapshot)


def test_recurplan_even_median_is_deterministic():
    even_median, competitor, filler = _hash(b"a"), _hash(b"b"), _hash(b"z")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=1)
    for key in (even_median, competitor, filler):
        assert tracker.register_chain([key])

    # At ordinal 22, even_median has latest gaps 1, 2, 3, 4, hence median
    # 5/2 and age 2: score 4/5. competitor has gaps 1, 4, 7, median 4,
    # and age 3: score 3/4. Exact arithmetic therefore selects even_median.
    even_ordinals = {10, 11, 13, 16, 20}
    competitor_ordinals = {7, 8, 12, 19}
    for ordinal in range(1, 23):
        if ordinal in even_ordinals:
            key = even_median
        elif ordinal in competitor_ordinals:
            key = competitor
        else:
            key = filler
        assert tracker.record_completed_access([key])

    assert tracker.protected_hashes(tracker.snapshot([even_median, competitor])) == {
        even_median
    }


def _assert_recurplan_exact_tie_selects(
    expected: BlockHashWithGroupId, other: BlockHashWithGroupId
) -> None:
    root, filler = _hash(b"r"), _hash(b"z")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=2)
    assert tracker.register_chain([root, expected])
    assert tracker.register_chain([root, other])
    assert tracker.register_chain([filler])

    # expected: gaps 3, 3, 3 and age 3. other: gaps 1, 2, 1 and
    # age 1. Both timing scores are exactly one. The root is selected first;
    # both children then have equal value and equal marginal cost one.
    for key in (
        expected,
        filler,
        filler,
        expected,
        filler,
        filler,
        expected,
        other,
        other,
        expected,
        other,
        other,
        filler,
    ):
        chain = [key] if key == filler else [root, key]
        assert tracker.record_completed_access(chain)

    assert tracker.protected_hashes(tracker.snapshot([other, root, expected])) == {
        root,
        expected,
    }


def test_recurplan_exact_ties_use_raw_digest_then_group_id():
    _assert_recurplan_exact_tie_selects(_hash(b"a", 9), _hash(b"b", 0))
    _assert_recurplan_exact_tie_selects(_hash(b"c", 1), _hash(b"c", 2))


def test_recurplan_uses_only_the_latest_four_positive_gaps():
    windowed, competitor, filler = _hash(b"a"), _hash(b"b"), _hash(b"z")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=1)
    for key in (windowed, competitor, filler):
        assert tracker.register_chain([key])

    # windowed has five gaps 100, 1, 2, 3, 4. Keeping only the latest four
    # yields median 5/2 and score 4/5 at age 2. Keeping all five would yield
    # median 3 and score 2/3. competitor has score 3/4, so the winner locks
    # both the four-gap window and its resulting exact timing score.
    windowed_ordinals = {1, 101, 102, 104, 107, 111}
    competitor_ordinals = {98, 99, 103, 110}
    for ordinal in range(1, 114):
        if ordinal in windowed_ordinals:
            key = windowed
        elif ordinal in competitor_ordinals:
            key = competitor
        else:
            key = filler
        assert tracker.record_completed_access([key])

    snapshot = tracker.snapshot([windowed, competitor])
    metadata = {item.block_hash: item for item in snapshot.metadata}
    assert metadata[windowed].gaps == (1, 2, 3, 4)
    assert all(len(item.gaps) <= 4 for item in snapshot.metadata)
    assert tracker.protected_hashes(snapshot) == {windowed}


@pytest.mark.parametrize(
    "policy",
    [PrefixRetentionPolicy.PREFIX_RECENCY, PrefixRetentionPolicy.LFU],
)
def test_ties_use_raw_digest_then_group_id(policy: PrefixRetentionPolicy):
    raw_first = _hash(b"a", 9)
    raw_second = _hash(b"b", 0)
    same_raw_lower_group = _hash(b"c", 1)
    same_raw_higher_group = _hash(b"c", 2)

    raw_tracker = _tracker(policy, budget=1)
    group_tracker = _tracker(policy, budget=1)
    for tracker, keys in (
        (raw_tracker, (raw_second, raw_first)),
        (group_tracker, (same_raw_higher_group, same_raw_lower_group)),
    ):
        for key in keys:
            assert tracker.register_chain([key])

    assert raw_tracker.protected_hashes(
        raw_tracker.snapshot([raw_second, raw_first])
    ) == {raw_first}
    assert group_tracker.protected_hashes(
        group_tracker.snapshot([same_raw_higher_group, same_raw_lower_group])
    ) == {same_raw_lower_group}


def test_invalid_and_noncompleted_accesses_do_not_advance_history_and_reset_is_stale():
    root, leaf, unknown = _hash(b"a"), _hash(b"b"), _hash(b"x")
    tracker = _tracker(PrefixRetentionPolicy.LFU, budget=2)
    assert tracker.register_chain([root, leaf])

    before = tracker.snapshot([root, leaf])
    assert not tracker.record_completed_access([])
    assert not tracker.record_completed_access([root, unknown])
    assert not tracker.record_completed_access([leaf])
    # Snapshots/policy queries model lookup, aborted, preempted, and error paths:
    # without a completed-access call they must be observationally pure.
    tracker.protected_hashes(before)
    assert tracker.snapshot([root, leaf]) == before

    tracker.reset()
    assert tracker.protected_hashes(before) == frozenset()
    stale = tracker.snapshot([root, leaf])
    assert stale.completed_ordinal == 0
    assert tracker.protected_hashes(stale) == frozenset()
    assert not tracker.record_completed_access([root, leaf])
