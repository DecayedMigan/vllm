# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    make_block_hash_with_group_id,
)
from vllm.v1.core.prefix_retention import (
    PrefixRetentionPolicy,
    PrefixRetentionTracker,
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
