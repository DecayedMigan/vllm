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


def test_recurplan_does_not_select_fewer_than_three_completed_gaps():
    key = _hash(b"a")
    tracker = _tracker(PrefixRetentionPolicy.RECURPLAN, budget=1)
    assert tracker.register_chain([key])
    for _ in range(3):
        assert tracker.record_completed_access([key])

    assert tracker.protected_hashes(tracker.snapshot([key])) == frozenset()


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
