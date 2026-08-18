# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure policy-fair shared prefix-retention selection.

Run this benchmark only in the frozen GPU environment:

    python benchmarks/benchmark_prefix_retention.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, replace

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    make_block_hash_with_group_id,
)
from vllm.v1.core.prefix_retention import (
    PrefixRetentionPolicy,
    PrefixRetentionSnapshot,
    PrefixRetentionTracker,
)

ANCESTOR_COUNT = 31
LEAF_COUNT = 4634
INVALID_RESIDENT_COUNT = 31
BUDGET_BLOCKS = 1174


@dataclass(frozen=True)
class PrefixRetentionBenchmarkFixture:
    tracker: PrefixRetentionTracker
    snapshot: PrefixRetentionSnapshot
    valid_terminal_count: int


def _key(label: str, group_id: int = 0) -> BlockHashWithGroupId:
    return make_block_hash_with_group_id(
        BlockHash(label.encode("ascii")), group_id
    )


def build_prefix_retention_fixture(
    policy: PrefixRetentionPolicy = PrefixRetentionPolicy.PREFIX_RECENCY,
) -> PrefixRetentionBenchmarkFixture:
    """Build 4696 residents with 4665 valid, shared-ancestor terminals."""
    tracker = PrefixRetentionTracker(
        policy=policy,
        budget_blocks=BUDGET_BLOCKS,
        hash_block_size=16,
    )
    ancestors = tuple(
        _key(f"ancestor-{index:02d}") for index in range(ANCESTOR_COUNT)
    )
    assert tracker.register_chain(ancestors)
    leaves = tuple(_key(f"leaf-{index:04d}") for index in range(LEAF_COUNT))
    for leaf in leaves:
        assert tracker.register_chain((*ancestors, leaf))
    invalid_residents = tuple(
        _key(f"invalid-{index:02d}") for index in range(INVALID_RESIDENT_COUNT)
    )
    residents = (*ancestors, *leaves, *invalid_residents)

    if policy is PrefixRetentionPolicy.PREFIX_RECENCY:
        leaf_ordinals = {
            leaf: index for index, leaf in enumerate(leaves, start=1)
        }
        snapshot = tracker.snapshot(residents)
        snapshot = replace(
            snapshot,
            completed_ordinal=LEAF_COUNT,
            metadata=tuple(
                replace(
                    item,
                    last_access_ordinal=leaf_ordinals.get(item.block_hash, 0),
                )
                for item in snapshot.metadata
            ),
        )
    else:
        for _ in range(4):
            for leaf in leaves:
                assert tracker.record_completed_access((*ancestors, leaf))
        snapshot = tracker.snapshot(residents)

    return PrefixRetentionBenchmarkFixture(
        tracker=tracker,
        snapshot=snapshot,
        valid_terminal_count=ANCESTOR_COUNT + LEAF_COUNT,
    )


def _legacy_lastseen_protected_hashes(
    tracker: PrefixRetentionTracker,
    snapshot: PrefixRetentionSnapshot,
) -> frozenset[BlockHashWithGroupId]:
    """Run the removed set-difference LastSeen loop for benchmark comparison."""
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
            ranking = tracker._ranking(
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


def _median_seconds(operation: Callable[[], object], repeats: int) -> float:
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def _peak_bytes(operation: Callable[[], object]) -> int:
    tracemalloc.start()
    operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _shared_report(
    policy: PrefixRetentionPolicy,
    repeats: int,
) -> dict[str, float | int | str]:
    fixture = build_prefix_retention_fixture(policy)
    protected, stats = fixture.tracker._protected_hashes_with_stats(
        fixture.snapshot
    )
    return {
        "policy": policy.value,
        "median_ms": _median_seconds(
            lambda: fixture.tracker.protected_hashes(fixture.snapshot), repeats
        )
        * 1000,
        "peak_bytes": _peak_bytes(
            lambda: fixture.tracker.protected_hashes(fixture.snapshot)
        ),
        "protected_hashes": len(protected),
        "candidate_evaluations": stats.candidate_evaluations,
        "closure_constructions": stats.closure_constructions,
        "mask_operations": stats.mask_operations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured samples per selector; report the median.",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    reports = [
        _shared_report(policy, args.repeats)
        for policy in (
            PrefixRetentionPolicy.PREFIX_RECENCY,
            PrefixRetentionPolicy.ARC,
            PrefixRetentionPolicy.RECURPLAN,
        )
    ]
    lastseen_fixture = build_prefix_retention_fixture(
        PrefixRetentionPolicy.PREFIX_RECENCY
    )
    shared_lastseen = reports[0]
    shared_lastseen_protected = lastseen_fixture.tracker.protected_hashes(
        lastseen_fixture.snapshot
    )
    legacy_lastseen_protected = _legacy_lastseen_protected_hashes(
        lastseen_fixture.tracker, lastseen_fixture.snapshot
    )
    if shared_lastseen_protected != legacy_lastseen_protected:
        raise AssertionError(
            "shared LastSeen selector diverged from the legacy set oracle"
        )
    legacy_lastseen_ms = (
        _median_seconds(
            lambda: _legacy_lastseen_protected_hashes(
                lastseen_fixture.tracker, lastseen_fixture.snapshot
            ),
            args.repeats,
        )
        * 1000
    )
    legacy_lastseen_peak = _peak_bytes(
        lambda: _legacy_lastseen_protected_hashes(
            lastseen_fixture.tracker, lastseen_fixture.snapshot
        )
    )
    print(
        json.dumps(
            {
                "fixture": {
                    "resident_count": 4696,
                    "valid_terminal_count": 4665,
                    "budget_blocks": BUDGET_BLOCKS,
                },
                "shared": reports,
                "legacy_lastseen": {
                    "median_ms": legacy_lastseen_ms,
                    "peak_bytes": legacy_lastseen_peak,
                },
                "lastseen_speedup": legacy_lastseen_ms
                / float(shared_lastseen["median_ms"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
