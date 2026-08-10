# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for prefix-retention allocation decision receipts."""

import asyncio
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.core.prefix_retention_observer as prefix_retention_observer
from vllm.distributed.kv_events import BlockRemoved
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    make_block_hash_with_group_id,
)
from vllm.v1.core.prefix_retention import (
    PrefixRetentionPolicy,
    PrefixRetentionTracker,
)
from vllm.v1.core.prefix_retention_observer import (
    PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION,
    PrefixRetentionBlockCategory,
    PrefixRetentionCompletedAccessReceipt,
    PrefixRetentionHashPreimage,
    PrefixRetentionReceiptBatch,
    PrefixRetentionResetReceipt,
)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    EngineCoreClient,
    InprocClient,
    MPClient,
    SyncMPClient,
)
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.cpu_test


def _key(raw: bytes, group_id: int) -> BlockHashWithGroupId:
    return make_block_hash_with_group_id(BlockHash(raw), group_id)


def _cache(pool: BlockPool, block_id: int, key: BlockHashWithGroupId) -> None:
    block = pool.blocks[block_id]
    block.block_hash = key
    pool.cached_block_hash_to_block.insert(key, block)


def _receipts(pool: BlockPool):
    batch = pool.take_prefix_retention_receipts()
    assert isinstance(batch, PrefixRetentionReceiptBatch)
    return batch.receipts


def test_inproc_drain_forwards_without_serialization():
    expected = prefix_retention_observer.disabled_prefix_retention_receipt_batch()
    client = object.__new__(InprocClient)
    client.engine_core = Mock()
    client.engine_core.take_prefix_retention_receipts.return_value = expected

    assert client.take_prefix_retention_receipts() is expected
    client.engine_core.take_prefix_retention_receipts.assert_called_once_with()

    engine = object.__new__(LLMEngine)
    engine.engine_core = client
    assert engine.take_prefix_retention_receipts() is expected


@pytest.mark.parametrize(
    "method",
    [
        EngineCoreClient.take_prefix_retention_receipts,
        MPClient.take_prefix_retention_receipts,
        EngineCoreProc.take_prefix_retention_receipts,
    ],
)
def test_non_inproc_receipt_drain_is_rejected(method):
    with pytest.raises(ValueError, match="prefix_retention_receipts_inproc_only"):
        method(Mock())


def test_multiprocess_utility_bypasses_are_rejected_before_transport():
    with pytest.raises(ValueError, match="prefix_retention_receipts_inproc_only"):
        SyncMPClient.call_utility(Mock(), "take_prefix_retention_receipts")

    async def invoke_async_bypass() -> None:
        with pytest.raises(
            ValueError, match="prefix_retention_receipts_inproc_only"
        ):
            await AsyncMPClient.call_utility_async(
                Mock(), "take_prefix_retention_receipts"
            )

    asyncio.run(invoke_async_bypass())


def test_reset_receipt_is_immutable_and_part_of_the_observer_union():
    receipt = PrefixRetentionResetReceipt(
        schema_version=PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION,
        reset_running_requests=False,
        connector_reset_requested=True,
        local_cache_reset=True,
        tracker_history_reset=True,
        connector_reset_successful=True,
        tracker_generation_before=2,
        tracker_generation_after=3,
        reset_successful=True,
    )

    assert receipt.reset_successful is True
    with pytest.raises(FrozenInstanceError):
        receipt.reset_successful = False  # type: ignore[misc]


def _full_attention_manager(pool: BlockPool) -> FullAttentionManager:
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return FullAttentionManager(
        spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=4,
    )


def test_real_single_type_allocations_preserve_request_identity():
    pool = BlockPool(
        num_gpu_blocks=6,
        enable_caching=True,
        hash_block_size=4,
        enable_prefix_retention_observer=True,
    )
    manager = _full_attention_manager(pool)

    manager.allocate_new_blocks("direct-request", 4, 4)
    manager.allocate_new_computed_blocks("external-request", (), 0, 4)

    receipts = _receipts(pool)
    assert [receipt.request_id for receipt in receipts] == [
        "direct-request",
        "external-request",
    ]


def test_lru_records_real_victims_with_group_aware_hashes_and_is_immutable():
    pool = BlockPool(
        num_gpu_blocks=6,
        enable_caching=True,
        hash_block_size=4,
        enable_kv_cache_events=True,
        enable_prefix_retention_observer=True,
    )
    first = _key(b"same", 1)
    second = _key(b"same", 2)
    _cache(pool, 1, first)
    _cache(pool, 2, second)

    selected = pool.get_new_blocks(3, request_id="lru-request")
    receipts = _receipts(pool)
    removed = [event for event in pool.take_events() if isinstance(event, BlockRemoved)]

    assert [block.block_id for block in selected] == [1, 2, 3]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.schema_version == PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION
    assert receipt.request_id == "lru-request"
    assert receipt.allocation_ordinal == 1
    assert receipt.policy == "lru"
    assert receipt.budget_blocks == 0
    assert receipt.protected_hashes_hex == ()
    assert [item.block_id for item in receipt.candidates] == [1, 2, 3, 4, 5]
    assert [item.block_id for item in receipt.selected] == [1, 2, 3]
    assert [item.block_id for item in receipt.victims] == [1, 2]
    assert [item.block_hash_hex for item in receipt.victims] == [
        bytes(first).hex(),
        bytes(second).hex(),
    ]
    assert [item.group_id for item in receipt.victims] == [1, 2]
    assert all(
        item.block_hash_hex is None
        or (
            item.block_hash_hex == item.block_hash_hex.lower()
            and bytes.fromhex(item.block_hash_hex)
        )
        for item in (*receipt.candidates, *receipt.selected, *receipt.victims)
    )
    assert receipt.free_blocks_before == 5
    assert receipt.free_blocks_after == 2
    assert receipt.non_null_used_blocks_after == 3
    assert len(removed) == len(receipt.victims) == 2
    assert _receipts(pool) == ()
    with pytest.raises(FrozenInstanceError):
        receipt.request_id = "mutated"  # type: ignore[misc]


def test_protection_receipt_freezes_candidate_classes_before_selection():
    tracker = PrefixRetentionTracker(
        policy=PrefixRetentionPolicy.RECURPLAN,
        budget_blocks=1,
        hash_block_size=4,
    )
    pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=4,
        enable_kv_cache_events=True,
        prefix_retention_tracker=tracker,
        enable_prefix_retention_observer=True,
    )
    protected = _key(b"protected", 7)
    unprotected = _key(b"unprotected", 8)
    _cache(pool, 1, protected)
    _cache(pool, 2, unprotected)

    selected = pool.get_new_blocks(
        3,
        protected_hashes=frozenset({protected}),
        request_id="protected-request",
    )
    (receipt,) = _receipts(pool)

    assert [block.block_id for block in selected] == [3, 4, 2]
    assert receipt.policy == "recurplan"
    assert receipt.budget_blocks == 1
    assert receipt.tracker_generation == 0
    assert receipt.completed_ordinal == 0
    assert receipt.protected_hashes_hex == (bytes(protected).hex(),)
    assert [item.category for item in receipt.candidates] == [
        PrefixRetentionBlockCategory.PROTECTED_CACHED,
        PrefixRetentionBlockCategory.UNPROTECTED_CACHED,
        PrefixRetentionBlockCategory.UNHASHED,
        PrefixRetentionBlockCategory.UNHASHED,
    ]
    assert [item.block_id for item in receipt.victims] == [2]


def test_arc_receipt_freezes_complete_adaptive_state():
    tracker = PrefixRetentionTracker(
        policy=PrefixRetentionPolicy.ARC,
        budget_blocks=2,
        hash_block_size=4,
    )
    pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=4,
        prefix_retention_tracker=tracker,
        enable_prefix_retention_observer=True,
    )
    first, second, third = (_key(raw, 0) for raw in (b"a", b"b", b"c"))
    for key in (first, second, third):
        assert tracker.register_chain([key])
    for key in (first, second, third, first):
        assert tracker.record_completed_access([key])
    for block_id, key in enumerate((first, second, third), start=1):
        _cache(pool, block_id, key)
    state = tracker.arc_state()
    protected = tracker.protected_hashes(
        tracker.snapshot(pool.get_resident_cached_hashes())
    )

    pool.get_new_blocks(1, protected_hashes=protected, request_id="arc-state")

    receipt = _receipts(pool)[0]
    assert receipt.arc_preimage.target_t1 == state.target_t1
    assert receipt.arc_preimage.t1_hashes_hex == tuple(
        bytes(key).hex() for key in state.t1_lru_to_mru
    )
    assert receipt.arc_preimage.t2_hashes_hex == tuple(
        bytes(key).hex() for key in state.t2_lru_to_mru
    )
    assert receipt.arc_preimage.b1_hashes_hex == tuple(
        bytes(key).hex() for key in state.b1_lru_to_mru
    )
    assert receipt.arc_preimage.b2_hashes_hex == tuple(
        bytes(key).hex() for key in state.b2_lru_to_mru
    )


def test_receipt_append_occurs_after_normal_eviction_mutations(monkeypatch):
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=4,
        enable_kv_cache_events=True,
        enable_prefix_retention_observer=True,
    )
    victim = _key(b"victim", 9)
    _cache(pool, 1, victim)
    observed: dict[str, object] = {}
    buffer = pool._prefix_retention_receipt_buffer
    append_unchecked = buffer._append_unchecked

    def inspect_then_append(receipt):
        block = pool.blocks[1]
        observed.update(
            block_hash=block.block_hash,
            ref_cnt=block.ref_cnt,
            still_cached=pool.cached_block_hash_to_block.get_one_block(victim),
            removed_events=tuple(
                event
                for event in pool.kv_event_queue
                if isinstance(event, BlockRemoved)
            ),
        )
        append_unchecked(receipt)

    monkeypatch.setattr(buffer, "_append_unchecked", inspect_then_append)

    pool.get_new_blocks(1, request_id="ordering")

    assert observed == {
        "block_hash": None,
        "ref_cnt": 1,
        "still_cached": None,
        "removed_events": tuple(pool.kv_event_queue),
    }
    assert len(_receipts(pool)) == 1


def test_append_failure_is_sticky_and_cannot_change_the_actual_victim(monkeypatch):
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=4,
        enable_kv_cache_events=True,
        enable_prefix_retention_observer=True,
    )
    victim = _key(b"failure-victim", 10)
    _cache(pool, 1, victim)
    buffer = pool._prefix_retention_receipt_buffer

    def fail_append(_receipt):
        raise RuntimeError("observer failed")

    monkeypatch.setattr(buffer, "_append_unchecked", fail_append)

    selected = pool.get_new_blocks(1, request_id="append-failure")

    assert [block.block_id for block in selected] == [1]
    assert pool.blocks[1].block_hash is None
    assert pool.cached_block_hash_to_block.get_one_block(victim) is None
    assert (
        len([event for event in pool.take_events() if isinstance(event, BlockRemoved)])
        == 1
    )
    assert pool.prefix_retention_observation_failed is True
    batch = pool.take_prefix_retention_receipts()
    assert batch.receipts == ()
    assert batch.observation_failed is True
    assert batch.append_failure_count == 1
    assert batch.dropped_count == 1
    assert pool.prefix_retention_observation_failed is True


def test_observer_return_value_cannot_replace_lru_selection(monkeypatch):
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=4,
        enable_kv_cache_events=True,
        enable_prefix_retention_observer=True,
    )
    first = _key(b"first", 1)
    second = _key(b"second", 1)
    _cache(pool, 1, first)
    _cache(pool, 2, second)
    buffer = pool._prefix_retention_receipt_buffer

    monkeypatch.setattr(buffer, "append", lambda _receipt: pool.blocks[2])

    selected = pool.get_new_blocks(1, request_id="malicious-return")

    assert [block.block_id for block in selected] == [1]
    assert pool.blocks[1].block_hash is None
    assert pool.blocks[2].block_hash == second


def test_observer_is_disabled_by_default_without_queue_snapshot(monkeypatch):
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=4)
    get_all_free_blocks = Mock(wraps=pool.free_block_queue.get_all_free_blocks)
    capture_preimage = Mock(side_effect=AssertionError("observer must stay disabled"))
    monkeypatch.setattr(
        pool.free_block_queue,
        "get_all_free_blocks",
        get_all_free_blocks,
    )
    monkeypatch.setattr(
        pool,
        "_capture_prefix_retention_preimage",
        capture_preimage,
    )

    assert [block.block_id for block in pool.get_new_blocks(1)] == [1]

    get_all_free_blocks.assert_not_called()
    capture_preimage.assert_not_called()
    batch = pool.take_prefix_retention_receipts()
    assert batch.observer_enabled is False
    assert batch.capacity == 0
    assert batch.receipts == ()
    assert pool.prefix_retention_observation_failed is False


def test_preimage_failure_is_sticky_and_preserves_exact_lru_victim(monkeypatch):
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=4,
        enable_kv_cache_events=True,
        enable_prefix_retention_observer=True,
    )
    first = _key(b"preimage-first", 1)
    second = _key(b"preimage-second", 1)
    _cache(pool, 1, first)
    _cache(pool, 2, second)

    def fail_preimage(*_args, **_kwargs):
        raise RuntimeError("preimage failed")

    monkeypatch.setattr(pool, "_capture_prefix_retention_preimage", fail_preimage)

    selected = pool.get_new_blocks(1, request_id="preimage-failure")

    assert [block.block_id for block in selected] == [1]
    assert pool.blocks[1].block_hash is None
    assert pool.blocks[2].block_hash == second
    assert pool.prefix_retention_observation_failed is True
    batch = pool.take_prefix_retention_receipts()
    assert batch.observer_enabled is True
    assert batch.capacity == 256
    assert batch.receipts == ()
    assert batch.observation_failed is True
    assert batch.dropped_count == 1


def test_bounded_buffer_overflow_is_sticky_and_never_blocks_allocation():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=4,
        enable_prefix_retention_observer=True,
        prefix_retention_observer_capacity=1,
    )

    assert [block.block_id for block in pool.get_new_blocks(1)] == [1]
    assert [block.block_id for block in pool.get_new_blocks(1)] == [2]

    assert pool.prefix_retention_observation_failed is True
    batch = pool.take_prefix_retention_receipts()
    assert batch.observation_failed is True
    assert batch.capacity == 1
    assert batch.overflow_count == 1
    assert batch.append_failure_count == 0
    assert batch.dropped_count == 1
    assert len(batch.receipts) == 1
    assert batch.receipts[0].allocation_ordinal == 1
    after = pool.take_prefix_retention_receipts()
    assert after.receipts == ()
    assert after.observation_failed is True
    assert after.overflow_count == 1


def test_decision_receipt_freezes_recomputable_tracker_metadata():
    tracker = PrefixRetentionTracker(
        policy=PrefixRetentionPolicy.RECURPLAN,
        budget_blocks=2,
        hash_block_size=4,
    )
    root = _key(b"metadata-root", 11)
    child = _key(b"metadata-child", 11)
    assert tracker.register_chain((root, child))
    assert tracker.record_completed_access((root, child))
    assert tracker.record_completed_access((root,))
    assert tracker.record_completed_access((root, child))
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=4,
        prefix_retention_tracker=tracker,
        enable_prefix_retention_observer=True,
    )
    _cache(pool, 1, root)
    _cache(pool, 2, child)

    pool.get_new_blocks(
        1,
        protected_hashes=frozenset({root, child}),
        request_id="metadata",
    )
    (receipt,) = _receipts(pool)

    assert [item.block_hash_hex for item in receipt.tracker_metadata_preimage] == [
        bytes(child).hex(),
        bytes(root).hex(),
    ]
    metadata = {item.block_hash_hex: item for item in receipt.tracker_metadata_preimage}
    assert metadata[bytes(root).hex()].parent_hash_hex is None
    assert metadata[bytes(root).hex()].group_id == 11
    assert metadata[bytes(root).hex()].completed_count == 3
    assert metadata[bytes(root).hex()].last_access_ordinal == 3
    assert metadata[bytes(root).hex()].last_four_gaps == (1, 1)
    assert metadata[bytes(child).hex()].parent_hash_hex == bytes(root).hex()
    assert metadata[bytes(child).hex()].parent_group_id == 11
    assert metadata[bytes(child).hex()].completed_count == 2
    assert metadata[bytes(child).hex()].last_four_gaps == (2,)


def test_successful_completed_access_records_ordered_chain_after_tracker_update():
    tracker = PrefixRetentionTracker(
        policy=PrefixRetentionPolicy.LFU,
        budget_blocks=2,
        hash_block_size=4,
    )
    root = _key(b"completed-root", 12)
    child = _key(b"completed-child", 12)
    assert tracker.register_chain((root, child))
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=4,
        prefix_retention_tracker=tracker,
        enable_prefix_retention_observer=True,
    )

    assert pool.record_completed_prefix_access("completed-request", (root, child))
    (receipt,) = _receipts(pool)

    assert isinstance(receipt, PrefixRetentionCompletedAccessReceipt)
    assert receipt.schema_version == PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION
    assert receipt.request_id == "completed-request"
    assert receipt.tracker_generation == 0
    assert receipt.completed_ordinal == 1
    assert receipt.ordered_hash_chain == (
        PrefixRetentionHashPreimage(bytes(root).hex(), 12),
        PrefixRetentionHashPreimage(bytes(child).hex(), 12),
    )
    assert tracker.completed_ordinal == 1


def test_failed_completed_access_does_not_emit_a_receipt():
    tracker = PrefixRetentionTracker(
        policy=PrefixRetentionPolicy.LFU,
        budget_blocks=1,
        hash_block_size=4,
    )
    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=4,
        prefix_retention_tracker=tracker,
        enable_prefix_retention_observer=True,
    )

    assert not pool.record_completed_prefix_access("missing", (_key(b"missing", 0),))

    assert tracker.completed_ordinal == 0
    assert _receipts(pool) == ()


def test_hot_path_freezes_raw_preimages_without_json_or_sha(monkeypatch):
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=4,
        enable_prefix_retention_observer=True,
    )
    victim = _key(b"no-hot-digest", 13)
    _cache(pool, 1, victim)

    def forbid_json(*_args, **_kwargs):
        raise AssertionError("JSON serialization is forbidden on the hot path")

    monkeypatch.setattr(prefix_retention_observer.json, "dumps", forbid_json)

    selected = pool.get_new_blocks(1, request_id="no-hot-digest")
    (receipt,) = _receipts(pool)

    assert [block.block_id for block in selected] == [1]
    assert [item.block_id for item in receipt.victims] == [1]
    assert pool.prefix_retention_observation_failed is False
    assert pool._prefix_retention_receipt_buffer.capacity == 256


def test_selected_preimage_failure_cannot_interrupt_eviction_mutation(monkeypatch):
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=4,
        enable_kv_cache_events=True,
        enable_prefix_retention_observer=True,
    )
    victim = _key(b"selected-map-failure", 15)
    _cache(pool, 1, victim)

    def fail_selected_mapping(*_args, **_kwargs):
        raise KeyError("selected preimage missing")

    monkeypatch.setattr(
        pool,
        "_select_prefix_retention_preimages",
        fail_selected_mapping,
        raising=False,
    )

    selected = pool.get_new_blocks(1, request_id="selected-map-failure")

    assert [block.block_id for block in selected] == [1]
    assert pool.blocks[1].block_hash is None
    assert pool.blocks[1].ref_cnt == 1
    assert pool.cached_block_hash_to_block.get_one_block(victim) is None
    assert (
        len([event for event in pool.take_events() if isinstance(event, BlockRemoved)])
        == 1
    )
    assert pool.prefix_retention_observation_failed is True
    batch = pool.take_prefix_retention_receipts()
    assert batch.receipts == ()
    assert batch.observation_failed is True
    assert batch.dropped_count == 1


def test_one_physical_instance_is_protected_for_each_logical_key():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=4,
        enable_kv_cache_events=True,
        enable_prefix_retention_observer=True,
    )
    duplicate = _key(b"duplicate-physical", 14)
    _cache(pool, 1, duplicate)
    _cache(pool, 2, duplicate)

    selected = pool.get_new_blocks(
        2,
        protected_hashes=frozenset({duplicate}),
        request_id="duplicate-physical",
    )
    (receipt,) = _receipts(pool)
    removed = [event for event in pool.take_events() if isinstance(event, BlockRemoved)]

    assert [block.block_id for block in selected] == [3, 1]
    assert receipt.protected_hashes_hex == (bytes(duplicate).hex(),)
    protected_instances = [
        item
        for item in receipt.candidates
        if item.category is PrefixRetentionBlockCategory.PROTECTED_CACHED
    ]
    assert [item.block_id for item in protected_instances] == [2]
    assert [item.group_id for item in protected_instances] == [14]
    assert [item.block_hash_hex for item in protected_instances] == [
        bytes(duplicate).hex(),
    ]
    assert [item.block_id for item in receipt.victims] == [1]
    assert len(removed) == len(receipt.victims) == 1
    assert pool.cached_block_hash_to_block.get_one_block(duplicate) is pool.blocks[2]
