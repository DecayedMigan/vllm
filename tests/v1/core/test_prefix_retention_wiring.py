# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.core.sched.scheduler as scheduler_module
from vllm.config import CacheConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.prefix_retention import PrefixRetentionPolicy
from vllm.v1.core.prefix_retention_observer import (
    PrefixRetentionCompletedAccessReceipt,
    PrefixRetentionDecisionReceipt,
    PrefixRetentionResetReceipt,
    PrefixRetentionRuntimeConfigReceipt,
)
from vllm.v1.core.sched.scheduler import (
    Scheduler,
    _create_prefix_retention_tracker,
)
from vllm.v1.kv_cache_interface import (
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

from .utils import create_requests

pytestmark = pytest.mark.cpu_test


def test_prefix_retention_cli_args():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    defaults = EngineArgs.from_cli_args(parser.parse_args([]))
    configured = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--prefix-retention-policy",
                "recurplan",
                "--prefix-retention-budget-blocks",
                "7",
                "--enable-prefix-retention-observer",
                "--prefix-retention-observer-capacity",
                "19",
            ]
        )
    )
    arc = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--prefix-retention-policy",
                "arc",
                "--prefix-retention-budget-blocks",
                "7",
            ]
        )
    )

    assert defaults.prefix_retention_policy == "lru"
    assert defaults.prefix_retention_budget_blocks == 0
    assert defaults.enable_prefix_retention_observer is False
    assert defaults.prefix_retention_observer_capacity == 256
    assert configured.prefix_retention_policy == "recurplan"
    assert configured.prefix_retention_budget_blocks == 7
    assert configured.enable_prefix_retention_observer is True
    assert configured.prefix_retention_observer_capacity == 19
    assert arc.prefix_retention_policy == "arc"
    assert arc.prefix_retention_budget_blocks == 7


def _make_gate_inputs(
    *,
    policy: str = "recurplan",
    budget: int = 1,
    num_blocks: int = 8,
):
    cache_config = SimpleNamespace(
        prefix_retention_policy=policy,
        prefix_retention_budget_blocks=budget,
        enable_prefix_caching=True,
        is_attention_free=False,
        sliding_window=None,
        kv_offloading_size=None,
        block_size=4,
        num_gpu_blocks=num_blocks,
    )
    model_config = SimpleNamespace(
        is_attention_free=False,
        is_hybrid=False,
        is_encoder_decoder=False,
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        nnodes=1,
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=parallel_config,
        kv_transfer_config=None,
        speculative_config=None,
    )
    group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
        is_eagle_group=False,
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[group],
    )
    return vllm_config, kv_cache_config


def _make_real_scheduler_inputs(
    *,
    policy: str = "recurplan",
    budget: int = 2,
    enable_prefix_caching: bool = True,
    enable_observer: bool = False,
    observer_capacity: int = 256,
):
    num_blocks = 8
    cache_config = CacheConfig(
        block_size=4,
        enable_prefix_caching=enable_prefix_caching,
        prefix_retention_policy=policy,
        prefix_retention_budget_blocks=budget,
        enable_prefix_retention_observer=enable_observer,
        prefix_retention_observer_capacity=observer_capacity,
    )
    cache_config.num_gpu_blocks = num_blocks
    model_config = SimpleNamespace(
        is_attention_free=False,
        is_hybrid=False,
        is_encoder_decoder=False,
        max_model_len=32,
        enable_return_routed_experts=False,
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        nnodes=1,
        data_parallel_index=0,
    )
    scheduler_config = SimpleNamespace(
        max_num_seqs=1,
        max_num_scheduled_tokens=None,
        max_num_batched_tokens=32,
        policy="fcfs",
        scheduler_reserve_full_isl=False,
    )
    vllm_config = SimpleNamespace(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        lora_config=None,
        kv_events_config=None,
        parallel_config=parallel_config,
        observability_config=SimpleNamespace(
            kv_cache_metrics=None,
            kv_cache_metrics_sample=1.0,
            enable_mfu_metrics=False,
        ),
        model_config=model_config,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
        use_v2_model_runner=False,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return vllm_config, kv_cache_config


def _make_real_scheduler(monkeypatch, **input_kwargs) -> Scheduler:
    vllm_config, kv_cache_config = _make_real_scheduler_inputs(**input_kwargs)
    monkeypatch.setattr(
        scheduler_module.EventPublisherFactory,
        "create",
        Mock(return_value=Mock()),
    )
    mm_registry = Mock()
    mm_registry.supports_multimodal_inputs.return_value = False
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=Mock(),
        block_size=4,
        hash_block_size=4,
        mm_registry=mm_registry,
    )


@pytest.mark.parametrize(
    ("policy", "budget", "capability_mode"),
    [
        ("lru", 0, "lru_bypass"),
        ("prefix_recency", 2, "non_lru_envelope_passed"),
        ("arc", 2, "non_lru_envelope_passed"),
        ("recurplan", 2, "non_lru_envelope_passed"),
    ],
)
def test_runtime_config_receipt_is_first_and_uses_effective_objects(
    monkeypatch, policy: str, budget: int, capability_mode: str
):
    scheduler = _make_real_scheduler(
        monkeypatch,
        policy=policy,
        budget=budget,
        enable_observer=True,
        observer_capacity=19,
    )

    batch = scheduler.take_prefix_retention_receipts()

    assert len(batch.receipts) == 1
    receipt = batch.receipts[0]
    assert isinstance(receipt, PrefixRetentionRuntimeConfigReceipt)
    assert receipt.policy == scheduler.prefix_retention_tracker.policy.value
    assert receipt.budget_blocks == scheduler.prefix_retention_tracker.budget_blocks
    assert (
        receipt.hash_block_size
        == scheduler.kv_cache_manager.block_pool.hash_block_size
        == 4
    )
    assert (
        receipt.num_gpu_blocks
        == scheduler.kv_cache_manager.block_pool.num_gpu_blocks
        == 8
    )
    assert receipt.prefix_caching_enabled is True
    assert receipt.scheduler_block_size == 4
    assert receipt.num_kv_groups == 1
    assert receipt.capability_mode == capability_mode
    assert receipt.tracker_binding_verified is True
    assert receipt.observer_enabled is True
    assert receipt.observer_capacity == batch.capacity == 19


def test_disabled_runtime_config_receipt_returns_before_construction(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "PrefixRetentionRuntimeConfigReceipt",
        Mock(side_effect=AssertionError("disabled observer must not build receipt")),
        raising=False,
    )
    monkeypatch.setattr(
        BlockPool,
        "record_prefix_retention_runtime_config",
        Mock(side_effect=AssertionError("disabled observer must not append receipt")),
        raising=False,
    )

    scheduler = _make_real_scheduler(monkeypatch, enable_observer=False)

    batch = scheduler.take_prefix_retention_receipts()
    assert batch.observer_enabled is False
    assert batch.receipts == ()


@pytest.mark.parametrize(
    ("policy", "budget"),
    [
        ("lru", 0),
        ("prefix_recency", 2),
        ("lfu", 2),
        ("arc", 2),
        ("recurplan", 2),
    ],
)
def test_capability_gate_builds_requested_tracker(policy: str, budget: int):
    vllm_config, kv_cache_config = _make_gate_inputs(
        policy=policy,
        budget=budget,
    )

    tracker, hash_block_size = _create_prefix_retention_tracker(
        vllm_config,
        kv_cache_config,
        block_size=4,
        hash_block_size=None,
    )

    assert tracker.policy is PrefixRetentionPolicy(policy)
    assert tracker.budget_blocks == budget
    assert tracker.hash_block_size == hash_block_size == 4


def test_real_scheduler_wires_one_tracker_to_manager_and_pool(monkeypatch):
    vllm_config, kv_cache_config = _make_real_scheduler_inputs(
        enable_observer=True,
        observer_capacity=19,
    )
    event_publisher = Mock()
    monkeypatch.setattr(
        scheduler_module.EventPublisherFactory,
        "create",
        Mock(return_value=event_publisher),
    )
    mm_registry = Mock()
    mm_registry.supports_multimodal_inputs.return_value = False

    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=Mock(),
        block_size=4,
        hash_block_size=4,
        mm_registry=mm_registry,
    )

    tracker = scheduler.prefix_retention_tracker
    assert tracker.policy is PrefixRetentionPolicy.RECURPLAN
    assert tracker is scheduler.kv_cache_manager.prefix_retention_tracker
    assert tracker is scheduler.kv_cache_manager.block_pool.prefix_retention_tracker
    batch = scheduler.take_prefix_retention_receipts()
    assert batch.observer_enabled is True
    assert batch.capacity == 19

    local_state_before = scheduler.kv_cache_manager.prefix_retention_local_state()
    reset = scheduler.reset_prefix_cache_with_receipt(reset_connector=True)
    local_state_after = scheduler.kv_cache_manager.prefix_retention_local_state()
    assert isinstance(reset, PrefixRetentionResetReceipt)
    assert reset.attempt_ordinal == 1
    assert reset.reset_running_requests_requested is False
    assert reset.reset_connector_requested is True
    assert reset.running_requests_before == 0
    assert reset.preempted_requests == 0
    assert reset.running_requests_after == 0
    assert reset.local_reset_attempted is True
    assert reset.local_reset_succeeded is True
    assert reset.connector_configured is False
    assert reset.connector_reset_attempted is False
    assert reset.connector_reset_succeeded is True
    assert reset.overall_succeeded is True
    assert reset.reason_tokens == ()
    assert reset.local_state_before == local_state_before
    assert reset.local_state_after == local_state_after
    assert local_state_after.tracker_generation == (
        local_state_before.tracker_generation + 1
    )
    assert reset.all_blocks_cleared_emitted is False
    buffer = scheduler.kv_cache_manager.block_pool._prefix_retention_receipt_buffer
    assert buffer is not None
    buffer.mark_failed()
    reset_batch = scheduler.take_prefix_retention_receipts()
    assert reset_batch.receipts == (reset,)
    assert reset_batch.receipts[0] is reset
    assert reset_batch.observation_failed is True


def test_real_scheduler_receipt_stream_preserves_causal_request_identity(monkeypatch):
    scheduler = _make_real_scheduler(
        monkeypatch,
        enable_observer=True,
        observer_capacity=19,
    )
    runtime_batch = scheduler.take_prefix_retention_receipts()
    assert [type(item) for item in runtime_batch.receipts] == [
        PrefixRetentionRuntimeConfigReceipt
    ]

    assert scheduler.reset_prefix_cache() is True
    request = create_requests(
        1,
        num_tokens=8,
        max_tokens=1,
        block_size=4,
        req_ids=["formal-request-001"],
    )[0]
    manager = scheduler.kv_cache_manager
    allocated = manager.allocate_slots(request, request.num_tokens)
    assert allocated is not None
    assert manager.record_completed_request_access(request) is True
    manager.free(request)

    batch = scheduler.take_prefix_retention_receipts()
    assert [type(item) for item in batch.receipts] == [
        PrefixRetentionResetReceipt,
        PrefixRetentionDecisionReceipt,
        PrefixRetentionCompletedAccessReceipt,
    ]
    reset, decision, completed = batch.receipts
    assert reset.attempt_ordinal == 1
    assert decision.allocation_ordinal == 1
    assert completed.completed_ordinal == 1
    assert decision.request_id == request.request_id == "formal-request-001"
    assert completed.request_id == request.request_id


def test_reset_receipt_reports_local_failure_without_resetting_tracker(monkeypatch):
    vllm_config, kv_cache_config = _make_real_scheduler_inputs(enable_observer=True)
    monkeypatch.setattr(
        scheduler_module.EventPublisherFactory,
        "create",
        Mock(return_value=Mock()),
    )
    mm_registry = Mock()
    mm_registry.supports_multimodal_inputs.return_value = False
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=Mock(),
        block_size=4,
        hash_block_size=4,
        mm_registry=mm_registry,
    )
    scheduler.take_prefix_retention_receipts()
    scheduler.kv_cache_manager.block_pool.get_new_blocks(1)
    scheduler.take_prefix_retention_receipts()
    local_state_before = scheduler.kv_cache_manager.prefix_retention_local_state()

    reset = scheduler.reset_prefix_cache_with_receipt()

    assert isinstance(reset, PrefixRetentionResetReceipt)
    assert reset.attempt_ordinal == 1
    assert reset.local_reset_attempted is True
    assert reset.local_reset_succeeded is False
    assert reset.reset_connector_requested is False
    assert reset.connector_configured is False
    assert reset.connector_reset_attempted is False
    assert reset.connector_reset_succeeded is None
    assert reset.overall_succeeded is False
    assert reset.reason_tokens == ("local_blocks_in_use",)
    assert reset.local_state_before == local_state_before
    assert reset.local_state_after == local_state_before
    assert reset.all_blocks_cleared_emitted is False
    reset_batch = scheduler.take_prefix_retention_receipts()
    assert reset_batch.receipts == (reset,)
    assert reset_batch.receipts[0] is reset


def test_reset_receipt_reports_connector_failure(monkeypatch):
    vllm_config, kv_cache_config = _make_real_scheduler_inputs(enable_observer=True)
    monkeypatch.setattr(
        scheduler_module.EventPublisherFactory,
        "create",
        Mock(return_value=Mock()),
    )
    mm_registry = Mock()
    mm_registry.supports_multimodal_inputs.return_value = False
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=Mock(),
        block_size=4,
        hash_block_size=4,
        mm_registry=mm_registry,
    )
    scheduler.take_prefix_retention_receipts()
    scheduler.connector = Mock()
    scheduler.connector.reset_cache.return_value = False
    local_state_before = scheduler.kv_cache_manager.prefix_retention_local_state()

    reset = scheduler.reset_prefix_cache_with_receipt(reset_connector=True)

    local_state_after = scheduler.kv_cache_manager.prefix_retention_local_state()
    assert isinstance(reset, PrefixRetentionResetReceipt)
    assert reset.attempt_ordinal == 1
    assert reset.local_reset_attempted is True
    assert reset.local_reset_succeeded is True
    assert reset.reset_connector_requested is True
    assert reset.connector_configured is True
    assert reset.connector_reset_attempted is True
    assert reset.connector_reset_succeeded is False
    assert reset.overall_succeeded is False
    assert reset.reason_tokens == ("connector_reset_failed",)
    assert reset.local_state_before == local_state_before
    assert reset.local_state_after == local_state_after
    assert local_state_after.tracker_generation == (
        local_state_before.tracker_generation + 1
    )
    assert reset.all_blocks_cleared_emitted is False
    reset_batch = scheduler.take_prefix_retention_receipts()
    assert reset_batch.receipts == (reset,)
    assert reset_batch.receipts[0] is reset


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_cache_config_rejects_invalid_observer_capacity(capacity):
    with pytest.raises(ValueError, match="invalid_observer_capacity"):
        CacheConfig(
            enable_prefix_retention_observer=True,
            prefix_retention_observer_capacity=capacity,
        )


def test_invalid_capability_emits_no_runtime_receipt_before_side_effects(monkeypatch):
    vllm_config, kv_cache_config = _make_real_scheduler_inputs(
        enable_prefix_caching=False
    )
    tracker_factory = Mock()
    connector_factory = Mock()
    event_factory = Mock()
    ec_factory = Mock()
    manager_factory = Mock()
    receipt_factory = Mock()
    monkeypatch.setattr(scheduler_module, "PrefixRetentionTracker", tracker_factory)
    monkeypatch.setattr(
        scheduler_module.KVConnectorFactory,
        "create_connector",
        connector_factory,
    )
    monkeypatch.setattr(
        scheduler_module.EventPublisherFactory,
        "create",
        event_factory,
    )
    monkeypatch.setattr(
        scheduler_module.ECConnectorFactory,
        "create_connector",
        ec_factory,
    )
    monkeypatch.setattr(scheduler_module, "KVCacheManager", manager_factory)
    monkeypatch.setattr(
        scheduler_module,
        "PrefixRetentionRuntimeConfigReceipt",
        receipt_factory,
        raising=False,
    )

    with pytest.raises(ValueError, match="prefix_caching_disabled"):
        Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=Mock(),
            block_size=4,
            hash_block_size=4,
            mm_registry=Mock(),
        )

    tracker_factory.assert_not_called()
    connector_factory.assert_not_called()
    event_factory.assert_not_called()
    ec_factory.assert_not_called()
    manager_factory.assert_not_called()
    receipt_factory.assert_not_called()


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("prefix_disabled", "prefix_caching_disabled"),
        ("attention_free", "attention_free"),
        ("hybrid", "hybrid_model"),
        ("encoder_decoder", "encoder_decoder"),
        ("multiple_groups", "multiple_kv_groups"),
        ("non_full", "non_full_attention"),
        ("block_mask", "block_mask_capable"),
        ("geometry", "block_geometry_mismatch"),
        ("connector", "kv_connector_or_offload"),
        ("speculative", "speculative_decode"),
        ("distributed", "distributed_execution"),
        ("budget", "invalid_budget"),
    ],
)
def test_capability_gate_fails_closed(case: str, reason: str):
    vllm_config, kv_cache_config = _make_gate_inputs()
    cache = vllm_config.cache_config
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    if case == "prefix_disabled":
        cache.enable_prefix_caching = False
    elif case == "attention_free":
        model.is_attention_free = True
    elif case == "hybrid":
        model.is_hybrid = True
    elif case == "encoder_decoder":
        model.is_encoder_decoder = True
    elif case == "multiple_groups":
        kv_cache_config.kv_cache_groups *= 2
    elif case == "non_full":
        kv_cache_config.kv_cache_groups[0].kv_cache_spec = object()
    elif case == "block_mask":
        cache.sliding_window = 8
    elif case == "geometry":
        cache.block_size = 8
    elif case == "connector":
        vllm_config.kv_transfer_config = object()
    elif case == "speculative":
        vllm_config.speculative_config = object()
    elif case == "distributed":
        parallel.tensor_parallel_size = 2
    elif case == "budget":
        cache.prefix_retention_budget_blocks = kv_cache_config.num_blocks
    else:  # pragma: no cover
        raise AssertionError(case)

    with pytest.raises(ValueError, match=reason):
        _create_prefix_retention_tracker(
            vllm_config,
            kv_cache_config,
            block_size=4,
            hash_block_size=4,
        )


@pytest.mark.parametrize(
    "spec",
    [
        SlidingWindowSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=8,
        ),
        MambaSpec(
            block_size=4,
            shapes=(1, 1),
            dtypes=(torch.float32,),
        ),
        CrossAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    ],
    ids=("sliding_window", "mamba", "cross_attention"),
)
def test_capability_gate_rejects_real_non_full_attention_specs(spec):
    vllm_config, kv_cache_config = _make_gate_inputs()
    kv_cache_config.kv_cache_groups[0].kv_cache_spec = spec

    with pytest.raises(ValueError, match="non_full_attention"):
        _create_prefix_retention_tracker(
            vllm_config,
            kv_cache_config,
            block_size=4,
            hash_block_size=4,
        )


@pytest.mark.parametrize("field", ("sliding_window", "attention_chunk_size"))
def test_capability_gate_rejects_real_block_mask_modes(field: str):
    vllm_config, kv_cache_config = _make_gate_inputs()
    kwargs = {field: 8}
    kv_cache_config.kv_cache_groups[0].kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        **kwargs,
    )

    with pytest.raises(ValueError, match="block_mask_capable"):
        _create_prefix_retention_tracker(
            vllm_config,
            kv_cache_config,
            block_size=4,
            hash_block_size=4,
        )


@pytest.mark.parametrize("mode", ("connector", "offload"))
def test_capability_gate_rejects_each_external_kv_path(mode: str):
    vllm_config, kv_cache_config = _make_gate_inputs()
    if mode == "connector":
        vllm_config.kv_transfer_config = object()
    else:
        vllm_config.cache_config.kv_offloading_size = 1.0

    with pytest.raises(ValueError, match="kv_connector_or_offload"):
        _create_prefix_retention_tracker(
            vllm_config,
            kv_cache_config,
            block_size=4,
            hash_block_size=4,
        )


@pytest.mark.parametrize(
    "field",
    (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "nnodes",
    ),
)
def test_capability_gate_rejects_each_distributed_degree(field: str):
    vllm_config, kv_cache_config = _make_gate_inputs()
    setattr(vllm_config.parallel_config, field, 2)

    with pytest.raises(ValueError, match="distributed_execution"):
        _create_prefix_retention_tracker(
            vllm_config,
            kv_cache_config,
            block_size=4,
            hash_block_size=4,
        )


@pytest.mark.parametrize("case", ("spec", "cache", "scheduler", "hash"))
def test_capability_gate_rejects_each_block_geometry_mismatch(case: str):
    vllm_config, kv_cache_config = _make_gate_inputs()
    scheduler_block_size = 4
    hash_block_size = 4
    if case == "spec":
        kv_cache_config.kv_cache_groups[0].kv_cache_spec = FullAttentionSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        )
    elif case == "cache":
        vllm_config.cache_config.block_size = 8
    elif case == "scheduler":
        scheduler_block_size = 8
    elif case == "hash":
        hash_block_size = 8

    with pytest.raises(ValueError, match="block_geometry_mismatch"):
        _create_prefix_retention_tracker(
            vllm_config,
            kv_cache_config,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )


@pytest.mark.parametrize(
    "case",
    ("actual_none", "actual_one", "configured_none", "mismatch", "capacity"),
)
def test_capability_gate_rejects_each_invalid_budget_boundary(case: str):
    vllm_config, kv_cache_config = _make_gate_inputs()
    if case == "actual_none":
        kv_cache_config.num_blocks = None
    elif case == "actual_one":
        kv_cache_config.num_blocks = 1
        vllm_config.cache_config.num_gpu_blocks = 1
    elif case == "configured_none":
        vllm_config.cache_config.num_gpu_blocks = None
    elif case == "mismatch":
        vllm_config.cache_config.num_gpu_blocks = 7
    elif case == "capacity":
        vllm_config.cache_config.prefix_retention_budget_blocks = 8

    with pytest.raises(ValueError, match="invalid_budget"):
        _create_prefix_retention_tracker(
            vllm_config,
            kv_cache_config,
            block_size=4,
            hash_block_size=4,
        )


def test_capability_gate_rejects_eagle_group_without_speculative_config():
    vllm_config, kv_cache_config = _make_gate_inputs()
    kv_cache_config.kv_cache_groups[0].is_eagle_group = True

    with pytest.raises(ValueError, match="speculative_decode"):
        _create_prefix_retention_tracker(
            vllm_config,
            kv_cache_config,
            block_size=4,
            hash_block_size=4,
        )


def test_lru_bypasses_experimental_capability_envelope():
    vllm_config, kv_cache_config = _make_gate_inputs(policy="lru", budget=0)
    vllm_config.cache_config.enable_prefix_caching = False
    vllm_config.model_config.is_hybrid = True
    vllm_config.parallel_config.tensor_parallel_size = 2
    kv_cache_config.kv_cache_groups *= 2

    tracker, _ = _create_prefix_retention_tracker(
        vllm_config,
        kv_cache_config,
        block_size=4,
        hash_block_size=4,
    )

    assert tracker.policy is PrefixRetentionPolicy.LRU


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefix_retention_policy": "lru", "prefix_retention_budget_blocks": 1},
        {
            "prefix_retention_policy": "recurplan",
            "prefix_retention_budget_blocks": 0,
        },
        {
            "prefix_retention_policy": "recurplan",
            "prefix_retention_budget_blocks": -1,
        },
    ],
)
def test_cache_config_rejects_invalid_policy_budget_pairs(kwargs):
    with pytest.raises(ValueError, match="invalid_budget"):
        CacheConfig(**kwargs)
