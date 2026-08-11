# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.core.prefix_retention import PrefixRetentionTracker
from vllm.v1.core.prefix_retention_observer import (
    PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION,
    PrefixRetentionARCPreimage,
    PrefixRetentionBlockCategory,
    PrefixRetentionBlockPreimage,
    PrefixRetentionCompletedAccessReceipt,
    PrefixRetentionDecisionReceipt,
    PrefixRetentionHashPreimage,
    PrefixRetentionReceiptBatch,
    PrefixRetentionReceiptBuffer,
    PrefixRetentionResetReceipt,
    PrefixRetentionTrackerMetadataPreimage,
    disabled_prefix_retention_receipt_batch,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def resident_keys(self) -> tuple[BlockHashWithGroupId, ...]:
        """Return a read-only, unique snapshot of live group-aware keys."""
        return tuple(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
        prefix_retention_tracker: Optional stable-hash retention metadata tracker.
        enable_prefix_retention_observer: Enable bounded read-only decision receipts.
        prefix_retention_observer_capacity: Maximum undrained observer receipts.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        prefix_retention_tracker: PrefixRetentionTracker | None = None,
        enable_prefix_retention_observer: bool = False,
        prefix_retention_observer_capacity: int = 256,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector
        self.prefix_retention_tracker = prefix_retention_tracker
        self._prefix_retention_receipt_buffer = (
            PrefixRetentionReceiptBuffer(prefix_retention_observer_capacity)
            if enable_prefix_retention_observer
            else None
        )
        self._prefix_retention_allocation_ordinal = 0

    @property
    def prefix_retention_observation_failed(self) -> bool:
        buffer = self._prefix_retention_receipt_buffer
        return buffer.observation_failed if buffer is not None else False

    def take_prefix_retention_receipts(
        self,
    ) -> PrefixRetentionReceiptBatch:
        """Take and clear allocation receipts without resetting failure state."""
        buffer = self._prefix_retention_receipt_buffer
        return (
            buffer.drain()
            if buffer is not None
            else disabled_prefix_retention_receipt_batch()
        )

    def record_prefix_retention_reset(
        self, receipt: PrefixRetentionResetReceipt
    ) -> None:
        """Append a reset outcome when observation is enabled."""
        buffer = self._prefix_retention_receipt_buffer
        if buffer is not None:
            buffer.append(receipt)

    @staticmethod
    def _prefix_retention_preimage(
        block: KVCacheBlock,
        queue_ordinal: int,
        protected_block_ids: AbstractSet[int],
    ) -> PrefixRetentionBlockPreimage:
        block_hash = block.block_hash
        if block_hash is None:
            category = PrefixRetentionBlockCategory.UNHASHED
            block_hash_hex = None
            group_id = None
        elif block.block_id in protected_block_ids:
            category = PrefixRetentionBlockCategory.PROTECTED_CACHED
            block_hash_hex = bytes(block_hash).hex()
            group_id = get_group_id(block_hash)
        else:
            category = PrefixRetentionBlockCategory.UNPROTECTED_CACHED
            block_hash_hex = bytes(block_hash).hex()
            group_id = get_group_id(block_hash)
        return PrefixRetentionBlockPreimage(
            queue_ordinal=queue_ordinal,
            block_id=block.block_id,
            block_hash_hex=block_hash_hex,
            group_id=group_id,
            category=category,
        )

    def _physical_protected_block_ids(
        self,
        protected_hashes: AbstractSet[BlockHashWithGroupId],
    ) -> frozenset[int]:
        """Choose one deterministic physical representative per logical key.

        Duplicate cached copies remain ordinary eviction candidates. The most
        recent free-queue instance is retained for each protected key, so the
        physical protection count can never exceed the logical key budget.
        """
        representatives: dict[BlockHashWithGroupId, int] = {}
        for block in self.free_block_queue.get_all_free_blocks():
            if block.block_hash in protected_hashes:
                representatives[block.block_hash] = block.block_id
        return frozenset(representatives.values())

    def _capture_prefix_retention_preimage(
        self,
        protected_hashes: AbstractSet[BlockHashWithGroupId],
        protected_block_ids: AbstractSet[int],
    ) -> tuple[
        tuple[PrefixRetentionBlockPreimage, ...],
        tuple[PrefixRetentionTrackerMetadataPreimage, ...],
        PrefixRetentionARCPreimage,
        str,
        int,
        int,
        int,
    ]:
        candidates = tuple(
            self._prefix_retention_preimage(block, queue_ordinal, protected_block_ids)
            for queue_ordinal, block in enumerate(
                self.free_block_queue.get_all_free_blocks()
            )
        )
        tracker = self.prefix_retention_tracker
        empty_arc = PrefixRetentionARCPreimage(0, (), (), (), ())
        if tracker is None:
            return candidates, (), empty_arc, "lru", 0, 0, 0
        state = tracker.observation_state()
        metadata = tuple(
            PrefixRetentionTrackerMetadataPreimage(
                block_hash_hex=bytes(item.block_hash).hex(),
                group_id=get_group_id(item.block_hash),
                parent_hash_hex=(
                    bytes(item.parent).hex() if item.parent is not None else None
                ),
                parent_group_id=(
                    get_group_id(item.parent) if item.parent is not None else None
                ),
                completed_count=item.completed_count,
                last_access_ordinal=item.last_access_ordinal,
                last_four_gaps=item.gaps,
            )
            for item in state.metadata
        )
        arc = state.arc_state
        arc_preimage = PrefixRetentionARCPreimage(
            target_t1=arc.target_t1,
            t1_hashes_hex=tuple(bytes(key).hex() for key in arc.t1_lru_to_mru),
            t2_hashes_hex=tuple(bytes(key).hex() for key in arc.t2_lru_to_mru),
            b1_hashes_hex=tuple(bytes(key).hex() for key in arc.b1_lru_to_mru),
            b2_hashes_hex=tuple(bytes(key).hex() for key in arc.b2_lru_to_mru),
        )
        return (
            candidates,
            metadata,
            arc_preimage,
            tracker.policy.value,
            tracker.budget_blocks,
            state.generation,
            state.completed_ordinal,
        )

    @staticmethod
    def _select_prefix_retention_preimages(
        candidates: tuple[PrefixRetentionBlockPreimage, ...],
        selected_blocks: Sequence[KVCacheBlock],
    ) -> tuple[PrefixRetentionBlockPreimage, ...]:
        candidate_by_id = {candidate.block_id: candidate for candidate in candidates}
        return tuple(candidate_by_id[block.block_id] for block in selected_blocks)

    def record_completed_prefix_access(
        self,
        request_id: str,
        ordered_hash_chain: Sequence[BlockHashWithGroupId],
    ) -> bool:
        """Record a true completion and append its observation afterwards."""
        tracker = self.prefix_retention_tracker
        if tracker is None or not tracker.record_completed_access(ordered_hash_chain):
            return False
        buffer = self._prefix_retention_receipt_buffer
        if buffer is not None:
            try:
                state = tracker.observation_state()
                chain = tuple(
                    PrefixRetentionHashPreimage(
                        block_hash_hex=bytes(block_hash).hex(),
                        group_id=get_group_id(block_hash),
                    )
                    for block_hash in ordered_hash_chain
                )
                buffer.append(
                    PrefixRetentionCompletedAccessReceipt(
                        schema_version=PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION,
                        request_id=request_id,
                        policy=tracker.policy.value,
                        budget_blocks=tracker.budget_blocks,
                        tracker_generation=state.generation,
                        completed_ordinal=state.completed_ordinal,
                        ordered_hash_chain=chain,
                    )
                )
            except Exception:
                buffer.mark_failed()
        return True

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def get_resident_cached_hashes(self) -> tuple[BlockHashWithGroupId, ...]:
        """Return a unique, group-aware snapshot without exposing the map."""
        return self.cached_block_hash_to_block.resident_keys()

    def prefix_retention_pool_state(self) -> tuple[int, int, int]:
        non_null_used = self.num_gpu_blocks - self.get_num_free_blocks() - 1
        resident_keys = len(self.cached_block_hash_to_block)
        hashed_physical = sum(
            block.block_hash is not None and not block.is_null for block in self.blocks
        )
        return non_null_used, resident_keys, hashed_physical

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.prefix_retention_tracker is not None:
            chain = tuple(
                make_block_hash_with_group_id(block_hash, kv_cache_group_id)
                for block_hash in block_hashes[:num_full_blocks]
            )
            if all(
                block.block_hash == block_hash
                for block, block_hash in zip(blocks[:num_full_blocks], chain)
            ):
                self.prefix_retention_tracker.register_chain(chain)

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null/masked-out blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                if block_mask is not None and not block_mask[i - num_cached_blocks]:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                    group_idx=kv_cache_group_id,
                )
            )

    def get_new_blocks(
        self,
        num_blocks: int,
        *,
        protected_hashes: AbstractSet[BlockHashWithGroupId] | None = None,
        request_id: str | None = None,
    ) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.
            protected_hashes: Group-aware cached block identities to prefer
                retaining. Protection is advisory and never prevents an
                otherwise feasible allocation.
            request_id: Optional scheduler request identity for observation.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        if num_blocks == 0:
            return []

        buffer = self._prefix_retention_receipt_buffer
        protected_block_ids = (
            self._physical_protected_block_ids(protected_hashes)
            if buffer is not None and protected_hashes
            else frozenset()
        )
        if buffer is None:
            if not protected_hashes:
                ret = self.free_block_queue.popleft_n(num_blocks)
            else:
                ret = self.free_block_queue.popleft_n_prefer_unprotected(
                    num_blocks, protected_hashes
                )
            if self.enable_caching:
                for block in ret:
                    self._maybe_evict_cached_block(block)
                    assert block.ref_cnt == 0
                    block.ref_cnt += 1
                    if self.metrics_collector:
                        self.metrics_collector.on_block_allocated(block)
            else:
                for block in ret:
                    assert block.ref_cnt == 0
                    block.ref_cnt += 1
                    if self.metrics_collector:
                        self.metrics_collector.on_block_allocated(block)
            return ret

        free_blocks_before = self.get_num_free_blocks()
        frozen_protected_hashes: frozenset[BlockHashWithGroupId] = frozenset()
        candidates: tuple[PrefixRetentionBlockPreimage, ...] | None = None
        tracker_metadata: tuple[PrefixRetentionTrackerMetadataPreimage, ...] = ()
        arc_preimage = PrefixRetentionARCPreimage(0, (), (), (), ())
        policy = "lru"
        budget_blocks = 0
        tracker_generation = 0
        completed_ordinal = 0
        try:
            frozen_protected_hashes = frozenset(protected_hashes or ())
            (
                candidates,
                tracker_metadata,
                arc_preimage,
                policy,
                budget_blocks,
                tracker_generation,
                completed_ordinal,
            ) = self._capture_prefix_retention_preimage(
                frozen_protected_hashes, protected_block_ids
            )
        except Exception:
            buffer.mark_failed()

        if not protected_hashes:
            ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
        else:
            ret = self.free_block_queue.popleft_n_prefer_unprotected(
                num_blocks, protected_hashes
            )

        selected: tuple[PrefixRetentionBlockPreimage, ...] | None = None
        if candidates is not None:
            try:
                selected = self._select_prefix_retention_preimages(candidates, ret)
            except Exception:
                buffer.mark_failed()
        victims: list[PrefixRetentionBlockPreimage] = []

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for index, block in enumerate(ret):
                if self._maybe_evict_cached_block(block) and selected is not None:
                    victims.append(selected[index])
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)

        if buffer is not None:
            self._prefix_retention_allocation_ordinal += 1
            if candidates is not None and selected is not None:
                try:
                    protected_hashes_hex = tuple(
                        sorted(
                            bytes(block_hash).hex()
                            for block_hash in frozen_protected_hashes
                        )
                    )
                    frozen_victims = tuple(victims)
                    receipt = PrefixRetentionDecisionReceipt(
                        schema_version=PREFIX_RETENTION_OBSERVER_SCHEMA_VERSION,
                        request_id=request_id,
                        allocation_ordinal=self._prefix_retention_allocation_ordinal,
                        policy=policy,
                        budget_blocks=budget_blocks,
                        tracker_generation=tracker_generation,
                        completed_ordinal=completed_ordinal,
                        tracker_metadata_preimage=tracker_metadata,
                        arc_preimage=arc_preimage,
                        protected_hashes_hex=protected_hashes_hex,
                        candidates=candidates,
                        selected=selected,
                        victims=frozen_victims,
                        free_blocks_before=free_blocks_before,
                        free_blocks_after=self.get_num_free_blocks(),
                        non_null_used_blocks_after=sum(
                            block.ref_cnt > 0 and not block.is_null
                            for block in self.blocks
                        ),
                    )
                    buffer.append(receipt)
                except Exception:
                    buffer.mark_failed()
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                    group_idx=get_group_id(block_hash),
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(
        self, ordered_blocks: Iterable[KVCacheBlock], prepend: bool = False
    ) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
            prepend: Whether to put newly-free blocks at the front of the free
                queue to be prioritized for reuse.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        freed_blocks = [
            block for block in blocks_list if block.ref_cnt == 0 and not block.is_null
        ]
        if prepend:
            self.free_block_queue.prepend_n(freed_blocks)
        else:
            self.free_block_queue.append_n(freed_blocks)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
