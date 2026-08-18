# Shared Prefix Retention Selector Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Give LastSeen, LFU, ARC, and RecurPlan one semantics-preserving
bitmask selection skeleton.

**Architecture:** The tracker builds canonical closures and masks once, then a
single greedy selector computes current marginal masks. Policy-specific ranking
is supplied only after current marginal cost is known. A test-only legacy set
selector is the semantic oracle.

**Tech Stack:** Python 3.12, pytest, vLLM V1 core, remote ROCm llm-finetune.

**Spec:** docs/superpowers/specs/2026-08-18-shared-prefix-retention-selector-design.md

## Global Constraints

- [ ] Branch from a5e8b11b6e1009795a886d6ce04937b136e06d7a as agent/gladius-kv-shared-selector.
- [ ] Do not touch feature/gladius-v3-vllm-engine or its untracked .serena directory.
- [ ] Do not call O(B x T) linear.
- [ ] Preserve all ranking tuples, marginal costs, and canonical tie-breaks.
- [ ] LRU and zero budget bypass mask construction.
- [ ] Use only zazzi@100.72.102.42 and its existing llm-finetune test environment.
- [ ] Keep the GLADIUS JCS seal repair separate.
- [ ] Use additive commits only; never amend or force push.

### Task 1: RED oracle and shared-selector contract

**Files:**
- Modify: tests/v1/core/test_prefix_retention.py

**Interfaces:**
- Produces: _legacy_protected_hashes(tracker, snapshot), a test-only copy of
  the old set algorithm.
- Produces: _select_protected_masks(snapshot, metadata, closures), required
  private shared execution point for every non-LRU policy.

- [ ] Write a test that attempts to wrap tracker._select_protected_masks,
  invokes protected_hashes for PrefixRetentionPolicy.PREFIX_RECENCY, and
  asserts one call. The pre-change RED must fail because the method does not
  exist.
- [ ] Run that single test remotely and preserve the expected AttributeError.
- [ ] Add the minimal legacy oracle in tests only. It must use the old closure
  difference loop and existing policy ranking methods.

### Task 2: One production bitmask selector

**Files:**
- Modify: vllm/v1/core/prefix_retention.py
- Test: tests/v1/core/test_prefix_retention.py

**Interfaces:**
- Consumes: snapshot, metadata, and terminal frozenset closures.
- Produces: _select_protected_masks returning a protected canonical hash set.

- [ ] Create canonical bit_for and closure_masks once.
- [ ] For each unconsidered terminal compute marginal from its closure mask,
  count it with bit_count, skip exhausted or over-budget candidates, and update
  the protected mask after selection.
- [ ] Preserve PrefixRecency, LFU, ARC, RecurPlan ratio, ARC fallback, cost,
  and canonical ordering exactly. Precompute each RecurPlan timing score once.
- [ ] Remove the RecurPlan-only loop rather than retaining a private fast path.
- [ ] Re-run the Task 1 test remotely until GREEN, then run the existing
  prefix-retention module remotely.

### Task 3: Semantic and allocation equivalence

**Files:**
- Modify: tests/v1/core/test_prefix_retention.py
- Modify: tests/v1/core/test_prefix_retention_receipts.py

**Interfaces:**
- Consumes: legacy test oracle and real BlockPool allocation.
- Produces: policy-equivalence and physical-ordering tests.

- [ ] Parametrize LastSeen, LFU, ARC, and RecurPlan over literal fixtures for
  chains, forks, many shared ancestors, group IDs, duplicate physical copies,
  budgets zero/one/exact/impossible, missing ancestors, non-resident history,
  and canonical ties.
- [ ] Compare legacy and shared protected sets and used budget. Assert LRU
  has zero protection and never invokes the shared selector.
- [ ] Use real BlockPool.get_new_blocks to assert equal physical
  representatives, victim block IDs, and remaining free block IDs.
- [ ] Run the two focused test modules remotely.

### Task 4: Deterministic counts and separate benchmark

**Files:**
- Create: benchmarks/benchmark_prefix_retention.py
- Modify: tests/v1/core/test_prefix_retention.py

**Interfaces:**
- Produces: deterministic selector operation counts and an explicitly invoked
  median benchmark report.

- [ ] Write failing tests for one closure construction per valid terminal and
  bounded candidate and mask operation accounting.
- [ ] Add test-only or benchmark-only instrumentation without changing receipt
  schema or retention semantics.
- [ ] Benchmark 4696 residents, 4665 terminals, budget 1174; report median
  decision time, candidate evaluations, closure constructions, mask
  operations, peak memory, and legacy LastSeen speedup.
- [ ] Run deterministic tests and repeated remote benchmark samples. A failed
  target stops GPU replay rather than shrinking the workload.

### Task 5: Verification and additive selector commit

**Files:**
- Modify only files listed in Tasks 1 through 4.

- [ ] Run focused tests, applicable remote full vLLM suite, Ruff on changed
  Python files, and git diff --check.
- [ ] Independently review shared routing, unchanged ranking semantics,
  absence of production set-difference loops, and benchmark dimensions.
- [ ] Commit with message: perf: share prefix retention bitmask selector.
