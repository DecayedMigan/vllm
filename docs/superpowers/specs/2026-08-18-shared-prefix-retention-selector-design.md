# Shared Prefix Retention Selector Design

## Goal

Make LastSeen, LFU, ARC, and RecurPlan use one bitmask-based greedy selector
without changing any policy ranking tuple, marginal cost, or canonical
tie-break.

## Design

The public protected_hashes entry point keeps generation validation and the LRU
or zero-budget bypass. For each non-LRU policy it constructs metadata,
resident closures, a canonical resident index, and closure masks once. One
private selector then evaluates each remaining candidate with the bitmask
operation closure_mask & ~protected_mask and marginal.bit_count(), applies
exact budget accounting, and returns the canonical protected hash set.

Policies retain their existing ranking logic. Prefix recency, LFU, and ARC use
their current ranking tuples. RecurPlan retains its timing-score and ARC
fallback ordering but contributes it to the shared selector rather than owning
a private bitmask loop. No production legacy set selector remains.

This reduces repeated marginal computation to O(B x T) after closures are
built. It is not linear: B can grow with the resident population.

## Correctness and performance

A minimal test-only legacy set selector is the semantic oracle. It must match
the shared selector on chains, forks, shared ancestors, group-aware hashes,
duplicate physical copies, boundary budgets, invalid ancestry, non-resident
history, canonical ties, and all executable non-LRU policies. Block-pool tests
also compare protected physical representatives, victims, and free-queue
order. LRU and budget zero bypass before mask construction.

Deterministic tests record closure construction, candidate evaluation, and mask
operation counts. A separately invoked benchmark uses about 4696 residents,
4665 terminals, and budget 1174; it reports median time, peak memory, counts,
and legacy LastSeen speedup. It does not weaken the workload. On the frozen GPU
environment selection must stay below one percent of request wall time, with a
120 ms target; otherwise replay remains stopped.

## Scope

No ARC state transition, receipt schema, physical allocation policy, or
GLADIUS failure-root sealing changes belong in this design. The JCS repair is a
later independent repository change and commit.
