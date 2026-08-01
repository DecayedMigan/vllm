# GLADIUS Shift-Cost Reduction: vLLM Execution-Plane Next Steps

**Date:** 2026-08-01

**Branch:** `feature/gladius-v3-vllm-engine`

**Starting revision:** `cde356011`

**Status:** implementation handoff; this document freezes the next execution-plane
milestone before runtime changes are committed.

## 1. Goal and evidence boundary

The next milestone reduces the cost of adapting after recurring workload shifts
without moving learning into vLLM. GLADIUS remains the control plane and owns
contextual experience, candidate probabilities, safety constraints, replay, and
policy selection. vLLM remains a deterministic, fail-open execution plane and
owns policy observation, safe admission clamping, scheduler-step telemetry, and
application acknowledgement.

Success is lexicographic:

1. no serving-safety or SLO regression relative to the accepted static policy;
2. lower post-shift area-under-regret and fewer recovery windows;
3. fewer exploratory requests and policy switches;
4. bounded control-plane overhead.

The first action space remains the canonical two-dimensional admission pair:

```text
(max_num_seqs, max_num_batched_tokens)
```

Prefix caching, chunked prefill, model weights, and other restart-sensitive
settings are not hot-switched in this milestone. A learned predictor may create
shadow candidates but may not directly control the scheduler.

## 2. Contract that does not change

`policy_snapshot.json` remains schema `1.0.0`. Its parser, engine/model matching,
generation high-watermark, TTL handling, startup clamp, and fallback behavior are
unchanged. vLLM does not accept legacy SMIG snapshots.

The execution directory gains one additive file:

```text
GLADIUS_POLICY_DIR/
  policy_snapshot.json
  policy_application.json
  telemetry.jsonl
```

The snapshot is still the only input that controls scheduling. The application
file and telemetry are observations; corrupt or unwritable observations must
never stop serving.

## 3. Policy application acknowledgement

After the scheduler first observes a new valid generation, it atomically writes
`policy_application.json`:

```json
{
  "schema_version": "1.0.0",
  "engine_id": "engine-a",
  "model_id": "Qwen/Qwen3-8B",
  "generation": 42,
  "policy_id": "policy-42",
  "decision_id": "policy-42",
  "observed_at": "2026-08-01T00:00:00Z",
  "scheduler_step": 1037,
  "state": "active",
  "requested_admission": {
    "max_num_seqs": 8,
    "max_num_batched_tokens": 2048
  },
  "effective_admission": {
    "max_num_seqs": 8,
    "max_num_batched_tokens": 2048
  },
  "clamped": {
    "max_num_seqs": false,
    "max_num_batched_tokens": false
  }
}
```

`state` has exactly three meanings:

- `active`: requested and effective ceilings match.
- `attriting`: a lower sequence ceiling is waiting for already-running requests
  to finish; no request is evicted.
- `fallback`: the scheduler is using startup defaults because no valid active
  policy exists.

The acknowledgement is rewritten only when generation/state/effective admission
changes. It uses a temporary file, flush, `fsync`, and `os.replace`. Equal or
regressed generations never produce a new acknowledgement. Observation I/O is
never allowed to raise through `schedule()`.

An acknowledgement does not make file publication synchronous: the scheduler
can observe a policy only on a scheduling step. GLADIUS may trigger one discarded
activation request when no real request is available, then wait for the matching
ack instead of scanning an unbounded telemetry stream.

## 4. Low-overhead telemetry and sealing

Per-step telemetry retains the canonical identities and requested/effective
admission fields. Additive timing fields report monotonic durations in
nanoseconds:

```text
policy_poll_ns
policy_apply_ns
telemetry_write_ns
```

Timing failures or serialization failures remain fail-open. Sampling and
size-based rotation remain enabled. The execution-plane P95 overhead target is
at most 1 ms per sampled scheduler step on H100.

Completed experiment data must not share a live writer with keepalive traffic.
The writer therefore exposes a lifecycle operation that:

1. flushes and closes the active JSONL handle;
2. records the final scheduler step and telemetry filenames;
3. atomically writes a seal manifest with SHA-256 values;
4. rejects subsequent writes from that writer instance without affecting
   scheduling.

The seal is an experimental evidence boundary, not a production availability
mechanism. A new policy directory and writer are required for later keepalive or
another run.

## 5. Parallel handoff to GLADIUS

GLADIUS will add continuous-context retrieval, confidence-weighted retained
priors, explicit shift episodes, and constrained contextual exploration. vLLM
must preserve enough observation data to link:

```text
ShiftEpisode
  -> WorkloadWindow
  -> CandidateSet (real propensities)
  -> PolicyDecision
  -> PublishedSnapshot
  -> PolicyApplication
  -> SchedulerStep*
  -> ServingOutcome
  -> RetainedExperience
```

The snapshot intentionally does not carry workload vectors, source experience,
or offline candidate fields. Those stay in the GLADIUS provenance ledger.

## 6. Implementation and test sequence

### P0: application observation

- Add a pure-Python application record/parser and atomic writer.
- Integrate it into the single `GladiusScheduler` state machine.
- Cover active, attriting, fallback, generation replacement, write failure, and
  clean close with CPU tests.

### P1: overhead and evidence lifecycle

- Add sampled poll/apply/write timing.
- Add telemetry sealing and a verifiable manifest.
- Prove that post-seal requests continue serving but cannot mutate certified
  telemetry.

### P2: cross-repository gate

- Share application fixtures with SMIG.
- Verify snapshot -> application -> scheduler-step -> DAG identity and step
  ranges across both parsers.
- Run real Phi-4 regression smoke, then Qwen3-8B calibration and pilot in the
  offline four-H100 `gladius` Conda environment.

## 7. Experiment gates

The next calibration uses a two-dimensional admission candidate set and a
workload with burst, prompt/output-length, prefix-sharing, and KV-pressure
changes. It must show at least 4% per-window-oracle headroom over the best static
candidate before an adaptive pilot starts.

For recurring shifts, GLADIUS must reduce area-under-regret by at least 20%
relative to no retention and 10% relative to the best preregistered SW-UCB. The
number of windows needed to sustain at least 98% of the oracle must not increase.
All application identities, real propensities, scheduler-step ranges, and seal
hashes must validate. Structural failure blocks expansion even when mean reward
looks favorable.

## 8. Explicit non-goals

- No learning, vector search, replay, or network call in vLLM scheduling.
- No direct gradient control of production scheduling.
- No runtime increase above startup CUDA-graph or token ceilings.
- No restart-sensitive policy fields in the hot-reload snapshot.
- No claim of continuous-learning serving completion before the cross-repository
  Qwen3-8B H100 gate passes.
