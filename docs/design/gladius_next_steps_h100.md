# GLADIUS vLLM H100 Blockers and Next Steps

**Date:** 2026-08-01

**Branch inspected:** `feature/gladius-v3-vllm-engine`

**Inspected commit:** `44cac872a`

**Purpose:** close the two vLLM-side blockers found by the real four-H100
contract smoke, then hand a canonical scheduler plugin back to the SMIG H100
pilot. This document is an implementation handoff. It does not change vLLM
runtime code itself.

## 1. Authoritative H100 environment

The offline H100 host has four NVIDIA H100 80 GB GPUs. The only verified
environment on that host with vLLM installed is:

```text
/opt/conda/private/envs/gladius/bin/python
Python 3.12.13
torch 2.11.0+cu129
vLLM 0.25.1
CUDA 12.9
```

Available local models are:

```text
/data/user/yge269/models/Phi-4-mini-instruct
/data/user/yge269/models/Qwen3-8B
```

The host cannot access the Internet. Set both `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; do not use a Hub model identifier in H100 tests.
The earlier `llm-finetune` environment instruction applies to the other
machine, not this H100 host. On this host, use the `gladius` environment above.

Do not add SSH credentials to commands, scripts, manifests, logs, or this
repository.

## 2. P0-A: support the installed Scheduler.schedule API

### Observed failure

The API server loads both local models and instantiates the canonical plugin,
but the first request kills EngineCore:

```text
TypeError: GladiusScheduler.schedule() takes 1 positional argument but 2 were given
```

vLLM 0.25.1 calls:

```python
self.scheduler.schedule(self._should_throttle_prefills())
```

and its base scheduler exposes:

```python
Scheduler.schedule(self, throttle_prefills: bool = False)
```

The current fork commit instead defines both the checked-in base scheduler and
`GladiusScheduler.schedule()` without that argument. The plugin is loaded on
top of the installed vLLM package during the offline H100 run, so its override
must tolerate both APIs.

### Required implementation

Keep `gladius_vllm.scheduler.GladiusScheduler` as the only state machine. Make
the override forward the arguments supplied by the active vLLM runtime:

```python
def schedule(self, *args: object, **kwargs: object) -> SchedulerOutput:
    decision = self._policy_loader.poll()
    # Existing safe-clamp logic remains unchanged.
    ...
    output = super().schedule(*args, **kwargs)
    self._telemetry_writer.record(self, output, decision)
    return output
```

Using forwarding rather than always passing `False` is important: the
checked-in fork currently has a zero-argument base method, while installed
vLLM 0.25.1 has the `throttle_prefills` argument. The same plugin must work in
both cases without version-string branching.

Add CPU tests for both call shapes:

1. `gladius.schedule()` forwards no arguments.
2. `gladius.schedule(True)` forwards `True` when the base API accepts it.
3. Policy clamp and telemetry still execute exactly once for either call.
4. Exceptions from telemetry remain fail-open and never take down scheduling.

### Existing H100 evidence

An explicitly labelled, uncommitted diagnostic overlay changed only this call
signature and forwarding behavior. With that overlay, both real-model smoke
tests completed successfully:

| Model | Telemetry | DAGs | Experiences | Generations | Peak running |
|---|---:|---:|---:|---|---|
| Phi-4-mini-instruct | 389 | 3 | 3 | 1, 2, 3 | 4 -> 1 -> 4 |
| Qwen3-8B | 389 | 3 | 3 | 1, 2, 3 | 4 -> 1 -> 4 |

These runs prove the compatibility change is sufficient to unblock the real
serving path, but they are diagnostic evidence only. The formal smoke must be
rerun from a committed vLLM source archive.

## 3. P0-B: make atomic snapshot replacement observable

### Observed failure

The transferred Linux CPU gate collected 91 plugin tests: 78 passed and 13
failed. Two implementation failures were:

```text
test_generation_regression_rejected
test_generation_equal_rejected
```

`PolicyLoader` currently caches only:

```python
(stat.st_mtime_ns, stat.st_size)
```

The tests publish through a temporary file plus `os.replace()`. An immediate,
same-size replacement can retain the same observed mtime and size on the HPC
filesystem, so the loader incorrectly treats the new inode as unchanged and
does not perform regression validation.

### Required implementation

Include the inode in the cached fingerprint:

```python
self._last_stat: tuple[int, int, int] | None = None
current_stat = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
```

This matches the canonical atomic-publication requirement: `os.replace()`
changes the file identity even when payload size and timestamp resolution do
not. Preserve the unchanged-file fast path and the never-raises behavior.

Required tests:

1. Same-size atomic replacement is reparsed.
2. Equal and lower generations are rejected.
3. A strictly higher generation is accepted.
4. A genuinely unchanged inode/mtime/size skips JSON parsing.
5. Missing, corrupt, expired, mismatched, and regressed files retain the
   documented fallback semantics.

## 4. P0-C: make offline real-model tests configurable

The other 11 H100 test failures were not scheduler failures. They hard-code
`Qwen/Qwen3-1.7B`, which is unavailable on the offline host. Parameterize the
real-model smoke fixture through an explicit environment variable, for example
`GLADIUS_TEST_MODEL`, while retaining a useful default for connected CI.

On the H100 host, run the smoke twice:

```text
GLADIUS_TEST_MODEL=/data/user/yge269/models/Phi-4-mini-instruct
GLADIUS_TEST_MODEL=/data/user/yge269/models/Qwen3-8B
```

Tests must not attempt a network fallback when the configured path is missing.
Fail early with a clear path error instead.

The broad transferred test collection also required `tblib`, which is absent
from the offline environment. Do not install from the Internet during the
campaign. Either keep the contract gate restricted to tests whose dependencies
are already present, or transfer a prebuilt dependency wheel as a separately
checksummed input.

## 5. Runtime-overlay rule

Do not put the clean fork root at the front of `PYTHONPATH` on the H100 host.
Doing so shadows the installed vLLM package and fails to import `vllm._C`
because the transferred source tree has no matching local build.

The supported offline layout is:

```text
installed site-packages: vllm 0.25.1 plus compiled extensions
PYTHONPATH overlay:       committed gladius_vllm package only
working directory:        outside the clean vLLM source root
```

Before starting a server, verify both origins:

```python
import vllm
import gladius_vllm

print(vllm.__version__, vllm.__file__)
print(gladius_vllm.__file__)
```

## 6. Required validation sequence

Do not declare the vLLM H100 handoff complete until all of the following pass:

1. Ruff and the pure-Python `tests/gladius` contract suite.
2. The shared fixture SHA-256 matches the SMIG fixture.
3. The two regression tests in P0-B pass repeatedly on the Linux HPC
   filesystem.
4. The canonical plugin imports on top of installed vLLM 0.25.1.
5. Phi-4 formal smoke observes generations 1/2/3 and admission 4 -> 1 -> 4.
6. Qwen3-8B formal smoke observes the same state transition.
7. Each smoke produces nonempty telemetry, three valid Provenance DAGs, three
   retained serving experiences, and matching engine/model/policy/decision
   identities.
8. API server and EngineCore exit cleanly and all four GPUs return to idle.

Use production settings during the formal run:

```text
GLADIUS_POLICY_DIR=<unique run directory>
GLADIUS_ENGINE_ID=<stable explicit engine id>
GLADIUS_POLICY_POLL_INTERVAL_MS=1
GLADIUS_TELEMETRY_SAMPLE_N=1
max_num_seqs startup ceiling=8
max_num_batched_tokens startup ceiling=8192
```

## 7. Handoff to SMIG

After the implementation commit is pushed, report:

- the exact vLLM commit SHA;
- tests run and their pass/fail counts;
- the committed `gladius_vllm` archive SHA-256;
- any remaining environment-only skips.

SMIG is prepared on branch `agent/h100-campaign` at commit `d314df1`. Once the
new vLLM commit is deployed and both formal smokes pass, SMIG will run the
three-seed/four-method pilot and only then open the 40-seed campaign gate.
