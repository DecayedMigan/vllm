# GLADIUS H100 Discovery Execution Plane — Second Review Requirements

**Date:** 2026-08-02

**Reviewed revision:** `63f722398320b8f4112369046f6465443e6abb82`

**Execution implementation:** `c9248aa01ea8e24496458549566a9f2044db5cda`

**Companion SMIG revision:** `c7b202800b7b2b4ecfbe2842293c8bbd914ae48d`

**Parent contract:**
[`gladius_h100_discovery_execution_plane_requirements.md`](gladius_h100_discovery_execution_plane_requirements.md)

**Third-review status:** the later implementation claim is **not accepted**.
Independent mutation probes found that the seal verifies sibling hashes but not
their deployment/application semantics, and no production lifecycle path calls
the seal. Additional receipt, GPU-identity, socket-ownership, and writer-race
gaps remain. Do not package this branch for H100 work. The controlling plan is
[`../superpowers/plans/2026-08-02-h100-execution-plane-third-review-remediation.md`](../superpowers/plans/2026-08-02-h100-execution-plane-third-review-remediation.md).

**Decision:** changes required. Do not package this revision for the formal H100
campaign. The following P0 issues invalidate physical-GPU identity or allow an
unattested or mutable telemetry stream to pass the current seal verifier.

## 1. Review evidence

The telemetry/evidence subset passes locally when isolated from the repository's
Linux-only global test setup:

```text
42 passed
```

The complete `tests/gladius` collection cannot be represented as a clean local
result on this macOS review host. The checked `.venv` initially lacks NumPy;
after bypassing the root conftest for the focused receipt suite, 46 tests passed
and 26 receipt tests failed because `/proc/<pid>/stat` is unavailable. These are
environment limitations, not the basis for the findings below. The blockers
were confirmed from source and with focused pure-Python probes.

## 2. P0-A: map a CUDA logical device to the correct physical NVML device

`receipt._resolve_physical_gpu()` obtains `torch.cuda.current_device()` and
passes that integer directly to `nvmlDeviceGetHandleByIndex()`. CUDA device
indices are logical inside `CUDA_VISIBLE_DEVICES`; NVML indices are physical
machine indices and are not remapped by that environment variable.

Under the required one-process-per-GPU launch, each replica normally sees
logical device `0`. The current NVML lookup therefore risks reporting physical
GPU 0 for all four replicas. This defeats the exact property the receipt is
meant to prove. The implementation handoff already records that the NVML/H100
branch was not exercised.

Resolve the physical handle by a device property that crosses the CUDA/NVML
namespace boundary, preferably PCI bus ID or CUDA's device UUID, and then read
the authoritative NVML UUID from that handle. Do not treat a logical ordinal as
a physical ordinal or trust the text of `CUDA_VISIBLE_DEVICES` alone.

Required tests cover numeric masks `0` through `3`, reordered masks such as
`3,1`, UUID-form masks, MIG where supported, and mismatch/ambiguity failure. A
four-process H100 smoke must yield four distinct receipt UUIDs matching the
intended devices in `nvidia-smi --query-gpu=uuid`.

## 3. P0-B: remove the unverified `--api-pid` attestation bypass

The attestor documentation says the API PID is derived from `/proc` socket
ownership, but `attest publish` accepts `--api-pid` and uses it directly when
present. The supplied PID is not checked against the listening socket. A
receipt can therefore bind a live but unrelated process to the requested
endpoint.

Remove the override from the formal interface, or require it to equal the PID
independently derived from the bound `host:port`. Receipt publication must also
reject a socket that is not listening, an endpoint owned by another process,
and a process whose start identity changes during assembly. Tests must cover
all three cases.

## 4. P0-C: make expected deployment digests mandatory at verification

The receipt records model, tokenizer, vLLM package, native-extension, and
GLADIUS overlay digests. `verify_server_start_receipt()` compares digests only
when an optional expected mapping is supplied, while `attest verify` exposes no
CLI arguments or signed manifest input for those expectations. The current
SMIG consumer also does not pass them.

The formal verifier must consume one immutable deployment manifest and require
exact nonce, endpoint, engine/model identity, GPU UUID, model/tokenizer/vLLM/
native-extension/overlay digests, hash algorithm version, and all frozen startup
settings. Missing expectations are a formal verification failure, not a skipped
check. The CLI, Python API, shared fixture, and SMIG caller must use the same
strict verification path.

## 5. P0-D: refuse unattested or sibling-less telemetry seals

`TelemetryWriter._verify_segments()` discards null instance IDs before counting
instances. A stream in which every record has `server_instance_id: null`
therefore derives a null instance instead of failing. `seal()` also treats
missing `server_start_receipt.json` and `policy_application.json` as null
digests and still returns success.

A focused probe against the reviewed commit produced:

```text
seal returned:                         true
seal server_instance_id:               null
attestation_receipt_sha256:            null
policy_application_sha256:             null
verify_telemetry_seal errors:           []
```

For a formal seal, every record must carry the same nonempty server instance,
and both sibling files must exist, parse strictly, share that identity, and be
hashed only after those semantic checks. A missing, null, or mismatched sibling
must make sealing and independent verification fail.

If pre-attestation native records are retained for production observability,
they must use a separate explicitly non-certifiable stream or non-formal mode
that SMIG can never accept.

## 6. P0-E: re-derive every seal field during independent verification

`verify_telemetry_seal()` re-derives record count and first/final step but does
not re-derive or compare `generation_high_watermark`. In the same probe,
changing the sealed manifest watermark from `7` to `999` still returned an
empty error list.

The verifier must re-derive exactly one nonempty instance, watermark, step
bounds, count, increasing step sequence, engine/model identity, receipt and
application digests, parsed sibling identities, and terminal application/action
consistency. The manifest parser must reject missing and unknown fields. Add one
mutation test per manifest field, plus sibling-content mutations whose file
digests are updated in the manifest; semantic checks must still reject them.

## 7. P0-F: reject half-native decisions in the writer

The parser correctly rejects a record whose generation, policy ID, and decision
ID are only partly present. The writer, however, calculates native state using
`generation is None OR policy_id is None`, then rewrites a half-native input
into an all-null record. This repairs an impossible state instead of exposing
it.

Writer and parser must enforce the same invariant. A half-native
`PolicyDecision` must disable formal evidence for that instance while serving
remains fail-open, and must never be serialized as valid native evidence.

## 8. P0-G: make telemetry record parsing schema-strict

`parse_telemetry_record_v2()` validates only a subset of the fields the writer
emits and returns the original mapping, including unknown fields. Formal
evidence parsing must require the exact 2.0.0 field set and validate types,
ranges, timestamps, queue/admission values, clamp invariants, policy status, and
engine/model identity. The byte-identical shared SMIG fixture must pin the same
accept/reject behavior in both repositories.

## 9. P0-H: prevent post-seal mutation by a new writer

Sealing freezes one `TelemetryWriter` object, but a new scheduler using the same
policy directory can reopen and append to the certified telemetry path before
receipt assembly rejects the reused directory. This destroys previously sealed
bytes even if verification later notices the damage.

Before opening any telemetry segment, detect a final receipt, seal, or
retirement marker and refuse writes for that directory. A completed formal
policy directory is immutable. Keepalive or a restarted server uses a new
directory and nonce. Add a test that constructs a second writer after sealing
and proves every certified byte remains unchanged.

## 10. Required acceptance sequence

Before another H100 handoff, provide one implementation commit and run:

1. Ruff check and format check on `gladius_vllm/` and `tests/gladius/`;
2. the complete `tests/gladius` suite in the documented Linux `uv` environment;
3. the byte-identical cross-repository fixture suite against exact SMIG commit;
4. focused negative probes for P0-A through P0-H;
5. a four-process H100 receipt smoke proving four expected, distinct GPU UUIDs;
6. strict receipt verification against the checksummed deployment manifest;
7. seal, copy, independent verify, then post-seal immutability verification.

Report exact commands, counts, skips, both repository commits, and archive
digests. Do not label the execution plane complete based on a synthetic receipt
or ROCm fallback: the NVIDIA NVML mapping and `/proc` socket-owner path are
formal H100 acceptance criteria.
