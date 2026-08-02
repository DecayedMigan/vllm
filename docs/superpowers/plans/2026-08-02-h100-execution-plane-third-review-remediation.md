# GLADIUS H100 Execution Plane Third-Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use superpowers:test-driven-development for every behavior change and superpowers:verification-before-completion before any completion claim.

**Goal:** Make vLLM emit and independently verify deployment-bound receipts, telemetry, policy applications, and terminal seals through a real production lifecycle path, with no semantic rewrite or physical-identity ambiguity accepted.

**Architecture:** The vLLM fork is the producer of execution evidence, not its own final trust root. Receipt publication binds the live endpoint and physical GPU to an immutable deployment expectation. Telemetry and application records form one server-instance stream. A production seal transition atomically retires that stream and validates every sibling against the receipt, expectation, and telemetry state. SMIG independently consumes the same exact-schema fixture corpus and anchors the resulting seal externally.

**Tech Stack:** Python 3.11+, vLLM EngineCore scheduler integration, JSON/JSONL, SHA-256, Linux `/proc`, NVML/CUDA identity APIs, pytest.

---

## 0. Third-review decision and corrected constraint model

The second-review implementation at `c311bc0` is not approved for packaging or
H100 execution. It fixed several concrete mechanics, but the acceptance contract
allowed local hash consistency and helper tests to masquerade as end-to-end
attestation.

The constraint failure was ours in part: we required “strict seal verification”
without enumerating same-instance semantic rewrites; required a “production
caller” without naming the lifecycle event and observable artifact; and required
physical identity without forcing CUDA/NVML cross-agreement and MIG cases. The
implementation team also overclaimed completion: repository search shows no
non-test caller of `seal_telemetry`, while a receipt or application can be
changed, rehashed, and still pass seal verification. Future claims therefore use
an executable four-part contract:

```text
attack fixture -> classified refusal -> production call site -> retained artifact
```

No task is complete if any part is absent. Helper coverage cannot substitute for
the production call site, and a checksum match cannot substitute for semantics.

## 1. Non-negotiable execution-plane contract

- Preserve the frozen Qwen3-8B four-replica experiment settings and production
  CUDA graph path; this plan changes evidence integrity, not experiment design.
- A formal receipt verifier always consumes an immutable deployment expectation.
  Optional expectation fields and skip-on-missing behavior are forbidden.
- Exact schemas are required. Unknown, missing, wrongly typed, or noncanonical
  fields fail in both vLLM and SMIG against one shared corpus.
- Every physical identity claim must be corroborated across CUDA and NVML. An
  ordinal is never a physical identity.
- Every seal field and sibling semantic is independently re-derived. Rehashing a
  modified sibling must not make it valid.
- Seal creation must be reachable from the deployed server lifecycle without a
  Python test holding an `EngineCore` object.
- Sealing is an atomic terminal transition for a policy directory. No existing or
  new writer may append afterward.
- Linux integration establishes `/proc` and socket facts. A mock establishes only
  branch behavior. H100 smoke establishes real CUDA/NVML/MIG identity.
- The delivery handoff must map each claim to source caller, adversarial test,
  command, artifact, and result. Missing evidence means “not delivered”.

## 2. Write the adversarial acceptance suite before implementation

**Files:**

- Create: `tests/gladius/test_third_review_adversarial.py`
- Modify: `tests/gladius/test_telemetry_seal_v2.py`
- Modify: `tests/gladius/test_telemetry_writer.py`
- Modify: `tests/gladius/test_server_start_receipt.py`
- Modify: `tests/gladius/fixtures/gladius-execution-evidence-v2.json`
- Modify: `tests/gladius/fixtures/gladius-control-protocol-v1.json`
- Create: `tests/gladius/fixtures/gladius-third-review-corpus.json`

Add and retain red results for these tests:

1. `test_seal_rejects_rehashed_application_action_rewrite`: change a valid
   policy application's requested/effective action, rebuild its digest in the
   seal, and require a semantic mismatch failure.
2. `test_seal_rejects_rehashed_receipt_startup_rewrite`: change
   `startup_max_model_len` or another deployment field, rebuild the digest, and
   require a deployment-expectation failure.
3. `test_seal_rejects_unlisted_and_missing_segments`: add a telemetry segment
   not listed by the manifest, omit one listed segment, reorder/duplicate a
   filename, and require exact-set failure.
4. `test_seal_binds_application_to_telemetry_step`: reject wrong engine/model,
   generation, action, watermark, or scheduler step not represented by the
   sealed telemetry stream.
5. `test_formal_verify_requires_complete_deployment_expectation`: reject a
   missing model path, graph mode, version/module path, startup setting, digest,
   endpoint, nonce, or physical UUID.
6. `test_receipt_recheck_detects_socket_owner_replacement`: keep the recorded API
   PID alive while another process owns the port; formal recheck must fail.
7. `test_gpu_resolution_rejects_cuda_nvml_disagreement`: reject PCI-resolved and
   UUID-resolved handles that disagree, ambiguity, and malformed MIG identity.
8. `test_live_server_seal_request_creates_terminal_artifacts`: exercise the real
   scheduler/control path, not direct `EngineCore.seal_telemetry()` invocation.
9. `test_writer_open_before_seal_cannot_append_after_retirement`: create two
   writers, seal through one lifecycle, and require the other append to fail.
10. `test_vllm_and_smig_reject_identical_invalid_corpus`: forged instance ID,
    unknown telemetry field, invalid booleans/integers, invalid steps and broken
    process identity receive the expected refusal from both consumers.

Assert classified error codes and the specific violated invariant. Do not
replace these tests with implementation-shaped happy paths.

## 3. Complete and require the deployment expectation

**Files:**

- Modify: `gladius_vllm/receipt.py`
- Modify: `gladius_vllm/attest.py`
- Modify: `tests/gladius/test_server_start_receipt.py`

Extend `DeploymentExpectation` and its manifest parser to require:

- nonce, host/port, engine ID, model ID and canonical model path;
- physical GPU UUID and, where applicable, MIG UUID/profile/parent UUID;
- `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, memory utilization,
  prefix caching, chunked prefill, and CUDA graph mode;
- model, tokenizer, vLLM tree, native extension, and overlay digests plus hash
  algorithm version;
- vLLM package version and attestor/module version identity.

The verifier must recompute the canonical `server_instance_id`, process-start
identities, and current socket owner. `attest verify` must require exactly one
deployment manifest and use the same Python function as the server/SMIG contract
tests. No formal API overload may verify a receipt without the expectation.

## 4. Prove physical GPU identity across CUDA, NVML, and MIG

**Files:**

- Modify: `gladius_vllm/receipt.py`
- Modify: `tests/gladius/test_server_start_receipt.py`
- Create: `tests/gladius/test_h100_execution_plane_smoke.py`

Resolve the CUDA device's PCI bus ID and CUDA UUID. Independently resolve both
through NVML and require the same handle and authoritative UUID. Handle numeric,
reordered, UUID-form, and GPU/MIG-form `CUDA_VISIBLE_DEVICES`; never manufacture
`GPU-` prefixes. If MIG identity is exposed, bind the MIG UUID and parent GPU
identity explicitly. Ambiguity, missing corroboration, or disagreement fails
receipt publication.

The H100 smoke starts four processes under four masks and retains receipts plus:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader
```

Expected: four distinct receipt identities, each matching its intended physical
device and endpoint. A platform without MIG may skip only the real-MIG smoke,
not the pure protocol tests.

## 5. Make seal verification semantic, exhaustive, and expectation-bound

**Files:**

- Modify: `gladius_vllm/telemetry.py`
- Modify: `gladius_vllm/scheduler.py`
- Modify: `tests/gladius/test_telemetry_seal_v2.py`
- Modify: `tests/gladius/test_telemetry_writer.py`

Change the formal verifier interface so it requires the policy directory and
deployment expectation. It must enumerate actual telemetry segment files and
require exact equality with the manifest list. For every segment, parse exact
schema, verify hash, sequence/order/continuity, one nonempty server instance,
and independently re-derive record count, first/final step, generation high
watermark, and all advertised bounds.

Parse and semantically verify both siblings:

- receipt matches the complete deployment expectation and seal instance;
- application matches engine/model/instance, monotonic generation, requested and
  effective action, clamp state, policy/decision IDs, and a scheduler step in the
  sealed telemetry;
- the telemetry record at that step agrees with the application's effective
  action, generation, instance, and watermark;
- terminal telemetry and seal summary agree on all derived fields.

Sibling hashes are then checked as transport integrity. They are not the source
of semantic truth. The same-instance rewrite probes in §2 must fail even after
all local hashes are rebuilt.

## 6. Add a real, terminal production seal lifecycle

**Files:**

- Modify: `gladius_vllm/scheduler.py`
- Modify: `gladius_vllm/telemetry.py`
- Modify: `gladius_vllm/attest.py`
- Create: `tests/gladius/test_telemetry_seal_lifecycle.py`

Implement an instance-bound seal request file in the policy directory. Its exact
schema includes request ID, server instance, deployment-manifest digest, expected
final generation, and nonce. The live scheduler polls it at a safe scheduler
boundary, writes the terminal telemetry/application state, verifies the request
belongs to its instance, seals, writes `RETIRED`, and acknowledges the request
with manifest digest or classified failure.

The operator-facing command writes that request atomically and waits for the
instance-bound acknowledgement. If the scheduler needs a final step to observe
the request, SMIG sends one explicit discarded activation request and records it
outside formal workload metrics. A seal request for another instance, stale
generation, or changed deployment digest fails. Retrying the identical request
is idempotent; a different request after retirement fails.

Acceptance requires a launched server process to produce the seal and retirement
artifacts through this interface. A test that calls a method directly does not
satisfy the task.

## 7. Make retirement atomic against every writer

**Files:**

- Modify: `gladius_vllm/telemetry.py`
- Modify: `gladius_vllm/application.py`
- Create: `tests/gladius/test_telemetry_retirement.py`

Use a Linux file lock covering writer construction, append, rotation, seal, and
retirement. Every append rechecks retirement while holding the same lock. Seal
acquires exclusive ownership, flushes/fsyncs all evidence, verifies it, writes
the manifest and retirement marker atomically, fsyncs the directory, then
releases. A writer opened before sealing cannot append afterward. Partial seal
files are never considered valid and retries cannot silently replace a terminal
manifest.

## 8. Keep producer and consumer on one protocol corpus

**Files:**

- Modify: `tests/gladius/fixtures/gladius-third-review-corpus.json`
- Modify: `tests/gladius/test_execution_evidence_v2.py`
- Modify: `tests/gladius/test_gladius_contract_e2e.py`
- Coordinate the matching SMIG fixture manifest and test

Each fixture has a canonical SHA-256, expected acceptance/refusal, and expected
error class. Include valid receipt/telemetry/application/seal sets generated by
the real producer, plus one-invalid-property mutations. Neither repository may
maintain a private “compatible” parser test using fabricated server IDs.

A protocol change is incomplete until the paired SMIG commit consumes the same
corpus. The handoff records both commit IDs and archive digests.

## 9. Verification ladder and claim discipline

Run and retain full logs in this order:

```bash
python -m pytest tests/gladius/test_third_review_adversarial.py -q
python -m pytest tests/gladius/test_server_start_receipt.py -q
python -m pytest tests/gladius/test_telemetry_seal_v2.py tests/gladius/test_telemetry_writer.py -q
python -m pytest tests/gladius -q
python -m ruff check gladius_vllm tests/gladius
python -m ruff format --check gladius_vllm tests/gladius
```

Then execute Linux process/socket integration, followed by the four-process H100
identity and live-seal smoke. Report every skip and failure; an aggregate pass
count is not a substitute. Preserve deployment manifest, four receipts,
application/telemetry files, seal request/acknowledgement, seal, retirement
marker, GPU query, commands, environment, commit IDs, and SHA-256 values.

Create `docs/design/gladius_h100_discovery_execution_plane_third_review_handoff.md`
with one row per claim: claim, production caller, attack test, command, artifact,
result. Its status must be “candidate for third review”, not “all P0s complete”,
until an independent checkout reproduces the ladder.

## 10. Definition of done

- Every §2 adversarial test fails on `c311bc0` and passes unchanged on the new
  candidate.
- Same-instance receipt/application rewrites remain invalid after all local
  hashes are coherently rebuilt.
- Formal verification requires and consumes the complete deployment expectation.
- Linux proves current process-start and socket ownership; H100 proves four
  distinct CUDA/NVML-correlated physical identities.
- A live deployed scheduler, reached through the operator interface, creates an
  expectation-bound seal and terminal retirement artifacts.
- No writer created before or after sealing can append to the retired directory.
- vLLM and SMIG agree on every fixture and publish paired commit/archive digests.
- An independent reviewer reproduces all gates before SMIG is authorized to run
  pre-calibration or formal discovery.
