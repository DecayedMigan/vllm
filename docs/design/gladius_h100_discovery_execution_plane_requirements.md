# GLADIUS H100 Discovery Execution-Plane Remediation Requirements

**Date:** 2026-08-02

**Reviewed vLLM revision:** `fbcec19ed`

**Companion SMIG implementation reviewed:**
`865527d709d523062b85ae1c8e31c47fa32389e8`

**Status:** required execution-plane handoff; the four-H100 discovery campaign is
blocked until the P0 contract and shared acceptance tests in this document pass.

## 1. Scope and ownership

The existing GLADIUS scheduler correctly keeps learning outside vLLM and already
implements two-dimensional clamping, policy application acknowledgements,
per-step telemetry, and telemetry sealing. The realistic discovery experiment
adds a stricter evidence boundary: the client must be able to prove which live
server instance applied an action and produced a telemetry stream.

vLLM owns only the execution-plane facts in this document:

- live process, model, tokenizer, GPU, endpoint, binary, and startup identity;
- the scheduler's generation high-watermark and effective admission;
- application and telemetry records tied to one server instance;
- immutable sealing of that instance's telemetry.

SMIG owns Azure trace generation, exact prompt construction, open-loop request
timing, HTTP/stream evidence, pre-calibration, reward recomputation, cell resume,
monitoring, certified snapshots, oracle analysis, and the adaptive gate. vLLM
must not implement those client-side responsibilities.

This is an additive experiment-evidence milestone. Scheduler observation remains
fail-open for production serving. If an evidence write fails, serving continues,
but the H100 runner must see the missing or invalid receipt and fail the formal
attempt before sending registered workload.

## 2. Why the current contract is insufficient

Revision `fbcec19ed` identifies application records by stable `engine_id` and
`model_id`, but those values can survive a process restart. Neither
`policy_application.json`, `telemetry.jsonl`, nor `telemetry_seal.json` binds a
specific serving process. The external CLI attestation measures its own process
environment rather than proving the identity of the EngineCore process that
loaded the model and applied the policy.

The current policy loader also keeps its accepted-generation high-watermark only
in memory. After a policy expires, the published fallback acknowledgement can
map the native decision to synthetic generation `0` and policy ID
`startup-default`, so a resumed client cannot determine the last generation the
still-running scheduler will reject. This creates a generation-reuse risk.

Finally, the protocol permits consumers to represent impossible half-native
states unless they separately enforce that generation, policy ID, and decision
ID are either all present or all absent.

## 3. P0-A: nonce-bound server-start receipt

When `GLADIUS_POLICY_DIR` is configured, the API process and the EngineCore
process that owns `GladiusScheduler` must jointly produce and atomically publish:

```text
GLADIUS_POLICY_DIR/server_start_receipt.json
```

The launcher provides a unique, unpredictable `GLADIUS_ATTESTATION_NONCE` and
the expected endpoint/visible-GPU mapping. The final receipt is emitted only
after the API socket is bound and the EngineCore confirms that the model,
tokenizer, CUDA device, scheduler, prefix cache, chunked prefill, and startup
execution mode are initialized. It contains strictly parsed fields covering:

```text
schema_version
attestation_nonce
server_instance_id
created_at
api_pid
api_process_start_identity
engine_core_pid
engine_core_process_start_identity
engine_id
model_id
listen_host
listen_port
cuda_visible_devices
physical_gpu_uuid
physical_gpu_name
model_path
model_tree_sha256
tokenizer_tree_sha256
vllm_version
vllm_module_path
vllm_package_tree_sha256
vllm_native_binary_sha256
gladius_overlay_path
gladius_overlay_tree_sha256
tree_hash_algorithm_version
startup_max_model_len
startup_max_num_seqs
startup_max_num_batched_tokens
gpu_memory_utilization
prefix_caching_enabled
chunked_prefill_enabled
enforce_eager
cuda_graph_mode
```

`server_instance_id` is unique for one API/EngineCore process pair and is derived
from or cryptographically bound to the nonce, both PIDs, both process-start
identities, engine/model identity, and physical GPU UUID. A PID alone is
insufficient because it can be reused. The API-process contribution proves the
bound listen socket; the EngineCore contribution proves the scheduler and model
identity. The receipt must report the physical GPU UUID observed inside the
model process, not infer it only from the launcher's `CUDA_VISIBLE_DEVICES`
string.

Model and tokenizer digests use canonical, documented tree-hash algorithms that
include relative path, file type, size, and content digest in sorted order. The
vLLM package digest covers the imported Python package, and the native-binary
digest covers the actual loaded extension or an ordered manifest of all loaded
vLLM native extensions. Symlink handling and excluded transient files are fixed
in the algorithm version; no digest may mean “current working tree” without
enumerating the imported files.

The formal Qwen3-8B receipt must prove these startup values:

```text
max_model_len = 8192
max_num_seqs = 32
max_num_batched_tokens = 8192
gpu_memory_utilization = 0.75
prefix_caching_enabled = true
chunked_prefill_enabled = true
enforce_eager = false
```

The receipt is written using a same-directory temporary file, file `fsync`,
`os.replace`, and directory `fsync`. A policy directory may contain only one
server-instance receipt. A second process must not overwrite a different
receipt; it must use a new directory. Receipt failure is logged and leaves the
server available, but formal readiness remains false to the external campaign.

## 4. P0-B: instance-bound application acknowledgement

The next application schema adds these required fields:

```text
server_instance_id
generation_high_watermark
```

Every acknowledgement must bind the server-start receipt by exact
`server_instance_id`, `engine_id`, and `model_id`. For a file policy in
`active` or `attriting` state:

```text
generation is a non-negative integer
policy_id == decision_id is nonempty
generation_high_watermark >= generation
```

For native/default fallback:

```text
generation == null
policy_id == null
decision_id == null
generation_high_watermark == null or the greatest generation previously
accepted by this server instance
```

The high-watermark is not reset by expiry, a corrupt snapshot, a mismatched
snapshot, fallback to startup admission, or an equal/regressed generation. It
resets only when `server_instance_id` changes. This value gives a resumed SMIG
lane a safe lower bound for its next generation without adding a network call
inside scheduling.

State semantics remain strict:

- `active` requires requested admission equal to effective admission and both
  clamp flags false;
- `attriting` is allowed only while an already-running sequence count prevents
  the requested lower sequence ceiling from becoming effective;
- every other mismatch is `fallback` and cannot certify a formal cell.

The writer still rewrites only on a meaningful state/identity/effective-value
change and remains fail-open. Its parser must reject unknown fields for the
selected schema, invalid timestamps, half-native identities, a high-watermark
below the active generation, and an application whose instance ID does not match
the receipt.

## 5. P0-C: instance-bound telemetry with explicit native invariants

Every sampled scheduler record adds `server_instance_id` and
`generation_high_watermark`. The following invariant is mandatory in both the
vLLM writer/parser and the shared SMIG parser:

```text
(generation is None) == (policy_id is None) == (decision_id is None)
```

When the decision is native/default, all three are null. When it is file-backed,
all three are present and `policy_id == decision_id`. A half-native record is
invalid; consumers must not repair it.

For a formal campaign, `GLADIUS_TELEMETRY_SAMPLE_N=1` is required. Scheduler
steps are strictly increasing within one server instance. The application step
must be present in the same instance's nonempty telemetry stream, and all
subsequent records used by a cell must report the cell's expected requested and
effective action without a clamp. Missing samples, duplicate steps, instance
changes, or a generation regression make the cell structurally invalid on the
SMIG side.

The execution plane need not attach request IDs or workload-window IDs to
scheduler telemetry. The client links requests to the application step and the
bounded telemetry interval. This avoids introducing request-trace knowledge into
the scheduler.

## 6. P0-D: stronger telemetry seal

`telemetry_seal.json` must add:

```text
server_instance_id
attestation_receipt_sha256
first_scheduler_step
final_scheduler_step
record_count
generation_high_watermark
policy_application_sha256
```

The seal hashes every telemetry segment in deterministic logical order. It must
reject duplicate filenames and confirm that JSONL records are parseable, belong
to exactly one server instance, have strictly increasing step identities, and
match the manifest's record count and bounds before returning success. An empty
telemetry set cannot be sealed successfully for a formal attempt.

The application digest is captured after the last scheduler step and binds the
last observed state. The receipt digest binds the immutable process/startup
identity. Sealing closes the writer before hashing and atomically publishes the
manifest. Post-seal requests continue serving but cannot append, rotate, replace,
or otherwise mutate any certified segment. Keepalive traffic must use a new
policy directory and new server instance/receipt.

## 7. Shared protocol fixture and schema transition

The vLLM and SMIG repositories must commit byte-identical fixtures for:

- a valid server-start receipt;
- valid active, attriting, and fallback applications;
- valid file-backed and native telemetry records;
- a valid nonempty telemetry seal;
- invalid half-native, regressed-generation, clamped-active, cross-instance,
  empty-seal, and receipt-mismatch cases.

Both repositories report the shared fixture's SHA-256 in their handoff.
`policy_snapshot.json` remains schema `1.0.0`. The server-start receipt,
application, telemetry, and seal defined here use execution-evidence schema
`2.0.0`; the new parser is strict and coexists with the historical 1.x parser.
A parser must never accept a 1.x artifact as if it provides the 2.0.0
server-instance guarantee. Formal discovery accepts only execution-evidence
schema `2.0.0`.

## 8. Required tests

The following pure-Python tests are P0 and must pass before an H100 archive is
created.

1. Two-process receipt assembly is atomic, directory-synced, nonce-bound, and
   refuses to overwrite another instance.
2. Receipt parsing rejects a wrong nonce, either PID/start mismatch, wrong GPU
   UUID, altered digest, eager mode, disabled prefix caching/chunked prefill, and
   any startup ceiling mismatch.
3. Application records carry the receipt's server instance and exact active
   requested/effective action.
4. Fallback after an accepted policy preserves `generation_high_watermark` while
   making generation/policy/decision all null.
5. Equal and lower generations remain rejected after expiry or corrupt input;
   the high-watermark does not regress.
6. Native telemetry has three null identity fields; file telemetry has all
   three present; every half-native permutation is rejected.
7. Telemetry and application records from a second server instance cannot be
   combined with the first instance's receipt or seal.
8. Sealing rejects no records, malformed JSONL, duplicate/nonmonotonic steps,
   mixed instance IDs, an altered application, or an altered receipt.
9. After a successful seal, scheduling remains functional and every certified
   file remains byte-identical.
10. The installed vLLM 0.25.1 compatibility smoke still accepts both supported
    `schedule()` call shapes and emits each new field exactly once per sampled
    step.

Run Ruff and the complete `tests/gladius` suite through the repository's `uv`
environment. Report exact commands, pass/fail/skip counts, and the shared fixture
digest. No test in receipt, policy loader, application, telemetry, scheduler, or
shared-contract code may be waived as environment-only.

## 9. H100 acceptance sequence

For each GPU 0-3, the implementation team must demonstrate this sequence before
SMIG starts pre-calibration:

1. launch Qwen3-8B in a new versioned policy directory with a unique nonce;
2. validate the receipt against the expected port, engine ID, GPU UUID, model,
   digests, and frozen startup settings;
3. send one discarded activation request and observe an unclamped active
   application plus a matching scheduler telemetry step;
4. publish a higher generation, observe the same server instance and the new
   high-watermark, then reject a deliberate equal/lower generation;
5. seal a short smoke stream and validate its receipt/application/telemetry
   linkage;
6. start keepalive only in a separate policy directory after the experimental
   writer has been retired.

The handoff reports the exact vLLM commit, checksummed `gladius_vllm` archive,
shared fixture SHA-256, imported package/native-binary digests, four receipt
SHA-256 values, and test results. A successful real completion without these
artifacts is a serving smoke, not certified discovery readiness.

## 10. Campaign authorization boundary

Completion of this vLLM contract is necessary but not sufficient for the H100
campaign. SMIG must separately pass its trace, exact-token, pre-calibration,
semantic-validator, resume, monitoring, certification, and gate acceptance
requirements. Until both repository handoffs pass independent review:

- do not SCP a formal campaign archive;
- do not start the 864-cell discovery run;
- do not describe either repository as H100 discovery ready;
- do not start an adaptive GLADIUS pilot.
