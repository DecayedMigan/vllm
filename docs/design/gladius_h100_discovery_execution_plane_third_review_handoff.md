# GLADIUS H100 Execution Plane — Third-Review Handoff

**Status: candidate for third review.** Not "all P0s complete". The plan is
explicit that this document may say "implemented" only after an independent
reviewer checks out the published commits in clean directories and
reproduces every gate. That has not happened.

Controlling plan:
[`2026-08-02-h100-execution-plane-third-review-remediation.md`](../superpowers/plans/2026-08-02-h100-execution-plane-third-review-remediation.md)
(commit `01dae4d2`). Rejected revision: `c311bc0`.

Paired SMIG commit and handoff are named in
[§6](#6-paired-commits-and-archive-digests).

---

## 1. What the retained red actually shows

The plan required each §2 test to fail on `c311bc0`. All 29 do
(`docs/superpowers/artifacts/third-review/red/vllm-adversarial-red-c311bc05.log`),
but most fail on the new signature, which is weak evidence. The substantive
artifact is
`docs/superpowers/artifacts/third-review/red/vllm-attacks-accepted-by-c311bc05.log`,
which runs the attacks against **that revision's own two-argument
`verify_telemetry_seal`**:

```
baseline valid                                          -> ACCEPTED
ATTACK application action rewrite + rehash              -> ACCEPTED
ATTACK receipt startup rewrite + rehash                 -> ACCEPTED
ATTACK unlisted extra segment on disk                   -> ACCEPTED
ATTACK application step 999 not in sealed telemetry     -> ACCEPTED

4 of 4 attacks ACCEPTED by the rejected revision c311bc05
```

The baseline being ACCEPTED is the load-bearing line: the attacks start from
evidence the rejected revision genuinely accepts, so their acceptance is the
defect and not a malformed fixture.

## 2. Claim matrix

Every row: claim → production call site → attack test → command → artifact →
result. A row with a missing cell is not delivered.

| # | Claim | Production call site | Attack test | Artifact | Result |
|---|---|---|---|---|---|
| A1 | A rehashed acknowledgement rewrite is refused | `telemetry.py::_verify_sealed_siblings` (called by `verify_telemetry_seal`, called by `attest verify`/SMIG) | `test_seal_rejects_rehashed_application_action_rewrite` | red log above | PASS |
| A2 | A rehashed receipt deployment rewrite is refused | same | `test_seal_rejects_rehashed_receipt_startup_rewrite` | red log above | PASS |
| A3 | The certified segment set equals the directory | `telemetry.py::telemetry_segment_names` | `test_seal_rejects_unlisted_and_missing_segments` (3 cases) | red log above | PASS |
| A4 | The acknowledgement is bound to a sealed step | `telemetry.py::_verify_sealed_siblings` | `test_seal_binds_application_to_telemetry_step` (4 cases) | red log above | PASS |
| B1 | Formal verification consumes the whole expectation | `receipt.py::DeploymentExpectation.from_dict` | `test_formal_verify_requires_complete_deployment_expectation` (11 fields) | — | PASS |
| B2 | No overload verifies without an expectation | `verify_server_start_receipt` **deleted**; `attest verify --deployment-manifest` is `required=True` | `test_no_formal_api_verifies_a_receipt_without_an_expectation`, `test_attest_verify_cannot_be_invoked_without_a_deployment_manifest` | — | PASS |
| C1 | Socket ownership is re-measured, not remembered | `receipt.py::_verify_live_identity` → `netowner.listen_socket_owner_pid` | `test_receipt_recheck_detects_socket_owner_replacement` (real second process holds the port) | — | PASS |
| D1 | CUDA and NVML must corroborate | `receipt.py::resolve_physical_gpu_identity`, called by `_resolve_physical_gpu` at startup | `test_gpu_resolution_rejects_cuda_nvml_disagreement` | — | PASS |
| D2 | One resolvable identity is not corroboration | same | `test_gpu_resolution_rejects_uncorroborated_identity`, `test_a_stale_pci_id_no_longer_falls_through_to_the_uuid` | — | PASS |
| D3 | `GPU-` is never prepended to a MIG UUID | `receipt.py::_nvml_uuid_candidate` | `test_gpu_resolution_never_manufactures_a_gpu_prefix` | — | PASS |
| E1 | The **live scheduler** produces the seal | `scheduler.py::GladiusScheduler._poll_seal_request`, called from `schedule()` | `test_live_server_seal_request_creates_terminal_artifacts` (real `GladiusScheduler`, real attestation handshake) | four-process smoke report | PASS |
| E2 | A foreign/stale/changed seal request is refused | `seal_lifecycle.py::classify_refusal` | `test_third_review_adversarial` seal-request cases; corpus `seal_request.*` | corpus | PASS |
| F1 | A writer opened before sealing cannot append | `telemetry.py::TelemetryWriter.append_record` under `evidence_lock` | `test_writer_open_before_seal_cannot_append_after_retirement` | — | PASS |
| F2 | Concurrent writers cannot interleave across a seal | `evidence_lock.py::evidence_lock` (exclusive on seal/retire) | `test_concurrent_writers_cannot_interleave_across_a_seal` | — | PASS |
| G1 | Both repositories refuse the same corpus | `corpus.py::evaluate_corpus_case` / SMIG `execution_evidence.evaluate_corpus_case` | `test_vllm_and_smig_reject_identical_invalid_corpus` (37 cases) | `tests/gladius/fixtures/gladius-third-review-corpus.json` | PASS |

**A finding the corpus produced, not a check that merely passed:** it caught
three real divergences where SMIG's telemetry parser accepted documents the
producer refuses — unknown field, missing field, unknown `policy_source`.
Those are fixed in the paired SMIG commit. That is the corpus doing the job
it exists for.

## 3. Verification ladder

```bash
python -m pytest tests/gladius/test_third_review_adversarial.py -q        # 29 passed
python -m pytest tests/gladius/test_server_start_receipt.py -q            # included below
python -m pytest tests/gladius/test_telemetry_seal_v2.py \
                tests/gladius/test_telemetry_writer.py -q                 # included below
python -m pytest tests/gladius -q                                         # 263 passed
python -m ruff check gladius_vllm tests/gladius                           # clean
python -m ruff format --check gladius_vllm tests/gladius                  # clean
```

`263 passed`, **0 failed, 0 skipped** in `tests/gladius`. The paired-corpus
test skips unless `GLADIUS_SMIG_ROOT` is set; the 263 figure is with it set.

## 4. Four-process launched-server smoke

`scripts/four_process_execution_plane_smoke.sh` launches four real
`vllm.entrypoints.openai.api_server` processes with four policy directories,
four nonces, and four ports, then drives the seal lifecycle through them.

Retained under `docs/superpowers/artifacts/third-review/four-process-smoke/`.

What it establishes on real processes rather than mocks: four distinct
`server_instance_id`s derived from real `/proc` process pairs, per-lane
listen-socket ownership read from `/proc/net/tcp`, `api_pid !=
engine_core_pid`, and the seal + `RETIRED` + acknowledgement produced by the
live scheduler with this script only writing the request file.

## 5. What this delivery does **not** establish

- **Four distinct physical GPU UUIDs.** One card. `physical_gpu_uuid` is the
  same for all four lanes, so SMIG's `validate_lane_set` refuses the set —
  correctly. The CUDA/NVML corroboration path is exercised against a fake
  NVML with four physically-indexed devices, never against four H100s.
- **Real MIG identity.** No MIG hardware. The MIG parse, parent resolution,
  and prefix rules are tested against a fake.
- **Formal-grade physical identity on this host.** ROCm has no NVML, so
  receipts here carry `physical_gpu_identity_source: "rocm-torch"`, which is
  excluded from `FORMAL_IDENTITY_SOURCES`. The smoke records the classified
  refusal a formal manifest produces against this host. That is the gate
  working, and it also means the smoke's seal lifecycle is driven by writing
  the request file directly rather than through `attest seal`, whose receipt
  pre-check correctly refuses this hardware.
- **Reviewer reproduction.** Nobody has checked this out clean and re-run the
  ladder. Until they do, the status stays "candidate".

## 6. Paired commits and archive digests

| Repository | Branch | Commit |
|---|---|---|
| vLLM fork | `feature/gladius-v3-vllm-engine` | see `git log` at handoff time |
| SMIG | `agent/h100-discovery` | see `git log` at handoff time |

| Shared artifact | sha256 |
|---|---|
| `gladius-execution-evidence-v2.json` | `5aa6974a06d78841ed414fd1dfa118cac0aee2cc2fbf32fa6d77b925c1d4e955` |
| `gladius-third-review-corpus.json` | `739250d4d929dcaca3032d7506b136db1cb5ac84354fef349cc7a58318182b4c` |

Both files are byte-identical in the two trees; the digests above are
asserted by `tests/integration/test_vllm_protocol_contract.py` on the SMIG
side and by `test_vllm_and_smig_reject_identical_invalid_corpus` here.

## 7. Authorization

None. No H100 pre-calibration, formal discovery run, or adaptive pilot is
authorized by this delivery. The plan reserves that for a separate reviewer
authorization after these gates are independently reproduced.
