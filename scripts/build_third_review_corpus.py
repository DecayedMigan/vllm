#!/usr/bin/env python3
"""Generate the shared third-review protocol corpus for both repositories.

Valid cases are taken from evidence the *real producer* wrote -- the shared
execution-evidence v2 fixture, whose receipt's `server_instance_id` is a
genuine digest over real `/proc` process-start identities and therefore
re-derives identically on any machine. Invalid cases are single-property
mutations of those valid payloads, so each one isolates exactly the
invariant it names.

The output is written byte-identically to both trees. A protocol change is
incomplete until both parsers and this corpus move in the same paired
commit.

    python scripts/build_third_review_corpus.py --smig-root ~/CODE/gladius-...
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gladius_vllm.corpus import CORPUS_FILENAME, CORPUS_VERSION  # noqa: E402
from gladius_vllm.evidence_codes import (  # noqa: E402
    APPLICATION_SCHEMA_INVALID,
    RECEIPT_EXPECTATION_INCOMPLETE,
    RECEIPT_SCHEMA_INVALID,
    SEAL_REQUEST_SCHEMA_INVALID,
    SEAL_SCHEMA_INVALID,
    TELEMETRY_SCHEMA_INVALID,
)

SHARED_FIXTURE = (
    REPO_ROOT / "tests" / "gladius" / "fixtures" / "gladius-execution-evidence-v2.json"
)


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _case(
    case_id: str,
    kind: str,
    payload: Any,
    *,
    accepted: bool,
    error_code: str | None,
    violates: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "kind": kind,
        "violates": violates,
        "payload": payload,
        "payload_sha256": _sha256(payload),
        "expected_accepted": accepted,
        "expected_error_code": error_code,
    }


def _mutate(base: dict, **changes: Any) -> dict:
    payload = copy.deepcopy(base)
    for key, value in changes.items():
        if value is _REMOVE:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


class _Remove:
    pass


_REMOVE = _Remove()


def build_corpus() -> dict[str, Any]:
    fixture = json.loads(SHARED_FIXTURE.read_text())
    receipt = fixture["valid_server_start_receipt"]
    application = fixture["valid_applications"]["active"]
    native_application = fixture["valid_applications"]["fallback_native"]
    telemetry = fixture["valid_telemetry"]["file_backed"]
    native_telemetry = fixture["valid_telemetry"]["native"]
    seal = fixture["valid_telemetry_seal"]

    deployment = {
        "schema_version": "2.0.0",
        "attestation_nonce": receipt["attestation_nonce"],
        "engine_id": receipt["engine_id"],
        "model_id": receipt["model_id"],
        "model_path": receipt["model_path"],
        "listen_host": receipt["listen_host"],
        "listen_port": receipt["listen_port"],
        "physical_gpu_uuid": receipt["physical_gpu_uuid"],
        "physical_gpu_identity_source": receipt["physical_gpu_identity_source"],
        "mig_uuid": receipt["mig_uuid"],
        "mig_profile": receipt["mig_profile"],
        "mig_parent_gpu_uuid": receipt["mig_parent_gpu_uuid"],
        "model_tree_sha256": receipt["model_tree_sha256"],
        "tokenizer_tree_sha256": receipt["tokenizer_tree_sha256"],
        "vllm_package_tree_sha256": receipt["vllm_package_tree_sha256"],
        "vllm_native_binary_sha256": receipt["vllm_native_binary_sha256"],
        "gladius_overlay_tree_sha256": receipt["gladius_overlay_tree_sha256"],
        "tree_hash_algorithm_version": receipt["tree_hash_algorithm_version"],
        "vllm_version": receipt["vllm_version"],
        "vllm_module_path": receipt["vllm_module_path"],
        "gladius_overlay_path": receipt["gladius_overlay_path"],
        "startup_max_model_len": receipt["startup_max_model_len"],
        "startup_max_num_seqs": receipt["startup_max_num_seqs"],
        "startup_max_num_batched_tokens": receipt["startup_max_num_batched_tokens"],
        "gpu_memory_utilization": receipt["gpu_memory_utilization"],
        "prefix_caching_enabled": receipt["prefix_caching_enabled"],
        "chunked_prefill_enabled": receipt["chunked_prefill_enabled"],
        "enforce_eager": receipt["enforce_eager"],
        "cuda_graph_mode": receipt["cuda_graph_mode"],
    }
    seal_request = {
        "schema_version": "2.0.0",
        "request_id": "seal-" + receipt["server_instance_id"],
        "server_instance_id": receipt["server_instance_id"],
        "deployment_manifest_sha256": "0" * 64,
        "expected_final_generation": 42,
        "attestation_nonce": receipt["attestation_nonce"],
        "requested_at": "2026-08-02T09:05:00Z",
    }

    cases: list[dict[str, Any]] = [
        # --- valid, from the real producer -------------------------------
        _case(
            "receipt.valid",
            "server_start_receipt",
            receipt,
            accepted=True,
            error_code=None,
            violates="nothing",
        ),
        _case(
            "application.valid_active",
            "policy_application",
            application,
            accepted=True,
            error_code=None,
            violates="nothing",
        ),
        _case(
            "application.valid_native_fallback",
            "policy_application",
            native_application,
            accepted=True,
            error_code=None,
            violates="nothing",
        ),
        _case(
            "telemetry.valid_file_backed",
            "telemetry_record",
            telemetry,
            accepted=True,
            error_code=None,
            violates="nothing",
        ),
        _case(
            "telemetry.valid_native",
            "telemetry_record",
            native_telemetry,
            accepted=True,
            error_code=None,
            violates="nothing",
        ),
        _case(
            "seal.valid",
            "telemetry_seal",
            seal,
            accepted=True,
            error_code=None,
            violates="nothing",
        ),
        _case(
            "deployment.valid",
            "deployment_manifest",
            deployment,
            accepted=True,
            error_code=None,
            violates="nothing",
        ),
        _case(
            "seal_request.valid",
            "seal_request",
            seal_request,
            accepted=True,
            error_code=None,
            violates="nothing",
        ),
        # --- forged instance identity ------------------------------------
        _case(
            "receipt.forged_server_instance_id",
            "server_start_receipt",
            _mutate(receipt, server_instance_id="srv-" + "0" * 32),
            accepted=False,
            error_code=RECEIPT_SCHEMA_INVALID,
            violates="server_instance_id must be a digest over the receipt's own "
            "identity fields",
        ),
        _case(
            "receipt.changed_process_start_identity",
            "server_start_receipt",
            _mutate(receipt, engine_core_process_start_identity="boot-id:1"),
            accepted=False,
            error_code=RECEIPT_SCHEMA_INVALID,
            violates="a changed process-start identity must break the derived "
            "instance id",
        ),
        _case(
            "receipt.unknown_field",
            "server_start_receipt",
            _mutate(receipt, unexpected_field="x"),
            accepted=False,
            error_code=RECEIPT_SCHEMA_INVALID,
            violates="exact field set",
        ),
        _case(
            "receipt.missing_field",
            "server_start_receipt",
            _mutate(receipt, physical_gpu_identity_source=_REMOVE),
            accepted=False,
            error_code=RECEIPT_SCHEMA_INVALID,
            violates="exact field set",
        ),
        _case(
            "receipt.half_declared_mig_identity",
            "server_start_receipt",
            _mutate(receipt, mig_uuid="MIG-1111"),
            accepted=False,
            error_code=RECEIPT_SCHEMA_INVALID,
            violates="a MIG uuid with no parent names a device of unknown physical "
            "identity",
        ),
        _case(
            "receipt.unknown_identity_source",
            "server_start_receipt",
            _mutate(receipt, physical_gpu_identity_source="trust-me"),
            accepted=False,
            error_code=RECEIPT_SCHEMA_INVALID,
            violates="identity source must name how the device was established",
        ),
        _case(
            "receipt.non_integer_port",
            "server_start_receipt",
            _mutate(receipt, listen_port="8000"),
            accepted=False,
            error_code=RECEIPT_SCHEMA_INVALID,
            violates="type",
        ),
        # --- telemetry ----------------------------------------------------
        _case(
            "telemetry.unknown_field",
            "telemetry_record",
            _mutate(telemetry, unexpected_field=1),
            accepted=False,
            error_code=TELEMETRY_SCHEMA_INVALID,
            violates="exact field set",
        ),
        _case(
            "telemetry.missing_field",
            "telemetry_record",
            _mutate(telemetry, kv_cache_usage=_REMOVE),
            accepted=False,
            error_code=TELEMETRY_SCHEMA_INVALID,
            violates="exact field set",
        ),
        _case(
            "telemetry.boolean_step",
            "telemetry_record",
            _mutate(telemetry, step=True),
            accepted=False,
            error_code=TELEMETRY_SCHEMA_INVALID,
            violates="a bool is not an integer step",
        ),
        _case(
            "telemetry.negative_counter",
            "telemetry_record",
            _mutate(telemetry, num_running_reqs=-1),
            accepted=False,
            error_code=TELEMETRY_SCHEMA_INVALID,
            violates="counters are non-negative",
        ),
        _case(
            "telemetry.half_native_decision",
            "telemetry_record",
            _mutate(telemetry, policy_id=None),
            accepted=False,
            error_code=TELEMETRY_SCHEMA_INVALID,
            violates="generation, policy_id and decision_id move together",
        ),
        _case(
            "telemetry.unknown_policy_source",
            "telemetry_record",
            _mutate(telemetry, policy_source="snapshot"),
            accepted=False,
            error_code=TELEMETRY_SCHEMA_INVALID,
            violates="policy_source enum",
        ),
        # Legal at the record level and *not* at the seal level: a server
        # writes honestly-null records in the window between startup and
        # the attestor publishing its receipt. Refusing them here would
        # make the writer lie; the seal is where they become fatal.
        _case(
            "telemetry.valid_unattested_before_receipt",
            "telemetry_record",
            _mutate(telemetry, server_instance_id=None),
            accepted=True,
            error_code=None,
            violates="nothing at record level; a seal refuses these separately",
        ),
        # --- application ---------------------------------------------------
        _case(
            "application.unknown_field",
            "policy_application",
            _mutate(application, unexpected_field=1),
            accepted=False,
            error_code=APPLICATION_SCHEMA_INVALID,
            violates="exact field set",
        ),
        _case(
            "application.half_native",
            "policy_application",
            _mutate(application, generation=None),
            accepted=False,
            error_code=APPLICATION_SCHEMA_INVALID,
            violates="generation, policy_id and decision_id move together",
        ),
        _case(
            "application.active_but_clamped",
            "policy_application",
            _mutate(
                application,
                clamped={"max_num_seqs": True, "max_num_batched_tokens": False},
            ),
            accepted=False,
            error_code=APPLICATION_SCHEMA_INVALID,
            violates="an active application is effective and unclamped",
        ),
        _case(
            "application.watermark_below_generation",
            "policy_application",
            _mutate(application, generation_high_watermark=1),
            accepted=False,
            error_code=APPLICATION_SCHEMA_INVALID,
            violates="the watermark cannot be below the applied generation",
        ),
        _case(
            "application.negative_step",
            "policy_application",
            _mutate(application, scheduler_step=-1),
            accepted=False,
            error_code=APPLICATION_SCHEMA_INVALID,
            violates="scheduler_step is non-negative",
        ),
        # --- seal -----------------------------------------------------------
        _case(
            "seal.unknown_field",
            "telemetry_seal",
            _mutate(seal, unexpected_field=1),
            accepted=False,
            error_code=SEAL_SCHEMA_INVALID,
            violates="exact field set",
        ),
        _case(
            "seal.certifies_nothing",
            "telemetry_seal",
            _mutate(seal, files=[]),
            accepted=False,
            error_code=SEAL_SCHEMA_INVALID,
            violates="a seal must certify at least one segment",
        ),
        _case(
            "seal.zero_record_count",
            "telemetry_seal",
            _mutate(seal, record_count=0),
            accepted=False,
            error_code=SEAL_SCHEMA_INVALID,
            violates="a seal certifies at least one record",
        ),
        _case(
            "seal.unattested_instance",
            "telemetry_seal",
            _mutate(seal, server_instance_id=""),
            accepted=False,
            error_code=SEAL_SCHEMA_INVALID,
            violates="a seal names exactly one serving instance",
        ),
        # --- deployment manifest ---------------------------------------------
        _case(
            "deployment.missing_model_path",
            "deployment_manifest",
            _mutate(deployment, model_path=_REMOVE),
            accepted=False,
            error_code=RECEIPT_EXPECTATION_INCOMPLETE,
            violates="every expectation field is required",
        ),
        _case(
            "deployment.missing_cuda_graph_mode",
            "deployment_manifest",
            _mutate(deployment, cuda_graph_mode=_REMOVE),
            accepted=False,
            error_code=RECEIPT_EXPECTATION_INCOMPLETE,
            violates="every expectation field is required",
        ),
        _case(
            "deployment.truncated_digest",
            "deployment_manifest",
            _mutate(deployment, model_tree_sha256="abc"),
            accepted=False,
            error_code=RECEIPT_EXPECTATION_INCOMPLETE,
            violates="digests are lowercase sha256",
        ),
        # --- seal request ------------------------------------------------------
        _case(
            "seal_request.unknown_field",
            "seal_request",
            _mutate(seal_request, unexpected_field=1),
            accepted=False,
            error_code=SEAL_REQUEST_SCHEMA_INVALID,
            violates="exact field set",
        ),
        _case(
            "seal_request.empty_instance",
            "seal_request",
            _mutate(seal_request, server_instance_id=""),
            accepted=False,
            error_code=SEAL_REQUEST_SCHEMA_INVALID,
            violates="a seal request names the instance it is for",
        ),
        _case(
            "seal_request.negative_generation",
            "seal_request",
            _mutate(seal_request, expected_final_generation=-1),
            accepted=False,
            error_code=SEAL_REQUEST_SCHEMA_INVALID,
            violates="generation is non-negative or null",
        ),
    ]

    return {
        "corpus_version": CORPUS_VERSION,
        "description": (
            "One protocol corpus, byte-identical in the vLLM fork and SMIG. "
            "Both repositories run every case through their own production "
            "parser and must reach the same classified verdict. Valid cases "
            "come from evidence the real producer wrote; each invalid case "
            "mutates exactly one property of a valid payload."
        ),
        "source_fixture_sha256": hashlib.sha256(
            SHARED_FIXTURE.read_bytes()
        ).hexdigest(),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smig-root", type=Path, required=True)
    args = parser.parse_args(argv)

    corpus = build_corpus()
    text = json.dumps(corpus, indent=2, sort_keys=True) + "\n"

    targets = [
        REPO_ROOT / "tests" / "gladius" / "fixtures" / CORPUS_FILENAME,
        args.smig_root / "tests" / "fixtures" / CORPUS_FILENAME,
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    digest = hashlib.sha256(text.encode()).hexdigest()
    print(
        json.dumps(
            {
                "corpus_sha256": digest,
                "case_count": len(corpus["cases"]),
                "accepted": sum(1 for c in corpus["cases"] if c["expected_accepted"]),
                "refused": sum(
                    1 for c in corpus["cases"] if not c["expected_accepted"]
                ),
                "written": [str(target) for target in targets],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
