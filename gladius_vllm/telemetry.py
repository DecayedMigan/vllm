"""Append-only telemetry.jsonl writer, one line per scheduling step.

Owned directly by GladiusScheduler (not the StatLoggerBase path) since it
already has everything it needs -- the scheduler instance, the fresh
SchedulerOutput, and the current PolicyDecision -- with no cross-process
ambiguity. See gladius_vllm.stat_logger for the optional secondary path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gladius_vllm.atomic import atomic_write_json
from gladius_vllm.evidence_codes import (
    SEAL_APPLICATION_SEMANTIC_MISMATCH,
    SEAL_APPLICATION_STEP_UNSEALED,
    SEAL_BOUNDS_MISMATCH,
    SEAL_INSTANCE_MISMATCH,
    SEAL_RECEIPT_EXPECTATION_MISMATCH,
    SEAL_SCHEMA_INVALID,
    SEAL_SEGMENT_CONTENT_MISMATCH,
    SEAL_SEGMENT_SET_MISMATCH,
    TELEMETRY_DIRECTORY_RETIRED,
    TELEMETRY_SCHEMA_INVALID,
    classify,
)
from gladius_vllm.evidence_lock import evidence_lock
from gladius_vllm.policy import PolicyDecision
from gladius_vllm.schema import (
    DEFAULT_TELEMETRY_MAX_BYTES,
    DEFAULT_TELEMETRY_SAMPLE_N,
    EXECUTION_EVIDENCE_SCHEMA_VERSION,
    POLICY_SOURCES,
    POLICY_STATUSES,
    format_iso8601,
    parse_int_env,
    parse_iso8601,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)

POLICY_APPLICATION_FILENAME = "policy_application.json"
SERVER_START_RECEIPT_FILENAME = "server_start_receipt.json"
TELEMETRY_SEAL_FILENAME = "telemetry_seal.json"
RETIRED_MARKER_FILENAME = "RETIRED"


def _is_retired(policy_dir: Path) -> bool:
    """True once a directory holds certified evidence that must not change."""
    return (policy_dir / TELEMETRY_SEAL_FILENAME).exists() or (
        policy_dir / RETIRED_MARKER_FILENAME
    ).exists()


def retire_policy_directory(policy_dir: Path) -> None:
    """Mark a policy directory closed to any further telemetry writer.

    Taken under the exclusive evidence lock so it cannot interleave with an
    append already in flight: after this returns, every writer -- including
    one opened long before -- observes the directory as retired.
    """
    policy_dir = Path(policy_dir)
    policy_dir.mkdir(parents=True, exist_ok=True)
    with evidence_lock(policy_dir, exclusive=True):
        _write_retirement_marker(policy_dir)


def _write_retirement_marker(policy_dir: Path) -> None:
    """Retirement without acquiring the lock.

    `flock` is per-open-file-description, so a caller that already holds the
    exclusive lock would block forever waiting on itself. The seal path is
    exactly such a caller.
    """
    atomic_write_json(
        policy_dir / RETIRED_MARKER_FILENAME,
        {
            "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
            "retired_at": format_iso8601(),
        },
    )


class TelemetrySealError(ValueError):
    """A telemetry stream could not be certified as immutable evidence."""


# The exact field set the writer emits. Formal evidence parsing requires
# all of it and nothing else: a record with an unknown field was produced by
# code this contract does not describe, and one with a missing field cannot
# be validated at all.
TELEMETRY_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "server_instance_id",
        "generation_high_watermark",
        "generation",
        "policy_id",
        "decision_id",
        "window_id",
        "model_id",
        "engine_id",
        "created_at",
        "expires_at",
        "step",
        "num_running_reqs",
        "num_waiting_reqs",
        "num_skipped_waiting_reqs",
        "num_scheduled_reqs",
        "num_scheduled_tokens",
        "num_prefill_reqs",
        "num_decode_reqs",
        "kv_cache_usage",
        "policy_status",
        "policy_source",
        "requested_admission",
        "effective_admission",
        "clamped",
        "policy_poll_ns",
        "policy_apply_ns",
        "telemetry_write_ns",
    }
)
_ADMISSION_FIELDS = frozenset({"max_num_seqs", "max_num_batched_tokens"})
_COUNTER_FIELDS = (
    "num_running_reqs",
    "num_waiting_reqs",
    "num_skipped_waiting_reqs",
    "num_scheduled_reqs",
    "num_scheduled_tokens",
    "num_prefill_reqs",
    "num_decode_reqs",
    "policy_poll_ns",
    "policy_apply_ns",
    "telemetry_write_ns",
)
_POLICY_STATUSES = frozenset(POLICY_STATUSES)
_POLICY_SOURCES = frozenset(POLICY_SOURCES)


def _seal_int(payload: dict, field: str, *, minimum: int) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TelemetrySealError(f"{field} must be an integer >= {minimum}")
    return value


def _seal_admission(payload: dict, field: str) -> tuple[int, int]:
    block = payload[field]
    if not isinstance(block, dict) or set(block) != set(_ADMISSION_FIELDS):
        raise TelemetrySealError(f"{field} does not match the contract")
    return (
        _seal_int(block, "max_num_seqs", minimum=1),
        _seal_int(block, "max_num_batched_tokens", minimum=1),
    )


def parse_telemetry_record_v2(payload: object) -> dict[str, Any]:
    """Strictly validate one execution-evidence 2.0.0 telemetry record.

    Schema-strict: the exact field set, with types, ranges, timestamps,
    queue/admission values, clamp invariants, policy status/source, and
    engine/model identity all checked. A subset check would let a record
    carrying an unrecognised field -- i.e. produced by something other than
    this writer -- into a formal seal.

    The native-state invariant is enforced rather than repaired: a record
    claiming a generation but no policy id describes a scheduler state that
    cannot exist, so accepting it would mean inventing evidence.
    """
    if not isinstance(payload, dict):
        raise TelemetrySealError("telemetry record must be a JSON object")
    fields = set(payload)
    if fields != set(TELEMETRY_RECORD_FIELDS):
        missing = sorted(TELEMETRY_RECORD_FIELDS - fields)
        unknown = sorted(fields - TELEMETRY_RECORD_FIELDS)
        raise TelemetrySealError(
            f"telemetry record fields mismatch: missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise TelemetrySealError(
            "telemetry requires execution-evidence schema "
            f"{EXECUTION_EVIDENCE_SCHEMA_VERSION}"
        )
    for field in ("engine_id", "model_id"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise TelemetrySealError(f"{field} must be a non-empty string")
    _seal_int(payload, "step", minimum=1)

    identity = (payload["generation"], payload["policy_id"], payload["decision_id"])
    present = tuple(value is not None for value in identity)
    if len(set(present)) != 1:
        raise TelemetrySealError(
            "half-native telemetry record: generation, policy_id, and "
            "decision_id must be all present or all null"
        )
    if present[0]:
        generation, policy_id, decision_id = identity
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise TelemetrySealError("generation must be an integer")
        if generation < 0:
            raise TelemetrySealError("generation must be non-negative")
        if policy_id != decision_id:
            raise TelemetrySealError("policy_id and decision_id must be identical")
        if not isinstance(policy_id, str) or not policy_id:
            raise TelemetrySealError("policy_id must be a non-empty string")

    watermark = payload["generation_high_watermark"]
    if watermark is not None:
        if (
            isinstance(watermark, bool)
            or not isinstance(watermark, int)
            or watermark < 0
        ):
            raise TelemetrySealError(
                "generation_high_watermark must be a non-negative integer or null"
            )
        if present[0] and watermark < payload["generation"]:
            raise TelemetrySealError(
                "generation_high_watermark must be >= the applied generation"
            )

    instance = payload["server_instance_id"]
    if instance is not None and (not isinstance(instance, str) or not instance):
        raise TelemetrySealError(
            "server_instance_id must be a non-empty string or null"
        )

    if payload["window_id"] is not None and not isinstance(payload["window_id"], str):
        raise TelemetrySealError("window_id must be a string or null")
    try:
        parse_iso8601(payload["created_at"])
    except (TypeError, ValueError) as error:
        raise TelemetrySealError(f"invalid created_at: {error}") from error
    if payload["expires_at"] is not None:
        try:
            parse_iso8601(payload["expires_at"])
        except (TypeError, ValueError) as error:
            raise TelemetrySealError(f"invalid expires_at: {error}") from error

    for field in _COUNTER_FIELDS:
        _seal_int(payload, field, minimum=0)
    usage = payload["kv_cache_usage"]
    if usage is not None:
        if isinstance(usage, bool) or not isinstance(usage, (int, float)):
            raise TelemetrySealError("kv_cache_usage must be a number or null")
        if not 0.0 <= float(usage) <= 1.0:
            raise TelemetrySealError("kv_cache_usage must be in [0, 1]")
    if (
        payload["num_prefill_reqs"] + payload["num_decode_reqs"]
        != (payload["num_scheduled_reqs"])
    ):
        raise TelemetrySealError(
            "num_prefill_reqs + num_decode_reqs must equal num_scheduled_reqs"
        )

    if payload["policy_status"] not in _POLICY_STATUSES:
        raise TelemetrySealError("policy_status is not a known status")
    if payload["policy_source"] not in _POLICY_SOURCES:
        raise TelemetrySealError("policy_source is not a known source")
    # A native/default decision is the only one that can have no identity,
    # and a file-backed decision always has one.
    if (payload["policy_source"] == "default") != (not present[0]):
        raise TelemetrySealError(
            "policy_source and the identity fields disagree about whether a "
            "file-backed policy is in force"
        )

    requested = _seal_admission(payload, "requested_admission")
    effective = _seal_admission(payload, "effective_admission")
    clamped_block = payload["clamped"]
    if not isinstance(clamped_block, dict) or set(clamped_block) != set(
        _ADMISSION_FIELDS
    ):
        raise TelemetrySealError("clamped does not match the contract")
    clamped = tuple(clamped_block[field] for field in sorted(_ADMISSION_FIELDS))
    if not all(isinstance(value, bool) for value in clamped):
        raise TelemetrySealError("clamped values must be boolean")
    # The clamp flags are derived, not independent: each must say exactly
    # whether that dimension's effective value differs from the requested one.
    expected_clamps = {
        "max_num_seqs": effective[0] != requested[0],
        "max_num_batched_tokens": effective[1] != requested[1],
    }
    if clamped_block != expected_clamps:
        raise TelemetrySealError(
            "clamp flags do not describe the requested/effective admission"
        )
    return payload


def _count_prefill_decode(scheduler: Any, output: SchedulerOutput) -> tuple[int, int]:
    """Derive prefill/decode counts from SchedulerOutput + Request state.

    A request counts as "prefill" this step if it's a brand-new admission (in
    scheduled_new_reqs) or a cached continuation still mid-prompt
    (num_computed_tokens < num_prompt_tokens, i.e. a chunked-prefill
    continuation); everything else scheduled this step is "decode".
    """
    new_req_ids = {req.req_id for req in output.scheduled_new_reqs}
    num_prefill = 0
    num_decode = 0
    for req_id in output.num_scheduled_tokens:
        if req_id in new_req_ids:
            num_prefill += 1
            continue
        request = scheduler.requests.get(req_id)
        if (
            request is not None
            and request.num_computed_tokens < request.num_prompt_tokens
        ):
            num_prefill += 1
        else:
            num_decode += 1
    return num_prefill, num_decode


def _resolve_sample_every_n_steps(explicit: int | None) -> int:
    """Never returns < 1: a 0 or negative value would ZeroDivisionError in
    record()'s modulo check, and telemetry must never be able to crash
    scheduling."""
    if explicit is not None:
        return explicit if explicit >= 1 else DEFAULT_TELEMETRY_SAMPLE_N
    return parse_int_env(
        "GLADIUS_TELEMETRY_SAMPLE_N", DEFAULT_TELEMETRY_SAMPLE_N, minimum=1
    )


def _resolve_max_bytes(explicit: int | None) -> int:
    if explicit is not None:
        return explicit if explicit >= 1 else DEFAULT_TELEMETRY_MAX_BYTES
    return parse_int_env(
        "GLADIUS_TELEMETRY_MAX_BYTES", DEFAULT_TELEMETRY_MAX_BYTES, minimum=1
    )


SEAL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "engine_id",
        "model_id",
        "sealed_at",
        "server_instance_id",
        "attestation_receipt_sha256",
        "policy_application_sha256",
        "first_scheduler_step",
        "final_scheduler_step",
        "record_count",
        "generation_high_watermark",
        "files",
    }
)


def parse_telemetry_seal(payload: object) -> dict[str, Any]:
    """Strictly validate a `telemetry_seal.json` manifest.

    Exact field set: a manifest missing a field cannot be re-derived, and one
    carrying an extra field was not produced by this writer.
    """
    if not isinstance(payload, dict):
        raise TelemetrySealError("telemetry seal must be a JSON object")
    fields = set(payload)
    if fields != set(SEAL_MANIFEST_FIELDS):
        missing = sorted(SEAL_MANIFEST_FIELDS - fields)
        unknown = sorted(fields - SEAL_MANIFEST_FIELDS)
        raise TelemetrySealError(
            f"telemetry seal fields mismatch: missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise TelemetrySealError(
            "telemetry seal requires execution-evidence schema "
            f"{EXECUTION_EVIDENCE_SCHEMA_VERSION}"
        )
    for field in (
        "engine_id",
        "model_id",
        "server_instance_id",
        "attestation_receipt_sha256",
        "policy_application_sha256",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            raise TelemetrySealError(f"{field} must be a non-empty string")
    for field in ("first_scheduler_step", "final_scheduler_step", "record_count"):
        _seal_int(payload, field, minimum=1)
    if payload["generation_high_watermark"] is not None:
        _seal_int(payload, "generation_high_watermark", minimum=0)
    try:
        parse_iso8601(payload["sealed_at"])
    except (TypeError, ValueError) as error:
        raise TelemetrySealError(f"invalid sealed_at: {error}") from error
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise TelemetrySealError("a seal must certify at least one segment")
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"name", "size", "sha256"}:
            raise TelemetrySealError(
                "each certified file entry must be name/size/sha256"
            )
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise TelemetrySealError("certified file name must be a non-empty string")
        _seal_int(entry, "size", minimum=0)
        if not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64:
            raise TelemetrySealError("certified file sha256 must be a sha256 digest")
    return payload


def telemetry_segment_names(policy_dir: Path, *, live_name: str) -> set[str]:
    """Every telemetry segment actually present in `policy_dir`.

    The verifier must enumerate the directory rather than walk the manifest.
    A manifest-only walk cannot see a segment the manifest does not list, so
    an attacker could drop an inconvenient segment from certification while
    leaving it on disk and the seal would still verify.
    """
    policy_dir = Path(policy_dir)
    names = {candidate.name for candidate in policy_dir.glob(f"{live_name}.*-*")}
    if (policy_dir / live_name).is_file():
        names.add(live_name)
    return names


def verify_telemetry_seal(
    manifest_path: Path,
    policy_dir: Path,
    *,
    expectation: Any,
    live_name: str = "telemetry.jsonl",
) -> list[str]:
    """Re-derive everything a seal claims, semantically, from disk.

    Two properties the reviewed verifier lacked and which this exists to
    provide:

    * the certified file *set* is compared against the segments actually
      present, not merely against the manifest's own list;
    * both siblings are checked for agreement with the deployment
      expectation and with the sealed telemetry stream, so a coherent
      rewrite -- edit a field, rebuild its digest, rebuild the manifest --
      still fails. A checksum match proves the bytes were copied intact and
      nothing more.

    `expectation` is required. There is no variant of this function that
    verifies a seal without one.

    Returns every discrepancy rather than raising on the first, so one pass
    gives the complete diagnosis.
    """
    manifest_path = Path(manifest_path)
    policy_dir = Path(policy_dir)
    if expectation is None:
        return [
            classify(
                SEAL_RECEIPT_EXPECTATION_MISMATCH,
                "a formal seal cannot be verified without a deployment "
                "expectation; a skipped check is not a pass",
            )
        ]
    try:
        manifest = parse_telemetry_seal(json.loads(manifest_path.read_text()))
    except (OSError, json.JSONDecodeError, TelemetrySealError) as error:
        return [classify(SEAL_SCHEMA_INVALID, f"telemetry seal unusable: {error}")]

    errors: list[str] = []
    listed = [entry["name"] for entry in manifest["files"]]
    if len(set(listed)) != len(listed):
        duplicates = sorted({name for name in listed if listed.count(name) > 1})
        errors.append(
            classify(
                SEAL_SEGMENT_SET_MISMATCH,
                f"telemetry seal lists {duplicates} more than once",
            )
        )
    present = telemetry_segment_names(policy_dir, live_name=live_name)
    if set(listed) != present:
        unlisted = sorted(present - set(listed))
        absent = sorted(set(listed) - present)
        errors.append(
            classify(
                SEAL_SEGMENT_SET_MISMATCH,
                f"certified segment set does not match the directory: "
                f"present-but-unlisted={unlisted}, listed-but-missing={absent}",
            )
        )

    observed_records = 0
    observed_first: int | None = None
    observed_last: int | None = None
    observed_watermark: int | None = None
    instances: set[str | None] = set()
    identities: set[tuple[str, str]] = set()
    records_by_step: dict[int, dict[str, Any]] = {}

    for entry in manifest["files"]:
        name = entry["name"]
        path = policy_dir / name
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            errors.append(
                classify(
                    SEAL_SEGMENT_CONTENT_MISMATCH,
                    f"certified segment {name} changed after sealing",
                )
            )
            continue
        if len(payload) != entry["size"]:
            errors.append(
                classify(
                    SEAL_SEGMENT_CONTENT_MISMATCH,
                    f"certified segment {name} has an unexpected size",
                )
            )
        for line_number, line in enumerate(payload.decode().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = parse_telemetry_record_v2(json.loads(line))
            except (json.JSONDecodeError, TelemetrySealError) as error:
                errors.append(
                    classify(
                        TELEMETRY_SCHEMA_INVALID, f"{name} line {line_number}: {error}"
                    )
                )
                continue
            step = record["step"]
            if observed_last is not None and step <= observed_last:
                errors.append(
                    classify(
                        SEAL_BOUNDS_MISMATCH,
                        f"{name} line {line_number}: step {step} does not increase",
                    )
                )
            observed_first = step if observed_first is None else observed_first
            observed_last = step
            observed_records += 1
            records_by_step[step] = record
            instances.add(record["server_instance_id"])
            identities.add((record["engine_id"], record["model_id"]))
            candidate = record["generation_high_watermark"]
            if candidate is not None:
                observed_watermark = (
                    candidate
                    if observed_watermark is None
                    else max(observed_watermark, candidate)
                )

    # A null instance id is *not* filtered out here: an unattested record
    # cannot certify anything, and discarding it (the reviewed behaviour)
    # made an entirely unattested stream derive a null instance and pass.
    if None in instances:
        errors.append(
            classify(
                SEAL_INSTANCE_MISMATCH,
                "certified telemetry contains unattested records "
                "(server_instance_id is null); a formal seal requires every "
                "record to name its instance",
            )
        )
    attested = {value for value in instances if value is not None}
    if len(attested) > 1:
        errors.append(
            classify(
                SEAL_INSTANCE_MISMATCH,
                "certified telemetry mixes multiple server instances",
            )
        )
    elif attested and manifest["server_instance_id"] not in attested:
        errors.append(
            classify(
                SEAL_INSTANCE_MISMATCH,
                "telemetry seal names a different server instance",
            )
        )
    if len(identities) > 1:
        errors.append(
            classify(
                SEAL_INSTANCE_MISMATCH,
                "certified telemetry mixes multiple engine/model identities",
            )
        )
    elif identities and next(iter(identities)) != (
        manifest["engine_id"],
        manifest["model_id"],
    ):
        errors.append(
            classify(
                SEAL_INSTANCE_MISMATCH,
                "telemetry seal names a different engine/model identity",
            )
        )

    if manifest["record_count"] != observed_records:
        errors.append(
            classify(
                SEAL_BOUNDS_MISMATCH,
                f"telemetry seal claims {manifest['record_count']} records, "
                f"found {observed_records}",
            )
        )
    for field, observed in (
        ("first_scheduler_step", observed_first),
        ("final_scheduler_step", observed_last),
        ("generation_high_watermark", observed_watermark),
    ):
        if manifest[field] != observed:
            errors.append(
                classify(
                    SEAL_BOUNDS_MISMATCH,
                    f"telemetry seal {field} is {manifest[field]}, found {observed}",
                )
            )

    errors.extend(
        _verify_sealed_siblings(
            manifest,
            policy_dir,
            expectation=expectation,
            records_by_step=records_by_step,
            observed_watermark=observed_watermark,
        )
    )
    return errors


def _verify_sealed_siblings(
    manifest: dict[str, Any],
    policy_dir: Path,
    *,
    expectation: Any,
    records_by_step: dict[int, dict[str, Any]],
    observed_watermark: int | None,
) -> list[str]:
    """Check both siblings against the expectation and the sealed stream.

    The digest comparison here is transport integrity only. Everything that
    decides whether the evidence is *true* is re-derived: the receipt against
    the immutable deployment manifest, and the acknowledgement against the
    telemetry record at the very step it claims.
    """
    from gladius_vllm.application import parse_policy_application_v2
    from gladius_vllm.receipt import (
        parse_server_start_receipt,
        verify_receipt_against_deployment,
    )

    errors: list[str] = []

    receipt_path = policy_dir / SERVER_START_RECEIPT_FILENAME
    if not receipt_path.is_file():
        errors.append(
            classify(
                SEAL_RECEIPT_EXPECTATION_MISMATCH,
                f"{SERVER_START_RECEIPT_FILENAME} is missing; a formal seal "
                "requires it",
            )
        )
    else:
        payload = receipt_path.read_bytes()
        expected = manifest["attestation_receipt_sha256"]
        if hashlib.sha256(payload).hexdigest() != expected:
            errors.append(
                classify(
                    SEAL_SEGMENT_CONTENT_MISMATCH,
                    f"{SERVER_START_RECEIPT_FILENAME} changed after sealing",
                )
            )
        try:
            receipt = parse_server_start_receipt(json.loads(payload))
        except (json.JSONDecodeError, ValueError) as error:
            errors.append(
                classify(
                    SEAL_RECEIPT_EXPECTATION_MISMATCH,
                    f"{SERVER_START_RECEIPT_FILENAME} does not parse strictly: {error}",
                )
            )
        else:
            if receipt.server_instance_id != manifest["server_instance_id"]:
                errors.append(
                    classify(
                        SEAL_INSTANCE_MISMATCH,
                        f"{SERVER_START_RECEIPT_FILENAME} names a different "
                        "server instance",
                    )
                )
            # The check a rehash cannot survive: the receipt's own content
            # must still describe the deployment the campaign froze.
            # Archival: a seal is verified after the server has exited, so
            # process liveness and socket ownership are not properties of a
            # valid seal. Everything about the receipt's *content* is still
            # compared against the expectation.
            for problem in verify_receipt_against_deployment(
                receipt,
                expectation,
                recheck_socket_owner=False,
                recheck_live_processes=False,
            ):
                errors.append(classify(SEAL_RECEIPT_EXPECTATION_MISMATCH, problem))

    application_path = policy_dir / POLICY_APPLICATION_FILENAME
    if not application_path.is_file():
        errors.append(
            classify(
                SEAL_APPLICATION_SEMANTIC_MISMATCH,
                f"{POLICY_APPLICATION_FILENAME} is missing; a formal seal requires it",
            )
        )
        return errors

    payload = application_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["policy_application_sha256"]:
        errors.append(
            classify(
                SEAL_SEGMENT_CONTENT_MISMATCH,
                f"{POLICY_APPLICATION_FILENAME} changed after sealing",
            )
        )
    try:
        application = parse_policy_application_v2(json.loads(payload))
    except (json.JSONDecodeError, ValueError) as error:
        errors.append(
            classify(
                SEAL_APPLICATION_SEMANTIC_MISMATCH,
                f"{POLICY_APPLICATION_FILENAME} does not parse strictly: {error}",
            )
        )
        return errors

    if application.server_instance_id != manifest["server_instance_id"]:
        errors.append(
            classify(
                SEAL_INSTANCE_MISMATCH,
                f"{POLICY_APPLICATION_FILENAME} names a different server instance",
            )
        )
    for label, actual, expected in (
        ("engine_id", application.engine_id, manifest["engine_id"]),
        ("model_id", application.model_id, manifest["model_id"]),
    ):
        if actual != expected:
            errors.append(
                classify(
                    SEAL_APPLICATION_SEMANTIC_MISMATCH,
                    f"acknowledged {label} {actual!r} is not the sealed {expected!r}",
                )
            )
    if application.generation_high_watermark != observed_watermark:
        errors.append(
            classify(
                SEAL_APPLICATION_SEMANTIC_MISMATCH,
                "acknowledged generation_high_watermark "
                f"{application.generation_high_watermark} is not the "
                f"{observed_watermark} the sealed telemetry proves",
            )
        )

    record = records_by_step.get(application.scheduler_step)
    if record is None:
        errors.append(
            classify(
                SEAL_APPLICATION_STEP_UNSEALED,
                f"acknowledgement names scheduler step "
                f"{application.scheduler_step}, which the sealed telemetry "
                "does not contain",
            )
        )
        return errors

    # The record at that exact step has to agree about what was applied.
    # This is what a coherently rehashed action rewrite cannot satisfy.
    application_payload = json.loads(payload)
    for field in ("requested_admission", "effective_admission", "clamped"):
        if application_payload[field] != record[field]:
            errors.append(
                classify(
                    SEAL_APPLICATION_SEMANTIC_MISMATCH,
                    f"acknowledged {field} {application_payload[field]!r} "
                    f"disagrees with the telemetry record at step "
                    f"{application.scheduler_step}: {record[field]!r}",
                )
            )
    for label, actual, expected in (
        ("generation", application.generation, record["generation"]),
        ("policy_id", application.policy_id, record["policy_id"]),
        ("decision_id", application.decision_id, record["decision_id"]),
    ):
        if actual != expected:
            errors.append(
                classify(
                    SEAL_APPLICATION_SEMANTIC_MISMATCH,
                    f"acknowledged {label} {actual!r} disagrees with the "
                    f"telemetry record at step {application.scheduler_step}: "
                    f"{expected!r}",
                )
            )
    return errors


class TelemetryWriter:
    """Appends one JSON line per scheduling step to telemetry.jsonl."""

    def __init__(
        self,
        path: Path | None,
        engine_id: str,
        model_id: str,
        sample_every_n_steps: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self._path = path
        self._engine_id = engine_id
        self._model_id = model_id
        self._sample_every_n_steps = _resolve_sample_every_n_steps(sample_every_n_steps)
        self._max_bytes = _resolve_max_bytes(max_bytes)
        self._step = 0
        self._rotation_count = 0
        self._last_write_ns = 0
        self._sealed = False
        self._half_native_seen = False
        self._file = None
        if self._path is not None and _is_retired(self._path.parent):
            # A directory that already holds a published seal or retirement
            # marker describes a *completed* formal attempt. Appending to it
            # would destroy certified bytes before any verifier could notice,
            # so this writer stays closed and serving continues without
            # telemetry. A restarted server uses a new directory and nonce.
            logger.warning(
                "GLADIUS telemetry disabled: %s is a retired policy directory; "
                "a new server instance must use a new directory",
                self._path.parent,
            )
            self._path = None
        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # Deliberately not a `with` block: this handle is kept open
                # across many record() calls over the writer's lifetime, not
                # scoped to one operation.
                self._file = open(self._path, "a")  # noqa: SIM115
            except OSError:
                logger.warning(
                    "GLADIUS telemetry disabled: could not open %s for writing",
                    self._path,
                    exc_info=True,
                )
                self._file = None

    def _rotate_if_needed(self) -> None:
        """Best-effort size-based rotation: never splits a JSONL record --
        only checked between records, before the next write -- and any
        failure here just skips rotation for this cycle (keeps appending to
        the existing file, growing past `max_bytes` this once) rather than
        disabling telemetry entirely. A failed rotation is a lesser problem
        than a failed write, so it gets its own, narrower fail-open handling
        instead of falling through to record()'s disable-on-failure path.
        """
        if self._file is None or self._path is None:
            return
        try:
            if self._path.stat().st_size < self._max_bytes:
                return
            self._file.close()
            self._rotation_count += 1
            # Counter guarantees uniqueness even for back-to-back rotations
            # within the same millisecond (small max_bytes, high QPS); the
            # timestamp prefix keeps rotated files human-orderable.
            rotated_path = self._path.with_name(
                f"{self._path.name}.{int(time.time() * 1000)}-{self._rotation_count}"
            )
            os.replace(self._path, rotated_path)
            self._file = open(self._path, "a")  # noqa: SIM115 (persistent handle)
        except OSError:
            logger.warning(
                "GLADIUS telemetry rotation failed for %s; continuing without rotating",
                self._path,
                exc_info=True,
            )
            if self._file is None or self._file.closed:
                try:
                    self._file = open(self._path, "a")  # noqa: SIM115
                except OSError:
                    self._file = None

    def record(
        self,
        scheduler: Any,
        output: SchedulerOutput,
        decision: PolicyDecision,
        *,
        policy_poll_ns: int = 0,
        policy_apply_ns: int = 0,
        server_instance_id: str | None = None,
        generation_high_watermark: int | None = None,
    ) -> None:
        self._step += 1
        if self._sealed:
            return
        if self._file is None:
            return
        if self._step % self._sample_every_n_steps != 0:
            return

        # Everything from here down -- stats construction, prefill/decode
        # derivation, dict build, JSON serialization, and the write/flush
        # itself -- is one fail-open region: telemetry must never be able to
        # take down schedule(). A single failure permanently disables this
        # writer (closes the handle) rather than retrying every subsequent
        # step, which both bounds it to exactly one warning (no separate
        # rate-limiting needed) and avoids repeated failing syscalls against
        # a persistently broken destination.
        try:
            self._rotate_if_needed()
            if self._file is None:
                return
            num_prefill, num_decode = _count_prefill_decode(scheduler, output)
            stats = scheduler.make_stats()

            requested_admission = {
                "max_num_seqs": decision.max_num_seqs,
                "max_num_batched_tokens": decision.max_num_batched_tokens,
            }
            effective_admission = {
                "max_num_seqs": scheduler.max_num_running_reqs,
                "max_num_batched_tokens": scheduler.max_num_scheduled_tokens,
            }
            clamped = {
                "max_num_seqs": (
                    effective_admission["max_num_seqs"]
                    != requested_admission["max_num_seqs"]
                ),
                "max_num_batched_tokens": (
                    effective_admission["max_num_batched_tokens"]
                    != requested_admission["max_num_batched_tokens"]
                ),
            }

            # A half-native decision -- a generation with no policy id, or
            # the reverse -- is an impossible scheduler state. Rewriting it
            # into an all-null record (the previous behaviour) would publish
            # it as valid native evidence and hide the defect. Instead the
            # writer refuses to certify anything further for this instance,
            # while serving continues untouched.
            generation = decision.generation
            policy_id = decision.policy_id
            if (generation is None) != (policy_id is None):
                self._half_native_seen = True
                logger.warning(
                    "GLADIUS telemetry disabled at step %d: the scheduler "
                    "reported a half-native decision (generation=%r, "
                    "policy_id=%r), which cannot be certified",
                    self._step,
                    generation,
                    policy_id,
                )
                with contextlib.suppress(OSError):
                    self._file.close()
                self._file = None
                return

            record = {
                "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
                "server_instance_id": server_instance_id,
                "generation_high_watermark": generation_high_watermark,
                "generation": generation,
                "policy_id": policy_id,
                "decision_id": policy_id,
                "window_id": None,  # vLLM has no notion of a control-plane window
                "model_id": self._model_id,
                "engine_id": self._engine_id,
                "created_at": format_iso8601(),
                "expires_at": None,
                "step": self._step,
                "num_running_reqs": stats.num_running_reqs
                if stats
                else len(scheduler.running),
                "num_waiting_reqs": stats.num_waiting_reqs
                if stats
                else len(scheduler.waiting),
                "num_skipped_waiting_reqs": (
                    stats.num_skipped_waiting_reqs
                    if stats
                    else len(scheduler.skipped_waiting)
                ),
                "num_scheduled_reqs": len(output.num_scheduled_tokens),
                "num_scheduled_tokens": output.total_num_scheduled_tokens,
                "num_prefill_reqs": num_prefill,
                "num_decode_reqs": num_decode,
                "kv_cache_usage": stats.kv_cache_usage if stats else None,
                "policy_status": decision.status,
                "policy_source": decision.source,
                "requested_admission": requested_admission,
                "effective_admission": effective_admission,
                "clamped": clamped,
                "policy_poll_ns": max(0, int(policy_poll_ns)),
                "policy_apply_ns": max(0, int(policy_apply_ns)),
                # Writing this record has not completed yet. Report the
                # previous sampled emission's measured serialization/write/
                # flush duration so the stream remains append-only.
                "telemetry_write_ns": self._last_write_ns,
            }
            write_started = time.perf_counter_ns()
            # One append path, so the retirement recheck cannot be bypassed
            # by the scheduler's hot loop.
            self.append_record(record)
            self._last_write_ns = time.perf_counter_ns() - write_started
        except Exception:
            logger.warning(
                "GLADIUS telemetry disabled at step %d: record/write failed",
                self._step,
                exc_info=True,
            )
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

    def append_record(self, record: dict[str, Any]) -> None:
        """Append one already-built record, refusing a retired directory.

        Public and fail-*closed*, unlike `record()`. The retirement check
        happens while holding the directory's shared evidence lock and is
        repeated on every append, so a writer opened before a seal cannot
        extend certified evidence afterwards -- the defect the third review
        found in the constructor-only check.
        """
        if self._path is None or self._file is None:
            raise TelemetrySealError(
                classify(
                    TELEMETRY_DIRECTORY_RETIRED,
                    "this telemetry writer holds no open stream",
                )
            )
        parse_telemetry_record_v2(record)
        with evidence_lock(self._path.parent, exclusive=False):
            if _is_retired(self._path.parent):
                with contextlib.suppress(OSError):
                    self._file.close()
                self._file = None
                self._sealed = True
                raise TelemetrySealError(
                    classify(
                        TELEMETRY_DIRECTORY_RETIRED,
                        f"{self._path.parent} was retired; its certified "
                        "evidence may not be extended",
                    )
                )
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

    def _ordered_segments(self) -> list[Path]:
        """Rotated segments in rotation order, then the live file.

        Sorted by the numeric `(millisecond, rotation_count)` suffix rather
        than lexicographically: `...-10` sorts before `...-2` as text, which
        would certify the stream in the wrong logical order.
        """
        assert self._path is not None
        segments: list[tuple[tuple[int, int], Path]] = []
        for candidate in self._path.parent.glob(f"{self._path.name}.*-*"):
            suffix = candidate.name[len(self._path.name) + 1 :]
            milliseconds, _, count = suffix.partition("-")
            if not milliseconds.isdigit() or not count.isdigit():
                raise TelemetrySealError(
                    f"unrecognised rotated telemetry segment {candidate.name}"
                )
            segments.append(((int(milliseconds), int(count)), candidate))
        ordered = [path for _, path in sorted(segments, key=lambda item: item[0])]
        if self._path.is_file():
            ordered.append(self._path)
        return ordered

    def _verify_segments(self, segments: list[Path]) -> dict[str, object]:
        """Parse every certified record and derive the manifest's bounds.

        Sealing an unverified byte range would certify only that files were
        copied intact. This walks the records so the seal also attests that
        they are parseable, come from exactly one server instance, and form
        one strictly increasing step sequence.
        """
        names = [path.name for path in segments]
        if len(set(names)) != len(names):
            raise TelemetrySealError("duplicate telemetry segment filenames")

        instances: set[str | None] = set()
        watermark: int | None = None
        first_step: int | None = None
        last_step: int | None = None
        record_count = 0
        for path in segments:
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    record = parse_telemetry_record_v2(json.loads(line))
                except (json.JSONDecodeError, TelemetrySealError) as error:
                    raise TelemetrySealError(
                        f"{path.name} line {line_number}: {error}"
                    ) from error
                step = record["step"]
                if last_step is not None and step <= last_step:
                    raise TelemetrySealError(
                        f"{path.name} line {line_number}: step {step} does not "
                        f"increase past {last_step}"
                    )
                first_step = step if first_step is None else first_step
                last_step = step
                record_count += 1
                instances.add(record["server_instance_id"])
                candidate = record.get("generation_high_watermark")
                if candidate is not None:
                    watermark = (
                        candidate if watermark is None else max(watermark, candidate)
                    )
        if record_count == 0:
            raise TelemetrySealError(
                "refusing to seal an empty telemetry stream: a formal attempt "
                "needs at least one certified scheduler record"
            )
        # An unattested record certifies nothing. Filtering nulls out before
        # counting (the reviewed behaviour) let a stream in which *every*
        # record was unattested derive a null instance and seal successfully.
        if None in instances:
            raise TelemetrySealError(
                "refusing to seal unattested telemetry: every record must name "
                "its server instance, which means the server-start receipt must "
                "have been published before the certified window began"
            )
        if len(instances) > 1:
            raise TelemetrySealError(
                f"telemetry mixes {len(instances)} server instances; a seal "
                "certifies exactly one serving process"
            )
        return {
            "server_instance_id": next(iter(instances)),
            "first_scheduler_step": first_step,
            "final_scheduler_step": last_step,
            "record_count": record_count,
            "generation_high_watermark": watermark,
        }

    def _sealed_sibling_digest(self, filename: str, *, instance_id: str) -> str:
        """Hash a sibling only after it parses and names the sealed instance.

        A digest taken before those checks certifies whatever bytes happened
        to be there -- including a missing file, which the reviewed revision
        recorded as a null digest and still called a successful seal.
        """
        from gladius_vllm.application import parse_policy_application_v2
        from gladius_vllm.receipt import parse_server_start_receipt

        assert self._path is not None
        path = self._path.parent / filename
        if not path.is_file():
            raise TelemetrySealError(
                f"refusing to seal without {filename}: a formal seal binds both "
                "the server-start receipt and the final acknowledgement"
            )
        payload = path.read_bytes()
        parser = (
            parse_server_start_receipt
            if filename == SERVER_START_RECEIPT_FILENAME
            else parse_policy_application_v2
        )
        try:
            parsed = parser(json.loads(payload))
        except (json.JSONDecodeError, ValueError) as error:
            raise TelemetrySealError(
                f"{filename} does not parse strictly: {error}"
            ) from error
        if parsed.server_instance_id != instance_id:
            raise TelemetrySealError(
                f"{filename} names server instance "
                f"{parsed.server_instance_id!r}, not the sealed {instance_id!r}"
            )
        return hashlib.sha256(payload).hexdigest()

    def seal(self, manifest_path: Path) -> bool:
        """Close and atomically certify this writer's telemetry segments.

        Taken under the directory's *exclusive* evidence lock, so no other
        writer -- in this process or another -- can be mid-append while the
        segments are hashed. The writer is closed before anything is hashed,
        and retirement is written before the lock is released, so the whole
        transition is atomic with respect to every appender.
        """
        self._sealed = True
        self.close()
        if self._path is None:
            return False
        with evidence_lock(self._path.parent, exclusive=True):
            return self._seal_locked(manifest_path)

    def _seal_locked(self, manifest_path: Path) -> bool:
        if self._half_native_seen:
            logger.warning(
                "GLADIUS refusing to seal %s: this instance emitted a "
                "half-native decision and can no longer certify evidence",
                self._path,
            )
            return False
        try:
            segments = self._ordered_segments()
            derived = self._verify_segments(segments)
            instance_id = str(derived["server_instance_id"])
            manifest = {
                "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
                "engine_id": self._engine_id,
                "model_id": self._model_id,
                "sealed_at": format_iso8601(),
                "attestation_receipt_sha256": self._sealed_sibling_digest(
                    SERVER_START_RECEIPT_FILENAME, instance_id=instance_id
                ),
                # Captured after the last scheduler step, so it binds the
                # final observed application state rather than whichever one
                # happened to be current when sealing began.
                "policy_application_sha256": self._sealed_sibling_digest(
                    POLICY_APPLICATION_FILENAME, instance_id=instance_id
                ),
                "files": [
                    {
                        "name": path.name,
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in segments
                ],
                **derived,
            }
            parse_telemetry_seal(manifest)
            atomic_write_json(manifest_path, manifest)
            # The directory now holds certified evidence: no later writer may
            # reopen it, even one in a freshly started server process. Written
            # before the exclusive lock is released, so no appender can
            # observe the seal without also observing the retirement.
            _write_retirement_marker(self._path.parent)
            return True
        except Exception:
            logger.warning(
                "GLADIUS telemetry seal failed for %s",
                self._path,
                exc_info=True,
            )
            return False

    @property
    def step(self) -> int:
        """Latest scheduler-step identity, including unsampled steps."""
        return self._step

    @property
    def sealed(self) -> bool:
        return self._sealed
