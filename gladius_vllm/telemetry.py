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
from gladius_vllm.policy import PolicyDecision
from gladius_vllm.schema import (
    DEFAULT_TELEMETRY_MAX_BYTES,
    DEFAULT_TELEMETRY_SAMPLE_N,
    EXECUTION_EVIDENCE_SCHEMA_VERSION,
    format_iso8601,
    parse_int_env,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)

POLICY_APPLICATION_FILENAME = "policy_application.json"
SERVER_START_RECEIPT_FILENAME = "server_start_receipt.json"


class TelemetrySealError(ValueError):
    """A telemetry stream could not be certified as immutable evidence."""


def parse_telemetry_record_v2(payload: object) -> dict[str, Any]:
    """Strictly validate one execution-evidence 2.0.0 telemetry record.

    The native-state invariant is enforced here rather than repaired: a
    record claiming a generation but no policy id (or vice versa) describes
    a scheduler state that cannot exist, so accepting it would mean
    inventing evidence.
    """
    if not isinstance(payload, dict):
        raise TelemetrySealError("telemetry record must be a JSON object")
    if payload.get("schema_version") != EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise TelemetrySealError(
            "telemetry requires execution-evidence schema "
            f"{EXECUTION_EVIDENCE_SCHEMA_VERSION}"
        )
    for field in ("engine_id", "model_id"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise TelemetrySealError(f"{field} must be a non-empty string")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise TelemetrySealError("step must be a positive integer")

    identity = (payload.get("generation"), payload.get("policy_id"))
    identity += (payload.get("decision_id"),)
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

    watermark = payload.get("generation_high_watermark")
    if watermark is not None and (
        isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0
    ):
        raise TelemetrySealError(
            "generation_high_watermark must be a non-negative integer or null"
        )
    if "server_instance_id" not in payload:
        raise TelemetrySealError("telemetry record must carry server_instance_id")
    instance = payload["server_instance_id"]
    if instance is not None and (not isinstance(instance, str) or not instance):
        raise TelemetrySealError(
            "server_instance_id must be a non-empty string or null"
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


def verify_telemetry_seal(manifest_path: Path, policy_dir: Path) -> list[str]:
    """Re-derive everything a seal claims, from the files still on disk.

    A seal that is only ever written is not evidence -- something has to be
    able to detect that a certified segment, the acknowledgement, or the
    receipt changed afterwards. Returns every discrepancy rather than
    raising on the first, so one pass gives the complete diagnosis.
    """
    manifest_path = Path(manifest_path)
    policy_dir = Path(policy_dir)
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f"telemetry seal unreadable: {error}"]

    if manifest.get("schema_version") != EXECUTION_EVIDENCE_SCHEMA_VERSION:
        errors.append(
            "telemetry seal requires execution-evidence schema "
            f"{EXECUTION_EVIDENCE_SCHEMA_VERSION}"
        )
    if not manifest.get("files"):
        errors.append("telemetry seal certifies no segments")
    if not manifest.get("record_count"):
        errors.append("telemetry seal certifies no records")

    seen_names: set[str] = set()
    observed_records = 0
    observed_first: int | None = None
    observed_last: int | None = None
    instances: set[str] = set()
    for entry in manifest.get("files", []):
        name = entry.get("name")
        if name in seen_names:
            errors.append(f"telemetry seal lists {name} twice")
            continue
        seen_names.add(name)
        path = policy_dir / name
        if not path.is_file():
            errors.append(f"certified segment {name} is missing")
            continue
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
            errors.append(f"certified segment {name} changed after sealing")
            continue
        if len(payload) != entry.get("size"):
            errors.append(f"certified segment {name} has an unexpected size")
        for line_number, line in enumerate(payload.decode().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = parse_telemetry_record_v2(json.loads(line))
            except (json.JSONDecodeError, TelemetrySealError) as error:
                errors.append(f"{name} line {line_number}: {error}")
                continue
            step = record["step"]
            if observed_last is not None and step <= observed_last:
                errors.append(
                    f"{name} line {line_number}: step {step} does not increase"
                )
            observed_first = step if observed_first is None else observed_first
            observed_last = step
            observed_records += 1
            if record["server_instance_id"] is not None:
                instances.add(record["server_instance_id"])

    if len(instances) > 1:
        errors.append("certified telemetry mixes multiple server instances")
    elif instances and manifest.get("server_instance_id") not in instances:
        errors.append("telemetry seal names a different server instance")
    if manifest.get("record_count") != observed_records:
        errors.append(
            f"telemetry seal claims {manifest.get('record_count')} records, "
            f"found {observed_records}"
        )
    for field, observed in (
        ("first_scheduler_step", observed_first),
        ("final_scheduler_step", observed_last),
    ):
        if manifest.get(field) != observed:
            errors.append(
                f"telemetry seal {field} is {manifest.get(field)}, found {observed}"
            )

    for field, filename in (
        ("attestation_receipt_sha256", SERVER_START_RECEIPT_FILENAME),
        ("policy_application_sha256", POLICY_APPLICATION_FILENAME),
    ):
        expected = manifest.get(field)
        path = policy_dir / filename
        actual = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if expected != actual:
            errors.append(f"{filename} changed after sealing")
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
        self._file = None
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

            # One source for all three identity fields, so the native
            # invariant `(generation is None) == (policy_id is None) ==
            # (decision_id is None)` cannot drift apart field by field.
            native = decision.generation is None or decision.policy_id is None
            generation = None if native else decision.generation
            policy_id = None if native else decision.policy_id

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
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()
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
        attested = {value for value in instances if value is not None}
        if len(attested) > 1:
            raise TelemetrySealError(
                f"telemetry mixes {len(attested)} server instances; a seal "
                "certifies exactly one serving process"
            )
        return {
            "server_instance_id": next(iter(attested), None),
            "first_scheduler_step": first_step,
            "final_scheduler_step": last_step,
            "record_count": record_count,
            "generation_high_watermark": watermark,
        }

    def _sibling_digest(self, filename: str) -> str | None:
        assert self._path is not None
        path = self._path.parent / filename
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def seal(self, manifest_path: Path) -> bool:
        """Close and atomically certify this writer's telemetry segments.

        The writer is closed *before* anything is hashed, so no record can be
        appended between the digest and the manifest. After this returns the
        stream is permanently frozen: `record()` keeps counting steps (so the
        scheduler's step identity stays continuous) but never writes again,
        and scheduling itself is unaffected.
        """
        self._sealed = True
        self.close()
        if self._path is None:
            return False
        try:
            segments = self._ordered_segments()
            derived = self._verify_segments(segments)
            manifest = {
                "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
                "engine_id": self._engine_id,
                "model_id": self._model_id,
                "sealed_at": format_iso8601(),
                "attestation_receipt_sha256": self._sibling_digest(
                    SERVER_START_RECEIPT_FILENAME
                ),
                # Captured after the last scheduler step, so it binds the
                # final observed application state rather than whichever one
                # happened to be current when sealing began.
                "policy_application_sha256": self._sibling_digest(
                    POLICY_APPLICATION_FILENAME
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
            atomic_write_json(manifest_path, manifest)
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
