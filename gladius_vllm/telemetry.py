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
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gladius_vllm.policy import PolicyDecision
from gladius_vllm.schema import (
    DEFAULT_TELEMETRY_MAX_BYTES,
    DEFAULT_TELEMETRY_SAMPLE_N,
    SCHEMA_VERSION,
    format_iso8601,
    parse_int_env,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)


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

            record = {
                "schema_version": SCHEMA_VERSION,
                "generation": decision.generation,
                "policy_id": decision.policy_id,
                "decision_id": decision.policy_id,  # == policy_id in schema 1.x
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

    def seal(self, manifest_path: Path) -> bool:
        """Close and atomically certify this writer's telemetry segments."""
        self._sealed = True
        self.close()
        if self._path is None:
            return False
        temporary_path: Path | None = None
        try:
            rotated = sorted(self._path.parent.glob(f"{self._path.name}.*-*"))
            files = [*rotated]
            if self._path.is_file():
                files.append(self._path)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "engine_id": self._engine_id,
                "model_id": self._model_id,
                "sealed_at": format_iso8601(),
                "final_scheduler_step": self._step,
                "files": [
                    {
                        "name": path.name,
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in files
                ],
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=manifest_path.parent,
                prefix=f".{manifest_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, manifest_path)
            temporary_path = None
            return True
        except Exception:
            logger.warning(
                "GLADIUS telemetry seal failed for %s",
                self._path,
                exc_info=True,
            )
            return False
        finally:
            if temporary_path is not None:
                with contextlib.suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    @property
    def step(self) -> int:
        """Latest scheduler-step identity, including unsampled steps."""
        return self._step
