"""GladiusScheduler: a vLLM V1 Scheduler that hot-reloads admission ceilings
from an externally-published policy_snapshot.json.

Wired in via vLLM's existing `--scheduler-cls` plugin mechanism (no upstream
vllm/ changes needed):

    vllm serve ... --scheduler-cls gladius_vllm.scheduler.GladiusScheduler

Configuration is via environment variables (not VllmConfig/EngineArgs
plumbing, to keep this a pure additive plugin):

    GLADIUS_POLICY_DIR                directory holding policy_snapshot.json
                                       and telemetry.jsonl. If unset, this
                                       scheduler behaves exactly like a
                                       vanilla Scheduler.
    GLADIUS_ENGINE_ID                 stable engine id across restarts.
    GLADIUS_POLICY_POLL_INTERVAL_MS   rate limit for policy file re-stat.
    GLADIUS_TELEMETRY_SAMPLE_N        write 1-in-N telemetry lines.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from gladius_vllm.application import PolicyApplicationWriter
from gladius_vllm.digest import sha256_file
from gladius_vllm.evidence_codes import SEAL_SCHEMA_INVALID
from gladius_vllm.policy import PolicyLoader
from gladius_vllm.receipt import publish_startup_attestation, resolve_attestation_nonce
from gladius_vllm.registry import register_scheduler
from gladius_vllm.schema import (
    DEFAULT_POLICY_POLL_INTERVAL_MS,
    parse_int_env,
    resolve_engine_id,
    resolve_model_id,
)
from gladius_vllm.seal_lifecycle import (
    ACK_REFUSED,
    ACK_SEALED,
    SEAL_REQUEST_FILENAME,
    classify_refusal,
    parse_seal_request,
    read_seal_ack,
    write_seal_ack,
)
from gladius_vllm.telemetry import (
    RETIRED_MARKER_FILENAME,
    TELEMETRY_SEAL_FILENAME,
    TelemetryWriter,
)
from vllm.v1.core.sched.scheduler import Scheduler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

POLICY_SNAPSHOT_FILENAME = "policy_snapshot.json"
TELEMETRY_FILENAME = "telemetry.jsonl"
POLICY_APPLICATION_FILENAME = "policy_application.json"


def _resolve_poll_interval_ms() -> int:
    """Production polling interval forbids 0: it would mean an unconditional
    `Path.stat()` on every single scheduling step regardless of QPS. The env
    var therefore enforces a minimum of 1.

    Tests that need deterministic, immediate re-polling (no rate-limit
    window between writing a new snapshot and observing it) should
    monkeypatch this function directly instead, e.g.:
    `monkeypatch.setattr(gladius_vllm.scheduler,
    "_resolve_poll_interval_ms", lambda: 0)`.
    """
    return parse_int_env(
        "GLADIUS_POLICY_POLL_INTERVAL_MS", DEFAULT_POLICY_POLL_INTERVAL_MS, minimum=1
    )


class GladiusScheduler(Scheduler):
    """Thin subclass: only __init__ and schedule() are overridden.

    The safe-clamp invariant `effective = min(policy_requested, startup)` is
    enforced fresh every step, so this scheduler degrades to byte-for-byte
    vanilla `Scheduler` behavior whenever no policy is configured/active
    (min(x, x) == x is a no-op), and can never admit more than the engine was
    started with -- max_num_seqs/max_num_batched_tokens are baked into CUDA
    graph capture sizes and torch.compile shapes at startup, so raising them
    post-hoc is unsafe; this scheduler only ever clamps down.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.engine_id = resolve_engine_id(self.vllm_config)
        self.model_id = resolve_model_id(self.vllm_config)

        # Captured once, after super().__init__ has set these -- these are
        # the values already baked into CUDA graph capture / compiled kernel
        # shapes, and are never mutated by this class.
        self.startup_max_num_seqs = self.max_num_running_reqs
        self.startup_max_num_batched_tokens = self.max_num_scheduled_tokens

        policy_dir_env = os.environ.get("GLADIUS_POLICY_DIR")
        policy_dir = Path(policy_dir_env) if policy_dir_env else None
        self._policy_dir = policy_dir
        # The deployment manifest this server was launched against, so a seal
        # request written for a *different* deployment is refused rather than
        # silently certifying whatever is running.
        self._deployment_manifest_sha256 = (
            os.environ.get("GLADIUS_DEPLOYMENT_MANIFEST_SHA256") or None
        )
        poll_interval_ms = _resolve_poll_interval_ms()

        self._policy_loader = PolicyLoader(
            snapshot_path=policy_dir / POLICY_SNAPSHOT_FILENAME if policy_dir else None,
            engine_id=self.engine_id,
            model_id=self.model_id,
            startup_max_num_seqs=self.startup_max_num_seqs,
            startup_max_num_batched_tokens=self.startup_max_num_batched_tokens,
            poll_interval_ms=poll_interval_ms,
        )
        self._telemetry_writer = TelemetryWriter(
            path=policy_dir / TELEMETRY_FILENAME if policy_dir else None,
            engine_id=self.engine_id,
            model_id=self.model_id,
        )
        self._application_writer = PolicyApplicationWriter(
            path=(policy_dir / POLICY_APPLICATION_FILENAME if policy_dir else None),
            engine_id=self.engine_id,
            model_id=self.model_id,
        )
        # EngineCore-side half of the server-start receipt: this process is
        # the only one that can prove which physical GPU the model landed on
        # and which binaries it loaded. The API process contributes its own
        # identity separately (`python -m gladius_vllm.attest publish`), and
        # this binding adopts the joined `server_instance_id` once that
        # lands. Fail-open: an unattested server still serves, it just
        # cannot certify a formal discovery cell.
        self._instance_binding = publish_startup_attestation(
            self.vllm_config,
            policy_dir=policy_dir,
            startup_max_num_seqs=self.startup_max_num_seqs,
            startup_max_num_batched_tokens=self.startup_max_num_batched_tokens,
        )

        register_scheduler(self)

    @property
    def server_instance_id(self) -> str | None:
        return self._instance_binding.server_instance_id

    def schedule(self, *args: object, **kwargs: object) -> SchedulerOutput:
        # Forward whatever the active vLLM runtime supplies rather than
        # assuming a fixed base signature: different vLLM versions call
        # Scheduler.schedule() differently (some zero-arg, some with a
        # `throttle_prefills` positional) -- see
        # docs/design/gladius_next_steps_h100.md P0-A. Hard-coding either
        # shape breaks the plugin on the other version.
        poll_started = time.perf_counter_ns()
        decision = self._policy_loader.poll()
        policy_poll_ns = time.perf_counter_ns() - poll_started
        apply_started = time.perf_counter_ns()
        target_max_num_seqs = min(decision.max_num_seqs, self.startup_max_num_seqs)
        # Never shrink below the number of requests already admitted: the
        # base Scheduler enforces `len(self.running) <= max_num_running_reqs`
        # as a forward-looking admission check only -- it has no preemption
        # path to evict already-running requests down to a newly-lowered
        # ceiling, and asserts the invariant unconditionally. A policy asking
        # for fewer than what's already running takes effect gradually, as
        # attrition (requests finishing) brings the running count back down.
        self.max_num_running_reqs = max(target_max_num_seqs, len(self.running))
        self.max_num_scheduled_tokens = min(
            decision.max_num_batched_tokens, self.startup_max_num_batched_tokens
        )
        policy_apply_ns = time.perf_counter_ns() - apply_started

        output = super().schedule(*args, **kwargs)

        # Cheap no-op once the receipt has been adopted; before then it is a
        # rate-unbounded stat() only while the server is still unattested.
        server_instance_id = self._instance_binding.refresh()
        generation_high_watermark = self._policy_loader.generation_high_watermark

        self._telemetry_writer.record(
            self,
            output,
            decision,
            policy_poll_ns=policy_poll_ns,
            policy_apply_ns=policy_apply_ns,
            server_instance_id=server_instance_id,
            generation_high_watermark=generation_high_watermark,
        )
        self._application_writer.record(
            decision,
            scheduler_step=self._telemetry_writer.step,
            effective_max_num_seqs=self.max_num_running_reqs,
            effective_max_num_batched_tokens=self.max_num_scheduled_tokens,
            server_instance_id=server_instance_id,
            generation_high_watermark=generation_high_watermark,
        )
        # The production caller `seal_telemetry()` never had. Placed after
        # both writers so a seal certifies this step too, and at the end of
        # schedule() so it is a safe boundary: no partial step is in flight.
        self._poll_seal_request()
        return output

    def _poll_seal_request(self) -> None:
        """Honour an operator's seal request at a safe scheduling boundary.

        Called once per `schedule()`, after the step's telemetry and
        acknowledgement have been written, so whatever this seals is the
        complete record of every step that has run. A missing request file
        is the overwhelmingly common case and costs one `stat()`.

        Fail-open like every other evidence path: a malformed or foreign
        request is answered with a classified refusal and serving continues.
        """
        if self._policy_dir is None or self._telemetry_writer.sealed:
            return
        request_path = self._policy_dir / SEAL_REQUEST_FILENAME
        try:
            if not request_path.is_file():
                return
            request = parse_seal_request(json.loads(request_path.read_text()))
        except (OSError, ValueError):
            logger.warning(
                "GLADIUS: unusable seal request in %s; ignoring",
                self._policy_dir,
                exc_info=True,
            )
            return

        try:
            existing = read_seal_ack(self._policy_dir)
            if existing is not None and existing["request_id"] == request.request_id:
                # Retrying an identical request is idempotent, so an operator
                # whose wait timed out can safely ask again.
                return

            refusal = classify_refusal(
                request,
                server_instance_id=self.server_instance_id,
                deployment_manifest_sha256=self._deployment_manifest_sha256,
                attestation_nonce=resolve_attestation_nonce(),
                observed_generation=self._policy_loader.generation_high_watermark,
                already_retired=(self._policy_dir / RETIRED_MARKER_FILENAME).exists(),
            )
            if refusal is not None:
                code, _, detail = refusal.partition(": ")
                write_seal_ack(
                    self._policy_dir,
                    request=request,
                    status=ACK_REFUSED,
                    error_code=code,
                    error_detail=detail,
                )
                logger.warning("GLADIUS refusing seal request: %s", refusal)
                return

            manifest_path = self._policy_dir / TELEMETRY_SEAL_FILENAME
            if self._telemetry_writer.seal(manifest_path):
                write_seal_ack(
                    self._policy_dir,
                    request=request,
                    status=ACK_SEALED,
                    telemetry_seal_sha256=sha256_file(manifest_path),
                )
                logger.info("GLADIUS sealed %s", self._policy_dir)
            else:
                write_seal_ack(
                    self._policy_dir,
                    request=request,
                    status=ACK_REFUSED,
                    error_code=SEAL_SCHEMA_INVALID,
                    error_detail="the telemetry stream could not be certified",
                )
        except Exception:  # noqa: BLE001 - sealing must never break serving
            logger.warning(
                "GLADIUS: seal request handling failed for %s",
                self._policy_dir,
                exc_info=True,
            )

    def shutdown(self) -> None:
        self._telemetry_writer.close()
        super().shutdown()
