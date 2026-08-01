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

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from gladius_vllm.application import PolicyApplicationWriter
from gladius_vllm.policy import PolicyLoader
from gladius_vllm.registry import register_scheduler
from gladius_vllm.schema import (
    DEFAULT_POLICY_POLL_INTERVAL_MS,
    parse_int_env,
    resolve_engine_id,
    resolve_model_id,
)
from gladius_vllm.telemetry import TelemetryWriter
from vllm.v1.core.sched.scheduler import Scheduler

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
            path=(
                policy_dir / POLICY_APPLICATION_FILENAME
                if policy_dir
                else None
            ),
            engine_id=self.engine_id,
            model_id=self.model_id,
        )

        register_scheduler(self)

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

        self._telemetry_writer.record(
            self,
            output,
            decision,
            policy_poll_ns=policy_poll_ns,
            policy_apply_ns=policy_apply_ns,
        )
        self._application_writer.record(
            decision,
            scheduler_step=self._telemetry_writer.step,
            effective_max_num_seqs=self.max_num_running_reqs,
            effective_max_num_batched_tokens=self.max_num_scheduled_tokens,
        )
        return output

    def seal_telemetry(self, manifest_path: Path) -> bool:
        """Certify the current experiment stream and reject later writes."""
        return self._telemetry_writer.seal(manifest_path)

    def shutdown(self) -> None:
        self._telemetry_writer.close()
        super().shutdown()
