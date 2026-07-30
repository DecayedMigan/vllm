"""vLLM V1 scheduler controlled by durable GLADIUS serving experience."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.v1.core.sched.gladius_protocol import (
    PolicyController,
    append_telemetry,
    utc_now,
)
from vllm.v1.core.sched.scheduler import Scheduler

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)


class GladiusScheduler(Scheduler):
    """Hot-reload GLADIUS admission policies around the V1 scheduler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        vllm_config = args[0] if args else kwargs["vllm_config"]
        self._gladius_startup_limit = self.scheduler_config.max_num_seqs
        self._gladius_model_id = vllm_config.model_config.model
        policy_value = os.environ.get("GLADIUS_POLICY_PATH")
        self._gladius_policy = PolicyController(
            policy_path=Path(policy_value) if policy_value else None,
            model_id=self._gladius_model_id,
            startup_admission_limit=self._gladius_startup_limit,
        )
        telemetry_value = os.environ.get("GLADIUS_TELEMETRY_PATH")
        self._gladius_telemetry_path = (
            Path(telemetry_value) if telemetry_value else None
        )
        self._gladius_engine_id = (
            os.environ.get("GLADIUS_ENGINE_ID") or f"pid-{os.getpid()}"
        )
        self._gladius_step = 0

    def schedule(self) -> SchedulerOutput:
        """Apply a live policy, schedule work, then emit best-effort telemetry."""

        self._gladius_step += 1
        state = self._gladius_policy.refresh(utc_now())
        # Do not evict requests already running when a lower admission policy
        # arrives. Matching their count prevents any new admission, and the
        # effective cap converges as those requests finish.
        self.max_num_running_reqs = max(state.admission_limit, len(self.running))
        scheduler_output = super().schedule()
        if self._gladius_telemetry_path is not None:
            try:
                running, waiting = self.get_request_counts()
                append_telemetry(
                    self._gladius_telemetry_path,
                    engine_id=self._gladius_engine_id,
                    model_id=self._gladius_model_id,
                    step=self._gladius_step,
                    state=state,
                    running=running,
                    waiting=waiting,
                    scheduled_requests=len(scheduler_output.num_scheduled_tokens),
                    scheduled_tokens=(scheduler_output.total_num_scheduled_tokens),
                    timestamp=utc_now(),
                )
            except Exception:
                logger.warning(
                    "Failed to append GLADIUS scheduler telemetry",
                    exc_info=True,
                )
        return scheduler_output
