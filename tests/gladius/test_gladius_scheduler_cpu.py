"""CPU-only tests for GladiusScheduler -- no GPU, no executor, real vLLM
Scheduler/Request objects via tests/v1/core/utils.py's create_scheduler()/
create_requests() helpers (upstream's own vehicle for scheduler-only unit
tests). GladiusScheduler isn't reachable through create_scheduler() directly
(it hardcodes Scheduler/AsyncScheduler), so _build_gladius_scheduler() below
is a small local wrapper that reuses a vanilla-built Scheduler's already-
constructed VllmConfig/KVCacheConfig/StructuredOutputManager to build a
GladiusScheduler with an identical configuration -- it does not modify the
upstream helper file.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from gladius_vllm.scheduler import GladiusScheduler
from tests.gladius._test_model import resolve_test_model
from tests.v1.core.utils import create_requests, create_scheduler

# See tests/gladius/conftest.py for the autouse HF_HUB_OFFLINE/no-proxy
# fixture required before any vllm config object (e.g. create_scheduler())
# is constructed.

MODEL = resolve_test_model()  # override via GLADIUS_TEST_MODEL, e.g. for offline hosts


def _write_snapshot(
    directory,
    *,
    engine_id: str,
    model_id: str,
    generation: int,
    max_num_seqs=None,
    max_num_batched_tokens=None,
    ttl_seconds: float = 30.0,
    created_at=None,
):
    created_at = created_at or datetime.now(timezone.utc)
    expires_at = created_at + timedelta(seconds=ttl_seconds)
    payload = {
        "schema_version": "1.0.0",
        "generation": generation,
        "policy_id": f"policy-{generation}",
        "model_id": model_id,
        "engine_id": engine_id,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        "admission": {
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
        },
    }
    path = directory / "policy_snapshot.json"
    tmp = directory / ".policy_snapshot.tmp"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    return path


def _build_gladius_scheduler(vanilla, block_size=16):
    return GladiusScheduler(
        vllm_config=vanilla.vllm_config,
        kv_cache_config=vanilla.kv_cache_config,
        structured_output_manager=vanilla.structured_output_manager,
        block_size=block_size,
        log_stats=True,
    )


@pytest.fixture(autouse=True)
def _fixed_engine_id(monkeypatch):
    monkeypatch.setenv("GLADIUS_ENGINE_ID", "test-engine")
    # Production forbids a 0ms polling interval via the env var (see
    # gladius_vllm.scheduler._resolve_poll_interval_ms) -- tests that need
    # deterministic immediate re-polling inject 0 directly instead.
    monkeypatch.setattr("gladius_vllm.scheduler._resolve_poll_interval_ms", lambda: 0)


def test_no_policy_file_matches_vanilla_scheduler_decisions(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))  # dir exists, no file in it

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=20, num_tokens=50, max_tokens=8):
        vanilla.add_request(req)
    for req in create_requests(num_requests=20, num_tokens=50, max_tokens=8):
        gladius.add_request(req)

    vanilla_output = vanilla.schedule()
    gladius_output = gladius.schedule()

    assert gladius.max_num_running_reqs == vanilla.max_num_running_reqs
    assert gladius.max_num_scheduled_tokens == vanilla.max_num_scheduled_tokens
    assert sorted(gladius_output.num_scheduled_tokens.items()) == sorted(
        vanilla_output.num_scheduled_tokens.items()
    )
    assert (
        gladius_output.total_num_scheduled_tokens
        == vanilla_output.total_num_scheduled_tokens
    )
    assert {r.req_id for r in gladius_output.scheduled_new_reqs} == {
        r.req_id for r in vanilla_output.scheduled_new_reqs
    }


def test_no_policy_dir_configured_at_all_matches_vanilla(monkeypatch):
    monkeypatch.delenv("GLADIUS_POLICY_DIR", raising=False)

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=5, num_tokens=50, max_tokens=8):
        vanilla.add_request(req)
    for req in create_requests(num_requests=5, num_tokens=50, max_tokens=8):
        gladius.add_request(req)

    vanilla_output = vanilla.schedule()
    gladius_output = gladius.schedule()
    assert (
        gladius_output.total_num_scheduled_tokens
        == vanilla_output.total_num_scheduled_tokens
    )


def test_policy_lowering_max_num_seqs_changes_admission(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=1,
        max_num_seqs=4,
    )

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)

    output = gladius.schedule()
    assert gladius.max_num_running_reqs == 4
    assert len(gladius.running) == 4
    assert len(output.scheduled_new_reqs) == 4


def test_policy_lowering_token_budget_changes_scheduled_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=1,
        max_num_batched_tokens=64,
    )

    for req in create_requests(num_requests=10, num_tokens=50, max_tokens=8):
        gladius.add_request(req)

    output = gladius.schedule()
    assert gladius.max_num_scheduled_tokens == 64
    assert output.total_num_scheduled_tokens <= 64


def test_schedule_publishes_matching_policy_application(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))
    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=7,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
    )
    for req in create_requests(num_requests=8, num_tokens=10, max_tokens=8):
        gladius.add_request(req)

    gladius.schedule()

    application = json.loads(
        (tmp_path / "policy_application.json").read_text(encoding="utf-8")
    )
    assert application["generation"] == 7
    assert application["policy_id"] == "policy-7"
    assert application["state"] == "active"
    assert application["scheduler_step"] == 1
    assert application["requested_admission"] == {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 2048,
    }
    assert application["effective_admission"] == application[
        "requested_admission"
    ]


def test_policy_above_startup_ceiling_is_clamped_not_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=1,
        max_num_seqs=999,
        max_num_batched_tokens=999999,
    )

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)
    gladius.schedule()

    # Effective ceiling never exceeds startup values, even though the policy
    # requested far more.
    assert gladius.max_num_running_reqs == gladius.startup_max_num_seqs == 16
    assert (
        gladius.max_num_scheduled_tokens
        == gladius.startup_max_num_batched_tokens
        == 8192
    )


def test_corrupt_then_valid_then_stale_generation_sequence(tmp_path, monkeypatch):
    # The lowered ceiling (4) is published *before* the first schedule() call
    # so the running count never starts out above it -- shrinking the
    # ceiling below an already-larger running count is a distinct, separate
    # safety behavior covered by test_running_count_floor_when_ceiling_drops
    # below (the base Scheduler has no preemption path to evict already-
    # admitted requests, so this scheduler floors the ceiling at the current
    # running count rather than violating vLLM's own admission invariant).
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)

    # Step 1: valid snapshot, generation 5, lowers ceiling to 4 from the start.
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=5,
        max_num_seqs=4,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 4

    # Step 2: corrupt write -> keep last-good (still 4).
    (tmp_path / "policy_snapshot.json").write_text("{not valid json")
    gladius.schedule()
    assert gladius.max_num_running_reqs == 4

    # Step 3: a regressed generation (3 < 5) -> rejected, still 4.
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=3,
        max_num_seqs=8,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 4

    # Step 4: a genuinely newer generation (6) -> accepted, ceiling raised to 8.
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=6,
        max_num_seqs=8,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 8


def test_running_count_floor_when_ceiling_drops_below_already_admitted(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)

    # No policy yet: admits up to the startup ceiling (16).
    gladius.schedule()
    assert len(gladius.running) == 16

    # Now lower the ceiling to 4 -- fewer than what's already running. This
    # must not crash (no forced eviction/preemption in phase 1 scope); the
    # scheduler floors the effective ceiling at the current running count
    # instead of violating the base Scheduler's own admission invariant.
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=1,
        max_num_seqs=4,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 16
    assert len(gladius.running) == 16


def test_engine_degrades_when_policy_expires_mid_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)

    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=1,
        max_num_seqs=4,
        ttl_seconds=0.05,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 4

    import time

    time.sleep(0.15)
    gladius.schedule()
    assert gladius.max_num_running_reqs == gladius.startup_max_num_seqs == 16


@pytest.mark.parametrize("bad_value", ["not-an-int", "-5"])
def test_invalid_poll_interval_env_var_falls_back_instead_of_crashing_startup(
    monkeypatch, bad_value
):
    monkeypatch.setenv("GLADIUS_POLICY_POLL_INTERVAL_MS", bad_value)
    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)  # must not raise
    assert gladius._policy_loader is not None


def test_schedule_forwards_call_args_for_cross_version_compat(monkeypatch):
    # Different vLLM versions call Scheduler.schedule() with different
    # signatures (some zero-arg, some with a `throttle_prefills` positional
    # -- see docs/design/gladius_next_steps_h100.md P0-A). GladiusScheduler
    # must forward whatever it's given rather than assuming one fixed shape.
    from vllm.v1.core.sched.scheduler import Scheduler

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)

    real_schedule = Scheduler.schedule
    calls = []

    def fake_base_schedule(self, *args, **kwargs):
        calls.append((args, kwargs))
        return real_schedule(self)

    monkeypatch.setattr(Scheduler, "schedule", fake_base_schedule)

    for req in create_requests(num_requests=2, num_tokens=10, max_tokens=4):
        gladius.add_request(req)

    gladius.schedule()  # zero-arg call shape
    gladius.schedule(True)  # one-arg call shape (e.g. throttle_prefills)

    assert calls == [((), {}), ((True,), {})]


def test_schedule_clamp_and_telemetry_run_exactly_once_per_call_shape(
    tmp_path, monkeypatch
):
    from vllm.v1.core.sched.scheduler import Scheduler

    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))
    real_schedule = Scheduler.schedule
    monkeypatch.setattr(
        Scheduler, "schedule", lambda self, *a, **kw: real_schedule(self)
    )

    vanilla = create_scheduler(
        model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192
    )
    gladius = _build_gladius_scheduler(vanilla)
    for req in create_requests(num_requests=2, num_tokens=10, max_tokens=4):
        gladius.add_request(req)

    gladius.schedule()
    assert gladius._telemetry_writer._step == 1
    gladius.schedule(True)
    assert gladius._telemetry_writer._step == 2
