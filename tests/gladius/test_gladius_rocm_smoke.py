"""RX 7900 XTX ROCm real-engine smoke test -- P1-C of
docs/design/gladius_next_steps_rocm.md.

Runs the doc's exact suggested sequence against a real vLLM engine on real
GPU hardware, at production-realistic settings (a nonzero
GLADIUS_POLICY_POLL_INTERVAL_MS, not the 0ms test-injection hook used by the
other contract tests):

1. Start at startup ceiling max_num_seqs=16 / max_num_batched_tokens=4096.
2. Publish generation 1: 8 / 2048.
3. Publish generation 2: 2 / 512.
4. Keep concurrent requests running; confirm lowering the ceiling doesn't
   evict them.
5. Let the policy expire; confirm reversion to startup default.
6. Audit telemetry.jsonl for full generation/policy/engine/model/step/
   requested-effective-ceiling/clamped correlation across the lifecycle.

Needs a real model load and a working GPU execution backend -- this is a
manual/CI-gate test, not part of the fast suite. Run:

    pytest tests/gladius/test_gladius_rocm_smoke.py -v -s
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

# See tests/gladius/conftest.py for the autouse HF_HUB_OFFLINE/no-proxy
# fixture required before any vllm config object is constructed.

MODEL = "Qwen/Qwen3-1.7B"  # small model per the doc -- avoid loading 8B first
STARTUP_MAX_NUM_SEQS = 16
STARTUP_MAX_NUM_BATCHED_TOKENS = 4096
POLL_INTERVAL_MS = 100  # production-realistic, per the doc -- not the 0ms test hook


def _write_snapshot(
    directory,
    *,
    engine_id,
    model_id,
    generation,
    max_num_seqs,
    max_num_batched_tokens,
    ttl_seconds,
):
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": "1.0.0",
        "generation": generation,
        "policy_id": f"policy-{generation}",
        "model_id": model_id,
        "engine_id": engine_id,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(seconds=ttl_seconds))
        .isoformat()
        .replace("+00:00", "Z"),
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


def _record_environment_evidence():
    import torch

    assert torch.cuda.is_available(), "no GPU visible to torch"
    device_name = torch.cuda.get_device_name(0)
    print(
        f"\n[env] torch={torch.__version__} hip={torch.version.hip} "
        f"device={device_name} cuda_available={torch.cuda.is_available()}"
    )
    return device_name


@pytest.mark.slow_test
def test_gladius_rocm_smoke(tmp_path, monkeypatch):
    device_name = _record_environment_evidence()

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("GLADIUS_ENGINE_ID", "rx7900xtx-engine-a")
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))
    monkeypatch.setenv("GLADIUS_POLICY_POLL_INTERVAL_MS", str(POLL_INTERVAL_MS))
    monkeypatch.setenv("GLADIUS_TELEMETRY_SAMPLE_N", "1")

    from gladius_vllm.scheduler import GladiusScheduler
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine

    # Step 1: start at the doc's suggested startup ceiling.
    engine_args = EngineArgs(
        model=MODEL,
        enforce_eager=True,
        scheduler_cls=GladiusScheduler,
        max_num_seqs=STARTUP_MAX_NUM_SEQS,
        max_num_batched_tokens=STARTUP_MAX_NUM_BATCHED_TOKENS,
        max_model_len=2048,
        gpu_memory_utilization=0.3,
    )
    engine = LLMEngine.from_engine_args(engine_args=engine_args)
    scheduler = engine.engine_core.engine_core.scheduler
    assert isinstance(scheduler, GladiusScheduler)
    assert scheduler.startup_max_num_seqs == STARTUP_MAX_NUM_SEQS
    assert scheduler.startup_max_num_batched_tokens == STARTUP_MAX_NUM_BATCHED_TOKENS

    sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)

    # A handful of long-running requests admitted under native/no-policy
    # behavior -- used in step 4 to prove lowering the ceiling doesn't evict
    # them.
    for i in range(6):
        engine.add_request(f"r{i}", "Hello, world! " * 4, sampling_params)
    engine.step()
    assert scheduler.max_num_running_reqs == STARTUP_MAX_NUM_SEQS
    running_after_native = {r.request_id for r in scheduler.running}
    assert len(running_after_native) == 6

    # Step 2: publish generation 1 (8 / 2048).
    _write_snapshot(
        tmp_path,
        engine_id=scheduler.engine_id,
        model_id=scheduler.model_id,
        generation=1,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        ttl_seconds=60.0,
    )
    time.sleep(POLL_INTERVAL_MS / 1000.0 + 0.05)  # clear the poll rate-limit window
    engine.step()
    # Step 4 (checked here, mid-sequence): the 6 already-running requests
    # exceed the new cap of 8? No -- 6 < 8, so this step doesn't yet exercise
    # the no-eviction floor; that happens at generation 2 below. For now,
    # just confirm the new ceiling is visible and the running set is intact.
    assert scheduler.max_num_running_reqs == 8
    assert {r.request_id for r in scheduler.running} == running_after_native

    # Step 3: publish generation 2 (2 / 512) -- now *below* the 6 already
    # running. This is the real no-eviction test.
    _write_snapshot(
        tmp_path,
        engine_id=scheduler.engine_id,
        model_id=scheduler.model_id,
        generation=2,
        max_num_seqs=2,
        max_num_batched_tokens=512,
        ttl_seconds=0.4,  # short-lived, so step 5 can observe expiry quickly
    )
    time.sleep(POLL_INTERVAL_MS / 1000.0 + 0.05)
    engine.step()

    # Step 4: confirm no eviction -- all 6 originally-running requests are
    # still running, and the effective ceiling floors at that count (2 would
    # be violated, so it holds at 6) rather than crashing or force-finishing
    # any of them.
    assert {r.request_id for r in scheduler.running} == running_after_native
    assert scheduler.max_num_running_reqs == len(scheduler.running) == 6

    # Step 5: let generation 2 expire, then confirm reversion to the startup
    # default (not generation 1, not a crash) on the next schedule() call.
    time.sleep(0.5)  # past the 0.4s ttl
    engine.step()
    assert scheduler.max_num_running_reqs == STARTUP_MAX_NUM_SEQS

    # Drain everything so telemetry has a settled trailing state.
    for _ in range(200):
        if not engine.has_unfinished_requests():
            break
        engine.step()

    # Step 6: audit telemetry.jsonl for full field correlation across the
    # whole lifecycle -- generation, policy, engine, model, step, and
    # requested/effective ceiling with clamped flags all present and self-
    # consistent on every line.
    telemetry_path = tmp_path / "telemetry.jsonl"
    assert telemetry_path.exists()
    lines = [
        json.loads(line) for line in telemetry_path.read_text().splitlines() if line
    ]
    assert len(lines) >= 4

    statuses_seen = set()
    prev_step = 0
    for line in lines:
        assert line["engine_id"] == "rx7900xtx-engine-a"
        assert line["model_id"] == scheduler.model_id
        assert line["decision_id"] == line["policy_id"]
        assert line["window_id"] is None
        assert line["step"] > prev_step  # strictly increasing
        prev_step = line["step"]
        assert set(line["requested_admission"]) == {
            "max_num_seqs",
            "max_num_batched_tokens",
        }
        assert set(line["effective_admission"]) == {
            "max_num_seqs",
            "max_num_batched_tokens",
        }
        assert set(line["clamped"]) == {"max_num_seqs", "max_num_batched_tokens"}
        # clamped must exactly reflect requested vs effective, every line.
        for key in ("max_num_seqs", "max_num_batched_tokens"):
            assert line["clamped"][key] == (
                line["effective_admission"][key] != line["requested_admission"][key]
            )
        statuses_seen.add(line["policy_status"])

    assert "no_policy" in statuses_seen
    assert "active" in statuses_seen
    assert "expired" in statuses_seen
    assert any(line["generation"] == 1 for line in lines)
    assert any(line["generation"] == 2 for line in lines)

    print(f"\n[smoke] PASSED on {device_name}: {len(lines)} telemetry lines audited")
