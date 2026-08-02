"""Pure-Python tests for TelemetryWriter -- hand-built fixtures, no vllm
executor/GPU, and no real Scheduler/SchedulerOutput instances needed since
TelemetryWriter only duck-types on a handful of attributes.
"""

import json
from types import SimpleNamespace

import pytest

from gladius_vllm.policy import PolicyDecision
from gladius_vllm.telemetry import TelemetryWriter

EXPECTED_FIELDS = {
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


def _fake_request(num_computed_tokens: int, num_prompt_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_computed_tokens=num_computed_tokens, num_prompt_tokens=num_prompt_tokens
    )


def _fake_scheduler(
    *,
    requests: dict,
    running: list,
    waiting: list,
    skipped_waiting: list,
    max_num_running_reqs: int = 64,
    max_num_scheduled_tokens: int = 4096,
    make_stats=None,
) -> SimpleNamespace:
    stats = SimpleNamespace(
        num_running_reqs=len(running),
        num_waiting_reqs=len(waiting),
        num_skipped_waiting_reqs=len(skipped_waiting),
        kv_cache_usage=0.42,
    )
    return SimpleNamespace(
        requests=requests,
        running=running,
        waiting=waiting,
        skipped_waiting=skipped_waiting,
        # The *effective* ceilings actually applied this step (post-clamp),
        # exactly as GladiusScheduler.schedule() would have set them.
        max_num_running_reqs=max_num_running_reqs,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        make_stats=make_stats or (lambda: stats),
    )


def _fake_output(new_req_ids: list, scheduled_tokens: dict) -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=rid) for rid in new_req_ids],
        num_scheduled_tokens=scheduled_tokens,
        total_num_scheduled_tokens=sum(scheduled_tokens.values()),
    )


def _decision(**overrides) -> PolicyDecision:
    defaults = dict(
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        policy_id="policy-1",
        generation=3,
        status="active",
        source="file",
    )
    defaults.update(overrides)
    return PolicyDecision(**defaults)


def test_writes_one_valid_json_line_with_all_fields(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="engine-1", model_id="model-1")
    scheduler = _fake_scheduler(
        requests={"r1": _fake_request(10, 10)},
        running=["r1"],
        waiting=[],
        skipped_waiting=[],
    )
    output = _fake_output(new_req_ids=["r1"], scheduled_tokens={"r1": 1})
    writer.record(
        scheduler,
        output,
        _decision(),
        policy_poll_ns=11,
        policy_apply_ns=13,
    )
    writer.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record.keys()) == EXPECTED_FIELDS
    assert record["policy_status"] == "active"
    assert record["policy_source"] == "file"
    assert record["decision_id"] == record["policy_id"] == "policy-1"
    assert record["window_id"] is None
    assert record["requested_admission"] == {
        "max_num_seqs": 64,
        "max_num_batched_tokens": 4096,
    }
    assert record["effective_admission"] == {
        "max_num_seqs": 64,
        "max_num_batched_tokens": 4096,
    }
    assert record["clamped"] == {"max_num_seqs": False, "max_num_batched_tokens": False}
    assert record["policy_poll_ns"] == 11
    assert record["policy_apply_ns"] == 13
    assert record["telemetry_write_ns"] >= 0


def test_prefill_decode_split_new_request_counts_as_prefill(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(
        requests={"new1": _fake_request(0, 50)},
        running=[],
        waiting=[],
        skipped_waiting=[],
    )
    output = _fake_output(new_req_ids=["new1"], scheduled_tokens={"new1": 50})
    writer.record(scheduler, output, _decision())
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["num_prefill_reqs"] == 1
    assert record["num_decode_reqs"] == 0


def test_prefill_decode_split_cached_mid_prompt_counts_as_prefill(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(
        requests={
            "cached1": _fake_request(num_computed_tokens=20, num_prompt_tokens=50)
        },
        running=["cached1"],
        waiting=[],
        skipped_waiting=[],
    )
    output = _fake_output(new_req_ids=[], scheduled_tokens={"cached1": 30})
    writer.record(scheduler, output, _decision())
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["num_prefill_reqs"] == 1
    assert record["num_decode_reqs"] == 0


def test_prefill_decode_split_cached_finished_prompt_counts_as_decode(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(
        requests={
            "cached1": _fake_request(num_computed_tokens=50, num_prompt_tokens=50)
        },
        running=["cached1"],
        waiting=[],
        skipped_waiting=[],
    )
    output = _fake_output(new_req_ids=[], scheduled_tokens={"cached1": 1})
    writer.record(scheduler, output, _decision())
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["num_prefill_reqs"] == 0
    assert record["num_decode_reqs"] == 1


def test_clamped_true_when_effective_differs_from_requested(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    # The scheduler clamped down to 16/2048 even though the policy requested
    # far more -- exactly what GladiusScheduler.schedule() does when a
    # policy exceeds the startup ceiling (or is floored by running count).
    scheduler = _fake_scheduler(
        requests={},
        running=[],
        waiting=[],
        skipped_waiting=[],
        max_num_running_reqs=16,
        max_num_scheduled_tokens=2048,
    )
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    decision = _decision(max_num_seqs=999, max_num_batched_tokens=999999)
    writer.record(scheduler, output, decision)
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["requested_admission"] == {
        "max_num_seqs": 999,
        "max_num_batched_tokens": 999999,
    }
    assert record["effective_admission"] == {
        "max_num_seqs": 16,
        "max_num_batched_tokens": 2048,
    }
    assert record["clamped"] == {"max_num_seqs": True, "max_num_batched_tokens": True}


def test_no_path_configured_is_a_silent_noop():
    writer = TelemetryWriter(path=None, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(scheduler, output, _decision())  # must not raise
    writer.close()


def test_sample_every_n_steps_downsamples(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(
        path=path, engine_id="e", model_id="m", sample_every_n_steps=3
    )
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    for _ in range(7):
        writer.record(scheduler, output, _decision())
    writer.close()
    lines = path.read_text().splitlines()
    assert len(lines) == 2  # steps 3 and 6
    assert json.loads(lines[0])["step"] == 3
    assert json.loads(lines[1])["step"] == 6


def test_expires_at_is_always_null(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(scheduler, output, _decision())
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["expires_at"] is None


@pytest.mark.parametrize("bad_value", ["not-an-int", "0", "-3", ""])
def test_invalid_sample_n_env_var_falls_back_to_default_instead_of_crashing(
    tmp_path, monkeypatch, bad_value
):
    monkeypatch.setenv("GLADIUS_TELEMETRY_SAMPLE_N", bad_value)
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")  # must not raise
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(scheduler, output, _decision())  # must not ZeroDivisionError
    writer.close()
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("bad_value", [0, -1])
def test_invalid_explicit_sample_n_falls_back_to_default_instead_of_crashing(
    tmp_path, bad_value
):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(
        path=path, engine_id="e", model_id="m", sample_every_n_steps=bad_value
    )
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(scheduler, output, _decision())  # must not ZeroDivisionError
    writer.close()
    assert len(path.read_text().splitlines()) == 1


def test_unwritable_directory_disables_telemetry_instead_of_raising(tmp_path):
    # Make the parent a file, not a directory, so mkdir/open both fail with
    # OSError -- construction must not raise; it should just disable writes.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    path = blocker / "nested" / "telemetry.jsonl"

    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")  # must not raise
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(scheduler, output, _decision())  # must not raise
    writer.close()  # must not raise


def test_write_failure_mid_run_disables_further_writes_instead_of_raising(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(scheduler, output, _decision())
    assert len(path.read_text().splitlines()) == 1

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    writer._file.write = _raise
    writer.record(scheduler, output, _decision())  # must not raise
    assert writer._file is None

    # Telemetry stays disabled (no crash) on subsequent calls too.
    writer.record(scheduler, output, _decision())
    assert len(path.read_text().splitlines()) == 1


def test_rotation_triggers_once_size_threshold_exceeded(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    # Small enough that a handful of records exceeds it, forcing a rotation.
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m", max_bytes=300)
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})

    for _ in range(20):
        writer.record(scheduler, output, _decision())
    writer.close()

    rotated = sorted(tmp_path.glob("telemetry.jsonl.*"))
    assert len(rotated) >= 1, "expected at least one rotated file"
    # The live file still exists and still has valid content.
    assert path.exists()
    for line in path.read_text().splitlines():
        json.loads(line)  # must not raise -- no split/truncated records


def test_rotation_never_splits_a_jsonl_record_across_files(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m", max_bytes=300)
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})

    for _ in range(30):
        writer.record(scheduler, output, _decision())
    writer.close()

    all_files = [path, *sorted(tmp_path.glob("telemetry.jsonl.*"))]
    total_lines = 0
    for f in all_files:
        for line in f.read_text().splitlines():
            record = json.loads(line)  # every line in every file parses whole
            assert record["engine_id"] == "e"
            total_lines += 1
    assert total_lines == 30


def test_rotation_failure_is_fail_open_write_continues(tmp_path, monkeypatch):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m", max_bytes=1)
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})

    def _raise(*args, **kwargs):
        raise OSError("rename failed")

    import os as os_module

    monkeypatch.setattr(os_module, "replace", _raise)
    writer.record(scheduler, output, _decision())  # must not raise
    writer.record(scheduler, output, _decision())  # must not raise
    writer.close()


def test_stats_construction_failure_is_fail_open_not_just_write_failure(tmp_path):
    # §2 of the canonical contract: "写入、stats 构造、序列化...均 fail-open" --
    # not just the final write/flush. A broken make_stats() must not escape
    # record() either.
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")

    def _broken_make_stats():
        raise RuntimeError("stats subsystem exploded")

    scheduler = _fake_scheduler(
        requests={},
        running=[],
        waiting=[],
        skipped_waiting=[],
        make_stats=_broken_make_stats,
    )
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(scheduler, output, _decision())  # must not raise
    assert writer._file is None
    assert path.read_text() == ""


def test_seal_hashes_telemetry_and_blocks_later_mutation(tmp_path):
    # A formal seal binds the receipt and the acknowledgement, so both have
    # to be real and name the sealed instance. The shared cross-repository
    # fixture supplies them.
    from pathlib import Path

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures" / "gladius-execution-evidence-v2.json"
        ).read_text()
    )
    receipt = fixture["valid_server_start_receipt"]
    instance_id = receipt["server_instance_id"]
    (tmp_path / "server_start_receipt.json").write_text(json.dumps(receipt))
    (tmp_path / "policy_application.json").write_text(
        json.dumps(fixture["valid_applications"]["active"])
    )

    path = tmp_path / "telemetry.jsonl"
    manifest_path = tmp_path / "telemetry_seal.json"
    writer = TelemetryWriter(
        path=path, engine_id=receipt["engine_id"], model_id=receipt["model_id"]
    )
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(
        scheduler,
        output,
        _decision(),
        server_instance_id=instance_id,
        generation_high_watermark=3,
    )
    certified_bytes = path.read_bytes()

    assert writer.seal(manifest_path) is True
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "2.0.0"
    assert manifest["engine_id"] == receipt["engine_id"]
    assert manifest["model_id"] == receipt["model_id"]
    assert manifest["server_instance_id"] == instance_id
    assert manifest["first_scheduler_step"] == 1
    assert manifest["final_scheduler_step"] == 1
    assert manifest["record_count"] == 1
    assert [item["name"] for item in manifest["files"]] == ["telemetry.jsonl"]
    assert len(manifest["files"][0]["sha256"]) == 64

    writer.record(scheduler, output, _decision())
    assert path.read_bytes() == certified_bytes
