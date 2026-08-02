"""P0-B/P0-C: instance-bound application and telemetry evidence.

Covers required tests 3-7 of the execution-plane remediation requirements.
All pure Python -- `PolicyLoader` reads real files from tmp_path, and the
writers duck-type on a handful of scheduler attributes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gladius_vllm.application import (
    PolicyApplicationWriter,
    parse_policy_application,
    parse_policy_application_v2,
    read_policy_application_v2,
    verify_application_binds_receipt,
)
from gladius_vllm.policy import PolicyDecision, PolicyLoader
from gladius_vllm.telemetry import (
    TelemetrySealError,
    TelemetryWriter,
    parse_telemetry_record_v2,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gladius-execution-evidence-v2.json"
INSTANCE = "srv-b98904472fa1ba70b59582d9d249925b"
OTHER_INSTANCE = "srv-40c55870b15263bf31864a1f3e1087d6"


@pytest.fixture(scope="module")
def shared_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _decision(**overrides) -> PolicyDecision:
    values = {
        "max_num_seqs": 8,
        "max_num_batched_tokens": 2048,
        "policy_id": "policy-42",
        "generation": 42,
        "status": "active",
        "source": "file",
    }
    values.update(overrides)
    return PolicyDecision(**values)


def _write_snapshot(
    path: Path,
    generation: int,
    *,
    created_offset_seconds: int = 0,
    expires_offset_seconds: int = 300,
) -> None:
    """Publish a snapshot whose validity window is relative to now.

    Offsets rather than a TTL so a test can express "already expired by the
    time the loader first reads it" without sleeping: the parser only
    requires `expires_at > created_at`, and the expiry check is against the
    wall clock.
    """
    now = datetime.now(timezone.utc)

    def stamp(offset: int) -> str:
        return (now + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")

    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generation": generation,
                "policy_id": f"policy-{generation}",
                "model_id": "model-a",
                "engine_id": "engine-a",
                "created_at": stamp(created_offset_seconds),
                "expires_at": stamp(expires_offset_seconds),
                "admission": {"max_num_seqs": 8, "max_num_batched_tokens": 2048},
            }
        )
    )


def _loader(path: Path) -> PolicyLoader:
    return PolicyLoader(
        snapshot_path=path,
        engine_id="engine-a",
        model_id="model-a",
        startup_max_num_seqs=32,
        startup_max_num_batched_tokens=8192,
        poll_interval_ms=0,
    )


# --- required test 3 -----------------------------------------------------


def test_application_carries_the_instance_and_the_exact_active_action(tmp_path):
    path = tmp_path / "policy_application.json"
    writer = PolicyApplicationWriter(path, "engine-a", "model-a")

    writer.record(
        _decision(),
        scheduler_step=901,
        effective_max_num_seqs=8,
        effective_max_num_batched_tokens=2048,
        server_instance_id=INSTANCE,
        generation_high_watermark=42,
    )

    application = read_policy_application_v2(path)
    assert application.server_instance_id == INSTANCE
    assert application.state == "active"
    assert application.generation == 42
    assert application.generation_high_watermark == 42
    payload = json.loads(path.read_text())
    assert payload["requested_admission"] == payload["effective_admission"]
    assert payload["requested_admission"] == {
        "max_num_seqs": 8,
        "max_num_batched_tokens": 2048,
    }
    assert payload["clamped"] == {
        "max_num_seqs": False,
        "max_num_batched_tokens": False,
    }


def test_unattested_application_fails_closed_for_a_formal_campaign(tmp_path):
    path = tmp_path / "policy_application.json"
    writer = PolicyApplicationWriter(path, "engine-a", "model-a")

    # The server keeps serving before its receipt is assembled...
    writer.record(
        _decision(),
        scheduler_step=3,
        effective_max_num_seqs=8,
        effective_max_num_batched_tokens=2048,
        server_instance_id=None,
    )
    assert json.loads(path.read_text())["server_instance_id"] is None

    # ...but a campaign reading it strictly refuses to certify the cell.
    with pytest.raises(ValueError, match="server_instance_id is required"):
        read_policy_application_v2(path)
    assert read_policy_application_v2(path, allow_unattested=True).state == "active"


def test_a_clamped_application_can_never_be_active(tmp_path):
    path = tmp_path / "policy_application.json"
    writer = PolicyApplicationWriter(path, "engine-a", "model-a")

    writer.record(
        _decision(),
        scheduler_step=5,
        effective_max_num_seqs=8,
        effective_max_num_batched_tokens=1024,
        server_instance_id=INSTANCE,
        generation_high_watermark=42,
    )

    application = read_policy_application_v2(path)
    assert application.state == "fallback"
    assert json.loads(path.read_text())["clamped"]["max_num_batched_tokens"] is True


# --- required test 4 -----------------------------------------------------


def test_fallback_after_an_accepted_policy_keeps_the_high_watermark(tmp_path):
    snapshot_path = tmp_path / "policy_snapshot.json"
    application_path = tmp_path / "policy_application.json"
    loader = _loader(snapshot_path)
    writer = PolicyApplicationWriter(application_path, "engine-a", "model-a")

    # Accepted on read, but its validity window already closed, so the very
    # same poll returns the expired fallback -- no sleeping required.
    _write_snapshot(
        snapshot_path, 7, created_offset_seconds=-10, expires_offset_seconds=-1
    )
    expired = loader.poll()
    assert loader.generation_high_watermark == 7
    assert expired.status == "expired"
    assert expired.source == "default"
    assert expired.generation is None
    assert loader.generation_high_watermark == 7

    writer.record(
        expired,
        scheduler_step=44,
        effective_max_num_seqs=32,
        effective_max_num_batched_tokens=8192,
        server_instance_id=INSTANCE,
        generation_high_watermark=loader.generation_high_watermark,
    )

    application = read_policy_application_v2(application_path)
    assert application.state == "fallback"
    assert application.generation is None
    assert application.policy_id is None
    assert application.decision_id is None
    # The client can still see which generations this scheduler will reject.
    assert application.generation_high_watermark == 7


# --- required test 5 -----------------------------------------------------


@pytest.mark.parametrize("replayed_generation", [7, 6, 0])
def test_equal_and_lower_generations_stay_rejected_without_regressing(
    tmp_path, replayed_generation
):
    snapshot_path = tmp_path / "policy_snapshot.json"
    loader = _loader(snapshot_path)
    _write_snapshot(snapshot_path, 7)
    assert loader.poll().generation == 7

    _write_snapshot(snapshot_path, replayed_generation)
    loader._last_stat = None
    decision = loader.poll()

    assert decision.status == "rejected_regression"
    assert decision.generation == 7
    assert loader.generation_high_watermark == 7


def test_corrupt_input_does_not_regress_the_high_watermark(tmp_path):
    snapshot_path = tmp_path / "policy_snapshot.json"
    loader = _loader(snapshot_path)
    _write_snapshot(snapshot_path, 11)
    assert loader.poll().generation == 11

    snapshot_path.write_text("{ truncated")
    loader._last_stat = None
    assert loader.poll().status == "corrupt"
    assert loader.generation_high_watermark == 11

    snapshot_path.unlink()
    loader._last_stat = None
    assert loader.poll().status == "no_policy"
    assert loader.generation_high_watermark == 11

    # And the next accepted policy must still be strictly above it.
    _write_snapshot(snapshot_path, 12)
    loader._last_stat = None
    assert loader.poll().generation == 12
    assert loader.generation_high_watermark == 12


def test_application_rejects_a_watermark_below_the_applied_generation(
    shared_fixture,
):
    payload = shared_fixture["invalid_cases"]["regressed_generation_application"]

    with pytest.raises(ValueError, match="generation_high_watermark must be >="):
        parse_policy_application_v2(payload)


# --- required test 6 -----------------------------------------------------


def _fake_scheduler():
    stats = SimpleNamespace(
        num_running_reqs=0,
        num_waiting_reqs=0,
        num_skipped_waiting_reqs=0,
        kv_cache_usage=0.0,
    )
    return SimpleNamespace(
        requests={},
        running=[],
        waiting=[],
        skipped_waiting=[],
        max_num_running_reqs=8,
        max_num_scheduled_tokens=2048,
        make_stats=lambda: stats,
    )


def _fake_output():
    return SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
    )


def test_native_telemetry_nulls_all_three_identity_fields(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")

    writer.record(
        _fake_scheduler(),
        _fake_output(),
        PolicyDecision(
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            policy_id=None,
            generation=None,
            status="no_policy",
            source="default",
        ),
        server_instance_id=INSTANCE,
    )
    writer.close()

    record = json.loads(path.read_text().splitlines()[0])
    assert record["generation"] is None
    assert record["policy_id"] is None
    assert record["decision_id"] is None
    assert record["server_instance_id"] == INSTANCE
    assert parse_telemetry_record_v2(record) is record


def test_file_backed_telemetry_carries_all_three_identity_fields(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")

    writer.record(
        _fake_scheduler(),
        _fake_output(),
        _decision(),
        server_instance_id=INSTANCE,
        generation_high_watermark=42,
    )
    writer.close()

    record = json.loads(path.read_text().splitlines()[0])
    assert record["generation"] == 42
    assert record["policy_id"] == record["decision_id"] == "policy-42"
    assert record["generation_high_watermark"] == 42
    parse_telemetry_record_v2(record)


@pytest.mark.parametrize(
    "nulled",
    [
        ("generation",),
        ("policy_id",),
        ("decision_id",),
        ("generation", "policy_id"),
        ("generation", "decision_id"),
        ("policy_id", "decision_id"),
    ],
)
def test_every_half_native_telemetry_permutation_is_rejected(shared_fixture, nulled):
    record = dict(shared_fixture["valid_telemetry"]["file_backed"])
    for field in nulled:
        record[field] = None

    if set(nulled) == {"policy_id", "decision_id"} or len(nulled) < 3:
        with pytest.raises(TelemetrySealError, match="half-native"):
            parse_telemetry_record_v2(record)


@pytest.mark.parametrize(
    "nulled",
    [
        ("generation",),
        ("policy_id",),
        ("decision_id",),
        ("generation", "policy_id"),
        ("generation", "decision_id"),
        ("policy_id", "decision_id"),
    ],
)
def test_every_half_native_application_permutation_is_rejected(shared_fixture, nulled):
    payload = dict(shared_fixture["valid_applications"]["active"])
    for field in nulled:
        payload[field] = None

    with pytest.raises(ValueError, match="half-native"):
        parse_policy_application_v2(payload)


def test_a_native_application_can_only_be_fallback(shared_fixture):
    payload = dict(shared_fixture["valid_applications"]["fallback_native"])
    payload["state"] = "active"

    with pytest.raises(ValueError, match="native/default decision"):
        parse_policy_application_v2(payload)


# --- required test 7 -----------------------------------------------------


def test_a_second_instances_records_cannot_join_the_first_instances_receipt(
    shared_fixture,
):
    receipt = SimpleNamespace(
        server_instance_id=INSTANCE,
        engine_id="gladius-h100-gpu0",
        model_id="/models/Qwen3-8B",
    )
    matching = parse_policy_application_v2(
        shared_fixture["valid_applications"]["active"]
    )
    assert verify_application_binds_receipt(matching, receipt) == []

    foreign = parse_policy_application_v2(
        shared_fixture["invalid_cases"]["receipt_mismatch_application"]
    )
    errors = verify_application_binds_receipt(foreign, receipt)
    assert len(errors) == 1
    assert OTHER_INSTANCE in errors[0]


def test_cross_instance_telemetry_is_structurally_distinguishable(shared_fixture):
    first = parse_telemetry_record_v2(shared_fixture["valid_telemetry"]["file_backed"])
    second = parse_telemetry_record_v2(
        shared_fixture["invalid_cases"]["cross_instance_telemetry"]
    )

    assert first["server_instance_id"] == INSTANCE
    assert second["server_instance_id"] == OTHER_INSTANCE


# --- schema transition ---------------------------------------------------


def test_a_1x_application_never_satisfies_the_2_0_0_requirement(shared_fixture):
    legacy = shared_fixture["invalid_cases"]["schema_1x_application"]

    # It still parses under the historical parser it was written for...
    assert parse_policy_application(legacy).generation == 42
    # ...and can never be mistaken for evidence that proves an instance.
    with pytest.raises(ValueError, match="do not match the contract"):
        parse_policy_application_v2(legacy)


def test_shared_fixture_matches_both_repositories(shared_fixture):
    assert shared_fixture["contract_version"] == "2.0.0"
    for name in ("active", "attriting", "fallback_native"):
        parse_policy_application_v2(shared_fixture["valid_applications"][name])
    for name in ("file_backed", "native"):
        parse_telemetry_record_v2(shared_fixture["valid_telemetry"][name])
    with pytest.raises(ValueError, match="active application"):
        parse_policy_application_v2(
            shared_fixture["invalid_cases"]["clamped_active_application"]
        )
    with pytest.raises(ValueError, match="server_instance_id is required"):
        parse_policy_application_v2(
            shared_fixture["invalid_cases"]["unattested_application"]
        )
