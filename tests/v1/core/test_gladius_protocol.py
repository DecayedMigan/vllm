import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "vllm/v1/core/sched/gladius_protocol.py"
FIXTURE_PATH = Path(__file__).with_name("fixtures") / (
    "gladius-control-protocol-v1.json"
)


def load_protocol():
    spec = importlib.util.spec_from_file_location("gladius_protocol", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_payload(**changes):
    payload = json.loads(FIXTURE_PATH.read_text())["valid_snapshot"]
    payload.update(changes)
    return payload


def write_snapshot(path, **changes):
    path.write_text(json.dumps(snapshot_payload(**changes)))


def test_parser_accepts_shared_v1_golden_snapshot():
    protocol = load_protocol()
    fixture = json.loads(FIXTURE_PATH.read_text())

    snapshot = protocol.parse_policy_snapshot(fixture["valid_snapshot"])

    assert snapshot.generation == 7
    assert snapshot.policy_id == "retained-admission-3"
    assert snapshot.model_id == "Qwen/Qwen3-8B"
    assert snapshot.admission_limit == 3
    assert snapshot.source_experience_id == "experience-17"
    assert snapshot.created_at == datetime(2026, 7, 31, tzinfo=UTC)
    assert snapshot.expires_at == datetime(2026, 7, 31, 0, 5, tzinfo=UTC)


def test_parser_rejects_every_shared_invalid_snapshot():
    protocol = load_protocol()
    fixture = json.loads(FIXTURE_PATH.read_text())

    for case in fixture["invalid_snapshots"]:
        with pytest.raises(protocol.ProtocolError, match=case["error"]):
            protocol.parse_policy_snapshot(case["payload"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation", True),
        ("generation", -1),
        ("admission_limit", True),
        ("policy_id", ""),
        ("source_experience_id", 7),
        ("created_at", "2026-07-31T00:00:00"),
    ],
)
def test_parser_rejects_invalid_types_and_values(field, value):
    protocol = load_protocol()

    with pytest.raises(protocol.ProtocolError, match=field):
        protocol.parse_policy_snapshot(snapshot_payload(**{field: value}))


def test_controller_caps_valid_policy_at_startup_limit(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "policy.json"
    write_snapshot(path, admission_limit=100)
    controller = protocol.PolicyController(
        policy_path=path,
        model_id="Qwen/Qwen3-8B",
        startup_admission_limit=8,
    )

    state = controller.refresh(datetime(2026, 7, 31, 0, 1, tzinfo=UTC))

    assert state.admission_limit == 8
    assert state.generation == 7
    assert state.policy_id == "retained-admission-3"
    assert state.error is None


def test_controller_ignores_stale_generation_and_retains_live_policy(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "policy.json"
    write_snapshot(path, generation=8, admission_limit=3)
    controller = protocol.PolicyController(
        policy_path=path,
        model_id="Qwen/Qwen3-8B",
        startup_admission_limit=8,
    )
    now = datetime(2026, 7, 31, 0, 1, tzinfo=UTC)
    assert controller.refresh(now).generation == 8
    write_snapshot(path, generation=7, admission_limit=1)

    state = controller.refresh(now)

    assert state.admission_limit == 3
    assert state.generation == 8
    assert "stale generation" in state.error


def test_controller_never_replays_stale_generation_after_newer_policy_expires(
    tmp_path,
):
    protocol = load_protocol()
    path = tmp_path / "policy.json"
    write_snapshot(path, generation=8, admission_limit=3)
    controller = protocol.PolicyController(
        policy_path=path,
        model_id="Qwen/Qwen3-8B",
        startup_admission_limit=8,
    )
    assert controller.refresh(datetime(2026, 7, 31, 0, 1, tzinfo=UTC)).generation == 8
    write_snapshot(
        path,
        generation=7,
        admission_limit=1,
        expires_at="2026-07-31T00:10:00Z",
    )

    state = controller.refresh(datetime(2026, 7, 31, 0, 6, tzinfo=UTC))

    assert state.admission_limit == 8
    assert state.generation == 0
    assert state.policy_id == "startup"
    assert "stale generation" in state.error


def test_controller_retains_live_policy_after_corrupt_update(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "policy.json"
    write_snapshot(path, admission_limit=3)
    controller = protocol.PolicyController(
        policy_path=path,
        model_id="Qwen/Qwen3-8B",
        startup_admission_limit=8,
    )
    now = datetime(2026, 7, 31, 0, 1, tzinfo=UTC)
    assert controller.refresh(now).admission_limit == 3
    path.write_text("{")

    state = controller.refresh(now)

    assert state.admission_limit == 3
    assert state.generation == 7
    assert "invalid JSON" in state.error


def test_controller_restores_startup_limit_after_policy_expires(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "policy.json"
    write_snapshot(path, admission_limit=3)
    controller = protocol.PolicyController(
        policy_path=path,
        model_id="Qwen/Qwen3-8B",
        startup_admission_limit=8,
    )
    assert (
        controller.refresh(datetime(2026, 7, 31, 0, 1, tzinfo=UTC)).admission_limit == 3
    )

    state = controller.refresh(datetime(2026, 7, 31, 0, 6, tzinfo=UTC))

    assert state.admission_limit == 8
    assert state.generation == 0
    assert state.policy_id == "startup"
    assert "expired" in state.error


def test_controller_rejects_model_mismatch(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "policy.json"
    write_snapshot(path, model_id="another/model")
    controller = protocol.PolicyController(
        policy_path=path,
        model_id="Qwen/Qwen3-8B",
        startup_admission_limit=8,
    )

    state = controller.refresh(datetime(2026, 7, 31, 0, 1, tzinfo=UTC))

    assert state.admission_limit == 8
    assert state.policy_id == "startup"
    assert "model_id" in state.error


def test_controller_rejects_future_policy(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "policy.json"
    write_snapshot(
        path,
        created_at="2026-07-31T00:02:00Z",
        expires_at="2026-07-31T00:07:00Z",
    )
    controller = protocol.PolicyController(
        policy_path=path,
        model_id="Qwen/Qwen3-8B",
        startup_admission_limit=8,
    )

    state = controller.refresh(datetime(2026, 7, 31, 0, 1, tzinfo=UTC))

    assert state.admission_limit == 8
    assert "not active" in state.error


def test_append_telemetry_emits_shared_schema_as_one_json_line(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "telemetry.jsonl"
    timestamp = datetime(2026, 7, 31, 0, 0, 1, tzinfo=UTC)

    protocol.append_telemetry(
        path,
        engine_id="engine-a",
        model_id="Qwen/Qwen3-8B",
        step=42,
        state=protocol.PolicyState(
            admission_limit=3,
            generation=7,
            policy_id="retained-admission-3",
            error=None,
            expires_at=timestamp + timedelta(minutes=4),
        ),
        running=3,
        waiting=9,
        scheduled_requests=2,
        scheduled_tokens=1024,
        timestamp=timestamp,
    )

    assert (
        json.loads(path.read_text())
        == json.loads(FIXTURE_PATH.read_text())["valid_telemetry"]
    )
    assert path.read_bytes().endswith(b"\n")
