"""Pure-Python policy-application acknowledgement tests."""

import json
from pathlib import Path

from gladius_vllm.application import (
    PolicyApplicationWriter,
    parse_policy_application,
)
from gladius_vllm.policy import PolicyDecision


def _decision(**overrides) -> PolicyDecision:
    values = {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 2048,
        "policy_id": "policy-42",
        "generation": 42,
        "status": "active",
        "source": "file",
    }
    values.update(overrides)
    return PolicyDecision(**values)


def test_writes_active_application_atomically(tmp_path) -> None:
    path = tmp_path / "policy_application.json"
    writer = PolicyApplicationWriter(path, "engine-a", "model-a")

    writer.record(
        _decision(),
        scheduler_step=1037,
        effective_max_num_seqs=4,
        effective_max_num_batched_tokens=2048,
    )

    payload = json.loads(path.read_text())
    assert payload["state"] == "active"
    assert payload["generation"] == 42
    assert payload["policy_id"] == payload["decision_id"] == "policy-42"
    assert payload["scheduler_step"] == 1037
    assert payload["requested_admission"] == {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 2048,
    }
    assert payload["effective_admission"] == payload["requested_admission"]
    assert payload["clamped"] == {
        "max_num_seqs": False,
        "max_num_batched_tokens": False,
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_attriting_application_becomes_active_without_new_generation(tmp_path) -> None:
    path = tmp_path / "policy_application.json"
    writer = PolicyApplicationWriter(path, "engine-a", "model-a")
    decision = _decision()

    writer.record(
        decision,
        scheduler_step=12,
        effective_max_num_seqs=8,
        effective_max_num_batched_tokens=2048,
    )
    assert json.loads(path.read_text())["state"] == "attriting"

    writer.record(
        decision,
        scheduler_step=15,
        effective_max_num_seqs=4,
        effective_max_num_batched_tokens=2048,
    )
    payload = json.loads(path.read_text())
    assert payload["state"] == "active"
    assert payload["scheduler_step"] == 15
    assert payload["generation"] == 42


def test_unchanged_application_does_not_replace_file(tmp_path) -> None:
    path = tmp_path / "policy_application.json"
    writer = PolicyApplicationWriter(path, "engine-a", "model-a")

    for step in (1, 2):
        writer.record(
            _decision(),
            scheduler_step=step,
            effective_max_num_seqs=4,
            effective_max_num_batched_tokens=2048,
        )
        if step == 1:
            first_inode = path.stat().st_ino

    assert path.stat().st_ino == first_inode
    assert json.loads(path.read_text())["scheduler_step"] == 1


def test_default_decision_is_reported_as_fallback(tmp_path) -> None:
    path = tmp_path / "policy_application.json"
    writer = PolicyApplicationWriter(path, "engine-a", "model-a")

    writer.record(
        _decision(
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            policy_id=None,
            generation=None,
            status="expired",
            source="default",
        ),
        scheduler_step=20,
        effective_max_num_seqs=16,
        effective_max_num_batched_tokens=8192,
    )

    payload = json.loads(path.read_text())
    assert payload["state"] == "fallback"
    # Schema 1.x reported a synthetic generation 0 / "startup-default"
    # identity here, which invented a policy the scheduler never accepted.
    # Execution-evidence 2.0.0 states the truth: no policy is applied.
    assert payload["generation"] is None
    assert payload["policy_id"] is None
    assert payload["decision_id"] is None


def test_application_write_failure_is_fail_open(tmp_path, monkeypatch) -> None:
    path = tmp_path / "policy_application.json"
    writer = PolicyApplicationWriter(path, "engine-a", "model-a")

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("gladius_vllm.atomic.os.replace", _raise)
    writer.record(
        _decision(),
        scheduler_step=1,
        effective_max_num_seqs=4,
        effective_max_num_batched_tokens=2048,
    )
    writer.record(
        _decision(),
        scheduler_step=2,
        effective_max_num_seqs=4,
        effective_max_num_batched_tokens=2048,
    )

    assert not path.exists()


def test_shared_fixture_parses_as_canonical_application() -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "gladius-control-protocol-v1.json"
    )
    payload = json.loads(fixture_path.read_text())["valid_policy_application"]

    application = parse_policy_application(payload)

    assert application.engine_id == "engine-a"
    assert application.generation == 42
    assert application.scheduler_step == 901
    assert application.state == "active"


def test_invalid_runtime_state_is_fail_open(tmp_path) -> None:
    path = tmp_path / "policy_application.json"
    writer = PolicyApplicationWriter(path, "engine-a", "model-a")

    writer.record(
        _decision(),
        scheduler_step=1,
        effective_max_num_seqs=0,
        effective_max_num_batched_tokens=2048,
    )

    assert not path.exists()
