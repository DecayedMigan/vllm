"""Pure-Python tests for PolicyLoader -- no GPU, no model, no vllm executor.

These do import `gladius_vllm.policy`, which has no vllm-internal
dependencies beyond the package's own `schema.py`/`errors.py`.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gladius_vllm.policy import PolicyCorruptError, PolicyLoader, parse_policy_snapshot

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gladius-control-protocol-v1.json"

ENGINE_ID = "engine-test"
MODEL_ID = "facebook/opt-125m"
STARTUP_SEQS = 64
STARTUP_TOKENS = 4096


def _write(path: Path, **overrides):
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": "1.0.0",
        "generation": 1,
        "policy_id": "policy-1",
        "model_id": MODEL_ID,
        "engine_id": ENGINE_ID,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(seconds=30)).isoformat().replace("+00:00", "Z"),
        "admission": {"max_num_seqs": 16, "max_num_batched_tokens": 512},
    }
    payload.update(overrides)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    return payload


def _loader(path: Path | None, poll_interval_ms: int = 0) -> PolicyLoader:
    return PolicyLoader(
        snapshot_path=path,
        engine_id=ENGINE_ID,
        model_id=MODEL_ID,
        startup_max_num_seqs=STARTUP_SEQS,
        startup_max_num_batched_tokens=STARTUP_TOKENS,
        poll_interval_ms=poll_interval_ms,
    )


def test_missing_file_returns_no_policy_status(tmp_path):
    loader = _loader(tmp_path / "policy_snapshot.json")
    decision = loader.poll()
    assert decision.status == "no_policy"
    assert decision.source == "default"
    assert decision.max_num_seqs == STARTUP_SEQS
    assert decision.max_num_batched_tokens == STARTUP_TOKENS


def test_none_path_returns_no_policy_status():
    loader = _loader(None)
    decision = loader.poll()
    assert decision.status == "no_policy"
    assert decision.max_num_seqs == STARTUP_SEQS


def test_first_valid_load_accepted(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path)
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "active"
    assert decision.source == "file"
    assert decision.max_num_seqs == 16
    assert decision.max_num_batched_tokens == 512
    assert decision.policy_id == "policy-1"
    assert decision.generation == 1


def test_null_admission_fields_fall_back_to_startup_ceiling(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path, admission={"max_num_seqs": None, "max_num_batched_tokens": None})
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "active"
    assert decision.max_num_seqs == STARTUP_SEQS
    assert decision.max_num_batched_tokens == STARTUP_TOKENS


def test_corrupt_json_first_load_falls_back_to_default(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    path.write_text("{not valid json")
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "corrupt"
    assert decision.source == "default"
    assert decision.max_num_seqs == STARTUP_SEQS


def test_corrupt_json_after_good_load_keeps_last_good(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path)
    loader = _loader(path)
    good = loader.poll()
    assert good.status == "active"

    path.write_text("{not valid json")
    decision = loader.poll()
    assert decision.status == "corrupt"
    assert decision.source == "file"
    assert decision.max_num_seqs == 16
    assert decision.generation == 1


def test_missing_required_field_rejected_as_corrupt(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": "1.0.0",
        "generation": 1,
        "policy_id": "policy-1",
        "model_id": MODEL_ID,
        # engine_id missing
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=30)).isoformat(),
    }
    path.write_text(json.dumps(payload))
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "corrupt"


def test_generation_regression_rejected(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path, generation=5)
    loader = _loader(path)
    first = loader.poll()
    assert first.generation == 5

    _write(path, generation=3)
    decision = loader.poll()
    assert decision.status == "rejected_regression"
    assert decision.generation == 5  # unchanged


def test_generation_equal_rejected(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path, generation=5)
    loader = _loader(path)
    loader.poll()

    _write(path, generation=5)
    decision = loader.poll()
    assert decision.status == "rejected_regression"


def test_engine_id_mismatch_rejected(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path, engine_id="some-other-engine")
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "rejected_engine_mismatch"
    assert decision.source == "default"


def test_model_id_mismatch_rejected(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path, model_id="some-other-model")
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "rejected_engine_mismatch"


def test_expired_policy_falls_back_to_default(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    now = datetime.now(timezone.utc)
    _write(
        path,
        created_at=(now - timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
        expires_at=(now - timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
    )
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "expired"
    assert decision.source == "default"
    assert decision.max_num_seqs == STARTUP_SEQS


def test_policy_expires_mid_run_after_being_valid(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    now = datetime.now(timezone.utc)
    _write(
        path,
        created_at=now.isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(milliseconds=50))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    loader = _loader(path, poll_interval_ms=0)
    first = loader.poll()
    assert first.status == "active"

    time.sleep(0.1)
    decision = loader.poll()
    assert decision.status == "expired"
    assert decision.source == "default"


def test_unchanged_file_skips_reparse(tmp_path, monkeypatch):
    path = tmp_path / "policy_snapshot.json"
    _write(path)
    loader = _loader(path)
    loader.poll()

    real_read_text = Path.read_text
    call_count = {"n": 0}

    def counting_read_text(self, *args, **kwargs):
        call_count["n"] += 1
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    loader.poll()
    loader.poll()
    assert call_count["n"] == 0


def test_schema_version_major_mismatch_rejected(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path, schema_version="2.0.0")
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "corrupt"


@pytest.mark.parametrize(
    "admission",
    [
        {"max_num_seqs": 0, "max_num_batched_tokens": 512},
        {"max_num_seqs": 16, "max_num_batched_tokens": -1},
        {"max_num_seqs": "not-an-int", "max_num_batched_tokens": 512},
    ],
)
def test_range_validation_rejects_bad_values(tmp_path, admission):
    path = tmp_path / "policy_snapshot.json"
    _write(path, admission=admission)
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "corrupt"


def test_policy_above_startup_ceiling_passes_through_unclamped_at_loader_level(
    tmp_path,
):
    path = tmp_path / "policy_snapshot.json"
    _write(
        path,
        admission={
            "max_num_seqs": STARTUP_SEQS * 10,
            "max_num_batched_tokens": STARTUP_TOKENS * 10,
        },
    )
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "active"
    # Loader itself does not clamp -- that's GladiusScheduler's job.
    assert decision.max_num_seqs == STARTUP_SEQS * 10
    assert decision.max_num_batched_tokens == STARTUP_TOKENS * 10


def test_unrecognized_admission_key_rejected(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path, admission={"max_num_seqs": 16, "enable_prefix_caching": True})
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "corrupt"


def test_expires_at_before_created_at_rejected(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    now = datetime.now(timezone.utc)
    _write(
        path,
        created_at=now.isoformat().replace("+00:00", "Z"),
        expires_at=(now - timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
    )
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "corrupt"


@pytest.mark.parametrize("payload", [None, [], 42, "just a string"])
def test_non_object_json_payload_rejected_as_corrupt(tmp_path, payload):
    path = tmp_path / "policy_snapshot.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    loader = _loader(path)
    decision = loader.poll()  # must not raise
    assert decision.status == "corrupt"
    assert decision.source == "default"


def test_naive_created_at_rejected_as_corrupt(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    now = datetime.now(timezone.utc)
    _write(
        path,
        created_at=now.replace(tzinfo=None).isoformat(),  # naive, no timezone
        expires_at=(now + timedelta(seconds=30)).isoformat().replace("+00:00", "Z"),
    )
    loader = _loader(path)
    decision = loader.poll()  # must not raise ValueError from naive-timestamp parsing
    assert decision.status == "corrupt"


def test_naive_expires_at_rejected_as_corrupt_not_crash_on_later_expiry_check(tmp_path):
    # A naive expires_at that slipped past parsing would later crash
    # PolicyLoader._is_expired()'s aware-vs-naive comparison; it must be
    # rejected up front instead.
    path = tmp_path / "policy_snapshot.json"
    now = datetime.now(timezone.utc)
    _write(
        path,
        created_at=now.isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(seconds=30)).replace(tzinfo=None).isoformat(),
    )
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "corrupt"
    # Re-polling must not raise even though nothing valid was ever accepted.
    decision = loader.poll()
    assert decision.status in ("corrupt", "no_policy")


def test_admission_missing_key_entirely_rejected(tmp_path):
    # Both admission keys must be *present* (value may be null) -- omitting
    # a key entirely is stricter than passing null for it.
    path = tmp_path / "policy_snapshot.json"
    _write(path, admission={"max_num_seqs": 16})
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "corrupt"


def test_unknown_top_level_field_rejected_as_corrupt(tmp_path):
    path = tmp_path / "policy_snapshot.json"
    _write(path, surprise=True)
    loader = _loader(path)
    decision = loader.poll()
    assert decision.status == "corrupt"


def test_corrupt_write_after_expiry_does_not_resurrect_expired_policy(tmp_path):
    # The exact bug from the design doc's state table: a corrupt (or
    # mismatched/regressed) write arriving *after* the last-good policy has
    # already expired must fall to startup default, not resurrect the
    # expired ceiling with the triggering event's status.
    path = tmp_path / "policy_snapshot.json"
    now = datetime.now(timezone.utc)
    _write(
        path,
        admission={"max_num_seqs": 4, "max_num_batched_tokens": 128},
        created_at=now.isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(milliseconds=50))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    loader = _loader(path, poll_interval_ms=0)
    first = loader.poll()
    assert first.status == "active"
    assert first.max_num_seqs == 4

    time.sleep(0.1)  # let it expire
    path.write_text("{not valid json")  # then a corrupt write arrives
    decision = loader.poll()
    assert decision.status == "corrupt"
    assert decision.source == "default"
    assert decision.max_num_seqs == STARTUP_SEQS  # not the expired 4


class TestParsePolicySnapshotAgainstGoldenFixture:
    """Cross-repo contract fixture -- an identical copy of this file lives in
    ~/CODE/gladius so both repos validate against the same golden data."""

    @staticmethod
    def _fixture():
        return json.loads(FIXTURE_PATH.read_text())

    def test_valid_snapshot_parses(self):
        fixture = self._fixture()
        snapshot = parse_policy_snapshot(fixture["valid_snapshot"])
        assert snapshot.policy_id == fixture["valid_snapshot"]["policy_id"]
        assert snapshot.generation == fixture["valid_snapshot"]["generation"]
        assert (
            snapshot.max_num_seqs
            == fixture["valid_snapshot"]["admission"]["max_num_seqs"]
        )

    def test_all_invalid_snapshots_rejected(self):
        fixture = self._fixture()
        for case in fixture["invalid_snapshots"]:
            with pytest.raises(PolicyCorruptError):
                parse_policy_snapshot(case["payload"])


def _valid_payload(**overrides):
    payload = json.loads(FIXTURE_PATH.read_text())["valid_snapshot"]
    payload = dict(payload)
    payload["admission"] = dict(payload["admission"])
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "schema_version",
    [
        "1",  # major only, no minor.patch
        "1.0",  # major.minor only, no patch
        "1.foo",  # non-numeric component
        1,  # not a string at all
        "01.0.0",  # leading zero -- not strict MAJOR.MINOR.PATCH
        "1.0.0-rc1",  # pre-release suffix not accepted by strict parser
        "",
    ],
)
def test_strict_semver_rejects_non_major_minor_patch_forms(schema_version):
    payload = _valid_payload(schema_version=schema_version)
    with pytest.raises(PolicyCorruptError):
        parse_policy_snapshot(payload)


@pytest.mark.parametrize("schema_version", ["1.0.0", "1.2.3", "1.9.99"])
def test_strict_semver_accepts_well_formed_major_1(schema_version):
    payload = _valid_payload(schema_version=schema_version)
    snapshot = parse_policy_snapshot(payload)
    assert snapshot.schema_version == schema_version


def test_strict_semver_rejects_major_2():
    payload = _valid_payload(schema_version="2.0.0")
    with pytest.raises(PolicyCorruptError):
        parse_policy_snapshot(payload)


def test_stat_permission_error_does_not_escape_poll_first_load(tmp_path, monkeypatch):
    # Path.stat() can raise more than FileNotFoundError (PermissionError,
    # transient I/O errors) -- none of these may escape poll(). With no
    # last-good policy yet, this falls to the startup default. The patch is
    # scoped tightly with monkeypatch.context() so it can't leak into
    # pytest's own Path.stat() usage during teardown/traceback formatting.
    path = tmp_path / "policy_snapshot.json"
    _write(path)  # file exists, but stat() will be made to fail below
    loader = _loader(path)

    def _raise_permission_error(self, *args, **kwargs):
        raise PermissionError("permission denied")

    with monkeypatch.context() as m:
        m.setattr(Path, "stat", _raise_permission_error)
        decision = loader.poll()  # must not raise
    assert decision.status == "corrupt"
    assert decision.source == "default"
    assert decision.max_num_seqs == STARTUP_SEQS


def test_stat_permission_error_after_good_load_keeps_unexpired_last_good(
    tmp_path, monkeypatch
):
    path = tmp_path / "policy_snapshot.json"
    _write(path, admission={"max_num_seqs": 4, "max_num_batched_tokens": 128})
    loader = _loader(path)
    good = loader.poll()
    assert good.status == "active"

    def _raise_permission_error(self, *args, **kwargs):
        raise PermissionError("permission denied")

    with monkeypatch.context() as m:
        m.setattr(Path, "stat", _raise_permission_error)
        decision = loader.poll()  # must not raise
    assert decision.status == "corrupt"
    assert decision.source == "file"
    assert decision.max_num_seqs == 4  # unexpired last-good preserved


def test_stat_generic_oserror_does_not_escape_poll(tmp_path, monkeypatch):
    path = tmp_path / "policy_snapshot.json"
    _write(path)
    loader = _loader(path)

    def _raise_oserror(self, *args, **kwargs):
        raise OSError(5, "I/O error")

    with monkeypatch.context() as m:
        m.setattr(Path, "stat", _raise_oserror)
        decision = loader.poll()  # must not raise
    assert decision.status == "corrupt"
