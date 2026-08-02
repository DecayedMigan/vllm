"""P0-D: the telemetry seal certifies a stream, not just a byte range.

Covers required tests 8 and 9. A seal that only hashes files proves the
files were copied intact; these tests pin the stronger property the
discovery campaign needs -- that the sealed records are parseable, come from
exactly one server instance, form one strictly increasing step sequence, and
that any later change to a certified segment, to the acknowledgement, or to
the receipt is detectable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gladius_vllm.policy import PolicyDecision
from gladius_vllm.telemetry import (
    POLICY_APPLICATION_FILENAME,
    SERVER_START_RECEIPT_FILENAME,
    TelemetryWriter,
    verify_telemetry_seal,
)
from tests.gladius.evidence_builders import (
    expectation_matching_receipt_on_disk,
)

SHARED_FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "gladius-execution-evidence-v2.json"
    ).read_text()
)
ENGINE_ID = SHARED_FIXTURE["valid_server_start_receipt"]["engine_id"]
MODEL_ID = SHARED_FIXTURE["valid_server_start_receipt"]["model_id"]


INSTANCE = "srv-b98904472fa1ba70b59582d9d249925b"
OTHER_INSTANCE = "srv-40c55870b15263bf31864a1f3e1087d6"


def _decision(generation: int = 42) -> PolicyDecision:
    return PolicyDecision(
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        policy_id=f"policy-{generation}",
        generation=generation,
        status="active",
        source="file",
    )


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


def _record(step: int, *, instance: str | None = INSTANCE) -> dict:
    return {
        "schema_version": "2.0.0",
        "server_instance_id": instance,
        "generation_high_watermark": 42,
        "generation": 42,
        "policy_id": "policy-42",
        "decision_id": "policy-42",
        "window_id": None,
        "model_id": MODEL_ID,
        "engine_id": ENGINE_ID,
        "created_at": "2026-08-02T09:05:00Z",
        "expires_at": None,
        "step": step,
        "num_running_reqs": 0,
        "num_waiting_reqs": 0,
        "num_skipped_waiting_reqs": 0,
        "num_scheduled_reqs": 0,
        "num_scheduled_tokens": 0,
        "num_prefill_reqs": 0,
        "num_decode_reqs": 0,
        "kv_cache_usage": 0.0,
        "policy_status": "active",
        "policy_source": "file",
        "requested_admission": {"max_num_seqs": 8, "max_num_batched_tokens": 2048},
        "effective_admission": {"max_num_seqs": 8, "max_num_batched_tokens": 2048},
        "clamped": {"max_num_seqs": False, "max_num_batched_tokens": False},
        "policy_poll_ns": 0,
        "policy_apply_ns": 0,
        "telemetry_write_ns": 0,
    }


def _write_siblings(tmp_path: Path, *, final_step: int | None = None) -> None:
    """Publish a real receipt and acknowledgement for the sealed instance.

    A seal now binds both siblings *semantically*, so a placeholder stub is
    no longer enough -- which is the point: the reviewed revision sealed
    happily with neither file present.

    `final_step` re-points the acknowledgement at a step the telemetry under
    test actually contains. A live server can only ever acknowledge a step it
    has run, so evidence where it does not is incoherent rather than merely
    inconvenient.
    """
    (tmp_path / SERVER_START_RECEIPT_FILENAME).write_text(
        json.dumps(SHARED_FIXTURE["valid_server_start_receipt"], sort_keys=True)
    )
    application = dict(SHARED_FIXTURE["valid_applications"]["active"])
    if final_step is not None:
        application["scheduler_step"] = final_step
    (tmp_path / POLICY_APPLICATION_FILENAME).write_text(
        json.dumps(application, sort_keys=True)
    )


def _last_step(lines: list[str]) -> int | None:
    """The final parseable step in a set of raw telemetry lines."""
    for line in reversed(lines):
        try:
            return int(json.loads(line)["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return None


def _seal_prewritten(
    tmp_path: Path, lines: list[str], *, with_siblings: bool = True
) -> bool:
    path = tmp_path / "telemetry.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines))
    if with_siblings:
        _write_siblings(tmp_path, final_step=_last_step(lines))
    writer = TelemetryWriter(path=path, engine_id=ENGINE_ID, model_id=MODEL_ID)
    return writer.seal(tmp_path / "telemetry_seal.json")


# --- required test 8: sealing refuses uncertifiable streams --------------


def test_sealing_refuses_an_empty_stream(tmp_path):
    assert _seal_prewritten(tmp_path, []) is False
    assert not (tmp_path / "telemetry_seal.json").exists()


def test_sealing_refuses_malformed_jsonl(tmp_path):
    assert (
        _seal_prewritten(tmp_path, [json.dumps(_record(1)), "{ this line is truncated"])
        is False
    )
    assert not (tmp_path / "telemetry_seal.json").exists()


@pytest.mark.parametrize("second_step", [1, 0])
def test_sealing_refuses_duplicate_or_nonmonotonic_steps(tmp_path, second_step):
    assert (
        _seal_prewritten(
            tmp_path,
            [json.dumps(_record(1)), json.dumps(_record(second_step))],
        )
        is False
    )


def test_sealing_refuses_mixed_server_instances(tmp_path):
    assert (
        _seal_prewritten(
            tmp_path,
            [
                json.dumps(_record(1)),
                json.dumps(_record(2, instance=OTHER_INSTANCE)),
            ],
        )
        is False
    )


def test_sealing_refuses_a_half_native_record(tmp_path):
    broken = _record(1)
    broken["policy_id"] = None

    assert _seal_prewritten(tmp_path, [json.dumps(broken)]) is False


def test_sealing_orders_rotated_segments_numerically(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    # Lexicographic ordering would place `-10` before `-2`, certifying the
    # stream in an order its steps do not follow.
    (tmp_path / "telemetry.jsonl.1700000000000-2").write_text(
        json.dumps(_record(1)) + "\n"
    )
    (tmp_path / "telemetry.jsonl.1700000000001-10").write_text(
        json.dumps(_record(2)) + "\n"
    )
    path.write_text(json.dumps(_record(3)) + "\n")
    _write_siblings(tmp_path)
    writer = TelemetryWriter(path=path, engine_id=ENGINE_ID, model_id=MODEL_ID)

    assert writer.seal(tmp_path / "telemetry_seal.json") is True

    manifest = json.loads((tmp_path / "telemetry_seal.json").read_text())
    assert [entry["name"] for entry in manifest["files"]] == [
        "telemetry.jsonl.1700000000000-2",
        "telemetry.jsonl.1700000000001-10",
        "telemetry.jsonl",
    ]
    assert manifest["record_count"] == 3
    assert manifest["first_scheduler_step"] == 1
    assert manifest["final_scheduler_step"] == 3


def test_seal_binds_the_receipt_and_the_final_application(tmp_path):
    assert _seal_prewritten(tmp_path, [json.dumps(_record(1))]) is True
    receipt_text = (tmp_path / SERVER_START_RECEIPT_FILENAME).read_text()
    application_text = (tmp_path / POLICY_APPLICATION_FILENAME).read_text()

    manifest = json.loads((tmp_path / "telemetry_seal.json").read_text())
    assert manifest["server_instance_id"] == INSTANCE
    assert manifest["generation_high_watermark"] == 42
    assert (
        manifest["attestation_receipt_sha256"]
        == hashlib.sha256(receipt_text.encode()).hexdigest()
    )
    assert (
        manifest["policy_application_sha256"]
        == hashlib.sha256(application_text.encode()).hexdigest()
    )
    assert (
        verify_telemetry_seal(
            tmp_path / "telemetry_seal.json",
            tmp_path,
            expectation=expectation_matching_receipt_on_disk(tmp_path),
        )
        == []
    )


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (SERVER_START_RECEIPT_FILENAME, "server_start_receipt.json changed"),
        (POLICY_APPLICATION_FILENAME, "policy_application.json changed"),
        ("telemetry.jsonl", "telemetry.jsonl changed after sealing"),
    ],
)
def test_verification_detects_a_post_seal_alteration(tmp_path, filename, expected):
    assert _seal_prewritten(tmp_path, [json.dumps(_record(1))]) is True

    (tmp_path / filename).write_text(json.dumps(_record(9)) + "\n")

    errors = verify_telemetry_seal(
        tmp_path / "telemetry_seal.json",
        tmp_path,
        expectation=expectation_matching_receipt_on_disk(tmp_path),
    )
    assert any(expected in error for error in errors), errors


def test_verification_detects_a_missing_certified_segment(tmp_path):
    assert _seal_prewritten(tmp_path, [json.dumps(_record(1))]) is True
    (tmp_path / "telemetry.jsonl").unlink()

    errors = verify_telemetry_seal(
        tmp_path / "telemetry_seal.json",
        tmp_path,
        expectation=expectation_matching_receipt_on_disk(tmp_path),
    )
    assert any("listed-but-missing" in error for error in errors), errors
    assert any(error.startswith("SEAL_SEGMENT_SET_MISMATCH") for error in errors)


def test_verification_rejects_a_seal_that_certifies_nothing(tmp_path):
    manifest_path = tmp_path / "telemetry_seal.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "files": [],
                "record_count": 0,
                "first_scheduler_step": None,
                "final_scheduler_step": None,
                "server_instance_id": INSTANCE,
                "attestation_receipt_sha256": None,
                "policy_application_sha256": None,
            }
        )
    )

    _write_siblings(tmp_path)
    errors = verify_telemetry_seal(
        manifest_path,
        tmp_path,
        expectation=expectation_matching_receipt_on_disk(tmp_path),
    )
    assert any("telemetry seal unusable" in error for error in errors)


# --- required test 9: sealing freezes evidence without stopping serving --


def test_scheduling_continues_and_certified_bytes_stay_identical(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    manifest_path = tmp_path / "telemetry_seal.json"
    _write_siblings(tmp_path, final_step=3)
    writer = TelemetryWriter(path=path, engine_id=ENGINE_ID, model_id=MODEL_ID)
    scheduler = _fake_scheduler()
    output = _fake_output()
    for _ in range(3):
        writer.record(
            scheduler,
            output,
            _decision(),
            server_instance_id=INSTANCE,
            generation_high_watermark=42,
        )

    assert writer.seal(manifest_path) is True
    certified = path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["record_count"] == 3
    assert manifest["final_scheduler_step"] == 3

    # Post-seal traffic keeps the scheduler's step identity moving forward
    # (so a later instance's stream is not silently renumbered) but must not
    # append, rotate, or otherwise mutate a certified segment.
    for _ in range(5):
        writer.record(
            scheduler,
            output,
            _decision(43),
            server_instance_id=INSTANCE,
            generation_high_watermark=43,
        )
    assert writer.sealed is True
    assert writer.step == 8
    assert path.read_bytes() == certified
    assert (
        verify_telemetry_seal(
            manifest_path,
            tmp_path,
            expectation=expectation_matching_receipt_on_disk(tmp_path),
        )
        == []
    )
