import importlib.util
import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[3]
SCHED_DIR = ROOT / "vllm/v1/core/sched"


def load_scheduler_module(monkeypatch):
    package_names = [
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    protocol_name = "vllm.v1.core.sched.gladius_protocol"
    protocol_spec = importlib.util.spec_from_file_location(
        protocol_name, SCHED_DIR / "gladius_protocol.py"
    )
    assert protocol_spec is not None and protocol_spec.loader is not None
    protocol = importlib.util.module_from_spec(protocol_spec)
    monkeypatch.setitem(sys.modules, protocol_name, protocol)
    protocol_spec.loader.exec_module(protocol)

    scheduler_base = types.ModuleType("vllm.v1.core.sched.scheduler")

    class Scheduler:
        def __init__(self, vllm_config, *args, **kwargs):
            self.vllm_config = vllm_config
            self.scheduler_config = vllm_config.scheduler_config
            self.max_num_running_reqs = self.scheduler_config.max_num_seqs
            self.running = [object(), object()]
            self.waiting = [object()]
            self.skipped_waiting = [object(), object()]

        def schedule(self):
            assert len(self.running) <= self.max_num_running_reqs
            return SimpleNamespace(
                num_scheduled_tokens={"a": 2, "b": 3},
                total_num_scheduled_tokens=5,
                admission_limit_seen=self.max_num_running_reqs,
            )

        def get_request_counts(self):
            return (
                len(self.running),
                len(self.waiting) + len(self.skipped_waiting),
            )

    scheduler_base.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", scheduler_base)

    module_name = "vllm.v1.core.sched.gladius_scheduler"
    spec = importlib.util.spec_from_file_location(
        module_name, SCHED_DIR / "gladius_scheduler.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(model="Qwen/Qwen3-8B"),
    )


def test_schedule_applies_policy_before_base_scheduler_and_writes_telemetry(
    monkeypatch, tmp_path
):
    module = load_scheduler_module(monkeypatch)
    policy_path = tmp_path / "policy.json"
    telemetry_path = tmp_path / "telemetry.jsonl"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": 7,
                "policy_id": "retained-admission-3",
                "model_id": "Qwen/Qwen3-8B",
                "admission_limit": 3,
                "source_experience_id": "experience-17",
                "created_at": "2026-07-31T00:00:00Z",
                "expires_at": "2026-07-31T00:05:00Z",
            }
        )
    )
    monkeypatch.setenv("GLADIUS_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("GLADIUS_TELEMETRY_PATH", str(telemetry_path))
    monkeypatch.setenv("GLADIUS_ENGINE_ID", "engine-a")
    monkeypatch.setattr(
        module, "utc_now", lambda: datetime(2026, 7, 31, 0, 1, tzinfo=UTC)
    )
    scheduler = module.GladiusScheduler(config(), None, None, 16)

    output = scheduler.schedule()

    assert scheduler.max_num_running_reqs == 3
    assert output.admission_limit_seen == 3
    assert output.total_num_scheduled_tokens == 5
    telemetry = json.loads(telemetry_path.read_text())
    assert telemetry["engine_id"] == "engine-a"
    assert telemetry["generation"] == 7
    assert telemetry["policy_id"] == "retained-admission-3"
    assert telemetry["running"] == 2
    assert telemetry["waiting"] == 3
    assert telemetry["scheduled_requests"] == 2
    assert telemetry["scheduled_tokens"] == 5


def test_telemetry_write_failure_does_not_fail_schedule(monkeypatch, tmp_path):
    module = load_scheduler_module(monkeypatch)
    monkeypatch.delenv("GLADIUS_POLICY_PATH", raising=False)
    monkeypatch.setenv(
        "GLADIUS_TELEMETRY_PATH", str(tmp_path / "missing" / "telemetry.jsonl")
    )
    scheduler = module.GladiusScheduler(config(), None, None, 16)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 5
    assert scheduler.max_num_running_reqs == 8


def test_lower_policy_does_not_evict_or_break_existing_running_requests(
    monkeypatch, tmp_path
):
    module = load_scheduler_module(monkeypatch)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": 7,
                "policy_id": "retained-admission-3",
                "model_id": "Qwen/Qwen3-8B",
                "admission_limit": 3,
                "source_experience_id": "experience-17",
                "created_at": "2026-07-31T00:00:00Z",
                "expires_at": "2026-07-31T00:05:00Z",
            }
        )
    )
    monkeypatch.setenv("GLADIUS_POLICY_PATH", str(policy_path))
    monkeypatch.delenv("GLADIUS_TELEMETRY_PATH", raising=False)
    monkeypatch.setattr(
        module, "utc_now", lambda: datetime(2026, 7, 31, 0, 1, tzinfo=UTC)
    )
    scheduler = module.GladiusScheduler(config(), None, None, 16)
    scheduler.running.extend([object(), object(), object()])

    scheduler.schedule()

    assert len(scheduler.running) == 5
    assert scheduler.max_num_running_reqs == 5
