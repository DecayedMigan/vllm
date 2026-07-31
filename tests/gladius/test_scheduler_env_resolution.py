"""Unit tests for GladiusScheduler's env-var resolution helpers, isolated
from the full scheduler construction (which tests/gladius/test_gladius_scheduler_cpu.py
covers, but whose autouse fixture monkeypatches _resolve_poll_interval_ms
directly for deterministic polling -- these tests target the real function).
"""

from gladius_vllm.scheduler import _resolve_poll_interval_ms
from gladius_vllm.schema import DEFAULT_POLICY_POLL_INTERVAL_MS


def test_env_poll_interval_of_zero_falls_back_to_production_default(monkeypatch):
    # Production forbids 0 (would mean unconditional Path.stat() on every
    # single scheduling step regardless of QPS); only the explicit
    # test-injection hook (monkeypatching _resolve_poll_interval_ms itself)
    # may force 0, never the env var.
    monkeypatch.setenv("GLADIUS_POLICY_POLL_INTERVAL_MS", "0")
    assert _resolve_poll_interval_ms() == DEFAULT_POLICY_POLL_INTERVAL_MS


def test_env_poll_interval_negative_falls_back_to_production_default(monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_POLL_INTERVAL_MS", "-5")
    assert _resolve_poll_interval_ms() == DEFAULT_POLICY_POLL_INTERVAL_MS


def test_env_poll_interval_valid_positive_value_is_used(monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_POLL_INTERVAL_MS", "250")
    assert _resolve_poll_interval_ms() == 250


def test_env_poll_interval_minimum_of_one_is_accepted(monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_POLL_INTERVAL_MS", "1")
    assert _resolve_poll_interval_ms() == 1


def test_missing_env_var_uses_default(monkeypatch):
    monkeypatch.delenv("GLADIUS_POLICY_POLL_INTERVAL_MS", raising=False)
    assert _resolve_poll_interval_ms() == DEFAULT_POLICY_POLL_INTERVAL_MS
