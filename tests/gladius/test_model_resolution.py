"""Pure-Python tests for the real-engine test model resolution helper --
no vllm import needed.
"""

import pytest

from tests.gladius._test_model import DEFAULT_TEST_MODEL, resolve_test_model


def test_no_override_returns_default(monkeypatch):
    monkeypatch.delenv("GLADIUS_TEST_MODEL", raising=False)
    assert resolve_test_model() == DEFAULT_TEST_MODEL


def test_hub_style_override_is_used_verbatim(monkeypatch):
    monkeypatch.setenv("GLADIUS_TEST_MODEL", "Qwen/Qwen3-8B")
    assert resolve_test_model() == "Qwen/Qwen3-8B"


def test_existing_local_path_override_is_used(monkeypatch, tmp_path):
    model_dir = tmp_path / "Phi-4-mini-instruct"
    model_dir.mkdir()
    monkeypatch.setenv("GLADIUS_TEST_MODEL", str(model_dir))
    assert resolve_test_model() == str(model_dir)


def test_missing_local_path_fails_early_no_network_fallback(monkeypatch, tmp_path):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("GLADIUS_TEST_MODEL", str(missing))
    with pytest.raises(FileNotFoundError):
        resolve_test_model()


def test_missing_relative_local_path_fails_early(monkeypatch):
    monkeypatch.setenv("GLADIUS_TEST_MODEL", "./no-such-model-dir")
    with pytest.raises(FileNotFoundError):
        resolve_test_model()
