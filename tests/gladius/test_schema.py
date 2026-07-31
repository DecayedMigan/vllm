"""Pure-Python tests for schema.py helpers -- no vllm import needed."""

import pytest

from gladius_vllm.schema import parse_int_env, parse_iso8601


def test_parse_int_env_missing_returns_default(monkeypatch):
    monkeypatch.delenv("GLADIUS_TEST_VAR", raising=False)
    assert parse_int_env("GLADIUS_TEST_VAR", 42) == 42


def test_parse_int_env_valid_value(monkeypatch):
    monkeypatch.setenv("GLADIUS_TEST_VAR", "7")
    assert parse_int_env("GLADIUS_TEST_VAR", 42) == 7


def test_parse_int_env_non_integer_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GLADIUS_TEST_VAR", "not-an-int")
    assert parse_int_env("GLADIUS_TEST_VAR", 42) == 42


def test_parse_int_env_empty_string_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GLADIUS_TEST_VAR", "")
    assert parse_int_env("GLADIUS_TEST_VAR", 42) == 42


def test_parse_int_env_below_minimum_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GLADIUS_TEST_VAR", "0")
    assert parse_int_env("GLADIUS_TEST_VAR", 42, minimum=1) == 42
    monkeypatch.setenv("GLADIUS_TEST_VAR", "-5")
    assert parse_int_env("GLADIUS_TEST_VAR", 42, minimum=0) == 42


def test_parse_int_env_at_minimum_is_accepted(monkeypatch):
    monkeypatch.setenv("GLADIUS_TEST_VAR", "0")
    assert parse_int_env("GLADIUS_TEST_VAR", 42, minimum=0) == 0


def test_parse_iso8601_accepts_z_suffix():
    dt = parse_iso8601("2026-07-30T12:00:00.000000Z")
    assert dt.tzinfo is not None


def test_parse_iso8601_accepts_explicit_offset():
    dt = parse_iso8601("2026-07-30T12:00:00+00:00")
    assert dt.tzinfo is not None


def test_parse_iso8601_rejects_naive_timestamp():
    with pytest.raises(ValueError):
        parse_iso8601("2026-07-30T12:00:00.000000")
