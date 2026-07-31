"""Shared fixtures for tests/gladius/.

Scoped, autouse env-var handling for the local HF cache: ModelConfig
construction eagerly builds an HTTP client that fails validation against a
SOCKS proxy URL scheme (this environment's ALL_PROXY) even when the model is
already cached locally, unless these are set before any vllm config object
is constructed. Applied via monkeypatch (not module-level os.environ
mutation) so it's undone after each test and can't leak into unrelated
tests collected in the same pytest session.
"""

import pytest


@pytest.fixture(autouse=True)
def _local_hf_cache_no_proxy(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)
