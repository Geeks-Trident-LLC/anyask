from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llmbridge.errors import ProviderAuthError
from llmbridge.providers.cerebras import CerebrasProvider


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="CEREBRAS_API_KEY"):
        CerebrasProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "sk-from-env")
    p = CerebrasProvider()
    assert p.name == "cerebras"


def test_init_sets_cerebras_base_url():
    p = CerebrasProvider(api_key="sk-test")
    assert "cerebras.ai" in str(p.sync_client.base_url)


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="CEREBRAS_API_KEY"):
        CerebrasProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "sk-from-env")
    p = CerebrasProvider.from_env()
    assert isinstance(p, CerebrasProvider)


def test_fetch_latest_models():
    p = CerebrasProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(id="llama3.1-8b"),
        SimpleNamespace(id="llama3.1-70b"),
    ]
    p.sync_client.models.list = MagicMock(
        return_value=SimpleNamespace(data=fake_models)
    )

    names = p.fetch_latest_models()

    assert names == ["llama3.1-8b", "llama3.1-70b"]
