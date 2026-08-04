from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from askllm.errors import ProviderAuthError
from askllm.providers.fireworks import FireworksProvider


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="FIREWORKS_API_KEY"):
        FireworksProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "sk-from-env")
    p = FireworksProvider()
    assert p.name == "fireworks"


def test_init_sets_fireworks_base_url():
    p = FireworksProvider(api_key="sk-test")
    assert "fireworks.ai" in str(p.sync_client.base_url)


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="FIREWORKS_API_KEY"):
        FireworksProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "sk-from-env")
    p = FireworksProvider.from_env()
    assert isinstance(p, FireworksProvider)


def test_fetch_latest_models():
    p = FireworksProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(id="accounts/fireworks/models/llama-v3p1-8b-instruct"),
        SimpleNamespace(id="accounts/fireworks/models/llama-v3p1-70b-instruct"),
    ]
    p.sync_client.models.list = MagicMock(
        return_value=SimpleNamespace(data=fake_models)
    )

    names = p.fetch_latest_models()

    assert names == [
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
    ]
