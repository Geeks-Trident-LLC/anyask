from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from askllm.errors import ProviderAuthError
from askllm.providers.xai import XAIProvider


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="XAI_API_KEY"):
        XAIProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "sk-from-env")
    p = XAIProvider()
    assert p.name == "xai"


def test_init_sets_xai_base_url():
    p = XAIProvider(api_key="sk-test")
    assert "x.ai" in str(p.sync_client.base_url)


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="XAI_API_KEY"):
        XAIProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "sk-from-env")
    p = XAIProvider.from_env()
    assert isinstance(p, XAIProvider)


def test_fetch_latest_models():
    p = XAIProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(id="grok-3-mini"),
        SimpleNamespace(id="grok-3"),
    ]
    p.sync_client.models.list = MagicMock(
        return_value=SimpleNamespace(data=fake_models)
    )

    names = p.fetch_latest_models()

    assert names == ["grok-3-mini", "grok-3"]
