from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llmbridge.errors import ProviderAuthError
from llmbridge.providers.deepseek import DeepSeekProvider


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="DEEPSEEK_API_KEY"):
        DeepSeekProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    p = DeepSeekProvider()
    assert p.name == "deepseek"


def test_init_sets_deepseek_base_url():
    p = DeepSeekProvider(api_key="sk-test")
    assert "deepseek.com" in str(p.sync_client.base_url)


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="DEEPSEEK_API_KEY"):
        DeepSeekProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    p = DeepSeekProvider.from_env()
    assert isinstance(p, DeepSeekProvider)


def test_fetch_latest_models():
    p = DeepSeekProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(id="deepseek-chat"),
        SimpleNamespace(id="deepseek-coder"),
    ]
    p.sync_client.models.list = MagicMock(
        return_value=SimpleNamespace(data=fake_models)
    )

    names = p.fetch_latest_models()

    assert names == ["deepseek-chat", "deepseek-coder"]
