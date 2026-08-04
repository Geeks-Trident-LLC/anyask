from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llmbridge.errors import ProviderAuthError
from llmbridge.providers.moonshot import MoonshotProvider


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="MOONSHOT_API_KEY"):
        MoonshotProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-from-env")
    p = MoonshotProvider()
    assert p.name == "moonshot"


def test_init_sets_moonshot_base_url():
    p = MoonshotProvider(api_key="sk-test")
    assert "moonshot.ai" in str(p.sync_client.base_url)


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="MOONSHOT_API_KEY"):
        MoonshotProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-from-env")
    p = MoonshotProvider.from_env()
    assert isinstance(p, MoonshotProvider)


def test_fetch_latest_models():
    p = MoonshotProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(id="moonshot-v1-8k"),
        SimpleNamespace(id="moonshot-v1-32k"),
    ]
    p.sync_client.models.list = MagicMock(
        return_value=SimpleNamespace(data=fake_models)
    )

    names = p.fetch_latest_models()

    assert names == ["moonshot-v1-8k", "moonshot-v1-32k"]
