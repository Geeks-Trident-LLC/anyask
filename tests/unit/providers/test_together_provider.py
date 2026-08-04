from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from askllm.errors import ProviderAuthError
from askllm.providers.together import TogetherProvider


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="TOGETHER_API_KEY"):
        TogetherProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "sk-from-env")
    p = TogetherProvider()
    assert p.name == "together"


def test_init_sets_together_base_url():
    p = TogetherProvider(api_key="sk-test")
    assert "together.xyz" in str(p.sync_client.base_url)


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="TOGETHER_API_KEY"):
        TogetherProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "sk-from-env")
    p = TogetherProvider.from_env()
    assert isinstance(p, TogetherProvider)


def test_fetch_latest_models():
    p = TogetherProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(id="meta-llama/Llama-3.1-8B-Instruct-Turbo"),
        SimpleNamespace(id="meta-llama/Llama-3.1-70B-Instruct-Turbo"),
    ]
    p.sync_client.models.list = MagicMock(
        return_value=SimpleNamespace(data=fake_models)
    )

    names = p.fetch_latest_models()

    assert names == [
        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    ]
