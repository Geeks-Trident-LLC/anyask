from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from anyask.errors import ProviderAuthError, ProviderError
from anyask.providers.azure import AzureOpenAIProvider, build_azure_endpoint


def _fake_result(
    content="hello",
    prompt_tokens=10,
    completion_tokens=20,
    total_tokens=30,
    finish_reason="stop",
):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_provider():
    return AzureOpenAIProvider(
        api_key="sk-test",
        endpoint="https://example.azure.com",
        api_version="2024-02-15-preview",
        deployment="gpt-4o-deployment",
    )


# ============================================================
# build_azure_endpoint
# ============================================================


def test_build_azure_endpoint_appends_deployment_path():
    result = build_azure_endpoint("https://example.azure.com", "my-deployment")
    assert result == "https://example.azure.com/openai/deployments/my-deployment"


def test_build_azure_endpoint_strips_trailing_slash():
    result = build_azure_endpoint("https://example.azure.com/", "my-deployment")
    assert result == "https://example.azure.com/openai/deployments/my-deployment"


def test_build_azure_endpoint_already_full_deployment_unchanged():
    full = "https://example.azure.com/openai/deployments/my-deployment"
    result = build_azure_endpoint(full, "ignored-deployment")
    assert result == full


# ============================================================
# AzureOpenAIProvider
# ============================================================


def test_init_builds_full_endpoint():
    p = _make_provider()
    assert (
        p.endpoint == "https://example.azure.com/openai/deployments/gpt-4o-deployment"
    )
    assert p.deployment == "gpt-4o-deployment"
    assert p.api_version == "2024-02-15-preview"


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="AZURE_API_KEY"):
        AzureOpenAIProvider(endpoint="https://example.azure.com", deployment="dep")


def test_init_missing_endpoint_raises(monkeypatch):
    monkeypatch.delenv("AZURE_ENDPOINT", raising=False)
    with pytest.raises(ProviderAuthError, match="AZURE_ENDPOINT"):
        AzureOpenAIProvider(api_key="sk-test", deployment="dep")


def test_init_missing_deployment_raises(monkeypatch):
    monkeypatch.delenv("AZURE_DEPLOYMENT", raising=False)
    with pytest.raises(ProviderAuthError, match="AZURE_DEPLOYMENT"):
        AzureOpenAIProvider(api_key="sk-test", endpoint="https://example.azure.com")


def test_supports_always_true():
    p = _make_provider()
    assert p.supports("anything")


@pytest.mark.asyncio
async def test_generate_success():
    p = _make_provider()
    p.client.complete = MagicMock(return_value=_fake_result())

    result = await p.generate(prompt="hi", model="gpt-4o-deployment")

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "stop"
    assert result.provider == "azure"


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error():
    p = _make_provider()
    p.client.complete = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        await p.generate(prompt="hi")


def test_generate_sync_success():
    p = _make_provider()
    p.client.complete = MagicMock(return_value=_fake_result())

    result = p.generate_sync(prompt="hi")

    assert result.content == "hello"
    assert result.usage.completion_tokens == 20


def test_generate_sync_wraps_exceptions_in_provider_error():
    p = _make_provider()
    p.client.complete = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi")


def test_fetch_latest_models_returns_deployment_only():
    p = _make_provider()
    assert p.fetch_latest_models() == ["gpt-4o-deployment"]


# ============================================================
# from_env
# ============================================================


def test_from_env_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_ENDPOINT", "https://example.azure.com")
    monkeypatch.setenv("AZURE_DEPLOYMENT", "dep")

    with pytest.raises(ProviderAuthError, match="AZURE_API_KEY"):
        AzureOpenAIProvider.from_env()


def test_from_env_missing_endpoint_raises(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "sk-test")
    monkeypatch.delenv("AZURE_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_DEPLOYMENT", "dep")

    with pytest.raises(ProviderAuthError, match="AZURE_ENDPOINT"):
        AzureOpenAIProvider.from_env()


def test_from_env_missing_deployment_raises(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "sk-test")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://example.azure.com")
    monkeypatch.delenv("AZURE_DEPLOYMENT", raising=False)

    with pytest.raises(ProviderAuthError, match="AZURE_DEPLOYMENT"):
        AzureOpenAIProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "sk-test")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://example.azure.com")
    monkeypatch.setenv("AZURE_DEPLOYMENT", "dep")
    monkeypatch.delenv("AZURE_API_VERSION", raising=False)

    p = AzureOpenAIProvider.from_env()

    assert isinstance(p, AzureOpenAIProvider)
    assert p.deployment == "dep"
    assert p.api_version == "2024-02-15-preview"


def test_from_env_custom_api_version(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "sk-test")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://example.azure.com")
    monkeypatch.setenv("AZURE_DEPLOYMENT", "dep")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-06-01")

    p = AzureOpenAIProvider.from_env()

    assert p.api_version == "2024-06-01"
