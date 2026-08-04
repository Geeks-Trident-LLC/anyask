from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from askllm.errors import ProviderAuthError, ProviderError
from askllm.providers.mistral import MistralProvider


def _fake_response(
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


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="MISTRAL_API_KEY"):
        MistralProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-from-env")
    p = MistralProvider()
    assert p.name == "mistral"


def test_supports_always_true():
    p = MistralProvider(api_key="sk-test")
    assert p.supports("anything")


@pytest.mark.asyncio
async def test_generate_success_applies_default_temperature_and_max_tokens():
    p = MistralProvider(api_key="sk-test")
    p.client.chat.complete_async = AsyncMock(return_value=_fake_response())

    result = await p.generate(prompt="hi", model="mistral-small-latest")

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "stop"

    _, kwargs = p.client.chat.complete_async.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_generate_respects_explicit_temperature_and_max_tokens():
    p = MistralProvider(api_key="sk-test")
    p.client.chat.complete_async = AsyncMock(return_value=_fake_response())

    await p.generate(
        prompt="hi", model="mistral-small-latest", temperature=0.9, max_tokens=100
    )

    _, kwargs = p.client.chat.complete_async.call_args
    assert kwargs["temperature"] == 0.9
    assert kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error():
    p = MistralProvider(api_key="sk-test")
    p.client.chat.complete_async = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        await p.generate(prompt="hi", model="mistral-small-latest")


def test_generate_sync_success():
    p = MistralProvider(api_key="sk-test")
    p.client.chat.complete = MagicMock(return_value=_fake_response())

    result = p.generate_sync(prompt="hi", model="mistral-small-latest")

    assert result.content == "hello"
    assert result.usage.total_tokens == 30

    _, kwargs = p.client.chat.complete.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 2048


def test_generate_sync_wraps_exceptions_in_provider_error():
    p = MistralProvider(api_key="sk-test")
    p.client.chat.complete = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi", model="mistral-small-latest")


def test_generate_sync_usage_none_gives_none_total():
    p = MistralProvider(api_key="sk-test")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason=None)
        ],
        usage=None,
    )
    p.client.chat.complete = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model="mistral-small-latest")

    assert result.usage.prompt_tokens is None
    assert result.usage.total_tokens is None


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="MISTRAL_API_KEY"):
        MistralProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-from-env")
    p = MistralProvider.from_env()
    assert isinstance(p, MistralProvider)


def test_fetch_latest_models():
    p = MistralProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(id="mistral-large-latest"),
        SimpleNamespace(id="mistral-small-latest"),
    ]
    p.client.models.list = MagicMock(return_value=SimpleNamespace(data=fake_models))

    names = p.fetch_latest_models()

    assert names == ["mistral-large-latest", "mistral-small-latest"]
