from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anyask.errors import ProviderAuthError, ProviderError, ProviderNotFoundError
from anyask.providers.cohere import CohereProvider


def _fake_response(
    text="hello", input_tokens=10, output_tokens=20, finish_reason="COMPLETE"
):
    content = [SimpleNamespace(text=text)]
    tokens = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    usage = SimpleNamespace(tokens=tokens)
    message = SimpleNamespace(content=content)
    return SimpleNamespace(message=message, usage=usage, finish_reason=finish_reason)


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="COHERE_API_KEY"):
        CohereProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "sk-from-env")
    p = CohereProvider()
    assert p.name == "cohere"


def test_supports_always_true():
    p = CohereProvider(api_key="sk-test")
    assert p.supports("anything")


@pytest.mark.asyncio
async def test_generate_reasoning_true_raises_provider_not_found_error():
    p = CohereProvider(api_key="sk-test")
    with pytest.raises(ProviderNotFoundError, match="reasoning"):
        await p.generate(prompt="hi", model="command-a", reasoning=True)


def test_generate_sync_reasoning_true_raises_provider_not_found_error():
    p = CohereProvider(api_key="sk-test")
    with pytest.raises(ProviderNotFoundError, match="reasoning"):
        p.generate_sync(prompt="hi", model="command-a", reasoning=True)


@pytest.mark.asyncio
async def test_generate_success_applies_default_temperature_and_max_tokens():
    p = CohereProvider(api_key="sk-test")
    p.client.chat = AsyncMock(return_value=_fake_response())

    result = await p.generate(prompt="hi", model="command-light")

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "COMPLETE"

    _, kwargs = p.client.chat.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_generate_respects_explicit_temperature_and_max_tokens():
    p = CohereProvider(api_key="sk-test")
    p.client.chat = AsyncMock(return_value=_fake_response())

    await p.generate(
        prompt="hi", model="command-light", temperature=0.9, max_tokens=100
    )

    _, kwargs = p.client.chat.call_args
    assert kwargs["temperature"] == 0.9
    assert kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error():
    p = CohereProvider(api_key="sk-test")
    p.client.chat = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        await p.generate(prompt="hi", model="command-light")


def test_generate_sync_success():
    p = CohereProvider(api_key="sk-test")
    p.sync_client.chat = MagicMock(return_value=_fake_response())

    result = p.generate_sync(prompt="hi", model="command-light")

    assert result.content == "hello"
    assert result.usage.total_tokens == 30

    _, kwargs = p.sync_client.chat.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 2048


def test_generate_sync_wraps_exceptions_in_provider_error():
    p = CohereProvider(api_key="sk-test")
    p.sync_client.chat = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi", model="command-light")


def test_generate_sync_usage_none_gives_none_total():
    p = CohereProvider(api_key="sk-test")
    response = SimpleNamespace(
        message=SimpleNamespace(content=[SimpleNamespace(text="hi")]),
        usage=None,
        finish_reason=None,
    )
    p.sync_client.chat = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model="command-light")

    assert result.usage.prompt_tokens is None
    assert result.usage.total_tokens is None
    assert result.finish_reason is None


def test_generate_sync_partial_tokens_gives_none_total():
    # Cohere's usage.tokens has no total_tokens field - total is computed
    # from input+output, but only when BOTH halves are present.
    p = CohereProvider(api_key="sk-test")
    tokens = SimpleNamespace(input_tokens=10, output_tokens=None)
    response = SimpleNamespace(
        message=SimpleNamespace(content=[SimpleNamespace(text="hi")]),
        usage=SimpleNamespace(tokens=tokens),
        finish_reason=None,
    )
    p.sync_client.chat = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model="command-light")

    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None


def test_generate_sync_content_as_plain_string():
    # Cohere's content field can be a plain str instead of a content-block
    # list, depending on the response shape.
    p = CohereProvider(api_key="sk-test")
    response = SimpleNamespace(
        message=SimpleNamespace(content="plain string content"),
        usage=None,
        finish_reason=None,
    )
    p.sync_client.chat = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model="command-light")

    assert result.content == "plain string content"


def test_generate_sync_joins_multiple_content_blocks():
    p = CohereProvider(api_key="sk-test")
    content = [SimpleNamespace(text="hello "), SimpleNamespace(text="world")]
    response = SimpleNamespace(
        message=SimpleNamespace(content=content), usage=None, finish_reason=None
    )
    p.sync_client.chat = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model="command-light")

    assert result.content == "hello world"


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="COHERE_API_KEY"):
        CohereProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "sk-from-env")
    p = CohereProvider.from_env()
    assert isinstance(p, CohereProvider)


def test_fetch_latest_models():
    p = CohereProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(name="command-a"),
        SimpleNamespace(name="command-r-plus"),
    ]
    p.sync_client.models.list = MagicMock(
        return_value=SimpleNamespace(models=fake_models)
    )

    names = p.fetch_latest_models()

    assert names == ["command-a", "command-r-plus"]

    _, kwargs = p.sync_client.models.list.call_args
    assert kwargs["endpoint"] == "chat"
