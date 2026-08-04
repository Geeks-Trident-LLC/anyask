from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anyask.errors import ProviderError
from anyask.providers.openai_compat import OpenAICompatProvider


def _fake_response(
    content="hello",
    prompt_tokens=5,
    completion_tokens=15,
    total_tokens=20,
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
    return OpenAICompatProvider(
        api_key="sk-test",
        endpoint="https://api.example.com",
        default_model="model-x",
    )


def test_supports_always_true():
    p = _make_provider()
    assert p.supports("anything-at-all")


def test_ignores_unknown_kwargs():
    p = OpenAICompatProvider(
        api_key="sk-test",
        endpoint="https://api.example.com",
        default_model="model-x",
        region="unused",
    )
    assert p.default_model == "model-x"


@pytest.mark.asyncio
async def test_generate_success():
    p = _make_provider()
    p.client.chat.completions.create = AsyncMock(return_value=_fake_response())

    result = await p.generate(prompt="hi", model="model-x")

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 5
    assert result.usage.total_tokens == 20
    assert result.finish_reason == "stop"
    assert result.provider == "openai_compat"
    assert result.model == "model-x"


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error():
    p = _make_provider()
    p.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        await p.generate(prompt="hi", model="model-x")


@pytest.mark.asyncio
async def test_generate_reasoning_true_sets_default_reasoning_effort():
    p = _make_provider()
    p.client.chat.completions.create = AsyncMock(return_value=_fake_response())

    await p.generate(prompt="hi", model="model-x", reasoning=True)

    _, kwargs = p.client.chat.completions.create.call_args
    assert kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_generate_reasoning_true_respects_explicit_reasoning_effort():
    p = _make_provider()
    p.client.chat.completions.create = AsyncMock(return_value=_fake_response())

    await p.generate(
        prompt="hi", model="model-x", reasoning=True, reasoning_effort="high"
    )

    _, kwargs = p.client.chat.completions.create.call_args
    assert kwargs["reasoning_effort"] == "high"


def test_generate_sync_reasoning_false_omits_reasoning_effort():
    p = _make_provider()
    p.sync_client.chat.completions.create = MagicMock(return_value=_fake_response())

    p.generate_sync(prompt="hi", model="model-x", reasoning=False)

    _, kwargs = p.sync_client.chat.completions.create.call_args
    assert "reasoning_effort" not in kwargs


def test_generate_sync_applies_default_temperature_and_max_tokens():
    p = _make_provider()
    p.sync_client.chat.completions.create = MagicMock(return_value=_fake_response())

    p.generate_sync(prompt="hi", model="model-x")

    _, kwargs = p.sync_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 2048


def test_generate_sync_reasoning_true_omits_temperature_and_max_tokens():
    p = _make_provider()
    p.sync_client.chat.completions.create = MagicMock(return_value=_fake_response())

    p.generate_sync(prompt="hi", model="model-x", reasoning=True)

    _, kwargs = p.sync_client.chat.completions.create.call_args
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


def test_generate_sync_success():
    p = _make_provider()
    p.sync_client.chat.completions.create = MagicMock(return_value=_fake_response())

    result = p.generate_sync(prompt="hi", model="model-x")

    assert result.content == "hello"
    assert result.usage.completion_tokens == 15


def test_generate_sync_wraps_exceptions_in_provider_error():
    p = _make_provider()
    p.sync_client.chat.completions.create = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi", model="model-x")
