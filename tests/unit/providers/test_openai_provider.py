from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anyask.errors import ProviderAuthError, ProviderError
from anyask.providers.openai import OpenAIProvider


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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="OPENAI_API_KEY"):
        OpenAIProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    p = OpenAIProvider()
    assert p.name == "openai"


@pytest.mark.asyncio
async def test_generate_success():
    p = OpenAIProvider(api_key="sk-test")
    p.client.chat.completions.create = AsyncMock(return_value=_fake_response())

    result = await p.generate(prompt="hi", model="gpt-4o-mini")

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "stop"
    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_generate_reasoning_true_sets_default_reasoning_effort():
    p = OpenAIProvider(api_key="sk-test")
    p.client.chat.completions.create = AsyncMock(return_value=_fake_response())

    await p.generate(prompt="hi", model="o3-mini", reasoning=True)

    _, kwargs = p.client.chat.completions.create.call_args
    assert kwargs["reasoning_effort"] == "medium"


def test_generate_sync_reasoning_false_omits_reasoning_effort():
    p = OpenAIProvider(api_key="sk-test")
    p.sync_client.chat.completions.create = MagicMock(return_value=_fake_response())

    p.generate_sync(prompt="hi", model="gpt-4o-mini", reasoning=False)

    _, kwargs = p.sync_client.chat.completions.create.call_args
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error():
    p = OpenAIProvider(api_key="sk-test")
    original = RuntimeError("boom")
    p.client.chat.completions.create = AsyncMock(side_effect=original)

    with pytest.raises(ProviderError) as exc_info:
        await p.generate(prompt="hi", model="gpt-4o-mini")

    assert exc_info.value.__cause__ is original


def test_generate_sync_success():
    p = OpenAIProvider(api_key="sk-test")
    p.sync_client.chat.completions.create = MagicMock(return_value=_fake_response())

    result = p.generate_sync(prompt="hi", model="gpt-4o-mini")

    assert result.content == "hello"
    assert result.usage.total_tokens == 30


def test_generate_sync_wraps_exceptions_in_provider_error():
    p = OpenAIProvider(api_key="sk-test")
    p.sync_client.chat.completions.create = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi", model="gpt-4o-mini")


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="OPENAI_API_KEY"):
        OpenAIProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    p = OpenAIProvider.from_env()
    assert isinstance(p, OpenAIProvider)


def test_fetch_latest_models():
    p = OpenAIProvider(api_key="sk-test")
    fake_models = [SimpleNamespace(id="gpt-4o"), SimpleNamespace(id="gpt-4o-mini")]
    p.sync_client.models.list = MagicMock(
        return_value=SimpleNamespace(data=fake_models)
    )

    names = p.fetch_latest_models()

    assert names == ["gpt-4o", "gpt-4o-mini"]


def test_supports_matches_gpt_and_o_series():
    p = OpenAIProvider(api_key="sk-test")
    assert p.supports("gpt-4o-mini")
    assert p.supports("o1-preview")
    assert not p.supports("claude-3-5-sonnet")
