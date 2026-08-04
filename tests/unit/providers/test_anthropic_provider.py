from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anyask.errors import ProviderAuthError, ProviderError
from anyask.providers.anthropic import AnthropicProvider


def _fake_response(
    text="hello", input_tokens=10, output_tokens=20, stop_reason="end_turn"
):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[block], usage=usage, stop_reason=stop_reason)


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    p = AnthropicProvider()
    assert p.name == "anthropic"


def test_init_ignores_unknown_kwargs():
    p = AnthropicProvider(api_key="sk-test", region="us-east-1", unknown="ignored")
    assert p.name == "anthropic"


def test_supports_always_true():
    p = AnthropicProvider(api_key="sk-test")
    assert p.supports("anything")


@pytest.mark.asyncio
async def test_generate_success_applies_default_temperature_and_max_tokens():
    p = AnthropicProvider(api_key="sk-test")
    p.client.messages.create = AsyncMock(return_value=_fake_response())

    result = await p.generate(prompt="hi", model="claude-sonnet-4-5")

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "end_turn"
    assert result.provider == "anthropic"
    assert result.model == "claude-sonnet-4-5"

    _, kwargs = p.client.messages.create.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_generate_respects_explicit_temperature_and_max_tokens():
    p = AnthropicProvider(api_key="sk-test")
    p.client.messages.create = AsyncMock(return_value=_fake_response())

    await p.generate(
        prompt="hi", model="claude-sonnet-4-5", temperature=0.9, max_tokens=100
    )

    _, kwargs = p.client.messages.create.call_args
    assert kwargs["temperature"] == 0.9
    assert kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error():
    p = AnthropicProvider(api_key="sk-test")
    original = RuntimeError("boom")
    p.client.messages.create = AsyncMock(side_effect=original)

    with pytest.raises(ProviderError) as exc_info:
        await p.generate(prompt="hi", model="claude-sonnet-4-5")

    assert exc_info.value.__cause__ is original


def test_generate_sync_success():
    p = AnthropicProvider(api_key="sk-test")
    p.sync_client.messages.create = MagicMock(return_value=_fake_response())

    result = p.generate_sync(prompt="hi", model="claude-sonnet-4-5")

    assert result.content == "hello"
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "end_turn"


def test_generate_sync_wraps_exceptions_in_provider_error():
    p = AnthropicProvider(api_key="sk-test")
    p.sync_client.messages.create = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi", model="claude-sonnet-4-5")


def test_generate_sync_usage_none_gives_none_total():
    p = AnthropicProvider(api_key="sk-test")
    response = SimpleNamespace(
        content=[SimpleNamespace(text="hi")], usage=None, stop_reason=None
    )
    p.sync_client.messages.create = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model="claude-sonnet-4-5")

    assert result.usage.prompt_tokens is None
    assert result.usage.total_tokens is None
    assert result.finish_reason is None


@pytest.mark.asyncio
async def test_generate_reasoning_true_enables_thinking_and_drops_temperature():
    p = AnthropicProvider(api_key="sk-test")
    p.client.messages.create = AsyncMock(return_value=_fake_response())

    await p.generate(prompt="hi", model="claude-sonnet-4-5", reasoning=True)

    _, kwargs = p.client.messages.create.call_args
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "temperature" not in kwargs


def test_generate_sync_reasoning_true_respects_explicit_thinking_budget():
    p = AnthropicProvider(api_key="sk-test")
    p.sync_client.messages.create = MagicMock(return_value=_fake_response())

    p.generate_sync(
        prompt="hi",
        model="claude-sonnet-4-5",
        reasoning=True,
        thinking_budget=4096,
    )

    _, kwargs = p.sync_client.messages.create.call_args
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "thinking_budget" not in kwargs


def test_generate_sync_reasoning_false_keeps_default_temperature():
    p = AnthropicProvider(api_key="sk-test")
    p.sync_client.messages.create = MagicMock(return_value=_fake_response())

    p.generate_sync(prompt="hi", model="claude-sonnet-4-5", reasoning=False)

    _, kwargs = p.sync_client.messages.create.call_args
    assert kwargs["temperature"] == 0.2
    assert "thinking" not in kwargs


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    p = AnthropicProvider.from_env()
    assert isinstance(p, AnthropicProvider)


def test_fetch_latest_models():
    p = AnthropicProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(id="claude-opus-4-8"),
        SimpleNamespace(id="claude-sonnet-4-5"),
    ]
    p.sync_client.models.list = MagicMock(
        return_value=SimpleNamespace(data=fake_models)
    )

    names = p.fetch_latest_models()

    assert names == ["claude-opus-4-8", "claude-sonnet-4-5"]
