from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llmbridge.errors import ProviderAuthError, ProviderError
from llmbridge.providers.gemini import GeminiProvider


def _fake_response(
    text="hello",
    prompt_tokens=10,
    completion_tokens=20,
    total_tokens=30,
    finish_reason="STOP",
):
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=total_tokens,
    )
    candidate = SimpleNamespace(finish_reason=finish_reason)
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=[candidate])


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-from-env")
    p = GeminiProvider()
    assert p.name == "gemini"


def test_supports_always_true():
    p = GeminiProvider(api_key="sk-test")
    assert p.supports("anything")


@pytest.mark.asyncio
async def test_generate_success():
    p = GeminiProvider(api_key="sk-test")
    p.client.models.generate_content = MagicMock(return_value=_fake_response())

    result = await p.generate(prompt="hi", model="gemini-2.5-flash")

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error():
    p = GeminiProvider(api_key="sk-test")
    p.client.models.generate_content = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        await p.generate(prompt="hi", model="gemini-2.5-flash")


@pytest.mark.asyncio
async def test_generate_passes_thinking_budget_and_temperature():
    p = GeminiProvider(api_key="sk-test")
    p.client.models.generate_content = MagicMock(return_value=_fake_response())

    await p.generate(
        prompt="hi", model="gemini-2.5-flash", thinking_budget=5, temperature=0.5
    )

    _, kwargs = p.client.models.generate_content.call_args
    config = kwargs["config"]
    assert config.temperature == 0.5
    assert config.thinking_config.thinking_budget == 5


def test_generate_sync_success():
    p = GeminiProvider(api_key="sk-test")
    p.client.models.generate_content = MagicMock(return_value=_fake_response())

    result = p.generate_sync(prompt="hi", model="gemini-2.5-flash")

    assert result.content == "hello"
    assert result.usage.total_tokens == 30


def test_generate_sync_no_candidates_gives_none_finish_reason():
    p = GeminiProvider(api_key="sk-test")
    response = _fake_response()
    response.candidates = []
    p.client.models.generate_content = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model="gemini-2.5-flash")

    assert result.finish_reason is None


def test_generate_sync_wraps_exceptions_in_provider_error():
    p = GeminiProvider(api_key="sk-test")
    p.client.models.generate_content = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi", model="gemini-2.5-flash")


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="GEMINI_API_KEY"):
        GeminiProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-from-env")
    p = GeminiProvider.from_env()
    assert isinstance(p, GeminiProvider)


def test_fetch_latest_models():
    p = GeminiProvider(api_key="sk-test")
    fake_models = [
        SimpleNamespace(name="models/gemini-2.5-flash"),
        SimpleNamespace(name="models/gemini-2.5-pro"),
    ]
    p.client.models.list = MagicMock(return_value=fake_models)

    names = p.fetch_latest_models()

    assert names == ["models/gemini-2.5-flash", "models/gemini-2.5-pro"]
