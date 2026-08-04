from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llmbridge.errors import ProviderAuthError, ProviderError
from llmbridge.providers.vertexai import VertexAIProvider


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


def test_init_missing_project_raises(monkeypatch):
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
    monkeypatch.setenv("VERTEXAI_REGION", "us-central1")
    with pytest.raises(ProviderAuthError, match="GCP project"):
        VertexAIProvider(project=None, region=None)


def test_init_missing_region_raises(monkeypatch):
    monkeypatch.setenv("VERTEXAI_PROJECT", "my-project")
    monkeypatch.delenv("VERTEXAI_REGION", raising=False)
    with pytest.raises(ProviderAuthError, match="GCP location"):
        VertexAIProvider(project=None, region=None)


def test_init_uses_env_vars_when_no_explicit_args(monkeypatch):
    monkeypatch.setenv("VERTEXAI_PROJECT", "env-project")
    monkeypatch.setenv("VERTEXAI_REGION", "europe-west4")
    p = VertexAIProvider()
    assert p.project == "env-project"
    assert p.region == "europe-west4"


def test_supports_always_true():
    p = VertexAIProvider(project="my-project", region="us-central1")
    assert p.supports("anything")


@pytest.mark.asyncio
async def test_generate_success():
    p = VertexAIProvider(project="my-project", region="us-central1")
    p.client.models.generate_content = MagicMock(return_value=_fake_response())

    result = await p.generate(prompt="hi", model="gemini-2.5-flash")

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error():
    p = VertexAIProvider(project="my-project", region="us-central1")
    p.client.models.generate_content = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        await p.generate(prompt="hi", model="gemini-2.5-flash")


@pytest.mark.asyncio
async def test_generate_passes_thinking_budget_and_temperature():
    p = VertexAIProvider(project="my-project", region="us-central1")
    p.client.models.generate_content = MagicMock(return_value=_fake_response())

    await p.generate(
        prompt="hi", model="gemini-2.5-flash", thinking_budget=5, temperature=0.5
    )

    _, kwargs = p.client.models.generate_content.call_args
    config = kwargs["config"]
    assert config.temperature == 0.5
    assert config.thinking_config.thinking_budget == 5


def test_generate_sync_success():
    p = VertexAIProvider(project="my-project", region="us-central1")
    p.client.models.generate_content = MagicMock(return_value=_fake_response())

    result = p.generate_sync(prompt="hi", model="gemini-2.5-flash")

    assert result.content == "hello"
    assert result.usage.total_tokens == 30


def test_generate_sync_wraps_exceptions_in_provider_error():
    p = VertexAIProvider(project="my-project", region="us-central1")
    p.client.models.generate_content = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi", model="gemini-2.5-flash")


def test_from_env_missing_project_raises(monkeypatch):
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
    monkeypatch.setenv("VERTEXAI_REGION", "us-central1")
    with pytest.raises(ProviderAuthError, match="VERTEXAI_PROJECT"):
        VertexAIProvider.from_env()


def test_from_env_missing_region_raises(monkeypatch):
    monkeypatch.setenv("VERTEXAI_PROJECT", "my-project")
    monkeypatch.delenv("VERTEXAI_REGION", raising=False)
    with pytest.raises(ProviderAuthError, match="VERTEXAI_REGION"):
        VertexAIProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("VERTEXAI_PROJECT", "my-project")
    monkeypatch.setenv("VERTEXAI_REGION", "us-central1")
    p = VertexAIProvider.from_env()
    assert isinstance(p, VertexAIProvider)


def test_fetch_latest_models():
    p = VertexAIProvider(project="my-project", region="us-central1")
    fake_models = [
        SimpleNamespace(name="gemini-2.5-pro"),
        SimpleNamespace(name="gemini-2.5-flash"),
    ]
    p.client.models.list = MagicMock(return_value=fake_models)

    names = p.fetch_latest_models()

    assert names == ["gemini-2.5-pro", "gemini-2.5-flash"]
