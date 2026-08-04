from unittest.mock import MagicMock

import pytest

from llmbridge.errors import ProviderAuthError, ProviderError
from llmbridge.providers.bedrock import BedrockProvider

MODEL_ID = "anthropic.claude-haiku-4-5-v1:0"


def _fake_response(
    text="hello",
    input_tokens=10,
    output_tokens=20,
    total_tokens=30,
    stop_reason="end_turn",
):
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens,
        },
        "stopReason": stop_reason,
    }


def test_init_missing_region_raises(monkeypatch):
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.delenv("BEDROCK_DEFAULT_REGION", raising=False)
    with pytest.raises(ProviderAuthError, match="AWS region"):
        BedrockProvider(region=None)


def test_init_uses_aws_region_env_var(monkeypatch):
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
    monkeypatch.delenv("BEDROCK_DEFAULT_REGION", raising=False)
    p = BedrockProvider()
    assert p.region == "us-east-1"


def test_init_uses_aws_default_region_env_var(monkeypatch):
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.setenv("BEDROCK_DEFAULT_REGION", "eu-central-1")
    p = BedrockProvider()
    assert p.region == "eu-central-1"


def test_supports_always_true():
    p = BedrockProvider(region="us-east-1")
    assert p.supports("anything")


@pytest.mark.asyncio
async def test_generate_success_applies_default_temperature_and_max_tokens():
    p = BedrockProvider(region="us-east-1")
    p.client.converse = MagicMock(return_value=_fake_response())

    result = await p.generate(prompt="hi", model=MODEL_ID)

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "end_turn"

    _, kwargs = p.client.converse.call_args
    assert kwargs["modelId"] == MODEL_ID
    assert kwargs["inferenceConfig"]["temperature"] == 0.2
    assert kwargs["inferenceConfig"]["maxTokens"] == 2048
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]


@pytest.mark.asyncio
async def test_generate_respects_explicit_temperature_and_max_tokens():
    p = BedrockProvider(region="us-east-1")
    p.client.converse = MagicMock(return_value=_fake_response())

    await p.generate(prompt="hi", model=MODEL_ID, temperature=0.9, max_tokens=100)

    _, kwargs = p.client.converse.call_args
    assert kwargs["inferenceConfig"]["temperature"] == 0.9
    assert kwargs["inferenceConfig"]["maxTokens"] == 100


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error():
    p = BedrockProvider(region="us-east-1")
    p.client.converse = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        await p.generate(prompt="hi", model=MODEL_ID)


def test_generate_sync_success():
    p = BedrockProvider(region="us-east-1")
    p.client.converse = MagicMock(return_value=_fake_response())

    result = p.generate_sync(prompt="hi", model=MODEL_ID)

    assert result.content == "hello"
    assert result.usage.total_tokens == 30


def test_generate_sync_wraps_exceptions_in_provider_error():
    p = BedrockProvider(region="us-east-1")
    p.client.converse = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi", model=MODEL_ID)


def test_generate_sync_usage_missing_gives_none_values():
    p = BedrockProvider(region="us-east-1")
    response = {"output": {"message": {"content": [{"text": "hi"}]}}}
    p.client.converse = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model=MODEL_ID)

    assert result.usage.prompt_tokens is None
    assert result.usage.total_tokens is None
    assert result.finish_reason is None


def test_generate_sync_joins_multiple_content_blocks():
    p = BedrockProvider(region="us-east-1")
    response = {
        "output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
    }
    p.client.converse = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model=MODEL_ID)

    assert result.content == "hello world"


def test_from_env_missing_region_raises(monkeypatch):
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.delenv("BEDROCK_DEFAULT_REGION", raising=False)
    with pytest.raises(ProviderAuthError, match="BEDROCK_REGION"):
        BedrockProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
    p = BedrockProvider.from_env()
    assert isinstance(p, BedrockProvider)


def test_fetch_latest_models(monkeypatch):
    p = BedrockProvider(region="us-east-1")

    fake_control_client = MagicMock()
    fake_control_client.list_foundation_models.return_value = {
        "modelSummaries": [
            {"modelId": "anthropic.claude-opus-4-8-v1:0"},
            {"modelId": "meta.llama4-maverick-v1:0"},
        ]
    }

    import llmbridge.providers.bedrock as bedrock_module

    monkeypatch.setattr(
        bedrock_module.boto3, "client", lambda *a, **k: fake_control_client
    )

    names = p.fetch_latest_models()

    assert names == ["anthropic.claude-opus-4-8-v1:0", "meta.llama4-maverick-v1:0"]
