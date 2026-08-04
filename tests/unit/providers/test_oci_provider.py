from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import anyask.providers.oci as oci_provider_module
from anyask.errors import ProviderAuthError, ProviderError, ProviderNotFoundError
from anyask.providers.oci import OCIProvider

MODEL_ID = "meta.llama-3.3-70b-instruct"
COMPARTMENT_ID = "ocid1.compartment.oc1..fake"


def _fake_response(
    text="hello",
    prompt_tokens=10,
    completion_tokens=20,
    total_tokens=30,
    finish_reason="COMPLETE",
):
    content_block = SimpleNamespace(text=text)
    message = SimpleNamespace(content=[content_block])
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    chat_response = SimpleNamespace(choices=[choice], usage=usage)
    return SimpleNamespace(data=SimpleNamespace(chat_response=chat_response))


def _patch_oci(monkeypatch, config=None, from_file_side_effect=None):
    if from_file_side_effect is not None:
        from_file = MagicMock(side_effect=from_file_side_effect)
    else:
        from_file = MagicMock(return_value=config if config is not None else {})
    monkeypatch.setattr(oci_provider_module.oci.config, "from_file", from_file)
    monkeypatch.setattr(
        oci_provider_module.oci.generative_ai_inference,
        "GenerativeAiInferenceClient",
        MagicMock(),
    )
    monkeypatch.setattr(
        oci_provider_module.oci.generative_ai, "GenerativeAiClient", MagicMock()
    )


def test_init_missing_compartment_id_raises(monkeypatch):
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})

    with pytest.raises(ProviderAuthError, match="compartment"):
        OCIProvider(compartment_id=None, region=None)


def test_init_missing_config_file_raises(monkeypatch):
    _patch_oci(monkeypatch, from_file_side_effect=FileNotFoundError("no config"))

    with pytest.raises(ProviderAuthError, match="OCI config"):
        OCIProvider(compartment_id=COMPARTMENT_ID, region="us-chicago-1")


def test_init_missing_region_raises(monkeypatch):
    monkeypatch.delenv("OCI_REGION", raising=False)
    _patch_oci(monkeypatch, config={})

    with pytest.raises(ProviderAuthError, match="region"):
        OCIProvider(compartment_id=COMPARTMENT_ID, region=None)


def test_init_uses_region_from_config_when_not_explicit(monkeypatch):
    monkeypatch.delenv("OCI_REGION", raising=False)
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})

    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)

    assert p.region == "us-chicago-1"


def test_init_explicit_region_overrides_config(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})

    p = OCIProvider(compartment_id=COMPARTMENT_ID, region="eu-frankfurt-1")

    assert p.region == "eu-frankfurt-1"


def test_supports_always_true(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})
    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)
    assert p.supports("anything")


@pytest.mark.asyncio
async def test_generate_reasoning_true_raises_provider_not_found_error(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})
    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)
    with pytest.raises(ProviderNotFoundError, match="reasoning"):
        await p.generate(prompt="hi", model=MODEL_ID, reasoning=True)


def test_generate_sync_reasoning_true_raises_provider_not_found_error(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})
    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)
    with pytest.raises(ProviderNotFoundError, match="reasoning"):
        p.generate_sync(prompt="hi", model=MODEL_ID, reasoning=True)


@pytest.mark.asyncio
async def test_generate_success(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})
    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)
    p.client.chat = MagicMock(return_value=_fake_response())

    result = await p.generate(prompt="hi", model=MODEL_ID)

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.finish_reason == "COMPLETE"


@pytest.mark.asyncio
async def test_generate_wraps_exceptions_in_provider_error(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})
    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)
    p.client.chat = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        await p.generate(prompt="hi", model=MODEL_ID)


def test_generate_sync_success(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})
    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)
    p.client.chat = MagicMock(return_value=_fake_response())

    result = p.generate_sync(prompt="hi", model=MODEL_ID)

    assert result.content == "hello"
    assert result.usage.total_tokens == 30


def test_generate_sync_wraps_exceptions_in_provider_error(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})
    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)
    p.client.chat = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        p.generate_sync(prompt="hi", model=MODEL_ID)


def test_generate_sync_joins_multiple_content_blocks(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})
    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)
    response = _fake_response()
    response.data.chat_response.choices[0].message.content = [
        SimpleNamespace(text="hello "),
        SimpleNamespace(text="world"),
    ]
    p.client.chat = MagicMock(return_value=response)

    result = p.generate_sync(prompt="hi", model=MODEL_ID)

    assert result.content == "hello world"


def test_from_env_missing_compartment_id_raises(monkeypatch):
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})

    with pytest.raises(ProviderAuthError, match="OCI_COMPARTMENT_ID"):
        OCIProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("OCI_COMPARTMENT_ID", COMPARTMENT_ID)
    monkeypatch.setenv("OCI_REGION", "us-chicago-1")
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})

    p = OCIProvider.from_env()

    assert isinstance(p, OCIProvider)


def test_fetch_latest_models(monkeypatch):
    _patch_oci(monkeypatch, config={"region": "us-chicago-1"})
    p = OCIProvider(compartment_id=COMPARTMENT_ID, region=None)

    fake_models = [
        SimpleNamespace(id="meta.llama-3.3-70b-instruct"),
        SimpleNamespace(id="xai.grok-4-fast-reasoning"),
    ]
    p._control_plane_client.list_models = MagicMock(
        return_value=SimpleNamespace(data=SimpleNamespace(items=fake_models))
    )

    names = p.fetch_latest_models()

    assert names == ["meta.llama-3.3-70b-instruct", "xai.grok-4-fast-reasoning"]
