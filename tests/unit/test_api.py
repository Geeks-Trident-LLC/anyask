from typing import Any, List

import pytest

from anyask import api as api_module
from anyask.errors import ProviderNotFoundError
from anyask.model_listing_mixin import ModelListingMixin
from anyask.provider import AskResponse, Provider, TokenUsage


class _FakeProvider(Provider):
    """A dummy provider that records how it was constructed and called,
    so tests can assert on the api.py construction/call-kwargs split
    without touching any real SDK."""

    name = "fake"
    instances: List["_FakeProvider"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.generate_calls: List[dict] = []
        self.generate_sync_calls: List[dict] = []
        _FakeProvider.instances.append(self)

    def supports(self, model: str) -> bool:
        return True

    async def generate(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        self.generate_calls.append({"prompt": prompt, "model": model, **kwargs})
        return AskResponse(
            content="async-content",
            usage=TokenUsage(1, 2, 3),
            finish_reason="stop",
            provider=self.name,
            model=model,
            raw=None,
        )

    def generate_sync(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        self.generate_sync_calls.append({"prompt": prompt, "model": model, **kwargs})
        return AskResponse(
            content="sync-content",
            usage=TokenUsage(4, 5, 9),
            finish_reason="stop",
            provider=self.name,
            model=model,
            raw=None,
        )


class _FakeListingProvider(_FakeProvider, ModelListingMixin):
    name = "fake-listing"

    def fetch_latest_models(self) -> List[str]:
        return ["model-a", "model-b"]


class _FakeNoListingProvider(_FakeProvider):
    name = "fake-no-listing"


@pytest.fixture(autouse=True)
def _reset_fake_provider_instances():
    _FakeProvider.instances = []
    yield
    _FakeProvider.instances = []


def _patch_registry(monkeypatch, provider_cls):
    monkeypatch.setattr(
        api_module.registry, "get", lambda name: provider_cls, raising=False
    )


# ------------------------------------------------------------
# ask() / ask_async(): construction vs call kwargs split
# ------------------------------------------------------------


def test_ask_splits_construction_and_call_kwargs(monkeypatch):
    _patch_registry(monkeypatch, _FakeProvider)

    result = api_module.ask(
        "hi",
        provider="fake",
        model="model-x",
        api_key="secret",
        region="us-east-1",
        temperature=0.7,
        max_tokens=100,
    )

    assert result.content == "sync-content"
    instance = _FakeProvider.instances[0]
    assert instance.init_kwargs == {"api_key": "secret", "region": "us-east-1"}
    assert instance.generate_sync_calls == [
        {
            "prompt": "hi",
            "model": "model-x",
            "reasoning": False,
            "temperature": 0.7,
            "max_tokens": 100,
        }
    ]


@pytest.mark.asyncio
async def test_ask_async_splits_construction_and_call_kwargs(monkeypatch):
    _patch_registry(monkeypatch, _FakeProvider)

    result = await api_module.ask_async(
        "hi",
        provider="fake",
        model="model-x",
        api_key="secret",
        compartment_id="ocid1.compartment",
        temperature=0.3,
    )

    assert result.content == "async-content"
    instance = _FakeProvider.instances[0]
    assert instance.init_kwargs == {
        "api_key": "secret",
        "compartment_id": "ocid1.compartment",
    }
    assert instance.generate_calls == [
        {"prompt": "hi", "model": "model-x", "reasoning": False, "temperature": 0.3}
    ]


def test_ask_unknown_provider_raises_provider_not_found_error():
    with pytest.raises(ProviderNotFoundError):
        api_module.ask("hi", provider="not-a-real-provider", model="x")


def test_ask_reasoning_true_is_forwarded_to_generate_sync(monkeypatch):
    _patch_registry(monkeypatch, _FakeProvider)

    api_module.ask("hi", provider="fake", model="model-x", reasoning=True)

    instance = _FakeProvider.instances[0]
    assert instance.generate_sync_calls == [
        {"prompt": "hi", "model": "model-x", "reasoning": True}
    ]


@pytest.mark.asyncio
async def test_ask_async_reasoning_true_is_forwarded_to_generate(monkeypatch):
    _patch_registry(monkeypatch, _FakeProvider)

    await api_module.ask_async("hi", provider="fake", model="model-x", reasoning=True)

    instance = _FakeProvider.instances[0]
    assert instance.generate_calls == [
        {"prompt": "hi", "model": "model-x", "reasoning": True}
    ]


# ------------------------------------------------------------
# list_models() / list_models_async()
# ------------------------------------------------------------


def test_list_models_raises_provider_not_found_for_non_listing_provider(monkeypatch):
    _patch_registry(monkeypatch, _FakeNoListingProvider)

    with pytest.raises(ProviderNotFoundError):
        api_module.list_models("fake-no-listing")


def test_list_models_returns_models_for_listing_provider(monkeypatch):
    _patch_registry(monkeypatch, _FakeListingProvider)

    names = api_module.list_models("fake-listing", api_key="secret")

    assert names == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_list_models_async_delegates_to_list_models(monkeypatch):
    _patch_registry(monkeypatch, _FakeListingProvider)

    names = await api_module.list_models_async("fake-listing", api_key="secret")

    assert names == ["model-a", "model-b"]


# ------------------------------------------------------------
# get_provider(): returns a live, reusable instance
# ------------------------------------------------------------


def test_get_provider_returns_provider_instance(monkeypatch):
    _patch_registry(monkeypatch, _FakeProvider)

    provider = api_module.get_provider("fake", api_key="secret")

    assert isinstance(provider, _FakeProvider)
    assert provider.init_kwargs == {"api_key": "secret"}


def test_get_provider_instance_is_reusable_across_calls(monkeypatch):
    _patch_registry(monkeypatch, _FakeProvider)

    provider = api_module.get_provider("fake", api_key="secret")

    provider.generate_sync("first", model="model-x")
    provider.generate_sync("second", model="model-x")

    # Constructed exactly once - the registry was not asked to rebuild the
    # provider between calls.
    assert len(_FakeProvider.instances) == 1
    assert len(provider.generate_sync_calls) == 2


def test_get_provider_unknown_provider_raises_provider_not_found_error():
    with pytest.raises(ProviderNotFoundError):
        api_module.get_provider("not-a-real-provider")
