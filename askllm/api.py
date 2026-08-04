# askllm/api.py

from __future__ import annotations

import asyncio
from typing import Any, List

from askllm.errors import ProviderNotFoundError
from askllm.model_listing_mixin import ModelListingMixin
from askllm.provider import AskResponse, Provider
from askllm.registry import registry

# Construction keys: split out of ask()/ask_async()/list_models()'s **kwargs
# and routed to the provider's constructor instead of generate()/
# generate_sync(). Everything else (temperature, max_tokens, ...) is
# forwarded to the generation call unchanged.
_CONSTRUCTION_KEYS = {
    "api_key",
    "endpoint",
    "api_version",
    "deployment",
    "region",
    "project",
    "compartment_id",
}


def _resolve_provider_cls(provider: str):
    try:
        return registry.get(provider)
    except KeyError:
        raise ProviderNotFoundError(f"Unknown provider: {provider!r}") from None
    except ImportError as exc:
        # registry.get() raises ImportError (with a pip-install hint) when
        # the provider is known but its SDK extra isn't installed - that's
        # a distinct failure mode from "unknown provider name", so it is
        # deliberately not caught/rewrapped here.
        raise exc


def _split_kwargs(kwargs: dict) -> tuple:
    config = {k: v for k, v in kwargs.items() if k in _CONSTRUCTION_KEYS}
    call_kwargs = {k: v for k, v in kwargs.items() if k not in _CONSTRUCTION_KEYS}
    return config, call_kwargs


def ask(prompt: str, *, provider: str, model: str, **kwargs: Any) -> AskResponse:
    """Call `provider`'s `model` synchronously and return its response.

    `**kwargs` is split: construction keys (api_key, endpoint, api_version,
    deployment, region, project, compartment_id) go to the provider's
    constructor; everything else (temperature, max_tokens, ...) is
    forwarded to `generate_sync()` unchanged.
    """
    provider_cls = _resolve_provider_cls(provider)
    config, call_kwargs = _split_kwargs(kwargs)
    return provider_cls(**config).generate_sync(prompt, model=model, **call_kwargs)


async def ask_async(
    prompt: str, *, provider: str, model: str, **kwargs: Any
) -> AskResponse:
    """Async counterpart of `ask()` - calls `generate()` instead of
    `generate_sync()`."""
    provider_cls = _resolve_provider_cls(provider)
    config, call_kwargs = _split_kwargs(kwargs)
    return await provider_cls(**config).generate(prompt, model=model, **call_kwargs)


def list_models(provider: str, **kwargs: Any) -> List[str]:
    """Return the live list of model IDs `provider` currently serves.

    Raises `ProviderNotFoundError` if the resolved provider class doesn't
    implement `ModelListingMixin` (e.g. it has no live listing endpoint).
    """
    provider_cls = _resolve_provider_cls(provider)
    if not issubclass(provider_cls, ModelListingMixin):
        raise ProviderNotFoundError(
            f"Provider {provider!r} does not support model listing"
        )

    config, _ = _split_kwargs(kwargs)
    return provider_cls(**config).fetch_latest_models()


async def list_models_async(provider: str, **kwargs: Any) -> List[str]:
    """Async counterpart of `list_models()`.

    No provider SDK exposes a genuinely async model-listing call today,
    so this just runs `list_models()` in a worker thread.
    """
    return await asyncio.to_thread(list_models, provider, **kwargs)


def get_provider(provider: str, **config: Any) -> Provider:
    """Construct and return a live, reusable `Provider` instance.

    This is not routing/fallback - it's the same object `ask()` builds
    internally, just handed back instead of discarded, so callers who
    want to construct a provider once (avoiding per-call SDK client
    reconstruction, e.g. re-reading ~/.oci/config or a fresh
    boto3.client()) and call `generate_sync()`/`generate()` many times
    can do so.
    """
    provider_cls = _resolve_provider_cls(provider)
    return provider_cls(**config)
