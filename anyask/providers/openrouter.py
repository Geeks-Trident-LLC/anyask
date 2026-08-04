# anyask/providers/openrouter.py

from __future__ import annotations

import os
from typing import Any, List

from anyask.errors import ProviderAuthError
from anyask.model_catalog import model as MODEL
from anyask.model_listing_mixin import ModelListingMixin

from .openai_compat import OpenAICompatProvider


class OpenRouterProvider(OpenAICompatProvider, ModelListingMixin):
    """
    OpenRouter provider using the OpenAI-compatible API surface.

    OpenRouter is a model aggregator/router: it re-exposes models from
    many upstream providers (OpenAI, Anthropic, Google, Meta, Mistral,
    DeepSeek, xAI, Qwen, etc.) under a single "vendor/model" namespace,
    plus its own "openrouter/auto" meta-model that lets OpenRouter pick
    the best underlying model for a given request.

    - same request format
    - same response format
    - same chat.completions.create()
    - same models.list()
    Only the base_url and API key differ.
    """

    name = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderAuthError("OPENROUTER_API_KEY is not set")

        super().__init__(
            api_key=api_key,
            endpoint=self.BASE_URL,
            default_model=MODEL.openrouter.default,
        )

    @classmethod
    def from_env(cls) -> "OpenRouterProvider":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderAuthError("OPENROUTER_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.sync_client.models.list().data
        return [m.id for m in models]
