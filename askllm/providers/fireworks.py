# askllm/providers/fireworks.py

from __future__ import annotations

import os
from typing import Any, List

from askllm.errors import ProviderAuthError
from askllm.model_catalog import model as MODEL
from askllm.model_listing_mixin import ModelListingMixin

from .openai_compat import OpenAICompatProvider


class FireworksProvider(OpenAICompatProvider, ModelListingMixin):
    """
    Fireworks AI provider using the OpenAI-compatible API surface.

    Fireworks' API intentionally mirrors the OpenAI API:
    - same request format
    - same response format
    - same chat.completions.create()
    - same models.list()
    Only the base_url and API key differ.
    """

    name = "fireworks"
    BASE_URL = "https://api.fireworks.ai/inference/v1"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ProviderAuthError("FIREWORKS_API_KEY is not set")

        super().__init__(
            api_key=api_key,
            endpoint=self.BASE_URL,
            default_model=MODEL.fireworks.default,
        )

    @classmethod
    def from_env(cls) -> "FireworksProvider":
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ProviderAuthError("FIREWORKS_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.sync_client.models.list().data
        return [m.id for m in models]
