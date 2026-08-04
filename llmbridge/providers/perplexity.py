# llmbridge/providers/perplexity.py

from __future__ import annotations

import os
from typing import Any, List

from llmbridge.errors import ProviderAuthError
from llmbridge.model_catalog import model as MODEL
from llmbridge.model_listing_mixin import ModelListingMixin

from .openai_compat import OpenAICompatProvider


class PerplexityProvider(OpenAICompatProvider, ModelListingMixin):
    """
    Perplexity provider using the OpenAI-compatible API surface.

    Perplexity's API intentionally mirrors the OpenAI API:
    - same request format
    - same response format
    - same chat.completions.create()
    Only the base_url and API key differ. Perplexity does not expose a
    models.list() endpoint, so fetch_latest_models() returns the known
    "sonar" family statically instead of querying the API.
    """

    name = "perplexity"
    BASE_URL = "https://api.perplexity.ai"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ProviderAuthError("PERPLEXITY_API_KEY is not set")

        super().__init__(
            api_key=api_key,
            endpoint=self.BASE_URL,
            default_model=MODEL.perplexity.default,
        )

    @classmethod
    def from_env(cls) -> "PerplexityProvider":
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ProviderAuthError("PERPLEXITY_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        return [
            "sonar",
            "sonar-pro",
            "sonar-reasoning",
            "sonar-reasoning-pro",
            "sonar-deep-research",
        ]
