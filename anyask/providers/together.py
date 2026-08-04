# anyask/providers/together.py

from __future__ import annotations

import os
from typing import Any, List

from anyask.errors import ProviderAuthError
from anyask.model_catalog import model as MODEL
from anyask.model_listing_mixin import ModelListingMixin

from .openai_compat import OpenAICompatProvider


class TogetherProvider(OpenAICompatProvider, ModelListingMixin):
    """
    Together AI provider using the OpenAI-compatible API surface.

    Together's API intentionally mirrors the OpenAI API:
    - same request format
    - same response format
    - same chat.completions.create()
    - same models.list()
    Only the base_url and API key differ.
    """

    name = "together"
    BASE_URL = "https://api.together.xyz/v1"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ProviderAuthError("TOGETHER_API_KEY is not set")

        super().__init__(
            api_key=api_key,
            endpoint=self.BASE_URL,
            default_model=MODEL.together.default,
        )

    @classmethod
    def from_env(cls) -> "TogetherProvider":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ProviderAuthError("TOGETHER_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.sync_client.models.list().data
        return [m.id for m in models]
