# anyask/providers/deepseek.py

from __future__ import annotations

import os
from typing import Any, List

from anyask.errors import ProviderAuthError
from anyask.model_catalog import model as MODEL
from anyask.model_listing_mixin import ModelListingMixin

from .openai_compat import OpenAICompatProvider


class DeepSeekProvider(OpenAICompatProvider, ModelListingMixin):
    """
    DeepSeek provider using the OpenAI-compatible API surface.

    DeepSeek intentionally mirrors the OpenAI API:
    - same request format
    - same response format
    - same chat.completions.create()
    - same models.list()
    Only the base_url and API key differ.
    """

    name = "deepseek"
    BASE_URL = "https://api.deepseek.com"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderAuthError("DEEPSEEK_API_KEY is not set")

        super().__init__(
            api_key=api_key,
            endpoint=self.BASE_URL,
            default_model=MODEL.deepseek.default,
        )

    @classmethod
    def from_env(cls) -> "DeepSeekProvider":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderAuthError("DEEPSEEK_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.sync_client.models.list().data
        return [m.id for m in models]
