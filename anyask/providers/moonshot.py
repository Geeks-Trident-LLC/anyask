# anyask/providers/moonshot.py

from __future__ import annotations

import os
from typing import Any, List

from anyask.errors import ProviderAuthError
from anyask.model_catalog import model as MODEL
from anyask.model_listing_mixin import ModelListingMixin

from .openai_compat import OpenAICompatProvider


class MoonshotProvider(OpenAICompatProvider, ModelListingMixin):
    """
    Moonshot AI (Kimi) provider using the OpenAI-compatible API surface.

    Moonshot's API intentionally mirrors the OpenAI API:
    - same request format
    - same response format
    - same chat.completions.create()
    - same models.list()
    Only the base_url and API key differ. Uses the global
    api.moonshot.ai endpoint (not the China-only api.moonshot.cn
    endpoint, which bills in RMB).
    """

    name = "moonshot"
    BASE_URL = "https://api.moonshot.ai/v1"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ProviderAuthError("MOONSHOT_API_KEY is not set")

        super().__init__(
            api_key=api_key,
            endpoint=self.BASE_URL,
            default_model=MODEL.moonshot.default,
        )

    @classmethod
    def from_env(cls) -> "MoonshotProvider":
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ProviderAuthError("MOONSHOT_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.sync_client.models.list().data
        return [m.id for m in models]
