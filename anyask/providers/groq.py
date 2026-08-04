# anyask/providers/groq.py

from __future__ import annotations

import os
from typing import Any, List

from anyask.errors import ProviderAuthError
from anyask.model_catalog import model as MODEL
from anyask.model_listing_mixin import ModelListingMixin

from .openai_compat import OpenAICompatProvider


class GroqProvider(OpenAICompatProvider, ModelListingMixin):
    """
    Groq provider using the OpenAI-compatible API surface.

    Groq hosts open models (Llama, Gemma, Qwen, Mixtral, DeepSeek-distill)
    behind an OpenAI-compatible chat.completions endpoint:
    - same request format
    - same response format
    - same chat.completions.create()
    - same models.list()
    Only the base_url and API key differ.
    """

    name = "groq"
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ProviderAuthError("GROQ_API_KEY is not set")

        super().__init__(
            api_key=api_key,
            endpoint=self.BASE_URL,
            default_model=MODEL.groq.default,
        )

    @classmethod
    def from_env(cls) -> "GroqProvider":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ProviderAuthError("GROQ_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.sync_client.models.list().data
        return [m.id for m in models]
