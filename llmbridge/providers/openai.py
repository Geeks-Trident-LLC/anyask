# llmbridge/providers/openai.py

from __future__ import annotations

import os
import re
from typing import Any, List

from openai import AsyncOpenAI, OpenAI

from llmbridge.errors import ProviderAuthError, ProviderError
from llmbridge.model_catalog import model as MODEL
from llmbridge.model_listing_mixin import ModelListingMixin
from llmbridge.provider import AskResponse, Provider, TokenUsage


class OpenAIProvider(Provider, ModelListingMixin):
    name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError("OPENAI_API_KEY is not set")

        self.client = AsyncOpenAI(api_key=api_key)
        self.sync_client = OpenAI(api_key=api_key)
        self.default_model = MODEL.openai.default

    def supports(self, model: str) -> bool:
        return bool(re.search(r"(gpt|o[0-9]+)-", model))

    async def generate(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        try:
            response = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            choice = response.choices[0]
            content = choice.message.content
            usage = getattr(response, "usage", None)

            return AskResponse(
                content=content,
                usage=TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                ),
                finish_reason=getattr(choice, "finish_reason", None),
                provider=self.name,
                model=model or self.default_model,
                raw=response,
            )

        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def generate_sync(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        try:
            response = self.sync_client.chat.completions.create(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            choice = response.choices[0]
            content = choice.message.content
            usage = getattr(response, "usage", None)

            return AskResponse(
                content=content,
                usage=TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                ),
                finish_reason=getattr(choice, "finish_reason", None),
                provider=self.name,
                model=model or self.default_model,
                raw=response,
            )

        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    @classmethod
    def from_env(cls) -> "OpenAIProvider":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError("OPENAI_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        return [m.id for m in self.sync_client.models.list().data]
