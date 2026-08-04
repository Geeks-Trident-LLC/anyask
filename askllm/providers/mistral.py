# askllm/providers/mistral.py

from __future__ import annotations

import os
from typing import Any, List

from mistralai import Mistral

from askllm.errors import ProviderAuthError, ProviderError
from askllm.model_catalog import model as MODEL
from askllm.model_listing_mixin import ModelListingMixin
from askllm.provider import AskResponse, Provider, TokenUsage


class MistralProvider(Provider, ModelListingMixin):
    """
    Mistral AI provider using the native `mistralai` SDK (Shape B).

    Unlike the OpenAI-compatible providers, Mistral's official Python SDK
    is NOT built on the OpenAI client library - it has its own generated
    client shape (client.chat.complete()/complete_async(),
    client.models.list()), even though the underlying REST API's JSON
    response shape (choices[0].message.content, usage.prompt_tokens/
    completion_tokens/total_tokens) closely mirrors OpenAI's
    chat-completions format.
    """

    name = "mistral"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ProviderAuthError("MISTRAL_API_KEY is not set")

        self.client = Mistral(api_key=api_key)
        self.default_model = MODEL.mistral.default

    def supports(self, model: str) -> bool:
        return True

    async def generate(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        kwargs.setdefault("temperature", 0.2)
        kwargs.setdefault("max_tokens", 2048)

        try:
            response = await self.client.chat.complete_async(
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
        kwargs.setdefault("temperature", 0.2)
        kwargs.setdefault("max_tokens", 2048)

        try:
            response = self.client.chat.complete(
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
    def from_env(cls) -> "MistralProvider":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ProviderAuthError("MISTRAL_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.client.models.list().data
        return [m.id for m in models]
