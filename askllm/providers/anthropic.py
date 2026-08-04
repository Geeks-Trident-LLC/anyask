# askllm/providers/anthropic.py

from __future__ import annotations

import os
from typing import Any, List

from anthropic import Anthropic, AsyncAnthropic

from askllm.errors import ProviderAuthError, ProviderError
from askllm.model_catalog import model as MODEL
from askllm.model_listing_mixin import ModelListingMixin
from askllm.provider import AskResponse, Provider, TokenUsage


class AnthropicProvider(Provider, ModelListingMixin):
    name = "anthropic"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY is not set")

        self.client = AsyncAnthropic(api_key=api_key)
        self.sync_client = Anthropic(api_key=api_key)
        self.default_model = MODEL.anthropic.default

    def supports(self, model: str) -> bool:
        return True

    async def generate(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        kwargs.setdefault("temperature", 0.2)
        kwargs.setdefault("max_tokens", 2048)

        try:
            response = await self.client.messages.create(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            # Anthropic returns a list of content blocks
            content = response.content[0].text
            usage = getattr(response, "usage", None)

            return AskResponse(
                content=content,
                usage=TokenUsage(
                    prompt_tokens=getattr(usage, "input_tokens", None),
                    completion_tokens=getattr(usage, "output_tokens", None),
                    total_tokens=(
                        (usage.input_tokens or 0) + (usage.output_tokens or 0)
                        if usage
                        else None
                    ),
                ),
                finish_reason=getattr(response, "stop_reason", None),
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
            response = self.sync_client.messages.create(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            content = response.content[0].text
            usage = getattr(response, "usage", None)

            return AskResponse(
                content=content,
                usage=TokenUsage(
                    prompt_tokens=getattr(usage, "input_tokens", None),
                    completion_tokens=getattr(usage, "output_tokens", None),
                    total_tokens=(
                        (usage.input_tokens or 0) + (usage.output_tokens or 0)
                        if usage
                        else None
                    ),
                ),
                finish_reason=getattr(response, "stop_reason", None),
                provider=self.name,
                model=model or self.default_model,
                raw=response,
            )

        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    @classmethod
    def from_env(cls) -> "AnthropicProvider":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.sync_client.models.list().data
        return [m.id for m in models]
