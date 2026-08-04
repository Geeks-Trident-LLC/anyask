# askllm/providers/gemini.py

from __future__ import annotations

import asyncio
import os
from typing import Any, List

import google.genai as genai

from askllm.errors import ProviderAuthError, ProviderError
from askllm.model_catalog import model as MODEL
from askllm.model_listing_mixin import ModelListingMixin
from askllm.provider import AskResponse, Provider, TokenUsage


class GeminiProvider(Provider, ModelListingMixin):
    name = "gemini"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderAuthError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
        self.default_model = MODEL.gemini.default

    def supports(self, model: str) -> bool:
        return True

    async def generate(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        try:
            thinking_budget = kwargs.pop("thinking_budget", 0)
            config = genai.types.GenerateContentConfig(
                temperature=kwargs.pop("temperature", None),
                max_output_tokens=kwargs.pop("max_tokens", None),
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=thinking_budget
                ),
                **kwargs,
            )

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model or self.default_model,
                contents=prompt,
                config=config,
            )

            usage = getattr(response, "usage_metadata", None)
            candidates = getattr(response, "candidates", None) or []
            finish_reason = (
                getattr(candidates[0], "finish_reason", None) if candidates else None
            )

            return AskResponse(
                content=response.text,
                usage=TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_token_count", None),
                    completion_tokens=getattr(usage, "candidates_token_count", None),
                    total_tokens=getattr(usage, "total_token_count", None),
                ),
                finish_reason=finish_reason,
                provider=self.name,
                model=model or self.default_model,
                raw=response,
            )

        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def generate_sync(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        try:
            thinking_budget = kwargs.pop("thinking_budget", 0)
            config = genai.types.GenerateContentConfig(
                temperature=kwargs.pop("temperature", None),
                max_output_tokens=kwargs.pop("max_tokens", None),
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=thinking_budget
                ),
                **kwargs,
            )

            response = self.client.models.generate_content(
                model=model or self.default_model,
                contents=prompt,
                config=config,
            )

            usage = getattr(response, "usage_metadata", None)
            candidates = getattr(response, "candidates", None) or []
            finish_reason = (
                getattr(candidates[0], "finish_reason", None) if candidates else None
            )

            return AskResponse(
                content=response.text,
                usage=TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_token_count", None),
                    completion_tokens=getattr(usage, "candidates_token_count", None),
                    total_tokens=getattr(usage, "total_token_count", None),
                ),
                finish_reason=finish_reason,
                provider=self.name,
                model=model or self.default_model,
                raw=response,
            )

        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    @classmethod
    def from_env(cls) -> "GeminiProvider":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderAuthError("GEMINI_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.client.models.list()
        return [m.name for m in models]
