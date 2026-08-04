# anyask/providers/openai_compat.py

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI

from anyask.errors import ProviderError
from anyask.provider import AskResponse, Provider, TokenUsage


class OpenAICompatProvider(Provider):
    """
    Base provider for any vendor exposing an OpenAI-compatible
    chat-completions API (same request format, same response format,
    same `chat.completions.create()`/`models.list()`) - only the
    `base_url` and API key differ per vendor. Concrete subclasses
    (DeepSeek, Groq, xAI, ...) hardcode their own base URL and resolve
    their own API key env var, then delegate here.

    Also directly usable/registered as the generic "openai_compat"
    provider for arbitrary OpenAI-compatible endpoints not otherwise
    named - pass `api_key`/`endpoint` explicitly in that case.
    """

    name = "openai_compat"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("endpoint")
        default_model = kwargs.get("default_model", "")

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.sync_client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model

    def supports(self, model: str) -> bool:
        return True

    async def generate(
        self, prompt: str, *, model: str, reasoning: bool = False, **kwargs: Any
    ) -> AskResponse:
        # OpenAI-compatible `reasoning_effort` convention - support varies
        # by vendor/model; an unsupported combination surfaces as a
        # ProviderError from the vendor's own API rejecting the field,
        # rather than being silently ignored here.
        if reasoning:
            kwargs.setdefault("reasoning_effort", "medium")
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

    def generate_sync(
        self, prompt: str, *, model: str, reasoning: bool = False, **kwargs: Any
    ) -> AskResponse:
        if reasoning:
            kwargs.setdefault("reasoning_effort", "medium")
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
