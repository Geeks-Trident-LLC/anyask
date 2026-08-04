# askllm/providers/cohere.py

from __future__ import annotations

import os
from typing import Any, List

from cohere import AsyncClientV2, ClientV2

from askllm.errors import ProviderAuthError, ProviderError
from askllm.model_catalog import model as MODEL
from askllm.model_listing_mixin import ModelListingMixin
from askllm.provider import AskResponse, Provider, TokenUsage


def _parse_cohere_response(
    response: Any, provider_name: str, model: str
) -> AskResponse:
    content_field = response.message.content

    if isinstance(content_field, str):
        content = content_field
    else:
        content = "".join(getattr(block, "text", "") for block in content_field or [])

    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "tokens", None)
    input_tokens = getattr(tokens, "input_tokens", None)
    output_tokens = getattr(tokens, "output_tokens", None)

    # Cohere's usage.tokens has no total_tokens field (unlike every other
    # provider so far) - compute it ourselves when both halves are present.
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )

    return AskResponse(
        content=content,
        usage=TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
        ),
        finish_reason=getattr(response, "finish_reason", None),
        provider=provider_name,
        model=model,
        raw=response,
    )


class CohereProvider(Provider, ModelListingMixin):
    """
    Cohere provider using the native `cohere` SDK's Chat v2 API (Shape B).

    Like Anthropic and Gemini (and unlike Mistral, whose single client
    exposes both complete()/complete_async()), the `cohere` SDK exposes
    genuinely separate `ClientV2`/`AsyncClientV2` classes - so this
    provider holds two client instances, `self.client` (async, used by
    generate()) and `self.sync_client` (sync, used by generate_sync()),
    matching AnthropicProvider's naming.
    """

    name = "cohere"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ProviderAuthError("COHERE_API_KEY is not set")

        self.client = AsyncClientV2(api_key=api_key)
        self.sync_client = ClientV2(api_key=api_key)
        self.default_model = MODEL.cohere.default

    def supports(self, model: str) -> bool:
        return True

    async def generate(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        kwargs.setdefault("temperature", 0.2)
        kwargs.setdefault("max_tokens", 2048)

        try:
            response = await self.client.chat(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return _parse_cohere_response(
                response, self.name, model or self.default_model
            )

        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def generate_sync(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        kwargs.setdefault("temperature", 0.2)
        kwargs.setdefault("max_tokens", 2048)

        try:
            response = self.sync_client.chat(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return _parse_cohere_response(
                response, self.name, model or self.default_model
            )

        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    @classmethod
    def from_env(cls) -> "CohereProvider":
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ProviderAuthError("COHERE_API_KEY is not set")
        return cls(api_key=api_key)

    def fetch_latest_models(self) -> List[str]:
        models = self.sync_client.models.list(endpoint="chat").models
        return [m.name for m in models]
