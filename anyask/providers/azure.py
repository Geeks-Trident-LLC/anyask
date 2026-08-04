# anyask/providers/azure.py

from __future__ import annotations

import asyncio
import os
from typing import Any, List

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from anyask.errors import ProviderAuthError, ProviderError, ProviderNotFoundError
from anyask.model_listing_mixin import ModelListingMixin
from anyask.provider import AskResponse, Provider, TokenUsage


def build_azure_endpoint(endpoint: str, deployment: str) -> str:
    """
    Normalize Azure endpoint for azure.ai.inference ChatCompletionsClient.

    If the user passes a full deployment endpoint, return it unchanged.
    If the user passes a base endpoint, append the deployment path.
    """
    endpoint = endpoint.rstrip("/")

    # Case 1: Already a full deployment endpoint
    # Example: https://.../openai/deployments/gpt-4.1-anyask
    if "/openai/deployments/" in endpoint:
        return endpoint

    # Case 2: Base endpoint -> build full deployment endpoint
    return f"{endpoint}/openai/deployments/{deployment}"


class AzureOpenAIProvider(Provider, ModelListingMixin):
    name = "azure"

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.get("api_key") or os.getenv("AZURE_API_KEY")
        endpoint = kwargs.get("endpoint") or os.getenv("AZURE_ENDPOINT")
        api_version = kwargs.get("api_version") or os.getenv(
            "AZURE_API_VERSION", "2024-02-15-preview"
        )
        deployment = kwargs.get("deployment") or os.getenv("AZURE_DEPLOYMENT")

        if not api_key:
            raise ProviderAuthError("AZURE_API_KEY is not set")
        if not endpoint:
            raise ProviderAuthError("AZURE_ENDPOINT is not set")
        if not deployment:
            raise ProviderAuthError("AZURE_DEPLOYMENT is not set")

        full_endpoint = build_azure_endpoint(endpoint, deployment)
        self.client = ChatCompletionsClient(
            endpoint=full_endpoint,
            credential=AzureKeyCredential(api_key),
            api_version=api_version,
        )
        self.api_key = api_key
        self.endpoint = full_endpoint
        self.api_version = api_version
        self.deployment = deployment

    def supports(self, model: str) -> bool:
        return True

    async def generate(
        self, prompt: str, *, model: str = "", reasoning: bool = False, **kwargs: Any
    ) -> AskResponse:
        if reasoning:
            raise ProviderNotFoundError(
                f"{self.name} does not support the reasoning parameter - "
                "azure.ai.inference is a generic gateway with no reasoning "
                "field consistent across the model families it can front"
            )
        try:
            # Azure SDK is synchronous -> run in thread
            result = await asyncio.to_thread(
                self.client.complete,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            choice = result.choices[0]
            content = choice.message.content
            usage = getattr(result, "usage", None)

            return AskResponse(
                content=content,
                usage=TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                ),
                finish_reason=getattr(choice, "finish_reason", None),
                provider=self.name,
                model=model or self.deployment,
                raw=result,
            )

        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def generate_sync(
        self, prompt: str, *, model: str = "", reasoning: bool = False, **kwargs: Any
    ) -> AskResponse:
        if reasoning:
            raise ProviderNotFoundError(
                f"{self.name} does not support the reasoning parameter - "
                "azure.ai.inference is a generic gateway with no reasoning "
                "field consistent across the model families it can front"
            )
        try:
            result = self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            choice = result.choices[0]
            content = choice.message.content
            usage = getattr(result, "usage", None)

            return AskResponse(
                content=content,
                usage=TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                ),
                finish_reason=getattr(choice, "finish_reason", None),
                provider=self.name,
                model=model or self.deployment,
                raw=result,
            )

        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    @classmethod
    def from_env(cls) -> "AzureOpenAIProvider":
        return cls()

    def fetch_latest_models(self) -> List[str]:
        return [self.deployment]
