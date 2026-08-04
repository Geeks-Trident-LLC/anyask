# anyask/providers/bedrock.py

from __future__ import annotations

import asyncio
import os
from typing import Any, List

import boto3

from anyask.errors import ProviderAuthError, ProviderError
from anyask.model_catalog import model as MODEL
from anyask.model_listing_mixin import ModelListingMixin
from anyask.provider import AskResponse, Provider, TokenUsage


class BedrockProvider(Provider, ModelListingMixin):
    """
    Amazon Bedrock provider using the native `boto3` bedrock-runtime
    Converse API (Shape B).

    Unlike every other provider, Bedrock authenticates via AWS's own
    credential chain (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/
    AWS_SESSION_TOKEN env vars, ~/.aws/credentials, or an IAM role) rather
    than a project-level API key - boto3 resolves credentials on its own,
    so this provider never handles a secret directly. The only
    Bedrock-specific parameter is `region`, since every Bedrock call is
    region-scoped. This resolves from the app-namespaced BEDROCK_REGION/
    BEDROCK_DEFAULT_REGION env vars (not boto3's own AWS_REGION/
    AWS_DEFAULT_REGION, since region_name is always passed explicitly to
    boto3.client() below, boto3 never gets a chance to fall back to its
    own env vars in this code path).

    boto3 has no native async client, so `generate()` wraps the sync
    `converse()` call in `asyncio.to_thread`, same as AzureOpenAIProvider.
    """

    name = "bedrock"

    def __init__(self, **kwargs: Any) -> None:
        region = (
            kwargs.get("region")
            or os.getenv("BEDROCK_REGION")
            or os.getenv("BEDROCK_DEFAULT_REGION")
        )
        if not region:
            raise ProviderAuthError(
                "AWS region is not set (pass region= or set BEDROCK_REGION/"
                "BEDROCK_DEFAULT_REGION)"
            )

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.region = region
        self.default_model = MODEL.bedrock.default

    def supports(self, model: str) -> bool:
        return True

    async def generate(
        self, prompt: str, *, model: str, reasoning: bool = False, **kwargs: Any
    ) -> AskResponse:
        try:
            return await asyncio.to_thread(
                self._converse, prompt, model, reasoning=reasoning, **kwargs
            )
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def generate_sync(
        self, prompt: str, *, model: str, reasoning: bool = False, **kwargs: Any
    ) -> AskResponse:
        try:
            return self._converse(prompt, model, reasoning=reasoning, **kwargs)
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def _converse(
        self, prompt: str, model: str, reasoning: bool = False, **kwargs: Any
    ) -> AskResponse:
        max_tokens = kwargs.pop("max_tokens", 2048)
        inference_config: dict = {"maxTokens": max_tokens}

        if reasoning:
            # Claude-on-Bedrock's thinking field, same shape as native
            # Anthropic - requires no fixed temperature while enabled, so
            # omit it from inferenceConfig entirely (model uses its
            # default) rather than send a value the API will reject. Only
            # applies to Claude models; other Bedrock model families raise
            # from the API itself if this field isn't recognized.
            kwargs.setdefault(
                "additionalModelRequestFields",
                {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": kwargs.pop("thinking_budget", 1024),
                    }
                },
            )
        else:
            inference_config["temperature"] = kwargs.pop("temperature", 0.2)

        response = self.client.converse(
            modelId=model or self.default_model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=inference_config,
            **kwargs,
        )

        content_blocks = response["output"]["message"]["content"]
        content = "".join(block.get("text", "") for block in content_blocks)
        usage = response.get("usage", {})

        return AskResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=usage.get("inputTokens"),
                completion_tokens=usage.get("outputTokens"),
                total_tokens=usage.get("totalTokens"),
            ),
            finish_reason=response.get("stopReason"),
            provider=self.name,
            model=model or self.default_model,
            raw=response,
        )

    @classmethod
    def from_env(cls) -> "BedrockProvider":
        region = os.getenv("BEDROCK_REGION") or os.getenv("BEDROCK_DEFAULT_REGION")
        if not region:
            raise ProviderAuthError(
                "BEDROCK_REGION or BEDROCK_DEFAULT_REGION is not set"
            )
        return cls(region=region)

    def fetch_latest_models(self) -> List[str]:
        control_client = boto3.client("bedrock", region_name=self.region)
        summaries = control_client.list_foundation_models().get("modelSummaries", [])
        return [m["modelId"] for m in summaries]
