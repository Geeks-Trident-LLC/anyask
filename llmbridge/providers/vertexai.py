# llmbridge/providers/vertexai.py

from __future__ import annotations

import asyncio
import os
from typing import Any, List

import google.genai as genai

from llmbridge.errors import ProviderAuthError, ProviderError
from llmbridge.model_catalog import model as MODEL
from llmbridge.model_listing_mixin import ModelListingMixin
from llmbridge.provider import AskResponse, Provider, TokenUsage


class VertexAIProvider(Provider, ModelListingMixin):
    """
    Google Vertex AI provider (Shape B).

    Reuses the SAME `google-genai` SDK as native Gemini - `google.genai
    .Client` supports Vertex AI directly via `vertexai=True, project=...,
    location=...`, so no new dependency is needed. Like Bedrock, Vertex AI
    has no project-level API key: when no api_key/credentials are passed,
    the SDK falls back to Google Cloud's Application Default Credentials
    (a service account key file via GOOGLE_APPLICATION_CREDENTIALS,
    `gcloud auth application-default login`, or workload identity on GCP
    infra) - so this provider never handles a secret directly. `project`/
    `region` (the "location" concept in Google's own SDK - reused here as
    `region`, the same generic keyword used by Bedrock/OCI) are resolved
    from the VERTEXAI_PROJECT/VERTEXAI_REGION env vars (not GCP's own
    GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION, since both are always
    passed explicitly to genai.Client() below).

    Vertex AI serves the SAME Gemini model catalog as the native Gemini
    Developer API under IDENTICAL model IDs (e.g. "gemini-2.5-pro") -
    unlike Bedrock/OpenRouter, there is no distinguishing namespace
    prefix, so Vertex AI must always be selected explicitly via
    provider="vertexai".

    generate()/generate_sync() mirror GeminiProvider's implementation
    exactly (same underlying SDK method, same asyncio.to_thread wrapping
    since generate_content() has no genuinely async variant used here) -
    only client construction differs.
    """

    name = "vertexai"

    def __init__(self, **kwargs: Any) -> None:
        project = kwargs.get("project") or os.getenv("VERTEXAI_PROJECT")
        region = kwargs.get("region") or os.getenv("VERTEXAI_REGION")

        if not project:
            raise ProviderAuthError(
                "GCP project is not set (pass project= or set VERTEXAI_PROJECT)"
            )
        if not region:
            raise ProviderAuthError(
                "GCP location is not set (pass region= or set VERTEXAI_REGION)"
            )

        self.client = genai.Client(vertexai=True, project=project, location=region)
        self.project = project
        self.region = region
        self.default_model = MODEL.vertexai.default

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
    def from_env(cls) -> "VertexAIProvider":
        project = os.getenv("VERTEXAI_PROJECT")
        region = os.getenv("VERTEXAI_REGION")

        if not project:
            raise ProviderAuthError("VERTEXAI_PROJECT is not set")
        if not region:
            raise ProviderAuthError("VERTEXAI_REGION is not set")

        return cls(project=project, region=region)

    def fetch_latest_models(self) -> List[str]:
        models = self.client.models.list()
        return [m.name for m in models]
