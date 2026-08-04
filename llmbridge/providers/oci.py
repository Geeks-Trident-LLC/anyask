# llmbridge/providers/oci.py

from __future__ import annotations

import asyncio
import os
from typing import Any, List

import oci

from llmbridge.errors import ProviderAuthError, ProviderError
from llmbridge.model_catalog import model as MODEL
from llmbridge.model_listing_mixin import ModelListingMixin
from llmbridge.provider import AskResponse, Provider, TokenUsage


class OCIProvider(Provider, ModelListingMixin):
    """
    Oracle Cloud Infrastructure (OCI) Generative AI provider (Shape B).

    Architecturally the most complex provider integrated so far, on three
    separate axes:

    1. Auth is neither a project-level API key (Mistral/Cohere) nor a
       simple ambient-credential-chain lookup (Bedrock/Vertex AI) - OCI
       uses per-request cryptographic signing. This provider supports
       only the most common local-dev/CI mode: a config file at
       `~/.oci/config` (DEFAULT profile), read via `oci.config.
       from_file()` - the OCI equivalent of AWS's `~/.aws/credentials`.
       Unlike boto3/google-genai's lazy credential resolution,
       `from_file()` validates eagerly at construction time, so a
       missing/malformed config fails fast with a clear error rather than
       on the first real API call. Instance/resource principal auth
       (for workloads running inside OCI itself) is NOT supported by this
       first pass - a deliberate scope decision, not an oversight.

    2. OCI's chat API genuinely splits into two incompatible request/
       response shapes depending on model vendor: GenericChatRequest
       (OpenAI-shaped messages=[...]/choices[] - used for Meta Llama and
       xAI Grok models) vs CohereChatRequest (message=/chat_history=,
       text response - used for Cohere models). This provider
       deliberately supports ONLY the Generic format - Cohere models are
       reachable through llmbridge's own `cohere` provider and via
       `bedrock` instead, so this is a scope choice, not a gap.

    3. Needs a THIRD app-specific field beyond region: `compartment_id`
       (an OCID identifying which OCI compartment/tenancy to bill and
       scope the request to) - on top of `region`, which is reused here
       the same way it's reused for Vertex AI's "location" concept.
    """

    name = "oci"

    def __init__(self, **kwargs: Any) -> None:
        compartment_id = kwargs.get("compartment_id") or os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            raise ProviderAuthError(
                "OCI compartment ID is not set (pass compartment_id= or set "
                "OCI_COMPARTMENT_ID)"
            )

        try:
            config = oci.config.from_file()
        except Exception as exc:
            raise ProviderAuthError(
                "OCI config file not found or invalid (expected ~/.oci/config "
                f"with a DEFAULT profile): {exc}"
            ) from exc

        region = kwargs.get("region") or os.getenv("OCI_REGION")
        if region:
            config["region"] = region
        elif not config.get("region"):
            raise ProviderAuthError(
                "OCI region is not set (pass region=, set OCI_REGION, or set "
                "it in ~/.oci/config)"
            )

        self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(config)
        self._control_plane_client = oci.generative_ai.GenerativeAiClient(config)
        self.compartment_id = compartment_id
        self.region = config["region"]
        self.default_model = MODEL.oci.default

    def supports(self, model: str) -> bool:
        return True

    async def generate(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        try:
            return await asyncio.to_thread(self._chat, prompt, model, **kwargs)
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def generate_sync(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        try:
            return self._chat(prompt, model, **kwargs)
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def _chat(self, prompt: str, model: str, **kwargs: Any) -> AskResponse:
        max_tokens = kwargs.pop("max_tokens", 2048)
        temperature = kwargs.pop("temperature", 0.2)

        chat_details = oci.generative_ai_inference.models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                model_id=model or self.default_model,
            ),
            chat_request=oci.generative_ai_inference.models.GenericChatRequest(
                api_format=(
                    oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
                ),
                messages=[
                    oci.generative_ai_inference.models.UserMessage(
                        content=[
                            oci.generative_ai_inference.models.TextContent(text=prompt)
                        ],
                    ),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            ),
        )

        response = self.client.chat(chat_details)
        chat_response = response.data.chat_response
        choice = chat_response.choices[0]
        content = "".join(
            getattr(block, "text", "") for block in choice.message.content
        )
        usage = getattr(chat_response, "usage", None)

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
            raw=response.data,
        )

    @classmethod
    def from_env(cls) -> "OCIProvider":
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            raise ProviderAuthError("OCI_COMPARTMENT_ID is not set")

        region = os.getenv("OCI_REGION")
        return cls(compartment_id=compartment_id, region=region)

    def fetch_latest_models(self) -> List[str]:
        models = self._control_plane_client.list_models(
            compartment_id=self.compartment_id,
        ).data.items
        return [m.id for m in models]
