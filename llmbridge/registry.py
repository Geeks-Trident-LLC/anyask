# llmbridge/registry.py

from __future__ import annotations

from importlib import import_module
from typing import Dict, NamedTuple, Optional, Type

from llmbridge.provider import Provider


class _LazySpec(NamedTuple):
    """Where to import a provider class from, and which pip extra installs it."""

    module: str
    class_name: str
    extra: str


# name -> (module path, class name, pip extra that installs its SDK)
#
# Each provider module does its own top-level `import <sdk>` (e.g.
# `import boto3`, `from cohere import ...`). Importing that module is
# deferred until a caller actually asks for that provider by name, so
# `import llmbridge` and building this registry never requires every
# provider SDK to be installed - only the one(s) actually used.
_LAZY_PROVIDERS: Dict[str, _LazySpec] = {
    "openai": _LazySpec("llmbridge.providers.openai", "OpenAIProvider", "openai"),
    "openai_compat": _LazySpec(
        "llmbridge.providers.openai_compat", "OpenAICompatProvider", "openai"
    ),
    "azure": _LazySpec("llmbridge.providers.azure", "AzureOpenAIProvider", "azure"),
    "anthropic": _LazySpec(
        "llmbridge.providers.anthropic", "AnthropicProvider", "anthropic"
    ),
    "gemini": _LazySpec("llmbridge.providers.gemini", "GeminiProvider", "gemini"),
    "deepseek": _LazySpec("llmbridge.providers.deepseek", "DeepSeekProvider", "deepseek"),
    "groq": _LazySpec("llmbridge.providers.groq", "GroqProvider", "groq"),
    "xai": _LazySpec("llmbridge.providers.xai", "XAIProvider", "xai"),
    "together": _LazySpec("llmbridge.providers.together", "TogetherProvider", "together"),
    "fireworks": _LazySpec(
        "llmbridge.providers.fireworks", "FireworksProvider", "fireworks"
    ),
    "cerebras": _LazySpec("llmbridge.providers.cerebras", "CerebrasProvider", "cerebras"),
    "perplexity": _LazySpec(
        "llmbridge.providers.perplexity", "PerplexityProvider", "perplexity"
    ),
    "openrouter": _LazySpec(
        "llmbridge.providers.openrouter", "OpenRouterProvider", "openrouter"
    ),
    "moonshot": _LazySpec("llmbridge.providers.moonshot", "MoonshotProvider", "moonshot"),
    "mistral": _LazySpec("llmbridge.providers.mistral", "MistralProvider", "mistral"),
    "bedrock": _LazySpec("llmbridge.providers.bedrock", "BedrockProvider", "bedrock"),
    "cohere": _LazySpec("llmbridge.providers.cohere", "CohereProvider", "cohere"),
    "vertexai": _LazySpec("llmbridge.providers.vertexai", "VertexAIProvider", "vertexai"),
    "oci": _LazySpec("llmbridge.providers.oci", "OCIProvider", "oci"),
}


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: Dict[str, Type[Provider]] = {}
        self._lazy: Dict[str, _LazySpec] = dict(_LAZY_PROVIDERS)

    def register(self, provider_cls: Type[Provider]) -> None:
        self._providers[provider_cls.name] = provider_cls

    def get(self, name: str) -> Type[Provider]:
        if name in self._providers:
            return self._providers[name]

        spec = self._lazy.get(name)
        if spec is None:
            raise KeyError(name)

        try:
            module = import_module(spec.module)
        except ImportError as ex:
            raise ImportError(
                f"Provider {name!r} requires additional dependencies that are "
                f"not installed. Install with: pip install llmbridge[{spec.extra}]"
            ) from ex

        provider_cls = getattr(module, spec.class_name)
        self._providers[name] = provider_cls
        return provider_cls

    def all(self) -> Dict[str, Optional[Type[Provider]]]:
        combined: Dict[str, Optional[Type[Provider]]] = {
            name: None for name in self._lazy
        }
        combined.update(self._providers)
        return combined


registry = ProviderRegistry()


def get_provider_by_name(provider_name: str) -> Type[Provider]:
    provider_name = provider_name.lower()
    try:
        return registry.get(provider_name)
    except KeyError:
        raise ValueError(f"Unknown provider name: {provider_name}")
