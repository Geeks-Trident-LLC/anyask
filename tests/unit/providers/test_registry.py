from unittest.mock import patch

import pytest

import anyask.registry as registry_module
from anyask.providers.anthropic import AnthropicProvider
from anyask.providers.bedrock import BedrockProvider
from anyask.providers.cerebras import CerebrasProvider
from anyask.providers.cohere import CohereProvider
from anyask.providers.fireworks import FireworksProvider
from anyask.providers.gemini import GeminiProvider
from anyask.providers.groq import GroqProvider
from anyask.providers.mistral import MistralProvider
from anyask.providers.moonshot import MoonshotProvider
from anyask.providers.oci import OCIProvider
from anyask.providers.openai import OpenAIProvider
from anyask.providers.openrouter import OpenRouterProvider
from anyask.providers.perplexity import PerplexityProvider
from anyask.providers.together import TogetherProvider
from anyask.providers.vertexai import VertexAIProvider
from anyask.providers.xai import XAIProvider
from anyask.registry import ProviderRegistry, get_provider_by_name, registry


def test_registry_get_returns_registered_class():
    assert registry.get("openai") is OpenAIProvider
    assert registry.get("anthropic") is AnthropicProvider
    assert registry.get("gemini") is GeminiProvider
    assert registry.get("groq") is GroqProvider
    assert registry.get("xai") is XAIProvider
    assert registry.get("together") is TogetherProvider
    assert registry.get("fireworks") is FireworksProvider
    assert registry.get("cerebras") is CerebrasProvider
    assert registry.get("perplexity") is PerplexityProvider
    assert registry.get("openrouter") is OpenRouterProvider
    assert registry.get("moonshot") is MoonshotProvider
    assert registry.get("mistral") is MistralProvider
    assert registry.get("bedrock") is BedrockProvider
    assert registry.get("cohere") is CohereProvider
    assert registry.get("vertexai") is VertexAIProvider
    assert registry.get("oci") is OCIProvider


def test_registry_get_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        registry.get("not-a-real-provider")


def test_registry_all_returns_copy_not_internal_dict():
    all_providers = registry.all()
    assert "openai" in all_providers

    all_providers["fake"] = object()
    assert "fake" not in registry.all()


def test_registry_register_adds_new_entry():
    class _FakeProvider:
        name = "fake-provider"

    r = ProviderRegistry()
    r.register(_FakeProvider)
    assert r.get("fake-provider") is _FakeProvider


def test_get_provider_by_name_success():
    assert get_provider_by_name("openai") is OpenAIProvider


def test_get_provider_by_name_lowercases_input():
    assert get_provider_by_name("OpenAI") is OpenAIProvider
    assert get_provider_by_name("ANTHROPIC") is AnthropicProvider


def test_get_provider_by_name_unknown_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown provider name"):
        get_provider_by_name("not-a-real-provider")


# ---------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------
def test_get_does_not_import_other_provider_modules():
    """Resolving one provider must not import every other provider's SDK."""
    import sys

    for mod in [
        "anyask.providers.bedrock",
        "anyask.providers.cohere",
        "anyask.providers.oci",
    ]:
        sys.modules.pop(mod, None)

    r = ProviderRegistry()
    r.get("anthropic")

    assert "anyask.providers.bedrock" not in sys.modules
    assert "anyask.providers.cohere" not in sys.modules
    assert "anyask.providers.oci" not in sys.modules


def test_get_caches_resolved_class():
    r = ProviderRegistry()
    first = r.get("anthropic")

    with patch.object(registry_module, "import_module") as mock_import:
        second = r.get("anthropic")

    assert second is first
    mock_import.assert_not_called()


def test_get_openai_compat_resolves():
    from anyask.providers.openai_compat import OpenAICompatProvider

    r = ProviderRegistry()
    assert r.get("openai_compat") is OpenAICompatProvider


def test_get_missing_dependency_raises_importerror_with_extra_hint():
    r = ProviderRegistry()

    with patch.object(
        registry_module,
        "import_module",
        side_effect=ImportError("No module named 'boto3'"),
    ):
        with pytest.raises(ImportError, match=r"pip install anyask\[bedrock\]"):
            r.get("bedrock")


def test_all_does_not_import_unloaded_providers():
    r = ProviderRegistry()

    with patch.object(registry_module, "import_module") as mock_import:
        all_providers = r.all()

    mock_import.assert_not_called()
    assert all_providers["bedrock"] is None
    assert "anthropic" in all_providers


def test_all_reflects_already_loaded_providers():
    r = ProviderRegistry()
    r.get("anthropic")

    all_providers = r.all()
    assert all_providers["anthropic"] is AnthropicProvider


def test_registry_module_attribute_is_not_shadowed_by_singleton():
    """
    Regression test: anyask/__init__.py must not re-export the `registry`
    singleton (e.g. `from .registry import registry`), since that rebinds
    the `registry` attribute on the `anyask` package to the
    ProviderRegistry instance - shadowing the `registry` *submodule*
    Python would otherwise expose there. `import anyask.registry as x`
    (and any other dotted-attribute resolution, including
    unittest.mock.patch's string-based targets) would then silently
    resolve to the singleton instance instead of the module.
    """
    assert type(registry_module).__name__ == "module"
    assert registry_module.__name__ == "anyask.registry"
