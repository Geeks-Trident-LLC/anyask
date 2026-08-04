import pytest

from askllm.errors import ProviderAuthError
from askllm.providers.perplexity import PerplexityProvider


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="PERPLEXITY_API_KEY"):
        PerplexityProvider(api_key=None)


def test_init_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "sk-from-env")
    p = PerplexityProvider()
    assert p.name == "perplexity"


def test_init_sets_perplexity_base_url():
    p = PerplexityProvider(api_key="sk-test")
    assert "perplexity.ai" in str(p.sync_client.base_url)


def test_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="PERPLEXITY_API_KEY"):
        PerplexityProvider.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "sk-from-env")
    p = PerplexityProvider.from_env()
    assert isinstance(p, PerplexityProvider)


def test_fetch_latest_models():
    # Perplexity has no models.list() endpoint, so this returns a static list.
    p = PerplexityProvider(api_key="sk-test")

    names = p.fetch_latest_models()

    assert names == [
        "sonar",
        "sonar-pro",
        "sonar-reasoning",
        "sonar-reasoning-pro",
        "sonar-deep-research",
    ]
