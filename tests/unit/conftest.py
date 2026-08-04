# tests/unit/conftest.py
#
# Several unit tests construct real SDK clients (openai.OpenAI(), anthropic
# .Anthropic(), google.genai.Client(), etc.) - directly in tests/unit/
# providers/ - and only mock the network-call methods afterward, so client
# construction itself never makes a real request. But every one of those
# SDKs - both the httpx-based ones and google-genai, which builds its own
# ssl.SSLContext directly - calls ssl.create_default_context() during
# construction, and that function reparses the ENTIRE CA certificate bundle
# from disk on every call instead of caching it, dominating test runtime
# with zero effect on test behavior - nothing here ever performs a real TLS
# handshake.
#
# Caching ssl.create_default_context() for the test session eliminates that
# redundant parsing.
#
# Must patch the `ssl` module's own attribute (not e.g. httpx's re-exported
# `create_ssl_context` wrapper): every caller here does `import ssl; ssl.
# create_default_context(...)`, an attribute lookup on the shared `ssl`
# module object at call time, so a patch on `ssl` itself is visible
# everywhere without needing to chase down every SDK's import style
# individually.

import ssl
from functools import lru_cache

import pytest


@pytest.fixture(autouse=True, scope="session")
def _cache_ssl_default_context():
    original = ssl.create_default_context
    ssl.create_default_context = lru_cache(maxsize=None)(original)
    yield
    ssl.create_default_context = original
