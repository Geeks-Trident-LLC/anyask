# askllm/errors.py

from __future__ import annotations


class AskLLMError(Exception):
    """Base class for all askllm errors."""


class ProviderNotFoundError(AskLLMError):
    """Raised when an unknown provider name is requested, or when the
    requested operation needs a capability (e.g. model listing) the
    resolved provider class doesn't implement."""


class ProviderError(AskLLMError):
    """Raised when a provider call fails. Always raised with `from exc`
    so `err.__cause__` is the original SDK exception."""


class ProviderAuthError(ProviderError):
    """Raised when a provider is missing required credentials/config at
    construction time (before any network call is made)."""
