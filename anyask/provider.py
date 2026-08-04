# anyask/provider.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting for a single generation call.

    Any field the underlying provider SDK doesn't report is left as
    `None` rather than coerced to 0, so callers can distinguish
    "not reported" from "reported as zero".
    """

    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass(frozen=True)
class AskResponse:
    """The normalized result of a single `generate`/`generate_sync` call.

    `finish_reason` is intentionally left as the RAW, provider-specific
    value (e.g. "end_turn" for Anthropic, "stop" for OpenAI-family, a
    Gemini `FinishReason` enum member, ...) - it is never normalized or
    collapsed to a bool, so callers can detect truncated completions
    themselves.

    `raw` holds the untouched original SDK response object, for callers
    that need something this dataclass doesn't expose.
    """

    content: str
    usage: TokenUsage
    finish_reason: Optional[str]
    provider: str
    model: str
    raw: Any = field(repr=False)


class Provider(ABC):
    """Base interface for all LLM providers.

    Construction is kwargs-only: every concrete provider's `__init__`
    accepts `**kwargs` and pulls only the keys it needs (e.g. `api_key`,
    `region`), ignoring the rest. This lets callers pass one uniform
    config dict to any provider class without an if/elif dispatch on
    provider name. See `anyask/api.py`'s `_CONSTRUCTION_KEYS` for the
    known keyword vocabulary.

    A provider instance is reusable across multiple `model=` values of
    the same vendor - `model` is always passed to `generate`/
    `generate_sync`, never fixed at construction time.
    """

    name: str

    def __init__(self, **kwargs: Any) -> None: ...

    @abstractmethod
    def supports(self, model: str) -> bool:
        """Return True if this provider can serve the given model identifier."""
        raise NotImplementedError

    @abstractmethod
    async def generate(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        """Async text generation call."""
        raise NotImplementedError

    @abstractmethod
    def generate_sync(self, prompt: str, *, model: str, **kwargs: Any) -> AskResponse:
        """Sync text generation call."""
        raise NotImplementedError

    @classmethod
    def from_env(cls) -> "Provider":
        raise NotImplementedError
