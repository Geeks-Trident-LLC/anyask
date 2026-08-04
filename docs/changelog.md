# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `anyask` console script: `anyask --version`/`anyask version`, `anyask check
  <provider>` (construct one provider without a network call, reporting whether
  its SDK is installed and credentials resolve; exits 0/1), and `anyask ready`
  (sweep every provider using only environment-variable credentials; always
  exits 0 - a status report, not a gate). Built on stdlib `argparse`, not
  `click`, to keep the bare install at 2 packages.
- Consistent `temperature=0.2`/`max_tokens=2048` defaults across all 17
  providers (previously only Anthropic, Mistral, Cohere, Bedrock, and
  OCI set these; OpenAI, the 9 OpenAI-compatible vendors, Gemini, Vertex
  AI, and Azure passed calls through with no default, meaning identical
  `ask()` calls could get very different sampling behavior purely
  depending on provider). Skipped for OpenAI/OpenAI-compatible vendors
  when `reasoning=True`, since reasoning models on some of them reject a
  temperature override and use `max_completion_tokens` instead of
  `max_tokens`.
- `reasoning: bool = False` on `ask()`/`ask_async()` — enables extended/
  deliberate reasoning using each provider's own real mechanism
  (Anthropic extended thinking, OpenAI/OpenAI-compatible
  `reasoning_effort`, Gemini/Vertex AI thinking budgets, Bedrock's Claude
  thinking field). Providers with no such mechanism (Azure, Mistral,
  Cohere, OCI) raise `ProviderNotFoundError` rather than silently
  ignoring it. See `docs/providers/index.md`'s Reasoning support table.

### Changed
-

### Fixed
-

### Removed
-

## [0.1.0] - 2026-08-04

### Added
- Initial extraction release: `ask()`, `ask_async()`, `list_models()`,
  `list_models_async()`, and `get_provider()` — one normalized
  `AskResponse` shape across 17 providers (OpenAI, Anthropic, Gemini,
  Vertex AI, Azure, Mistral, Bedrock, Cohere, OCI, and nine
  OpenAI-compatible vendors: DeepSeek, Groq, xAI, Together AI, Fireworks
  AI, Cerebras, Perplexity, OpenRouter, Moonshot)
- Lazy per-provider SDK imports (`anyask/registry.py`) — `import anyask`
  never requires any provider SDK to be installed; each provider's
  extra is only needed once that provider is actually used
- `AskLLMError`/`ProviderNotFoundError`/`ProviderError`/
  `ProviderAuthError` error hierarchy — every provider call failure is
  raised with `from exc`, preserving the original SDK exception as
  `__cause__`
- No routing, fallback, or retry logic — `provider` and `model` are
  always explicit, required arguments
