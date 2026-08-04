# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
-

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
- Lazy per-provider SDK imports (`llmbridge/registry.py`) — `import llmbridge`
  never requires any provider SDK to be installed; each provider's
  extra is only needed once that provider is actually used
- `AskLLMError`/`ProviderNotFoundError`/`ProviderError`/
  `ProviderAuthError` error hierarchy — every provider call failure is
  raised with `from exc`, preserving the original SDK exception as
  `__cause__`
- No routing, fallback, or retry logic — `provider` and `model` are
  always explicit, required arguments
