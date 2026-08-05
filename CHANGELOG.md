# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.2] - 2026-08-05

### Changed
- `anyask --version`/`anyask version` now print `anyask v<version>` (e.g.
  `anyask v0.2.2`) instead of `anyask <version>`, matching this repo's
  `vX.Y.Z` git tag convention.

## [0.2.1] - 2026-08-05

### Changed
- Simplified `README.md`: restructured around description / features /
  installation / supported providers / what is anyask / why do you need
  anyask, trimming the detailed API/Errors/CLI reference sections down to
  pointers at the hosted docs (`docs/reference`, `docs/cli`,
  `docs/providers` already cover that ground) instead of duplicating it.

### Fixed
- `README.md`'s `SPEC.md` links used repo-relative paths, which break on
  PyPI (README.md also serves as the PyPI project description, and PyPI
  has no repo tree to resolve a relative link against — it just 404s).
  Switched to absolute GitHub blob URLs, matching the pattern already
  used for the same file in `docs/index.md`.

## [0.2.0] - 2026-08-05

### Added
- `SPEC.md` — architecture, the full public API contract, and seven Mermaid
  diagrams (component overview, a single `ask()` call as a sequence diagram,
  lazy per-provider imports, the Provider/AskResponse/error class hierarchy,
  the construction-kwargs split, the `reasoning=True` per-provider decision
  flow, and the release pipeline), plus an explicit out-of-scope list.
- "What anyask does (and doesn't) do" section in `README.md`/`docs/index.md`:
  documents that `prompt`/`AskResponse.content` are always plain strings, so
  there's no multi-turn conversation state, no normalized system-prompt
  support, no multi-modal input/output, and no streaming — spelled out
  explicitly rather than left implicit.

### Changed
- `pyproject.toml`'s description now says "single text prompt" / "raw text
  response" instead of just "raw response", matching the capabilities
  clarification above.
- Fixed the provider count from 17 to 18 everywhere (`pyproject.toml`,
  `README.md`, `SPEC.md`, `CHANGELOG.md`, and three docs pages) — the actual
  registry has 18 named vendor providers plus the internal `openai_compat`
  base, which was never meant to be counted; "17" was wrong from the first
  release and had just been copied forward since.

### Removed
- `scripts/verify-version.ps1` — unreferenced anywhere in the repo, and
  unlike this repo's other `.ps1` scripts (which exist because their bash
  counterparts need a native-PowerShell equivalent), `verify-version.py`
  already runs identically on Windows with no bash dependency to work
  around. Also a strictly weaker check (missing `.bumpversion.cfg`).
- `scripts/verify-version.py` and its two references (the local pre-commit
  hook, the Makefile's `verify-version` target). Version-drift detection
  now relies solely on `tests/unit/test_version_consistency.py`, which
  already runs automatically in CI via `tox` on every push.
- `.pre-commit-config.yaml` — not installed (missing from the `dev` extra),
  not documented, not invoked by CI. Everything it checked (ruff,
  ruff-format, black, mypy) is the same tooling at the same pinned versions
  already enforced via `tox -e lint/format/typecheck`.

### Fixed
- Replaced the `Makefile`'s `tomllib`/`tomli`-fallback version extraction
  (which assumed `tomli` was installed on Python <3.11, but never declared
  it as a dependency anywhere) and `scripts/release.ps1`'s equivalent (which
  had no fallback at all) with a plain regex against the raw file text — no
  stdlib-version gate, no extra dependency, and now also supports
  single-quoted `version = '0.1.1'` in addition to double-quoted.

## [0.1.1] - 2026-08-04

### Added
- `anyask` console script: `anyask --version`/`anyask version`, `anyask check
  <provider>` (construct one provider without a network call, reporting whether
  its SDK is installed and credentials resolve; exits 0/1), and `anyask ready`
  (sweep every provider using only environment-variable credentials; always
  exits 0 - a status report, not a gate). Built on stdlib `argparse`, not
  `click`, to keep the bare install at 2 packages.
- Consistent `temperature=0.2`/`max_tokens=2048` defaults across all 18
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

## [0.1.0] - 2026-08-04

### Added
- Initial extraction release: `ask()`, `ask_async()`, `list_models()`,
  `list_models_async()`, and `get_provider()` — one normalized
  `AskResponse` shape across 18 providers (OpenAI, Anthropic, Gemini,
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
