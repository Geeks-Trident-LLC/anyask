# Installation

`llmbridge` supports Python 3.9+ and works on all major platforms.

## Install from PyPI

```bash
pip install llmbridge
```

This installs the core API only — `PyYAML` (for the built-in model
catalog) is the only dependency, and zero provider SDKs. Pick the
provider(s) you actually use as an extra:

```bash
pip install llmbridge[openai]        # OpenAI (also covers deepseek, groq,
                                   # xai, together, fireworks, cerebras,
                                   # perplexity, openrouter, moonshot -
                                   # all OpenAI-compatible, no extra
                                   # SDK needed beyond openai)
pip install llmbridge[anthropic]     # Anthropic Claude
pip install llmbridge[gemini]        # Google Gemini
pip install llmbridge[vertexai]      # Google Vertex AI
pip install llmbridge[azure]         # Azure AI Inference / Azure OpenAI
pip install llmbridge[mistral]       # Mistral AI
pip install llmbridge[bedrock]       # Amazon Bedrock
pip install llmbridge[cohere]        # Cohere
pip install llmbridge[oci]           # Oracle Cloud Infrastructure
```

Need more than one? Combine extras: `pip install llmbridge[anthropic,gemini]`.

Or install every provider SDK at once:

```bash
pip install llmbridge[all]
```

Provider SDK imports are lazy — resolving one provider by name never
imports another provider's SDK, so `import llmbridge` always succeeds even in
an environment with no provider SDKs installed at all. Only *calling*
`ask()`/`get_provider()` with a given `provider=` requires that provider's
extra (or `[all]`) to be installed.

See the [Dependency Footprint](../guides/dependency-footprint.md) guide
for exactly how many packages each extra installs.

## Verify installation

```bash
python -c "import llmbridge; print(llmbridge.__version__)"
```

## Optional: Install development tools

```bash
pip install -e ".[all,dev]"
```

This includes every provider SDK plus:

- pytest + pytest-asyncio + pytest-cov
- ruff, black, mypy

Only working on one or two providers and don't want the rest installed?
Combine `-e .`, `dev`, and just the extra(s) you need instead of
`[all,dev]`:

```bash
pip install -e ".[dev]"
pip install -e ".[anthropic]"
pip install -e ".[openai]"
```

For docs tooling (mkdocs + mkdocstrings), see
[CONTRIBUTING.md](https://github.com/Geeks-Trident-LLC/llmbridge/blob/main/CONTRIBUTING.md).
