# Installation

`anyask` supports Python 3.9+ and works on all major platforms.

## Install from PyPI

```bash
pip install anyask
```

This installs the core API only — `PyYAML` (for the built-in model
catalog) is the only dependency, and zero provider SDKs. Pick the
provider(s) you actually use as an extra:

```bash
pip install anyask[openai]        # OpenAI (also covers deepseek, groq,
                                   # xai, together, fireworks, cerebras,
                                   # perplexity, openrouter, moonshot -
                                   # all OpenAI-compatible, no extra
                                   # SDK needed beyond openai)
pip install anyask[anthropic]     # Anthropic Claude
pip install anyask[gemini]        # Google Gemini
pip install anyask[vertexai]      # Google Vertex AI
pip install anyask[azure]         # Azure AI Inference / Azure OpenAI
pip install anyask[mistral]       # Mistral AI
pip install anyask[bedrock]       # Amazon Bedrock
pip install anyask[cohere]        # Cohere
pip install anyask[oci]           # Oracle Cloud Infrastructure
```

Need more than one? Combine extras: `pip install anyask[anthropic,gemini]`.

Or install every provider SDK at once:

```bash
pip install anyask[all]
```

Provider SDK imports are lazy — resolving one provider by name never
imports another provider's SDK, so `import anyask` always succeeds even in
an environment with no provider SDKs installed at all. Only *calling*
`ask()`/`get_provider()` with a given `provider=` requires that provider's
extra (or `[all]`) to be installed.

See the [Dependency Footprint](../guides/dependency-footprint.md) guide
for exactly how many packages each extra installs.

## Verify installation

```bash
python -c "import anyask; print(anyask.__version__)"
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
[CONTRIBUTING.md](https://github.com/Geeks-Trident-LLC/anyask/blob/main/CONTRIBUTING.md).
