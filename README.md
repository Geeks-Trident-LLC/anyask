# anyask

A single-responsibility Python package for one job: call an LLM provider, get its raw
response back. No routing, no fallback, no retries - you always name the provider and
model explicitly, and you always get the same normalized `AskResponse` shape back
regardless of which of the 17 supported vendors you called.

```python
import anyask

response = anyask.ask(
    "Say hello in one word.",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    api_key="sk-...",  # or set ANTHROPIC_API_KEY
)

print(response.content)          # "Hello"
print(response.usage)            # TokenUsage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
print(response.finish_reason)    # "end_turn" (raw, provider-specific - never normalized)
print(response.provider)         # "anthropic"
print(response.model)            # "claude-haiku-4-5-20251001"
```

## Documentation

Full documentation, including the [Providers](https://geeks-trident-llc.github.io/anyask/latest/providers/)
reference table, [Quickstart](https://geeks-trident-llc.github.io/anyask/latest/getting-started/quickstart/),
and generated [API Reference](https://geeks-trident-llc.github.io/anyask/latest/reference/),
is available at:

- **Latest docs:** [https://geeks-trident-llc.github.io/anyask/latest/](https://geeks-trident-llc.github.io/anyask/latest/)
- **All versions:** [https://geeks-trident-llc.github.io/anyask/](https://geeks-trident-llc.github.io/anyask/)

## Supported providers

`openai`, `anthropic`, `gemini`, `vertexai`, `azure`, `mistral`, `cohere`, `bedrock`,
`oci`, and the nine OpenAI-compatible vendors: `deepseek`, `groq`, `xai`, `together`,
`fireworks`, `cerebras`, `perplexity`, `openrouter`, `moonshot`.

## Install

A bare `pip install anyask` pulls in zero provider SDKs - only `PyYAML` (for the
built-in model catalog). Install the extra(s) for the provider(s) you actually use:

```bash
pip install anyask[anthropic]
pip install anyask[openai,gemini]
pip install anyask[all]       # every provider SDK
```

Provider SDK imports are lazy: resolving one provider by name never imports another
provider's SDK, so `import anyask` always succeeds even in an environment with no
provider SDKs installed at all.

## API

```python
def ask(prompt: str, *, provider: str, model: str, reasoning: bool = False, **kwargs) -> AskResponse: ...
async def ask_async(prompt: str, *, provider: str, model: str, reasoning: bool = False, **kwargs) -> AskResponse: ...
def list_models(provider: str, **kwargs) -> list[str]: ...
async def list_models_async(provider: str, **kwargs) -> list[str]: ...
def get_provider(provider: str, **config) -> Provider: ...
```

`**kwargs` passed to `ask`/`ask_async`/`list_models` is split automatically: construction
keys (`api_key`, `endpoint`, `api_version`, `deployment`, `region`, `project`,
`compartment_id`) go to the provider's constructor; everything else (`temperature`,
`max_tokens`, ...) is forwarded to the generation call unchanged. Any key a given
provider doesn't recognize is simply ignored, so the same kwargs dict can be handed to
any provider.

### `ask()` / `ask_async()`

```python
response = anyask.ask("What is 2+2?", provider="openai", model="gpt-4o-mini")

response = await anyask.ask_async(
    "What is 2+2?", provider="openai", model="gpt-4o-mini", temperature=0.0,
)
```

Pass `reasoning=True` for extended/deliberate reasoning, using each provider's own
real mechanism (Anthropic extended thinking, OpenAI/OpenAI-compatible
`reasoning_effort`, Gemini/Vertex AI thinking budgets, Bedrock's Claude thinking
field). Azure, Mistral, Cohere, and OCI have no documented per-call toggle and raise
`ProviderNotFoundError` rather than silently ignoring it - see the
[Providers](https://geeks-trident-llc.github.io/anyask/latest/providers/#reasoning-support)
page for the full support matrix.

```python
response = anyask.ask(
    "What's 17 * 24?", provider="anthropic", model="claude-opus-4-8", reasoning=True,
)
```

### `list_models()` / `list_models_async()`

```python
models = anyask.list_models("anthropic", api_key="sk-...")
# ['claude-opus-4-8', 'claude-sonnet-4-5', ...]
```

Raises `anyask.ProviderNotFoundError` if the resolved provider doesn't expose a live
model-listing endpoint (e.g. Perplexity returns a static list instead).

### `get_provider()` - reusable provider instances

`ask()` builds a fresh provider (and its underlying SDK client) on every call. If you're
making many calls against the same provider/credentials - e.g. resolving dozens of
prompts against one Anthropic API key in a loop - construct once and reuse:

```python
provider = anyask.get_provider("anthropic", api_key="sk-...")

for prompt in prompts:
    response = provider.generate_sync(prompt, model="claude-haiku-4-5-20251001")
```

This avoids rebuilding the SDK client (a fresh `boto3.client()`, re-reading
`~/.oci/config`, etc.) on every single call.

## Errors

```python
class AskLLMError(Exception): ...
class ProviderNotFoundError(AskLLMError): ...   # unknown provider name, or missing capability
class ProviderError(AskLLMError): ...            # provider call failed - raised with `from exc`
class ProviderAuthError(ProviderError): ...       # missing/invalid credentials at construction
```

Every provider call failure is raised as `ProviderError(...) from exc`, so
`err.__cause__` is always the original SDK exception - inspect it if you need
vendor-specific error details (status codes, error types, etc.).

## CLI

A small `anyask` console script installs alongside the Python API - it doesn't call
any provider, it only checks readiness (SDK installed, credentials resolve) without
making a network call:

```bash
anyask --version
anyask check anthropic                    # exit 0/1
anyask check bedrock --region us-east-1
anyask ready                              # sweep every provider (always exits 0)
```

See the [CLI Guide](https://geeks-trident-llc.github.io/anyask/latest/cli/) for the
full command reference.

## License

MIT
