# askllm

A single-responsibility Python package for one job: call an LLM provider, get its raw
response back. No routing, no fallback, no retries - you always name the provider and
model explicitly, and you always get the same normalized `AskResponse` shape back
regardless of which of the 17 supported vendors you called.

```python
import askllm

response = askllm.ask(
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

## Supported providers

`openai`, `anthropic`, `gemini`, `vertexai`, `azure`, `mistral`, `cohere`, `bedrock`,
`oci`, and the nine OpenAI-compatible vendors: `deepseek`, `groq`, `xai`, `together`,
`fireworks`, `cerebras`, `perplexity`, `openrouter`, `moonshot`.

## Install

A bare `pip install askllm` pulls in zero provider SDKs - only `PyYAML` (for the
built-in model catalog). Install the extra(s) for the provider(s) you actually use:

```bash
pip install askllm[anthropic]
pip install askllm[openai,gemini]
pip install askllm[all]       # every provider SDK
```

Provider SDK imports are lazy: resolving one provider by name never imports another
provider's SDK, so `import askllm` always succeeds even in an environment with no
provider SDKs installed at all.

## API

```python
def ask(prompt: str, *, provider: str, model: str, **kwargs) -> AskResponse: ...
async def ask_async(prompt: str, *, provider: str, model: str, **kwargs) -> AskResponse: ...
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
response = askllm.ask("What is 2+2?", provider="openai", model="gpt-4o-mini")

response = await askllm.ask_async(
    "What is 2+2?", provider="openai", model="gpt-4o-mini", temperature=0.0,
)
```

### `list_models()` / `list_models_async()`

```python
models = askllm.list_models("anthropic", api_key="sk-...")
# ['claude-opus-4-8', 'claude-sonnet-4-5', ...]
```

Raises `askllm.ProviderNotFoundError` if the resolved provider doesn't expose a live
model-listing endpoint (e.g. Perplexity returns a static list instead).

### `get_provider()` - reusable provider instances

`ask()` builds a fresh provider (and its underlying SDK client) on every call. If you're
making many calls against the same provider/credentials - e.g. resolving dozens of
prompts against one Anthropic API key in a loop - construct once and reuse:

```python
provider = askllm.get_provider("anthropic", api_key="sk-...")

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

## License

MIT
