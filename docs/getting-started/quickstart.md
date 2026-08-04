# Quickstart

Five minutes to your first call: install, set a credential, and use the
four public functions — `ask()`, `ask_async()`, `list_models()`, and
`get_provider()`.

## Prerequisites

You need an API key for at least one supported provider, set as an
environment variable. This guide uses Anthropic as the running example.

```bash
pip install llmbridge[anthropic]
export ANTHROPIC_API_KEY=sk-ant-...
```

See [Providers](../providers/index.md) for the full list of supported
providers and their credential environment variables.

## 1. `ask()` — a single synchronous call

```python
import llmbridge

response = llmbridge.ask(
    "Say hello in one word.",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
)

print(response.content)          # "Hello"
print(response.usage)            # TokenUsage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
print(response.finish_reason)    # "end_turn" (raw, provider-specific)
print(response.provider)         # "anthropic"
print(response.model)            # "claude-haiku-4-5-20251001"
```

`api_key` can also be passed explicitly instead of relying on the
`ANTHROPIC_API_KEY` environment variable:

```python
response = llmbridge.ask(
    "Say hello in one word.",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    api_key="sk-ant-...",
)
```

Extra keyword arguments are split automatically: construction keys
(`api_key`, `endpoint`, `api_version`, `deployment`, `region`, `project`,
`compartment_id`) go to the provider's constructor; everything else
(`temperature`, `max_tokens`, ...) is forwarded to the generation call
unchanged.

## 2. `ask_async()` — the async counterpart

```python
import asyncio
import llmbridge


async def main():
    response = await llmbridge.ask_async(
        "Say hello in one word.",
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
    )
    print(response.content)


asyncio.run(main())
```

## 3. `list_models()` — see what a provider currently serves

```python
models = llmbridge.list_models("anthropic")
print(models)   # ['claude-opus-4-8', 'claude-sonnet-4-5', ...]
```

Not every provider supports live model listing — `list_models()` raises
`llmbridge.ProviderNotFoundError` for providers with no listing endpoint
(e.g. Perplexity).

## 4. `get_provider()` — reuse a client across many calls

`ask()` builds a fresh provider (and its underlying SDK client) on every
call. If you're making many calls against the same provider/credentials,
construct once and reuse:

```python
provider = llmbridge.get_provider("anthropic")

for prompt in ["What is 2+2?", "What is the capital of France?"]:
    response = provider.generate_sync(prompt, model="claude-haiku-4-5-20251001")
    print(response.content)
```

This avoids rebuilding the SDK client (a fresh `boto3.client()`,
re-reading `~/.oci/config`, etc.) on every single call. `provider` also
exposes an async `generate()` method for use inside an `async def`.

## Next steps

- See [Providers](../providers/index.md) for the full list of supported
  providers, their credentials, and cloud-gateway providers that take
  extra constructor parameters instead of `api_key`.
- See the [API Reference](../reference/index.md) for every function,
  type, and error class.
