![banner](assets/images/banner.svg)

`anyask` is a single-responsibility Python package for one job: call an LLM
provider, get its raw response back. No routing, no fallback, no retries —
you always name the provider and model explicitly, and you always get the
same normalized `AskResponse` shape back regardless of which of the 17
supported vendors you called.

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
```

It is designed for:

- anyone who wants a single, uniform call signature across every major LLM
  vendor without adopting a full agent/orchestration framework
- applications that already do their own retry, fallback, and routing logic
  and just need a thin, predictable client underneath
- teams that want to add or drop a provider by installing/removing one pip
  extra, with no import-time cost for providers they don't use

## Features

- One function, 17 vendors — `ask()`/`ask_async()` take the same
  `provider="..."`/`model="..."` signature for OpenAI, Anthropic, Gemini,
  Vertex AI, Azure, Mistral, Bedrock, Cohere, OCI, and nine
  OpenAI-compatible vendors (DeepSeek, Groq, xAI, Together AI, Fireworks
  AI, Cerebras, Perplexity, OpenRouter, Moonshot)
- Frozen, normalized `AskResponse` — `content`, `usage`, `finish_reason`
  (kept raw and provider-specific, never coerced), `provider`, `model`,
  and the untouched `raw` SDK response
- Lazy per-provider imports — installing none of the 17 provider extras
  still gives a fully working `import anyask`; each provider's SDK is
  only imported the moment it's actually used
- No routing/fallback magic — you always name the provider and model;
  `anyask` never silently swaps one for another
- `get_provider()` for reuse — construct a provider once and call
  `generate_sync()`/`generate()` many times instead of rebuilding an SDK
  client on every call

## Explore the Docs

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Providers](providers/index.md)
- [Dependency Footprint](guides/dependency-footprint.md)
- [API Reference](reference/index.md)
