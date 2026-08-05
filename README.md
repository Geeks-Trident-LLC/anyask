# anyask

Call an LLM provider with a single text prompt, get its raw response back —
one function, 18 vendors, zero routing/fallback magic.

```python
import anyask

response = anyask.ask(
    "Say hello in one word.",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    api_key="sk-...",  # or set ANTHROPIC_API_KEY
)

print(response.content)   # "Hello"
print(response.usage)     # TokenUsage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
```

## Features

- **Lightweight installation** — a bare `pip install anyask` pulls in exactly
  one dependency (`PyYAML`); each provider you actually use adds only that
  provider's own SDK, never the other 17. Cheap to bake into a container
  image, fast to install in CI.
- **One function, 18 vendors** — `ask()`/`ask_async()` take the same
  `provider=`/`model=` signature everywhere; swap providers by changing one
  string.
- **No routing/fallback magic** — `provider` and `model` are always explicit;
  anyask never silently swaps one vendor for another.
- **Real reasoning support, not simulated** — `reasoning=True` enables each
  provider's own actual extended-thinking mechanism where one exists.
- **A small built-in CLI** — `anyask check`/`anyask ready` answer "is this
  provider's SDK installed and its credentials set," with no network call.

## Installation

```bash
pip install anyask                 # minimal install — no provider usable yet
pip install anyask[anthropic]      # + Anthropic only
pip install anyask[openai,gemini]  # + multiple providers
pip install anyask[all]            # + every provider SDK
```

Verify:

```bash
anyask version                     # installed version
anyask ready                       # sweep every provider's SDK/credential status
anyask check anthropic             # check one provider specifically, exit 0/1
```

## Supported providers

`openai`, `anthropic`, `gemini`, `vertexai`, `azure`, `mistral`, `cohere`, `bedrock`,
`oci`, and the nine OpenAI-compatible vendors: `deepseek`, `groq`, `xai`, `together`,
`fireworks`, `cerebras`, `perplexity`, `openrouter`, `moonshot`.

## What is anyask

A single-responsibility package for one job: call an LLM provider, get its raw
response back. `prompt` is always a plain string and `AskResponse.content` is
always a plain string — no multi-turn history, no system prompts, no
multi-modal input/output, no streaming. That narrow contract is deliberate,
not a missing feature — see [SPEC.md](SPEC.md) for the full design and
[SPEC.md's out-of-scope list](SPEC.md#10-explicitly-out-of-scope) for exactly
what's excluded and why.

## Why do you need anyask

If you're calling more than one LLM vendor, you're already facing 18 SDKs with
18 different client shapes, auth patterns, and response formats. anyask is the
thin, uniform layer underneath: one call signature, one normalized response
shape (`content`/`usage`/`finish_reason`/`raw`), and nothing else — no
opinions about retries, fallback, cost tracking, or agent orchestration. Build
those on top if you need them, or just use anyask directly when you don't.
It's also the option to reach for when install footprint actually matters —
CI pipelines, container images, or a library that can't afford to drag in
SDKs its own users never touch.

## Documentation

Full docs, including the [Quickstart](https://geeks-trident-llc.github.io/anyask/latest/getting-started/quickstart/),
[Providers reference](https://geeks-trident-llc.github.io/anyask/latest/providers/),
[CLI Guide](https://geeks-trident-llc.github.io/anyask/latest/cli/), and generated
[API Reference](https://geeks-trident-llc.github.io/anyask/latest/reference/):

- **Latest:** https://geeks-trident-llc.github.io/anyask/latest/
- **All versions:** https://geeks-trident-llc.github.io/anyask/

## License

MIT
