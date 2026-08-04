# Providers

anyask integrates 17 LLM providers behind one interface — `ask()`,
`ask_async()`, `list_models()`, `list_models_async()`, and
`get_provider()` all take the same `provider="..."` value. Most providers
just need an API key; three cloud-gateway providers (Bedrock, Vertex AI,
OCI) use their own ambient credential chains instead and take extra
constructor parameters in place of `api_key`.

There is no automatic routing or fallback anywhere in anyask — `provider`
is always an explicit, required argument on every call.

## All providers

| Provider | `provider` value | Credential | Extra params | Default model |
|---|---|---|---|---|
| OpenAI | `"openai"` | `OPENAI_API_KEY` | — | `gpt-4o-mini` |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | — | `claude-haiku-4-5-20251001` |
| Google Gemini | `"gemini"` | `GEMINI_API_KEY` | — | `gemini-2.5-flash` |
| DeepSeek | `"deepseek"` | `DEEPSEEK_API_KEY` | — | `deepseek-v4-flash` |
| Groq | `"groq"` | `GROQ_API_KEY` | — | `llama-3.1-8b-instant` |
| xAI (Grok) | `"xai"` | `XAI_API_KEY` | — | `grok-3-mini` |
| Together AI | `"together"` | `TOGETHER_API_KEY` | — | `meta-llama/Llama-3.1-8B-Instruct-Turbo` |
| Fireworks AI | `"fireworks"` | `FIREWORKS_API_KEY` | — | `accounts/fireworks/models/llama-v3p1-8b-instruct` |
| Cerebras | `"cerebras"` | `CEREBRAS_API_KEY` | — | `llama3.1-8b` |
| Perplexity | `"perplexity"` | `PERPLEXITY_API_KEY` | — | `sonar` |
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` | — | `google/gemini-2.5-flash-lite` |
| Moonshot AI (Kimi) | `"moonshot"` | `MOONSHOT_API_KEY` | — | `moonshot-v1-8k` |
| Mistral AI | `"mistral"` | `MISTRAL_API_KEY` | — | `mistral-small-latest` |
| Azure OpenAI | `"azure"` | `AZURE_API_KEY` | `endpoint` (`AZURE_ENDPOINT`), `api_version` (`AZURE_API_VERSION`), `deployment` (`AZURE_DEPLOYMENT`) — `model` is passed as usual but the deployment name drives the actual call | *(your deployment name)* |
| Amazon Bedrock | `"bedrock"` | *(none — AWS credential chain)* | `region` (`BEDROCK_REGION`/`BEDROCK_DEFAULT_REGION`, required) | `anthropic.claude-haiku-4-5-v1:0` |
| Cohere | `"cohere"` | `COHERE_API_KEY` | — | `command-light` |
| Google Vertex AI | `"vertexai"` | *(none — GCP ADC credential chain)* | `project` (`VERTEXAI_PROJECT`, required), `region` (`VERTEXAI_REGION`, required) | `gemini-2.5-flash` |
| Oracle OCI | `"oci"` | *(none — `~/.oci/config` credential file)* | `compartment_id` (`OCI_COMPARTMENT_ID`, required), `region` (`OCI_REGION`, optional — falls back to the config file) | `meta.llama-3.3-70b-instruct` |

`openai`'s extra also installs the SDK backing nine OpenAI-compatible
vendors (`deepseek`, `groq`, `xai`, `together`, `fireworks`, `cerebras`,
`perplexity`, `openrouter`, `moonshot`) — they share one HTTP client
implementation (`anyask/providers/openai_compat.py`) and differ only in
base URL, env var, and default model. Each still needs its own extra
installed (e.g. `pip install anyask[groq]`) since credentials and default
models are provider-specific, but no additional package beyond `openai`
itself is pulled in.

Credentials resolve in this order everywhere: explicit function/keyword
argument (`api_key=...`, `region=...`, ...) > provider-specific
environment variable. There is no config-file fallback.

## Two implementation shapes

Internally, providers fall into two groups — this only matters if you're
extending anyask itself, not for calling it:

- **Shape A** — OpenAI-compatible chat-completions API (DeepSeek, Groq,
  xAI, Together AI, Fireworks AI, Cerebras, Perplexity, OpenRouter,
  Moonshot). These share one implementation
  (`anyask/providers/openai_compat.py`) and differ only in base URL, env
  var, and default model.
- **Shape B** — a native SDK with its own request/response shape (OpenAI,
  Anthropic, Gemini, Azure, Mistral, Bedrock, Cohere, Vertex AI, OCI).
  Bedrock, Vertex AI, and OCI additionally take extra constructor
  parameters (`region`/`project`/`compartment_id`) instead of `api_key`,
  since they authenticate via their cloud platform's own credential chain
  rather than a project-level API key.

## Reasoning support

`ask()`/`ask_async()` take a `reasoning: bool = False` keyword. When
`True`, each provider that supports it enables extended/deliberate
reasoning using its own real mechanism — nothing is simulated, and
support is not uniform:

| Provider | `reasoning=True` behavior |
|---|---|
| Anthropic | Extended thinking (`thinking={"type": "enabled", "budget_tokens": ...}`); default budget 1024, override with `thinking_budget=N`. Temperature is left unset, since the API rejects an override while thinking is enabled. |
| OpenAI, and the 9 OpenAI-compatible vendors | `reasoning_effort="medium"` by default; override with `reasoning_effort="low"/"high"/...`. Support and accepted values depend on the specific model — an unsupported combination surfaces as a `ProviderError` from the vendor's own API. |
| Gemini, Vertex AI | `thinking_config.thinking_budget=-1` (dynamic — model decides) by default; override with `thinking_budget=N` (or `0` to force off even when `reasoning=True`). |
| Amazon Bedrock | Same `thinking` field as native Anthropic, via `additionalModelRequestFields` — only applies to Claude models on Bedrock; other model families error from the API if it's not recognized. |
| Azure, Mistral, Cohere, OCI | Not supported — `reasoning=True` raises `ProviderNotFoundError` immediately rather than silently doing nothing. |

`reasoning=False` (the default) leaves every provider's existing
non-reasoning behavior completely unchanged.

## A note on Vertex AI and OCI model IDs

**Vertex AI** serves the exact same Gemini model ID strings as the native
`gemini` provider (e.g. `gemini-2.5-pro` means the same thing to both).
**OCI** uses `vendor.model-name` IDs (`meta.llama-3.3-70b-instruct`)
that share the `meta.` vendor prefix with Bedrock's own re-hosted model
namespace. Since anyask has no auto-routing to begin with, this is never
ambiguous in practice — you always select `provider="vertexai"` or
`provider="oci"` explicitly — but it's worth knowing if you're
cross-referencing model IDs between providers.
