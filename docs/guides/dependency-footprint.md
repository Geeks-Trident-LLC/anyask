# Dependency Footprint

`llmbridge` supports 17 LLM providers, but no single install needs all of
their SDKs at once. `pip install llmbridge` installs only `PyYAML` (used by
the built-in default-model catalog) — every provider SDK is an opt-in
extra via `pip install llmbridge[<provider>]` (see
[Installation](../getting-started/installation.md)). This page documents
what a representative sample of extras actually pull in, verified with
real, clean-venv installs — useful when you're sizing a container image
or just curious where the weight goes.

## Bare install: 2 packages

```bash
pip install llmbridge
```

Installs only `llmbridge` itself plus its single dependency, `PyYAML`. This
is fully functional on its own — `import llmbridge`, `llmbridge.ask(...)`,
`llmbridge.get_provider(...)` all work for constructing/inspecting objects.
Only *calling* a provider requires its extra:

```pycon
>>> import llmbridge
>>> llmbridge.ask("hi", provider="anthropic", model="claude-haiku-4-5-20251001")
ImportError: Provider 'anthropic' requires additional dependencies that
are not installed. Install with: pip install llmbridge[anthropic]
```

## Per-provider package counts (measured sample)

| Extra | Total packages (incl. llmbridge) | What makes up the difference |
|---|---:|---|
| `[bedrock]` | 9 | Reuses `boto3`'s own credential chain — no `httpx`/`pydantic` stack, just `boto3`/`botocore` + their small support libs (`jmespath`, `s3transfer`, `python-dateutil`, `six`, `urllib3`) |
| `[oci]` | 15 | Per-request cryptographic signing needs `cryptography` + `pyOpenSSL` + `PyJWT` (no bearer API key at all), plus `circuitbreaker`/`crc32c`/`pytz` support libs |
| `[anthropic]` | 18 | `httpx` (sync+async client) + `pydantic` (typed request/response models), plus `jiter`/`distro`/`docstring_parser` |
| `[openai]` (+ 9 aliases below) | 19 | Same `httpx` + `pydantic` stack as `anthropic`, plus `tqdm`/`colorama` |
| `[mistral]` | 31 | **Heaviest** — a full OpenTelemetry SDK (`opentelemetry-api`/`-sdk`/exporters) plus `protobuf`, `googleapis-common-protos`, and `invoke` (a task-runner library), on top of its own `httpx`+`pydantic` stack |

`[openai]` also covers `deepseek`, `groq`, `xai`, `together`, `fireworks`,
`cerebras`, `perplexity`, `openrouter`, and `moonshot` at no extra
package cost — all nine subclass the same OpenAI-compatible chat-
completions client (`llmbridge/providers/openai_compat.py`) and need
nothing beyond the `openai` package itself.

`[gemini]`/`[vertexai]` (both use `google-genai`), `[azure]`, and
`[cohere]` were not individually re-verified in a clean venv for this
page — they weren't called out as a lightest/heaviest/unusual case in
the sample above, so treat their counts as "similar shape to `anthropic`/
`openai`'s `httpx`+`pydantic` stack, plus `google-auth`→`cryptography`
for the Google-backed ones" rather than exact numbers.

**Package count and disk size are two different axes.** `oci` pulls in
fewer total packages (15) than `mistral` (31) despite being the single
heaviest *file* on PyPI (~36MB) — a heavy SDK author can produce either a
few large files or many small ones.

## Why some SDKs are heavier than others

The differences trace back to each provider's own upstream SDK design,
not anything this package controls:

- **HTTP client choice**: `httpx` (async-capable, used by `openai`,
  `anthropic`, and most `httpx`-based SDKs) pulls in `httpcore`, `h11`,
  `anyio`, and `sniffio` on top of itself. `bedrock` reuses `boto3`'s own
  request machinery instead.
- **Request/response validation**: `pydantic` (used by most of the
  `httpx`-based SDKs) brings its own compiled-Rust `pydantic_core` plus
  `annotated-types`/`typing-inspection`.
- **Credential machinery**: `oci` needs `cryptography`+`pyOpenSSL`+
  `PyJWT` because it signs every request itself rather than sending a
  bearer token; `bedrock` needs nothing extra since `boto3` resolves AWS
  credentials internally.
- **Unrelated SDK features**: `mistral`'s OpenTelemetry stack (for
  built-in tracing) has nothing to do with making a chat-completions
  call, but ships as a hard requirement of that SDK regardless.

## How this was verified

Each number above comes from an actual `pip install -e ".[<extra>]"` into
a fresh, empty virtual environment (not from reading `pyproject.toml`
alone), followed by `pip list`:

```bash
python -m venv /tmp/llmbridge-check-<extra>
/tmp/llmbridge-check-<extra>/bin/pip install -e ".[<extra>]"
/tmp/llmbridge-check-<extra>/bin/pip list
```

Verified extras: bare (no extra), `anthropic`, `openai`, `bedrock`,
`oci`, `mistral` — the same representative sample (lightest/heaviest/
unusual) called out above. Counts were produced on Python 3.14; exact
transitive versions will differ slightly on Python 3.9/3.12 (this
package's actual CI matrix), but the package *count* and relative
ordering should hold. See `llmbridge/registry.py` for the lazy-loading
mechanism that makes per-extra installs possible in the first place.
