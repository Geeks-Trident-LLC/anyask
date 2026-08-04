# CLI Guide

`anyask` installs a small `anyask` console script alongside the Python
API. It doesn't call any provider's `generate`/`generate_sync` — it only
answers "is this provider ready to use," by attempting the same
construction `get_provider()`/`ask()` do internally, without making a
network call.

## Version

```bash
anyask --version
anyask version
```

Both print the installed version; `--version` is the conventional flag,
`version` is an explicit subcommand for scripts that prefer a subcommand
over a top-level flag.

## `check` — is one specific provider ready?

```bash
anyask check anthropic
```

Attempts to construct the named provider - the same construction
`get_provider()`/`ask()` use internally - and reports one of:

- `ready` - the SDK is installed and credentials resolved (exit code `0`)
- `SDK not installed - ...` - the provider's pip extra isn't installed
  (exit code `1`)
- `credentials missing - ...` - the SDK is installed but no credentials
  resolved (exit code `1`)
- `unknown provider` - the name isn't one anyask recognizes (exit code `1`)

By default, credentials resolve from the provider's usual environment
variable (see [Providers](../providers/index.md)). Pass them explicitly
instead with the same flags `ask()`'s construction kwargs accept:

```bash
anyask check azure \
  --api-key sk-... \
  --endpoint https://example.azure.com \
  --deployment my-deployment \
  --api-version 2024-06-01

anyask check bedrock --region us-east-1
anyask check oci --compartment-id ocid1.compartment.oc1..xxx --region us-chicago-1
```

`check`'s exit code makes it usable directly in scripts:

```bash
anyask check anthropic || echo "anthropic isn't configured"
```

## `ready` — sweep every provider

```bash
anyask ready
```

```text
[--] anthropic   credentials missing - ANTHROPIC_API_KEY is not set
[--] azure       credentials missing - AZURE_API_KEY is not set
[--] bedrock     credentials missing - AWS region is not set (pass region= or set BEDROCK_REGION/BEDROCK_DEFAULT_REGION)
...
[OK] openai      ready
```

Checks every registered provider using only its environment-variable
credentials (no flags - use `check <provider>` to test explicit values).
`ready` **always exits `0`** - it's a status report, not a gate. Most
setups only configure a handful of the 17 providers, so a report full of
"credentials missing" lines for the rest is expected, not a failure.

## What `check`/`ready` don't verify

Neither command makes a real API call - "ready" means the provider
constructed successfully (SDK importable, required config/credentials
present), not that a live call would succeed. A revoked API key, an
expired token, or insufficient account permissions won't show up here -
only an actual `ask()` call surfaces those.
