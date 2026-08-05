# anyask/cli.py

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence

from anyask import __version__
from anyask.api import get_provider
from anyask.errors import ProviderAuthError, ProviderError, ProviderNotFoundError
from anyask.registry import registry

# Same construction-kwarg vocabulary as api.py's ask()/ask_async() - see
# _CONSTRUCTION_KEYS there. Kept as an explicit {kwarg: flag} map (not
# derived from that set) so flag spelling is a CLI concern, not an API one.
_CONSTRUCTION_FLAGS = {
    "api_key": "--api-key",
    "endpoint": "--endpoint",
    "api_version": "--api-version",
    "deployment": "--deployment",
    "region": "--region",
    "project": "--project",
    "compartment_id": "--compartment-id",
}

# "openai_compat" is the internal generic base for arbitrary
# OpenAI-compatible endpoints (no env var of its own, requires an
# explicit --endpoint) - it isn't one of the named vendors `ready` sweeps,
# though `check openai_compat --endpoint ... --api-key ...` still works.
_READY_SWEEP_EXCLUDE = {"openai_compat"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anyask",
        description="Call an LLM provider, get its raw response back.",
    )
    parser.add_argument("--version", action="version", version=f"anyask v{__version__}")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print the installed anyask version")

    check_parser = subparsers.add_parser(
        "check",
        help="Check whether one provider's SDK is installed and its "
        "credentials resolve",
        description=(
            "Attempts to construct the named provider - the same "
            "construction get_provider()/ask() do internally - without "
            "making any API call. Reports whether the provider's SDK "
            "extra is installed and whether credentials resolve (from "
            "the flags below, or the provider's usual environment "
            "variable). Exits 0 if the provider is ready, 1 otherwise."
        ),
    )
    check_parser.add_argument("provider", help="Provider name, e.g. anthropic")
    for key, flag in _CONSTRUCTION_FLAGS.items():
        check_parser.add_argument(flag, dest=key, default=None)

    subparsers.add_parser(
        "ready",
        help="Check readiness of every registered provider",
        description=(
            "Sweeps every registered provider using only its "
            "environment-variable credentials (no CLI flags - use "
            "`check <provider>` to test explicit values). Always exits "
            "0 - this is a status report, not a gate; most setups only "
            "configure a handful of providers, so 'ready' reports on "
            "purpose don't fail the command."
        ),
    )

    return parser


def _describe_status(provider: str, config: Dict[str, Any]) -> tuple:
    """Attempt to construct `provider` and classify the outcome.

    Returns (ready: bool, message: str). Never makes a network call -
    this only exercises each provider's __init__ (SDK import + the same
    fail-fast credential checks generate()/generate_sync() would hit),
    the same construction path get_provider()/ask() use internally.
    """
    try:
        get_provider(provider, **config)
    except ProviderNotFoundError:
        return False, "unknown provider"
    except ImportError as exc:
        return False, f"SDK not installed - {exc}"
    except ProviderAuthError as exc:
        return False, f"credentials missing - {exc}"
    except ProviderError as exc:
        return False, f"error - {exc}"
    return True, "ready"


def _cmd_check(provider: str, config: Dict[str, Any]) -> int:
    ready, message = _describe_status(provider, config)
    print(f"{provider}: {message}")
    return 0 if ready else 1


def _cmd_ready() -> int:
    names = sorted(name for name in registry.all() if name not in _READY_SWEEP_EXCLUDE)
    width = max(len(name) for name in names)
    for name in names:
        ready, message = _describe_status(name, {})
        marker = "OK" if ready else "--"
        print(f"[{marker}] {name.ljust(width)}  {message}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "version":
        print(f"anyask v{__version__}")
        return 0

    if args.command == "check":
        config = {
            key: value
            for key in _CONSTRUCTION_FLAGS
            if (value := getattr(args, key)) is not None
        }
        return _cmd_check(args.provider, config)

    if args.command == "ready":
        return _cmd_ready()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
