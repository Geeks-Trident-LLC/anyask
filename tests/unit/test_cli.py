from typing import Any, Dict

import pytest

from anyask import __version__, cli
from anyask.errors import ProviderAuthError, ProviderError, ProviderNotFoundError


def test_main_no_args_prints_help(capsys):
    exit_code = cli.main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "usage: anyask" in out


def test_main_version_command_prints_version(capsys):
    exit_code = cli.main(["version"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == f"anyask {__version__}"


def test_main_dash_dash_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"anyask {__version__}"


# ------------------------------------------------------------
# check <provider>
# ------------------------------------------------------------


def _patch_get_provider(monkeypatch, fn):
    monkeypatch.setattr(cli, "get_provider", fn)


def test_check_ready_provider_exits_zero(monkeypatch, capsys):
    _patch_get_provider(monkeypatch, lambda provider, **config: object())

    exit_code = cli.main(["check", "anthropic"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "anthropic: ready"


def test_check_unknown_provider_exits_one(monkeypatch, capsys):
    def _raise(provider, **config):
        raise ProviderNotFoundError(f"Unknown provider: {provider!r}")

    _patch_get_provider(monkeypatch, _raise)

    exit_code = cli.main(["check", "not-a-real-provider"])

    assert exit_code == 1
    assert "unknown provider" in capsys.readouterr().out


def test_check_missing_sdk_exits_one(monkeypatch, capsys):
    def _raise(provider, **config):
        raise ImportError(
            "Provider 'bedrock' requires additional dependencies that are "
            "not installed. Install with: pip install anyask[bedrock]"
        )

    _patch_get_provider(monkeypatch, _raise)

    exit_code = cli.main(["check", "bedrock"])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "SDK not installed" in out
    assert "pip install anyask[bedrock]" in out


def test_check_missing_credentials_exits_one(monkeypatch, capsys):
    def _raise(provider, **config):
        raise ProviderAuthError("ANTHROPIC_API_KEY is not set")

    _patch_get_provider(monkeypatch, _raise)

    exit_code = cli.main(["check", "anthropic"])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "credentials missing" in out
    assert "ANTHROPIC_API_KEY" in out


def test_check_generic_provider_error_exits_one(monkeypatch, capsys):
    def _raise(provider, **config):
        raise ProviderError("boom")

    _patch_get_provider(monkeypatch, _raise)

    exit_code = cli.main(["check", "oci"])

    assert exit_code == 1
    assert "error - boom" in capsys.readouterr().out


def test_check_forwards_cli_flags_as_construction_kwargs(monkeypatch):
    seen: Dict[str, Any] = {}

    def _capture(provider, **config):
        seen["provider"] = provider
        seen["config"] = config
        return object()

    _patch_get_provider(monkeypatch, _capture)

    cli.main(
        [
            "check",
            "azure",
            "--api-key",
            "sk-test",
            "--endpoint",
            "https://example.azure.com",
            "--deployment",
            "my-deployment",
            "--api-version",
            "2024-06-01",
        ]
    )

    assert seen["provider"] == "azure"
    assert seen["config"] == {
        "api_key": "sk-test",
        "endpoint": "https://example.azure.com",
        "deployment": "my-deployment",
        "api_version": "2024-06-01",
    }


def test_check_omits_unset_flags_from_config(monkeypatch):
    seen: Dict[str, Any] = {}

    def _capture(provider, **config):
        seen["config"] = config
        return object()

    _patch_get_provider(monkeypatch, _capture)

    cli.main(["check", "anthropic", "--api-key", "sk-test"])

    assert seen["config"] == {"api_key": "sk-test"}


# ------------------------------------------------------------
# ready
# ------------------------------------------------------------


class _FakeRegistry:
    def __init__(self, names):
        self._names = names

    def all(self):
        return {name: None for name in self._names}


def test_ready_excludes_openai_compat(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "registry", _FakeRegistry(["anthropic", "openai_compat", "openai"])
    )
    _patch_get_provider(monkeypatch, lambda provider, **config: object())

    exit_code = cli.main(["ready"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "openai_compat" not in out
    assert "anthropic" in out
    assert "openai" in out


def test_ready_always_exits_zero_even_when_none_ready(monkeypatch, capsys):
    monkeypatch.setattr(cli, "registry", _FakeRegistry(["anthropic", "openai"]))

    def _raise(provider, **config):
        raise ProviderAuthError(f"{provider.upper()}_API_KEY is not set")

    _patch_get_provider(monkeypatch, _raise)

    exit_code = cli.main(["ready"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "[--] anthropic" in out
    assert "[--] openai" in out
    assert "credentials missing" in out


def test_ready_shows_ok_marker_for_ready_providers(monkeypatch, capsys):
    monkeypatch.setattr(cli, "registry", _FakeRegistry(["anthropic"]))
    _patch_get_provider(monkeypatch, lambda provider, **config: object())

    cli.main(["ready"])

    assert "[OK] anthropic" in capsys.readouterr().out
