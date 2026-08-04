# tests/unit/test_version_consistency.py

import re
from pathlib import Path

import anyask

ROOT = Path(anyask.__file__).resolve().parent.parent


def extract_version_from_pyproject():
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError("Version not found in pyproject.toml")
    return match.group(1)


def normalize(version: str) -> str:
    # Keep only the numeric core version (e.g., "0.3.5" from "0.3.5-dev")
    return version.split("-", 1)[0].split("+", 1)[0].strip()


def test_version_consistency():
    # Load version from __init__.py
    init_file = ROOT / "anyask" / "__init__.py"
    init_version = re.search(r'__version__ = "(.+?)"', init_file.read_text()).group(1)

    # Load version from pyproject.toml
    pyproject_version = extract_version_from_pyproject()

    # Load version from .bumpversion.cfg
    cfg_file = ROOT / ".bumpversion.cfg"
    cfg_version = (
        re.search(r"current_version = (.+)", cfg_file.read_text()).group(1).strip()
    )

    assert (
        normalize(init_version)
        == normalize(pyproject_version)
        == normalize(cfg_version)
    )
