#!/usr/bin/env python3
"""
Ensures version consistency across:
- pyproject.toml
- llmbridge/__init__.py
- .bumpversion.cfg

Exit codes:
0 = versions match (after normalization)
1 = mismatch detected
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Extraction helpers
# -----------------------------


def extract_init_version() -> str:
    """Return __version__ from llmbridge/__init__.py."""
    text = (ROOT / "llmbridge" / "__init__.py").read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError("Version not found in llmbridge/__init__.py")
    return match.group(1)


def extract_pyproject_version() -> str:
    """Return version from pyproject.toml (regex-based, no tomllib)."""
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError("Version not found in pyproject.toml")
    return match.group(1)


def extract_cfg_version() -> str:
    """Return current_version from .bumpversion.cfg."""
    text = (ROOT / ".bumpversion.cfg").read_text()
    match = re.search(r"current_version\s*=\s*(.+)", text)
    if not match:
        raise RuntimeError("current_version not found in .bumpversion.cfg")
    return match.group(1).strip()


# -----------------------------
# Normalization
# -----------------------------


def normalize(version: str) -> str:
    """
    Normalize version for comparison.

    Examples:
    - "0.3.5-dev"   -> "0.3.5"
    - "0.3.5+local" -> "0.3.5"
    - "0.3.5"       -> "0.3.5"
    """
    core = version.split("-", 1)[0]
    core = core.split("+", 1)[0]
    return core.strip()


# -----------------------------
# Main logic
# -----------------------------


def main() -> int:
    init_v = extract_init_version()
    proj_v = extract_pyproject_version()
    cfg_v = extract_cfg_version()

    n_init = normalize(init_v)
    n_proj = normalize(proj_v)
    n_cfg = normalize(cfg_v)

    if n_init == n_proj == n_cfg:
        print(f"Version OK: {init_v}")
        return 0

    print("Version mismatch detected:")
    print(f"  __init__.py:        {init_v}  (normalized: {n_init})")
    print(f"  pyproject.toml:     {proj_v}  (normalized: {n_proj})")
    print(f"  .bumpversion.cfg:   {cfg_v}  (normalized: {n_cfg})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
