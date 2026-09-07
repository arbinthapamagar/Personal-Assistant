"""Lazy dependency resolution.

The point: a fresh clone installs a small core, then pulls the heavy optional
packages (provider SDKs, Playwright, Pillow) the first time a feature is
actually used. Nobody downloads a browser driver to chat with a local model.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import subprocess
import sys
import sysconfig
from types import ModuleType

from .errors import DependencyError

# import name -> pip requirement, when the two differ.
PIP_NAMES = {
    "google.genai": "google-genai",
    "PIL": "pillow",
    "playwright.sync_api": "playwright",
    "yaml": "pyyaml",
}

_asked: set[str] = set()


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def externally_managed() -> bool:
    """True on Debian/Ubuntu system Python, where pip refuses to write (PEP 668)."""
    stdlib = sysconfig.get_paths().get("stdlib")
    return bool(stdlib) and (pathlib.Path(stdlib) / "EXTERNALLY-MANAGED").exists()


def pip_requirement(import_name: str) -> str:
    return PIP_NAMES.get(import_name, import_name.split(".")[0].replace("_", "-"))


def install(import_name: str, *, quiet: bool = False) -> None:
    """pip-install the package providing `import_name` into the running interpreter."""
    req = pip_requirement(import_name)
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", req]

    if not in_virtualenv():
        if externally_managed():
            raise DependencyError(
                f"cannot install {req!r}: this Python is externally managed (PEP 668) and "
                f"not a virtualenv.\nRun the bundled ./install.sh, which creates a venv, "
                f"or install manually:\n  python3 -m venv .venv && .venv/bin/pip install {req}"
            )
        cmd.insert(4, "--user")

    if not quiet:
        print(f"[pa] installing {req} ...", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise DependencyError(f"pip install {req} failed:\n" + "\n".join(tail))
    importlib.invalidate_caches()


def require(import_name: str, *, auto: bool = True, purpose: str = "") -> ModuleType:
    """Import a module, installing it on first use if allowed.

    Raises DependencyError with a copy-pasteable fix when auto-install is off
    or the install fails.
    """
    try:
        return importlib.import_module(import_name)
    except ImportError:
        pass

    req = pip_requirement(import_name)
    if not auto:
        hint = f" (needed for {purpose})" if purpose else ""
        raise DependencyError(
            f"missing dependency {req!r}{hint}. Install it with:\n"
            f"  {sys.executable} -m pip install {req}\n"
            f"or set auto_install_deps: true in your config."
        )

    install(import_name)
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:  # installed but still unimportable - bad wheel, name mismatch
        raise DependencyError(f"installed {req!r} but cannot import {import_name!r}: {exc}") from exc


def available(import_name: str) -> bool:
    """Is the module importable right now, without installing anything?"""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False
