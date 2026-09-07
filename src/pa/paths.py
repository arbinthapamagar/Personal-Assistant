"""XDG-correct locations, so the agent behaves the same on every machine."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "personal-assistant"


def _xdg(var: str, default: str) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else Path.home() / default


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
        return base / APP
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support" / APP
    return _xdg("XDG_CONFIG_HOME", ".config") / APP


def data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        return base / APP
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support" / APP
    return _xdg("XDG_DATA_HOME", ".local/share") / APP


def cache_dir() -> Path:
    if sys.platform in ("win32", "darwin"):
        return data_dir() / "cache"
    return _xdg("XDG_CACHE_HOME", ".cache") / APP


def config_file() -> Path:
    return config_dir() / "config.yaml"


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def log_file() -> Path:
    return data_dir() / "pa.log"


def ensure_dirs() -> None:
    for d in (config_dir(), data_dir(), cache_dir(), sessions_dir()):
        d.mkdir(parents=True, exist_ok=True)
