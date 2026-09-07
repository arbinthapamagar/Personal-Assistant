"""XDG-correct locations, so the agent behaves the same on every machine."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "arbin-assistant"
OLD_APP = "personal-assistant"


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


def _migrate_from_old_name() -> None:
    """Carry over config/data written under the previous app name, once.

    The tool was renamed to arbin-assistant; a user who ran the old name should
    not silently lose their config, sessions, or memory. Move each old dir into
    place only if the new one does not exist yet.
    """
    import shutil

    for new, old in (
        (config_dir(), config_dir().with_name(OLD_APP)),
        (data_dir(), data_dir().with_name(OLD_APP)),
    ):
        if old.exists() and not new.exists():
            try:
                shutil.move(str(old), str(new))
            except OSError:
                pass


def ensure_dirs() -> None:
    _migrate_from_old_name()
    for d in (config_dir(), data_dir(), cache_dir(), sessions_dir()):
        d.mkdir(parents=True, exist_ok=True)
