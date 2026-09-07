"""systemd user-service management.

Installs `pa --serve` as a user service so the daemon starts at login and
survives logout (with linger). Deliberately a *user* service, never system:
the agent runs as you, with your sessions and your permissions, and has no
business running as root or for other users.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import PAError

SERVICE_NAME = "arbin-assistant.service"


def _unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def _pa_binary() -> str:
    # Prefer the installed console script; fall back to `python -m pa`.
    found = shutil.which("pa")
    if found:
        return found
    return f"{sys.executable} -m pa"


def _unit_text() -> str:
    exec_start = f"{_pa_binary()} --serve --quiet"
    return f"""\
[Unit]
Description=arbin-assistant agent daemon
After=default.target

[Service]
Type=simple
ExecStart={exec_start}
Restart=on-failure
RestartSec=3
# Keep the model client and caches warm; do not kill on idle.
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    if not shutil.which("systemctl"):
        raise PAError("systemctl not found - this system does not use systemd")
    return subprocess.run(
        ["systemctl", "--user", *args], capture_output=True, text=True, check=False,
    )


def install() -> str:
    unit_dir = _unit_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / SERVICE_NAME
    unit_path.write_text(_unit_text())

    _systemctl("daemon-reload")
    enable = _systemctl("enable", "--now", SERVICE_NAME)
    if enable.returncode != 0:
        raise PAError(f"failed to enable the service:\n{enable.stderr.strip()}")

    # Offer to enable linger so it survives logout, but do not fail if we can't
    # (it needs a polkit prompt or root on some systems).
    linger_note = ""
    check = subprocess.run(
        ["loginctl", "show-user", os.environ.get("USER", ""), "--property=Linger"],
        capture_output=True, text=True,
    )
    if "Linger=yes" not in check.stdout:
        linger_note = (
            "\nTo keep it running after you log out:\n"
            f"  sudo loginctl enable-linger {os.environ.get('USER', '$USER')}"
        )

    return (
        f"installed and started {SERVICE_NAME}\n"
        f"  unit:   {unit_path}\n"
        f"  status: systemctl --user status {SERVICE_NAME}\n"
        f"  logs:   journalctl --user -u {SERVICE_NAME} -f"
        f"{linger_note}"
    )


def uninstall() -> str:
    _systemctl("disable", "--now", SERVICE_NAME)
    unit_path = _unit_dir() / SERVICE_NAME
    removed = unit_path.exists()
    unit_path.unlink(missing_ok=True)
    _systemctl("daemon-reload")
    return f"removed {SERVICE_NAME}" if removed else f"{SERVICE_NAME} was not installed"


def status() -> str:
    result = _systemctl("is-active", SERVICE_NAME)
    return result.stdout.strip() or result.stderr.strip() or "unknown"
