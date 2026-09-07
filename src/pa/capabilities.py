"""Machine probe.

Run once at startup. Tools consult it to pick a backend, refuse cleanly, or
tell the user exactly which system package is missing - which is what makes the
same code behave sanely on a different PC.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache


def which(name: str) -> str | None:
    return shutil.which(name)


@dataclass
class Capabilities:
    os: str
    arch: str
    python: str
    session: str = "unknown"          # x11 | wayland | windows | macos | headless
    desktop_backend: str | None = None  # xdotool | ydotool | pyautogui | applescript | none
    screenshot_backend: str | None = None
    chrome_path: str | None = None
    has_ollama: bool = False
    ollama_running: bool = False
    ollama_models: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def can_control_desktop(self) -> bool:
        return self.desktop_backend is not None

    @property
    def can_screenshot(self) -> bool:
        return self.screenshot_backend is not None

    @property
    def can_drive_browser(self) -> bool:
        return self.chrome_path is not None

    def summary(self) -> str:
        rows = [
            f"os          {self.os} ({self.arch})",
            f"python      {self.python}",
            f"session     {self.session}",
            f"desktop     {self.desktop_backend or 'unavailable'}",
            f"screenshot  {self.screenshot_backend or 'unavailable'}",
            f"browser     {self.chrome_path or 'no chrome/chromium found'}",
            f"ollama      {self._ollama_line()}",
        ]
        return "\n".join(rows + [f"note: {n}" for n in self.notes])

    def _ollama_line(self) -> str:
        if not self.has_ollama:
            return "not installed"
        if not self.ollama_running:
            return "installed but not serving - run: ollama serve"
        if not self.ollama_models:
            return "serving, no models pulled - run: ollama pull qwen2.5-coder:14b"
        return f"serving - {', '.join(self.ollama_models[:5])}"


def _detect_session() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    stype = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if stype in ("wayland", "x11"):
        return stype
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "headless"


def _detect_chrome() -> str | None:
    candidates = {
        "linux": ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave-browser", "microsoft-edge"],
        "darwin": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                   "/Applications/Chromium.app/Contents/MacOS/Chromium"],
        "win32": [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"],
    }
    for cand in candidates.get(sys.platform, []):
        if cand.startswith("/") or cand[1:3] == ":\\":
            if os.path.exists(cand):
                return cand
        elif found := which(cand):
            return found
    return None


def _detect_desktop(session: str) -> tuple[str | None, list[str]]:
    """Pick an input-injection backend, and explain the choice when it's lossy."""
    notes: list[str] = []
    if session == "x11":
        if which("xdotool"):
            return "xdotool", notes
        notes.append("install xdotool for desktop control: sudo apt install xdotool wmctrl")
        return None, notes
    if session == "wayland":
        # Wayland forbids cross-app input injection; ydotool works via the kernel
        # uinput device, which needs permission set up once.
        if which("ydotool"):
            notes.append(
                "Wayland: using ydotool via /dev/uinput. If keystrokes silently do nothing, "
                "run: sudo systemctl enable --now ydotoold && sudo usermod -aG input $USER "
                "(then log out and back in)."
            )
            return "ydotool", notes
        if which("xdotool"):
            notes.append(
                "Wayland: only xdotool found - it can drive XWayland apps but not native "
                "Wayland ones. Install ydotool for full coverage: sudo apt install ydotool"
            )
            return "xdotool", notes
        notes.append("Wayland desktop control needs ydotool: sudo apt install ydotool")
        return None, notes
    if session in ("windows", "macos"):
        return "pyautogui", notes
    notes.append("no graphical session detected - desktop and browser tools are unavailable")
    return None, notes


def _detect_screenshot(session: str) -> str | None:
    """Pick a capture backend.

    On Wayland the compositor decides, so the desktop environment matters more
    than which binaries are installed. Two traps, both verified by hand on
    GNOME 4x / Ubuntu 24.04:

    * `grim` needs wlr-screencopy, which GNOME does not implement - it exits
      with an error.
    * X11 tools (`scrot`, `maim`, `import`) *appear* to succeed under XWayland
      and write a fully black image. That is far worse than failing, because a
      model will confidently describe an empty screen. Never select one on
      Wayland.
    * `org.gnome.Shell.Screenshot` over D-Bus is restricted to GNOME's own
      screenshot UI and answers AccessDenied for outside callers, so it sits
      last as a best-effort only.
    """
    if session == "wayland":
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
        if "GNOME" in desktop:
            order = ["gnome-screenshot", "gnome-shell-dbus"]
        elif "KDE" in desktop or "PLASMA" in desktop:
            order = ["spectacle", "grim"]
        else:
            order = ["grim", "gnome-screenshot", "spectacle"]
        for tool in order:
            if tool == "gnome-shell-dbus":
                if which("gdbus"):
                    return tool
            elif which(tool):
                return tool
        return None  # deliberately not falling back to an X11 tool
    if session == "x11":
        for tool in ("maim", "scrot", "import"):
            if which(tool):
                return tool
        return None
    if session in ("windows", "macos"):
        return "pyautogui"
    return None


def _detect_ollama() -> tuple[bool, bool, list[str]]:
    """(installed, serving, models). The daemon being down is the common case
    and deserves a different message than the binary being absent."""
    if not which("ollama"):
        return False, False, []
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, OSError):
        return True, False, []
    combined = (out.stdout + out.stderr).lower()
    # `ollama list` exits 0 even when the daemon is unreachable, so match the text.
    if "could not connect" in combined or "connection refused" in combined:
        return True, False, []
    models = [line.split()[0] for line in out.stdout.splitlines()[1:] if line.strip()]
    return True, True, models


@lru_cache(maxsize=1)
def probe() -> Capabilities:
    session = _detect_session()
    desktop, notes = _detect_desktop(session)
    has_ollama, ollama_running, models = _detect_ollama()
    screenshot = _detect_screenshot(session)
    if session == "wayland" and screenshot is None:
        notes.append(
            "no usable screenshot backend on this Wayland session. Install one: "
            "sudo apt install gnome-screenshot   (X11 tools like scrot silently "
            "capture a black screen under Wayland, so they are not used)"
        )
    caps = Capabilities(
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        python=platform.python_version(),
        session=session,
        desktop_backend=desktop,
        screenshot_backend=screenshot,
        chrome_path=_detect_chrome(),
        has_ollama=has_ollama,
        ollama_running=ollama_running,
        ollama_models=models,
        notes=notes,
    )
    return caps
