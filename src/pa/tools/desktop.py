"""Desktop control: keyboard, mouse, windows, screenshots.

Backend choice is a platform fact, not a preference:

* X11      - xdotool, which can target a specific window.
* Wayland  - ydotool, which injects at the kernel level via /dev/uinput. It has
             no concept of windows, so it types into whatever has focus. That is
             a real limitation, not a bug: Wayland deliberately forbids one app
             from synthesising input into another.
* Win/mac  - pyautogui.

Screenshots go through a separate backend, because on Wayland the compositor
owns the screen: GNOME answers over its own D-Bus interface, wlroots
compositors answer `grim`, and X11 tools only ever see XWayland windows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .. import deps, paths
from ..errors import ToolError
from .base import Tool, ToolContext


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ToolError(f"{cmd[0]} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"{cmd[0]} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise ToolError(f"{cmd[0]} failed: {detail or f'exit {proc.returncode}'}")
    return proc.stdout


class DesktopTool(Tool):
    group = "desktop"
    requires = ("can_control_desktop",)

    def lock_key(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        # One physical keyboard and mouse. Concurrent input would interleave
        # characters and clicks.
        return "desktop-input"


class TypeTextTool(DesktopTool):
    name = "desktop_type"
    description = (
        "Type text into whatever window currently has keyboard focus. There is no "
        "targeting - focus the right window first (desktop_focus) and verify with "
        "desktop_screenshot if it matters."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "delay_ms": {"type": "integer", "description": "Per-keystroke delay. Default 12."},
        },
        "required": ["text"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"desktop.type:{(args.get('text') or '')[:60]}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        text = (args.get("text") or "")[:50]
        return f"type {text!r} into the focused window"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        text = args.get("text") or ""
        if not text:
            raise ToolError("no text to type")
        delay = str(int(args.get("delay_ms") or 12))
        backend = ctx.caps.desktop_backend
        if backend == "ydotool":
            _run(["ydotool", "type", "--key-delay", delay, text], timeout=60)
        elif backend == "xdotool":
            _run(["xdotool", "type", "--delay", delay, text], timeout=60)
        else:
            pyautogui = deps.require("pyautogui", auto=ctx.config.auto_install_deps,
                                     purpose="desktop control")
            pyautogui.typewrite(text, interval=int(delay) / 1000)
        return f"typed {len(text)} characters"


class KeyTool(DesktopTool):
    name = "desktop_key"
    description = (
        "Press a key combination, e.g. 'ctrl+c', 'alt+Tab', 'super', 'Return'. "
        "Combine modifiers with '+'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "keys": {"type": "string", "description": "Key combo such as 'ctrl+shift+t'."},
            "repeat": {"type": "integer", "description": "Times to repeat. Default 1."},
        },
        "required": ["keys"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"desktop.key:{args.get('keys', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"press {args.get('keys')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        combo = (args.get("keys") or "").strip()
        if not combo:
            raise ToolError("no key given")
        repeat = max(int(args.get("repeat") or 1), 1)
        backend = ctx.caps.desktop_backend
        if backend == "ydotool":
            # ydotool accepts named keys joined by '+', e.g. ctrl+alt+t
            _run(["ydotool", "key", "--repeat", str(repeat), combo])
        elif backend == "xdotool":
            for _ in range(repeat):
                _run(["xdotool", "key", "--clearmodifiers", combo])
        else:
            pyautogui = deps.require("pyautogui", auto=ctx.config.auto_install_deps,
                                     purpose="desktop control")
            for _ in range(repeat):
                pyautogui.hotkey(*combo.split("+"))
        return f"pressed {combo}" + (f" x{repeat}" if repeat > 1 else "")


class ClickTool(DesktopTool):
    name = "desktop_click"
    description = (
        "Move the mouse to absolute screen coordinates and click. Take a "
        "screenshot first to find the coordinates."
    )
    parameters = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "double": {"type": "boolean"},
        },
        "required": ["x", "y"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"desktop.click:{args.get('x')},{args.get('y')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"{'double-' if args.get('double') else ''}click at ({args.get('x')}, {args.get('y')})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        x, y = int(args["x"]), int(args["y"])
        button = args.get("button") or "left"
        backend = ctx.caps.desktop_backend

        if backend == "ydotool":
            # ydotool mousemove --absolute needs the compositor to honour it;
            # button codes: 0=left 1=right 2=middle, 0x40|n = click (down+up).
            codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
            _run(["ydotool", "mousemove", "--absolute", "-x", str(x), "-y", str(y)])
            time.sleep(0.08)
            _run(["ydotool", "click", codes[button]])
            if args.get("double"):
                time.sleep(0.05)
                _run(["ydotool", "click", codes[button]])
        elif backend == "xdotool":
            num = {"left": "1", "middle": "2", "right": "3"}[button]
            _run(["xdotool", "mousemove", str(x), str(y)])
            _run(["xdotool", "click", "--repeat", "2" if args.get("double") else "1", num])
        else:
            pyautogui = deps.require("pyautogui", auto=ctx.config.auto_install_deps,
                                     purpose="desktop control")
            pyautogui.click(x=x, y=y, button=button, clicks=2 if args.get("double") else 1)
        return f"clicked {button} at ({x}, {y})"


class WindowsTool(Tool):
    name = "desktop_windows"
    group = "desktop"
    description = "List open windows with their titles, so you can pick one to focus."
    parameters = {"type": "object", "properties": {}}

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "desktop.windows:list"

    def available(self, ctx: ToolContext) -> tuple[bool, str]:
        if shutil.which("wmctrl") or shutil.which("xdotool"):
            return True, ""
        return False, "desktop_windows needs wmctrl (sudo apt install wmctrl)"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        if shutil.which("wmctrl"):
            out = _run(["wmctrl", "-l"])
            rows = [ln for ln in out.splitlines() if ln.strip()]
            if not rows and ctx.caps.session == "wayland":
                return (
                    "no windows listed - wmctrl only sees XWayland windows, and native "
                    "Wayland apps are invisible to it. Use desktop_screenshot instead."
                )
            return "\n".join(rows) or "[no windows]"
        return _run(["xdotool", "search", "--name", ".", "getwindowname", "%@"])


class FocusTool(Tool):
    name = "desktop_focus"
    group = "desktop"
    description = "Bring a window to the front by (partial) title match, so typing lands in it."
    parameters = {
        "type": "object",
        "properties": {"title": {"type": "string", "description": "Substring of the window title."}},
        "required": ["title"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"desktop.focus:{args.get('title', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"focus window matching {args.get('title')!r}"

    def available(self, ctx: ToolContext) -> tuple[bool, str]:
        if shutil.which("wmctrl") or shutil.which("xdotool"):
            return True, ""
        return False, "desktop_focus needs wmctrl (sudo apt install wmctrl)"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        title = args["title"]
        if shutil.which("wmctrl"):
            try:
                _run(["wmctrl", "-a", title])
            except ToolError as exc:
                raise ToolError(
                    f"could not focus {title!r}: {exc}. On Wayland only XWayland windows "
                    f"can be focused this way."
                ) from exc
            time.sleep(0.3)
            return f"focused window matching {title!r}"
        _run(["xdotool", "search", "--name", title, "windowactivate"])
        return f"focused window matching {title!r}"


class ScreenshotTool(Tool):
    name = "desktop_screenshot"
    group = "desktop"
    description = (
        "Capture the whole screen to a PNG file and return its path. Use this to "
        "see what is on screen before clicking, and to verify what happened after."
    )
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Where to save the PNG."}},
    }
    requires = ("can_screenshot",)

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "desktop.screenshot:screen"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "take a screenshot of the whole screen"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        target = (
            Path(args["path"]).expanduser()
            if args.get("path")
            else paths.cache_dir() / f"screen-{int(time.time())}.png"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        backend = ctx.caps.screenshot_backend

        if backend == "gnome-screenshot":
            _run(["gnome-screenshot", "-f", str(target)])
        elif backend == "gnome-shell-dbus":
            _run([
                "gdbus", "call", "--session",
                "--dest", "org.gnome.Shell.Screenshot",
                "--object-path", "/org/gnome/Shell/Screenshot",
                "--method", "org.gnome.Shell.Screenshot.Screenshot",
                "false", "false", str(target),
            ])
        elif backend == "grim":
            _run(["grim", str(target)])
        elif backend == "maim":
            _run(["maim", str(target)])
        elif backend == "scrot":
            _run(["scrot", "-o", str(target)])
        elif backend == "import":
            _run(["import", "-window", "root", str(target)])
        elif backend == "pyautogui":
            pyautogui = deps.require("pyautogui", auto=ctx.config.auto_install_deps,
                                     purpose="screenshots")
            pyautogui.screenshot(str(target))
        else:
            raise ToolError("no screenshot backend available on this machine")

        if not target.exists() or target.stat().st_size == 0:
            raise ToolError(
                f"{backend} reported success but wrote no image to {target}. On Wayland this "
                f"usually means the compositor denied the capture."
            )
        if blank := _looks_blank(target):
            raise ToolError(
                f"{backend} wrote {target} but the image is {blank}. This is the classic "
                f"Wayland failure - the capture was denied and a black frame returned. "
                f"Do not describe the screen from this file. Install gnome-screenshot "
                f"(GNOME) or grim (wlroots), or ask the user what is on screen."
            )
        return f"saved {target} ({target.stat().st_size} bytes)"


def _looks_blank(path: Path) -> str | None:
    """Detect an all-black capture, the signature of a denied Wayland grab.

    Uses Pillow when it is already installed, otherwise ImageMagick's
    `identify`, otherwise gives up rather than installing anything just to
    check - a false negative here only costs a confusing screenshot.
    """
    try:
        from PIL import Image, ImageStat  # noqa: PLC0415 - optional, checked lazily

        with Image.open(path) as img:
            mean = sum(ImageStat.Stat(img.convert("L")).mean) / 1
        return "entirely black" if mean < 1.0 else None
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - an unreadable image is reported elsewhere
        return None

    if not shutil.which("identify"):
        return None
    try:
        out = subprocess.run(
            ["identify", "-format", "%[fx:mean]", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and float(out.stdout.strip() or 1) < 0.004:
            return "entirely black"
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    return None


TOOLS = [TypeTextTool, KeyTool, ClickTool, WindowsTool, FocusTool, ScreenshotTool]
