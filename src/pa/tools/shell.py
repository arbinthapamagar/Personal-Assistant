"""Shell execution.

One tool, because the model already knows shell. The interesting parts are the
guardrails: a persistent working directory, a hard timeout, output truncation,
and a permission key that carries the actual command text so the gate can match
on it.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ..errors import ToolError
from .base import Tool, ToolContext


class ShellTool(Tool):
    name = "shell"
    group = "shell"
    description = (
        "Run a shell command on this machine and return its combined stdout and "
        "stderr. The working directory persists between calls. Use this for git, "
        "package managers, builds, system queries - anything you would type in a "
        "terminal. Prefer the file tools for reading and editing files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command line to run, as typed in a shell.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional directory to run in. Defaults to the persistent cwd.",
            },
            "timeout": {
                "type": "integer",
                "description": "Seconds before the command is killed. Default 120, max 900.",
            },
        },
        "required": ["command"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"shell:{args.get('command', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        cwd = args.get("cwd") or "."
        return f"run `{args.get('command', '')}` in {cwd}"

    def lock_key(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        # A bare `cd` mutates the shared cwd; everything else is independent
        # and runs in parallel.
        command = (args.get("command") or "").strip()
        return "shell-cwd" if command.startswith("cd ") else None

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        command = (args.get("command") or "").strip()
        if not command:
            raise ToolError("no command given")

        cwd = Path(args["cwd"]).expanduser() if args.get("cwd") else ctx.cwd
        if not cwd.is_dir():
            raise ToolError(f"working directory does not exist: {cwd}")

        timeout = min(int(args.get("timeout") or 120), 900)

        # `cd` alone would be lost when the subprocess exits, so handle it here
        # and let the change persist for later calls.
        if command.startswith("cd ") and "&&" not in command and ";" not in command:
            target = (cwd / Path(shlex.split(command[3:])[0]).expanduser()).resolve()
            if not target.is_dir():
                raise ToolError(f"not a directory: {target}")
            ctx.cwd = target
            return f"cwd is now {target}"

        env = dict(os.environ, PAGER="cat", GIT_PAGER="cat", TERM="dumb")
        try:
            proc = subprocess.run(
                ["bash", "-lc", command] if os.name != "nt" else command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                shell=(os.name == "nt"),
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command exceeded {timeout}s and was killed: {command}") from None
        except OSError as exc:
            raise ToolError(f"could not run command: {exc}") from exc

        body = (proc.stdout or "") + (proc.stderr or "")
        body = ctx.truncate(body.strip())
        status = "" if proc.returncode == 0 else f"\n[exit status {proc.returncode}]"
        return (body or "[no output]") + status


class CwdTool(Tool):
    name = "cwd"
    group = "shell"
    description = "Report the agent's current working directory and the configured workspace root."
    parameters = {"type": "object", "properties": {}}

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "cwd:read"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"cwd: {ctx.cwd}\nworkspace: {ctx.workspace}\nhome: {Path.home()}"


TOOLS = [ShellTool, CwdTool]
