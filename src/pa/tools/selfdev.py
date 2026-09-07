"""Self-development: the assistant works on its own source code.

The agent can already read and edit files, so it can technically edit itself.
What makes that *safe* is a checkpoint-and-verify loop, which is what these
tools add:

* `self_locate` - find its own installed source tree, so it never guesses.
* `self_checkpoint` - commit the current state to git (a rollback point).
* `self_test` - run its own test suite to prove a change did not break it.
* `self_rollback` - restore the last good checkpoint if a change went wrong.

The rule the `self-development` skill enforces: checkpoint, edit, test, and roll
back on failure. Git is the seatbelt - a self-edit that breaks the assistant is
always one command from undone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..errors import ToolError
from .base import Tool, ToolContext


def _source_root() -> Path | None:
    """The editable source tree, found from this module's own location. Only
    returned if it is a git working tree - self-editing without version control
    is not something to encourage."""
    here = Path(__file__).resolve()
    # .../src/pa/tools/selfdev.py -> project root is three parents above src/pa
    for parent in here.parents:
        if (parent / ".git").exists() and (parent / "src" / "pa").is_dir():
            return parent
    return None


def _git(root: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, timeout=timeout,
    )


class SelfLocateTool(Tool):
    name = "self_locate"
    group = "selfdev"
    description = (
        "Find your own source code on disk so you can read and modify yourself. "
        "Returns the project root, the package directory, and current git "
        "status. Use this before editing your own code."
    )
    parameters = {"type": "object", "properties": {}}

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "selfdev.locate"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        root = _source_root()
        if root is None:
            raise ToolError(
                "could not locate your own source as a git repo. You may be "
                "running from an installed wheel rather than an editable "
                "checkout, in which case self-modification is not available."
            )
        status = _git(root, "status", "--short")
        head = _git(root, "log", "-1", "--oneline")
        dirty = status.stdout.strip() or "(clean)"
        return (
            f"project root: {root}\n"
            f"package:      {root / 'src' / 'pa'}\n"
            f"tests:        {root / 'tests'}\n"
            f"HEAD:         {head.stdout.strip()}\n"
            f"uncommitted:\n{dirty}"
        )


class SelfCheckpointTool(Tool):
    name = "self_checkpoint"
    group = "selfdev"
    description = (
        "Commit your current source to git as a restore point before making "
        "risky changes to yourself. Always checkpoint before a self-edit."
    )
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string", "description": "Commit message."}},
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "selfdev.checkpoint"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "commit a self-development checkpoint"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        root = _source_root()
        if root is None:
            raise ToolError("no source repo found")
        message = args.get("message") or "self-development checkpoint"
        _git(root, "add", "-A")
        commit = _git(
            root, "-c", "user.name=arbin-assistant",
            "-c", "user.email=santosh@matat.io", "commit", "-m", message,
        )
        if commit.returncode != 0:
            if "nothing to commit" in (commit.stdout + commit.stderr):
                return "nothing to checkpoint - working tree already clean"
            raise ToolError(f"checkpoint failed: {commit.stderr.strip()}")
        head = _git(root, "log", "-1", "--oneline")
        return f"checkpoint saved: {head.stdout.strip()}"


class SelfTestTool(Tool):
    name = "self_test"
    group = "selfdev"
    description = (
        "Run your own test suite to verify a change to your code did not break "
        "you. Do this after every self-edit, before trusting the change."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Optional test path or -k filter."},
        },
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "selfdev.test"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "run your own test suite"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        root = _source_root()
        if root is None:
            raise ToolError("no source repo found")
        python = root / ".venv" / "bin" / "python"
        cmd = [str(python) if python.exists() else "python3", "-m", "pytest",
               "-q", "-p", "no:cacheprovider"]
        target = (args.get("target") or "").strip()
        if target:
            # A path (has a slash or is a .py file) runs that file - resolving a
            # bare filename into tests/ - anything else is a -k name filter.
            looks_path = "/" in target or target.endswith(".py")
            if looks_path:
                candidate = root / target
                if not candidate.exists() and (root / "tests" / target).exists():
                    candidate = root / "tests" / target
                cmd.append(str(candidate))
            else:
                cmd += ["-k", target]
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise ToolError("tests exceeded 5 minutes") from None
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        verdict = "PASSED" if proc.returncode == 0 else "FAILED"
        return f"[tests {verdict}]\n{tail}"


class SelfRollbackTool(Tool):
    name = "self_rollback"
    group = "selfdev"
    description = (
        "Undo uncommitted changes to your own code, restoring the last "
        "checkpoint. Use this when a self-edit broke your tests and you cannot "
        "quickly fix it - it is always safe."
    )
    parameters = {
        "type": "object",
        "properties": {
            "hard": {"type": "boolean",
                     "description": "Also discard the last commit, not just uncommitted work."},
        },
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "selfdev.rollback"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "roll back your own code to the last checkpoint"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        root = _source_root()
        if root is None:
            raise ToolError("no source repo found")
        _git(root, "reset", "--hard", "HEAD~1" if args.get("hard") else "HEAD")
        _git(root, "clean", "-fd", "src", "tests")
        head = _git(root, "log", "-1", "--oneline")
        return f"rolled back to: {head.stdout.strip()}\nRestart the assistant to load the restored code."


TOOLS = [SelfLocateTool, SelfCheckpointTool, SelfTestTool, SelfRollbackTool]
