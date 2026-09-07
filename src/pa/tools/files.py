"""File tools.

Granular tools rather than one do-everything file tool: the model chooses more
accurately between read/write/edit/list/search when they are separate, and the
permission gate gets a meaningfully different key for each.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path
from typing import Any

from ..errors import ToolError
from .base import Tool, ToolContext

MAX_READ_BYTES = 400_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", "dist", "build"}


def _resolve(raw: str, ctx: ToolContext) -> Path:
    if not raw:
        raise ToolError("no path given")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ctx.cwd / path
    return path


def _path_lock(raw_path: str, ctx: ToolContext) -> str | None:
    """Exclusive key for a mutating call on one file.

    Concurrent edits to the same file lose data: both read the old text, both
    write, and the later write silently discards the earlier one.
    """
    try:
        return f"file:{_resolve(raw_path, ctx)}"
    except ToolError:
        return f"file:{raw_path}"


def _write_key(verb: str, raw_path: str, ctx: ToolContext) -> str:
    """Permission key for a mutating file call.

    A write outside the configured workspace gets its own key, so it is
    matched, prompted, and audited separately from an ordinary in-workspace
    write - and the user sees *why* they are being asked in the first prompt
    rather than a second one.
    """
    if not ctx.config.security.confirm_writes_outside_workspace:
        return f"files.{verb}:{raw_path}"
    try:
        path = _resolve(raw_path, ctx)
    except ToolError:
        return f"files.{verb}:{raw_path}"
    if ctx.inside_workspace(path):
        return f"files.{verb}:{path}"
    return f"files.write-outside-workspace:{path}"


def _write_summary(verb: str, raw_path: str, detail: str, ctx: ToolContext) -> str:
    try:
        path = _resolve(raw_path, ctx)
    except ToolError:
        return f"{verb} {raw_path}"
    if ctx.config.security.confirm_writes_outside_workspace and not ctx.inside_workspace(path):
        return f"{verb} {path} - OUTSIDE the workspace ({ctx.workspace}){detail}"
    return f"{verb} {path}{detail}"


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(4096)
    except OSError:
        return False


class ReadFileTool(Tool):
    name = "read_file"
    group = "files"
    description = (
        "Read a text file. Returns the content with 1-based line numbers. "
        "Use offset and limit for large files rather than reading everything."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to read."},
            "offset": {"type": "integer", "description": "First line to return (1-based)."},
            "limit": {"type": "integer", "description": "How many lines to return."},
        },
        "required": ["path"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"files.read:{args.get('path', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"read {args.get('path')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = _resolve(args["path"], ctx)
        if not path.exists():
            raise ToolError(f"no such file: {path}")
        if path.is_dir():
            raise ToolError(f"{path} is a directory - use list_dir")
        if _is_binary(path):
            return f"[binary file, {path.stat().st_size} bytes] {path}"
        if path.stat().st_size > MAX_READ_BYTES and not args.get("limit"):
            raise ToolError(
                f"{path} is {path.stat().st_size} bytes - pass offset/limit to read part of it"
            )

        lines = path.read_text(errors="replace").splitlines()
        start = max(int(args.get("offset") or 1), 1)
        count = int(args.get("limit") or len(lines))
        chunk = lines[start - 1 : start - 1 + count]
        width = len(str(start + len(chunk)))
        body = "\n".join(f"{start + i:>{width}}  {line}" for i, line in enumerate(chunk))
        tail = ""
        if start - 1 + count < len(lines):
            tail = f"\n\n[{len(lines) - (start - 1 + count)} more lines; total {len(lines)}]"
        return ctx.truncate(body + tail) or "[empty file]"


class WriteFileTool(Tool):
    name = "write_file"
    group = "files"
    description = (
        "Create a file or replace its entire content. For a small change to an "
        "existing file, prefer edit_file - it is safer and shows a smaller diff."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string", "description": "The complete new file content."},
        },
        "required": ["path", "content"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return _write_key("write", args.get("path", ""), ctx)

    def lock_key(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        return _path_lock(args.get("path", ""), ctx)

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        size = len(args.get("content") or "")
        return _write_summary("write", args.get("path", ""), f" ({size} bytes)", ctx)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = _resolve(args["path"], ctx)
        content = args.get("content") or ""
        existed = path.exists()
        if existed and path.is_dir():
            raise ToolError(f"{path} is a directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        verb = "overwrote" if existed else "created"
        return f"{verb} {path} ({len(content)} bytes, {content.count(chr(10)) + 1} lines)"


class EditFileTool(Tool):
    name = "edit_file"
    group = "files"
    description = (
        "Replace an exact string in a file. `old` must appear exactly once "
        "unless replace_all is true. Include surrounding context in `old` to "
        "make it unique."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Exact text to find, including indentation."},
            "new": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
        },
        "required": ["path", "old", "new"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return _write_key("edit", args.get("path", ""), ctx)

    def lock_key(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        return _path_lock(args.get("path", ""), ctx)

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        detail = f" (replace {len(args.get('old') or '')} bytes)"
        return _write_summary("edit", args.get("path", ""), detail, ctx)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = _resolve(args["path"], ctx)
        if not path.is_file():
            raise ToolError(f"no such file: {path}")
        old, new = args.get("old") or "", args.get("new") or ""
        if old == new:
            raise ToolError("old and new are identical")

        text = path.read_text(errors="replace")
        hits = text.count(old)
        if hits == 0:
            raise ToolError(f"text not found in {path} - read the file and match it exactly")
        if hits > 1 and not args.get("replace_all"):
            raise ToolError(
                f"text appears {hits} times in {path} - add context to make it unique, "
                f"or pass replace_all: true"
            )
        path.write_text(text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1))
        return f"edited {path} ({hits if args.get('replace_all') else 1} replacement(s))"


class ListDirTool(Tool):
    name = "list_dir"
    group = "files"
    description = "List a directory. Shows type, size, and name; skips noisy build directories."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list. Defaults to cwd."},
            "all": {"type": "boolean", "description": "Include dotfiles."},
        },
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"files.list:{args.get('path', '.')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = _resolve(args.get("path") or str(ctx.cwd), ctx)
        if not path.is_dir():
            raise ToolError(f"not a directory: {path}")
        rows = []
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError as exc:
            raise ToolError(f"permission denied: {path}") from exc
        for entry in entries:
            if entry.name.startswith(".") and not args.get("all"):
                continue
            if entry.is_dir():
                rows.append(f"dir   {'-':>9}  {entry.name}/")
            else:
                try:
                    rows.append(f"file  {entry.stat().st_size:>9}  {entry.name}")
                except OSError:
                    rows.append(f"file  {'?':>9}  {entry.name}")
        return ctx.truncate(f"{path}\n" + ("\n".join(rows) or "[empty]"))


class SearchTool(Tool):
    name = "search"
    group = "files"
    description = (
        "Search file contents for a regular expression, or find files by name "
        "glob. Uses ripgrep when installed and falls back to a pure-Python walk."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex to search for in file contents."},
            "name": {"type": "string", "description": "Filename glob, e.g. '*.py'. Use instead of pattern to find files."},
            "path": {"type": "string", "description": "Directory to search. Defaults to cwd."},
            "max_results": {"type": "integer", "description": "Default 100."},
        },
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"files.search:{args.get('path', '.')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        root = _resolve(args.get("path") or str(ctx.cwd), ctx)
        limit = int(args.get("max_results") or 100)
        pattern, name = args.get("pattern"), args.get("name")
        if not pattern and not name:
            raise ToolError("give either 'pattern' (content search) or 'name' (filename glob)")

        if name and not pattern:
            hits = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for filename in filenames:
                    if fnmatch.fnmatch(filename, name):
                        hits.append(str(Path(dirpath) / filename))
                        if len(hits) >= limit:
                            return ctx.truncate("\n".join(hits) + f"\n[stopped at {limit}]")
            return ctx.truncate("\n".join(hits) or f"no files matching {name!r} under {root}")

        from shutil import which

        if which("rg"):
            cmd = ["rg", "--line-number", "--no-heading", "--color=never", "-m", str(limit)]
            if name:
                cmd += ["--glob", name]
            cmd += [pattern, str(root)]
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if proc.returncode not in (0, 1):
                raise ToolError(f"ripgrep failed: {proc.stderr.strip()}")
            return ctx.truncate(proc.stdout.strip() or f"no matches for {pattern!r}")

        import re

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"bad regex {pattern!r}: {exc}") from exc
        hits = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                if name and not fnmatch.fnmatch(filename, name):
                    continue
                file = Path(dirpath) / filename
                if _is_binary(file):
                    continue
                try:
                    for lineno, line in enumerate(file.read_text(errors="replace").splitlines(), 1):
                        if regex.search(line):
                            hits.append(f"{file}:{lineno}:{line.strip()[:200]}")
                            if len(hits) >= limit:
                                return ctx.truncate("\n".join(hits) + f"\n[stopped at {limit}]")
                except OSError:
                    continue
        return ctx.truncate("\n".join(hits) or f"no matches for {pattern!r}")


TOOLS = [ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, SearchTool]
