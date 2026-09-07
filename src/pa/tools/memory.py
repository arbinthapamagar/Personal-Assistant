"""Memory and semantic code search tools.

`memory_save` / `memory_search` are how the agent carries knowledge between
sessions. `code_index` / `code_search` are how it finds things by meaning
rather than by exact string - complementary to `search`, which is regex.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..memory import Memory
from .base import Tool, ToolContext


def _memory(ctx: ToolContext) -> Memory:
    memory = ctx.state.get("memory")
    if memory is None:
        memory = Memory(auto_install=ctx.config.auto_install_deps)
        ctx.state["memory"] = memory
    return memory


class MemoryTool(Tool):
    group = "memory"
    # Embedding runs on the CPU and Chroma's client is not thread-safe, so all
    # memory work shares one thread.
    affinity = "memory"

    def lock_key(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        return "memory"


class MemorySaveTool(MemoryTool):
    name = "memory_save"
    description = (
        "Remember something for future sessions: a decision and its reason, how "
        "this machine is configured, a preference the user stated, a gotcha you "
        "hit. Save the durable insight, not a transcript - and not anything the "
        "code or git history already records."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The fact, in one or two sentences."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short labels for filtering, e.g. ['ubuntu', 'wayland'].",
            },
            "kind": {
                "type": "string",
                "enum": ["note", "preference", "machine", "project", "gotcha"],
                "description": "What sort of memory this is.",
            },
        },
        "required": ["text"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "memory.save:note"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"remember: {(args.get('text') or '')[:80]}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        note_id = _memory(ctx).remember(
            args["text"],
            tags=args.get("tags") or [],
            kind=args.get("kind") or "note",
            source=f"session:{ctx.state.get('session_id', '')}",
        )
        return f"saved as {note_id}"


class MemorySearchTool(MemoryTool):
    name = "memory_search"
    description = (
        "Search everything remembered from past sessions, by meaning rather "
        "than keyword. Worth doing before asking the user something they may "
        "have already told you."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "Default 6."},
            "kind": {"type": "string", "description": "Restrict to one kind of memory."},
        },
        "required": ["query"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "memory.search:notes"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        hits = _memory(ctx).recall(
            args["query"], k=int(args.get("limit") or 6), kind=args.get("kind")
        )
        if not hits:
            return "nothing remembered that matches"
        return "\n\n".join(
            f"[{h.score:.2f}] {h.cite()}  (id {h.id})\n{h.text}" for h in hits
        )


class MemoryForgetTool(MemoryTool):
    name = "memory_forget"
    description = "Delete a remembered note by id. Use when a memory turns out to be wrong."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"memory.forget:{args.get('id', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"delete memory {args.get('id')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        _memory(ctx).forget(args["id"])
        return f"forgot {args['id']}"


class CodeIndexTool(MemoryTool):
    name = "code_index"
    description = (
        "Build a semantic index of a project directory so code_search can find "
        "things by meaning. Run once per project, and again after large changes. "
        "Cheap to re-run: unchanged chunks overwrite themselves."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Project root. Defaults to cwd."},
            "suffixes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Limit to these extensions, e.g. ['.py', '.ts'].",
            },
        },
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"memory.index:{args.get('path', '.')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"index {args.get('path') or ctx.cwd} for semantic search"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        root = Path(args["path"]).expanduser() if args.get("path") else ctx.cwd
        if not root.is_dir():
            raise ToolError(f"not a directory: {root}")
        stats = _memory(ctx).index_files(root, suffixes=args.get("suffixes"))
        return (
            f"indexed {root}: {stats['files']} files, {stats['chunks']} chunks"
            + (f", {stats['skipped']} skipped (too large or unreadable)" if stats["skipped"] else "")
        )


class CodeSearchTool(MemoryTool):
    name = "code_search"
    description = (
        "Search indexed code by meaning - 'where do we validate permissions', "
        "'the retry logic' - and get back file paths with line ranges. Use the "
        "regex `search` tool instead when you know the exact string. Requires "
        "code_index to have been run."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "Default 8."},
            "path_prefix": {"type": "string", "description": "Only results under this path."},
        },
        "required": ["query"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "memory.search:code"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        hits = _memory(ctx).search_code(
            args["query"],
            k=int(args.get("limit") or 8),
            path_prefix=args.get("path_prefix") or "",
        )
        if not hits:
            return (
                "no matches. If this project has not been indexed yet, run "
                "code_index first."
            )
        blocks = []
        for hit in hits:
            body = hit.text if len(hit.text) < 1200 else hit.text[:1200] + "\n..."
            blocks.append(f"[{hit.score:.2f}] {hit.cite()}\n{body}")
        return ctx.truncate("\n\n".join(blocks))


TOOLS = [
    MemorySaveTool, MemorySearchTool, MemoryForgetTool, CodeIndexTool, CodeSearchTool,
]
