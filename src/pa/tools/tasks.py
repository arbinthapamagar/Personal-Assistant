"""Tools for background tasks.

The agent's side of `pa.tasks`: start a long job, keep working, check back.
`task_start` returns in milliseconds no matter how long the work takes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..tasks import TaskManager
from .base import Tool, ToolContext


def _manager(ctx: ToolContext) -> TaskManager:
    if ctx.store is None:
        raise ToolError("background tasks need a state backend; none is attached")
    manager = ctx.state.get("tasks")
    if manager is None:
        manager = TaskManager(ctx.store)
        ctx.state["tasks"] = manager
    return manager


class TaskStartTool(Tool):
    name = "task_start"
    group = "tasks"
    description = (
        "Launch a shell command in the background and return its task id "
        "immediately. Use this for anything slow - builds, test suites, "
        "downloads, long scripts - then carry on and check it later with "
        "task_list or task_output. The task keeps running even if this "
        "conversation ends."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
            "label": {
                "type": "string",
                "description": "Short human-readable name, e.g. 'pytest suite'.",
            },
            "cwd": {"type": "string", "description": "Directory to run in."},
        },
        "required": ["command"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        # Same key shape as the foreground shell, so one allow-rule covers
        # both and the gate cannot be sidestepped by backgrounding a command.
        return f"shell:{args.get('command', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"run `{args.get('command', '')}` in the background"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        command = (args.get("command") or "").strip()
        if not command:
            raise ToolError("no command given")
        cwd = Path(args["cwd"]).expanduser() if args.get("cwd") else ctx.cwd
        if not cwd.is_dir():
            raise ToolError(f"working directory does not exist: {cwd}")
        task = _manager(ctx).start(command, cwd=cwd, label=args.get("label") or "")
        return (
            f"started task {task.id} (pid {task.pid})\n"
            f"check it with task_output({task.id!r}) or task_list()"
        )


class TaskListTool(Tool):
    name = "task_list"
    group = "tasks"
    description = "List background tasks with their status, runtime, and exit code."
    parameters = {
        "type": "object",
        "properties": {
            "running_only": {"type": "boolean", "description": "Hide finished tasks."},
        },
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "tasks.list:all"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        tasks = _manager(ctx).list(include_finished=not args.get("running_only"))
        if not tasks:
            return "no background tasks"
        return "\n".join(t.describe() for t in tasks)


class TaskOutputTool(Tool):
    name = "task_output"
    group = "tasks"
    description = (
        "Read a background task's output so far. Works while it is still "
        "running, so use it to follow progress."
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "tail": {"type": "integer", "description": "Last N lines. Default 200, 0 for all."},
        },
        "required": ["id"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"tasks.output:{args.get('id', '')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        manager = _manager(ctx)
        task = manager.require(args["id"])
        tail = args.get("tail")
        body = manager.output(task.id, tail=200 if tail is None else int(tail))
        return f"{task.describe()}\n\n{ctx.truncate(body)}"


class TaskWaitTool(Tool):
    name = "task_wait"
    group = "tasks"
    description = (
        "Block until the named background tasks finish, then report how each "
        "ended. Use it when you genuinely need the result before continuing; "
        "otherwise prefer checking task_list later."
    )
    parameters = {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Task ids to wait for.",
            },
            "timeout": {"type": "integer", "description": "Seconds to wait. Default 300."},
        },
        "required": ["ids"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"tasks.wait:{','.join(args.get('ids') or [])}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        ids = args.get("ids") or []
        if not ids:
            raise ToolError("no task ids given")
        timeout = min(max(int(args.get("timeout") or 300), 1), 3600)
        finished = _manager(ctx).wait(ids, timeout=timeout)
        still = [t for t in finished if t.status == "running"]
        lines = [t.describe() for t in finished]
        if still:
            lines.append(f"[{len(still)} still running after {timeout}s]")
        return "\n".join(lines)


class TaskCancelTool(Tool):
    name = "task_cancel"
    group = "tasks"
    description = "Stop a running background task and everything it started."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"tasks.cancel:{args.get('id', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"kill background task {args.get('id')} and its child processes"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        task = _manager(ctx).cancel(args["id"])
        return f"cancelled {task.id} after {task.elapsed:.1f}s"


TOOLS = [TaskStartTool, TaskListTool, TaskOutputTool, TaskWaitTool, TaskCancelTool]
