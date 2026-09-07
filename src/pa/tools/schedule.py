"""Tools to create and manage autonomous scheduled jobs.

The jobs are stored in the shared backend; the daemon's scheduler thread runs
whatever is due. Creating a schedule from a non-daemon session works fine - the
job is saved and the running daemon (or the next one started) picks it up.
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from ..scheduler import CadenceError, ScheduleStore, parse_cadence
from .base import Tool, ToolContext


def _store(ctx: ToolContext) -> ScheduleStore:
    if ctx.store is None:
        raise ToolError("scheduling needs a state backend; none is attached")
    return ScheduleStore(ctx.store)


class ScheduleAddTool(Tool):
    name = "schedule_add"
    group = "schedule"
    description = (
        "Schedule a prompt to run automatically on a cadence - the assistant "
        "will run it on its own (via the daemon) and save the result. Cadence "
        "examples: 'every 30m', 'every 2h', 'daily 09:00', 'once in 1h'. Use "
        "for recurring chores like a morning summary or a periodic check. The "
        "daemon must be running (arbin-assistant --serve or the service) for "
        "jobs to actually fire."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What to run each time."},
            "cadence": {"type": "string",
                        "description": "'every 30m' | 'every 2h' | 'daily 09:00' | 'once in 1h'"},
            "profile": {"type": "string", "description": "Optional model profile to run it on."},
        },
        "required": ["prompt", "cadence"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"schedule.add:{args.get('cadence', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"schedule {args.get('prompt','')[:40]!r} to run {args.get('cadence')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            raise ToolError("no prompt given")
        try:
            job = _store(ctx).add(prompt, args.get("cadence", ""),
                                  profile=args.get("profile") or "")
        except CadenceError as exc:
            raise ToolError(str(exc)) from exc
        running = _daemon_note(ctx)
        return f"scheduled job {job.id}: {parse_cadence(job.cadence).describe()}\n{running}"


class ScheduleListTool(Tool):
    name = "schedule_list"
    group = "schedule"
    description = "List scheduled jobs: their cadence, next run time, and last result."
    parameters = {"type": "object", "properties": {}}

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "schedule.list"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        jobs = _store(ctx).list()
        if not jobs:
            return "no scheduled jobs"
        return "\n".join(j.describe() for j in jobs)


class ScheduleCancelTool(Tool):
    name = "schedule_cancel"
    group = "schedule"
    description = "Cancel a scheduled job by id."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"schedule.cancel:{args.get('id', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"cancel scheduled job {args.get('id')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        if not _store(ctx).cancel(args.get("id", "")):
            raise ToolError(f"no scheduled job {args.get('id')!r}")
        return f"cancelled {args['id']}"


def _daemon_note(ctx: ToolContext) -> str:
    from .. import daemon

    if daemon.is_running():
        return "the running daemon will fire it on schedule."
    return ("note: no daemon is running, so it will not fire until you start "
            "one - arbin-assistant --serve (or --install-service).")


TOOLS = [ScheduleAddTool, ScheduleListTool, ScheduleCancelTool]
