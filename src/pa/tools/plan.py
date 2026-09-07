"""A live task plan the agent keeps as it works.

Complex requests fall apart when a model holds the whole plan in its head - it
drifts, repeats steps, or forgets the last one. This is worse on small local
models. The plan tool externalises the checklist: the model writes the steps
once, marks them off as it goes, and the current plan is injected into every
turn's prompt so it can always see where it is. Same idea as a coding agent's
to-do list.

The plan lives in session state (not the model's context history), so it
survives a long turn without eating tokens on every step, and the agent can
revise it freely.
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from .base import Tool, ToolContext

_STATUS = {"todo": "[ ]", "doing": "[~]", "done": "[x]", "blocked": "[!]"}


def get_plan(ctx: ToolContext) -> list[dict[str, str]]:
    return ctx.state.setdefault("plan", [])


def render_plan(ctx: ToolContext) -> str:
    """The block injected into the prompt each turn. Empty when there's no plan."""
    plan = ctx.state.get("plan") or []
    if not plan:
        return ""
    lines = ["Your current plan (keep it updated with the plan tool; mark steps "
             "done as you finish them):"]
    for i, step in enumerate(plan, 1):
        mark = _STATUS.get(step.get("status", "todo"), "[ ]")
        lines.append(f"  {mark} {i}. {step['text']}")
    return "\n".join(lines)


class PlanTool(Tool):
    name = "plan"
    group = "plan"
    description = (
        "Keep a task plan while you work on something multi-step. Set the steps "
        "at the start, then mark each done as you finish it - the plan stays "
        "visible to you every turn so you do not lose track or repeat work. "
        "Actions: 'set' (replace all steps, pass `steps`), 'check' (mark step "
        "`n` done), 'status' (set step `n` to a `status`), 'add' (append a "
        "`step`), 'show' (display), 'clear'. Use it for anything with more than "
        "about three steps."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "check", "status", "add", "show", "clear"],
            },
            "steps": {
                "type": "array", "items": {"type": "string"},
                "description": "For 'set': the full list of step descriptions.",
            },
            "step": {"type": "string", "description": "For 'add': one step to append."},
            "n": {"type": "integer", "description": "1-based step number for check/status."},
            "status": {
                "type": "string", "enum": ["todo", "doing", "done", "blocked"],
                "description": "For 'status': the new state of step n.",
            },
        },
        "required": ["action"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"plan.{args.get('action', 'show')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        plan = get_plan(ctx)
        action = args.get("action", "show")

        if action == "set":
            steps = args.get("steps") or []
            if not steps:
                raise ToolError("set needs a non-empty 'steps' list")
            plan.clear()
            plan.extend({"text": str(s), "status": "todo"} for s in steps)
        elif action == "add":
            if not args.get("step"):
                raise ToolError("add needs a 'step'")
            plan.append({"text": str(args["step"]), "status": "todo"})
        elif action in ("check", "status"):
            n = int(args.get("n") or 0)
            if not 1 <= n <= len(plan):
                raise ToolError(f"step {n} out of range (1-{len(plan)})")
            plan[n - 1]["status"] = "done" if action == "check" else (
                args.get("status") or "done")
        elif action == "clear":
            plan.clear()
            return "plan cleared"
        elif action != "show":
            raise ToolError(f"unknown action {action!r}")

        rendered = render_plan(ctx)
        done = sum(1 for s in plan if s.get("status") == "done")
        return (rendered + f"\n({done}/{len(plan)} done)") if plan else "no plan set"


TOOLS = [PlanTool]
