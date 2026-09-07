"""The tool for loading a skill's full instructions on demand.

Descriptions live in the system prompt; this tool pulls in a body. Keeping the
mechanism a tool (rather than something the harness does invisibly) means the
model chooses to load a playbook the same way it chooses any other action, and
the loaded text is a normal tool result the conversation can see and reason
about.
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from ..skills import SkillLibrary
from .base import Tool, ToolContext


def _library(ctx: ToolContext) -> SkillLibrary:
    library = ctx.state.get("skills")
    if library is None:
        library = SkillLibrary.load(ctx.cwd)
        ctx.state["skills"] = library
    return library


class UseSkillTool(Tool):
    name = "use_skill"
    group = "skills"
    description = (
        "Load the full instructions for one of the available skills, listed in "
        "the system prompt. Do this at the start of a task that matches a "
        "skill's description - it returns a focused playbook for that kind of "
        "work. Follow the loaded instructions over your default approach."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The skill name, e.g. 'debugging'."},
        },
        "required": ["name"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"skills.use:{args.get('name', '')}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"load the {args.get('name')!r} skill"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        library = _library(ctx)
        skill = library.get(args.get("name", ""))
        if skill is None:
            known = ", ".join(library.names()) or "none installed"
            raise ToolError(f"no skill {args.get('name')!r}. Available: {known}")
        return f"# Skill: {skill.name}\n\n{skill.body}"


TOOLS = [UseSkillTool]
