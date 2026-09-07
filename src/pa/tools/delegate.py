"""Sub-agents.

`delegate` runs one or more short-lived agents in parallel, each with its own
fresh conversation, each returning a summary. It is how wide work gets done
without flooding the main conversation's context with material needed only once
- reviewing six files, checking four hypotheses, researching several topics at
the same time.

Two deliberate constraints:

* A sub-agent inherits a **read-mostly** tool set. It can read, search, fetch,
  and run read-only shell, but the destructive and stateful tools are withheld:
  a sub-agent cannot answer an approval prompt (there is no terminal in front
  of it), so anything that would prompt must not be reachable. Mutations stay
  with the main agent, where you are watching.
* Sub-agents run on the profile's model but can be pointed at a cheaper one via
  config, because fan-out is exactly where a smaller model pays off.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from ..config import Config
from ..errors import ToolError
from .base import Tool, ToolContext

# Tools a sub-agent may use. Everything here is safe without a human watching:
# read-only, or self-contained. Notably absent: write_file, edit_file, the
# desktop and browser groups, task_* (stateful), and anything that prompts.
SUBAGENT_TOOLS = {
    "shell", "cwd", "read_file", "list_dir", "search",
    "web_search", "web_fetch", "code_search", "memory_search", "use_skill",
}
# Shell commands a sub-agent may run: read-only inspection only. A sub-agent
# cannot answer an approval prompt, so its shell is allow-listed to commands
# that change nothing.
SUBAGENT_SHELL_ALLOW = [
    "shell:ls*", "shell:cat*", "shell:head*", "shell:tail*", "shell:grep*",
    "shell:rg*", "shell:find*", "shell:git log*", "shell:git status*",
    "shell:git show*", "shell:git diff*", "shell:wc*", "shell:file*",
    "shell:tree*", "shell:stat*", "shell:which*", "shell:pwd*", "shell:echo*",
]


class DelegateTool(Tool):
    name = "delegate"
    group = "agents"
    description = (
        "Run one or more sub-agents in parallel, each on its own task, and get "
        "back their findings. Use this for work that fans out - reviewing "
        "several files, investigating several leads, researching several topics "
        "at once - especially when reading it all yourself would bury the main "
        "thread. Each sub-agent starts fresh with no memory of this "
        "conversation, so give it a complete, self-contained brief and say "
        "exactly what to report back. Sub-agents can read, search, fetch, and "
        "run read-only commands; they cannot make changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "brief": {
                            "type": "string",
                            "description": "The complete, standalone instruction for this sub-agent.",
                        },
                        "label": {"type": "string", "description": "Short name for its output."},
                    },
                    "required": ["brief"],
                },
                "description": "One entry per sub-agent; they run concurrently.",
            },
        },
        "required": ["tasks"],
    }

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"agents.delegate:{len(args.get('tasks') or [])}"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        tasks = args.get("tasks") or []
        return f"run {len(tasks)} sub-agent(s) in parallel"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        tasks = args.get("tasks") or []
        if not tasks:
            raise ToolError("no tasks given")
        if len(tasks) > 8:
            raise ToolError("at most 8 sub-agents at once; batch the rest in a later call")

        briefs = []
        for index, task in enumerate(tasks):
            brief = (task.get("brief") or "").strip()
            if not brief:
                raise ToolError(f"task {index} has an empty brief")
            briefs.append((task.get("label") or f"agent-{index + 1}", brief))

        # Guard against unbounded recursion: a sub-agent's own delegate is
        # disabled (see _subagent_config), so depth is at most one, but this
        # keeps the invariant explicit.
        if ctx.state.get("is_subagent"):
            raise ToolError("sub-agents cannot themselves delegate")

        max_parallel = min(len(briefs), ctx.config.max_parallel_tools)
        with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="pa-subagent") as pool:
            futures = {
                pool.submit(self._run_one, label, brief, ctx): label
                for label, brief in briefs
            }
            outputs: dict[str, str] = {}
            for future in futures:
                label = futures[future]
                try:
                    outputs[label] = future.result()
                except Exception as exc:  # noqa: BLE001 - one failure must not sink the batch
                    outputs[label] = f"[sub-agent failed: {type(exc).__name__}: {exc}]"

        blocks = [f"## {label}\n{outputs[label]}" for label, _ in briefs]
        return ctx.truncate("\n\n".join(blocks))

    def _run_one(self, label: str, brief: str, ctx: ToolContext) -> str:
        # Imported here, not at module top: agent.py imports the tool registry,
        # and importing it back at load time would be circular.
        from ..agent import Agent
        from ..providers import build as build_provider
        from ..security import Gate
        from .base import ToolContext as Ctx, build_registry

        config = self._subagent_config(ctx.config)
        provider = build_provider(config.profile, config)
        try:
            # A gate that answers every prompt with "no", so a sub-agent that
            # reaches for something outside its allow-list is refused rather
            # than hanging on an unanswerable question.
            gate = Gate(config.security)
            sub_ctx = Ctx(
                config=config,
                caps=ctx.caps,
                gate=gate,
                approve=lambda key, desc: False,
                cwd=ctx.cwd,
                store=None,
            )
            sub_ctx.state["is_subagent"] = True
            registry = build_registry(config, ctx.caps)
            agent = Agent(config, provider, registry, sub_ctx, ctx.caps)
            agent.session.model = provider.model

            final = []
            for step in agent.turn(brief):
                if step.kind == "text":
                    final.append(step.text)
                elif step.kind == "error":
                    final.append(f"[{step.text}]")
            agent.close()
            return "\n".join(final).strip() or "[sub-agent produced no answer]"
        finally:
            provider.close()

    def _subagent_config(self, config: Config) -> Config:
        """A locked-down copy: read-mostly tools, no prompting, optional cheaper
        model."""
        sub = replace(
            config,
            tools=["shell", "files", "web", "skills", "memory", "agents"],
            max_steps=min(config.max_steps, 20),
            stream=False,
        )
        # Read-only posture: allow the safe shell verbs, deny everything that
        # would prompt, and set the mode so an unmatched call is refused rather
        # than asked.
        from ..config import Security

        sub.security = Security(
            mode="deny",
            allow=[
                *SUBAGENT_SHELL_ALLOW,
                "files.read:*", "files.list:*", "files.search:*",
                "web.search:*", "web.fetch:*",
                "memory.search:*", "skills.use:*", "cwd:read",
            ],
            deny=["agents.delegate:*"],
            workspace=config.security.workspace,
        )

        # Point sub-agents at a cheaper model if one is configured.
        worker = config.profile.extra.get("subagent_model")
        if worker:
            sub.profiles = dict(sub.profiles)
            sub.profiles[sub.active_profile] = replace(
                sub.profile, model=worker
            )
        return sub


TOOLS = [DelegateTool]
