"""Tool contract and registry.

A tool declares a JSON Schema for its arguments, a permission key derived from
those arguments, and a `run` that returns a string for the model. Tools never
prompt the user directly - they ask the context, which owns the gate and the
console. That keeps them testable and keeps approval logic in one place.
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..capabilities import Capabilities
from ..concurrency import KeyedLocks
from ..config import Config
from ..errors import PermissionDenied, ToolError
from ..messages import ToolSpec
from ..security import Gate

#: (key, description) -> True if the user approved.
Approver = Callable[[str, str], bool]


@dataclass
class ToolContext:
    """Everything a tool is allowed to reach."""

    config: Config
    caps: Capabilities
    gate: Gate
    approve: Approver
    #: The agent's working directory, mutable across calls (so `cd` sticks).
    cwd: Path = field(default_factory=Path.cwd)
    #: Scratch space shared between tools in one session (browser handles, etc.).
    state: dict[str, Any] = field(default_factory=dict)
    #: Mutual exclusion between concurrent tool calls, keyed by resource.
    locks: KeyedLocks = field(default_factory=KeyedLocks)
    #: Held only while an approval question is on screen. Two prompts printed
    #: at once are unreadable, so prompting is serialized even though the work
    #: that follows is not.
    approval_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Shared state backend, when one is attached (background tasks, memory).
    store: Any = None

    @property
    def workspace(self) -> Path:
        raw = self.config.security.workspace
        return Path(raw).expanduser().resolve() if raw else Path.home()

    def inside_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.workspace)
            return True
        except ValueError:
            return False

    def check(self, key: str, description: str) -> None:
        """Run `key` past policy, prompting if the policy says to ask."""
        decision = self.gate.decide(key)
        if decision.allowed:
            return
        if decision.verdict.value == "deny":
            raise PermissionDenied(f"{key} - {decision.reason}")
        with self.approval_lock:
            # Re-check under the lock: while this call waited its turn, the user
            # may have answered "always allow" to an identical prompt, and
            # asking the same question twice is a bug users notice.
            decision = self.gate.decide(key)
            if decision.allowed:
                return
            if decision.verdict.value == "deny":
                raise PermissionDenied(f"{key} - {decision.reason}")
            if not self.approve(key, description):
                raise PermissionDenied(f"{key} - declined by user")

    def truncate(self, text: str) -> str:
        limit = self.config.security.max_output_bytes
        if len(text) <= limit:
            return text
        half = limit // 2
        omitted = len(text) - 2 * half
        return f"{text[:half]}\n\n... [{omitted} bytes omitted] ...\n\n{text[-half:]}"


class Tool(abc.ABC):
    """One capability exposed to the model."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    #: Which tool group this belongs to, matched against config `tools.enabled`.
    group: str = ""
    #: Capability attributes that must be truthy on the probe, e.g. "can_screenshot".
    requires: Sequence[str] = ()
    #: Pin every call of this tool to one named thread. Required for
    #: thread-bound resources - Playwright's sync API raises if touched from a
    #: thread other than the one that created it.
    affinity: str | None = None

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    def available(self, ctx: ToolContext) -> tuple[bool, str]:
        """Can this tool run on this machine right now?"""
        for attr in self.requires:
            if not getattr(ctx.caps, attr, False):
                return False, f"{self.name}: machine lacks {attr}"
        return True, ""

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """The permission key for a specific call.

        Takes the context so a key can reflect circumstance, not just
        arguments - a write inside the workspace and one outside it are
        different keys, and so get different policy and a different prompt.
        """
        return self.name

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """One line shown to the user when asking for approval."""
        return f"{self.name}({', '.join(f'{k}={v!r}'[:60] for k, v in args.items())})"

    def lock_key(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        """Resource this call needs exclusively, or None if it is safe to run
        alongside anything. Two calls sharing a key never overlap."""
        return None

    @abc.abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Do the work. Return text for the model, or raise ToolError."""

    def __call__(self, args: dict[str, Any], ctx: ToolContext) -> str:
        ok, why = self.available(ctx)
        if not ok:
            raise ToolError(why)
        # Approval first, then the resource lock: prompting while holding a
        # contended lock would stall every other call behind a question.
        ctx.check(self.key(args, ctx), self.summary(args, ctx))
        key = self.lock_key(args, ctx)
        if key is None:
            return self.run(args, ctx)
        with ctx.locks.acquire(key):
            return self.run(args, ctx)


class Registry:
    """The set of tools active for a session."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if not tool.name:
            raise ToolError(f"{type(tool).__name__} has no name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            known = ", ".join(sorted(self._tools)) or "<none>"
            raise ToolError(f"unknown tool {name!r}; available: {known}")
        return self._tools[name]

    def specs(self, ctx: ToolContext, *, limit: int = 0) -> list[ToolSpec]:
        """Specs for tools this machine can actually run - never advertise a
        tool that will only fail, or the model wastes turns on it.

        `limit` caps how many are shown (0 = all). Insertion order puts the
        core groups first (shell, files, web ...), so a cap keeps the essential
        tools and drops the specialised ones - which is what a small local
        model needs to avoid drowning in choices."""
        out = []
        for tool in self._tools.values():
            ok, _ = tool.available(ctx)
            if ok:
                out.append(tool.spec())
                if limit and len(out) >= limit:
                    break
        return out

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())


# Groups whose import is cheap and dependency-free: loaded eagerly.
_LIGHT_GROUPS = {
    "shell": "shell", "files": "files", "web_fetch": "web",
    "search": "search", "research": "research", "skills": "skills",
    "tasks": "tasks", "agents": "delegate", "desktop": "desktop", "voice": "voice",
    "selfdev": "selfdev", "plan": "plan",
}
# Groups that pull a heavy optional dependency: imported only when enabled, so a
# machine that never uses them never imports (or installs) Playwright/Chroma.
_HEAVY_GROUPS = {
    "browser": "browser",   # Playwright
    "memory": "memory",     # Chroma
}
# "web" is a convenience alias enabling both fetch and search.
_ALIASES = {"web": ["web_fetch", "search", "research"]}


def build_registry(config: Config, caps: Capabilities) -> Registry:
    """Assemble the registry from the enabled groups, importing lazily.

    A group's module is imported only if that group is enabled, so optional
    heavy dependencies stay unimported on machines that do not use them.
    """
    import importlib

    registry = Registry()
    wanted: list[str] = []
    for name in config.tools:
        wanted.extend(_ALIASES.get(name, [name]))

    seen_modules: set[str] = set()
    for group in wanted:
        module_name = _LIGHT_GROUPS.get(group) or _HEAVY_GROUPS.get(group)
        if module_name is None or module_name in seen_modules:
            continue
        seen_modules.add(module_name)
        module = importlib.import_module(f"{__package__}.{module_name}")
        for factory in module.TOOLS:
            registry.add(factory())
    return registry
