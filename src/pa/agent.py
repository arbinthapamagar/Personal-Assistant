"""The agent loop.

Provider-neutral by construction: it speaks only canonical Messages and
ToolSpecs, so switching from Claude to a local Ollama model changes nothing
here. Responsibilities are narrow - drive the request/tool/response cycle,
enforce the step budget, and turn tool failures into results the model can
recover from rather than exceptions that kill the turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Iterator

from .capabilities import Capabilities
from .concurrency import Job, Scheduler
from .config import Config
from .errors import PAError, PermissionDenied, ProviderError, RateLimitError, ToolError
from .harness import build_harness, looks_like_unparsed_tool_call
from .messages import Completion, Message, Text, ToolCall, ToolResult, Usage
from .prompts import build as build_prompt
from .providers.base import Provider
from .session import Session
from .skills import SkillLibrary
from .tools.base import Registry, ToolContext


@dataclass
class Step:
    """One observable thing the agent did, for the UI to render."""

    kind: str  # "text" | "thinking" | "tool" | "result" | "usage" | "error"
    text: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    is_error: bool = False


class Agent:
    def __init__(
        self,
        config: Config,
        provider: Provider,
        registry: Registry,
        ctx: ToolContext,
        caps: Capabilities,
        session: Session | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.registry = registry
        self.ctx = ctx
        self.caps = caps
        self.session = session or Session(
            profile=config.active_profile, model=provider.model
        )
        self.scheduler = Scheduler(max_workers=config.max_parallel_tools)
        # "auto" begins native and self-heals to prompted if the model writes
        # tool calls as text (the classic small-local-model failure).
        self.harness_mode = config.harness
        self._resolved_mode = "native" if config.harness == "auto" else config.harness
        self.harness = build_harness(self._resolved_mode)
        self.skills = SkillLibrary.load(ctx.cwd)
        # Make the loaded library available to use_skill without reloading.
        ctx.state.setdefault("skills", self.skills)
        ctx.state["session_id"] = self.session.id
        self._system = build_prompt(config, caps, registry.names(), self.skills)
        # Fallback chain: profile names still to try when the current model is
        # rate-limited. Consumed as we fall through them this session.
        self._fallback_remaining = [
            name for name in config.fallback if name in config.profiles
        ]
        #: Info steps queued by inner calls (e.g. a fallback switch) for turn()
        #: to surface to the UI.
        self._events: list[Step] = []

    # ---- one user turn ------------------------------------------------------

    def turn(
        self, user_input: str, on_text: Callable[[str], None] | None = None
    ) -> Iterator[Step]:
        """Run until the model stops calling tools. Yields Steps as they happen."""
        self.session.add(Message.user(user_input))

        # Auto-memory: recall relevant notes for this turn, capture new facts.
        self._recall_block = self._auto_recall(user_input)
        if self._recall_block:
            yield Step("usage", f"recalled {self._recall_block.count(chr(10))} memories")
        self._auto_capture(user_input)

        for step_index in range(self.config.max_steps):
            try:
                completion = self._complete(on_text)
            except ProviderError as exc:
                yield from self._drain_events()
                yield Step("error", str(exc), is_error=True)
                return
            except KeyboardInterrupt:
                yield Step("error", "interrupted", is_error=True)
                return

            # Surface any fallback switches that happened inside _complete.
            yield from self._drain_events()
            self.session.usage = self.session.usage + completion.usage
            self.session.add(completion.message)

            for part in completion.message.parts:
                if part.type == "thinking" and part.text:
                    yield Step("thinking", part.text)
            # Text was already streamed into on_text; only emit it when it was not.
            if on_text is None:
                for part in completion.message.parts:
                    if part.type == "text" and part.text:
                        yield Step("text", part.text)

            yield Step("usage", _usage_line(completion))

            calls = completion.message.tool_calls
            if not calls:
                self._persist()
                return

            # Announce every call before any of them runs, so the user can see
            # the whole batch that is about to execute concurrently.
            for call in calls:
                yield Step("tool", tool=call.name, args=call.arguments)

            yield from self._execute(calls)
            self._persist()
        else:
            note = (
                f"stopped after {self.config.max_steps} tool steps without finishing. "
                f"Raise max_steps in the config, or narrow the request."
            )
            self.session.add(Message.assistant(f"[{note}]"))
            yield Step("error", note, is_error=True)

    def _complete(self, on_text):
        """One provider call, routed through the active harness.

        In auto mode, watch the first native reply: if it is a tool call the
        model wrote as text rather than as a structured call, switch this
        session to the prompted harness and retry once. After that the mode is
        fixed for the session, so the cost is paid at most once."""
        tools = self.registry.specs(self.ctx, limit=self._tool_limit())
        # Recalled memory rides in the system prompt for this turn only, so it
        # never accumulates in the stored history. Placed after the stable
        # prompt so the recall block is the only part that varies per turn.
        base_system = self._system
        if getattr(self, "_recall_block", ""):
            base_system = f"{base_system}\n\n{self._recall_block}"
        # The live task plan rides in the prompt too, so the model always sees
        # its checklist and does not lose the thread on a long task.
        from .tools.plan import render_plan

        plan_block = render_plan(self.ctx)
        if plan_block:
            base_system = f"{base_system}\n\n{plan_block}"
        system, messages, provider_tools = self.harness.prepare(
            base_system, list(self.session.messages), tools
        )
        # Streaming tokens are meaningless once we may re-parse the whole text,
        # so only stream under the native harness.
        stream_sink = on_text if self._resolved_mode == "native" else None
        try:
            completion = self._provider_complete(
                system=system, messages=messages, tools=provider_tools, on_text=stream_sink
            )
        except ProviderError as exc:
            # Some local models (e.g. dolphin-mistral) have no native tool
            # support and the backend hard-rejects a request that carries
            # tools. That is a definitive "use the prompted protocol" signal -
            # switch and retry, rather than failing the turn.
            if self._should_switch_to_prompted(exc):
                self._switch_to_prompted()
                return self._complete_prompted(tools)
            raise
        completion = self.harness.interpret(completion)

        if (
            self.harness_mode == "auto"
            and self._resolved_mode == "native"
            and looks_like_unparsed_tool_call(completion)
        ):
            # Self-heal: this model wrote a tool call as text -> prompted.
            self._switch_to_prompted()
            return self._complete_prompted(tools)
        return completion

    def _should_switch_to_prompted(self, exc: ProviderError) -> bool:
        if self.harness_mode != "auto" or self._resolved_mode != "native":
            return False
        text = str(exc).lower()
        return "does not support tools" in text or "does not support insert" in text \
            or ("tool" in text and "support" in text)

    def _switch_to_prompted(self) -> None:
        self._resolved_mode = "prompted"
        self.harness = build_harness("prompted")

    def _complete_prompted(self, tools) -> Completion:
        """Re-run the request under the prompted harness (no native tools)."""
        base_system = self._system
        if getattr(self, "_recall_block", ""):
            base_system = f"{base_system}\n\n{self._recall_block}"
        from .tools.plan import render_plan

        plan_block = render_plan(self.ctx)
        if plan_block:
            base_system = f"{base_system}\n\n{plan_block}"
        system, messages, provider_tools = self.harness.prepare(
            base_system, list(self.session.messages), tools
        )
        completion = self._provider_complete(
            system=system, messages=messages, tools=provider_tools, on_text=None
        )
        return self.harness.interpret(completion)

    def _provider_complete(self, **kwargs) -> Completion:
        """Call the provider, falling through the fallback chain on a rate
        limit. The first model that answers becomes this session's provider, so
        a daily-capped free-tier model is left behind for good rather than
        retried on every turn."""
        while True:
            try:
                return self.provider.complete(**kwargs)
            except RateLimitError as exc:
                nxt = self._advance_fallback()
                if nxt is None:
                    raise
                self._events.append(Step(
                    "error",
                    f"{self.provider.model} is rate-limited/out of quota - "
                    f"switching to {nxt!r}",
                    is_error=True,
                ))
                self._switch_fallback(nxt)

    def _drain_events(self) -> Iterator[Step]:
        while self._events:
            yield self._events.pop(0)

    def _advance_fallback(self) -> str | None:
        while self._fallback_remaining:
            name = self._fallback_remaining.pop(0)
            if name != self.config.active_profile:
                return name
        return None

    def _switch_fallback(self, name: str) -> None:
        from .providers import build as build_provider

        profile = self.config.profiles[name]
        provider = build_provider(profile, self.config)
        self.provider.close()
        self.provider = provider
        self.config = self.config.with_profile(name)
        self.session.model = provider.model
        # A different model may need a different harness; let auto re-detect.
        if self.harness_mode == "auto":
            self._resolved_mode = "native"
            self.harness = build_harness("native")

    def _tool_limit(self) -> int:
        """How many tools to show. Explicit config wins; otherwise the prompted
        harness (a weak model) gets a sane cap so it is not overwhelmed."""
        if self.config.max_tools:
            return self.config.max_tools
        if self._resolved_mode == "prompted":
            return 14
        return 0

    # ---- automatic memory ---------------------------------------------------

    def _memory(self):
        """The shared Memory instance, or None if unavailable. Cached on ctx so
        the tools and auto-memory use the same store."""
        settings = self.config.auto_memory or {}
        if not settings.get("enabled", True):
            return None
        memory = self.ctx.state.get("memory")
        if memory is None:
            from .memory import Memory

            memory = Memory(auto_install=self.config.auto_install_deps)
            if not memory.available():
                # Do not trigger an install just for auto-recall; it is a bonus,
                # not a requirement. The explicit memory tools can still install.
                return None
            self.ctx.state["memory"] = memory
        return memory

    def _auto_recall(self, user_input: str) -> str:
        """Find notes relevant to this turn and format them for the prompt."""
        if len(user_input.strip()) < 8:
            return ""  # too short to match anything meaningfully
        memory = self._memory()
        if memory is None:
            return ""
        settings = self.config.auto_memory
        try:
            hits = memory.recall(user_input, k=int(settings.get("recall_k", 4)))
        except Exception:  # noqa: BLE001 - recall is best-effort, never fatal
            return ""
        floor = float(settings.get("min_score", 0.35))
        hits = [h for h in hits if h.score >= floor]
        from .automemory import format_recall

        return format_recall(hits)

    def _auto_capture(self, user_input: str) -> None:
        """Save durable facts the user explicitly stated this turn."""
        if not (self.config.auto_memory or {}).get("capture", True):
            return
        from .automemory import extract_memorable

        facts = extract_memorable(user_input)
        if not facts:
            return
        memory = self._memory()
        if memory is None:
            return
        for fact in facts:
            try:
                # Skip a near-duplicate already remembered, so repeating a
                # preference does not pile up copies.
                existing = memory.recall(fact.text, k=1)
                if existing and existing[0].score > 0.9:
                    continue
                memory.remember(fact.text, kind=fact.kind, tags=["auto"])
            except Exception:  # noqa: BLE001 - capture is best-effort
                continue

    # ---- tool execution -----------------------------------------------------

    def _execute(self, calls: list[ToolCall]) -> Iterator[Step]:
        """Run a batch of tool calls concurrently, yielding each as it lands.

        Results are collected by original index and appended to the transcript
        in that order: the model must see results in a deterministic order even
        though they finished in whatever order the work allowed.
        """
        jobs = [
            Job(
                index=index,
                label=call.name,
                run=partial(self._run_tool, call),
                affinity=self._affinity_of(call.name),
            )
            for index, call in enumerate(calls)
        ]

        results: list[ToolResult | None] = [None] * len(calls)
        try:
            for done in self.scheduler.run(jobs):
                call = calls[done.index]
                if done.ok:
                    result = done.value
                else:
                    # _run_tool catches ordinary failures itself, so reaching
                    # here means something escaped it - record it and continue,
                    # because every call still owes the model a result.
                    result = ToolResult(
                        call.id,
                        f"Error: {type(done.error).__name__}: {done.error}",
                        is_error=True,
                    )
                results[done.index] = result
                yield Step(
                    "result", text=result.content, tool=call.name, is_error=result.is_error
                )
        except KeyboardInterrupt:
            # Any call with no result yet still needs one, or the next request
            # is malformed: the API rejects a tool_use with no tool_result.
            for index, call in enumerate(calls):
                if results[index] is None:
                    results[index] = ToolResult(
                        call.id, "Cancelled: the user interrupted this turn.", is_error=True
                    )
            self.session.add(Message("user", [r for r in results if r]))
            raise

        # All results for one assistant turn go back in a single message -
        # splitting them teaches the model to stop calling tools in parallel.
        self.session.add(Message("user", [r for r in results if r is not None]))

    def _affinity_of(self, tool_name: str) -> str | None:
        try:
            return self.registry.get(tool_name).affinity
        except ToolError:
            return None  # unknown tool; _run_tool reports it properly

    def _run_tool(self, call: ToolCall) -> ToolResult:
        """Execute one call. Every failure becomes a result, never an exception:
        the model can read an error and choose another route."""
        try:
            tool = self.registry.get(call.name)
            output = tool(call.arguments, self.ctx)
            return ToolResult(call.id, output or "[no output]")
        except PermissionDenied as exc:
            return ToolResult(
                call.id,
                f"Denied: {exc}. Do not retry this or work around it - "
                f"tell the user what you needed and why.",
                is_error=True,
            )
        except ToolError as exc:
            return ToolResult(call.id, f"Error: {exc}", is_error=True)
        except KeyboardInterrupt:
            return ToolResult(call.id, "Cancelled by the user.", is_error=True)
        except PAError as exc:
            return ToolResult(call.id, f"Error: {exc}", is_error=True)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the session
            return ToolResult(
                call.id, f"Unexpected {type(exc).__name__}: {exc}", is_error=True
            )

    def _persist(self) -> None:
        try:
            self.session.save()
        except OSError:
            pass  # losing the transcript is not worth killing the turn over

    # ---- runtime controls ---------------------------------------------------

    def switch_profile(self, name: str, provider: Provider) -> None:
        """Change model mid-conversation. History carries over unchanged."""
        self.provider.close()
        self.provider = provider
        self.config = self.config.with_profile(name)
        self.session.profile = name
        self.session.model = provider.model
        # Machine facts did not change, but the tool list may differ per config.
        self._system = build_prompt(
            self.config, self.caps, self.registry.names(), self.skills
        )

    def close(self) -> None:
        """Release worker threads. Safe to call twice."""
        self.scheduler.shutdown(wait=False)

    def stats(self) -> str:
        usage = self.session.usage
        return (
            f"session {self.session.id}  profile {self.session.profile}  "
            f"model {self.provider.model}\n"
            f"messages {len(self.session.messages)}  "
            f"in {usage.input_tokens:,}  out {usage.output_tokens:,}  "
            f"cache-read {usage.cache_read_tokens:,}\n"
            f"parallelism up to {self.scheduler.max_workers} tools at once\n"
            f"harness {self._resolved_mode}" +
            (" (auto)" if self.harness_mode == "auto" else "")
        )


def _usage_line(completion: Completion) -> str:
    usage = completion.usage
    bits = [f"in {usage.input_tokens:,}", f"out {usage.output_tokens:,}"]
    if usage.cache_read_tokens:
        bits.append(f"cached {usage.cache_read_tokens:,}")
    return f"{completion.model}  {'  '.join(bits)}  [{completion.stop_reason}]"
