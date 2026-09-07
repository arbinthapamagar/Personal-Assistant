"""Tool harnesses: how tools are shown to a model and parsed back.

The Anthropic and OpenAI tool APIs are reliable on frontier models. Small local
models are not: give a 3B model seven tools and a long system prompt and it will
often dump tool calls as plain text instead of using the structured API, or
emit near-JSON garbage. A single fixed contract - "always use the native tools
field" - therefore fails exactly where we most want local support.

So the harness is pluggable:

* **NativeHarness** - trust the provider's tool API. What frontier models want.
* **PromptedHarness** - describe the tools in the system prompt, tell the model
  to answer with a small JSON object to call one, and parse that back out of
  the text with a forgiving reader. Works with *any* model, including ones with
  no tool API at all, and with weak models that have one but misuse it.

The agent's `auto` mode starts native and, the first time a model answers with
text that is obviously a tool call but carries no structured call, switches
that session to prompted and retries. The failure heals itself instead of
silently dropping the call.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from .messages import Completion, Message, Text, ToolCall, ToolResult, ToolSpec


class Harness:
    name = "native"

    def prepare(
        self, system: str, messages: list[Message], tools: Sequence[ToolSpec]
    ) -> tuple[str, list[Message], Sequence[ToolSpec]]:
        """Return (system, messages, tools) as they should go to the provider."""
        return system, messages, tools

    def interpret(self, completion: Completion) -> Completion:
        """Post-process a completion (e.g. parse tool calls out of text)."""
        return completion


class NativeHarness(Harness):
    """Pass tools through the provider's native API. No transformation."""

    name = "native"


# ---------------------------------------------------------------------------
# Prompted harness
# ---------------------------------------------------------------------------

_PROTOCOL = """\
You are an agent running on the user's computer. You ACT for the user - you do \
the work yourself with tools. You never tell the user to run something, never \
describe which tools they "could" use, and never list steps for them to follow. \
If a task needs a tool, you CALL it. If it needs several, you call them.

You cannot see files, run commands, or read the web except by calling a tool, \
so never guess or invent a result - call a tool to get it. Making up file \
names, command output, or search results is a serious error.

HOW TO CALL A TOOL - reply with ONE line that is exactly one JSON object and \
nothing else (no explanation before or after):
{{"tool": "<tool_name>", "args": {{ ... }}}}
Put one such object per line to call several at once. Only when you have all \
the information you need do you reply with a plain prose answer and NO JSON. In \
any one message: either call tools, or give the final answer - never both, and \
never describe a tool instead of calling it.

Example - the user says "check trengo". WRONG: explaining that they could use \
list_dir and task_list. RIGHT: actually call the tools, e.g.
{{"tool": "shell", "args": {{"command": "systemctl status trengo 2>&1 | head; pgrep -af trengo"}}}}
then read the result and answer.

Available tools:
{tools}
"""


def _tool_docs(tools: Sequence[ToolSpec]) -> str:
    lines = []
    for spec in tools:
        props = (spec.parameters or {}).get("properties", {}) or {}
        required = set((spec.parameters or {}).get("required", []) or [])
        params = []
        for name, meta in props.items():
            typ = meta.get("type", "any")
            mark = "" if name in required else "?"
            params.append(f"{name}{mark}:{typ}")
        sig = ", ".join(params)
        # One line each - a small model cannot hold paragraphs per tool in mind.
        desc = (spec.description or "").split(".")[0][:120]
        lines.append(f"- {spec.name}({sig}) - {desc}")
    return "\n".join(lines)


# Ordered so the most explicit shapes win. Each returns (name, args) or None.
def _coerce(obj: dict) -> tuple[str, dict] | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool") or obj.get("name") or obj.get("function")
    if isinstance(name, dict):  # {"function": {"name": ..., "arguments": ...}}
        args = name.get("arguments") or name.get("parameters") or {}
        name = name.get("name")
    else:
        args = obj.get("args")
        if args is None:
            args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters")
        if args is None:
            args = obj.get("input")
        if args is None:
            args = {}
    if not isinstance(name, str) or not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return (name, args if isinstance(args, dict) else {})


def _iter_json_objects(text: str):
    """Yield top-level {...} spans from text, tolerating prose around them.

    A brace counter rather than a regex, because tool arguments nest objects
    and a regex cannot match balanced braces. Strings are tracked so a brace
    inside a JSON string does not throw off the depth.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def parse_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Pull tool calls out of a text response.

    Returns (calls, leftover_text). `leftover_text` is what remains once the
    call objects are stripped - kept so a model that narrates alongside a call
    does not lose the narration entirely.
    """
    calls: list[ToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    # Strip code fences first; models love to wrap the JSON in ```.
    scan = text
    cursor = 0
    for span in _iter_json_objects(text):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        coerced = _coerce(obj)
        if coerced is None:
            continue
        name, args = coerced
        idx = text.find(span, cursor)
        if idx >= 0:
            consumed_spans.append((idx, idx + len(span)))
            cursor = idx + len(span)
        calls.append(ToolCall(f"{name}:{len(calls)}", name, args))

    if not calls:
        return [], text

    # Remove the consumed spans to recover any prose.
    leftover_parts = []
    prev = 0
    for lo, hi in consumed_spans:
        leftover_parts.append(text[prev:lo])
        prev = hi
    leftover_parts.append(text[prev:])
    leftover = re.sub(r"\n{3,}", "\n\n", "".join(leftover_parts)).strip()
    return calls, leftover


class PromptedHarness(Harness):
    """Describe tools in the prompt; parse tool calls from the reply text."""

    name = "prompted"

    def prepare(self, system, messages, tools):
        if tools:
            system = f"{system}\n\n{_PROTOCOL.format(tools=_tool_docs(tools))}"
        rewritten = [self._rewrite(message) for message in messages]
        # Do NOT pass native tools - we are driving the model by prompt alone.
        return system, rewritten, ()

    def _rewrite(self, message: Message) -> Message:
        """Turn structured tool_call / tool_result parts back into text, so a
        model that only understands prose still sees a coherent history."""
        needs = any(isinstance(p, (ToolCall, ToolResult)) for p in message.parts)
        if not needs:
            return message
        parts = []
        for part in message.parts:
            if isinstance(part, ToolCall):
                parts.append(Text(json.dumps({"tool": part.name, "args": part.arguments})))
            elif isinstance(part, ToolResult):
                tag = "ERROR" if part.is_error else "OBSERVATION"
                parts.append(Text(f"{tag} from {part.call_id.rsplit(':', 1)[0]}:\n{part.content}"))
            else:
                parts.append(part)
        # Tool results were carried on a user turn; keep that role.
        return Message(message.role, parts)

    def interpret(self, completion: Completion) -> Completion:
        # If the provider already produced structured calls (a capable local
        # model), leave them be.
        if any(isinstance(p, ToolCall) for p in completion.message.parts):
            return completion
        text = completion.message.text
        calls, leftover = parse_tool_calls(text)
        if not calls:
            return completion
        parts: list = []
        if leftover:
            parts.append(Text(leftover))
        parts.extend(calls)
        completion.message = Message("assistant", parts)
        completion.stop_reason = "tool_use"
        return completion


def looks_like_unparsed_tool_call(completion: Completion) -> bool:
    """Did a native-harness completion smell like a tool call the model wrote
    as text? The signal the agent's auto mode watches for before switching."""
    if any(isinstance(p, ToolCall) for p in completion.message.parts):
        return False
    text = completion.message.text
    if not text or "{" not in text:
        return False
    calls, _ = parse_tool_calls(text)
    return bool(calls)


_ANSWER_KEYS = ("text", "response", "answer", "content", "message", "output", "reply")


def unwrap_json_answer(text: str) -> str:
    """Some small models wrap their whole reply in JSON, e.g.
    {"text": "Hello"} or {"response": "..."}. Unwrap that to the bare text.

    Only unwraps when the object is *purely* a wrapped answer (all its keys are
    answer-ish), so genuine JSON data the user asked for is left untouched.
    """
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return text
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return text
    if not isinstance(obj, dict) or not obj:
        return text
    if not set(obj.keys()) <= set(_ANSWER_KEYS):
        return text  # has non-answer keys -> real data, leave it
    for key in _ANSWER_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return text


def build_harness(mode: str) -> Harness:
    if mode == "prompted":
        return PromptedHarness()
    return NativeHarness()
