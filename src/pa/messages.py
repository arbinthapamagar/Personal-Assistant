"""Canonical message types.

Every provider adapter translates between its own wire format and these types,
so the agent loop, the tools, and the transcript never learn provider details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Role = Literal["user", "assistant", "system"]


@dataclass
class Text:
    """A run of plain text."""

    text: str
    type: Literal["text"] = "text"


@dataclass
class Thinking:
    """Reasoning the model exposed. Replayed verbatim when the provider wants it."""

    text: str
    signature: str | None = None
    raw: Any = None
    type: Literal["thinking"] = "thinking"


@dataclass
class ToolCall:
    """A model's request to run a tool."""

    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolResult:
    """The outcome of a tool call, headed back to the model."""

    call_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass
class Image:
    """An image, as raw bytes plus its media type."""

    data: bytes
    media_type: str = "image/png"
    type: Literal["image"] = "image"


Part = Text | Thinking | ToolCall | ToolResult | Image


@dataclass
class Message:
    """One turn in the conversation."""

    role: Role
    parts: list[Part] = field(default_factory=list)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls("user", [Text(text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls("assistant", [Text(text)])

    @property
    def text(self) -> str:
        """All text parts joined - what a human reads."""
        return "\n".join(p.text for p in self.parts if isinstance(p, Text))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [p for p in self.parts if isinstance(p, ToolCall)]


@dataclass
class Usage:
    """Token accounting for one request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "error"]


@dataclass
class Completion:
    """A provider's normalized response to one request."""

    message: Message
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: Any = None


@dataclass
class ToolSpec:
    """A tool as the model sees it: a name, a description, and a JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {"name": self.name, "description": self.description, "parameters": self.parameters},
            sort_keys=True,
        )


def transcript_preview(messages: Sequence[Message], limit: int = 400) -> str:
    """A short human-readable dump of a conversation, for logs and debugging."""
    lines = []
    for m in messages:
        body = m.text or ", ".join(f"{c.name}(...)" for c in m.tool_calls) or "<non-text>"
        lines.append(f"{m.role}: {body[:limit]}")
    return "\n".join(lines)
