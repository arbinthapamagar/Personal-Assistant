"""A scripted provider for local testing without any API key.

Enabled only when a profile explicitly selects provider 'mock'. It echoes the
user's text and, if the text starts with 'run:', emits a shell tool call - just
enough to exercise the daemon, the socket, and the approval bridge end to end.
Never used in normal operation.
"""

from __future__ import annotations

from typing import Sequence

from ..messages import Completion, Message, Text, ToolCall, Usage
from .base import Provider, TextSink


class MockProvider(Provider):
    name = "mock"
    sdk = None

    def __init__(self, profile, config) -> None:
        super().__init__(profile, config)
        self._pending_tool = False

    def complete(self, *, system, messages, tools=(), on_text=None):
        last = messages[-1] if messages else None
        # If we just got a tool result, wrap up.
        if last and any(p.type == "tool_result" for p in last.parts):
            text = "Done - the command ran."
            if on_text:
                on_text(text)
            return Completion(Message("assistant", [Text(text)]), "end_turn",
                              Usage(3, 3), self.model)

        user_text = last.text if last else ""
        if user_text.startswith("run:"):
            command = user_text[4:].strip() or "echo hello"
            return Completion(
                Message("assistant", [ToolCall("m1", "shell", {"command": command})]),
                "tool_use", Usage(3, 3), self.model,
            )
        reply = f"(mock) you said: {user_text}"
        if on_text:
            on_text(reply)
        return Completion(Message("assistant", [Text(reply)]), "end_turn",
                          Usage(3, 2), self.model)

    def check(self) -> str:
        return "ok - mock provider (no network)"
