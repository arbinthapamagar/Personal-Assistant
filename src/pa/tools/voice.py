"""The `speak` tool.

Lets the agent choose to say something aloud - reading out an answer, giving a
heads-up when a long task finishes, confirming an action hands-free. Input
(listening) is driven by the CLI's voice mode, not by a tool, because the agent
does not decide when the user talks.
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from .base import Tool, ToolContext


class SpeakTool(Tool):
    name = "speak"
    group = "voice"
    # Audio playback is a single output device; two utterances at once would
    # talk over each other.
    affinity = "voice"

    description = (
        "Say something aloud through the computer's speakers. Use it to read "
        "back a short answer, announce that a background task finished, or "
        "confirm an action when the user is working hands-free. Keep it to a "
        "sentence or two - this is speech, not a document. The full text still "
        "goes in your normal reply as well."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "What to say. One or two sentences."},
        },
        "required": ["text"],
    }

    def lock_key(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        return "voice-output"

    def key(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "voice.speak:text"

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"say aloud: {(args.get('text') or '')[:60]}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        text = (args.get("text") or "").strip()
        if not text:
            raise ToolError("nothing to say")

        voice = ctx.state.get("voice")
        if voice is None:
            # Build one on demand, so `speak` works even outside voice mode.
            from ..voice import Voice

            voice = Voice(ctx.config)
            ctx.state["voice"] = voice
        try:
            voice.speak(text)
        except Exception as exc:  # noqa: BLE001 - normalised for the model
            raise ToolError(f"could not speak: {exc}") from exc
        return f"spoke {len(text)} characters aloud"


TOOLS = [SpeakTool]
