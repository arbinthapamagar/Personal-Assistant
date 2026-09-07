"""Conversation persistence.

Sessions are stored as JSON so a crashed or closed terminal does not lose the
thread, and so `pa --resume` can pick up where it left off. Thinking blocks are
dropped on save: they are bound to the model turn that produced them and are not
valid to replay into a later process.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .messages import Image, Message, Text, Thinking, ToolCall, ToolResult, Usage


def _part_to_json(part: Any) -> dict[str, Any] | None:
    if isinstance(part, Text):
        return {"t": "text", "text": part.text}
    if isinstance(part, ToolCall):
        return {"t": "call", "id": part.id, "name": part.name, "args": part.arguments}
    if isinstance(part, ToolResult):
        return {
            "t": "result",
            "id": part.call_id,
            "content": part.content,
            "error": part.is_error,
        }
    if isinstance(part, Image):
        return {"t": "image", "media_type": part.media_type, "bytes": len(part.data)}
    if isinstance(part, Thinking):
        return None  # not replayable across processes
    return None


def _part_from_json(raw: dict[str, Any]) -> Any:
    kind = raw.get("t")
    if kind == "text":
        return Text(raw.get("text", ""))
    if kind == "call":
        return ToolCall(raw["id"], raw["name"], raw.get("args") or {})
    if kind == "result":
        return ToolResult(raw["id"], raw.get("content", ""), bool(raw.get("error")))
    if kind == "image":
        # The bytes are not persisted; leave a placeholder so history stays coherent.
        return Text(f"[image dropped on reload: {raw.get('bytes', 0)} bytes]")
    return None


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.time)
    profile: str = ""
    model: str = ""
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def path(self) -> Path:
        return paths.sessions_dir() / f"{self.id}.json"

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def save(self) -> Path:
        paths.sessions_dir().mkdir(parents=True, exist_ok=True)
        payload = {
            "id": self.id,
            "started": self.started,
            "profile": self.profile,
            "model": self.model,
            "usage": self.usage.__dict__,
            "messages": [
                {
                    "role": m.role,
                    "parts": [p for p in (_part_to_json(x) for x in m.parts) if p],
                }
                for m in self.messages
            ],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self.path)  # atomic, so a kill mid-save cannot corrupt history
        return self.path

    @classmethod
    def load(cls, path: Path) -> Session:
        raw = json.loads(path.read_text())
        session = cls(
            id=raw.get("id", path.stem),
            started=raw.get("started", time.time()),
            profile=raw.get("profile", ""),
            model=raw.get("model", ""),
            usage=Usage(**raw.get("usage", {})),
        )
        for entry in raw.get("messages", []):
            parts = [p for p in (_part_from_json(x) for x in entry.get("parts", [])) if p]
            if parts:
                session.messages.append(Message(entry["role"], parts))
        return session

    @classmethod
    def latest(cls) -> Session | None:
        files = sorted(
            paths.sessions_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return cls.load(files[0]) if files else None

    @classmethod
    def listing(cls, limit: int = 20) -> list[tuple[str, str, int]]:
        """(id, when, message count), newest first."""
        rows = []
        for file in sorted(
            paths.sessions_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:limit]:
            try:
                raw = json.loads(file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(raw.get("started", 0)))
            rows.append((raw.get("id", file.stem), when, len(raw.get("messages", []))))
        return rows
