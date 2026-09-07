"""Client for the control socket.

`pa send "..."` uses this to reach a running daemon. It streams the response,
renders tool calls and tokens as they arrive, and answers approval prompts the
daemon forwards - so a request to the background daemon feels the same as a
turn typed into the foreground REPL.
"""

from __future__ import annotations

import socket
from typing import Any

from . import daemon, protocol
from .errors import PAError


class DaemonUnavailable(PAError):
    """No daemon is listening."""


def _connect(timeout: float = 5.0) -> socket.socket:
    path = daemon.socket_path()
    if not path.exists():
        raise DaemonUnavailable(
            "no daemon running. Start one with:  pa --serve   "
            "(or install it as a service: pa --install-service)"
        )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(path))
    except OSError as exc:
        raise DaemonUnavailable(f"daemon socket present but not accepting: {exc}") from exc
    sock.settimeout(None)
    return sock


def status() -> dict[str, Any]:
    sock = _connect()
    try:
        sock.sendall(protocol.encode({"op": "status"}))
        for frame in protocol.read_lines(sock):
            if frame.get("type") == "status":
                return frame
        return {}
    finally:
        sock.close()


def stop() -> bool:
    sock = _connect()
    try:
        sock.sendall(protocol.encode({"op": "stop"}))
        for frame in protocol.read_lines(sock):
            if frame.get("type") in ("stopping", "done"):
                return True
        return False
    finally:
        sock.close()


def interrupt() -> bool:
    sock = _connect()
    try:
        sock.sendall(protocol.encode({"op": "interrupt"}))
        for frame in protocol.read_lines(sock):
            if frame.get("type") == "interrupted":
                return bool(frame.get("was_running"))
        return False
    finally:
        sock.close()


def send(text: str, *, ui, interactive: bool = True, speak=None) -> bool:
    """Submit a prompt to the daemon and render the streamed reply.

    Returns False if the turn reported an error. `ui` is the same UI object the
    REPL uses, so rendering is identical; `speak`, if given, reads the final
    answer aloud.
    """
    sock = _connect()
    ok = True
    try:
        sock.sendall(protocol.encode({
            "op": "prompt", "text": text, "interactive": interactive,
        }))
        streaming = False
        for frame in protocol.read_lines(sock):
            kind = frame.get("type")
            if kind == "queued":
                ui.info("daemon busy - queued behind the current turn")
            elif kind == "token":
                streaming = True
                ui.stream(frame.get("text", ""))
            elif kind == "text":
                if streaming:
                    ui.end_stream(); streaming = False
                ui.markdown(frame.get("text", ""))
            elif kind == "thinking":
                if streaming:
                    ui.end_stream(); streaming = False
                ui.thinking(frame.get("text", ""))
            elif kind == "tool":
                if streaming:
                    ui.end_stream(); streaming = False
                ui.tool_call(frame.get("tool", ""), frame.get("args") or {})
            elif kind == "result":
                ui.tool_result(frame.get("tool", ""), frame.get("text", ""),
                               bool(frame.get("is_error")))
            elif kind == "usage":
                if streaming:
                    ui.end_stream(); streaming = False
                ui.usage(frame.get("text", ""))
            elif kind == "approval":
                self_answer = _answer_approval(sock, ui, frame)
            elif kind == "error":
                ok = False
                if streaming:
                    ui.end_stream(); streaming = False
                ui.error(frame.get("text", ""))
            elif kind == "done":
                if streaming:
                    ui.end_stream()
                if speak and frame.get("answer"):
                    speak(frame["answer"])
                break
        return ok
    finally:
        sock.close()


def _answer_approval(sock: socket.socket, ui, frame: dict[str, Any]) -> None:
    """The daemon is asking permission for a tool call; ask the user here."""
    decision, pattern = ui.ask_approval(frame.get("key", ""), frame.get("description", ""))
    sock.sendall(protocol.encode({
        "type": "approval_reply", "decision": decision, "pattern": pattern,
    }))
