"""Daemon tests over a real Unix socket with a scripted provider.

The socket is redirected into a temp XDG_RUNTIME_DIR so the tests never touch a
real daemon, and a FakeProvider means no API key or network. The approval
bridge - the daemon forwarding a permission question to the client that
submitted the turn - is the piece most worth pinning, so it gets its own test.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa import capabilities, config as config_mod, protocol, store as store_mod
from pa.agent import Agent
from pa.daemon import Daemon, is_running, socket_path
from pa.messages import Completion, Message, Text, ToolCall, Usage
from pa.providers.base import Provider
from pa.security import Gate
from pa.tools.base import ToolContext, build_registry


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, script, profile, config):
        super().__init__(profile, config)
        self.script = list(script)

    def complete(self, *, system, messages, tools=(), on_text=None):
        parts, stop = self.script.pop(0)
        if on_text:
            for part in parts:
                if isinstance(part, Text):
                    on_text(part.text)
        return Completion(Message("assistant", list(parts)), stop, Usage(4, 2), "fake")


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """A running daemon on an isolated socket, torn down after the test."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    def make(script, *, mode="ask"):
        cfg = config_mod.default_config()
        cfg.active_profile = "local"
        cfg.security.mode = mode
        cfg.security.workspace = str(tmp_path)
        cfg.tools = ["shell", "files"]
        caps = capabilities.probe()
        store = store_mod.SqliteStore(tmp_path / "state.db")
        ctx = ToolContext(cfg, caps, Gate(cfg.security), lambda k, d: False, store=store)
        registry = build_registry(cfg, caps)
        agent = Agent(cfg, FakeProvider(script, cfg.profile, cfg), registry, ctx, caps)
        server = Daemon(agent)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        for _ in range(50):
            if is_running():
                break
            time.sleep(0.05)
        return server, store

    servers = []

    def factory(script, **kw):
        server, store = make(script, **kw)
        servers.append((server, store))
        return server

    yield factory

    for server, store in servers:
        _send_op({"op": "stop"})
        store.close()


def _connect() -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(socket_path()))
    return sock


def _send_op(request: dict) -> list[dict]:
    try:
        sock = _connect()
    except OSError:
        return []
    frames = []
    try:
        sock.sendall(protocol.encode(request))
        for frame in protocol.read_lines(sock):
            frames.append(frame)
            if frame.get("type") in ("done", "pong", "stopping", "interrupted"):
                break
    finally:
        sock.close()
    return frames


def test_ping_and_status(daemon):
    daemon([([Text("hi")], "end_turn")])
    assert is_running()
    frames = _send_op({"op": "status"})
    status = next(f for f in frames if f["type"] == "status")
    assert status["tools"] == 7  # shell,cwd + read,write,edit,list,search
    assert status["busy"] is False


def test_prompt_streams_answer(daemon):
    daemon([([Text("Hello from the daemon.")], "end_turn")])
    frames = _send_op({"op": "prompt", "text": "hi", "interactive": True})
    done = next(f for f in frames if f["type"] == "done")
    assert "daemon" in done["answer"]


def test_approval_is_forwarded_to_the_client(daemon, tmp_path):
    target = tmp_path / "x.txt"
    daemon([
        ([ToolCall("c1", "shell", {"command": f"echo hi > {target}"})], "tool_use"),
        ([Text("wrote it")], "end_turn"),
    ])
    sock = _connect()
    sock.sendall(protocol.encode({"op": "prompt", "text": "write", "interactive": True}))
    asked = None
    result_ok = False
    for frame in protocol.read_lines(sock):
        if frame["type"] == "approval":
            asked = frame["key"]
            sock.sendall(protocol.encode(
                {"type": "approval_reply", "decision": "once", "pattern": frame["key"]}
            ))
        elif frame["type"] == "result":
            result_ok = not frame.get("is_error")
        elif frame["type"] == "done":
            break
    sock.close()
    assert asked and asked.startswith("shell:")
    assert result_ok and target.exists()


def test_declined_approval_denies_the_tool(daemon):
    daemon([
        ([ToolCall("c1", "shell", {"command": "echo nope"})], "tool_use"),
        ([Text("understood")], "end_turn"),
    ])
    sock = _connect()
    sock.sendall(protocol.encode({"op": "prompt", "text": "run", "interactive": True}))
    denied = False
    for frame in protocol.read_lines(sock):
        if frame["type"] == "approval":
            sock.sendall(protocol.encode(
                {"type": "approval_reply", "decision": "skip", "pattern": frame["key"]}
            ))
        elif frame["type"] == "result" and frame.get("is_error"):
            denied = "declined" in frame["text"].lower() or "denied" in frame["text"].lower()
        elif frame["type"] == "done":
            break
    sock.close()
    assert denied


def test_non_interactive_turn_falls_back_to_policy(daemon):
    """A scheduled (non-interactive) turn cannot prompt, so an unmatched call
    is denied rather than silently run."""
    daemon([
        ([ToolCall("c1", "shell", {"command": "echo scheduled"})], "tool_use"),
        ([Text("done")], "end_turn"),
    ], mode="ask")
    frames = _send_op({"op": "prompt", "text": "run", "interactive": False})
    results = [f for f in frames if f["type"] == "result"]
    assert results and results[0]["is_error"]


def test_stop_shuts_down(daemon):
    daemon([([Text("hi")], "end_turn")])
    assert is_running()
    _send_op({"op": "stop"})
    for _ in range(40):
        if not is_running():
            break
        time.sleep(0.05)
    assert not is_running()
