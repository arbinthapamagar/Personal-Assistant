"""Reflection tests: after a real task the agent distils a lesson to memory,
saves nothing on NONE, and is off unless enabled / enough work was done.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa import capabilities, config as config_mod, deps
from pa.agent import Agent
from pa.messages import Completion, Message, Text, ToolCall, Usage
from pa.providers.base import Provider
from pa.security import Gate
from pa.tools.base import ToolContext, build_registry


class Scripted(Provider):
    """Runs one shell tool, answers, then returns a scripted reflection."""

    name = "fake"

    def __init__(self, profile, config, lesson):
        super().__init__(profile, config)
        self._lesson = lesson
        self._phase = 0

    def complete(self, *, system, messages, tools=(), on_text=None):
        # A reflection call has no tools and a "distil" system prompt.
        if "distil" in (system or "").lower():
            return Completion(Message("assistant", [Text(self._lesson)]),
                              "end_turn", Usage(1, 1), "fake")
        self._phase += 1
        if self._phase == 1:
            return Completion(
                Message("assistant", [ToolCall("c1", "shell", {"command": "echo hi"})]),
                "tool_use", Usage(1, 1), "fake")
        return Completion(Message("assistant", [Text("Done.")]), "end_turn", Usage(1, 1), "fake")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    if not deps.available("chromadb"):
        pytest.skip("chromadb not installed")


def _agent(lesson, *, enabled=True, min_tools=1):
    cfg = config_mod.default_config()
    cfg.active_profile = "local"; cfg.security.mode = "allow"; cfg.tools = ["shell"]
    cfg.auto_memory = {"enabled": True, "recall_k": 4, "min_score": 0.6, "capture": False}
    cfg.reflect = {"enabled": enabled, "min_tools": min_tools}
    caps = capabilities.probe()
    ctx = ToolContext(cfg, caps, Gate(cfg.security), lambda k, d: True)
    provider = Scripted(cfg.profile, cfg, lesson)
    return Agent(cfg, provider, build_registry(cfg, caps), ctx, caps)


def test_saves_a_lesson_after_a_real_task():
    agent = _agent("On this machine, tests run with the venv python.")
    steps = list(agent.turn("run the tests"))
    assert any(s.kind == "usage" and "learned:" in s.text for s in steps)
    notes = agent.ctx.state["memory"].notes()
    assert any("venv python" in n.text for n in notes)


def test_none_saves_nothing():
    agent = _agent("NONE")
    list(agent.turn("run the tests"))
    notes = agent.ctx.state["memory"].notes()
    assert not any(n.metadata.get("kind") == "lesson" for n in notes)


def test_disabled_by_default_does_not_reflect():
    agent = _agent("a real lesson here", enabled=False)
    list(agent.turn("run the tests"))
    # memory backend was never even touched for a lesson
    mem = agent.ctx.state.get("memory")
    if mem is not None:
        assert not any(n.metadata.get("kind") == "lesson" for n in mem.notes())


def test_skips_when_too_little_work():
    # min_tools high -> a one-tool task should not trigger reflection
    agent = _agent("a lesson", enabled=True, min_tools=5)
    steps = list(agent.turn("run the tests"))
    assert not any("learned:" in s.text for s in steps if s.kind == "usage")
