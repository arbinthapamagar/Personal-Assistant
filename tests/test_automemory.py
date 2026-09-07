"""Automatic-memory tests: extraction precision and the agent recall/capture
loop. Precision is the thing that matters most - a false capture pollutes
recall for every future turn - so the negative cases are as important as the
positive ones.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa.automemory import Captured, extract_memorable, format_recall


# ---- extraction: it captures real statements -------------------------------


@pytest.mark.parametrize("text,expect_substr", [
    ("My name is Arbin", "Arbin"),
    ("call me Arbin please", "Arbin"),
    ("Remember that I deploy on Fridays", "Fridays"),
    ("I prefer verbose test output", "verbose"),
    ("I use Ubuntu 24.04 with Wayland", "Ubuntu"),
    ("For future reference, the key is in the vault", "vault"),
    ("Note that the build breaks on Python 3.9", "3.9"),
    ("I don't like tabs", "tabs"),
])
def test_captures_real_facts(text, expect_substr):
    facts = extract_memorable(text)
    assert facts, f"expected a capture from {text!r}"
    assert any(expect_substr.lower() in f.text.lower() for f in facts)


# ---- extraction: it ignores non-facts (precision) --------------------------


@pytest.mark.parametrize("text", [
    "What is my name?",
    "Do you remember what I told you?",
    "How do I prefer to set this up?",
    "List the files in this directory",
    "I don't remember the password",
    "I can't remember if I saved it",
    "Can you note the current time?",
    "please run the tests",
])
def test_ignores_non_facts(text):
    assert extract_memorable(text) == []


def test_name_is_not_polluted_by_following_words():
    facts = extract_memorable("My name is Arbin and I use Wayland")
    # The normalized name fact captures only the name, not the trailing clause.
    normalized = [f.text for f in facts if f.text.startswith("The user's name is")]
    assert normalized == ["The user's name is Arbin."]


def test_compound_sentence_yields_both_facts():
    facts = extract_memorable("My name is Arbin and I prefer verbose output")
    joined = " ".join(f.text for f in facts).lower()
    assert "arbin" in joined and "verbose" in joined


def test_format_recall_is_empty_without_hits():
    assert format_recall([]) == ""


# ---- agent integration: recall + capture round trip -----------------------


class _Fake:
    """Minimal provider capturing the system prompt it was handed."""

    name = "fake"

    def __init__(self, profile, config):
        self.profile, self.config = profile, config
        self.systems: list[str] = []

    @property
    def model(self):
        return "fake"

    def complete(self, *, system, messages, tools=(), on_text=None):
        from pa.messages import Completion, Message, Text, Usage

        self.systems.append(system)
        return Completion(Message("assistant", [Text("ok")]), "end_turn", Usage(1, 1), "fake")

    def close(self):
        pass


@pytest.fixture
def isolated_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    # Skip if chromadb is not installed in this environment.
    from pa import deps

    if not deps.available("chromadb"):
        pytest.skip("chromadb not installed")


def _agent():
    from pa import capabilities, config as config_mod
    from pa.agent import Agent
    from pa.security import Gate
    from pa.tools.base import ToolContext, build_registry

    cfg = config_mod.default_config()
    cfg.active_profile = "local"
    cfg.security.mode = "allow"
    cfg.tools = ["shell", "memory"]
    cfg.auto_memory = {"enabled": True, "recall_k": 5, "min_score": 0.6, "capture": True}
    caps = capabilities.probe()
    ctx = ToolContext(cfg, caps, Gate(cfg.security), lambda k, d: True)
    provider = _Fake(cfg.profile, cfg)
    return Agent(cfg, provider, build_registry(cfg, caps), ctx, caps), provider


def test_capture_then_recall_across_sessions(isolated_memory):
    a1, _ = _agent()
    list(a1.turn("My name is Arbin and I prefer verbose test output."))
    a1.close()

    # A brand-new agent (new session) must recall the fact without being told.
    a2, provider = _agent()
    list(a2.turn("what output style do I like for tests?"))
    recalled = provider.systems[-1].lower()
    assert "verbose" in recalled
    assert "arbin" in recalled
    a2.close()


def test_recall_is_absent_when_nothing_matches(isolated_memory):
    a1, _ = _agent()
    list(a1.turn("My name is Arbin."))
    a1.close()

    a2, provider = _agent()
    list(a2.turn("what is the capital of France?"))
    # An unrelated question should not drag in the name at the score floor.
    block = provider.systems[-1]
    assert "Things you remember" not in block or "Arbin" not in block


def test_disabling_auto_memory_skips_it(isolated_memory):
    from pa import capabilities, config as config_mod
    from pa.agent import Agent
    from pa.security import Gate
    from pa.tools.base import ToolContext, build_registry

    cfg = config_mod.default_config()
    cfg.active_profile = "local"; cfg.security.mode = "allow"; cfg.tools = ["shell", "memory"]
    cfg.auto_memory = {"enabled": False}
    caps = capabilities.probe()
    ctx = ToolContext(cfg, caps, Gate(cfg.security), lambda k, d: True)
    provider = _Fake(cfg.profile, cfg)
    agent = Agent(cfg, provider, build_registry(cfg, caps), ctx, caps)
    list(agent.turn("My name is Arbin and I prefer verbose output."))
    # Nothing recalled, nothing captured.
    assert "Things you remember" not in provider.systems[-1]
    assert "memory" not in agent.ctx.state
    agent.close()


def test_lowercase_name_is_captured():
    facts = extract_memorable("my name is arbin")
    assert facts and facts[0].text == "The user's name is Arbin."
    # and it doesn't sweep in a trailing clause
    facts2 = extract_memorable("my name is arbin and i use wayland")
    names = [f.text for f in facts2 if f.text.startswith("The user's name")]
    assert names == ["The user's name is Arbin."]
