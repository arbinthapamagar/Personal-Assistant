"""Harness tests: tool-call parsing and the prompted/native transforms.

These are the guardrails for local-model support. The parser cases are drawn
from shapes real local models actually emit (llama, qwen, and garbage), so a
regression here means a class of local model silently stops working.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pa.harness import (
    NativeHarness,
    PromptedHarness,
    build_harness,
    looks_like_unparsed_tool_call,
    parse_tool_calls,
)
from pa.messages import Completion, Message, Text, ToolCall, ToolResult, ToolSpec, Usage

SPEC = ToolSpec(
    "list_dir", "List a directory.",
    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


# ---- parser ----------------------------------------------------------------


def test_parses_our_format():
    calls, _ = parse_tool_calls('{"tool": "shell", "args": {"command": "ls"}}')
    assert len(calls) == 1 and calls[0].name == "shell"
    assert calls[0].arguments == {"command": "ls"}


def test_parses_name_parameters_shape():
    calls, _ = parse_tool_calls('{"name": "list_dir", "parameters": {"path": "."}}')
    assert calls[0].name == "list_dir" and calls[0].arguments == {"path": "."}


def test_parses_name_arguments_shape():
    calls, _ = parse_tool_calls('{"name": "shell", "arguments": {"command": "pwd"}}')
    assert calls[0].arguments == {"command": "pwd"}


def test_parses_nested_function_shape():
    calls, _ = parse_tool_calls('{"function": {"name": "s", "arguments": {"q": "x"}}}')
    assert calls[0].name == "s" and calls[0].arguments == {"q": "x"}


def test_handles_nested_braces_in_arguments():
    calls, _ = parse_tool_calls('{"tool": "write", "args": {"content": "{a:1}", "path": "f"}}')
    assert calls[0].arguments["content"] == "{a:1}"


def test_multiple_calls_on_separate_lines():
    calls, _ = parse_tool_calls('{"tool":"a","args":{}}\n{"tool":"b","args":{}}')
    assert [c.name for c in calls] == ["a", "b"]


def test_recovers_narration_alongside_a_call():
    calls, leftover = parse_tool_calls('I will list them.\n{"tool":"list_dir","args":{}}')
    assert len(calls) == 1
    assert "list them" in leftover


def test_plain_prose_is_not_a_call():
    calls, leftover = parse_tool_calls("The answer is 42. Nothing to call here.")
    assert calls == [] and "42" in leftover


def test_string_encoded_arguments_are_decoded():
    calls, _ = parse_tool_calls('{"name": "shell", "arguments": "{\\"command\\": \\"ls\\"}"}')
    assert calls[0].arguments == {"command": "ls"}


def test_the_exact_llama32_failure_output():
    """The literal blob llama3.2:3b produced when it failed the native path."""
    blob = (
        '{"name": "list_dir", "parameters": {"all": true, "path": "."}}\n'
        '{"name": "read_file", "parameters": {"limit": 0, "offset": 1, "path": "x"}}'
    )
    calls, _ = parse_tool_calls(blob)
    assert [c.name for c in calls] == ["list_dir", "read_file"]


# ---- prompted harness ------------------------------------------------------


def test_prompted_prepare_injects_protocol_and_drops_native_tools():
    harness = PromptedHarness()
    system, messages, tools = harness.prepare("BASE", [Message.user("hi")], [SPEC])
    assert "list_dir" in system and '"tool"' in system
    assert tools == ()  # native tools withheld - we drive by prompt


def test_prompted_rewrites_tool_history_to_text():
    harness = PromptedHarness()
    history = [
        Message.user("do it"),
        Message("assistant", [ToolCall("list_dir:0", "list_dir", {"path": "."})]),
        Message("user", [ToolResult("list_dir:0", "a.txt\nb.txt", False)]),
    ]
    _, rewritten, _ = harness.prepare("BASE", history, [SPEC])
    # The tool call and result survive as readable text, no structured parts.
    assert all(
        p.type == "text" for m in rewritten for p in m.parts
    )
    joined = "\n".join(m.text for m in rewritten)
    assert "list_dir" in joined and "OBSERVATION" in joined and "a.txt" in joined


def test_prompted_interpret_extracts_calls_from_text():
    harness = PromptedHarness()
    completion = Completion(
        Message("assistant", [Text('{"tool": "list_dir", "args": {"path": "/tmp"}}')]),
        "end_turn", Usage(), "local",
    )
    result = harness.interpret(completion)
    assert result.stop_reason == "tool_use"
    assert result.message.tool_calls[0].name == "list_dir"


def test_prompted_interpret_leaves_structured_calls_alone():
    """A capable local model that DID use the native path is not second-guessed."""
    harness = PromptedHarness()
    completion = Completion(
        Message("assistant", [ToolCall("c1", "shell", {"command": "ls"})]),
        "tool_use", Usage(), "local",
    )
    result = harness.interpret(completion)
    assert len(result.message.tool_calls) == 1


# ---- native harness + auto detection ---------------------------------------


def test_native_harness_is_passthrough():
    harness = NativeHarness()
    msgs = [Message.user("hi")]
    system, messages, tools = harness.prepare("BASE", msgs, [SPEC])
    assert system == "BASE" and messages == msgs and list(tools) == [SPEC]


def test_auto_detects_a_text_tool_call():
    text_call = Completion(
        Message("assistant", [Text('{"name": "list_dir", "parameters": {"path": "."}}')]),
        "end_turn", Usage(), "local",
    )
    assert looks_like_unparsed_tool_call(text_call) is True


def test_auto_ignores_a_real_answer():
    answer = Completion(
        Message("assistant", [Text("There are 5 files in the directory.")]),
        "end_turn", Usage(), "local",
    )
    assert looks_like_unparsed_tool_call(answer) is False


def test_auto_ignores_a_proper_structured_call():
    structured = Completion(
        Message("assistant", [ToolCall("c1", "list_dir", {"path": "."})]),
        "tool_use", Usage(), "local",
    )
    assert looks_like_unparsed_tool_call(structured) is False


def test_build_harness_selects_by_mode():
    assert build_harness("prompted").name == "prompted"
    assert build_harness("native").name == "native"
    assert build_harness("auto").name == "native"  # auto starts native


# ---- error-driven prompted switch (models with no native tool support) -----


def test_agent_switches_to_prompted_when_backend_rejects_tools(tmp_path):
    """A model like dolphin-mistral 400s on a request carrying tools. The agent
    must switch to the prompted protocol and retry, not fail the turn."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from pa import capabilities, config as config_mod
    from pa.agent import Agent
    from pa.errors import ProviderError
    from pa.messages import Completion, Message, Text, Usage
    from pa.providers.base import Provider
    from pa.security import Gate
    from pa.tools.base import ToolContext, build_registry

    class NoNativeTools(Provider):
        name = "fake"
        def __init__(self, profile, config):
            super().__init__(profile, config)
            self.calls = 0
        def complete(self, *, system, messages, tools=(), on_text=None):
            self.calls += 1
            # First call carries native tools -> reject like Ollama does.
            if tools:
                raise ProviderError("Ollama HTTP 400: dolphin-mistral does not support tools")
            # Prompted retry sends no native tools -> answer.
            return Completion(Message("assistant", [Text("hello from prompted")]),
                              "end_turn", Usage(1, 1), "dolphin")

    cfg = config_mod.default_config()
    cfg.active_profile = "local"; cfg.security.mode = "allow"
    cfg.tools = ["shell"]; cfg.harness = "auto"; cfg.auto_memory = {"enabled": False}
    caps = capabilities.probe()
    ctx = ToolContext(cfg, caps, Gate(cfg.security), lambda k, d: True)
    agent = Agent(cfg, NoNativeTools(cfg.profile, cfg), build_registry(cfg, caps), ctx, caps)

    out = [s for s in agent.turn("hi") if s.kind == "text"]
    assert any("prompted" in s.text for s in out)
    assert agent._resolved_mode == "prompted"   # healed


def test_prompted_reply_is_shown_even_when_streaming_is_on(tmp_path):
    """Regression: a prompted-harness model (e.g. dolphin) does not stream, so
    its text must be emitted as a step even when the caller passed on_text -
    otherwise the reply is generated and silently dropped."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from pa import capabilities, config as config_mod
    from pa.agent import Agent
    from pa.messages import Completion, Message, Text, Usage
    from pa.providers.base import Provider
    from pa.security import Gate
    from pa.tools.base import ToolContext, build_registry

    class PromptedProvider(Provider):
        name = "fake"
        def complete(self, *, system, messages, tools=(), on_text=None):
            # A prompted-mode reply: plain text, and it never calls on_text.
            return Completion(Message("assistant", [Text("hello from dolphin")]),
                              "end_turn", Usage(1, 1), "dolphin")

    cfg = config_mod.default_config()
    cfg.active_profile = "local"; cfg.security.mode = "allow"
    cfg.tools = ["shell"]; cfg.harness = "prompted"; cfg.auto_memory = {"enabled": False}
    caps = capabilities.probe()
    ctx = ToolContext(cfg, caps, Gate(cfg.security), lambda k, d: True)
    agent = Agent(cfg, PromptedProvider(cfg.profile, cfg), build_registry(cfg, caps), ctx, caps)

    # on_text IS provided (streaming interactive path) but must not swallow text.
    texts = [s.text for s in agent.turn("hi", on_text=lambda d: None) if s.kind == "text"]
    assert any("hello from dolphin" in t for t in texts)
