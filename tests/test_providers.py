"""Provider translation tests.

The adapters are where a bug is quietest: a mistranslated tool result does not
raise, it just makes the model behave strangely. These tests pin the wire
shapes for the adapters that need no SDK to construct (Ollama and Gemini talk
raw HTTP), plus the pure helpers from the others.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa import config as config_mod
from pa.config import Profile
from pa.messages import Image, Message, Text, ToolCall, ToolResult, ToolSpec
from pa.providers.google_provider import GoogleProvider, _sanitize_schema
from pa.providers.ollama_provider import OllamaProvider
from pa.providers.openai_provider import _parse_args

SPEC = ToolSpec(
    "shell",
    "Run a command",
    {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
)

CONVERSATION = [
    Message.user("run echo hi"),
    Message("assistant", [Text("Sure."), ToolCall("c1", "shell", {"command": "echo hi"})]),
    Message("user", [ToolResult("c1", "hi", False)]),
]


def _cfg():
    cfg = config_mod.default_config()
    cfg.stream = False
    return cfg


# ---- Ollama ----------------------------------------------------------------


def test_ollama_wire_shape():
    cfg = _cfg()
    provider = OllamaProvider(Profile("local", "ollama"), cfg)
    wire = provider._to_wire("SYSTEM", CONVERSATION)

    assert wire[0] == {"role": "system", "content": "SYSTEM"}
    assert wire[1] == {"role": "user", "content": "run echo hi"}

    assistant = wire[2]
    assert assistant["role"] == "assistant"
    # Ollama takes arguments as a real object, not a JSON string.
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"command": "echo hi"}

    result = wire[3]
    assert result["role"] == "tool" and result["content"] == "hi"
    provider.close()


def test_ollama_parses_string_arguments_defensively():
    """Some builds send arguments as a JSON string; both must work."""
    cfg = _cfg()
    provider = OllamaProvider(Profile("local", "ollama"), cfg)
    parts = provider._parts_from_message(
        {"tool_calls": [{"function": {"name": "shell", "arguments": '{"command": "ls"}'}}]}
    )
    assert parts[0].arguments == {"command": "ls"}
    provider.close()


def test_ollama_tool_calls_set_a_tool_use_stop_reason():
    cfg = _cfg()
    provider = OllamaProvider(Profile("local", "ollama"), cfg)
    completion = provider._finish([ToolCall("c1", "shell", {})], {"done": True})
    assert completion.stop_reason == "tool_use"
    provider.close()


# ---- Gemini ----------------------------------------------------------------


def test_gemini_schema_sanitizer_strips_rejected_keywords():
    cleaned = _sanitize_schema(SPEC.parameters)
    assert "additionalProperties" not in cleaned
    assert cleaned["type"] == "object"
    assert cleaned["properties"]["command"]["type"] == "string"
    assert cleaned["required"] == ["command"]


def test_gemini_sanitizer_recurses_and_adds_missing_object_type():
    schema = {
        "properties": {
            "nested": {"properties": {"x": {"type": "string"}}, "$ref": "#/nope"},
        },
        "oneOf": [{"type": "string"}],
    }
    cleaned = _sanitize_schema(schema)
    assert "oneOf" not in cleaned
    assert cleaned["type"] == "object"                        # added
    assert cleaned["properties"]["nested"]["type"] == "object"  # added, recursively
    assert "$ref" not in cleaned["properties"]["nested"]


def test_gemini_wire_shape_and_result_keying():
    cfg = _cfg()
    provider = GoogleProvider(Profile("gemini", "google", api_key="test-key"), cfg)
    wire = provider._to_wire(CONVERSATION)

    assert wire[0]["role"] == "user"
    assert wire[1]["role"] == "model"                     # not "assistant"
    assert wire[1]["parts"][1]["functionCall"]["name"] == "shell"

    # Gemini matches results to calls by NAME, so the id must round-trip to it.
    response = wire[2]["parts"][0]["functionResponse"]
    assert response["name"] == "c1".rsplit(":", 1)[0]
    provider.close()


def test_gemini_synthesises_ids_that_survive_the_round_trip():
    """A parsed functionCall gets id 'name:index'; sending the result back must
    recover exactly 'name'."""
    cfg = _cfg()
    provider = GoogleProvider(Profile("gemini", "google", api_key="k"), cfg)
    completion = provider._from_wire(
        {
            "candidates": [
                {
                    "content": {"parts": [{"functionCall": {"name": "shell", "args": {"a": 1}}}]},
                    "finishReason": "STOP",
                }
            ]
        }
    )
    call = completion.message.tool_calls[0]
    assert call.id == "shell:0" and completion.stop_reason == "tool_use"

    wire = provider._to_wire([Message("user", [ToolResult(call.id, "out", False)])])
    assert wire[0]["parts"][0]["functionResponse"]["name"] == "shell"
    provider.close()


def test_gemini_safety_block_surfaces_as_refusal():
    cfg = _cfg()
    provider = GoogleProvider(Profile("gemini", "google", api_key="k"), cfg)
    completion = provider._from_wire(
        {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
    )
    assert completion.stop_reason == "refusal"
    assert "safety" in completion.message.text.lower()
    provider.close()


def test_gemini_requires_a_key():
    from pa.errors import AuthError

    cfg = _cfg()
    profile = Profile("gemini", "google", api_key_env="PA_DEFINITELY_UNSET_KEY")
    with pytest.raises(AuthError):
        GoogleProvider(profile, cfg)


# ---- OpenAI helpers --------------------------------------------------------


def test_openai_argument_parsing_never_raises():
    assert _parse_args('{"a": 1}') == {"a": 1}
    assert _parse_args("") == {}
    assert _parse_args("not json")["__unparsed__"] == "not json"
    assert _parse_args("[1,2]") == {"value": [1, 2]}   # non-object JSON is wrapped


# ---- config / profiles -----------------------------------------------------


def test_unknown_provider_is_rejected_at_config_time():
    from pa.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown provider"):
        Profile("x", "not-a-provider")


def test_openai_compat_demands_a_base_url():
    from pa.errors import ConfigError
    from pa.providers.openai_provider import OpenAICompatProvider

    cfg = _cfg()
    cfg.auto_install_deps = False
    with pytest.raises(ConfigError, match="base_url"):
        OpenAICompatProvider(Profile("groq", "openai-compat"), cfg)


def test_profile_defaults_fill_in_per_provider():
    assert Profile("a", "anthropic").model == "claude-opus-5"
    assert Profile("b", "ollama").base_url == "http://localhost:11434"
    assert Profile("c", "openai").api_key_env == "OPENAI_API_KEY"
