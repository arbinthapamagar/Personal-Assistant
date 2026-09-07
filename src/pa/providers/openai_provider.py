"""OpenAI adapter, plus the generic OpenAI-compatible variant.

Uses Chat Completions rather than the Responses API on purpose: it is the shape
every compatible endpoint implements (Groq, Together, OpenRouter, vLLM,
LM Studio, llama.cpp), so one adapter covers OpenAI and everything wearing its
clothes. `openai-compat` is the same code with a required base_url.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .. import deps
from ..config import Config, Profile
from ..errors import AuthError, ConfigError, ProviderError, RateLimitError
from ..messages import (
    Completion,
    Image,
    Message,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from .base import Provider, TextSink

_STOP_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAIProvider(Provider):
    name = "openai"
    sdk = "openai"
    requires_base_url = False

    def __init__(self, profile: Profile, config: Config) -> None:
        super().__init__(profile, config)
        if self.requires_base_url and not profile.base_url:
            raise ConfigError(
                f"profile {profile.name!r}: provider 'openai-compat' needs a base_url "
                f"(e.g. https://api.groq.com/openai/v1)"
            )
        self._sdk = deps.require(
            "openai", auto=config.auto_install_deps, purpose="the OpenAI-compatible provider"
        )
        kwargs: dict[str, Any] = {}
        if key := profile.resolve_key():
            kwargs["api_key"] = key
        elif self.requires_base_url:
            # Local servers usually ignore the key but the SDK insists on one.
            kwargs["api_key"] = "not-needed"
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        self._client = self._sdk.OpenAI(**kwargs)

    # ---- translation --------------------------------------------------------

    def _to_wire(self, system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "system":
                wire.append({"role": "system", "content": msg.text})
                continue

            # Tool results are standalone `tool` messages here, not parts of a turn.
            results = [p for p in msg.parts if isinstance(p, ToolResult)]
            others = [p for p in msg.parts if not isinstance(p, ToolResult)]

            if others or not results:
                content: Any
                images = [p for p in others if isinstance(p, Image)]
                text = "\n".join(p.text for p in others if isinstance(p, Text) and p.text)
                if images:
                    import base64

                    content = [{"type": "text", "text": text}] if text else []
                    for img in images:
                        b64 = base64.standard_b64encode(img.data).decode()
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{img.media_type};base64,{b64}"},
                        })
                else:
                    content = text

                calls = [p for p in others if isinstance(p, ToolCall)]
                entry: dict[str, Any] = {"role": msg.role, "content": content or None}
                if calls:
                    entry["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in calls
                    ]
                if entry["content"] is not None or calls:
                    wire.append(entry)

            for res in results:
                wire.append({
                    "role": "tool",
                    "tool_call_id": res.call_id,
                    "content": res.content,
                })
        return wire

    def _params(self, system, messages, tools) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire(system, messages),
            "max_completion_tokens": self.profile.max_tokens,
        }
        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        if self.profile.temperature is not None:
            params["temperature"] = self.profile.temperature
        params.update(self.profile.extra.get("params", {}))
        return params

    # ---- request ------------------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        on_text: TextSink | None = None,
    ) -> Completion:
        params = self._params(system, messages, tools)
        try:
            if self.config.stream:
                return self._stream(params, on_text)
            response = self._client.chat.completions.create(**params)
            return self._from_choice(response)
        except Exception as exc:  # noqa: BLE001
            raise self._normalize(exc) from exc

    def _stream(self, params: dict[str, Any], on_text: TextSink | None) -> Completion:
        text_parts: list[str] = []
        reasoning: list[str] = []
        # tool calls arrive as indexed fragments; assemble by index.
        pending: dict[int, dict[str, str]] = {}
        finish = "stop"
        usage = Usage()
        model = self.model

        stream = self._client.chat.completions.create(
            **params, stream=True, stream_options={"include_usage": True}
        )
        for chunk in stream:
            if getattr(chunk, "model", None):
                model = chunk.model
            if chunk_usage := getattr(chunk, "usage", None):
                usage = Usage(
                    input_tokens=getattr(chunk_usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(chunk_usage, "completion_tokens", 0) or 0,
                )
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
                if on_text:
                    on_text(delta.content)
            # Some compatible servers expose chain-of-thought here.
            if chunk_reason := getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None):
                reasoning.append(chunk_reason)
            for frag in getattr(delta, "tool_calls", None) or []:
                slot = pending.setdefault(frag.index, {"id": "", "name": "", "args": ""})
                if frag.id:
                    slot["id"] = frag.id
                fn = getattr(frag, "function", None)
                if fn is not None:
                    if fn.name:
                        slot["name"] = fn.name
                    if fn.arguments:
                        slot["args"] += fn.arguments

        parts: list[Any] = []
        if reasoning:
            parts.append(Thinking("".join(reasoning)))
        if any(text_parts):
            parts.append(Text("".join(text_parts)))
        for index in sorted(pending):
            slot = pending[index]
            if slot["name"]:
                parts.append(
                    ToolCall(
                        slot["id"] or f"call_{index}",
                        slot["name"],
                        _parse_args(slot["args"]),
                    )
                )
        return Completion(
            message=Message("assistant", parts),
            stop_reason=_STOP_MAP.get(finish, "end_turn"),
            usage=usage,
            model=model,
        )

    def _from_choice(self, response: Any) -> Completion:
        choice = response.choices[0]
        msg = choice.message
        parts: list[Any] = []
        if getattr(msg, "content", None):
            parts.append(Text(msg.content))
        for call in getattr(msg, "tool_calls", None) or []:
            parts.append(
                ToolCall(call.id, call.function.name, _parse_args(call.function.arguments))
            )
        raw_usage = getattr(response, "usage", None)
        return Completion(
            message=Message("assistant", parts),
            stop_reason=_STOP_MAP.get(choice.finish_reason or "stop", "end_turn"),
            usage=Usage(
                input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            ),
            model=getattr(response, "model", self.model),
            raw=response,
        )

    def _normalize(self, exc: Exception) -> Exception:
        sdk = self._sdk
        if isinstance(exc, getattr(sdk, "AuthenticationError", ())):
            return AuthError(str(exc))
        if isinstance(exc, getattr(sdk, "RateLimitError", ())):
            return RateLimitError(str(exc))
        if isinstance(exc, getattr(sdk, "APIError", ())):
            return ProviderError(str(exc))
        return exc

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except Exception:  # noqa: BLE001
            return []

    def check(self) -> str:
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=1,
            )
            return f"ok - {self.model} reachable"
        except Exception as exc:  # noqa: BLE001
            return f"failed - {self._normalize(exc)}"

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


class OpenAICompatProvider(OpenAIProvider):
    """Any third-party endpoint speaking the OpenAI wire format."""

    name = "openai-compat"
    requires_base_url = True


def _parse_args(raw: str) -> dict[str, Any]:
    """Tool arguments arrive as a JSON *string*; never string-match on it."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__unparsed__": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
