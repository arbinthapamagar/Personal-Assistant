"""Claude adapter, built on the official `anthropic` SDK.

Model-feature handling is deliberately defensive: the user can type any model
string into their config, and the current API rejects some parameters on older
models (and *requires* omitting others on newer ones). So we resolve features
from the model id, then degrade and retry once if the server still objects.
That keeps a five-year-old model id working without a code change.
"""

from __future__ import annotations

from typing import Any, Sequence

from .. import deps
from ..config import Config, Profile
from ..errors import AuthError, ProviderError, RateLimitError
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

# Model families that take `thinking: {type: "adaptive"}` and `output_config.effort`.
# `budget_tokens` is rejected on these; older models need it instead.
_ADAPTIVE = (
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6",
    "claude-fable-5", "claude-mythos-5",
)
# Families where a policy decline can be rescued server-side by a fallback model.
_FALLBACK = ("claude-opus-5", "claude-fable-5", "claude-mythos-5")

_STOP_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
    "stop_sequence": "end_turn",
    "pause_turn": "tool_use",
}


def _dump(obj: Any) -> Any:
    """SDK content blocks are pydantic models; turn one back into a wire dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return obj


class AnthropicProvider(Provider):
    name = "anthropic"
    sdk = "anthropic"

    def __init__(self, profile: Profile, config: Config) -> None:
        super().__init__(profile, config)
        self._sdk = deps.require(
            "anthropic", auto=config.auto_install_deps, purpose="the Claude provider"
        )
        key = profile.resolve_key()
        # A bare client also picks up an `ant auth login` profile, so an unset
        # ANTHROPIC_API_KEY is not automatically an error.
        self._client = self._sdk.Anthropic(api_key=key) if key else self._sdk.Anthropic()
        self._degraded: set[str] = set()

    # ---- feature resolution -------------------------------------------------

    def _family(self) -> str:
        return self.model.lower()

    def _supports(self, feature: str) -> bool:
        if feature in self._degraded:
            return False
        model = self._family()
        if feature in ("adaptive_thinking", "effort"):
            return any(model.startswith(p) for p in _ADAPTIVE)
        if feature == "fallbacks":
            return any(model.startswith(p) for p in _FALLBACK)
        return False

    # ---- translation --------------------------------------------------------

    def _to_wire(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for msg in messages:
            blocks: list[Any] = []
            for part in msg.parts:
                if isinstance(part, Text):
                    if part.text:
                        blocks.append({"type": "text", "text": part.text})
                elif isinstance(part, Thinking):
                    # Replayed verbatim - the API validates these against the
                    # turn that produced them, so never reconstruct by hand.
                    if part.raw is not None:
                        blocks.append(_dump(part.raw))
                elif isinstance(part, ToolCall):
                    blocks.append({
                        "type": "tool_use",
                        "id": part.id,
                        "name": part.name,
                        "input": part.arguments,
                    })
                elif isinstance(part, ToolResult):
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": part.call_id,
                        "content": part.content,
                    }
                    if part.is_error:
                        block["is_error"] = True
                    blocks.append(block)
                elif isinstance(part, Image):
                    import base64

                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": base64.standard_b64encode(part.data).decode(),
                        },
                    })
            if not blocks:
                continue
            # tool_result blocks are carried by a user turn, per the API shape.
            role = "user" if msg.role == "user" else msg.role
            wire.append({"role": role, "content": blocks})
        return wire

    def _from_wire(self, response: Any) -> Completion:
        parts: list[Any] = []
        for block in response.content or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                parts.append(Text(block.text))
            elif kind == "thinking":
                parts.append(Thinking(getattr(block, "thinking", "") or "", raw=block))
            elif kind == "redacted_thinking":
                parts.append(Thinking("", raw=block))
            elif kind == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                parts.append(ToolCall(block.id, block.name, args))

        raw_usage = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
        )
        stop = _STOP_MAP.get(getattr(response, "stop_reason", "") or "", "end_turn")

        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            parts.append(Text(f"[declined by policy: {category}]"))

        return Completion(
            message=Message("assistant", parts),
            stop_reason=stop,
            usage=usage,
            model=getattr(response, "model", self.model),
            raw=response,
        )

    # ---- request ------------------------------------------------------------

    def _params(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]):
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.profile.max_tokens,
            "messages": self._to_wire(messages),
        }
        if system:
            # Cache the stable system prefix - it is identical on every turn.
            params["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        if self._supports("adaptive_thinking"):
            # "summarized" so the user sees reasoning in the terminal; the
            # default on these models is to return empty thinking blocks.
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self._supports("effort"):
            params["output_config"] = {"effort": self.profile.effort}
        elif self.profile.temperature is not None:
            params["temperature"] = self.profile.temperature

        betas: list[str] = []
        if self._supports("fallbacks"):
            # A policy decline otherwise just stops the turn; this re-runs the
            # same request on a suitable fallback model inside the same call.
            betas.append("server-side-fallback-2026-07-01")
            params["fallbacks"] = "default"
        return params, betas

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        on_text: TextSink | None = None,
    ) -> Completion:
        for attempt in (1, 2):
            params, betas = self._params(system, messages, tools)
            if betas:
                params["betas"] = betas
            target = self._client.beta.messages if betas else self._client.messages
            try:
                # Always stream: max_tokens here is large enough that a
                # non-streaming request can outlive the HTTP timeout.
                with target.stream(**params) as stream:
                    if on_text is not None:
                        for event in stream:
                            if getattr(event, "type", "") == "text" and getattr(event, "text", ""):
                                on_text(event.text)
                    response = stream.get_final_message()
                return self._from_wire(response)
            except Exception as exc:  # noqa: BLE001 - normalized below
                if attempt == 1 and self._degrade_from(exc):
                    continue
                raise self._normalize(exc) from exc
        raise ProviderError("unreachable")

    def _degrade_from(self, exc: Exception) -> bool:
        """On a 400 about a parameter we opted into, drop it and retry once."""
        if getattr(exc, "status_code", None) != 400:
            return False
        text = str(exc).lower()
        dropped = False
        for feature, needles in (
            ("fallbacks", ("fallback",)),
            ("adaptive_thinking", ("thinking", "adaptive", "budget_tokens")),
            ("effort", ("effort", "output_config")),
        ):
            if feature not in self._degraded and any(n in text for n in needles):
                self._degraded.add(feature)
                dropped = True
        return dropped

    def _normalize(self, exc: Exception) -> Exception:
        sdk = self._sdk
        for cls, wrap in (
            (getattr(sdk, "AuthenticationError", ()), AuthError),
            (getattr(sdk, "PermissionDeniedError", ()), AuthError),
            (getattr(sdk, "RateLimitError", ()), RateLimitError),
        ):
            if cls and isinstance(exc, cls):
                return wrap(str(exc))
        if isinstance(exc, getattr(sdk, "APIError", ())):
            return ProviderError(str(exc))
        return exc

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self._client.models.list()]
        except Exception:  # noqa: BLE001 - listing is a convenience, never fatal
            return []

    def check(self) -> str:
        try:
            self._client.messages.create(
                model=self.model, max_tokens=1, messages=[{"role": "user", "content": "hi"}]
            )
            return f"ok - {self.model} reachable"
        except Exception as exc:  # noqa: BLE001
            return f"failed - {self._normalize(exc)}"

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
