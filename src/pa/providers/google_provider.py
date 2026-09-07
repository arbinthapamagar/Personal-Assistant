"""Gemini adapter, over the REST API.

Deliberately no SDK: the `generateContent` wire format is stable and this keeps
the dependency count at httpx (already core), so a Gemini-only machine installs
nothing extra. Two Gemini-specific quirks are handled here - function results
are keyed by function *name* rather than a call id, and the schema dialect is a
restricted OpenAPI subset that rejects several common JSON Schema keywords.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import httpx

from ..config import Config, Profile
from ..errors import AuthError, ProviderError, RateLimitError
from ..messages import (
    Completion,
    Image,
    Message,
    Text,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from .base import Provider, TextSink

DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"

_STOP_MAP = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "refusal",
    "RECITATION": "refusal",
    "PROHIBITED_CONTENT": "refusal",
    "BLOCKLIST": "refusal",
}

# Keywords the Gemini schema dialect rejects outright.
_SCHEMA_DROP = {
    "additionalProperties", "$schema", "$id", "$ref", "definitions", "$defs",
    "exclusiveMinimum", "exclusiveMaximum", "const", "examples", "default",
    "oneOf", "allOf", "not", "patternProperties", "unevaluatedProperties",
}


def _sanitize_schema(node: Any) -> Any:
    """Strip JSON Schema keywords Gemini refuses, recursively."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in _SCHEMA_DROP:
                continue
            out[key] = _sanitize_schema(value)
        # Gemini requires a type on every object node.
        if "properties" in out and "type" not in out:
            out["type"] = "object"
        return out
    if isinstance(node, list):
        return [_sanitize_schema(v) for v in node]
    return node


class GoogleProvider(Provider):
    name = "google"
    sdk = None

    def __init__(self, profile: Profile, config: Config) -> None:
        super().__init__(profile, config)
        self._key = profile.resolve_key()
        if not self._key:
            env = profile.api_key_env or "GEMINI_API_KEY"
            raise AuthError(f"no Gemini API key - set ${env} or profiles.{profile.name}.api_key")
        self._base = (profile.base_url or DEFAULT_BASE).rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(600.0, connect=15.0))

    # ---- translation --------------------------------------------------------

    def _to_wire(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            for part in msg.parts:
                if isinstance(part, Text):
                    if part.text:
                        parts.append({"text": part.text})
                elif isinstance(part, ToolCall):
                    call_part: dict[str, Any] = {
                        "functionCall": {"name": part.name, "args": part.arguments}
                    }
                    # Thinking Gemini models attach a thought_signature to each
                    # call and reject the next turn unless it is echoed back.
                    if part.meta and part.meta.get("thought_signature"):
                        call_part["thoughtSignature"] = part.meta["thought_signature"]
                    parts.append(call_part)
                elif isinstance(part, ToolResult):
                    # Gemini keys results by name; our ids are "name:index".
                    name = part.call_id.rsplit(":", 1)[0]
                    parts.append({
                        "functionResponse": {
                            "name": name,
                            "response": {"output": part.content, "error": part.is_error},
                        }
                    })
                elif isinstance(part, Image):
                    import base64

                    parts.append({
                        "inlineData": {
                            "mimeType": part.media_type,
                            "data": base64.standard_b64encode(part.data).decode(),
                        }
                    })
            if parts:
                contents.append({"role": role, "parts": parts})
        return contents

    def _body(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]):
        body: dict[str, Any] = {
            "contents": self._to_wire(messages),
            "generationConfig": {"maxOutputTokens": self.profile.max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{
                "functionDeclarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": _sanitize_schema(t.parameters),
                    }
                    for t in tools
                ]
            }]
        if self.profile.temperature is not None:
            body["generationConfig"]["temperature"] = self.profile.temperature
        return body

    def _from_wire(self, payload: dict[str, Any]) -> Completion:
        candidates = payload.get("candidates") or [{}]
        candidate = candidates[0]
        parts: list[Any] = []
        counter = 0
        for raw in (candidate.get("content") or {}).get("parts") or []:
            if "text" in raw and raw["text"]:
                parts.append(Text(raw["text"]))
            elif "functionCall" in raw:
                call = raw["functionCall"]
                name = call.get("name", "")
                meta = None
                if sig := raw.get("thoughtSignature"):
                    meta = {"thought_signature": sig}
                parts.append(ToolCall(f"{name}:{counter}", name, call.get("args") or {}, meta=meta))
                counter += 1

        meta = payload.get("usageMetadata") or {}
        finish = candidate.get("finishReason") or "STOP"
        stop = _STOP_MAP.get(finish, "end_turn")
        if any(isinstance(p, ToolCall) for p in parts):
            stop = "tool_use"
        if stop == "refusal":
            parts.append(Text(f"[blocked by Gemini safety filter: {finish}]"))

        return Completion(
            message=Message("assistant", parts),
            stop_reason=stop,
            usage=Usage(
                input_tokens=meta.get("promptTokenCount", 0),
                output_tokens=meta.get("candidatesTokenCount", 0),
                cache_read_tokens=meta.get("cachedContentTokenCount", 0),
            ),
            model=payload.get("modelVersion", self.model),
            raw=payload,
        )

    # ---- request ------------------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        on_text: TextSink | None = None,
    ) -> Completion:
        body = self._body(system, messages, tools)
        params = {"key": self._key}
        # 429 (rate limit) and 503 (model overloaded) are transient and common
        # on the free tier, so ride them out with exponential backoff rather
        # than failing the turn. Other errors raise immediately.
        import time

        attempts = 5
        for attempt in range(attempts):
            try:
                if self.config.stream and on_text is not None:
                    return self._stream(body, params, on_text)
                url = f"{self._base}/models/{self.model}:generateContent"
                resp = self._http.post(url, params=params, json=body)
                self._raise_for_status(resp)
                return self._from_wire(resp.json())
            except RateLimitError:
                if attempt == attempts - 1:
                    raise
                time.sleep(min(2 ** attempt, 20))
            except ProviderError as exc:
                if "503" in str(exc) and attempt < attempts - 1:
                    time.sleep(min(2 ** attempt, 20))
                    continue
                raise
            except httpx.HTTPError as exc:
                raise ProviderError(f"Gemini request failed: {exc}") from exc
        raise ProviderError("Gemini: exhausted retries")

    def _stream(self, body, params, on_text: TextSink) -> Completion:
        url = f"{self._base}/models/{self.model}:streamGenerateContent"
        merged: dict[str, Any] = {}
        text_acc: list[str] = []
        calls: list[ToolCall] = []
        counter = 0

        with self._http.stream(
            "POST", url, params={**params, "alt": "sse"}, json=body
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                self._raise_for_status(resp)
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    payload = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                merged = payload  # keep the newest, for usage + finishReason
                candidate = (payload.get("candidates") or [{}])[0]
                for raw in (candidate.get("content") or {}).get("parts") or []:
                    if raw.get("text"):
                        text_acc.append(raw["text"])
                        on_text(raw["text"])
                    elif "functionCall" in raw:
                        call = raw["functionCall"]
                        name = call.get("name", "")
                        meta = None
                        if sig := raw.get("thoughtSignature"):
                            meta = {"thought_signature": sig}
                        calls.append(
                            ToolCall(f"{name}:{counter}", name, call.get("args") or {}, meta=meta)
                        )
                        counter += 1

        completion = self._from_wire(merged or {})
        parts: list[Any] = ([Text("".join(text_acc))] if text_acc else []) + list(calls)
        completion.message = Message("assistant", parts)
        completion.stop_reason = "tool_use" if calls else completion.stop_reason
        return completion

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("error", {}).get("message", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        if resp.status_code in (401, 403):
            raise AuthError(f"Gemini rejected the API key: {detail}")
        if resp.status_code == 429:
            raise RateLimitError(f"Gemini rate limit: {detail}")
        raise ProviderError(f"Gemini HTTP {resp.status_code}: {detail}")

    def list_models(self) -> list[str]:
        try:
            resp = self._http.get(f"{self._base}/models", params={"key": self._key})
            resp.raise_for_status()
            return sorted(
                m["name"].removeprefix("models/") for m in resp.json().get("models", [])
            )
        except Exception:  # noqa: BLE001
            return []

    def check(self) -> str:
        models = self.list_models()
        if not models:
            return "failed - could not list models (bad key or no network)"
        if self.model not in models:
            return f"ok - key valid, but {self.model!r} is not in the model list"
        return f"ok - {self.model} reachable"

    def close(self) -> None:
        self._http.close()
