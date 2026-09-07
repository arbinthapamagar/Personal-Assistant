"""Ollama adapter - the fully local, no-key, no-network path.

Talks to /api/chat over HTTP (no SDK). Two differences from OpenAI's format
matter: Ollama returns tool-call arguments as a real object rather than a JSON
string, and it streams newline-delimited JSON instead of SSE.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import httpx

from ..config import Config, Profile
from ..errors import ProviderError
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

DEFAULT_BASE = "http://localhost:11434"


class OllamaProvider(Provider):
    name = "ollama"
    sdk = None

    def __init__(self, profile: Profile, config: Config) -> None:
        super().__init__(profile, config)
        self._base = (profile.base_url or DEFAULT_BASE).rstrip("/")
        # No read timeout: a large local model on CPU can think for minutes.
        self._http = httpx.Client(timeout=httpx.Timeout(None, connect=10.0))

    # ---- translation --------------------------------------------------------

    def _to_wire(self, system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})

        for msg in messages:
            for res in [p for p in msg.parts if isinstance(p, ToolResult)]:
                entry = {"role": "tool", "content": res.content}
                # Newer Ollama builds match results to calls by name.
                name = res.call_id.rsplit(":", 1)[0]
                if name:
                    entry["tool_name"] = name
                wire.append(entry)

            texts = [p.text for p in msg.parts if isinstance(p, Text) and p.text]
            calls = [p for p in msg.parts if isinstance(p, ToolCall)]
            images = [p for p in msg.parts if isinstance(p, Image)]
            if not (texts or calls or images):
                continue

            entry = {"role": msg.role, "content": "\n".join(texts)}
            if calls:
                entry["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                ]
            if images:
                import base64

                entry["images"] = [
                    base64.standard_b64encode(i.data).decode() for i in images
                ]
            wire.append(entry)
        return wire

    def _body(self, system, messages, tools, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire(system, messages),
            "stream": stream,
            "options": {"num_predict": self.profile.max_tokens},
            # Release the model from RAM after idle, so switching between local
            # models (e.g. dolphin 7B <-> llama 3B) frees memory for the next
            # one instead of pinning both. Overridable per profile; some
            # machines set OLLAMA_KEEP_ALIVE=-1 globally, which this defeats.
            "keep_alive": self.profile.extra.get("keep_alive", "5m"),
        }
        if tools:
            body["tools"] = [
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
            body["options"]["temperature"] = self.profile.temperature
        body["options"].update(self.profile.extra.get("options", {}))
        return body

    def _parts_from_message(self, raw: dict[str, Any], index: int = 0) -> list[Any]:
        parts: list[Any] = []
        if raw.get("thinking"):
            parts.append(Thinking(raw["thinking"]))
        if raw.get("content"):
            parts.append(Text(raw["content"]))
        for i, call in enumerate(raw.get("tool_calls") or []):
            fn = call.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments")
            if isinstance(args, str):  # some builds still send a string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"__unparsed__": args}
            parts.append(ToolCall(f"{name}:{index + i}", name, args or {}))
        return parts

    @staticmethod
    def _usage(payload: dict[str, Any]) -> Usage:
        return Usage(
            input_tokens=payload.get("prompt_eval_count", 0) or 0,
            output_tokens=payload.get("eval_count", 0) or 0,
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
        streaming = bool(self.config.stream and on_text is not None)
        body = self._body(system, messages, tools, stream=streaming)
        url = f"{self._base}/api/chat"
        try:
            if not streaming:
                resp = self._http.post(url, json=body)
                self._raise_for_status(resp)
                payload = resp.json()
                parts = self._parts_from_message(payload.get("message") or {})
                return self._finish(parts, payload)
            return self._stream(url, body, on_text)
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"cannot reach Ollama at {self._base} - is it running? "
                f"Start it with: ollama serve"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

    def _stream(self, url: str, body: dict[str, Any], on_text: TextSink) -> Completion:
        text_acc: list[str] = []
        think_acc: list[str] = []
        calls: list[ToolCall] = []
        final: dict[str, Any] = {}

        with self._http.stream("POST", url, json=body) as resp:
            if resp.status_code >= 400:
                resp.read()
                self._raise_for_status(resp)
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = payload.get("message") or {}
                if chunk := message.get("content"):
                    text_acc.append(chunk)
                    on_text(chunk)
                if chunk := message.get("thinking"):
                    think_acc.append(chunk)
                for i, call in enumerate(message.get("tool_calls") or []):
                    fn = call.get("function") or {}
                    name = fn.get("name", "")
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"__unparsed__": args}
                    calls.append(ToolCall(f"{name}:{len(calls) + i}", name, args or {}))
                if payload.get("done"):
                    final = payload

        parts: list[Any] = []
        if think_acc:
            parts.append(Thinking("".join(think_acc)))
        if text_acc:
            parts.append(Text("".join(text_acc)))
        parts.extend(calls)
        return self._finish(parts, final)

    def _finish(self, parts: list[Any], payload: dict[str, Any]) -> Completion:
        has_calls = any(isinstance(p, ToolCall) for p in parts)
        reason = payload.get("done_reason") or ""
        stop = "tool_use" if has_calls else ("max_tokens" if reason == "length" else "end_turn")
        return Completion(
            message=Message("assistant", parts),
            stop_reason=stop,
            usage=self._usage(payload),
            model=payload.get("model", self.model),
            raw=payload,
        )

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        if resp.status_code == 404 and "model" in str(detail).lower():
            raise ProviderError(
                f"Ollama has no model {self.model!r}. Pull it first:\n  ollama pull {self.model}"
            )
        raise ProviderError(f"Ollama HTTP {resp.status_code}: {detail}")

    def list_models(self) -> list[str]:
        try:
            resp = self._http.get(f"{self._base}/api/tags", timeout=5.0)
            resp.raise_for_status()
            return sorted(m["name"] for m in resp.json().get("models", []))
        except Exception:  # noqa: BLE001
            return []

    def check(self) -> str:
        models = self.list_models()
        if not models:
            return f"failed - no response from {self._base} (try: ollama serve)"
        if self.model not in models:
            return (
                f"failed - {self.model!r} not pulled. Available: {', '.join(models[:6])}\n"
                f"  fix: ollama pull {self.model}"
            )
        return f"ok - {self.model} loaded locally"

    def close(self) -> None:
        self._http.close()
