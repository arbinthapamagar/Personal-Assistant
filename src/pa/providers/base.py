"""The provider contract.

One method matters: `complete()` takes canonical messages plus tool specs and
returns a canonical Completion. Everything provider-specific - wire formats,
thinking blocks, tool-call encodings, streaming events - is confined to the
adapter that implements it.
"""

from __future__ import annotations

import abc
from typing import Callable, Sequence

from ..config import Config, Profile
from ..messages import Completion, Message, ToolSpec

TextSink = Callable[[str], None]


class Provider(abc.ABC):
    """Adapter for one model backend."""

    #: Registry key, e.g. "anthropic".
    name: str = ""
    #: Import name of the SDK this adapter needs, or None for pure HTTP.
    sdk: str | None = None

    def __init__(self, profile: Profile, config: Config) -> None:
        self.profile = profile
        self.config = config

    @property
    def model(self) -> str:
        return self.profile.model

    @abc.abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        on_text: TextSink | None = None,
    ) -> Completion:
        """Run one request. If `on_text` is given, stream text deltas into it."""

    def list_models(self) -> list[str]:
        """Model IDs this backend offers. Empty when the backend has no listing API."""
        return []

    def check(self) -> str:
        """Cheap credential/reachability probe. Returns a one-line status."""
        return "no check implemented"

    def close(self) -> None:
        """Release sockets. Safe to call twice."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} model={self.model!r}>"
