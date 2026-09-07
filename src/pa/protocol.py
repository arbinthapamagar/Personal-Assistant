"""Wire protocol for the control socket.

Newline-delimited JSON, one object per line, both directions. Chosen over
anything cleverer because it is trivial to speak from a shell script, a hotkey
binding, or another language - `echo '{"op":"prompt","text":"hi"}' | nc -U
<sock>` works - which is the whole point of exposing a socket.

A request is one object. A response is a *stream* of objects ending in a
`done` (or `error`) frame, so the client can render tool calls and tokens as
they happen rather than waiting for the whole turn.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

# Guard against a client (or a corrupted peer) sending an unbounded line.
MAX_LINE = 8 * 1024 * 1024


def encode(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def read_lines(sock, *, chunk: int = 65536) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a socket until it closes.

    Tolerates partial reads and multiple objects per packet. A malformed line
    yields an `{"_error": ...}` marker rather than raising, so one bad frame
    does not kill the connection.
    """
    buffer = b""
    while True:
        try:
            data = sock.recv(chunk)
        except (ConnectionResetError, OSError):
            return
        if not data:
            return
        buffer += data
        if len(buffer) > MAX_LINE:
            yield {"_error": "line exceeded maximum length"}
            return
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"_error": f"bad json: {exc}"}


def read_one(sock, *, timeout: float | None = None) -> dict[str, Any] | None:
    """Read a single object, for request/response exchanges like approvals."""
    if timeout is not None:
        sock.settimeout(timeout)
    try:
        for obj in read_lines(sock):
            return obj
    except OSError:
        return None
    finally:
        if timeout is not None:
            sock.settimeout(None)
    return None
