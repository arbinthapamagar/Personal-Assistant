"""Shared HTTP layer with optional Tor routing.

One place that builds the httpx client for every web tool, so a single switch
routes fetching and searching over Tor. Tor earns its place here for three
real reasons, all legitimate:

* **Reach** - .onion services and sites blocked on the normal internet.
* **Rate limits** - each Tor circuit is a fresh exit IP, so the free search
  backends stop throttling under heavy use (the "lots of searching" problem).
* **Privacy** - requests do not carry the machine's real address.

`.onion` URLs are always routed through Tor (they resolve nowhere else). Normal
URLs go direct unless Tor is switched on in config or per call.
"""

from __future__ import annotations

import socket
from typing import Any

import httpx

DEFAULT_SOCKS = "socks5h://127.0.0.1:9050"
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def tor_running(socks_url: str = DEFAULT_SOCKS) -> bool:
    """Is a Tor SOCKS proxy actually listening? Cheap TCP probe."""
    host, port = _socks_host_port(socks_url)
    try:
        with socket.socket() as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def _socks_host_port(socks_url: str) -> tuple[str, int]:
    # socks5h://host:port
    rest = socks_url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    return host or "127.0.0.1", int(port or "9050")


def tor_settings(config: Any) -> dict[str, Any]:
    base = {"enabled": False, "auto_onion": True, "socks": DEFAULT_SOCKS}
    base.update(getattr(config, "tor", None) or {})
    return base


def use_tor_for(url: str, config: Any, *, force: bool | None = None) -> str | None:
    """Return the SOCKS proxy to use for `url`, or None for a direct request.

    An .onion always needs Tor. Otherwise: an explicit per-call `force` wins,
    then the config switch.
    """
    settings = tor_settings(config)
    socks = settings["socks"]
    host = url.split("://", 1)[-1].split("/", 1)[0].lower()
    if host.endswith(".onion"):
        return socks
    if force is True:
        return socks
    if force is None and settings["enabled"]:
        return socks
    return None


def client(
    *,
    proxy: str | None = None,
    timeout: float = 30.0,
    follow_redirects: bool = True,
) -> httpx.Client:
    """Build an httpx client, over Tor when `proxy` is given."""
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(timeout, connect=min(timeout, 20.0)),
        "follow_redirects": follow_redirects,
        "headers": {"User-Agent": UA},
    }
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


def new_identity(socks_url: str = DEFAULT_SOCKS, control_port: int = 9051,
                 password: str = "") -> bool:
    """Ask Tor for a fresh circuit (new exit IP) via the control port.

    Best-effort: the control port is often closed by default, in which case a
    new circuit simply isn't forced. Returns whether the signal was sent.
    """
    try:
        with socket.create_connection(("127.0.0.1", control_port), timeout=3) as sock:
            auth = f'AUTHENTICATE "{password}"\r\n'.encode()
            sock.sendall(auth)
            if b"250" not in sock.recv(256):
                return False
            sock.sendall(b"SIGNAL NEWNYM\r\n")
            return b"250" in sock.recv(256)
    except OSError:
        return False
