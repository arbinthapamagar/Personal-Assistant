"""The persistent agent daemon.

One warm `pa` process holds the model client, the tool registry, semantic
memory, background tasks, and (optionally) a live browser, and answers requests
over a Unix socket. Any terminal, script, or keyboard shortcut can reach it
with `pa send`, so the assistant is "everywhere on the machine" without paying
cold-start on every question and without losing state between them.

Design decisions worth knowing:

* **One turn at a time.** A single conversation is stateful, so prompt turns
  are serialized by a lock. Control ops (status, ping, stop, interrupt) skip
  the lock and are answered immediately, even mid-turn.
* **Approvals travel to the asking client.** The daemon has no terminal, so
  when a tool needs approval it forwards the question down the socket to the
  client that submitted the turn, and blocks on that client's answer. A turn
  submitted with no interactive client (a scheduled job) falls back to the
  configured policy - unmatched calls are denied, never silently run.
* **The socket is private.** It lives in $XDG_RUNTIME_DIR at mode 0600, owned
  by the user. No network listener, ever.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import paths, protocol
from .agent import Agent
from .errors import PAError


def socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else paths.data_dir()
    directory = base / "arbin-assistant"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "pa.sock"


def pid_path() -> Path:
    return socket_path().with_name("pa.pid")


def is_running() -> bool:
    """True if a daemon is listening on the socket right now."""
    sock = socket_path()
    if not sock.exists():
        return False
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(1.0)
    try:
        client.connect(str(sock))
        client.sendall(protocol.encode({"op": "ping"}))
        reply = protocol.read_one(client, timeout=1.0)
        return bool(reply and reply.get("type") == "pong")
    except OSError:
        return False
    finally:
        client.close()


class Daemon:
    def __init__(self, agent: Agent, ui=None) -> None:
        self.agent = agent
        self.ui = ui
        self._turn_lock = threading.Lock()
        self._server: socket.socket | None = None
        self._stop = threading.Event()
        self._started = time.time()
        self._turns = 0
        # The connection currently driving a turn, so its approvals route back.
        self._active_conn: socket.socket | None = None
        self._active_lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------------

    def serve(self) -> int:
        path = socket_path()
        if is_running():
            self._log(f"a daemon is already running on {path}")
            return 1
        # Clear a stale socket left by a crash.
        if path.exists():
            path.unlink()

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        os.chmod(path, 0o600)
        self._server.listen(16)
        self._server.settimeout(1.0)
        pid_path().write_text(str(os.getpid()))
        self._log(f"listening on {path} (pid {os.getpid()})")
        self._notify("arbin-assistant daemon started", "Reach it with: arbin-assistant --daemon \"...\"")

        try:
            while not self._stop.is_set():
                try:
                    conn, _ = self._server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(
                    target=self._handle, args=(conn,), daemon=True
                ).start()
        finally:
            self._cleanup()
        return 0

    def _cleanup(self) -> None:
        if self._server is not None:
            self._server.close()
        for stale in (socket_path(), pid_path()):
            stale.unlink(missing_ok=True)
        self.agent.close()
        self._log("stopped")

    # ---- connection handling ------------------------------------------------

    def _handle(self, conn: socket.socket) -> None:
        try:
            request = protocol.read_one(conn, timeout=30.0)
            if not request:
                return
            op = request.get("op", "prompt")
            handler = {
                "ping": self._op_ping,
                "status": self._op_status,
                "stop": self._op_stop,
                "interrupt": self._op_interrupt,
                "prompt": self._op_prompt,
            }.get(op)
            if handler is None:
                self._send(conn, {"type": "error", "text": f"unknown op {op!r}"})
                return
            handler(conn, request)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001 - one bad client must not kill the daemon
            self._log(f"connection error: {exc}")
            self._safe_send(conn, {"type": "error", "text": str(exc)})
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _op_ping(self, conn, request) -> None:
        self._send(conn, {"type": "pong", "pid": os.getpid()})

    def _op_status(self, conn, request) -> None:
        self._send(conn, {
            "type": "status",
            "pid": os.getpid(),
            "uptime": round(time.time() - self._started, 1),
            "turns": self._turns,
            "busy": self._turn_lock.locked(),
            "profile": self.agent.config.active_profile,
            "model": self.agent.provider.model,
            "session": self.agent.session.id,
            "tools": len(self.agent.registry),
        })
        self._send(conn, {"type": "done"})

    def _op_stop(self, conn, request) -> None:
        self._send(conn, {"type": "stopping"})
        self._send(conn, {"type": "done"})
        self._stop.set()
        # Nudge the accept loop out of its timeout wait.
        try:
            self._server and self._server.close()
        except OSError:
            pass

    def _op_interrupt(self, conn, request) -> None:
        # Interrupt whatever turn is running by closing its driving connection,
        # which makes the blocked approval read fail and the turn unwind.
        with self._active_lock:
            active = self._active_conn
        interrupted = False
        if active is not None:
            try:
                active.shutdown(socket.SHUT_RDWR)
                interrupted = True
            except OSError:
                pass
        self._send(conn, {"type": "interrupted", "was_running": interrupted})
        self._send(conn, {"type": "done"})

    def _op_prompt(self, conn, request) -> None:
        text = (request.get("text") or "").strip()
        if not text:
            self._send(conn, {"type": "error", "text": "empty prompt"})
            return
        interactive = bool(request.get("interactive", True))

        # Serialize turns; tell a waiting client it is queued rather than hung.
        if self._turn_lock.locked():
            self._send(conn, {"type": "queued"})
        with self._turn_lock:
            self._turns += 1
            with self._active_lock:
                self._active_conn = conn
            # Route approvals for this turn back to this client, or fall to
            # policy for a non-interactive (scheduled) submission.
            self.agent.ctx.approve = (
                self._make_approver(conn) if interactive else (lambda k, d: False)
            )
            try:
                self._run_turn(conn, text)
            finally:
                with self._active_lock:
                    self._active_conn = None

    def _run_turn(self, conn, text: str) -> None:
        def on_text(delta: str) -> None:
            self._safe_send(conn, {"type": "token", "text": delta})

        try:
            for step in self.agent.turn(text, on_text=on_text):
                frame = {"type": step.kind, "text": step.text}
                if step.kind in ("tool", "result"):
                    frame["tool"] = step.tool
                if step.kind == "tool":
                    frame["args"] = step.args
                if step.kind == "result":
                    frame["is_error"] = step.is_error
                self._send(conn, frame)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._log("client disconnected mid-turn")
            return
        # Hand back the final assistant text as a discrete field, so a client
        # that ignored the stream can still get the answer.
        answer = ""
        for message in reversed(self.agent.session.messages):
            if message.role == "assistant" and message.text.strip():
                answer = message.text
                break
        self._safe_send(conn, {"type": "done", "answer": answer})

    # ---- approval bridge ----------------------------------------------------

    def _make_approver(self, conn: socket.socket) -> Callable[[str, str], bool]:
        """Approvals for this turn are asked of the client that submitted it."""

        def approve(key: str, description: str) -> bool:
            try:
                self._send(conn, {
                    "type": "approval",
                    "key": key,
                    "description": description,
                })
                reply = protocol.read_one(conn, timeout=300.0)
            except OSError:
                return False  # client vanished -> treat as declined
            if not reply or reply.get("type") != "approval_reply":
                return False
            # The client may also teach the gate a session rule.
            decision = reply.get("decision", "skip")
            pattern = reply.get("pattern")
            if decision == "always" and pattern:
                self.agent.ctx.gate.remember_allow(pattern)
            elif decision == "never" and pattern:
                self.agent.ctx.gate.remember_deny(pattern)
            return decision in ("once", "always")

        return approve

    # ---- helpers ------------------------------------------------------------

    def _send(self, conn, obj) -> None:
        conn.sendall(protocol.encode(obj))

    def _safe_send(self, conn, obj) -> None:
        try:
            conn.sendall(protocol.encode(obj))
        except OSError:
            pass

    def _log(self, message: str) -> None:
        line = f"[daemon] {message}"
        if self.ui is not None:
            self.ui.info(line)
        else:
            print(line, flush=True)

    def _notify(self, title: str, body: str) -> None:
        from shutil import which

        if which("notify-send"):
            try:
                import subprocess

                subprocess.run(
                    ["notify-send", "-a", "arbin-assistant", title, body],
                    check=False, capture_output=True, timeout=5,
                )
            except (OSError, PAError):
                pass
