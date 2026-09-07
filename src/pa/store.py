"""Shared state backend.

Background tasks, cross-session locks, and task notifications all need state
that outlives one turn and is visible to every `pa` process on the machine.
Two backends implement the same contract:

* **sqlite** (default) - zero setup, works on any machine, uses WAL so several
  `pa` processes can read and write concurrently. Notifications are polled.
* **redis** (opt-in) - real pub/sub, so a finished task wakes a waiting session
  instantly instead of on the next poll, and state can be shared beyond one
  machine.

SQLite is the honest default: it covers everything here except instant
notification. Redis is selected only when the config asks for it and the server
actually answers, and selection falls back rather than failing.
"""

from __future__ import annotations

import abc
import contextlib
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import paths
from .errors import PAError


class StoreError(PAError):
    """The state backend failed."""


@dataclass
class LockToken:
    name: str
    owner: str


class Store(abc.ABC):
    """Documents, locks, and notifications."""

    name: str = ""

    # ---- documents ----------------------------------------------------------

    @abc.abstractmethod
    def put(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None: ...

    @abc.abstractmethod
    def list(self, collection: str) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def delete(self, collection: str, doc_id: str) -> None: ...

    # ---- coordination -------------------------------------------------------

    @abc.abstractmethod
    def _acquire(self, name: str, ttl: float, timeout: float) -> LockToken | None: ...

    @abc.abstractmethod
    def _release(self, token: LockToken) -> None: ...

    @contextlib.contextmanager
    def lock(self, name: str, *, ttl: float = 300.0, timeout: float = 60.0):
        """Cross-process mutual exclusion. Raises StoreError on timeout."""
        token = self._acquire(name, ttl, timeout)
        if token is None:
            raise StoreError(f"timed out after {timeout}s waiting for lock {name!r}")
        try:
            yield token
        finally:
            self._release(token)

    @abc.abstractmethod
    def publish(self, channel: str, payload: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def listen(self, channel: str, *, timeout: float) -> Iterator[dict[str, Any]]:
        """Yield messages until `timeout` seconds have passed with none."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


class SqliteStore(Store):
    name = "sqlite"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (paths.data_dir() / "state.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._owner = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        """One connection per thread - sqlite3 objects are not thread-safe."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL lets readers and one writer proceed at once, which is what
            # makes several `pa` processes usable against one file.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                collection TEXT NOT NULL,
                doc_id     TEXT NOT NULL,
                body       TEXT NOT NULL,
                updated    REAL NOT NULL,
                PRIMARY KEY (collection, doc_id)
            );
            CREATE TABLE IF NOT EXISTS locks (
                name    TEXT PRIMARY KEY,
                owner   TEXT NOT NULL,
                expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                seq     INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                body    TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_channel ON events(channel, seq);
            """
        )

    # ---- documents ----------------------------------------------------------

    def put(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO docs(collection, doc_id, body, updated) VALUES(?,?,?,?) "
            "ON CONFLICT(collection, doc_id) DO UPDATE SET body=excluded.body, "
            "updated=excluded.updated",
            (collection, doc_id, json.dumps(doc), time.time()),
        )

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT body FROM docs WHERE collection=? AND doc_id=?", (collection, doc_id)
        ).fetchone()
        return json.loads(row["body"]) if row else None

    def list(self, collection: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT body FROM docs WHERE collection=? ORDER BY updated DESC", (collection,)
        ).fetchall()
        return [json.loads(r["body"]) for r in rows]

    def delete(self, collection: str, doc_id: str) -> None:
        self._conn.execute(
            "DELETE FROM docs WHERE collection=? AND doc_id=?", (collection, doc_id)
        )

    # ---- coordination -------------------------------------------------------

    def _acquire(self, name: str, ttl: float, timeout: float) -> LockToken | None:
        owner = f"{self._owner}:{uuid.uuid4().hex[:6]}"
        deadline = time.monotonic() + timeout
        while True:
            now = time.time()
            self._conn.execute("DELETE FROM locks WHERE expires < ?", (now,))
            try:
                self._conn.execute(
                    "INSERT INTO locks(name, owner, expires) VALUES(?,?,?)",
                    (name, owner, now + ttl),
                )
                return LockToken(name, owner)
            except sqlite3.IntegrityError:
                pass  # held by someone else
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def _release(self, token: LockToken) -> None:
        self._conn.execute(
            "DELETE FROM locks WHERE name=? AND owner=?", (token.name, token.owner)
        )

    def publish(self, channel: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO events(channel, body, created) VALUES(?,?,?)",
            (channel, json.dumps(payload), time.time()),
        )
        # Keep the log bounded; it is a notification bus, not a record.
        self._conn.execute(
            "DELETE FROM events WHERE seq < (SELECT MAX(seq) - 2000 FROM events)"
        )

    def listen(self, channel: str, *, timeout: float) -> Iterator[dict[str, Any]]:
        row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) AS s FROM events").fetchone()
        cursor = row["s"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rows = self._conn.execute(
                "SELECT seq, body FROM events WHERE channel=? AND seq>? ORDER BY seq",
                (channel, cursor),
            ).fetchall()
            for entry in rows:
                cursor = entry["seq"]
                deadline = time.monotonic() + timeout  # reset idle timer
                yield json.loads(entry["body"])
            time.sleep(0.15)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


class RedisStore(Store):
    name = "redis"

    def __init__(self, url: str = "redis://localhost:6379/0", *, auto: bool = True) -> None:
        from . import deps

        redis_mod = deps.require("redis", auto=auto, purpose="the redis state backend")
        self._redis = redis_mod.Redis.from_url(url, decode_responses=True)
        self._redis.ping()  # fail fast, so the caller can fall back to sqlite
        self._prefix = "pa"
        self._owner = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def _key(self, collection: str, doc_id: str) -> str:
        return f"{self._prefix}:doc:{collection}:{doc_id}"

    def put(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        pipe = self._redis.pipeline()
        pipe.set(self._key(collection, doc_id), json.dumps(doc))
        pipe.sadd(f"{self._prefix}:idx:{collection}", doc_id)
        pipe.execute()

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        raw = self._redis.get(self._key(collection, doc_id))
        return json.loads(raw) if raw else None

    def list(self, collection: str) -> list[dict[str, Any]]:
        ids = self._redis.smembers(f"{self._prefix}:idx:{collection}")
        if not ids:
            return []
        raws = self._redis.mget([self._key(collection, i) for i in ids])
        return [json.loads(r) for r in raws if r]

    def delete(self, collection: str, doc_id: str) -> None:
        pipe = self._redis.pipeline()
        pipe.delete(self._key(collection, doc_id))
        pipe.srem(f"{self._prefix}:idx:{collection}", doc_id)
        pipe.execute()

    def _acquire(self, name: str, ttl: float, timeout: float) -> LockToken | None:
        owner = f"{self._owner}:{uuid.uuid4().hex[:6]}"
        key = f"{self._prefix}:lock:{name}"
        deadline = time.monotonic() + timeout
        while True:
            if self._redis.set(key, owner, nx=True, px=int(ttl * 1000)):
                return LockToken(name, owner)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def _release(self, token: LockToken) -> None:
        # Compare-and-delete, so an expired lock reacquired by someone else is
        # not deleted out from under them.
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] "
            "then return redis.call('del', KEYS[1]) else return 0 end"
        )
        try:
            self._redis.eval(script, 1, f"{self._prefix}:lock:{token.name}", token.owner)
        except Exception:  # noqa: BLE001 - releasing must never raise
            pass

    def publish(self, channel: str, payload: dict[str, Any]) -> None:
        self._redis.publish(f"{self._prefix}:ch:{channel}", json.dumps(payload))

    def listen(self, channel: str, *, timeout: float) -> Iterator[dict[str, Any]]:
        pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(f"{self._prefix}:ch:{channel}")
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                message = pubsub.get_message(timeout=0.5)
                if message and message.get("type") == "message":
                    deadline = time.monotonic() + timeout
                    with contextlib.suppress(json.JSONDecodeError):
                        yield json.loads(message["data"])
        finally:
            pubsub.close()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._redis.close()


# ---------------------------------------------------------------------------


def build(config: Any) -> Store:
    """Pick a backend from config, degrading to sqlite rather than failing.

    A machine that was set up with redis and later loses it should keep
    working, so an unreachable redis is a warning path, not an error path.
    """
    requested = getattr(config, "state_backend", "auto")
    url = getattr(config, "redis_url", "redis://localhost:6379/0")

    if requested in ("redis", "auto"):
        try:
            return RedisStore(url, auto=getattr(config, "auto_install_deps", True))
        except Exception as exc:  # noqa: BLE001 - any failure means "use sqlite"
            if requested == "redis":
                raise StoreError(
                    f"state_backend is 'redis' but {url} is unusable: {exc}\n"
                    f"Start it (sudo systemctl start redis-server) or set "
                    f"state_backend: sqlite"
                ) from exc
    return SqliteStore()
