"""Autonomous scheduling: run prompts on a cadence, without being asked.

Paired with the daemon, this makes the assistant proactive - "every morning at
09:00, summarise what changed in my repos", "every 30 minutes, check the build".
A job is a prompt plus a cadence; the daemon's scheduler thread wakes on the
minute, runs whatever is due as a non-interactive turn, and stores the result.

Cadence grammar, deliberately small and human:
    every <n><m|h|d>     -> fixed interval   ("every 30m", "every 2h")
    daily <HH:MM>        -> once a day at a local time
    at <HH:MM>           -> alias for daily
    once in <n><m|h|d>   -> a single future run
    in <n><m|h|d>        -> alias for "once in"

The next-run computation is pure and clock-injectable, so it can be tested to
the minute rather than by waiting.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

COLLECTION = "schedules"

_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}
_INTERVAL = re.compile(r"^(?:every\s+)?(\d+)\s*([mhd])$", re.I)
_DAILY = re.compile(r"^(?:daily|at)\s+(\d{1,2}):(\d{2})$", re.I)
_ONCE = re.compile(r"^(?:once\s+in|in)\s+(\d+)\s*([mhd])$", re.I)


class CadenceError(ValueError):
    """The cadence string could not be parsed."""


@dataclass
class Cadence:
    kind: str            # "interval" | "daily" | "once"
    seconds: int = 0     # for interval / once
    hour: int = 0        # for daily
    minute: int = 0

    def describe(self) -> str:
        if self.kind == "interval":
            return f"every {_human(self.seconds)}"
        if self.kind == "daily":
            return f"daily at {self.hour:02d}:{self.minute:02d}"
        return f"once, {_human(self.seconds)} from creation"

    def first_run(self, now: float) -> float:
        """When this cadence should first fire, given creation time `now`."""
        if self.kind == "interval":
            return now + self.seconds
        if self.kind == "once":
            return now + self.seconds
        return _next_daily(now, self.hour, self.minute)

    def next_run(self, after: float) -> float | None:
        """The run after a firing at time `after`. None = never again (once)."""
        if self.kind == "once":
            return None
        if self.kind == "interval":
            return after + self.seconds
        return _next_daily(after, self.hour, self.minute)


def parse_cadence(spec: str) -> Cadence:
    spec = (spec or "").strip()
    if m := _INTERVAL.match(spec):
        n, unit = int(m.group(1)), m.group(2).lower()
        if n <= 0:
            raise CadenceError("interval must be positive")
        return Cadence("interval", seconds=n * _UNIT_SECONDS[unit])
    if m := _ONCE.match(spec):
        n, unit = int(m.group(1)), m.group(2).lower()
        return Cadence("once", seconds=n * _UNIT_SECONDS[unit])
    if m := _DAILY.match(spec):
        hour, minute = int(m.group(1)), int(m.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise CadenceError(f"invalid time {hour:02d}:{minute:02d}")
        return Cadence("daily", hour=hour, minute=minute)
    raise CadenceError(
        f"could not understand schedule {spec!r}. Try 'every 30m', 'every 2h', "
        f"'daily 09:00', or 'once in 1h'."
    )


def _next_daily(now: float, hour: int, minute: int) -> float:
    dt = datetime.fromtimestamp(now)
    target = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= dt:
        target += timedelta(days=1)
    return target.timestamp()


def _human(seconds: int) -> str:
    for unit, s in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % s == 0 and seconds >= s:
            return f"{seconds // s}{unit}"
    return f"{seconds}s"


@dataclass
class Job:
    id: str
    prompt: str
    cadence: str                 # the raw spec, re-parsed on load
    profile: str = ""            # optional profile override
    next_run: float = 0.0
    created: float = field(default_factory=time.time)
    last_run: float | None = None
    last_status: str = ""
    runs: int = 0
    enabled: bool = True

    def to_doc(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> Job:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in doc.items() if k in known})

    def describe(self) -> str:
        state = "on" if self.enabled else "off"
        when = time.strftime("%H:%M", time.localtime(self.next_run)) if self.next_run else "-"
        last = self.last_status or "never run"
        return (f"[{state}] {self.id}  {parse_cadence(self.cadence).describe()}  "
                f"next {when}  ({self.runs} runs, {last})\n     {self.prompt[:70]}")


class ScheduleStore:
    """Job persistence over the shared state backend, so schedules survive
    restarts and are visible to every process."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def add(self, prompt: str, cadence: str, *, profile: str = "") -> Job:
        cad = parse_cadence(cadence)  # validate before saving
        job = Job(
            id=uuid.uuid4().hex[:8],
            prompt=prompt.strip(),
            cadence=cadence.strip(),
            profile=profile,
            next_run=cad.first_run(time.time()),
        )
        self.store.put(COLLECTION, job.id, job.to_doc())
        return job

    def list(self) -> list[Job]:
        jobs = [Job.from_doc(d) for d in self.store.list(COLLECTION)]
        return sorted(jobs, key=lambda j: j.next_run)

    def get(self, job_id: str) -> Job | None:
        doc = self.store.get(COLLECTION, job_id)
        return Job.from_doc(doc) if doc else None

    def save(self, job: Job) -> None:
        self.store.put(COLLECTION, job.id, job.to_doc())

    def cancel(self, job_id: str) -> bool:
        if self.store.get(COLLECTION, job_id) is None:
            return False
        self.store.delete(COLLECTION, job_id)
        return True

    def due(self, now: float | None = None) -> list[Job]:
        """Jobs whose next_run has passed and are enabled."""
        now = time.time() if now is None else now
        return [j for j in self.list() if j.enabled and j.next_run and j.next_run <= now]

    def mark_ran(self, job: Job, status: str, now: float | None = None) -> Job:
        """Record a run and compute the next fire time (or disable a once-job)."""
        now = time.time() if now is None else now
        job.last_run = now
        job.last_status = status
        job.runs += 1
        nxt = parse_cadence(job.cadence).next_run(now)
        if nxt is None:
            job.enabled = False       # a "once" job is done
            job.next_run = 0.0
        else:
            job.next_run = nxt
        self.save(job)
        return job
