"""Scheduler tests: cadence parsing to the minute, and the job store's
due/mark-ran lifecycle against a real sqlite backend. The time-dependent parts
use injected clocks so they're deterministic, not wall-clock races.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa.scheduler import Cadence, CadenceError, ScheduleStore, parse_cadence
from pa.store import SqliteStore


# ---- cadence parsing -------------------------------------------------------


@pytest.mark.parametrize("spec,kind,seconds", [
    ("every 30m", "interval", 1800),
    ("2h", "interval", 7200),
    ("every 1d", "interval", 86400),
    ("once in 1h", "once", 3600),
    ("in 45m", "once", 2700),
])
def test_interval_and_once_parsing(spec, kind, seconds):
    c = parse_cadence(spec)
    assert c.kind == kind and c.seconds == seconds


def test_daily_parsing():
    c = parse_cadence("daily 09:00")
    assert c.kind == "daily" and c.hour == 9 and c.minute == 0
    assert parse_cadence("at 14:30").minute == 30


@pytest.mark.parametrize("bad", ["", "every 0m", "daily 25:00", "nonsense", "every 5x"])
def test_bad_cadence_rejected(bad):
    with pytest.raises(CadenceError):
        parse_cadence(bad)


def test_daily_next_run_crosses_midnight():
    c = parse_cadence("daily 09:00")
    before = datetime(2026, 6, 15, 8, 0).timestamp()
    after = datetime(2026, 6, 15, 10, 0).timestamp()
    assert datetime.fromtimestamp(c.first_run(before)).day == 15   # later today
    assert datetime.fromtimestamp(c.first_run(after)).day == 16    # tomorrow


def test_once_never_repeats():
    assert parse_cadence("in 1h").next_run(1000.0) is None


def test_interval_repeats():
    assert parse_cadence("every 30m").next_run(1000.0) == 2800.0


# ---- job store lifecycle ---------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "state.db")
    yield s
    s.close()


def test_add_and_list(store):
    sched = ScheduleStore(store)
    job = sched.add("summarise my repos", "daily 09:00")
    assert job.id
    listed = sched.list()
    assert len(listed) == 1 and listed[0].prompt == "summarise my repos"


def test_due_returns_only_past_jobs(store):
    sched = ScheduleStore(store)
    job = sched.add("check build", "every 30m")
    # Not due yet (next_run is 30m out).
    assert sched.due(now=job.created + 60) == []
    # Due after the interval.
    due = sched.due(now=job.created + 1900)
    assert len(due) == 1 and due[0].id == job.id


def test_mark_ran_reschedules_interval(store):
    sched = ScheduleStore(store)
    job = sched.add("check build", "every 30m")
    ran_at = job.created + 1900
    sched.mark_ran(job, "ok", now=ran_at)
    reloaded = sched.get(job.id)
    assert reloaded.runs == 1 and reloaded.last_status == "ok"
    assert reloaded.next_run == ran_at + 1800   # rescheduled 30m out
    assert reloaded.enabled


def test_once_job_disables_after_running(store):
    sched = ScheduleStore(store)
    job = sched.add("one-time reminder", "once in 1h")
    sched.mark_ran(job, "ok", now=job.created + 3700)
    reloaded = sched.get(job.id)
    assert reloaded.runs == 1 and reloaded.enabled is False
    assert sched.due(now=job.created + 999999) == []   # never fires again


def test_cancel(store):
    sched = ScheduleStore(store)
    job = sched.add("x", "every 1h")
    assert sched.cancel(job.id) is True
    assert sched.get(job.id) is None
    assert sched.cancel("nonexistent") is False


def test_jobs_survive_a_new_store_instance(store):
    ScheduleStore(store).add("persist me", "daily 08:00")
    # A fresh ScheduleStore over the same db (a new process) sees the job.
    again = ScheduleStore(SqliteStore(store.path))
    assert any(j.prompt == "persist me" for j in again.list())
