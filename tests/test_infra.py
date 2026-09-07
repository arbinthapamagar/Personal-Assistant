"""Tests for the multi-tasking infrastructure: store, tasks, concurrency,
memory, skills, and the overridable denial floor. These hit real backends
(sqlite, real subprocesses, a real local vector store) rather than mocks,
because the value of these pieces is precisely in the edges mocks paper over.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa import config as config_mod
from pa.concurrency import Job, KeyedLocks, Scheduler
from pa.security import Gate, Verdict
from pa.store import SqliteStore
from pa.tasks import TaskManager


# ---- store -----------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "state.db")
    yield s
    s.close()


def test_store_document_roundtrip(store):
    store.put("things", "a", {"value": 1, "nested": {"x": [1, 2]}})
    assert store.get("things", "a") == {"value": 1, "nested": {"x": [1, 2]}}
    store.put("things", "b", {"value": 2})
    assert len(store.list("things")) == 2
    store.delete("things", "a")
    assert store.get("things", "a") is None


def test_store_lock_is_exclusive(store):
    with store.lock("resource", ttl=5):
        # A second acquire with a short timeout must fail while we hold it.
        assert store._acquire("resource", 5, 0.2) is None
    # Released now, so it is available again.
    token = store._acquire("resource", 5, 0.2)
    assert token is not None
    store._release(token)


def test_store_pubsub_polling(store):
    store.publish("chan", {"n": 1})
    store.publish("chan", {"n": 2})
    # A listener started after these still sees only future messages, so
    # publish once more from another "thread" of control.
    seen = []

    import threading

    def publisher():
        time.sleep(0.2)
        store2 = SqliteStore(store.path)
        store2.publish("chan", {"n": 3})
        store2.close()

    threading.Thread(target=publisher).start()
    for message in store.listen("chan", timeout=0.5):
        seen.append(message)
        break
    assert seen == [{"n": 3}]


# ---- concurrency -----------------------------------------------------------


def test_scheduler_runs_in_parallel():
    scheduler = Scheduler(max_workers=6)
    jobs = [Job(i, f"j{i}", lambda: time.sleep(0.3) or "ok") for i in range(6)]
    start = time.monotonic()
    results = list(scheduler.run(jobs))
    elapsed = time.monotonic() - start
    scheduler.shutdown(wait=True)
    assert len(results) == 6
    assert all(r.ok for r in results)
    assert elapsed < 1.0  # 6 x 0.3s sequential would be 1.8s


def test_scheduler_preserves_indices_and_isolates_failures():
    scheduler = Scheduler(max_workers=4)

    def boom():
        raise ValueError("nope")

    jobs = [
        Job(0, "ok", lambda: "fine"),
        Job(1, "bad", boom),
        Job(2, "ok2", lambda: "also fine"),
    ]
    by_index = {r.index: r for r in scheduler.run(jobs)}
    scheduler.shutdown(wait=True)
    assert by_index[0].value == "fine"
    assert not by_index[1].ok and isinstance(by_index[1].error, ValueError)
    assert by_index[2].value == "also fine"


def test_affinity_pins_to_one_thread():
    scheduler = Scheduler(max_workers=8)
    import threading

    threads = []

    def record():
        threads.append(threading.get_ident())
        time.sleep(0.05)

    jobs = [Job(i, "x", record, affinity="solo") for i in range(5)]
    list(scheduler.run(jobs))
    scheduler.shutdown(wait=True)
    # All five ran on the same thread, because they shared an affinity.
    assert len(set(threads)) == 1


def test_keyed_locks_serialize_same_key():
    locks = KeyedLocks()
    order = []

    import threading

    def worker(name):
        with locks.acquire("shared"):
            order.append(f"{name}-in")
            time.sleep(0.1)
            order.append(f"{name}-out")

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start(); t2.start(); t1.join(); t2.join()
    # Whoever went first finished before the other started - never interleaved.
    assert order in (
        ["a-in", "a-out", "b-in", "b-out"],
        ["b-in", "b-out", "a-in", "a-out"],
    )


# ---- background tasks ------------------------------------------------------


def test_task_completes_and_records_exit_code(store, tmp_path):
    manager = TaskManager(store)
    task = manager.start("echo hello; exit 0", cwd=tmp_path, label="greet")
    finished = manager.wait([task.id], timeout=10)[0]
    assert finished.status == "done"
    assert finished.exit_code == 0
    assert "hello" in manager.output(task.id)


def test_task_failure_is_reported(store, tmp_path):
    manager = TaskManager(store)
    task = manager.start("echo oops >&2; exit 3", cwd=tmp_path)
    finished = manager.wait([task.id], timeout=10)[0]
    assert finished.status == "failed" and finished.exit_code == 3
    assert "oops" in manager.output(task.id)


def test_task_survives_a_new_manager(store, tmp_path):
    """State is on disk, so a fresh manager (a new `pa` process) sees it."""
    manager = TaskManager(store)
    task = manager.start("sleep 0.3; echo done", cwd=tmp_path)
    reader = TaskManager(SqliteStore(store.path))
    assert reader.get(task.id).status == "running"
    manager.wait([task.id], timeout=10)
    assert reader.get(task.id).status == "done"


def test_task_cancel_stops_a_long_job(store, tmp_path):
    manager = TaskManager(store)
    task = manager.start("sleep 30", cwd=tmp_path)
    time.sleep(0.3)
    cancelled = manager.cancel(task.id)
    assert cancelled.status == "cancelled"


# ---- overridable denial floor ---------------------------------------------


def test_unrestrict_disables_a_specific_rule():
    policy = config_mod.Security(mode="allow", unrestrict=["force-push"])
    gate = Gate(policy)
    # The unlocked rule now falls through to allow-mode...
    assert gate.decide("shell:git push --force origin main").verdict is Verdict.ALLOW
    # ...but every other floor rule still holds.
    assert gate.decide("shell:mkfs.ext4 /dev/sda").verdict is Verdict.DENY
    assert "force-push" in gate.disabled_rules


def test_floor_holds_by_default():
    gate = Gate(config_mod.Security(mode="allow"))
    assert gate.disabled_rules == []
    assert gate.decide("shell:git push --force").verdict is Verdict.DENY
