"""Parallel tool execution.

When the model asks for several tools in one turn, they run at once. Three
constraints make this less trivial than handing everything to a thread pool,
and each is a real property of the thing being driven rather than caution:

* **Playwright's sync API is bound to its creating thread.** Every browser call
  must therefore run on one dedicated thread, or it raises. That is `affinity`.
* **There is one keyboard and one mouse.** Two `desktop_type` calls interleaving
  would produce shuffled characters, so desktop input takes a shared lock. That
  is `lock_key`.
* **Approval prompts share one terminal.** Two prompts printed at once are
  unreadable and unanswerable, so prompting is serialized separately from the
  work - held only while the question is on screen, never while a tool runs.

Locks are acquired inside `Tool.__call__`, so the scheduler only has to know
about threads.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class KeyedLocks:
    """Named re-entrant locks, created on first use."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def get(self, key: str) -> threading.RLock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    def acquire(self, key: str) -> threading.RLock:
        """Usable as a context manager: `with locks.acquire("browser"):`."""
        return self.get(key)

    def held_keys(self) -> list[str]:
        with self._guard:
            return sorted(self._locks)


@dataclass
class Job:
    """One unit of work for the scheduler."""

    index: int
    label: str
    run: Callable[[], Any]
    #: Pin to a named single-thread executor, for thread-bound resources.
    affinity: str | None = None


@dataclass
class Completed:
    index: int
    label: str
    value: Any = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Scheduler:
    """Runs jobs concurrently, honouring thread affinity.

    Results are yielded as they finish so the terminal stays responsive, and
    each carries its original index so the caller can restore order - which
    matters, because tool results must go back to the model deterministically.
    """

    def __init__(self, max_workers: int = 8) -> None:
        self.max_workers = max(1, max_workers)
        self._pool: ThreadPoolExecutor | None = None
        self._affinity_pools: dict[str, ThreadPoolExecutor] = {}
        self._guard = threading.Lock()
        self._closed = False

    def _general(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="pa-tool"
            )
        return self._pool

    def _affinity(self, name: str) -> ThreadPoolExecutor:
        """A single-thread executor per affinity name, so a thread-bound
        object is always touched by the same thread."""
        with self._guard:
            pool = self._affinity_pools.get(name)
            if pool is None:
                pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"pa-{name}")
                self._affinity_pools[name] = pool
            return pool

    def run(self, jobs: list[Job]) -> Iterator[Completed]:
        """Submit every job, yielding each as it completes."""
        if self._closed:
            raise RuntimeError("scheduler is shut down")
        if not jobs:
            return
        if len(jobs) == 1:
            # One job needs no thread at all - keep tracebacks and Ctrl-C
            # behaviour identical to the sequential path.
            job = jobs[0]
            try:
                yield Completed(job.index, job.label, value=job.run())
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                yield Completed(job.index, job.label, error=exc)
            return

        futures: dict[Future, Job] = {}
        for job in jobs:
            pool = self._affinity(job.affinity) if job.affinity else self._general()
            futures[pool.submit(job.run)] = job

        # Not using as_completed(): it cannot be interrupted cleanly, and a
        # Ctrl-C mid-turn should surface promptly.
        pending = set(futures)
        while pending:
            done, pending = _wait_any(pending)
            for future in done:
                job = futures[future]
                try:
                    yield Completed(job.index, job.label, value=future.result())
                except BaseException as exc:  # noqa: BLE001
                    yield Completed(job.index, job.label, error=exc)

    def shutdown(self, *, wait: bool = False) -> None:
        self._closed = True
        pools = [self._pool, *self._affinity_pools.values()]
        self._pool = None
        self._affinity_pools = {}
        for pool in pools:
            if pool is not None:
                pool.shutdown(wait=wait, cancel_futures=not wait)


def _wait_any(pending: set[Future]) -> tuple[list[Future], set[Future]]:
    """Block until at least one future finishes; return (done, still pending)."""
    from concurrent.futures import FIRST_COMPLETED, wait

    result = wait(pending, return_when=FIRST_COMPLETED)
    return list(result.done), set(result.not_done)
