"""Background tasks.

Long jobs - a build, a test suite, a download, a training run - should not hold
the conversation hostage. `task_start` launches one detached and returns
immediately; the agent keeps talking and checks back later.

Completion is recorded by the *wrapper shell*, not by a watcher thread:

    bash -lc '<command>; printf %s $? > <id>.status'

That one detail is what makes tasks survive `pa` exiting. A watcher thread dies
with its process and would leave a task permanently "running"; a status file on
disk can be read by any later `pa` session, so state is never lost and no
daemon is needed.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import paths
from .errors import ToolError

COLLECTION = "tasks"
CHANNEL = "tasks"


def _runs_dir() -> Path:
    directory = paths.data_dir() / "tasks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass
class Task:
    id: str
    command: str
    label: str = ""
    cwd: str = ""
    pid: int = 0
    started: float = field(default_factory=time.time)
    finished: float | None = None
    exit_code: int | None = None
    status: str = "running"  # running | done | failed | cancelled | lost

    @property
    def output_path(self) -> Path:
        return _runs_dir() / f"{self.id}.log"

    @property
    def status_path(self) -> Path:
        return _runs_dir() / f"{self.id}.status"

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def to_doc(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> Task:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in doc.items() if k in known})

    def describe(self) -> str:
        mark = {
            "running": "..", "done": "ok", "failed": "xx",
            "cancelled": "--", "lost": "??",
        }.get(self.status, "??")
        name = self.label or self.command
        code = "" if self.exit_code is None else f" exit={self.exit_code}"
        return f"[{mark}] {self.id}  {self.elapsed:6.1f}s{code}  {name[:70]}"


class TaskManager:
    """Owns the task registry. State lives in the store, output on disk."""

    def __init__(self, store: Any) -> None:
        self.store = store

    # ---- lifecycle ----------------------------------------------------------

    def start(self, command: str, *, cwd: Path, label: str = "") -> Task:
        task_id = uuid.uuid4().hex[:8]
        task = Task(id=task_id, command=command, label=label, cwd=str(cwd))

        # The wrapper records the exit code where any process can read it.
        # The command runs in a SUBSHELL: a bare `exit` in the user's command
        # would otherwise terminate the wrapper before the status line ran,
        # leaving the task forever "running". The subshell contains the exit,
        # and $? still carries its code out.
        wrapped = f"( {command}\n)\nprintf %s $? > {task.status_path}\n"
        with task.output_path.open("wb") as log:
            proc = subprocess.Popen(
                ["bash", "-lc", wrapped],
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=dict(os.environ, PAGER="cat", GIT_PAGER="cat", TERM="dumb"),
                # Its own process group, so cancelling kills the whole tree
                # rather than orphaning children.
                start_new_session=True,
            )
        task.pid = proc.pid
        self._save(task)
        return task

    def cancel(self, task_id: str) -> Task:
        task = self.require(task_id)
        if task.status != "running":
            raise ToolError(f"task {task_id} is already {task.status}")
        try:
            os.killpg(os.getpgid(task.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone; fall through and mark it
        except PermissionError as exc:
            raise ToolError(f"not permitted to kill task {task_id}: {exc}") from exc
        else:
            # Give the tree a moment to exit before insisting.
            for _ in range(20):
                time.sleep(0.05)
                if not _alive(task.pid):
                    break
            else:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(task.pid), signal.SIGKILL)

        task.status = "cancelled"
        task.finished = time.time()
        self._save(task)
        self._announce(task)
        return task

    # ---- inspection ---------------------------------------------------------

    def refresh(self, task: Task) -> Task:
        """Reconcile a running task against reality.

        Order matters: check the status file first. A process can exit between
        the liveness check and the file read, and calling that "lost" when the
        exit code is sitting on disk would be wrong.
        """
        if task.status != "running":
            return task

        if task.status_path.exists():
            raw = task.status_path.read_text().strip()
            task.exit_code = int(raw) if raw.isdigit() else -1
            task.status = "done" if task.exit_code == 0 else "failed"
            task.finished = task.status_path.stat().st_mtime
            self._save(task)
            self._announce(task)
            return task

        if not _alive(task.pid):
            # No status file and no process: killed by the OOM killer, a reboot,
            # or SIGKILL. Say so rather than reporting success.
            task.status = "lost"
            task.finished = time.time()
            self._save(task)
            self._announce(task)
        return task

    def list(self, *, include_finished: bool = True) -> list[Task]:
        tasks = [Task.from_doc(d) for d in self.store.list(COLLECTION)]
        tasks = [self.refresh(t) for t in tasks]
        if not include_finished:
            tasks = [t for t in tasks if t.status == "running"]
        return sorted(tasks, key=lambda t: t.started, reverse=True)

    def get(self, task_id: str) -> Task | None:
        doc = self.store.get(COLLECTION, task_id)
        return self.refresh(Task.from_doc(doc)) if doc else None

    def require(self, task_id: str) -> Task:
        task = self.get(task_id)
        if task is None:
            known = ", ".join(t.id for t in self.list()[:8]) or "none"
            raise ToolError(f"no task {task_id!r}. Known: {known}")
        return task

    def output(self, task_id: str, *, tail: int = 200) -> str:
        task = self.require(task_id)
        if not task.output_path.exists():
            return "[no output yet]"
        lines = task.output_path.read_text(errors="replace").splitlines()
        shown = lines[-tail:] if tail > 0 else lines
        omitted = len(lines) - len(shown)
        head = f"[{omitted} earlier lines omitted]\n" if omitted > 0 else ""
        return head + "\n".join(shown)

    def wait(self, task_ids: list[str], *, timeout: float) -> list[Task]:
        """Block until every named task finishes, or the timeout expires."""
        deadline = time.monotonic() + timeout
        pending = list(task_ids)
        finished: list[Task] = []
        while pending and time.monotonic() < deadline:
            still: list[str] = []
            for task_id in pending:
                task = self.require(task_id)
                if task.status == "running":
                    still.append(task_id)
                else:
                    finished.append(task)
            pending = still
            if pending:
                time.sleep(0.25)
        finished.extend(self.require(t) for t in pending)  # timed out, report as-is
        return finished

    def prune(self, *, older_than: float = 86_400) -> int:
        """Drop finished tasks and their logs. Returns how many went."""
        removed = 0
        cutoff = time.time() - older_than
        for task in self.list():
            if task.status == "running" or (task.finished or 0) > cutoff:
                continue
            task.output_path.unlink(missing_ok=True)
            task.status_path.unlink(missing_ok=True)
            self.store.delete(COLLECTION, task.id)
            removed += 1
        return removed

    # ---- internals ----------------------------------------------------------

    def _save(self, task: Task) -> None:
        self.store.put(COLLECTION, task.id, task.to_doc())

    def _announce(self, task: Task) -> None:
        try:
            self.store.publish(
                CHANNEL,
                {"id": task.id, "status": task.status, "label": task.label or task.command},
            )
        except Exception:  # noqa: BLE001 - notification is a nicety
            pass


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True
