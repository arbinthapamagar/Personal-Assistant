"""Self-development tool tests. These confirm the assistant can find its own
source, checkpoint/test/rollback safely, and that the safety loop's pieces
actually work - without performing a destructive rollback on the real repo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa import capabilities, config as config_mod
from pa.security import Gate
from pa.tools.base import ToolContext
from pa.tools import selfdev


@pytest.fixture
def ctx():
    cfg = config_mod.default_config()
    return ToolContext(cfg, capabilities.probe(), Gate(cfg.security), lambda k, d: True)


def test_locate_finds_the_source_tree(ctx):
    out = selfdev.SelfLocateTool()({}, ctx)
    assert "project root:" in out
    assert "src/pa" in out and "tests" in out
    assert "HEAD:" in out


def test_self_test_runs_the_suite(ctx):
    # Run a tiny, fast slice so the test-of-tests stays quick.
    out = selfdev.SelfTestTool()({"target": "test_harness.py"}, ctx)
    assert "tests PASSED" in out


def test_self_test_reports_failure_without_crashing(ctx, monkeypatch):
    # A filter that matches nothing -> pytest exits non-zero; tool reports it.
    out = selfdev.SelfTestTool()({"target": "definitely_no_such_test_xyz"}, ctx)
    assert "tests FAILED" in out or "no tests ran" in out


def test_locate_requires_a_git_repo(monkeypatch, ctx, tmp_path):
    # If the source root cannot be found, the tool errors clearly rather than
    # guessing - simulate by pointing _source_root at nothing.
    monkeypatch.setattr(selfdev, "_source_root", lambda: None)
    from pa.errors import ToolError

    with pytest.raises(ToolError, match="could not locate"):
        selfdev.SelfLocateTool()({}, ctx)


def test_checkpoint_and_rollback_on_a_scratch_repo(tmp_path, monkeypatch, ctx):
    """Exercise checkpoint + rollback against a throwaway git repo, so the real
    project is never touched."""
    repo = tmp_path / "proj"
    (repo / "src" / "pa").mkdir(parents=True)
    (repo / "src" / "pa" / "x.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-q", "-m", "base"], check=True,
    )
    monkeypatch.setattr(selfdev, "_source_root", lambda: repo)

    # A change, then checkpoint captures it.
    (repo / "src" / "pa" / "x.py").write_text("VALUE = 2\n")
    out = selfdev.SelfCheckpointTool()({"message": "change"}, ctx)
    assert "checkpoint saved" in out

    # A second, bad change, then rollback restores the checkpoint.
    (repo / "src" / "pa" / "x.py").write_text("VALUE = BROKEN(\n")
    selfdev.SelfRollbackTool()({}, ctx)
    assert (repo / "src" / "pa" / "x.py").read_text() == "VALUE = 2\n"
