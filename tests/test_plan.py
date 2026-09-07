"""Planner tests: the live task checklist the agent maintains."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa import capabilities, config as config_mod
from pa.security import Gate
from pa.tools.base import ToolContext
from pa.tools.plan import PlanTool, render_plan


@pytest.fixture
def ctx():
    cfg = config_mod.default_config()
    return ToolContext(cfg, capabilities.probe(), Gate(cfg.security), lambda k, d: True)


def test_set_and_render(ctx):
    PlanTool()({"action": "set", "steps": ["find the bug", "fix it", "add a test"]}, ctx)
    block = render_plan(ctx)
    assert "find the bug" in block and "add a test" in block
    assert block.count("[ ]") == 3


def test_check_marks_done(ctx):
    PlanTool()({"action": "set", "steps": ["a", "b"]}, ctx)
    out = PlanTool()({"action": "check", "n": 1}, ctx)
    assert "[x] 1. a" in out and "[ ] 2. b" in out
    assert "1/2 done" in out


def test_status_and_add(ctx):
    PlanTool()({"action": "set", "steps": ["a"]}, ctx)
    PlanTool()({"action": "status", "n": 1, "status": "doing"}, ctx)
    PlanTool()({"action": "add", "step": "b"}, ctx)
    block = render_plan(ctx)
    assert "[~] 1. a" in block and "[ ] 2. b" in block


def test_out_of_range_is_an_error(ctx):
    from pa.errors import ToolError

    PlanTool()({"action": "set", "steps": ["only one"]}, ctx)
    with pytest.raises(ToolError, match="out of range"):
        PlanTool()({"action": "check", "n": 5}, ctx)


def test_clear(ctx):
    PlanTool()({"action": "set", "steps": ["a"]}, ctx)
    PlanTool()({"action": "clear"}, ctx)
    assert render_plan(ctx) == ""


def test_empty_plan_renders_nothing(ctx):
    assert render_plan(ctx) == ""
