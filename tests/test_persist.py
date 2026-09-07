"""Config persistence: a /profile or /model switch is remembered next launch,
via targeted line edits that keep the user's comments intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pa import config as C

SAMPLE = """\
# my assistant config
active_profile: gemini

profiles:
  gemini:
    provider: google
    model: gemini-flash-latest
  dolphin:
    provider: ollama
    model: dolphin-mistral

harness: auto
"""


def test_persist_active_profile(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(SAMPLE)
    assert C.persist_selection("dolphin", path=p) is True
    s = p.read_text()
    assert "active_profile: dolphin" in s
    assert "active_profile: gemini" not in s
    assert "# my assistant config" in s          # comment preserved
    assert "harness: auto" in s                   # unrelated keys preserved


def test_persist_reloads_as_default(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(SAMPLE)
    C.persist_selection("dolphin", path=p)
    cfg = C.load(p)
    assert cfg.active_profile == "dolphin"        # next launch starts here


def test_persist_model_updates_only_that_block(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(SAMPLE)
    C.persist_selection("dolphin", model="dolphin-llama3", path=p)
    s = p.read_text()
    assert "model: dolphin-llama3" in s
    assert "model: gemini-flash-latest" in s      # gemini's model untouched


def test_persist_creates_file_if_missing(tmp_path):
    p = tmp_path / "new.yaml"
    assert C.persist_selection("local", path=p) is True
    assert "active_profile: local" in p.read_text()
