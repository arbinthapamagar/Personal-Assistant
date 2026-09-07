"""Deep-research tool tests. The fetch and search layers are stubbed so the
tests are deterministic and offline; what's under test is the tool's own logic:
domain de-duplication, binary skipping, parallel aggregation, citation
numbering, and graceful degradation when pages fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa import capabilities, config as config_mod
from pa.security import Gate
from pa.tools.base import ToolContext
from pa.tools.search import Result
import pa.tools.research as research


@pytest.fixture
def ctx():
    cfg = config_mod.default_config()
    return ToolContext(cfg, capabilities.probe(), Gate(cfg.security), lambda k, d: True)


def _stub_search(results):
    return lambda config, query, limit: results


def _stub_fetch(pages):
    # pages: {url: (text, error)}
    return lambda url, proxy=None: pages.get(url, ("", "not stubbed"))


def test_reads_and_cites_multiple_sources(ctx, monkeypatch):
    results = [
        Result("Alpha", "https://a.com/x", "snippet a"),
        Result("Beta", "https://b.com/y", "snippet b"),
    ]
    pages = {
        "https://a.com/x": ("Alpha page content about cats.", ""),
        "https://b.com/y": ("Beta page content about dogs.", ""),
    }
    monkeypatch.setattr(research, "run_search", _stub_search(results))
    monkeypatch.setattr(research, "_fetch_one", _stub_fetch(pages))

    out = research.WebResearchTool()({"query": "pets", "sources": 2}, ctx)
    assert "[1]" in out and "[2]" in out
    assert "cats" in out and "dogs" in out
    assert "read 2 of 2" in out


def test_deduplicates_by_domain(ctx, monkeypatch):
    # Two results from the same domain - only the first should be read.
    results = [
        Result("One", "https://same.com/a", "s1"),
        Result("Two", "https://same.com/b", "s2"),
        Result("Three", "https://other.com/c", "s3"),
    ]
    pages = {
        "https://same.com/a": ("first from same", ""),
        "https://other.com/c": ("from other", ""),
    }
    monkeypatch.setattr(research, "run_search", _stub_search(results))
    monkeypatch.setattr(research, "_fetch_one", _stub_fetch(pages))

    out = research.WebResearchTool()({"query": "x", "sources": 5}, ctx)
    assert "same.com/a" in out and "same.com/b" not in out
    assert "other.com/c" in out


def test_skips_binary_urls(ctx, monkeypatch):
    results = [
        Result("PDF", "https://x.com/report.pdf", "a pdf"),
        Result("Page", "https://y.com/page", "a page"),
    ]
    pages = {"https://y.com/page": ("real content", "")}
    monkeypatch.setattr(research, "run_search", _stub_search(results))
    monkeypatch.setattr(research, "_fetch_one", _stub_fetch(pages))

    out = research.WebResearchTool()({"query": "x", "sources": 5}, ctx)
    assert "report.pdf" not in out
    assert "real content" in out


def test_failed_pages_fall_back_to_snippet(ctx, monkeypatch):
    results = [Result("Broken", "https://dead.com/x", "the snippet survives")]
    pages = {"https://dead.com/x": ("", "HTTP 503")}
    monkeypatch.setattr(research, "run_search", _stub_search(results))
    monkeypatch.setattr(research, "_fetch_one", _stub_fetch(pages))

    out = research.WebResearchTool()({"query": "x"}, ctx)
    assert "could not read" in out and "503" in out
    assert "read 0 of 1" in out


def test_no_results_is_reported(ctx, monkeypatch):
    monkeypatch.setattr(research, "run_search", _stub_search([]))
    out = research.WebResearchTool()({"query": "nothing"}, ctx)
    assert "no search results" in out


def test_untrusted_header_present(ctx, monkeypatch):
    results = [Result("A", "https://a.com/x", "s")]
    monkeypatch.setattr(research, "run_search", _stub_search(results))
    monkeypatch.setattr(research, "_fetch_one", _stub_fetch({"https://a.com/x": ("content", "")}))
    out = research.WebResearchTool()({"query": "x"}, ctx)
    assert "untrusted" in out.lower() and "do not obey" in out.lower()
