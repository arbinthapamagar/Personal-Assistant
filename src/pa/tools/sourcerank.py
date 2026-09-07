"""Rank web results by authority, so research leads with official sources.

The goal is to answer from the primary source - the project's own docs, the
standards body, the vendor, the government agency - the way a careful
researcher does, rather than from whatever SEO-optimised blog happens to rank
first. This is a heuristic scorer, not a search engine: it reorders results the
backend already returned.

Signals, highest first:
* the query's own subject appearing as the domain (asking about "ffmpeg" ->
  ffmpeg.org is almost certainly the official site)
* documentation hosts (docs.*, *.readthedocs.io, developer.*, /docs/ paths)
* primary-source TLDs and hosts (.gov, .edu, standards bodies, package indexes)
* the vendor/official domain for well-known projects
Penalised: content farms, SEO aggregators, and scraper sites.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Hosts (or host suffixes) that are authoritative by nature.
_AUTHORITATIVE = {
    # standards / reference
    "w3.org": 6, "ietf.org": 6, "rfc-editor.org": 6, "iso.org": 5,
    "unicode.org": 5, "khronos.org": 5, "ecma-international.org": 5,
    # code / packages (the project's real home)
    "github.com": 4, "gitlab.com": 3, "pypi.org": 5, "npmjs.com": 5,
    "crates.io": 5, "pkg.go.dev": 5, "readthedocs.io": 5, "rubygems.org": 5,
    # vendor docs
    "developer.mozilla.org": 6, "developer.apple.com": 5, "learn.microsoft.com": 5,
    "cloud.google.com": 5, "docs.aws.amazon.com": 5, "kubernetes.io": 5,
    "python.org": 6, "nodejs.org": 5, "postgresql.org": 5, "kernel.org": 6,
    "wikipedia.org": 3,
}
# TLD-level authority.
_TLD_SCORE = {".gov": 5, ".edu": 4, ".int": 4, ".mil": 4}
# Known content farms / low-authority aggregators, penalised.
_LOW = {
    "w3schools.com": -2, "geeksforgeeks.org": -2, "tutorialspoint.com": -3,
    "javatpoint.com": -3, "medium.com": -1, "quora.com": -2, "pinterest.com": -4,
    "answers.com": -3, "coursehero.com": -3, "scribd.com": -3,
}
_DOC_HINT = re.compile(r"\b(docs?|documentation|reference|manual|developer|api)\b", re.I)
_STOP = {"the", "a", "an", "of", "for", "how", "to", "what", "is", "are", "in",
         "on", "and", "or", "do", "does", "with", "using", "use", "official",
         "site", "website", "com", "org", "www"}


def _query_terms(query: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9.+-]{2,}", query.lower())
    return {w for w in words if w not in _STOP}


def score(url: str, title: str, query: str) -> float:
    """Authority score for one result. Higher is more official."""
    host = urlparse(url if "://" in url else f"http://{url}").netloc.lower()
    host = host.removeprefix("www.")
    if not host:
        return 0.0
    s = 0.0

    # Exact/suffix authoritative hosts.
    for known, weight in _AUTHORITATIVE.items():
        if host == known or host.endswith("." + known):
            s += weight
            break
    for known, penalty in _LOW.items():
        if host == known or host.endswith("." + known):
            s += penalty
            break
    for tld, weight in _TLD_SCORE.items():
        if host.endswith(tld):
            s += weight
            break

    # Documentation subdomains / paths read as primary sources.
    if host.startswith(("docs.", "developer.", "dev.", "api.")):
        s += 4
    if _DOC_HINT.search(url):
        s += 1.5

    # The subject of the query appearing in the domain is a strong "official
    # site" signal: "ffmpeg codecs" -> ffmpeg.org, "django orm" -> djangoproject.com.
    terms = _query_terms(query)
    domain_word = host.split(".")[0]
    for term in terms:
        if len(term) >= 3 and (term == domain_word or term in host.split(".")):
            s += 5
            break
        if len(term) >= 4 and term in domain_word:
            s += 3
            break

    # A shallow path (the root or a top section) is usually more canonical than
    # a deep, dated article URL.
    depth = urlparse(url).path.strip("/").count("/")
    if depth == 0:
        s += 1.0
    elif depth >= 4:
        s -= 1.0

    return s


def rerank(results, query: str):
    """Stable-sort results by authority, keeping original order as tiebreak."""
    scored = [(i, score(r.url, r.title, query), r) for i, r in enumerate(results)]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [r for _, _, r in scored]
