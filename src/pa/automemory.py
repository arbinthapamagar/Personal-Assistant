"""Automatic memory: recall relevant context, capture durable facts.

The memory *tools* (memory_save / memory_search) only fire when the model
chooses to call them - which a weak local model rarely does, and even a strong
one forgets. Automatic memory closes that gap without a model call:

* **Recall** - before each turn, the user's message is embedded and matched
  against saved notes; strong matches are injected into that turn's system
  prompt as "things you remember". Retrieval-augmented, every turn, for free.
* **Capture** - after a turn, the user's message is scanned for the handful of
  phrasings that reliably signal a durable fact ("my name is", "remember that",
  "I prefer", ...) and those are saved. Deterministic and precise: it captures
  what the user explicitly stated, not the model's guess about what mattered,
  so it almost never saves noise.

Both are deliberately model-free. That keeps them cheap, private, and reliable
on a local setup - the "learning" is retrieval, not fine-tuning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern captures the memorable clause in group 1. Anchored to the start
# of a sentence so a passing mention ("I don't remember that") does not match.
# Kept intentionally narrow - a false save is worse than a missed one, because
# it pollutes recall for every future turn.
_CAPTURE_PATTERNS = [
    # Capture a single name token (case-insensitive, so lowercase "arbin"
    # works). One token only, so "my name is arbin and i..." grabs just the
    # name, not the trailing clause.
    (re.compile(r"\bmy name is\s+([A-Za-z][\w'-]*)", re.I), "preference"),
    (re.compile(r"\bcall me\s+([A-Za-z][\w'-]*)", re.I), "preference"),
    # Imperative "remember ..." anchored to the clause start, so "I don't
    # remember", "can't remember", etc. never match.
    (re.compile(r"^(?:please\s+)?remember\b(?:\s+that)?\s+(.+?)(?:[.!?]|$)", re.I), "note"),
    (re.compile(r"^note that\s+(.+?)(?:[.!?]|$)", re.I), "note"),
    (re.compile(r"\bfor future reference[,:]?\s+(.+?)(?:[.!?]|$)", re.I), "note"),
    (re.compile(
        r"\bI (?:prefer|like|want|always|usually|never|don't (?:like|want|use))\s+(.+?)(?:[.!?]|$)",
        re.I,
    ), "preference"),
    (re.compile(r"\bI (?:use|work with|work on|run)\s+(.+?)(?:[.!?]|$)", re.I), "preference"),
    (re.compile(r"\bI (?:am|'m)\s+(a\s+.+?|an\s+.+?)(?:[.!?]|$)", re.I), "preference"),
]

# Never capture a sentence whose "remember"-ish verb is negated - a guard on
# top of the anchoring above, for the preference patterns too.
_NEGATED = re.compile(r"\b(?:don't|do not|can't|cannot|won't|didn't|never)\s+remember\b", re.I)

# If any of these appear, skip capture: the user is asking or hypothesising,
# not stating a durable fact.
_QUESTION_HINT = re.compile(r"\?|^(what|who|when|where|why|how|do|does|is|are|can|could|should)\b", re.I)

_MIN_FACT_LEN = 3
_MAX_FACT_LEN = 200


@dataclass
class Captured:
    text: str
    kind: str


def extract_memorable(user_text: str) -> list[Captured]:
    """Pull explicitly-stated durable facts out of a user message.

    Returns at most a few, de-duplicated. Empty for questions, commands, and
    anything that does not match a clear "here is a fact about me" phrasing.
    """
    text = (user_text or "").strip()
    if not text or len(text) > 2000:
        return []

    out: list[Captured] = []
    seen: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        if not sentence or _QUESTION_HINT.search(sentence) or _NEGATED.search(sentence):
            continue
        # Try every pattern, not just the first: a compound sentence like
        # "My name is Arbin and I prefer verbose output" holds two facts.
        for pattern, kind in _CAPTURE_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            fact = match.group(1).strip().rstrip(".!?, ")
            statement = _phrase(pattern.pattern, fact, sentence)
            key = statement.lower()
            if _MIN_FACT_LEN <= len(fact) <= _MAX_FACT_LEN and key not in seen:
                seen.add(key)
                out.append(Captured(statement, kind))
    return out[:3]


def _phrase(pattern_src: str, fact: str, sentence: str) -> str:
    """Turn a match into a stored statement. For the strongly-shaped ones we
    rewrite to a clean fact; otherwise keep the user's own sentence, which
    preserves meaning better than a fragment."""
    if "name is" in pattern_src or "all me" in pattern_src:
        return f"The user's name is {fact[:1].upper() + fact[1:]}."
    # For preference/note patterns the whole sentence is the clearest record.
    return sentence.rstrip()


def format_recall(hits) -> str:
    """Render recalled notes as a compact block for the system prompt."""
    if not hits:
        return ""
    lines = ["Things you remember about this user or their work (from past "
             "sessions - use if relevant, ignore if not):"]
    for hit in hits:
        lines.append(f"- {hit.text.strip()}")
    return "\n".join(lines)
