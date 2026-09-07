"""The permission gate.

Every tool call is reduced to a single string key - `shell:rm -rf /`,
`files.write:/etc/passwd`, `browser.click:#buy-now` - and that key is matched
against three layers, most restrictive first:

  1. hard denials  - patterns that are refused no matter what the config says
  2. config deny   - the user's own blocklist
  3. config allow  - the user's own passlist, plus this session's approvals

Anything unmatched falls through to the configured default mode. The hard
denials exist because an agent with shell access is one bad inference away from
an unrecoverable machine, and a config file is too easy to loosen by accident.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import Security


class Verdict(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class Decision:
    verdict: Verdict
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW


# Patterns refused unless the config explicitly unlocks them by id. Each entry
# is (id, regex, reason). These are a floor, not a cage: it is the user's
# machine, so any of them can be disabled with `security.unrestrict: [id, ...]`
# in the config. That is a deliberate, auditable act - unlike loosening an
# allow-list by accident, which is what the floor exists to prevent.
HARD_DENY_RULES: list[tuple[str, str, str]] = [
    ("rm-root", r"rm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|/\*|~|~/\*|\$HOME)\s*$", "recursive delete of home or root"),
    ("mkfs", r"\bmkfs(\.\w+)?\b", "filesystem format"),
    ("dd-device", r"\bdd\b.*\bof=/dev/(sd|nvme|vd|hd)", "raw write to a block device"),
    ("redirect-device", r">\s*/dev/(sd|nvme|vd|hd)\w*", "raw write to a block device"),
    ("fork-bomb", r":\(\)\s*\{.*\}\s*;?\s*:", "fork bomb"),
    ("chmod-root", r"\bchmod\s+-R\s+777\s+/\s*$", "world-writable root"),
    ("power", r"\b(shutdown|reboot|poweroff|halt)\b", "power state change"),
    ("history", r"\bhistory\s+-c\b|\brm\b.*\.bash_history", "shell history tampering"),
    ("curl-pipe-sh", r"\bcurl\b[^|]*\|\s*(sudo\s+)?(ba)?sh", "pipe-from-internet to shell"),
    ("wget-pipe-sh", r"\bwget\b[^|]*\|\s*(sudo\s+)?(ba)?sh", "pipe-from-internet to shell"),
    ("account", r"\b(userdel|groupdel|passwd)\b", "account modification"),
    ("force-push", r"\bgit\b.*\bpush\b.*--force", "force push"),
    ("system-creds", r"/etc/(shadow|sudoers)", "system credential file"),
    ("ssh-key", r"\.ssh/id_(rsa|ed25519|ecdsa)(?!\.pub)", "private SSH key"),
    ("cloud-creds", r"\.aws/credentials|\.config/gcloud|\.kube/config", "cloud credential file"),
]


@dataclass
class Gate:
    """Evaluates tool calls against policy. One instance per session."""

    policy: Security
    session_allow: list[str] = field(default_factory=list)
    session_deny: list[str] = field(default_factory=list)
    #: Keys already decided this session, for the audit trail.
    log: list[tuple[str, Verdict, str]] = field(default_factory=list)
    _hard: list[tuple[Any, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Compile the denial floor once, dropping any rule the config unlocks
        # via security.unrestrict. Disabling a rule is a deliberate, auditable
        # act; loosening an allow-list by accident is the mistake the floor
        # exists to catch, which is why the two are separate mechanisms.
        disabled = set(getattr(self.policy, "unrestrict", None) or [])
        # "all"/"*" removes the entire floor - a single deliberate opt-out for a
        # user who wants zero oversight on their own machine.
        self.floor_off = bool(disabled & {"all", "*"})
        self._hard = [] if self.floor_off else [
            (re.compile(pattern, re.IGNORECASE), rule_id, why)
            for rule_id, pattern, why in HARD_DENY_RULES
            if rule_id not in disabled
        ]

    @property
    def disabled_rules(self) -> list[str]:
        active = {rid for _, rid, _ in self._hard}
        return [rid for rid, _, _ in HARD_DENY_RULES if rid not in active]

    def decide(self, key: str) -> Decision:
        decision = self._decide(key)
        self.log.append((key, decision.verdict, decision.reason))
        return decision

    def _decide(self, key: str) -> Decision:
        for pattern, rule_id, why in self._hard:
            if pattern.search(key):
                return Decision(Verdict.DENY, f"blocked [{rule_id}]: {why}")

        for pattern in self.session_deny:
            if fnmatch.fnmatch(key, pattern):
                return Decision(Verdict.DENY, f"denied this session ({pattern})")

        for pattern in self.policy.deny:
            if fnmatch.fnmatch(key, pattern):
                return Decision(Verdict.DENY, f"denied by config ({pattern})")

        for pattern in self.session_allow:
            if fnmatch.fnmatch(key, pattern):
                return Decision(Verdict.ALLOW, f"approved this session ({pattern})")

        for pattern in self.policy.allow:
            if fnmatch.fnmatch(key, pattern):
                return Decision(Verdict.ALLOW, f"allowed by config ({pattern})")

        mode = self.policy.mode
        if mode == "allow":
            return Decision(Verdict.ALLOW, "mode: allow")
        if mode == "deny":
            return Decision(Verdict.DENY, "mode: deny - no tool runs without an allow rule")
        return Decision(Verdict.ASK, "no rule matched")

    def remember_allow(self, pattern: str) -> None:
        """Approve a pattern for the rest of this session only."""
        if pattern not in self.session_allow:
            self.session_allow.append(pattern)

    def remember_deny(self, pattern: str) -> None:
        if pattern not in self.session_deny:
            self.session_deny.append(pattern)

    def audit(self) -> str:
        if not self.log:
            return "no tool calls yet"
        counts: dict[Verdict, int] = {}
        for _, verdict, _ in self.log:
            counts[verdict] = counts.get(verdict, 0) + 1
        head = "  ".join(f"{v.value}={n}" for v, n in counts.items())
        recent = "\n".join(
            f"  {v.value:5} {k[:90]}" for k, v, _ in self.log[-15:]
        )
        return f"{head}\n{recent}"


def generalize(key: str) -> str:
    """Suggest a reusable glob for a one-off key, for 'always allow' prompts.

    `shell:git status --short` -> `shell:git status*`
    `files.read:/home/a/x.py`  -> `files.read:*`
    """
    tool, _, detail = key.partition(":")
    if not detail:
        return f"{tool}*"
    if tool.startswith("files.") or tool.startswith("browser."):
        return f"{tool}:*"
    words = detail.split()
    stem = " ".join(words[:2]) if len(words) > 1 else words[0] if words else ""
    return f"{tool}:{stem}*" if stem else f"{tool}:*"
