"""Skills: playbooks the agent loads on demand.

A skill is a Markdown file with YAML front matter. Only its one-line
description sits in the system prompt; the body is loaded when the agent
decides it is relevant. That split is the whole point - a dozen full playbooks
would crowd out the conversation, while a dozen descriptions cost almost
nothing and let the model choose.

Three search paths, later ones overriding earlier by name:

1. bundled  - shipped with the package
2. user     - ~/.config/personal-assistant/skills
3. project  - ./.pa/skills, so a repository can carry its own conventions

Write a new one by dropping a file in the user or project directory. No code
change, no restart beyond the next launch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import paths

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    origin: str  # bundled | user | project
    path: Path

    @property
    def summary_line(self) -> str:
        return f"- {self.name}: {self.description}"


def _parse(path: Path, origin: str) -> Skill | None:
    try:
        raw = path.read_text(errors="replace")
    except OSError:
        return None

    name = path.stem
    description = ""
    body = raw

    if match := FRONT_MATTER.match(raw):
        body = raw[match.end():]
        # Deliberately not importing yaml for two flat keys - a skill file with
        # a typo should still load with a degraded description rather than
        # failing the whole session.
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip().strip("\"'")
            if key == "name" and value:
                name = value
            elif key == "description" and value:
                description = value

    if not description:
        # Fall back to the first non-heading line, so a plain Markdown file
        # with no front matter is still usable.
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                description = stripped[:200]
                break

    return Skill(name=name, description=description, body=body.strip(), origin=origin, path=path)


def bundled_dir() -> Path:
    return Path(__file__).parent / "skills"


def user_dir() -> Path:
    return paths.config_dir() / "skills"


def project_dir(cwd: Path) -> Path:
    return cwd / ".pa" / "skills"


class SkillLibrary:
    """Everything loadable, keyed by name."""

    def __init__(self, skills: dict[str, Skill] | None = None) -> None:
        self._skills = skills or {}

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self):
        return iter(sorted(self._skills.values(), key=lambda s: s.name))

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name.strip().lower().removeprefix("/"))

    def names(self) -> list[str]:
        return sorted(self._skills)

    def catalogue(self) -> str:
        """The block that goes in the system prompt: names and descriptions."""
        if not self._skills:
            return ""
        return "\n".join(skill.summary_line for skill in self)

    @classmethod
    def load(cls, cwd: Path | None = None) -> SkillLibrary:
        skills: dict[str, Skill] = {}
        sources = [
            (bundled_dir(), "bundled"),
            (user_dir(), "user"),
            (project_dir(cwd or Path.cwd()), "project"),
        ]
        for directory, origin in sources:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.md")):
                if path.name.lower() == "readme.md":
                    continue  # documentation, not a skill
                if skill := _parse(path, origin):
                    skills[skill.name.strip().lower()] = skill
        return cls(skills)
