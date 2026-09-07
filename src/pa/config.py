"""Layered configuration.

Precedence, lowest to highest: built-in defaults, the config file, environment
variables, then explicit CLI overrides. A machine with no config file at all
still starts, which is what makes `git clone && install` work anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from . import paths
from .errors import ConfigError

# Sensible model per provider, used when a profile names no model.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5",
    "google": "gemini-2.5-pro",
    "ollama": "qwen2.5-coder:14b",
    "openai-compat": "",
    "mock": "mock-1",
}

# Where each provider looks for credentials, and where it talks by default.
PROVIDER_DEFAULTS = {
    "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "base_url": None},
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": None},
    "google": {"api_key_env": "GEMINI_API_KEY", "base_url": None},
    "ollama": {"api_key_env": None, "base_url": "http://localhost:11434"},
    "openai-compat": {"api_key_env": "PA_API_KEY", "base_url": None},
    "mock": {"api_key_env": None, "base_url": None},
}


@dataclass
class Profile:
    """One (provider, model, settings) combination the user can switch to by name."""

    name: str
    provider: str
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    max_tokens: int = 16000
    effort: str = "high"
    temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDER_DEFAULTS:
            known = ", ".join(sorted(PROVIDER_DEFAULTS))
            raise ConfigError(
                f"profile {self.name!r}: unknown provider {self.provider!r} (known: {known})"
            )
        defaults = PROVIDER_DEFAULTS[self.provider]
        self.model = self.model or DEFAULT_MODELS[self.provider]
        self.base_url = self.base_url or defaults["base_url"]
        self.api_key_env = self.api_key_env or defaults["api_key_env"]

    def resolve_key(self) -> str | None:
        """An explicit key wins; otherwise read the environment."""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class Security:
    """Which tool calls run unattended, and which stop for a human."""

    mode: str = "ask"  # ask | allow | deny
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    #: Built-in denial rule ids to switch OFF (see security.HARD_DENY_RULES).
    #: Deliberately unlocks a specific dangerous pattern; empty by default.
    unrestrict: list[str] = field(default_factory=list)
    workspace: str | None = None
    confirm_writes_outside_workspace: bool = True
    max_output_bytes: int = 200_000


@dataclass
class Config:
    active_profile: str = "claude"
    profiles: dict[str, Profile] = field(default_factory=dict)
    tools: list[str] = field(
        default_factory=lambda: [
            "shell", "files", "web", "skills", "tasks", "memory",
            "agents", "voice", "browser", "desktop",
        ]
    )
    security: Security = field(default_factory=Security)
    system_prompt: str | None = None
    max_steps: int = 40
    auto_install_deps: bool = True
    stream: bool = True
    #: How many tool calls may run at once when the model requests a batch.
    max_parallel_tools: int = 8
    #: "auto" prefers redis and falls back to sqlite; or force "sqlite"/"redis".
    state_backend: str = "auto"
    redis_url: str = "redis://localhost:6379/0"
    #: Web-search backend config: {backend, base_url?, api_key?, api_key_env?}.
    search: dict[str, Any] = field(default_factory=lambda: {"backend": "duckduckgo"})
    #: Voice config: {tts_backend, voice_model?, stt_model, rate, silence_end, ...}.
    voice: dict[str, Any] = field(default_factory=dict)
    #: Speak assistant replies aloud automatically in interactive mode.
    speak_replies: bool = False
    #: Tool-calling strategy: "auto" (native, self-healing to prompted for weak
    #: local models), "native", or "prompted".
    harness: str = "auto"
    #: Cap on tools shown to the model. Small local models drown in a big tool
    #: list; 0 means no cap. Applied by the agent when building the prompt.
    max_tools: int = 0
    #: Curated tool groups used when the active provider is local, unless the
    #: user set `tools` explicitly. Keeps weak models focused.
    local_tools: list[str] = field(
        default_factory=lambda: ["shell", "files", "web", "tasks"]
    )

    @property
    def profile(self) -> Profile:
        if self.active_profile not in self.profiles:
            available = ", ".join(sorted(self.profiles)) or "<none>"
            raise ConfigError(
                f"active profile {self.active_profile!r} is not defined (have: {available})"
            )
        return self.profiles[self.active_profile]

    def with_profile(self, name: str) -> Config:
        return replace(self, active_profile=name)


def default_config() -> Config:
    """What you get with no config file - every provider we support, ready to switch to."""
    profiles = {
        "claude": Profile("claude", "anthropic"),
        "gpt": Profile("gpt", "openai"),
        "gemini": Profile("gemini", "google"),
        "local": Profile("local", "ollama"),
    }
    return Config(active_profile="claude", profiles=profiles)


def _profiles_from_raw(raw: dict[str, Any]) -> dict[str, Profile]:
    out: dict[str, Profile] = {}
    for name, body in (raw or {}).items():
        if not isinstance(body, dict):
            raise ConfigError(f"profile {name!r} must be a mapping, got {type(body).__name__}")
        provider = body.get("provider")
        if not provider:
            raise ConfigError(f"profile {name!r} is missing 'provider'")
        known = {f for f in Profile.__dataclass_fields__ if f not in ("name", "extra")}
        kwargs = {k: v for k, v in body.items() if k in known}
        extra = {k: v for k, v in body.items() if k not in known}
        out[name] = Profile(name=name, extra=extra, **kwargs)
    return out


def load(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Read the config file if present, layer env vars and overrides on top."""
    cfg = default_config()
    path = path or paths.config_file()

    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")

        if raw.get("profiles"):
            # File profiles merge over the built-ins, so a partial file still works.
            cfg.profiles = {**cfg.profiles, **_profiles_from_raw(raw["profiles"])}
        if "active_profile" in raw:
            cfg.active_profile = str(raw["active_profile"])
        if "tools" in raw:
            tools = raw["tools"]
            cfg.tools = tools.get("enabled", cfg.tools) if isinstance(tools, dict) else list(tools)
            if isinstance(tools, dict) and tools.get("workspace"):
                cfg.security.workspace = tools["workspace"]
        if isinstance(raw.get("search"), dict):
            cfg.search = {**cfg.search, **raw["search"]}
        if isinstance(raw.get("voice"), dict):
            cfg.voice = {**cfg.voice, **raw["voice"]}
        if "speak_replies" in raw:
            cfg.speak_replies = bool(raw["speak_replies"])
        if isinstance(raw.get("security"), dict):
            sec = raw["security"]
            known = set(Security.__dataclass_fields__)
            unknown = set(sec) - known
            if unknown:
                raise ConfigError(f"{path}: unknown security keys: {', '.join(sorted(unknown))}")
            cfg.security = Security(**{**cfg.security.__dict__, **sec})
        for key in (
            "system_prompt", "max_steps", "auto_install_deps", "stream",
            "max_parallel_tools", "state_backend", "redis_url",
            "harness", "max_tools",
        ):
            if key in raw:
                setattr(cfg, key, raw[key])

    # Environment overrides - handy for CI, containers, and throwaway machines.
    if env_profile := os.environ.get("PA_PROFILE"):
        cfg.active_profile = env_profile
    if env_model := os.environ.get("PA_MODEL"):
        cfg.profiles[cfg.active_profile].model = env_model
    if os.environ.get("PA_YOLO") == "1":
        cfg.security.mode = "allow"

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "profile":
            cfg.active_profile = value
        elif key == "model":
            cfg.profiles[cfg.active_profile].model = value
        elif key == "mode":
            cfg.security.mode = value
        else:
            setattr(cfg, key, value)

    cfg.profile  # validate the selection eagerly, so errors surface at startup
    return cfg


def write_example(path: Path) -> Path:
    """Drop a commented starter config next to the user's other config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE)
    return path


EXAMPLE = """\
# Personal Assistant configuration.
# Switch provider live with `/profile <name>`, or at launch with `pa --profile <name>`.

active_profile: claude

profiles:
  claude:
    provider: anthropic
    model: claude-opus-5
    effort: high            # low | medium | high | xhigh | max

  gpt:
    provider: openai
    model: gpt-5

  gemini:
    provider: google
    model: gemini-2.5-pro

  local:                    # no API key, no network, runs on this machine
    provider: ollama
    model: qwen2.5-coder:14b
    base_url: http://localhost:11434

  groq:                     # any OpenAI-compatible endpoint
    provider: openai-compat
    model: llama-3.3-70b-versatile
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY

tools:
  enabled: [shell, files, web, skills, tasks, memory, agents, voice, browser, desktop]
  workspace: ~/             # writes outside this need confirmation

security:
  mode: ask                 # ask | allow | deny
  allow:                    # glob patterns that never prompt
    - "files.read:*"
    - "shell:git status*"
    - "shell:ls*"
  deny:                     # patterns that are always refused
    - "shell:*rm -rf /*"
    - "shell:*mkfs*"
    - "shell:*dd if=*of=/dev/*"

max_steps: 40               # tool-call budget per user turn
max_parallel_tools: 8       # how many tool calls run at once in a batch
harness: auto               # auto | native | prompted - how tools are called
                            #   auto self-heals to prompted for weak local models
max_tools: 0                # cap tools shown to the model (0 = no cap); small
                            #   local models do better with a handful
auto_install_deps: true     # fetch optional packages on first use
stream: true

# Shared state for background tasks and cross-session locks.
state_backend: auto         # auto (redis if reachable, else sqlite) | sqlite | redis
redis_url: redis://localhost:6379/0

# Web search backend for the web_search tool.
search:
  backend: duckduckgo       # duckduckgo (no key) | searxng | brave
  # base_url: http://localhost:8888     # for searxng
  # api_key_env: BRAVE_API_KEY          # for brave

# Voice: local speech in and out. Talk with `pa --voice`, or /voice in the REPL.
speak_replies: false        # also speak every reply aloud in text mode
voice:
  tts_backend: auto         # auto (piper if a voice model exists, else espeak) | piper | espeak
  # voice_model: ~/.local/share/personal-assistant/voices/en_US-amy-medium.onnx
  stt_model: base           # faster-whisper size: tiny | base | small | medium
  rate: 175                 # espeak words-per-minute (ignored by piper)
  silence_end: 1.2          # seconds of quiet that end a spoken turn
  max_seconds: 30
"""
