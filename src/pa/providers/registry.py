"""Provider lookup.

Adapters are imported lazily so that a machine with only `ollama` installed
never imports - or installs - the Anthropic or OpenAI SDKs.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ..config import Config, Profile
from ..errors import ConfigError

if TYPE_CHECKING:
    from .base import Provider

# registry key -> "module:class" within this package
PROVIDERS = {
    "anthropic": "anthropic_provider:AnthropicProvider",
    "openai": "openai_provider:OpenAIProvider",
    "openai-compat": "openai_provider:OpenAICompatProvider",
    "google": "google_provider:GoogleProvider",
    "ollama": "ollama_provider:OllamaProvider",
    "mock": "mock_provider:MockProvider",
}


def build(profile: Profile, config: Config) -> Provider:
    """Instantiate the adapter a profile names."""
    target = PROVIDERS.get(profile.provider)
    if target is None:
        raise ConfigError(
            f"no adapter for provider {profile.provider!r}; known: {', '.join(sorted(PROVIDERS))}"
        )
    mod_name, cls_name = target.split(":")
    module = importlib.import_module(f"{__package__}.{mod_name}")
    cls = getattr(module, cls_name)
    return cls(profile, config)
