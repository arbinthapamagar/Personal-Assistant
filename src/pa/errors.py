"""Exception hierarchy. Every failure the agent can recover from is one of these."""


class PAError(Exception):
    """Base class for all Personal Assistant errors."""


class ConfigError(PAError):
    """Configuration is missing, malformed, or contradictory."""


class DependencyError(PAError):
    """An optional dependency is missing and could not be installed."""


class ProviderError(PAError):
    """A model provider failed in a way the agent cannot retry through."""


class AuthError(ProviderError):
    """Credentials are missing or rejected."""


class RateLimitError(ProviderError):
    """Provider asked us to slow down."""


class ToolError(PAError):
    """A tool failed. Surfaced back to the model as an error result."""


class PermissionDenied(ToolError):
    """The user (or policy) refused a tool call."""
