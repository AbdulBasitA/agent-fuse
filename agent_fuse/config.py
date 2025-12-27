"""
Agent Fuse Configuration

Uses Pydantic Settings for environment variable support with sensible defaults.
All settings can be overridden via environment variables prefixed with AGENTFUSE_.
"""

from __future__ import annotations

import os
from typing import cast
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentFuseSettings(BaseSettings):
    """
    Configuration for Agent Fuse.

    All settings can be configured via environment variables:
    - AGENTFUSE_BUDGET: Maximum spend in USD (default: 1.00)
    - AGENTFUSE_FAIL_SAFE: Block on errors if True (default: True)
    - AGENTFUSE_DB_PATH: Path to SQLite database
    - AGENTFUSE_MAX_RETRIES: Max DB retry attempts (default: 3)
    - AGENTFUSE_RETRY_DELAY: Base delay between retries in seconds (default: 0.1)
    - AGENTFUSE_LOG_LEVEL: Logging level (default: WARNING)
    - AGENTFUSE_LOOP_THRESHOLD: Identical calls before loop error (default: 5)
    - AGENTFUSE_LOOP_DETECTION_ENABLED: Enable loop detection (default: True)
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTFUSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Budget Configuration
    budget: float = Field(
        default=1.00,
        ge=0.0,
        description="Maximum budget in USD. Set to 0 for unlimited (not recommended).",
    )

    # Resilience Configuration
    fail_safe: bool = Field(
        default=True,
        description=(
            "If True (default): Block agent on DB errors (prioritize safety). "
            "If False: Log warning and allow agent to proceed (prioritize availability)."
        ),
    )

    # Storage Configuration
    db_path: Path = Field(
        default_factory=lambda: Path.home() / ".agent_fuse" / "guard_v1.db",
        description="Path to the SQLite database file.",
    )

    # Retry Configuration
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts for database operations.",
    )

    retry_delay: float = Field(
        default=0.1,
        ge=0.01,
        le=5.0,
        description="Base delay in seconds between retry attempts (exponential backoff).",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="WARNING",
        description="Logging level for Agent Fuse messages.",
    )

    # Session Configuration
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for grouping usage logs.",
    )

    # Pricing Configuration
    pricing_url: str | None = Field(
        default="https://raw.githubusercontent.com/agentfuse/agent-fuse/main/agent_fuse/core/default_pricing.json",
        description=(
            "URL to fetch latest model pricing. Set to None to disable remote fetching. "
            "Falls back to bundled defaults if fetch fails."
        ),
    )

    pricing_fetch_timeout: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Timeout in seconds for fetching remote pricing.",
    )

    # Loop Detection Configuration
    loop_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of identical tool calls before triggering SentinelLoopError.",
    )

    loop_detection_enabled: bool = Field(
        default=True,
        description="Enable/disable loop detection. When False, check_loop() is a no-op.",
    )

    @field_validator("db_path", mode="before")
    @classmethod
    def expand_db_path(cls, v: str | Path) -> Path:
        """Expand ~ and environment variables in path."""
        if isinstance(v, str):
            v = Path(os.path.expandvars(os.path.expanduser(v)))
        return v

    def ensure_db_directory(self) -> None:
        """Create the database directory if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def configure(
    budget: float | None = None,
    fail_safe: bool | None = None,
    db_path: Path | str | None = None,
    session_id: str | None = None,
    max_retries: int | None = None,
    retry_delay: float | None = None,
    loop_threshold: int | None = None,
    loop_detection_enabled: bool | None = None,
) -> AgentFuseSettings:
    """
    Configure Agent Fuse programmatically.

    This clears the cached settings and creates a new instance with
    the provided overrides. Unspecified values use environment variables
    or defaults.

    Args:
        budget: Maximum spend in USD
        fail_safe: Block on errors if True
        db_path: Path to SQLite database
        session_id: Optional session identifier
        max_retries: Maximum retry attempts for DB operations
        retry_delay: Base delay between retries in seconds
        loop_threshold: Number of identical tool calls before SentinelLoopError
        loop_detection_enabled: Enable/disable loop detection

    Returns:
        The new AgentFuseSettings instance

    Example:
        >>> from agent_fuse.config import configure
        >>> settings = configure(budget=5.00, fail_safe=False)
    """
    overrides = {}
    if budget is not None:
        overrides["budget"] = budget
    if fail_safe is not None:
        overrides["fail_safe"] = fail_safe
    if db_path is not None:
        overrides["db_path"] = Path(db_path) if isinstance(db_path, str) else db_path
    if session_id is not None:
        overrides["session_id"] = session_id
    if max_retries is not None:
        overrides["max_retries"] = max_retries
    if retry_delay is not None:
        overrides["retry_delay"] = retry_delay
    if loop_threshold is not None:
        overrides["loop_threshold"] = loop_threshold
    if loop_detection_enabled is not None:
        overrides["loop_detection_enabled"] = loop_detection_enabled

    # Create new settings with overrides
    settings = AgentFuseSettings(**overrides)

    # Store in cache for get_settings() to return
    _settings_cache[None] = settings

    return settings


# Internal cache for programmatic configuration
_settings_cache: dict[None, AgentFuseSettings] = {}


def get_settings() -> AgentFuseSettings:
    """
    Get settings instance (supports programmatic configuration).

    Returns cached settings if configured programmatically,
    otherwise loads from environment.
    """
    if None in _settings_cache:
        return _settings_cache[None]
    return AgentFuseSettings()


def reset_settings() -> None:
    """Reset settings to defaults (clears programmatic configuration)."""
    _settings_cache.clear()
