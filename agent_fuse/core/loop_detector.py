"""
Loop Detector - Detects repeated tool/action patterns to prevent infinite loops.

Tracks tool call fingerprints (tool_name + canonical args) and raises
SentinelLoopError when the same action is repeated more than the configured threshold.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from agent_fuse.config import get_settings
from agent_fuse.core.exceptions import SentinelLoopError

if TYPE_CHECKING:
    from agent_fuse.config import AgentFuseSettings

logger = logging.getLogger("agent_fuse.loop_detector")


class LoopDetector:
    """
    Thread-safe singleton for detecting tool call loops.

    Tracks tool call fingerprints in-memory and raises SentinelLoopError
    when the same call is made more than the configured threshold.

    Example:
        >>> detector = LoopDetector.get_instance()
        >>> detector.check("search_web", {"query": "python"})  # OK
        >>> detector.check("search_web", {"query": "python"})  # OK
        >>> # ... after threshold exceeded ...
        >>> detector.check("search_web", {"query": "python"})  # Raises!
    """

    _instance: LoopDetector | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, settings: AgentFuseSettings | None = None) -> None:
        """
        Initialize the loop detector.

        Args:
            settings: Optional settings override (uses get_settings() if None)
        """
        self._settings = settings or get_settings()
        # Structure: {session_id: {signature: count}}
        self._call_counts: dict[str | None, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._data_lock = threading.RLock()

    @classmethod
    def get_instance(cls, settings: AgentFuseSettings | None = None) -> LoopDetector:
        """
        Get the singleton LoopDetector instance (thread-safe).

        Args:
            settings: Optional settings override for first initialization

        Returns:
            The singleton LoopDetector instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(settings)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None

    @property
    def threshold(self) -> int:
        """Get the configured loop threshold."""
        return self._settings.loop_threshold

    @property
    def enabled(self) -> bool:
        """Check if loop detection is enabled."""
        return self._settings.loop_detection_enabled

    def _create_signature(self, tool_name: str, args: dict[str, Any] | None) -> str:
        """
        Create a canonical signature for a tool call.

        Args:
            tool_name: Name of the tool being called
            args: Arguments passed to the tool

        Returns:
            Canonical string signature for the call
        """
        # Sort keys for consistent hashing regardless of dict order
        canonical_args = json.dumps(args or {}, sort_keys=True, separators=(",", ":"))
        return f"{tool_name}|{canonical_args}"

    def check(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Check if a tool call would create a loop.

        Records the call and raises SentinelLoopError if the same call
        has been made more than threshold times.

        Args:
            tool_name: Name of the tool being called
            args: Arguments passed to the tool
            session_id: Optional session override (uses settings.session_id if None)

        Raises:
            SentinelLoopError: If loop threshold is exceeded
        """
        if not self.enabled:
            return

        # Use provided session_id or fall back to settings
        effective_session = session_id or self._settings.session_id

        signature = self._create_signature(tool_name, args)

        with self._data_lock:
            self._call_counts[effective_session][signature] += 1
            count = self._call_counts[effective_session][signature]

        logger.debug(
            "Loop check: %s (session=%s, count=%d/%d)",
            signature[:50],
            effective_session,
            count,
            self.threshold,
        )

        if count > self.threshold:
            logger.warning(
                "Loop detected: '%s' called %d times (threshold: %d)",
                tool_name,
                count,
                self.threshold,
            )
            raise SentinelLoopError(
                f"Loop detected: '{tool_name}' called {count} times with identical args",
                call_count=count,
                time_window=0.0,  # Not using time-based detection
                pattern=signature,
            )

    def reset_session(self, session_id: str | None = None) -> None:
        """
        Reset call counts for a session.

        Args:
            session_id: Session to reset (uses settings.session_id if None)
        """
        effective_session = session_id or self._settings.session_id

        with self._data_lock:
            if effective_session in self._call_counts:
                del self._call_counts[effective_session]

        logger.debug("Reset loop counts for session: %s", effective_session)

    def reset_all(self) -> None:
        """Reset all call counts for all sessions."""
        with self._data_lock:
            self._call_counts.clear()

        logger.debug("Reset all loop counts")

    def get_counts(self, session_id: str | None = None) -> dict[str, int]:
        """
        Get current call counts for a session (for debugging/monitoring).

        Args:
            session_id: Session to query (uses settings.session_id if None)

        Returns:
            Dict mapping signatures to call counts
        """
        effective_session = session_id or self._settings.session_id

        with self._data_lock:
            return dict(self._call_counts.get(effective_session, {}))


# Module-level singleton accessor
def get_loop_detector() -> LoopDetector:
    """Get the global LoopDetector instance."""
    return LoopDetector.get_instance()


def reset_loop_detector() -> None:
    """Reset the global LoopDetector instance."""
    LoopDetector.reset_instance()


def check_loop(
    tool_name: str,
    args: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> None:
    """
    Module-level loop check.

    Convenience function that uses the global loop detector.
    See LoopDetector.check for details.

    Args:
        tool_name: Name of the tool being called
        args: Arguments passed to the tool (will be canonicalized)
        session_id: Optional session override

    Raises:
        SentinelLoopError: If loop threshold is exceeded

    Example:
        >>> from agent_fuse import check_loop
        >>> check_loop("search_web", {"query": "python"})  # OK first time
        >>> # ... after 5 identical calls ...
        >>> check_loop("search_web", {"query": "python"})  # Raises!
    """
    return get_loop_detector().check(tool_name, args, session_id)
