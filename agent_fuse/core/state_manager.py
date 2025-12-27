"""
State Manager - Thread-safe singleton for tracking usage state.

Combines in-memory counters for the current session with
persistent storage via SQLite.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_fuse.config import get_settings
from agent_fuse.core.cost_tracker import calculate_cost
from agent_fuse.storage.sqlite import AgentFuseDB, get_db

if TYPE_CHECKING:
    from agent_fuse.config import AgentFuseSettings


@dataclass
class UsageStats:
    """Usage statistics for monitoring."""

    total_spend_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    session_spend_usd: float = 0.0
    session_calls: int = 0
    budget_remaining_usd: float = 0.0
    budget_limit_usd: float = 0.0


@dataclass
class SessionState:
    """In-memory state for the current session."""

    spend_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    start_time: float = field(default_factory=time.time)
    last_call_time: float = 0.0


class StateManager:
    """
    Thread-safe singleton for managing usage state.

    Tracks both in-memory session stats and persisted totals.

    Example:
        >>> manager = StateManager.get_instance()
        >>> manager.add_usage("gpt-4o", 1000, 500)
        >>> stats = manager.get_stats()
        >>> print(f"Spent: ${stats.total_spend_usd:.4f}")
    """

    _instance: StateManager | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, settings: AgentFuseSettings | None = None) -> None:
        """
        Initialize the state manager.

        Note: Use get_instance() for singleton access.
        """
        self._settings = settings or get_settings()
        self._db: AgentFuseDB | None = None
        self._session = SessionState()
        self._session_lock = threading.RLock()

    @classmethod
    def get_instance(cls, settings: AgentFuseSettings | None = None) -> StateManager:
        """
        Get the singleton StateManager instance.

        Thread-safe lazy initialization.

        Args:
            settings: Optional settings override (only used on first call)

        Returns:
            The global StateManager instance
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
    def db(self) -> AgentFuseDB:
        """Get the database instance (lazy initialization)."""
        if self._db is None:
            self._db = get_db()
        return self._db

    @property
    def budget(self) -> float:
        """Get the configured budget limit."""
        return self._settings.budget

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._settings.session_id

    def get_total_spend(self) -> float:
        """
        Get total spend from database.

        Returns:
            Total spend in USD across all sessions
        """
        return self.db.get_total_spend()

    def get_session_spend(self) -> float:
        """
        Get spend for current in-memory session.

        Returns:
            Session spend in USD
        """
        with self._session_lock:
            return self._session.spend_usd

    def get_current_spend(self) -> float:
        """
        Get current spend (session + persisted).

        For budget checks, we use the database total to ensure
        accurate tracking across multiple processes.

        Returns:
            Current total spend in USD
        """
        return self.get_total_spend()

    def check_budget(self, estimated_cost: float) -> bool:
        """
        Check if an estimated cost would exceed the budget.

        Args:
            estimated_cost: Estimated cost of the planned call

        Returns:
            True if within budget, False if would exceed
        """
        if self.budget <= 0:
            # Budget of 0 means unlimited (not recommended)
            return True

        current_spend = self.get_current_spend()
        return (current_spend + estimated_cost) <= self.budget

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: str | None = None,
    ) -> float:
        """
        Record usage after an LLM call.

        Updates both in-memory session state and persistent storage.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Optional JSON metadata

        Returns:
            Cost of this call in USD
        """
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Update in-memory session state
        with self._session_lock:
            self._session.spend_usd += cost
            self._session.input_tokens += input_tokens
            self._session.output_tokens += output_tokens
            self._session.call_count += 1
            self._session.last_call_time = time.time()

        # Persist to database
        self.db.log_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            session_id=self.session_id,
            metadata=metadata,
        )

        return cost

    def get_stats(self) -> UsageStats:
        """
        Get comprehensive usage statistics.

        Returns:
            UsageStats with current usage information
        """
        db_stats = self.db.get_usage_stats(session_id=None)

        with self._session_lock:
            session_spend = self._session.spend_usd
            session_calls = self._session.call_count

        total_spend = db_stats.get("total_spend_usd", 0.0)

        return UsageStats(
            total_spend_usd=total_spend,
            total_input_tokens=db_stats.get("total_input_tokens", 0),
            total_output_tokens=db_stats.get("total_output_tokens", 0),
            total_calls=db_stats.get("total_calls", 0),
            session_spend_usd=session_spend,
            session_calls=session_calls,
            budget_remaining_usd=max(0, self.budget - total_spend),
            budget_limit_usd=self.budget,
        )

    def reset_session(self) -> None:
        """Reset the in-memory session state."""
        with self._session_lock:
            self._session = SessionState()


def get_state_manager() -> StateManager:
    """
    Get the global StateManager instance.

    Convenience function for accessing the singleton.

    Returns:
        The global StateManager instance
    """
    return StateManager.get_instance()
