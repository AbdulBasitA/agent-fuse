"""
Sentinel Guard Exceptions

Custom exceptions for circuit breaker behavior and error handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decimal import Decimal


class SentinelError(Exception):
    """Base exception for all Sentinel Guard errors."""

    pass


class SentinelBudgetExceeded(SentinelError):
    """
    Raised when an LLM call would exceed the configured budget.

    This is the primary circuit breaker exception. When raised, the agent
    should stop making LLM calls to prevent runaway costs.

    Attributes:
        current_spend: Total amount spent so far in USD
        estimated_cost: Estimated cost of the blocked call in USD
        budget_limit: Configured budget limit in USD
        model: The model that was being called
    """

    def __init__(
        self,
        message: str,
        *,
        current_spend: float,
        estimated_cost: float,
        budget_limit: float,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.current_spend = current_spend
        self.estimated_cost = estimated_cost
        self.budget_limit = budget_limit
        self.model = model

    def __str__(self) -> str:
        return (
            f"Budget exceeded: ${self.current_spend:.4f} spent + "
            f"${self.estimated_cost:.4f} estimated = "
            f"${self.current_spend + self.estimated_cost:.4f} "
            f"(limit: ${self.budget_limit:.2f})"
        )

    @property
    def overage(self) -> float:
        """Amount by which the call would exceed the budget."""
        return (self.current_spend + self.estimated_cost) - self.budget_limit


class SentinelLoopError(SentinelError):
    """
    Raised when a suspected infinite loop is detected.

    This exception is raised when the agent appears to be stuck in a loop,
    making the same or similar calls repeatedly without progress.

    Attributes:
        call_count: Number of similar calls detected
        time_window: Time window in seconds over which calls were counted
        pattern: The tool call signature that was repeated (tool_name|args)
    """

    def __init__(
        self,
        message: str,
        *,
        call_count: int,
        time_window: float,
        pattern: str | None = None,
    ) -> None:
        super().__init__(message)
        self.call_count = call_count
        self.time_window = time_window
        self.pattern = pattern

    def __str__(self) -> str:
        base = f"Loop detected: {self.call_count} identical calls"
        if self.pattern:
            base += f"\nSignature: {self.pattern}"
        return base

    @property
    def signature(self) -> str | None:
        """Alias for pattern - the tool call signature that triggered the loop."""
        return self.pattern


class SentinelSystemError(SentinelError):
    """
    Raised when Sentinel Guard encounters a system error in FAIL_SAFE mode.

    This exception is raised when the database is unreachable, locked, or
    otherwise unavailable AND the FAIL_SAFE configuration is True.

    When FAIL_SAFE is False, system errors are logged but the agent is
    allowed to proceed (at the risk of untracked spending).

    Attributes:
        operation: The operation that failed (e.g., "connect", "write", "read")
        original_error: The underlying exception that caused the failure
        retry_count: Number of retries attempted before failing
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        original_error: Exception | None = None,
        retry_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error
        self.retry_count = retry_count

    def __str__(self) -> str:
        base = f"Sentinel system error during '{self.operation}'"
        if self.retry_count > 0:
            base += f" (after {self.retry_count} retries)"
        if self.original_error:
            base += f": {type(self.original_error).__name__}: {self.original_error}"
        return base
