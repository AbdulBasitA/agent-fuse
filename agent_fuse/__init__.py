"""
Agent Fuse - A Circuit Breaker for AI Agents

Fail Safe by Default.

Example:
    >>> from agent_fuse import init, guard
    >>> init(budget=5.00)  # Set $5 budget
    >>>
    >>> @guard
    ... def call_llm(prompt):
    ...     return openai.chat.completions.create(...)
"""

from agent_fuse.config import configure, get_settings
from agent_fuse.core.exceptions import (
    SentinelBudgetExceeded,
    SentinelError,
    SentinelLoopError,
    SentinelSystemError,
)

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "init",
    "configure",
    "get_settings",
    # Core functions
    "guard",
    "monitor",
    "pre_flight",
    "post_flight",
    # Exceptions
    "SentinelError",
    "SentinelBudgetExceeded",
    "SentinelLoopError",
    "SentinelSystemError",
]


def init(
    budget: float | None = None,
    fail_safe: bool | None = None,
    session_id: str | None = None,
    db_path: str | None = None,
) -> None:
    """
    Initialize Sentinel Guard.

    This is the primary entry point for configuring Sentinel Guard.
    Call this once at the start of your application.

    Args:
        budget: Maximum spend in USD (default: $1.00)
        fail_safe: If True, block on DB errors (default: True)
        session_id: Optional session identifier for grouping usage
        db_path: Path to SQLite database (default: ~/.sentinel/guard_v1.db)

    Example:
        >>> import agent_fuse
        >>> agent_fuse.init(budget=10.00)
    """
    from agent_fuse.storage.sqlite import get_db

    # Configure settings
    configure(budget=budget, fail_safe=fail_safe, session_id=session_id, db_path=db_path)

    # Initialize database
    db = get_db()
    db.initialize()


def guard(
    model: str = "gpt-4o",
):
    """
    Decorator to guard a function with pre-flight budget checks.

    Args:
        model: Default model to use for cost estimation

    Returns:
        Decorated function that checks budget before execution

    Example:
        >>> @guard(model="gpt-4o")
        ... def call_llm(prompt: str) -> str:
        ...     return client.chat.completions.create(...)
    """
    from agent_fuse.core.circuit_breaker import guard as _guard

    return _guard(model=model)


def monitor():
    """
    Get current usage statistics.

    Returns:
        UsageStats with spend, tokens, and budget info

    Example:
        >>> stats = agent_fuse.monitor()
        >>> print(f"Spent: ${stats.total_spend_usd:.4f}")
        >>> print(f"Remaining: ${stats.budget_remaining_usd:.2f}")
    """
    from agent_fuse.core.state_manager import get_state_manager

    return get_state_manager().get_stats()


def pre_flight(
    model: str,
    estimated_input_tokens: int | None = None,
    estimated_output_tokens: int | None = None,
    messages: list[dict[str, str]] | None = None,
    prompt: str | None = None,
) -> float:
    """
    Perform a pre-flight budget check before an LLM call.

    Args:
        model: Model identifier (e.g., 'gpt-4o')
        estimated_input_tokens: Pre-calculated input token count
        estimated_output_tokens: Expected output tokens
        messages: Chat messages for token estimation
        prompt: Text prompt for token estimation

    Returns:
        Estimated cost for this call

    Raises:
        SentinelBudgetExceeded: If call would exceed budget
    """
    from agent_fuse.core.circuit_breaker import pre_flight as _pre_flight

    return _pre_flight(
        model=model,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        messages=messages,
        prompt=prompt,
    )


def post_flight(
    model: str,
    input_tokens: int,
    output_tokens: int,
    metadata: str | None = None,
) -> float:
    """
    Record actual usage after an LLM call.

    Args:
        model: Model identifier
        input_tokens: Actual input token count
        output_tokens: Actual output token count
        metadata: Optional JSON metadata

    Returns:
        Actual cost of the call
    """
    from agent_fuse.core.circuit_breaker import post_flight as _post_flight

    return _post_flight(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata=metadata,
    )
