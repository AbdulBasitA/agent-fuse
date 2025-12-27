"""
Agent Fuse - A Circuit Breaker for AI Agents

Fail Safe by Default.

Example:
    >>> from agent_fuse import init, guard, check_loop
    >>> init(budget=5.00)  # Set $5 budget
    >>>
    >>> @guard
    ... def call_llm(prompt):
    ...     return openai.chat.completions.create(...)
    >>>
    >>> # Loop detection for tool calls
    >>> check_loop("search_web", {"query": "python"})
"""

from agent_fuse.config import configure, get_settings
from agent_fuse.core.exceptions import (
    SentinelBudgetExceeded,
    SentinelError,
    SentinelLoopError,
    SentinelSystemError,
)
from agent_fuse.core.loop_detector import check_loop, get_loop_detector, reset_loop_detector

__version__ = "0.2.1"
__all__ = [
    # Configuration
    "init",
    "configure",
    "get_settings",
    # Core functions
    "guard",
    "loop_guard",
    "monitor",
    "pre_flight",
    "post_flight",
    # Loop detection
    "check_loop",
    "get_loop_detector",
    "reset_loop_detector",
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
    loop_threshold: int | None = None,
    loop_detection_enabled: bool | None = None,
) -> None:
    """
    Initialize Agent Fuse.

    This is the primary entry point for configuring Agent Fuse.
    Call this once at the start of your application.

    Args:
        budget: Maximum spend in USD (default: $1.00)
        fail_safe: If True, block on DB errors (default: True)
        session_id: Optional session identifier for grouping usage
        db_path: Path to SQLite database (default: ~/.agent_fuse/guard_v1.db)
        loop_threshold: Number of identical tool calls before error (default: 5)
        loop_detection_enabled: Enable loop detection (default: True)

    Example:
        >>> import agent_fuse
        >>> agent_fuse.init(budget=10.00, loop_threshold=5)
    """
    from agent_fuse.storage.sqlite import get_db

    # Configure settings
    configure(
        budget=budget,
        fail_safe=fail_safe,
        session_id=session_id,
        db_path=db_path,
        loop_threshold=loop_threshold,
        loop_detection_enabled=loop_detection_enabled,
    )

    # Initialize database
    db = get_db()
    db.initialize()

    # Reset loop detector to clear any stale state
    reset_loop_detector()


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


def loop_guard(tool_name: str | None = None):
    """
    Decorator to guard a function with loop detection.

    Checks for repeated identical calls before each execution.
    Raises SentinelLoopError if the same function is called with
    the same arguments more than the configured threshold.

    Args:
        tool_name: Override the tool name (defaults to function name)

    Returns:
        Decorated function that checks for loops before execution

    Example:
        >>> @loop_guard()
        ... def search_web(query: str) -> list[str]:
        ...     return api.search(query)
        >>>
        >>> search_web("python")  # OK
        >>> search_web("python")  # OK (count: 2)
        >>> # ... after threshold (default 5) ...
        >>> search_web("python")  # Raises SentinelLoopError!

    Note:
        The decorator captures function arguments to create a signature.
        Different arguments = different counters.
    """
    import functools
    import inspect
    from typing import Any, Callable, TypeVar

    F = TypeVar("F", bound=Callable[..., Any])

    def decorator(func: F) -> F:
        effective_name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build args dict from function signature
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = dict(bound.arguments)

            # Check for loop
            check_loop(effective_name, args_dict)

            # Call the function
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
