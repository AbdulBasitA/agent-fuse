"""
Circuit Breaker - The core pre-flight and post-flight logic.

Implements the "Should I allow this call?" decision and
tracks actual usage after calls complete.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from agent_fuse.core.cost_tracker import estimate_cost, format_cost
from agent_fuse.core.exceptions import SentinelBudgetExceeded
from agent_fuse.core.state_manager import StateManager, get_state_manager
from agent_fuse.utils.token_heuristics import estimate_messages_tokens, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("agent_fuse.circuit_breaker")

F = TypeVar("F", bound=Callable[..., Any])


class CircuitBreaker:
    """
    Circuit breaker for LLM calls.

    Performs pre-flight budget checks before calls and
    post-flight usage recording after calls.

    Example:
        >>> breaker = CircuitBreaker()
        >>> breaker.pre_flight("gpt-4o", estimated_tokens=1000)  # Raises if over budget
        >>> # ... make LLM call ...
        >>> breaker.post_flight("gpt-4o", input_tokens=950, output_tokens=500)
    """

    def __init__(self, state_manager: StateManager | None = None) -> None:
        """
        Initialize the circuit breaker.

        Args:
            state_manager: Optional state manager override
        """
        self._state = state_manager or get_state_manager()

    @property
    def state(self) -> StateManager:
        """Get the state manager."""
        return self._state

    def pre_flight(
        self,
        model: str,
        estimated_input_tokens: int | None = None,
        estimated_output_tokens: int | None = None,
        messages: Sequence[dict[str, str]] | None = None,
        prompt: str | None = None,
    ) -> float:
        """
        Pre-flight check before an LLM call.

        Estimates cost and checks against budget. Raises SentinelBudgetExceeded
        if the call would exceed the budget.

        Args:
            model: Model identifier (e.g., 'gpt-4o')
            estimated_input_tokens: Pre-calculated input token count
            estimated_output_tokens: Expected output tokens (optional)
            messages: Chat messages (used if estimated_input_tokens not provided)
            prompt: Text prompt (used if messages not provided)

        Returns:
            Estimated cost for this call

        Raises:
            SentinelBudgetExceeded: If call would exceed budget

        Example:
            >>> breaker.pre_flight("gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        """
        # Estimate input tokens if not provided
        if estimated_input_tokens is None:
            if messages is not None:
                estimated_input_tokens = estimate_messages_tokens(messages, model)
            elif prompt is not None:
                estimated_input_tokens = estimate_tokens(prompt, model)
            else:
                # No input provided - use a minimal estimate
                estimated_input_tokens = 10
                logger.warning(
                    "pre_flight called without token estimate or content. "
                    "Using minimal estimate of %d tokens.",
                    estimated_input_tokens,
                )

        # Calculate estimated cost
        cost = estimate_cost(model, estimated_input_tokens, estimated_output_tokens)

        # Check budget
        current_spend = self._state.get_current_spend()
        budget = self._state.budget

        if budget > 0 and (current_spend + cost) > budget:
            logger.warning(
                "Budget exceeded: %s spent + %s estimated > %s limit",
                format_cost(current_spend),
                format_cost(cost),
                format_cost(budget),
            )
            raise SentinelBudgetExceeded(
                f"Budget would be exceeded: {format_cost(current_spend)} spent + "
                f"{format_cost(cost)} estimated > {format_cost(budget)} limit",
                current_spend=current_spend,
                estimated_cost=cost,
                budget_limit=budget,
                model=model,
            )

        logger.debug(
            "Pre-flight passed: %s estimated, %s remaining",
            format_cost(cost),
            format_cost(budget - current_spend - cost) if budget > 0 else "unlimited",
        )

        return cost

    def post_flight(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: str | None = None,
    ) -> float:
        """
        Post-flight recording after an LLM call.

        Records the actual token usage and cost to the database.

        Args:
            model: Model identifier
            input_tokens: Actual input token count
            output_tokens: Actual output token count
            metadata: Optional JSON metadata

        Returns:
            Actual cost of the call

        Example:
            >>> cost = breaker.post_flight("gpt-4o", input_tokens=950, output_tokens=500)
            >>> print(f"Call cost: ${cost:.4f}")
        """
        cost = self._state.add_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata,
        )

        logger.debug(
            "Post-flight recorded: %s for %d/%d tokens on %s",
            format_cost(cost),
            input_tokens,
            output_tokens,
            model,
        )

        return cost


# Module-level singleton
_breaker: CircuitBreaker | None = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get the global CircuitBreaker instance."""
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker()
    return _breaker


def reset_circuit_breaker() -> None:
    """Reset the global CircuitBreaker instance."""
    global _breaker
    _breaker = None


def pre_flight(
    model: str,
    estimated_input_tokens: int | None = None,
    estimated_output_tokens: int | None = None,
    messages: Sequence[dict[str, str]] | None = None,
    prompt: str | None = None,
) -> float:
    """
    Module-level pre-flight check.

    Convenience function that uses the global circuit breaker.
    See CircuitBreaker.pre_flight for details.
    """
    return get_circuit_breaker().pre_flight(
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
    Module-level post-flight recording.

    Convenience function that uses the global circuit breaker.
    See CircuitBreaker.post_flight for details.
    """
    return get_circuit_breaker().post_flight(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata=metadata,
    )


def guard(
    model: str = "gpt-4o",
    estimate_tokens_from_args: Callable[..., int] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to guard a function with pre-flight budget checks.

    Note: This decorator only performs pre-flight checks. For full
    pre-flight and post-flight tracking, use the SentinelOpenAI shim
    or LangChain callback handler.

    Args:
        model: Default model to use for cost estimation
        estimate_tokens_from_args: Optional function to estimate tokens from args

    Returns:
        Decorated function

    Example:
        >>> @guard(model="gpt-4o")
        ... def call_llm(prompt: str) -> str:
        ...     return client.chat.completions.create(...)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Estimate tokens if estimator provided
            estimated_tokens = None
            if estimate_tokens_from_args is not None:
                try:
                    estimated_tokens = estimate_tokens_from_args(*args, **kwargs)
                except Exception as e:
                    logger.warning("Token estimation failed: %s", e)

            # Pre-flight check
            pre_flight(
                model=model,
                estimated_input_tokens=estimated_tokens,
            )

            # Call the function
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
