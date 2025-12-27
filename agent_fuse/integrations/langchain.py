"""
LangChain Integration - Callback handler for budget protection.

AgentFuseCallbackHandler hooks into LangChain's callback system to
perform pre-flight checks and post-flight recording.

Example:
    >>> from agent_fuse.integrations.langchain import AgentFuseCallbackHandler
    >>> from langchain_openai import ChatOpenAI
    >>>
    >>> handler = AgentFuseCallbackHandler()
    >>> llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
    >>>
    >>> # Budget protection is now active
    >>> response = llm.invoke("Hello!")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from agent_fuse.core.circuit_breaker import get_circuit_breaker
from agent_fuse.utils.token_heuristics import estimate_tokens

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

logger = logging.getLogger("agent_fuse.integrations.langchain")


class AgentFuseCallbackHandler:
    """
    LangChain callback handler for Agent Fuse budget protection.

    Hooks into on_llm_start for pre-flight checks and on_llm_end
    for post-flight usage recording.

    Example:
        >>> import agent_fuse
        >>> from agent_fuse.integrations.langchain import AgentFuseCallbackHandler
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> agent_fuse.init(budget=5.00)
        >>> handler = AgentFuseCallbackHandler()
        >>>
        >>> llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
        >>> response = llm.invoke("What is 2+2?")

    Note:
        This handler works with both ChatModels and LLMs in LangChain.
        Token counting uses the model's built-in counting when available,
        falling back to heuristics otherwise.
    """

    def __init__(self, default_model: str = "gpt-4o") -> None:
        """
        Initialize the callback handler.

        Args:
            default_model: Default model name for cost estimation
                          when model info is not available from LangChain
        """
        self._breaker = get_circuit_breaker()
        self._default_model = default_model
        # Track runs for matching start/end
        self._runs: dict[UUID, dict[str, Any]] = {}

    # Required callback interface methods

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when LLM starts processing.

        Performs pre-flight budget check. Raises SentinelBudgetExceeded
        if the estimated cost would exceed the budget.
        """
        # Extract model name
        model = self._extract_model(serialized, kwargs)

        # Estimate input tokens from prompts
        total_input_tokens = sum(estimate_tokens(p, model) for p in prompts)

        # Store run info for post-flight
        self._runs[run_id] = {
            "model": model,
            "estimated_input_tokens": total_input_tokens,
            "prompts": prompts,
        }

        # Pre-flight check (raises if over budget)
        self._breaker.pre_flight(
            model=model,
            estimated_input_tokens=total_input_tokens,
        )

        logger.debug(
            "LangChain LLM start: model=%s, estimated_tokens=%d",
            model,
            total_input_tokens,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when LLM finishes processing.

        Records actual token usage from the response.
        """
        run_info = self._runs.pop(run_id, None)
        if run_info is None:
            logger.warning("on_llm_end called without matching on_llm_start")
            return

        model = run_info["model"]

        # Try to get actual token usage from response
        input_tokens, output_tokens = self._extract_usage(response, run_info)

        # Post-flight recording
        self._breaker.post_flight(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        logger.debug(
            "LangChain LLM end: model=%s, tokens=%d/%d",
            model,
            input_tokens,
            output_tokens,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM encounters an error."""
        # Clean up run tracking
        self._runs.pop(run_id, None)
        logger.debug("LangChain LLM error: %s", error)

    # Optional callback methods (no-op implementations)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts."""
        pass

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends."""
        pass

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        pass

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts."""
        pass

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends."""
        pass

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        pass

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when text is produced."""
        pass

    def on_retry(
        self,
        retry_state: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called on retry."""
        pass

    # Helper methods

    def _extract_model(
        self,
        serialized: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> str:
        """Extract model name from serialized config or kwargs."""
        # Try kwargs first
        if "invocation_params" in kwargs:
            params = kwargs["invocation_params"]
            if "model" in params:
                return params["model"]
            if "model_name" in params:
                return params["model_name"]

        # Try serialized config
        if "kwargs" in serialized:
            sk = serialized["kwargs"]
            if "model" in sk:
                return sk["model"]
            if "model_name" in sk:
                return sk["model_name"]

        # Try name field
        if "name" in serialized:
            name = serialized["name"]
            # Common patterns: "ChatOpenAI", "OpenAI", etc.
            if "gpt" in name.lower():
                return "gpt-4o"
            if "claude" in name.lower():
                return "claude-3-sonnet"

        return self._default_model

    def _extract_usage(
        self,
        response: LLMResult,
        run_info: dict[str, Any],
    ) -> tuple[int, int]:
        """Extract token usage from LLMResult."""
        model = run_info["model"]

        # Try to get from llm_output
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                return (
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )

            # Alternative key names
            if "usage" in response.llm_output:
                usage = response.llm_output["usage"]
                return (
                    usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                    usage.get("output_tokens", usage.get("completion_tokens", 0)),
                )

        # Fallback to estimation
        input_tokens = run_info.get("estimated_input_tokens", 0)

        # Estimate output tokens from generations
        output_text = ""
        for gen_list in response.generations:
            for gen in gen_list:
                if gen.text:
                    output_text += gen.text

        output_tokens = estimate_tokens(output_text, model) if output_text else 0

        return input_tokens, output_tokens
