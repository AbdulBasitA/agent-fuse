"""
OpenAI Shim - Drop-in replacement for openai.OpenAI with budget protection.

AgentFuseOpenAI wraps the OpenAI client and intercepts chat.completions.create()
to perform pre-flight budget checks and post-flight usage recording.

Example:
    >>> from agent_fuse.integrations.openai_shim import AgentFuseOpenAI
    >>> client = AgentFuseOpenAI()  # Uses OPENAI_API_KEY from env
    >>> response = client.chat.completions.create(
    ...     model="gpt-4o",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, overload

from agent_fuse.core.circuit_breaker import get_circuit_breaker
from agent_fuse.utils.token_heuristics import estimate_messages_tokens

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

logger = logging.getLogger("agent_fuse.integrations.openai")


class AgentFuseCompletions:
    """Wrapper for chat.completions that adds budget protection."""

    def __init__(self, client: OpenAI, breaker: Any) -> None:
        self._client = client
        self._breaker = breaker

    @overload
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = True,
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]: ...

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """
        Create a chat completion with budget protection.

        Performs pre-flight check before the API call and post-flight
        recording after. Supports both streaming and non-streaming modes.

        Args:
            model: Model identifier (e.g., 'gpt-4o')
            messages: List of message dicts
            stream: Whether to stream the response
            **kwargs: Additional arguments passed to OpenAI

        Returns:
            ChatCompletion or stream of ChatCompletionChunk

        Raises:
            SentinelBudgetExceeded: If call would exceed budget
        """
        # Estimate tokens for pre-flight
        estimated_input = estimate_messages_tokens(messages, model)

        # Pre-flight check (raises SentinelBudgetExceeded if over budget)
        self._breaker.pre_flight(
            model=model,
            estimated_input_tokens=estimated_input,
        )

        # Make the actual API call
        if stream:
            return self._create_streaming(model, messages, **kwargs)
        else:
            return self._create_non_streaming(model, messages, **kwargs)

    def _create_non_streaming(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Handle non-streaming completion."""
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **kwargs,
        )

        # Extract token usage from response
        usage = response.usage
        if usage:
            self._breaker.post_flight(
                model=model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            )
        else:
            logger.warning("No usage data in response, estimating tokens")
            # Estimate from response content
            output_text = ""
            if response.choices:
                content = response.choices[0].message.content
                if content:
                    output_text = content
            from agent_fuse.utils.token_heuristics import estimate_tokens
            estimated_output = estimate_tokens(output_text, model)
            self._breaker.post_flight(
                model=model,
                input_tokens=estimate_messages_tokens(messages, model),
                output_tokens=estimated_output,
            )

        return response

    def _create_streaming(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion with post-flight on completion."""
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},  # Request usage in stream
            **kwargs,
        )

        return _StreamingWrapper(
            stream=stream,
            model=model,
            messages=messages,
            breaker=self._breaker,
        )


class _StreamingWrapper:
    """
    Wrapper for streaming responses that records usage on completion.

    Accumulates the streamed content and records usage when the stream
    is exhausted.
    """

    def __init__(
        self,
        stream: Iterator[Any],
        model: str,
        messages: list[dict[str, Any]],
        breaker: Any,
    ) -> None:
        self._stream = stream
        self._model = model
        self._messages = messages
        self._breaker = breaker
        self._accumulated_content = ""
        self._usage_recorded = False
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)

            # Accumulate content
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    self._accumulated_content += delta.content

            # Check for usage data (sent in final chunk with stream_options)
            if hasattr(chunk, "usage") and chunk.usage:
                self._prompt_tokens = chunk.usage.prompt_tokens
                self._completion_tokens = chunk.usage.completion_tokens

            return chunk

        except StopIteration:
            # Stream exhausted - record usage
            self._record_usage()
            raise

    def _record_usage(self) -> None:
        """Record usage after stream completes."""
        if self._usage_recorded:
            return
        self._usage_recorded = True

        if self._prompt_tokens is not None and self._completion_tokens is not None:
            # Use actual usage from API
            self._breaker.post_flight(
                model=self._model,
                input_tokens=self._prompt_tokens,
                output_tokens=self._completion_tokens,
            )
        else:
            # Fallback to estimation
            logger.debug("No usage in stream, estimating from content")
            from agent_fuse.utils.token_heuristics import estimate_tokens
            input_tokens = estimate_messages_tokens(self._messages, self._model)
            output_tokens = estimate_tokens(self._accumulated_content, self._model)
            self._breaker.post_flight(
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )


class AgentFuseChat:
    """Wrapper for client.chat namespace."""

    def __init__(self, client: OpenAI, breaker: Any) -> None:
        self._client = client
        self._breaker = breaker
        self._completions: AgentFuseCompletions | None = None

    @property
    def completions(self) -> AgentFuseCompletions:
        """Get the completions wrapper."""
        if self._completions is None:
            self._completions = AgentFuseCompletions(self._client, self._breaker)
        return self._completions


class AgentFuseOpenAI:
    """
    Drop-in replacement for openai.OpenAI with budget protection.

    Wraps the OpenAI client to add pre-flight budget checks and
    post-flight usage recording for chat completions.

    Example:
        >>> from agent_fuse.integrations.openai_shim import AgentFuseOpenAI
        >>> import agent_fuse
        >>>
        >>> agent_fuse.init(budget=5.00)
        >>> client = AgentFuseOpenAI()
        >>>
        >>> # Use exactly like openai.OpenAI
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )

    Note:
        Currently only wraps chat.completions.create(). Other endpoints
        (embeddings, images, etc.) are passed through without protection.
    """

    def __init__(
        self,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the AgentFuse-wrapped OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            **kwargs: Additional arguments passed to openai.OpenAI
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for AgentFuseOpenAI. "
                "Install it with: pip install agent-fuse[openai]"
            ) from e

        self._client = OpenAI(api_key=api_key, **kwargs)
        self._breaker = get_circuit_breaker()
        self._chat: AgentFuseChat | None = None

    @property
    def chat(self) -> AgentFuseChat:
        """Get the chat namespace wrapper."""
        if self._chat is None:
            self._chat = AgentFuseChat(self._client, self._breaker)
        return self._chat

    def __getattr__(self, name: str) -> Any:
        """
        Proxy attribute access to the underlying OpenAI client.

        This allows SentinelOpenAI to be used as a drop-in replacement,
        with non-wrapped endpoints passing through directly.
        """
        return getattr(self._client, name)
