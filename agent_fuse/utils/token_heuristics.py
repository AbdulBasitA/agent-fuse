"""
Token Heuristics - Estimate token counts for pre-flight checks.

Provides fast token estimation without requiring tiktoken.
Falls back to character-based heuristics when tiktoken is unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("sentinel_guard.utils")

# Try to import tiktoken for accurate counts
_tiktoken_available = False
_encoders: dict[str, object] = {}

try:
    import tiktoken
    _tiktoken_available = True
except ImportError:
    tiktoken = None  # type: ignore


# Model to tiktoken encoding mapping
MODEL_ENCODINGS: dict[str, str] = {
    # OpenAI GPT-4 family
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    # OpenAI GPT-3.5 family
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    # OpenAI o1 family
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    # Default for unknown models
    "default": "cl100k_base",
}

# Average characters per token by model family (for heuristic fallback)
CHARS_PER_TOKEN: dict[str, float] = {
    "gpt-4": 4.0,
    "gpt-3.5": 4.0,
    "claude": 3.5,  # Claude tends to be slightly more efficient
    "default": 4.0,
}


def _get_encoder(model: str) -> object | None:
    """Get or create a tiktoken encoder for the model."""
    if not _tiktoken_available or tiktoken is None:
        return None

    encoding_name = MODEL_ENCODINGS.get(model, MODEL_ENCODINGS["default"])

    if encoding_name not in _encoders:
        try:
            _encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning("Failed to load tiktoken encoding %s: %s", encoding_name, e)
            return None

    return _encoders.get(encoding_name)


def _get_chars_per_token(model: str) -> float:
    """Get the characters-per-token ratio for heuristic estimation."""
    model_lower = model.lower()

    for prefix, ratio in CHARS_PER_TOKEN.items():
        if prefix in model_lower:
            return ratio

    return CHARS_PER_TOKEN["default"]


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate the number of tokens in a text string.

    Uses tiktoken if available for accurate counts, otherwise
    falls back to character-based heuristics (len(text) / 4).

    Args:
        text: The text to estimate tokens for
        model: The model name (used to select appropriate tokenizer)

    Returns:
        Estimated token count (always at least 1 for non-empty text)

    Example:
        >>> estimate_tokens("Hello, world!")
        4
        >>> estimate_tokens("", model="gpt-4")
        0
    """
    if not text:
        return 0

    # Try tiktoken first
    encoder = _get_encoder(model)
    if encoder is not None:
        try:
            tokens = encoder.encode(text)  # type: ignore
            return len(tokens)
        except Exception as e:
            logger.debug("tiktoken encoding failed, using heuristic: %s", e)

    # Fallback to heuristic
    chars_per_token = _get_chars_per_token(model)
    estimated = int(len(text) / chars_per_token)

    # Always return at least 1 for non-empty text
    return max(1, estimated)


def estimate_messages_tokens(
    messages: Sequence[dict[str, str]],
    model: str = "gpt-4",
) -> int:
    """
    Estimate tokens for a list of chat messages.

    Accounts for message structure overhead (role, content separators).

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: The model name

    Returns:
        Estimated total token count

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"},
        ... ]
        >>> estimate_messages_tokens(messages)
        12
    """
    if not messages:
        return 0

    total = 0

    # Overhead per message (role tokens, separators)
    # OpenAI uses ~4 tokens per message for structure
    tokens_per_message = 4

    for message in messages:
        total += tokens_per_message
        content = message.get("content", "")
        if content:
            total += estimate_tokens(content, model)

        # Role name tokens
        role = message.get("role", "")
        if role:
            total += estimate_tokens(role, model)

        # Name field if present
        name = message.get("name", "")
        if name:
            total += estimate_tokens(name, model) + 1  # +1 for name separator

    # Add priming tokens (assistant response priming)
    total += 3

    return total


def is_tiktoken_available() -> bool:
    """Check if tiktoken is available for accurate token counting."""
    return _tiktoken_available
