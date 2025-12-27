"""
Cost Tracker - Calculate costs based on model pricing.

Supports dynamic pricing updates from a remote JSON file with
fallback to bundled defaults. Pricing is loaded once on first use.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_fuse.config import get_settings

logger = logging.getLogger("agent_fuse.core.cost_tracker")

# Path to bundled default pricing
_DEFAULT_PRICING_PATH = Path(__file__).parent / "default_pricing.json"


@dataclass(frozen=True)
class ModelPricing:
    """Pricing information for a model (per 1M tokens)."""

    input_cost_per_million: float
    output_cost_per_million: float

    @property
    def input_cost_per_token(self) -> float:
        """Cost per input token."""
        return self.input_cost_per_million / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        """Cost per output token."""
        return self.output_cost_per_million / 1_000_000


# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING = ModelPricing(10.00, 30.00)


class PricingRegistry:
    """
    Singleton registry for model pricing.

    Loads pricing from remote URL on first access, falling back
    to bundled defaults if fetch fails.
    """

    _instance: PricingRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._pricing: dict[str, ModelPricing] = {}
        self._loaded = False
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> PricingRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _load_pricing(self) -> None:
        """Load pricing from remote URL or fallback to local file."""
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            settings = get_settings()
            pricing_data: dict[str, Any] | None = None

            # Try remote fetch first
            if settings.pricing_url:
                pricing_data = self._fetch_remote_pricing(
                    settings.pricing_url,
                    settings.pricing_fetch_timeout,
                )

            # Fallback to bundled defaults
            if pricing_data is None:
                pricing_data = self._load_bundled_pricing()

            # Parse into ModelPricing objects
            if pricing_data:
                self._parse_pricing_data(pricing_data)

            self._loaded = True

    def _fetch_remote_pricing(
        self,
        url: str,
        timeout: float,
    ) -> dict[str, Any] | None:
        """
        Fetch pricing from remote URL.

        Args:
            url: URL to fetch pricing JSON from
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON dict, or None if fetch failed
        """
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "agent-fuse/0.1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read().decode("utf-8")
                pricing = json.loads(data)
                logger.debug("Loaded remote pricing from %s", url)
                return pricing

        except urllib.error.URLError as e:
            logger.debug("Failed to fetch remote pricing: %s", e)
        except urllib.error.HTTPError as e:
            logger.debug("HTTP error fetching pricing: %s %s", e.code, e.reason)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in remote pricing: %s", e)
        except TimeoutError:
            logger.debug("Timeout fetching remote pricing")
        except Exception as e:
            logger.debug("Unexpected error fetching pricing: %s", e)

        return None

    def _load_bundled_pricing(self) -> dict[str, Any] | None:
        """Load pricing from bundled JSON file."""
        try:
            with open(_DEFAULT_PRICING_PATH, "r") as f:
                pricing = json.load(f)
                logger.debug("Loaded bundled pricing from %s", _DEFAULT_PRICING_PATH)
                return pricing
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load bundled pricing: %s", e)
            return None

    def _parse_pricing_data(self, data: dict[str, Any]) -> None:
        """Parse JSON pricing data into ModelPricing objects."""
        for model, pricing in data.items():
            try:
                if isinstance(pricing, dict):
                    input_cost = float(pricing.get("input", 10.0))
                    output_cost = float(pricing.get("output", 30.0))
                    self._pricing[model] = ModelPricing(input_cost, output_cost)
            except (TypeError, ValueError) as e:
                logger.warning("Invalid pricing for model %s: %s", model, e)

    def get_pricing(self, model: str) -> ModelPricing:
        """
        Get pricing for a model.

        Args:
            model: Model identifier

        Returns:
            ModelPricing for the model, or DEFAULT_PRICING if unknown
        """
        self._load_pricing()

        # Exact match first
        if model in self._pricing:
            return self._pricing[model]

        # Try lowercase
        model_lower = model.lower()
        if model_lower in self._pricing:
            return self._pricing[model_lower]

        # Try prefix matching for versioned models
        for known_model, pricing in self._pricing.items():
            if model_lower.startswith(known_model.lower()):
                return pricing

        return DEFAULT_PRICING


def get_model_pricing(model: str) -> ModelPricing:
    """
    Get pricing for a model.

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-opus')

    Returns:
        ModelPricing for the model, or DEFAULT_PRICING if unknown
    """
    return PricingRegistry.get_instance().get_pricing(model)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate the cost for a model call.

    Args:
        model: Model identifier
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens

    Returns:
        Cost in USD

    Example:
        >>> calculate_cost("gpt-4o", 1000, 500)
        0.0075  # $0.0025 input + $0.005 output
    """
    pricing = get_model_pricing(model)

    input_cost = input_tokens * pricing.input_cost_per_token
    output_cost = output_tokens * pricing.output_cost_per_token

    return input_cost + output_cost


def estimate_cost(
    model: str,
    estimated_input_tokens: int,
    estimated_output_tokens: int | None = None,
) -> float:
    """
    Estimate cost for a planned model call (pre-flight).

    If output tokens aren't specified, estimates based on typical
    response ratios.

    Args:
        model: Model identifier
        estimated_input_tokens: Estimated input token count
        estimated_output_tokens: Estimated output tokens (optional)

    Returns:
        Estimated cost in USD
    """
    if estimated_output_tokens is None:
        # Assume output is roughly 50% of input for estimation
        # This is conservative for most use cases
        estimated_output_tokens = max(100, estimated_input_tokens // 2)

    return calculate_cost(model, estimated_input_tokens, estimated_output_tokens)


def format_cost(cost: float) -> str:
    """
    Format a cost value for display.

    Args:
        cost: Cost in USD

    Returns:
        Formatted string (e.g., "$0.0025", "$1.50")
    """
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.00:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def reload_pricing() -> None:
    """
    Force reload of pricing data.

    Useful if you want to refresh pricing without restarting.
    """
    PricingRegistry.reset()
    # Trigger reload on next access
    PricingRegistry.get_instance()._load_pricing()
