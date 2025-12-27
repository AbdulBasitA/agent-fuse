"""Agent Fuse Integrations - Shims for popular LLM libraries."""

__all__ = [
    "AgentFuseOpenAI",
    "AgentFuseCallbackHandler",
    # Legacy aliases for backwards compatibility
    "SentinelOpenAI",
    "SentinelCallbackHandler",
]


def __getattr__(name: str):
    """Lazy imports to avoid requiring optional dependencies."""
    if name in ("AgentFuseOpenAI", "SentinelOpenAI"):
        from agent_fuse.integrations.openai_shim import AgentFuseOpenAI
        return AgentFuseOpenAI
    elif name in ("AgentFuseCallbackHandler", "SentinelCallbackHandler"):
        from agent_fuse.integrations.langchain import AgentFuseCallbackHandler
        return AgentFuseCallbackHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
