# AgentFuse

**The Circuit Breaker for AI Agents.**

[![PyPI version](https://badge.fury.io/py/agent-fuse.svg)](https://badge.fury.io/py/agent-fuse)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/agent-fuse)](https://pepy.tech/project/agent-fuse)

Stop infinite loops and wallet-draining bugs *before* they hit the API.

```bash
pip install agent-fuse
```

---

## The Problem

You write an Agent. You give it a loop. You go to sleep.

You wake up to a **$500 OpenAI bill** because your agent got stuck in a `while True` loop trying to fix a typo.

This happens more than you think:
- AutoGen agents retrying failed tool calls
- LangChain chains with bad exit conditions
- RAG pipelines re-embedding the same documents
- Any agent with access to `gpt-4` and poor error handling

## The Solution

**Agent Fuse** is a local circuit breaker for your LLM calls. It sits between your code and OpenAI, enforcing:

1. **Hard Budget Limits** - "Max $2.00 per session"
2. **Pre-flight Checks** - Block calls before they're made
3. **Fail-Safe Architecture** - Won't break your app if the guard fails

Zero latency. Zero external dependencies. Just SQLite.

---

## Quick Start (30 seconds)

### 1. Install

```bash
pip install agent-fuse
```

Or with optional integrations:
```bash
pip install agent-fuse[openai]      # OpenAI support
pip install agent-fuse[langchain]   # LangChain support
pip install agent-fuse[all]         # Everything
```

### 2. Protect Your Agent

Just change your import. Agent Fuse is a drop-in replacement.

```python
# Before
from openai import OpenAI
client = OpenAI()

# After
from agent_fuse import init
from agent_fuse.integrations import AgentFuseOpenAI as OpenAI

init(budget=5.00)  # Max $5.00 for this session
client = OpenAI()  # Works exactly the same!
```

That's it. If your agent tries to exceed $5.00, Agent Fuse raises `SentinelBudgetExceeded` and your wallet stays intact.

---

## Features

### Drop-in OpenAI Replacement

```python
from agent_fuse import init
from agent_fuse.integrations import AgentFuseOpenAI

init(budget=10.00)
client = AgentFuseOpenAI()

# Use exactly like openai.OpenAI
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True  # Streaming supported!
)
```

### LangChain Integration

```python
from agent_fuse import init
from agent_fuse.integrations import AgentFuseCallbackHandler
from langchain_openai import ChatOpenAI

init(budget=5.00)
handler = AgentFuseCallbackHandler()

llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
response = llm.invoke("What is 2+2?")  # Protected!
```

### Manual Pre/Post Flight

For custom integrations or non-OpenAI models:

```python
from agent_fuse import init, pre_flight, post_flight, monitor

init(budget=5.00)

# Before your LLM call
pre_flight("gpt-4o", estimated_input_tokens=1000)

# ... make your API call ...

# After your LLM call (with actual usage)
post_flight("gpt-4o", input_tokens=950, output_tokens=500)

# Check your spend
stats = monitor()
print(f"Spent: ${stats.total_spend_usd:.4f}")
print(f"Remaining: ${stats.budget_remaining_usd:.2f}")
```

### Fail-Safe Mode

By default, Agent Fuse prioritizes **safety** - if the database fails, your agent stops.

For high-availability scenarios, you can flip this:

```python
from agent_fuse import init

# Safety mode (default): Block agent if Agent Fuse fails
init(budget=10.00, fail_safe=True)

# Availability mode: Log warning but let agent continue
init(budget=10.00, fail_safe=False)
```

---

## Configuration

All settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTFUSE_BUDGET` | `1.00` | Maximum spend in USD |
| `AGENTFUSE_FAIL_SAFE` | `True` | Block on errors (safety) or continue (availability) |
| `AGENTFUSE_DB_PATH` | `~/.agent_fuse/guard_v1.db` | SQLite database location |
| `AGENTFUSE_MAX_RETRIES` | `3` | Retry attempts for DB operations |

Or configure programmatically:

```python
from agent_fuse import init

init(
    budget=10.00,
    fail_safe=True,
    session_id="my-agent-run-123",
    db_path="/tmp/agent_fuse.db"
)
```

---

## Supported Models

Agent Fuse includes pricing for 25+ models:

**OpenAI:** gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini

**Anthropic:** claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet, claude-3.5-haiku

Unknown models fall back to conservative pricing estimates.

---

## Architecture

```
Your Agent
    |
    v
+------------------+
| AgentFuseOpenAI  |  <- Drop-in replacement
+--------+---------+
         |
    +----+----+
    v         v
Pre-flight  Post-flight
 (check)     (record)
    |         |
    +----+----+
         v
+------------------+
| SQLite (WAL)     |  <- Thread-safe, concurrent writes
+------------------+
         |
         v
   ~/.agent_fuse/
   guard_v1.db
```

**Why SQLite?**
- Zero configuration
- WAL mode handles concurrent agents
- No network dependency
- Works offline

---

## API Reference

### `init(budget, fail_safe, session_id, db_path)`
Initialize Agent Fuse. Call once at startup.

### `pre_flight(model, estimated_input_tokens, ...)`
Check if a call would exceed budget. Raises `SentinelBudgetExceeded` if over.

### `post_flight(model, input_tokens, output_tokens)`
Record actual usage after a call completes.

### `monitor()`
Returns `UsageStats` with current spend, remaining budget, and call counts.

### `guard(model)`
Decorator for wrapping functions with pre-flight checks.

---

## Exceptions

```python
from agent_fuse.core.exceptions import (
    SentinelBudgetExceeded,  # Budget limit hit
    SentinelSystemError,     # DB failed (in fail_safe mode)
    SentinelLoopError,       # Loop detected (coming soon)
)
```

---

## Development

```bash
# Clone
git clone https://github.com/agentfuse/agent-fuse
cd agent-fuse

# Install dev dependencies
python -m venv env
source env/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run verification
python verify_phase1.py
```

---

## Roadmap

- [x] Budget limits
- [x] OpenAI shim (sync + streaming)
- [x] LangChain callback handler
- [x] Fail-safe architecture
- [ ] Loop detection (repeated similar calls)
- [ ] Rate limiting (calls per minute)
- [ ] Multi-agent session tracking
- [ ] Dashboard / CLI stats viewer

---

## Limitations

**Transparency builds trust.** Here's what AgentFuse does NOT do (yet):

- **No loop detection** - AgentFuse tracks spend, not call patterns. If your agent makes 1000 cheap calls, it won't stop until budget is hit.
- **No rate limiting** - It's a budget guard, not a rate limiter. Use OpenAI's built-in rate limits for that.
- **Single-machine only** - The SQLite DB is local. For distributed agents, you'd need a shared backend (PRs welcome!).
- **Estimates can drift** - Pre-flight estimates use heuristics. Actual costs are recorded post-flight, but a large call could slip through if estimates are low.

---

## License

MIT

---

## Contributing

PRs welcome. Please include tests.

---

**Built with paranoia by developers who've seen $500 bills.**
