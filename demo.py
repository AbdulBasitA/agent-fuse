#!/usr/bin/env python3
"""
Sentinel Guard Demo - Real OpenAI Integration Test

This script demonstrates the circuit breaker in action by:
1. Setting a tiny budget ($0.0001)
2. Attempting an LLM call that would exceed it
3. Verifying that Sentinel blocks the call

Run: export OPENAI_API_KEY=sk-... && python demo.py
"""

import os
import sys

# Check for API key first
if not os.environ.get("OPENAI_API_KEY"):
    print("[!] OPENAI_API_KEY not set in environment")
    print("    Run: export OPENAI_API_KEY=sk-...")
    sys.exit(1)

from agent_fuse import init, monitor
from agent_fuse.integrations import AgentFuseOpenAI
from agent_fuse.core.exceptions import SentinelBudgetExceeded

print("=" * 50)
print("AGENT FUSE - Integration Demo")
print("=" * 50)

# 1. Initialize with a tiny budget ($0.0001) to force a block immediately
print("\n[1] Initializing Agent Fuse with $0.0001 budget...")
init(budget=0.0001)
print("    Budget set: $0.0001")

# 2. Use the "Shim" client
print("\n[2] Creating AgentFuseOpenAI client...")
client = AgentFuseOpenAI()  # Automatically reads OPENAI_API_KEY from env
print("    Client ready (wrapping openai.OpenAI)")

# 3. Attempt to make a call that would exceed budget
print("\n[3] Attempting LLM call (estimated cost: ~$0.0002)...")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a 50-word poem about circuit breakers."}]
    )
    print("\n[FAIL] The call went through! Budget check failed.")
    print(f"       Response: {response.choices[0].message.content[:100]}...")
    sys.exit(1)

except SentinelBudgetExceeded as e:
    print(f"\n[OK] Agent Fuse blocked the call!")
    print(f"     Reason: {e}")
    print("     Your wallet is safe.")

except Exception as e:
    print(f"\n[ERROR] Unexpected error: {type(e).__name__}: {e}")
    sys.exit(1)

# 4. Check stats
print("\n[4] Checking usage stats...")
stats = monitor()
print(f"    Total spend:     ${stats.total_spend_usd:.6f}")
print(f"    Total calls:     {stats.total_calls}")
print(f"    Budget limit:    ${stats.budget_limit_usd:.6f}")
print(f"    Budget remaining: ${stats.budget_remaining_usd:.6f}")

print("\n" + "=" * 50)
print("DEMO COMPLETE - Agent Fuse is working!")
print("=" * 50)


# Bonus: Test with a real call if budget allows
print("\n[BONUS] Testing with $1.00 budget (real API call)...")
init(budget=1.00, db_path="./demo_test.db")

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Sentinel Guard works!' in exactly 3 words."}]
    )
    print(f"\n[OK] Real API call succeeded!")
    print(f"     Response: {response.choices[0].message.content}")

    stats = monitor()
    print(f"     Cost: ${stats.total_spend_usd:.6f}")

except SentinelBudgetExceeded as e:
    print(f"\n[BLOCKED] {e}")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")

# Cleanup
import os
if os.path.exists("./demo_test.db"):
    os.remove("./demo_test.db")
if os.path.exists("./demo_test.db-wal"):
    os.remove("./demo_test.db-wal")
if os.path.exists("./demo_test.db-shm"):
    os.remove("./demo_test.db-shm")
