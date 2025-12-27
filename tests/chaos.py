#!/usr/bin/env python3
"""
Chaos Test - Stress test for concurrent access and budget resets.

Simulates 20 agents running wildly in parallel while the budget
is being reset mid-execution. If this passes, the library is solid.
"""

import threading
import random
import time
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_fuse import init, pre_flight, post_flight
from agent_fuse.core.exceptions import SentinelBudgetExceeded

crash_count = 0
crash_lock = threading.Lock()


def chaos_agent(agent_id):
    """Simulates a rogue agent trying to spend money fast."""
    global crash_count

    for i in range(50):
        model = random.choice(["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"])
        tokens = random.randint(100, 5000)

        try:
            # 1. Pre-Flight Check
            pre_flight(model, tokens)

            # Simulate network latency
            time.sleep(random.uniform(0.001, 0.01))

            # 2. Post-Flight Update
            post_flight(model, tokens, tokens)

        except SentinelBudgetExceeded:
            # This is GOOD. The guard worked.
            pass
        except Exception as e:
            # This is BAD. The library crashed.
            with crash_lock:
                crash_count += 1
                print(f"[CRASH] Agent {agent_id}: {type(e).__name__}: {e}")
            return


def admin_monkey():
    """Simulates a user resetting the config while agents are running."""
    for _ in range(5):
        time.sleep(0.5)
        # Randomly change the budget mid-execution
        new_limit = random.uniform(0.1, 10.0)
        init(budget=new_limit, db_path="./test_chaos.db")
        print(f"   [Admin] Reset budget to ${new_limit:.2f}")


def run_chaos():
    print("=" * 50)
    print("CHAOS TEST - 20 concurrent threads")
    print("=" * 50)

    # 1. Init with test database
    init(budget=1.0, db_path="./test_chaos.db")

    # 2. Spawn Agents
    print("\n[1] Spawning 20 chaos agents...")
    threads = []
    for i in range(20):
        t = threading.Thread(target=chaos_agent, args=(i,))
        threads.append(t)
        t.start()

    # 3. Spawn Admin
    print("[2] Spawning admin monkey (will reset budget 5 times)...")
    admin = threading.Thread(target=admin_monkey)
    admin.start()

    # 4. Wait
    print("[3] Waiting for chaos to complete...\n")
    for t in threads:
        t.join()
    admin.join()

    # 5. Report
    print("\n" + "=" * 50)
    if crash_count == 0:
        print("[OK] CHAOS TEST PASSED. No deadlocks or crashes.")
    else:
        print(f"[FAIL] CHAOS TEST FAILED. {crash_count} crashes detected.")
    print("=" * 50)

    # Cleanup
    for f in ["./test_chaos.db", "./test_chaos.db-wal", "./test_chaos.db-shm"]:
        if os.path.exists(f):
            os.remove(f)

    return crash_count == 0


if __name__ == "__main__":
    success = run_chaos()
    sys.exit(0 if success else 1)
