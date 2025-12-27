"""
Phase 2 Verification - Test Core Logic

Tests:
1. Token heuristics
2. Cost calculation
3. Pre-flight budget checks
4. Post-flight recording
5. Budget exceeded exception
"""

import os
import shutil
from pathlib import Path

from sentinel_guard.config import configure, reset_settings
from sentinel_guard.storage.sqlite import reset_db
from sentinel_guard.core.exceptions import SentinelBudgetExceeded


def setup():
    """Reset state for clean test."""
    reset_settings()
    reset_db()
    # Reset state manager singleton
    from sentinel_guard.core.state_manager import StateManager
    StateManager.reset_instance()
    # Reset circuit breaker singleton
    from sentinel_guard.core.circuit_breaker import reset_circuit_breaker
    reset_circuit_breaker()


def test_token_heuristics():
    print("[Test 1] Token Heuristics...")
    from sentinel_guard.utils.token_heuristics import estimate_tokens, estimate_messages_tokens

    # Test basic text estimation
    text = "Hello, world! This is a test message."
    tokens = estimate_tokens(text)
    assert tokens > 0, "Token count should be positive"
    print(f"   Text: '{text[:30]}...' -> {tokens} tokens")

    # Test message estimation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    msg_tokens = estimate_messages_tokens(messages)
    assert msg_tokens > tokens, "Message tokens should include overhead"
    print(f"   Messages (2 msgs) -> {msg_tokens} tokens")

    print("[OK] Token heuristics working.")


def test_cost_calculation():
    print("\n[Test 2] Cost Calculation...")
    from sentinel_guard.core.cost_tracker import calculate_cost, get_model_pricing, format_cost

    # Test GPT-4o pricing
    pricing = get_model_pricing("gpt-4o")
    print(f"   GPT-4o: ${pricing.input_cost_per_million}/1M input, ${pricing.output_cost_per_million}/1M output")

    # Calculate cost for 1000 input, 500 output tokens
    cost = calculate_cost("gpt-4o", 1000, 500)
    expected_input = 1000 * (2.50 / 1_000_000)  # $2.50 per 1M
    expected_output = 500 * (10.00 / 1_000_000)  # $10.00 per 1M
    expected_total = expected_input + expected_output

    assert abs(cost - expected_total) < 0.0001, f"Cost mismatch: {cost} vs {expected_total}"
    print(f"   1000 in + 500 out = {format_cost(cost)}")

    print("[OK] Cost calculation accurate.")


def test_pre_flight_pass():
    print("\n[Test 3] Pre-flight Check (Should Pass)...")
    setup()

    import sentinel_guard
    # Use init with db_path for clean database
    sentinel_guard.init(budget=1.00, db_path="./test_env/phase2_pass.db")

    # Small request should pass
    estimated = sentinel_guard.pre_flight(
        model="gpt-4o",
        estimated_input_tokens=100,
        estimated_output_tokens=50,
    )

    print(f"   Estimated cost: ${estimated:.6f}")
    assert estimated < 1.00, "Estimated cost should be under budget"

    print("[OK] Pre-flight passed for small request.")


def test_pre_flight_fail():
    print("\n[Test 4] Pre-flight Check (Should Fail - Budget Exceeded)...")
    setup()

    import sentinel_guard
    # Use init with db_path for clean database
    sentinel_guard.init(budget=0.001, db_path="./test_env/phase2_fail.db")

    # Large request should fail
    try:
        sentinel_guard.pre_flight(
            model="gpt-4o",
            estimated_input_tokens=10000,
            estimated_output_tokens=5000,
        )
        print("[FAIL] Should have raised SentinelBudgetExceeded")
        exit(1)
    except SentinelBudgetExceeded as e:
        print(f"   Caught: {e}")
        print(f"   Overage: ${e.overage:.4f}")

    print("[OK] Pre-flight correctly blocked over-budget request.")


def test_post_flight_recording():
    print("\n[Test 5] Post-flight Recording...")
    setup()

    import sentinel_guard
    # Use init with db_path for clean database
    sentinel_guard.init(budget=10.00, db_path="./test_env/phase2_recording.db")

    # Record some usage
    cost1 = sentinel_guard.post_flight("gpt-4o", input_tokens=1000, output_tokens=500)
    cost2 = sentinel_guard.post_flight("gpt-4o", input_tokens=2000, output_tokens=1000)

    print(f"   Call 1: ${cost1:.6f}")
    print(f"   Call 2: ${cost2:.6f}")

    # Check stats
    stats = sentinel_guard.monitor()
    print(f"   Total spend: ${stats.total_spend_usd:.6f}")
    print(f"   Total calls: {stats.total_calls}")
    print(f"   Budget remaining: ${stats.budget_remaining_usd:.4f}")

    assert stats.total_calls == 2, f"Expected 2 calls, got {stats.total_calls}"
    assert abs(stats.total_spend_usd - (cost1 + cost2)) < 0.0001, "Spend mismatch"

    print("[OK] Post-flight recording and monitoring working.")


def cleanup():
    if os.path.exists("./test_env"):
        shutil.rmtree("./test_env")


if __name__ == "__main__":
    cleanup()
    os.makedirs("./test_env", exist_ok=True)

    try:
        test_token_heuristics()
        test_cost_calculation()
        test_pre_flight_pass()
        test_pre_flight_fail()
        test_post_flight_recording()
        print("\n[PHASE 2 COMPLETE] Core logic is solid.")
    finally:
        cleanup()
