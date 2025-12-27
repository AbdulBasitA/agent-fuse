"""
Resilience Tests - Verify FAIL_SAFE behavior.

Tests that Sentinel Guard correctly handles database failures
based on the FAIL_SAFE configuration.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_fuse.config import configure, reset_settings
from agent_fuse.core.exceptions import SentinelBudgetExceeded, SentinelSystemError
from agent_fuse.storage.sqlite import AgentFuseDB, reset_db


@pytest.fixture(autouse=True)
def clean_state():
    """Reset state before each test."""
    reset_settings()
    reset_db()
    # Reset singletons
    from agent_fuse.core.state_manager import StateManager
    from agent_fuse.core.circuit_breaker import reset_circuit_breaker
    StateManager.reset_instance()
    reset_circuit_breaker()
    yield
    reset_settings()
    reset_db()


@pytest.fixture
def test_dir(tmp_path):
    """Provide a temporary test directory."""
    return tmp_path


class TestFailSafeTrue:
    """Tests for FAIL_SAFE=True (default, prioritize safety)."""

    def test_db_connection_failure_raises_system_error(self, test_dir):
        """When DB connection fails and FAIL_SAFE=True, should raise SentinelSystemError."""
        # Create a directory where the DB file should be (causes OperationalError)
        bad_path = test_dir / "bad_db"
        bad_path.mkdir()
        db_file = bad_path / "file.db"
        db_file.mkdir()  # Make it a directory to cause error

        settings = configure(db_path=db_file, fail_safe=True)
        storage = AgentFuseDB(settings)

        with pytest.raises(SentinelSystemError) as exc_info:
            storage.log_usage("gpt-4o", 100, 50, 0.01)

        assert "connect" in str(exc_info.value).lower() or "initialize" in str(exc_info.value).lower()

    def test_db_write_failure_raises_system_error(self, test_dir):
        """When DB write fails and FAIL_SAFE=True, should raise SentinelSystemError."""
        db_path = test_dir / "test.db"
        settings = configure(db_path=db_path, fail_safe=True)
        storage = AgentFuseDB(settings)
        storage.initialize()

        # Mock execute_with_retry to simulate failure
        with patch.object(storage, "_get_connection") as mock_conn:
            mock_conn.return_value.execute.side_effect = sqlite3.OperationalError("database is locked")

            with pytest.raises(SentinelSystemError):
                storage.execute_with_retry("test_op", "SELECT 1")

    def test_budget_check_blocked_on_db_failure(self, test_dir):
        """Budget checks should fail safely when DB is unavailable."""
        bad_path = test_dir / "bad_db"
        bad_path.mkdir()
        db_file = bad_path / "file.db"
        db_file.mkdir()

        settings = configure(db_path=db_file, fail_safe=True, budget=10.00)

        from agent_fuse.core.state_manager import StateManager
        StateManager.reset_instance()
        manager = StateManager(settings)

        with pytest.raises(SentinelSystemError):
            manager.get_total_spend()


class TestFailSafeFalse:
    """Tests for FAIL_SAFE=False (prioritize availability)."""

    def test_db_connection_failure_continues(self, test_dir, capsys):
        """When DB connection fails and FAIL_SAFE=False, should log warning and continue."""
        bad_path = test_dir / "bad_db"
        bad_path.mkdir()
        db_file = bad_path / "file.db"
        db_file.mkdir()

        settings = configure(db_path=db_file, fail_safe=False)
        storage = AgentFuseDB(settings)

        # Should not raise, returns None
        result = storage.log_usage("gpt-4o", 100, 50, 0.01)
        assert result is None

        # Should have printed warning to stderr
        captured = capsys.readouterr()
        assert "AGENT FUSE FAILED" in captured.err

    def test_db_failure_allows_spend_query(self, test_dir, capsys):
        """Spend queries should return 0 when DB fails and FAIL_SAFE=False."""
        bad_path = test_dir / "bad_db"
        bad_path.mkdir()
        db_file = bad_path / "file.db"
        db_file.mkdir()

        settings = configure(db_path=db_file, fail_safe=False)
        storage = AgentFuseDB(settings)

        # Should return 0.0 instead of raising
        result = storage.get_total_spend()
        assert result == 0.0

    def test_pre_flight_passes_on_db_failure(self, test_dir, capsys):
        """Pre-flight should pass when DB fails and FAIL_SAFE=False."""
        bad_path = test_dir / "bad_db"
        bad_path.mkdir()
        db_file = bad_path / "file.db"
        db_file.mkdir()

        settings = configure(db_path=db_file, fail_safe=False, budget=1.00)

        from agent_fuse.core.state_manager import StateManager
        from agent_fuse.core.circuit_breaker import CircuitBreaker

        StateManager.reset_instance()
        manager = StateManager(settings)
        breaker = CircuitBreaker(manager)

        # Should not raise - DB failure means spend is 0, budget check passes
        cost = breaker.pre_flight("gpt-4o", estimated_input_tokens=100)
        assert cost > 0  # Cost was estimated


class TestRetryBehavior:
    """Tests for retry logic on transient errors."""

    def test_retry_settings_are_configured(self, test_dir):
        """Verify retry settings are properly configured."""
        db_path = test_dir / "retry_test.db"
        settings = configure(db_path=db_path, fail_safe=True, max_retries=5, retry_delay=0.5)

        assert settings.max_retries == 5
        assert settings.retry_delay == 0.5

    def test_default_retry_settings(self, test_dir):
        """Verify default retry settings."""
        db_path = test_dir / "default_retry_test.db"
        settings = configure(db_path=db_path, fail_safe=True)

        # Defaults from SentinelSettings
        assert settings.max_retries == 3
        assert settings.retry_delay == 0.1


class TestBudgetEnforcement:
    """Tests for budget enforcement with working DB."""

    def test_budget_exceeded_raises(self, test_dir):
        """Should raise SentinelBudgetExceeded when over budget."""
        db_path = test_dir / "test.db"
        configure(db_path=db_path, fail_safe=True, budget=0.01)

        import agent_fuse
        from agent_fuse.core.state_manager import StateManager
        from agent_fuse.core.circuit_breaker import reset_circuit_breaker

        StateManager.reset_instance()
        reset_circuit_breaker()
        agent_fuse.init(budget=0.01)

        # Small request might pass
        # Large request should fail
        with pytest.raises(SentinelBudgetExceeded) as exc_info:
            agent_fuse.pre_flight("gpt-4o", estimated_input_tokens=100000)

        assert exc_info.value.budget_limit == 0.01
        assert exc_info.value.estimated_cost > 0.01

    def test_budget_tracking_accumulates(self, test_dir):
        """Usage should accumulate correctly."""
        # Use unique db path for this test
        db_path = test_dir / "accumulate_test.db"

        import agent_fuse
        from agent_fuse.core.state_manager import StateManager
        from agent_fuse.core.circuit_breaker import reset_circuit_breaker
        from agent_fuse.storage.sqlite import reset_db

        # Full reset
        reset_settings()
        reset_db()
        StateManager.reset_instance()
        reset_circuit_breaker()

        # Use init with db_path to ensure fresh database
        agent_fuse.init(budget=10.00, db_path=str(db_path))

        # Record usage
        agent_fuse.post_flight("gpt-4o", 1000, 500)
        agent_fuse.post_flight("gpt-4o", 1000, 500)

        stats = agent_fuse.monitor()
        assert stats.total_calls == 2, f"Expected 2 calls, got {stats.total_calls}"
        assert stats.total_spend_usd > 0


class TestConcurrency:
    """Tests for concurrent access scenarios."""

    def test_thread_safety(self, test_dir):
        """Multiple threads should be able to write safely."""
        import threading

        # Use unique db path
        db_path = test_dir / "thread_test.db"

        import agent_fuse
        from agent_fuse.core.state_manager import StateManager
        from agent_fuse.core.circuit_breaker import reset_circuit_breaker
        from agent_fuse.storage.sqlite import reset_db

        # Full reset
        reset_settings()
        reset_db()
        StateManager.reset_instance()
        reset_circuit_breaker()

        # Use init with db_path to ensure fresh database
        agent_fuse.init(budget=100.00, db_path=str(db_path))

        errors = []
        success_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal success_count
            try:
                for _ in range(5):
                    agent_fuse.post_flight("gpt-4o", 100, 50)
                with lock:
                    success_count += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors occurred: {errors}"
        assert success_count == 4

        stats = agent_fuse.monitor()
        assert stats.total_calls == 20  # 4 threads * 5 calls
