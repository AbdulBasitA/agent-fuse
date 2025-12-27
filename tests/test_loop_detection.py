"""
Loop Detection Tests - Verify loop detection behavior.

Tests that Agent Fuse correctly detects repeated tool calls
and raises SentinelLoopError when threshold is exceeded.
"""

from __future__ import annotations

import threading

import pytest

from agent_fuse.config import configure, reset_settings
from agent_fuse.core.exceptions import SentinelLoopError
from agent_fuse.core.loop_detector import (
    LoopDetector,
    check_loop,
    get_loop_detector,
    reset_loop_detector,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset state before each test."""
    reset_settings()
    reset_loop_detector()
    yield
    reset_settings()
    reset_loop_detector()


class TestLoopDetectionBasic:
    """Core loop detection tests."""

    def test_no_error_under_threshold(self):
        """Should not raise when under threshold."""
        configure(loop_threshold=3)

        check_loop("test_tool", {"arg": "value"})
        check_loop("test_tool", {"arg": "value"})
        check_loop("test_tool", {"arg": "value"})
        # No error - exactly at threshold

    def test_raises_over_threshold(self):
        """Should raise SentinelLoopError when over threshold."""
        configure(loop_threshold=3)

        check_loop("test_tool", {"arg": "value"})
        check_loop("test_tool", {"arg": "value"})
        check_loop("test_tool", {"arg": "value"})

        with pytest.raises(SentinelLoopError) as exc_info:
            check_loop("test_tool", {"arg": "value"})  # 4th call

        assert exc_info.value.call_count == 4
        assert "test_tool" in exc_info.value.pattern

    def test_default_threshold_is_five(self):
        """Default threshold should be 5."""
        configure()  # Use defaults

        # Should succeed 5 times
        for _ in range(5):
            check_loop("tool", {"x": 1})

        # 6th should fail
        with pytest.raises(SentinelLoopError) as exc_info:
            check_loop("tool", {"x": 1})

        assert exc_info.value.call_count == 6

    def test_different_args_separate_counters(self):
        """Different args should have separate counters."""
        configure(loop_threshold=2)

        check_loop("search", {"query": "python"})
        check_loop("search", {"query": "python"})
        check_loop("search", {"query": "rust"})  # Different args
        check_loop("search", {"query": "rust"})

        # Both at threshold, not over - no error

    def test_different_tools_separate_counters(self):
        """Different tools should have separate counters."""
        configure(loop_threshold=2)

        check_loop("tool_a", {"arg": "same"})
        check_loop("tool_a", {"arg": "same"})
        check_loop("tool_b", {"arg": "same"})  # Different tool
        check_loop("tool_b", {"arg": "same"})

        # Both at threshold, not over - no error


class TestLoopDetectionException:
    """Tests for exception attributes."""

    def test_signature_in_exception(self):
        """Exception should include signature for debugging."""
        configure(loop_threshold=1)

        check_loop("my_tool", {"key": "value"})

        with pytest.raises(SentinelLoopError) as exc_info:
            check_loop("my_tool", {"key": "value"})

        assert exc_info.value.signature is not None
        assert "my_tool" in exc_info.value.signature
        assert "key" in exc_info.value.signature

    def test_signature_alias_for_pattern(self):
        """signature property should be alias for pattern."""
        configure(loop_threshold=1)
        check_loop("tool", {})

        with pytest.raises(SentinelLoopError) as exc_info:
            check_loop("tool", {})

        assert exc_info.value.signature == exc_info.value.pattern

    def test_exception_str_includes_signature(self):
        """Exception __str__ should include signature."""
        configure(loop_threshold=1)
        check_loop("edit_file", {"path": "/tmp/test.py"})

        with pytest.raises(SentinelLoopError) as exc_info:
            check_loop("edit_file", {"path": "/tmp/test.py"})

        error_str = str(exc_info.value)
        assert "Signature:" in error_str
        assert "edit_file" in error_str


class TestLoopDetectionDisabled:
    """Tests for disabled loop detection."""

    def test_no_error_when_disabled(self):
        """Should never raise when detection is disabled."""
        configure(loop_threshold=1, loop_detection_enabled=False)

        for _ in range(100):
            check_loop("test_tool", {"arg": "value"})
        # No error

    def test_enabled_by_default(self):
        """Loop detection should be enabled by default."""
        configure(loop_threshold=1)

        check_loop("tool", {})

        with pytest.raises(SentinelLoopError):
            check_loop("tool", {})


class TestSessionIsolation:
    """Tests for per-session tracking."""

    def test_separate_sessions_isolated(self):
        """Different sessions should have isolated counters."""
        configure(loop_threshold=2, session_id="session_a")

        check_loop("tool", {"arg": "val"})
        check_loop("tool", {"arg": "val"})  # At threshold for session_a

        # Different session should start fresh
        check_loop("tool", {"arg": "val"}, session_id="session_b")
        check_loop("tool", {"arg": "val"}, session_id="session_b")
        # No error - separate counters

    def test_session_from_settings(self):
        """Should use session_id from settings when not provided."""
        configure(loop_threshold=2, session_id="my_session")

        check_loop("tool", {"arg": "val"})
        check_loop("tool", {"arg": "val"})

        # Same signature, different session - should work
        check_loop("tool", {"arg": "val"}, session_id="other_session")

    def test_reset_session(self):
        """Resetting session should clear counters."""
        configure(loop_threshold=2)
        detector = get_loop_detector()

        check_loop("tool", {"arg": "val"})
        check_loop("tool", {"arg": "val"})

        detector.reset_session()

        check_loop("tool", {"arg": "val"})  # Should work again
        check_loop("tool", {"arg": "val"})
        # No error - counter was reset

    def test_reset_all(self):
        """reset_all should clear all sessions."""
        configure(loop_threshold=2)
        detector = get_loop_detector()

        check_loop("tool", {"arg": "val"}, session_id="a")
        check_loop("tool", {"arg": "val"}, session_id="a")
        check_loop("tool", {"arg": "val"}, session_id="b")
        check_loop("tool", {"arg": "val"}, session_id="b")

        detector.reset_all()

        # Both sessions should be reset
        check_loop("tool", {"arg": "val"}, session_id="a")
        check_loop("tool", {"arg": "val"}, session_id="b")


class TestArgumentNormalization:
    """Tests for argument canonicalization."""

    def test_dict_order_independent(self):
        """Dict key order should not affect signature."""
        configure(loop_threshold=1)

        check_loop("tool", {"a": 1, "b": 2})

        with pytest.raises(SentinelLoopError):
            check_loop("tool", {"b": 2, "a": 1})  # Same args, different order

    def test_none_args_handled(self):
        """None args should work."""
        configure(loop_threshold=1)

        check_loop("tool", None)

        with pytest.raises(SentinelLoopError):
            check_loop("tool", None)

    def test_empty_args_handled(self):
        """Empty args should work."""
        configure(loop_threshold=1)

        check_loop("tool", {})

        with pytest.raises(SentinelLoopError):
            check_loop("tool", {})

    def test_none_and_empty_equivalent(self):
        """None and {} should be equivalent."""
        configure(loop_threshold=1)

        check_loop("tool", None)

        with pytest.raises(SentinelLoopError):
            check_loop("tool", {})

    def test_nested_dict_args(self):
        """Nested dicts should be handled correctly."""
        configure(loop_threshold=1)

        check_loop("tool", {"outer": {"inner": "value"}})

        with pytest.raises(SentinelLoopError):
            check_loop("tool", {"outer": {"inner": "value"}})

    def test_list_args(self):
        """List args should be handled correctly."""
        configure(loop_threshold=1)

        check_loop("tool", {"items": [1, 2, 3]})

        with pytest.raises(SentinelLoopError):
            check_loop("tool", {"items": [1, 2, 3]})


class TestThreadSafety:
    """Tests for concurrent access."""

    def test_concurrent_checks(self):
        """Multiple threads should be able to check safely."""
        configure(loop_threshold=100)  # High threshold

        errors = []
        success_count = [0]  # Use list for mutability in closure
        lock = threading.Lock()

        def worker():
            try:
                for i in range(20):
                    check_loop("tool", {"iter": i})
                with lock:
                    success_count[0] += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"
        assert success_count[0] == 4

    def test_concurrent_same_signature(self):
        """Multiple threads checking same signature should be thread-safe."""
        configure(loop_threshold=50)

        errors = []
        total_calls = [0]
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(10):
                    check_loop("same_tool", {"same": "args"})
                    with lock:
                        total_calls[0] += 1
            except SentinelLoopError:
                # Expected when threshold hit
                pass
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors: {errors}"


class TestDecoratorIntegration:
    """Tests for @loop_guard decorator."""

    def test_decorator_tracks_function_calls(self):
        """Decorator should track calls based on function args."""
        from agent_fuse import loop_guard

        configure(loop_threshold=2)

        @loop_guard()
        def my_function(x: int, y: str = "default"):
            return f"{x}-{y}"

        my_function(1, "test")
        my_function(1, "test")

        with pytest.raises(SentinelLoopError):
            my_function(1, "test")

    def test_decorator_custom_name(self):
        """Decorator should use custom tool name if provided."""
        from agent_fuse import loop_guard

        configure(loop_threshold=1)

        @loop_guard(tool_name="custom_name")
        def my_function(x: int):
            return x

        my_function(1)

        with pytest.raises(SentinelLoopError) as exc_info:
            my_function(1)

        assert "custom_name" in exc_info.value.signature

    def test_decorator_different_args_allowed(self):
        """Decorator should allow different args."""
        from agent_fuse import loop_guard

        configure(loop_threshold=2)

        @loop_guard()
        def my_function(x: int):
            return x

        my_function(1)
        my_function(1)
        my_function(2)  # Different arg - separate counter
        my_function(2)
        # No error

    def test_decorator_uses_function_name(self):
        """Decorator should default to function name."""
        from agent_fuse import loop_guard

        configure(loop_threshold=1)

        @loop_guard()
        def search_web(query: str):
            return query

        search_web("test")

        with pytest.raises(SentinelLoopError) as exc_info:
            search_web("test")

        assert "search_web" in exc_info.value.signature


class TestGetCounts:
    """Tests for get_counts debugging method."""

    def test_get_counts_empty(self):
        """get_counts should return empty dict initially."""
        configure()
        detector = get_loop_detector()

        assert detector.get_counts() == {}

    def test_get_counts_tracks_calls(self):
        """get_counts should track call counts."""
        configure()
        detector = get_loop_detector()

        check_loop("tool_a", {"x": 1})
        check_loop("tool_a", {"x": 1})
        check_loop("tool_b", {"y": 2})

        counts = detector.get_counts()
        assert len(counts) == 2
        assert any("tool_a" in k for k in counts.keys())
        assert any("tool_b" in k for k in counts.keys())


class TestSingletonBehavior:
    """Tests for singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """get_instance should return the same instance."""
        configure()

        detector1 = LoopDetector.get_instance()
        detector2 = LoopDetector.get_instance()

        assert detector1 is detector2

    def test_reset_clears_singleton(self):
        """reset_instance should clear the singleton."""
        configure()

        detector1 = LoopDetector.get_instance()
        LoopDetector.reset_instance()
        detector2 = LoopDetector.get_instance()

        assert detector1 is not detector2

    def test_module_functions_use_singleton(self):
        """Module-level functions should use singleton."""
        configure()

        # First call creates singleton
        check_loop("tool", {})

        # Get the detector and verify it tracked the call
        detector = get_loop_detector()
        counts = detector.get_counts()
        assert len(counts) == 1
