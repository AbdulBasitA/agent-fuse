"""
Agent Fuse SQLite Storage

A robust SQLite wrapper with:
- WAL mode for concurrent access
- Retry logic with exponential backoff
- Thread-safe connection management
- FAIL_SAFE behavior (block or warn on errors)
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

from agent_fuse.config import get_settings
from agent_fuse.core.exceptions import SentinelSystemError

if TYPE_CHECKING:
    from agent_fuse.config import AgentFuseSettings

logger = logging.getLogger("agent_fuse.storage")

# Load schema SQL at module level
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _load_schema() -> str:
    """Load the SQL schema from file."""
    return _SCHEMA_PATH.read_text()


class AgentFuseDB:
    """
    Thread-safe SQLite database wrapper for Agent Fuse.

    This class manages database connections with:
    - WAL mode enabled for concurrent read/write
    - Automatic retry on transient errors
    - FAIL_SAFE behavior based on configuration
    - Thread-local connection management

    Example:
        >>> db = AgentFuseDB()
        >>> with db.connection() as conn:
        ...     cursor = conn.execute("SELECT * FROM usage_logs")
        ...     rows = cursor.fetchall()
    """

    def __init__(self, settings: AgentFuseSettings | None = None) -> None:
        """
        Initialize the database wrapper.

        Args:
            settings: Optional settings override. If not provided,
                     uses the global settings from get_settings().
        """
        self._settings = settings or get_settings()
        self._local = threading.local()
        self._lock = threading.RLock()
        self._initialized = False

    @property
    def db_path(self) -> Path:
        """Get the database file path."""
        return self._settings.db_path

    def _get_connection(self) -> sqlite3.Connection | None:
        """Get or create a thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = self._create_connection()
        return self._local.connection

    def _create_connection(self) -> sqlite3.Connection | None:
        """
        Create a new database connection with proper configuration.

        Returns:
            Configured SQLite connection

        Raises:
            SentinelSystemError: If connection fails and FAIL_SAFE is True
        """
        # Ensure directory exists
        self._settings.ensure_db_directory()

        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=5.0,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode for explicit transaction control
            )

            # Enable WAL mode for concurrent access (CRITICAL)
            conn.execute("PRAGMA journal_mode=WAL;")

            # Set busy timeout for lock handling
            conn.execute("PRAGMA busy_timeout=5000;")

            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON;")

            # Row factory for dict-like access
            conn.row_factory = sqlite3.Row

            return conn

        except sqlite3.Error as e:
            return self._handle_error("connect", e)

    def initialize(self) -> None:
        """
        Initialize the database schema.

        This method is idempotent - safe to call multiple times.
        It creates all required tables and indices if they don't exist.

        Raises:
            SentinelSystemError: If initialization fails and FAIL_SAFE is True
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                conn = self._get_connection()
                if conn is None:
                    # Connection failed but FAIL_SAFE=False allowed it
                    # Mark as "initialized" to prevent repeated attempts
                    self._initialized = True
                    return

                schema = _load_schema()
                conn.executescript(schema)
                self._initialized = True
                logger.debug("Database initialized at %s", self.db_path)

            except (sqlite3.Error, OSError) as e:
                self._handle_error("initialize", e)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database operations.

        Provides a connection with automatic initialization and
        error handling based on FAIL_SAFE configuration.

        Yields:
            SQLite connection

        Example:
            >>> with db.connection() as conn:
            ...     conn.execute("INSERT INTO usage_logs ...")
        """
        self.initialize()
        yield self._get_connection()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for transactional database operations.

        Wraps operations in BEGIN/COMMIT with ROLLBACK on error.

        Yields:
            SQLite connection within a transaction

        Example:
            >>> with db.transaction() as conn:
            ...     conn.execute("INSERT ...")
            ...     conn.execute("UPDATE ...")
        """
        self.initialize()
        conn = self._get_connection()

        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def execute_with_retry(
        self,
        operation: str,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
    ) -> sqlite3.Cursor | None:
        """
        Execute SQL with automatic retry on transient errors.

        Args:
            operation: Description of the operation (for error messages)
            sql: SQL statement to execute
            params: Parameters for the SQL statement

        Returns:
            Cursor with results, or None if failed and FAIL_SAFE=False

        Raises:
            SentinelSystemError: If all retries fail and FAIL_SAFE is True
        """
        self.initialize()

        # If connection is None (FAIL_SAFE=False case), return None
        conn = self._get_connection()
        if conn is None:
            return None

        last_error: Exception | None = None
        retry_count = 0

        for attempt in range(self._settings.max_retries):
            try:
                return conn.execute(sql, params)

            except sqlite3.OperationalError as e:
                last_error = e
                retry_count = attempt + 1

                # Check if this is a retryable error
                error_msg = str(e).lower()
                if "locked" in error_msg or "busy" in error_msg:
                    delay = self._settings.retry_delay * (2**attempt)
                    logger.warning(
                        "Database locked, retrying in %.2fs (attempt %d/%d)",
                        delay,
                        retry_count,
                        self._settings.max_retries,
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable error
                    break

            except sqlite3.Error as e:
                last_error = e
                retry_count = attempt + 1
                break

        # All retries exhausted
        return self._handle_error(operation, last_error, retry_count)

    def _handle_error(
        self,
        operation: str,
        error: Exception | None,
        retry_count: int = 0,
    ) -> Any:
        """
        Handle database errors based on FAIL_SAFE configuration.

        Args:
            operation: Description of the failed operation
            error: The exception that occurred
            retry_count: Number of retries attempted

        Returns:
            None if FAIL_SAFE is False (allows agent to proceed)

        Raises:
            SentinelSystemError: If FAIL_SAFE is True
        """
        if self._settings.fail_safe:
            raise SentinelSystemError(
                f"Database operation '{operation}' failed",
                operation=operation,
                original_error=error,
                retry_count=retry_count,
            )
        else:
            # Log to stderr and allow agent to proceed
            msg = (
                f"AGENT FUSE FAILED: Database operation '{operation}' failed "
                f"after {retry_count} retries. "
                f"Error: {type(error).__name__}: {error}. "
                "Agent will proceed without tracking (FAIL_SAFE=False)."
            )
            print(msg, file=sys.stderr)
            logger.error(msg)
            return None

    def log_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        session_id: str | None = None,
        metadata: str | None = None,
    ) -> int | None:
        """
        Log an LLM usage record.

        Args:
            model: Model identifier (e.g., 'gpt-4')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD
            session_id: Optional session identifier
            metadata: Optional JSON metadata string

        Returns:
            Row ID of the inserted record, or None if insert failed
        """
        session_id = session_id or self._settings.session_id

        cursor = self.execute_with_retry(
            "log_usage",
            """
            INSERT INTO usage_logs (model, input_tokens, output_tokens, cost_usd, session_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model, input_tokens, output_tokens, cost_usd, session_id, metadata),
        )

        if cursor is None:
            return None

        return cursor.lastrowid

    def get_total_spend(self, session_id: str | None = None) -> float:
        """
        Get total spend in USD.

        Args:
            session_id: Optional session to filter by. If None, returns total across all sessions.

        Returns:
            Total spend in USD, or 0.0 if query fails
        """
        if session_id:
            cursor = self.execute_with_retry(
                "get_total_spend",
                "SELECT COALESCE(SUM(cost_usd), 0) FROM usage_logs WHERE session_id = ?",
                (session_id,),
            )
        else:
            cursor = self.execute_with_retry(
                "get_total_spend",
                "SELECT COALESCE(SUM(cost_usd), 0) FROM usage_logs",
            )

        if cursor is None:
            return 0.0

        result = cursor.fetchone()
        return float(result[0]) if result else 0.0

    def get_usage_stats(self, session_id: str | None = None) -> dict[str, Any]:
        """
        Get usage statistics.

        Args:
            session_id: Optional session to filter by

        Returns:
            Dictionary with usage statistics
        """
        if session_id:
            cursor = self.execute_with_retry(
                "get_usage_stats",
                """
                SELECT
                    COALESCE(SUM(cost_usd), 0) as total_spend_usd,
                    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COUNT(*) as total_calls
                FROM usage_logs
                WHERE session_id = ?
                """,
                (session_id,),
            )
        else:
            cursor = self.execute_with_retry(
                "get_usage_stats",
                """
                SELECT
                    COALESCE(SUM(cost_usd), 0) as total_spend_usd,
                    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COUNT(*) as total_calls
                FROM usage_logs
                """,
            )

        if cursor is None:
            return {
                "total_spend_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
            }

        row = cursor.fetchone()
        return dict(row) if row else {}

    def close(self) -> None:
        """Close the thread-local database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            try:
                self._local.connection.close()
            except sqlite3.Error:
                pass
            self._local.connection = None

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.close()


# Module-level singleton instance
_db_instance: AgentFuseDB | None = None
_db_lock = threading.Lock()


def get_db() -> AgentFuseDB:
    """
    Get the global AgentFuseDB instance.

    Creates the instance on first call (lazy initialization).
    Thread-safe.

    Returns:
        The global AgentFuseDB instance
    """
    global _db_instance

    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = AgentFuseDB()

    return _db_instance


def reset_db() -> None:
    """Reset the global database instance (useful for testing)."""
    global _db_instance

    with _db_lock:
        if _db_instance is not None:
            _db_instance.close()
            _db_instance = None
