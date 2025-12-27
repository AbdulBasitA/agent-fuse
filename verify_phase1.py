import os
import sqlite3
import shutil
from pathlib import Path
from agent_fuse.config import configure, get_settings, reset_settings
from agent_fuse.storage.sqlite import AgentFuseDB, reset_db
from agent_fuse.core.exceptions import SentinelSystemError

def test_wal_mode():
    print("[Test 1] SQLite WAL Mode Concurrency...")

    # 1. Initialize DB
    reset_settings()
    reset_db()
    settings = configure(db_path=Path("./test_env/agent_fuse_test.db"))
    storage = AgentFuseDB(settings)
    storage.initialize()

    # 2. Verify WAL mode is actually active
    conn = storage._get_connection()
    cursor = conn.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]

    if mode.lower() == 'wal':
        print("[OK] SUCCESS: Database is in WAL mode.")
    else:
        print(f"[FAIL] FAILURE: Database is in '{mode}' mode (Expected 'wal').")
        exit(1)

def test_fail_safe_logic():
    print("\n[Test 2] Fail-Safe Configuration...")

    # Path to a broken DB (directory instead of file forces an error)
    bad_path = Path("./test_env/bad_db")
    os.makedirs(bad_path, exist_ok=True)
    # Create a directory where the DB file should be (this will cause sqlite3.OperationalError)
    os.makedirs(bad_path / "file.db", exist_ok=True)

    # Case A: FAIL_SAFE = True (Should Crash)
    print("   [A] Testing FAIL_SAFE=True (Expect Crash)...")
    reset_settings()
    reset_db()
    settings = configure(db_path=bad_path / "file.db", fail_safe=True)

    try:
        # Try to write to the broken path
        storage = AgentFuseDB(settings)
        storage.log_usage("test_model", 10, 10, 0.05)
        print("[FAIL] FAILURE: AgentFuse did NOT raise error (Unsafe).")
        exit(1)
    except SentinelSystemError:
        print("[OK] SUCCESS: AgentFuse raised SystemError as expected.")
    except Exception as e:
        print(f"[FAIL] FAILURE: Raised wrong error: {type(e).__name__}: {e}")
        exit(1)

    # Case B: FAIL_SAFE = False (Should Warn but Continue)
    print("   [B] Testing FAIL_SAFE=False (Expect Warning)...")
    reset_settings()
    reset_db()
    settings = configure(db_path=bad_path / "file.db", fail_safe=False)

    try:
        # This should print a red error to stderr but NOT crash the script
        storage = AgentFuseDB(settings)
        storage.log_usage("test_model", 10, 10, 0.05)
        print("[OK] SUCCESS: AgentFuse swallowed the error and continued.")
    except Exception as e:
        print(f"[FAIL] FAILURE: AgentFuse crashed with {e} (Availability failed).")
        exit(1)

def cleanup():
    if os.path.exists("./test_env"):
        shutil.rmtree("./test_env")

if __name__ == "__main__":
    cleanup()
    os.makedirs("./test_env", exist_ok=True)

    try:
        test_wal_mode()
        test_fail_safe_logic()
        print("\n[PHASE 1 COMPLETE] Foundation is solid.")
    finally:
        cleanup()
