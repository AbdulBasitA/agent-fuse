-- Sentinel Guard Database Schema
-- Version: 1.0
--
-- This schema supports tracking LLM usage and enforcing budget limits.
-- Designed for SQLite with WAL mode for concurrent access.

-- Usage logs table: Records each LLM call with token counts and costs
CREATE TABLE IF NOT EXISTS usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Timestamp of the LLM call (UTC)
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),

    -- Model identifier (e.g., 'gpt-4', 'claude-3-opus')
    model TEXT NOT NULL,

    -- Token counts
    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),

    -- Cost in USD (calculated based on model pricing)
    cost_usd REAL NOT NULL CHECK (cost_usd >= 0),

    -- Optional session identifier for grouping related calls
    session_id TEXT,

    -- Optional metadata (JSON string for extensibility)
    metadata TEXT
);

-- Limits table: Stores configurable limits (key-value pairs)
CREATE TABLE IF NOT EXISTS limits (
    -- Limit key (e.g., 'budget_usd', 'max_calls_per_minute')
    key TEXT PRIMARY KEY,

    -- Limit value (numeric)
    value REAL NOT NULL,

    -- Last update timestamp
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indices for efficient querying
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_logs(model);

-- View for quick spend summary
CREATE VIEW IF NOT EXISTS spend_summary AS
SELECT
    COALESCE(SUM(cost_usd), 0) as total_spend_usd,
    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
    COUNT(*) as total_calls,
    MIN(timestamp) as first_call,
    MAX(timestamp) as last_call
FROM usage_logs;

-- View for spend by session
CREATE VIEW IF NOT EXISTS spend_by_session AS
SELECT
    session_id,
    COALESCE(SUM(cost_usd), 0) as total_spend_usd,
    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
    COUNT(*) as call_count,
    MIN(timestamp) as first_call,
    MAX(timestamp) as last_call
FROM usage_logs
GROUP BY session_id;

-- View for spend by model
CREATE VIEW IF NOT EXISTS spend_by_model AS
SELECT
    model,
    COALESCE(SUM(cost_usd), 0) as total_spend_usd,
    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
    COUNT(*) as call_count
FROM usage_logs
GROUP BY model;
