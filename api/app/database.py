from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import psycopg2
import psycopg2.errors
import psycopg2.extras
import psycopg2.pool

from .config import settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Cursor:
    """Thin wrapper to make sqlite3 and psycopg2 look the same to the repository layer."""

    def __init__(self, cur, *, backend: str) -> None:
        self._cur = cur
        self._backend = backend

    def _normalize_sql(self, sql: str) -> str:
        if self._backend == "sqlite":
            return sql.replace("%s", "?")
        return sql

    def execute(self, sql: str, params=None) -> "_Cursor":
        normalized = self._normalize_sql(sql)
        if params is None:
            self._cur.execute(normalized)
        else:
            self._cur.execute(normalized, params)
        return self

    def executemany(self, sql: str, params_seq) -> "_Cursor":
        self._cur.executemany(self._normalize_sql(sql), params_seq)
        return self

    def fetchall(self):
        return self._cur.fetchall()

    def fetchone(self):
        return self._cur.fetchone()

    @property
    def lastrowid(self) -> int:
        if self._backend == "sqlite":
            row = self._cur.fetchone()
            if row is not None:
                return int(row["id"])
            return int(self._cur.lastrowid)
        row = self._cur.fetchone()
        return row["id"]


def _sqlite_database_path(database_url: str) -> Path:
    if not database_url:
        return settings.data_dir / "app.db"
    if not database_url.startswith("sqlite:///"):
        raise RuntimeError(f"Unsupported database URL: {database_url}")
    raw_path = database_url.removeprefix("sqlite:///")
    if raw_path == ":memory:":
        return Path(":memory:")
    return Path(raw_path)


def _database_backend() -> str:
    if not settings.database_url or settings.database_url.startswith("sqlite:///"):
        return "sqlite"
    return "postgres"


def _safe_database_target(database_url: str) -> str:
    if not database_url:
        return f"sqlite={_sqlite_database_path(database_url)}"
    if database_url.startswith("sqlite:///"):
        return f"sqlite={_sqlite_database_path(database_url)}"
    parsed = urlparse(database_url)
    host = parsed.hostname or "<missing>"
    port = parsed.port or "<missing>"
    # The username is deliberately omitted: /health/details is unauthenticated
    # and a database user name is useful to an attacker.
    return f"host={host} port={port}"


def _database_connection_error(exc: Exception) -> RuntimeError:
    details = str(exc).strip()
    message = f"Database connection failed for {_safe_database_target(settings.database_url)}."
    if "Tenant or user not found" in details:
        message += (
            " Supabase pooler credentials look invalid. If you are using the Supabase pooler on port 6543, "
            "the username must include the project ref, for example "
            "`postgres.<project-ref>`, and the password must be the database password for that project."
        )
    return RuntimeError(message)


def _connect_sqlite():
    database_path = _sqlite_database_path(settings.database_url)
    if database_path != Path(":memory:"):
        database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(database_path)
    except sqlite3.Error as exc:
        raise _database_connection_error(exc) from exc
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Lazily create the shared Postgres pool.

    Opening a fresh connection per query is prohibitively expensive against a
    hosted pooler, so Postgres connections are reused. SQLite stays per-call
    because it is local and cheap.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                try:
                    _pool = psycopg2.pool.ThreadedConnectionPool(
                        settings.database_pool_min_size,
                        settings.database_pool_max_size,
                        settings.database_url,
                    )
                except psycopg2.OperationalError as exc:
                    raise _database_connection_error(exc) from exc
    return _pool


def reset_pool() -> None:
    """Drop the shared pool so the next call reconnects with current settings."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


@contextmanager
def get_connection() -> Iterator[_Cursor]:
    backend = _database_backend()
    pool = None
    if backend == "sqlite":
        conn = _connect_sqlite()
        cur = conn.cursor()
    else:
        pool = _get_pool()
        conn = pool.getconn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    wrapper = _Cursor(cur, backend=backend)
    try:
        yield wrapper
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        if pool is None:
            conn.close()
        else:
            pool.putconn(conn)


def _create_shared_schema(connection: _Cursor) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL DEFAULT 'legacy',
            name TEXT NOT NULL,
            support_domain_description TEXT NOT NULL,
            allowed_topics_json TEXT NOT NULL,
            disallowed_topics_json TEXT NOT NULL,
            routing_notes TEXT NOT NULL,
            agent_model TEXT NOT NULL,
            max_rounds INTEGER NOT NULL,
            target_macro_f1 REAL NOT NULL,
            target_out_of_scope_precision REAL NOT NULL,
            sandbox_profile TEXT NOT NULL,
            promoted_run_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS examples (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            source TEXT NOT NULL,
            approved BOOLEAN NOT NULL,
            split TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            stop_reason TEXT,
            best_round_id TEXT,
            best_macro_f1 REAL,
            summary TEXT,
            workspace_root TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # No surrogate key: this keeps the DDL identical on SQLite and PostgreSQL.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rate_limit_hits (
            bucket TEXT NOT NULL,
            hit_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rounds (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            round_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            candidate_file TEXT,
            dataset_summary_file TEXT,
            review_file TEXT,
            evaluation_file TEXT,
            holdout_file TEXT,
            holdout_evaluation_file TEXT,
            metrics_json TEXT NOT NULL,
            holdout_metrics_json TEXT NOT NULL DEFAULT '{}',
            checkpoint_path TEXT,
            note TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def _init_sqlite_schema(connection: _Cursor) -> None:
    _create_shared_schema(connection)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_projects_session_id ON projects(session_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_examples_project_id ON examples(project_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id)"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_rounds_run_round ON rounds(run_id, round_index)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_rate_limit_hits_bucket ON rate_limit_hits(bucket)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)"
    )


def _init_postgres_schema(connection: _Cursor) -> None:
    _create_shared_schema(connection)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS run_events (
            id BIGSERIAL PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        ALTER TABLE projects
        ADD COLUMN IF NOT EXISTS session_id TEXT NOT NULL DEFAULT 'legacy'
        """
    )
    connection.execute(
        """
        ALTER TABLE rounds
        ADD COLUMN IF NOT EXISTS holdout_file TEXT
        """
    )
    connection.execute(
        """
        ALTER TABLE rounds
        ADD COLUMN IF NOT EXISTS holdout_evaluation_file TEXT
        """
    )
    connection.execute(
        """
        ALTER TABLE rounds
        ADD COLUMN IF NOT EXISTS holdout_metrics_json TEXT NOT NULL DEFAULT '{}'
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_projects_session_id
        ON projects(session_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_examples_project_id
        ON examples(project_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_project_id
        ON runs(project_id)
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_rounds_run_round
        ON rounds(run_id, round_index)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_run_events_run_id
        ON run_events(run_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rate_limit_hits_bucket
        ON rate_limit_hits(bucket)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_status
        ON runs(status)
        """
    )


def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        if _database_backend() == "sqlite":
            _init_sqlite_schema(connection)
        else:
            _init_postgres_schema(connection)


def decode_json(value: str) -> list[str] | dict[str, object]:
    return json.loads(value)


def encode_json(value: object) -> str:
    return json.dumps(value)
