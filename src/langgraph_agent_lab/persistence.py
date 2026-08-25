"""Checkpointer adapters for the lab workflow."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpointer(
    path: str | Path = "checkpoints.db",
    database_url: str | Path | None = None,
) -> BaseCheckpointSaver | None:
    """Create a LangGraph checkpointer.

    A filesystem path creates a SQLite checkpointer. The ``memory``/``none``
    values and ``database_url`` argument are retained for CLI compatibility.
    SQLite resources can be released with ``close()`` or a ``with`` block.
    """
    requested = str(path)
    if requested == "none":
        return None
    if requested == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if requested == "postgres":
        raise NotImplementedError("Postgres persistence is outside Gate 3 scope")

    if requested == "sqlite":
        database_path = str(database_url or "checkpoints.db")
    else:
        database_path = requested

    return _build_sqlite_saver(database_path)


def _build_sqlite_saver(database_path: str) -> BaseCheckpointSaver:
    """Open a SQLite connection and wrap it in the current SqliteSaver API."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    class ManagedSqliteSaver(SqliteSaver):
        def close(self) -> None:
            self.conn.close()

        def __enter__(self) -> ManagedSqliteSaver:
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            self.close()

    target = Path(database_path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.commit()
    return ManagedSqliteSaver(connection)
