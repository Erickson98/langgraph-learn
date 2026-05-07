"""SQLite-backed LangGraph store for module 5 long-term memory."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchOp,
)


class SqliteStore(BaseStore):
    """Persistent key-value store backed by SQLite.

    Provides the same interface as ``InMemoryStore`` but persists data to a
    SQLite database file so that profile, todo, and instruction memory survive
    process restarts.

    The store shares the same file as the SQLite checkpointer (different table)
    so the module only needs one configured path.
    """

    _TABLE = "module5_store"
    _SEP = "\x1f"

    def __init__(self, db_path: str) -> None:
        """Open or create the SQLite store.

        Args:
            db_path: Path to the SQLite database file.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_table()


    def _init_table(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    namespace  TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """)
            conn.commit()

    def _ns(self, namespace: tuple[str, ...]) -> str:
        return self._SEP.join(namespace)

    def _parse_ns(self, s: str) -> tuple[str, ...]:
        return tuple(s.split(self._SEP))

    def _to_item(self, row: tuple[Any, ...]) -> Item:
        ns_str, key, value_json, created_ts, updated_ts = row
        return Item(
            namespace=self._parse_ns(ns_str),
            key=key,
            value=json.loads(value_json),
            created_at=datetime.fromtimestamp(created_ts, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(updated_ts, tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Per-operation handlers (called by _run_batch)
    # ------------------------------------------------------------------

    def _exec_put(self, conn: sqlite3.Connection, op: PutOp) -> None:
        ns = self._ns(op.namespace)
        if op.value is None:
            conn.execute(
                f"DELETE FROM {self._TABLE} WHERE namespace=? AND key=?",
                (ns, op.key),
            )
        else:
            now = datetime.now(timezone.utc).timestamp()
            conn.execute(
                f"""
                INSERT INTO {self._TABLE}
                    (namespace, key, value, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value      = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (ns, op.key, json.dumps(op.value), now, now),
            )

    def _exec_get(self, conn: sqlite3.Connection, op: GetOp) -> Item | None:
        row = conn.execute(
            f"""
            SELECT namespace, key, value, created_at, updated_at
            FROM {self._TABLE}
            WHERE namespace=? AND key=?
            """,
            (self._ns(op.namespace), op.key),
        ).fetchone()
        return self._to_item(row) if row else None

    def _exec_search(self, conn: sqlite3.Connection, op: SearchOp) -> list[Item]:
        ns_prefix = self._ns(op.namespace_prefix)
        limit = op.limit or 10
        offset = op.offset or 0
        like_prefix = ns_prefix + self._SEP + "%"

        if op.filter:
            # Fetch all rows before filtering: applying LIMIT/OFFSET in SQL
            # before the in-process filter would drop matching rows that fall
            # beyond the first N results.
            rows = conn.execute(
                f"""
                SELECT namespace, key, value, created_at, updated_at
                FROM {self._TABLE}
                WHERE namespace = ? OR namespace LIKE ?
                ORDER BY updated_at DESC
                """,
                (ns_prefix, like_prefix),
            ).fetchall()
            items = [
                item
                for item in (self._to_item(r) for r in rows)
                if all(item.value.get(k) == v for k, v in op.filter.items())
            ]
            return items[offset : offset + limit]

        rows = conn.execute(
            f"""
            SELECT namespace, key, value, created_at, updated_at
            FROM {self._TABLE}
            WHERE namespace = ? OR namespace LIKE ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (ns_prefix, like_prefix, limit, offset),
        ).fetchall()
        return [self._to_item(r) for r in rows]

    def _exec_list_namespaces(
        self, conn: sqlite3.Connection, op: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        rows = conn.execute(f"SELECT DISTINCT namespace FROM {self._TABLE}").fetchall()
        namespaces: list[tuple[str, ...]] = [self._parse_ns(r[0]) for r in rows]

        if op.max_depth is not None:
            namespaces = _truncate_to_depth(namespaces, op.max_depth)

        if op.match_conditions:
            namespaces = [
                ns for ns in namespaces if _ns_matches(ns, op.match_conditions)
            ]

        limit = op.limit or 100
        offset = op.offset or 0
        return namespaces[offset : offset + limit]


    def _run_batch(self, ops: list[Any]) -> list[Any]:
        results: list[Any] = []
        with sqlite3.connect(self._db_path) as conn:
            for op in ops:
                if isinstance(op, PutOp):
                    self._exec_put(conn, op)
                    results.append(None)
                elif isinstance(op, GetOp):
                    results.append(self._exec_get(conn, op))
                elif isinstance(op, SearchOp):
                    results.append(self._exec_search(conn, op))
                elif isinstance(op, ListNamespacesOp):
                    results.append(self._exec_list_namespaces(conn, op))
                else:
                    results.append(None)
            conn.commit()
        return results


    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute a batch of store operations synchronously.

        Args:
            ops: Sequence of store operations.

        Returns:
            Result for each operation in the same order.
        """
        return self._run_batch(list(ops))

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute a batch of store operations asynchronously.

        Delegates to ``_run_batch`` in a thread pool so the event loop is
        never blocked by SQLite I/O.

        Args:
            ops: Sequence of store operations.

        Returns:
            Result for each operation in the same order.
        """
        return await asyncio.to_thread(self._run_batch, list(ops))



def _truncate_to_depth(
    namespaces: list[tuple[str, ...]], max_depth: int
) -> list[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for ns in namespaces:
        truncated = ns[:max_depth]
        if truncated not in seen:
            seen.add(truncated)
            result.append(truncated)
    return result


def _pattern_matches(
    ns: tuple[str, ...], path: tuple[str, ...], segments: tuple[str, ...]
) -> bool:
    if len(ns) < len(path):
        return False
    return all(pat == "*" or pat == seg for pat, seg in zip(path, segments))


def _ns_matches(ns: tuple[str, ...], conditions: Any) -> bool:
    for cond in conditions:
        path: tuple[str, ...] = cond.path
        if cond.match_type == "prefix":
            if not _pattern_matches(ns, path, ns):
                return False
        elif cond.match_type == "suffix":
            if not _pattern_matches(ns, path, ns[-len(path) :]):
                return False
    return True


def build_store(memory_db: str) -> BaseStore:
    """Return a store backed by ``memory_db``.

    Args:
        memory_db: SQLite file path, or ``':memory:'`` for an ephemeral
            in-process store (useful in tests and the CLI).

    Returns:
        Persistent ``SqliteStore`` for file paths, or ``InMemoryStore``
        for the ``':memory:'`` sentinel.
    """
    if memory_db == ":memory:":
        from langgraph.store.memory import InMemoryStore

        return InMemoryStore()
    return SqliteStore(memory_db)
