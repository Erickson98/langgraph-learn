"""Tests for the module 5 SQLite store."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.module5.services.sqlite_store import SqliteStore, build_store


class SqliteStoreTests(unittest.TestCase):
    """Verify SqliteStore persistence and interface correctness."""

    def setUp(self) -> None:
        """Create a fresh temporary directory for each test."""
        self.tmp = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmp) / "test.sqlite")

    def tearDown(self) -> None:
        """Remove the temporary directory after each test."""
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _store(self) -> SqliteStore:
        return SqliteStore(self.db_path)

    def test_put_and_get_roundtrip(self) -> None:
        """A value written with put should be readable with get."""
        store = self._store()
        store.put(("profile", "user-1"), "profile", {"name": "Ada"})

        item = store.get(("profile", "user-1"), "profile")

        self.assertIsNotNone(item)
        self.assertEqual(item.value["name"], "Ada")
        self.assertEqual(item.key, "profile")
        self.assertEqual(item.namespace, ("profile", "user-1"))

    def test_get_returns_none_for_missing_key(self) -> None:
        """get should return None for a key that was never written."""
        self.assertIsNone(self._store().get(("profile", "user-1"), "missing"))

    def test_put_none_deletes_item(self) -> None:
        """Putting None as the value should remove the item from the store."""
        store = self._store()
        store.put(("profile", "user-1"), "profile", {"name": "Ada"})
        store.put(("profile", "user-1"), "profile", None)

        self.assertIsNone(store.get(("profile", "user-1"), "profile"))

    def test_put_updates_existing_value(self) -> None:
        """A second put on the same key should overwrite the value."""
        store = self._store()
        store.put(("profile", "user-1"), "profile", {"name": "Ada"})
        store.put(("profile", "user-1"), "profile", {"name": "Ada Lovelace"})

        item = store.get(("profile", "user-1"), "profile")
        self.assertEqual(item.value["name"], "Ada Lovelace")

    def test_search_returns_all_items_in_namespace(self) -> None:
        """search on an exact namespace should return all items stored there."""
        store = self._store()
        store.put(("todo", "user-1"), "todo-1", {"task": "Ship module 5"})
        store.put(("todo", "user-1"), "todo-2", {"task": "Write tests"})
        store.put(("todo", "user-2"), "todo-1", {"task": "Other user task"})

        items = store.search(("todo", "user-1"))
        tasks = {item.value["task"] for item in items}

        self.assertEqual(len(items), 2)
        self.assertIn("Ship module 5", tasks)
        self.assertIn("Write tests", tasks)
        self.assertNotIn("Other user task", tasks)

    def test_search_returns_empty_for_unknown_namespace(self) -> None:
        """search on a namespace with no data should return an empty list."""
        self.assertEqual(self._store().search(("todo", "nobody")), [])

    def test_search_applies_filter(self) -> None:
        """search with a filter dict should exclude non-matching items."""
        store = self._store()
        store.put(("todo", "u1"), "t1", {"task": "A", "status": "done"})
        store.put(("todo", "u1"), "t2", {"task": "B", "status": "pending"})

        items = store.search(("todo", "u1"), filter={"status": "done"})

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].value["task"], "A")

    def test_data_persists_across_separate_instances(self) -> None:
        """Data written by one store instance should be visible from a new one.

        This simulates a process restart: the first instance writes, the
        second instance (same file) reads back the persisted value.
        """
        store1 = self._store()
        store1.put(("profile", "user-1"), "profile", {"name": "Ada"})

        store2 = self._store()  # new connection, same file
        item = store2.get(("profile", "user-1"), "profile")

        self.assertIsNotNone(item)
        self.assertEqual(item.value["name"], "Ada")

    def test_todos_accumulate_across_instances(self) -> None:
        """Multiple todos written across instances should all be readable."""
        store1 = self._store()
        store1.put(("todo", "user-1"), "todo-1", {"task": "Task A"})

        store2 = self._store()
        store2.put(("todo", "user-1"), "todo-2", {"task": "Task B"})

        store3 = self._store()
        items = store3.search(("todo", "user-1"))
        tasks = {i.value["task"] for i in items}

        self.assertEqual({"Task A", "Task B"}, tasks)

    def test_instructions_persist_and_update(self) -> None:
        """User preferences should survive restarts and support in-place updates."""
        store1 = self._store()
        store1.put(
            ("instructions", "user-1"),
            "user_instructions",
            {"memory": "Prioritize urgent tasks."},
        )

        store2 = self._store()
        item = store2.get(("instructions", "user-1"), "user_instructions")
        self.assertEqual(item.value["memory"], "Prioritize urgent tasks.")

        store2.put(
            ("instructions", "user-1"),
            "user_instructions",
            {"memory": "Always show deadlines first."},
        )

        store3 = self._store()
        item = store3.get(("instructions", "user-1"), "user_instructions")
        self.assertEqual(item.value["memory"], "Always show deadlines first.")

    def test_item_timestamps_are_populated(self) -> None:
        """Items should carry created_at and updated_at datetime values."""
        store = self._store()
        store.put(("profile", "user-1"), "p", {"x": 1})

        item = store.get(("profile", "user-1"), "p")
        self.assertIsNotNone(item.created_at)
        self.assertIsNotNone(item.updated_at)

    def test_created_at_is_stable_on_update(self) -> None:
        """Updating a value should preserve the original created_at."""
        store = self._store()
        store.put(("profile", "u"), "k", {"v": 1})
        original_created = store.get(("profile", "u"), "k").created_at

        store.put(("profile", "u"), "k", {"v": 2})
        updated_created = store.get(("profile", "u"), "k").created_at

        self.assertEqual(original_created, updated_created)

    def test_creates_parent_directory_if_missing(self) -> None:
        """SqliteStore should create missing parent directories automatically."""
        nested = str(Path(self.tmp) / "a" / "b" / "store.sqlite")
        store = SqliteStore(nested)
        store.put(("x",), "k", {"v": 1})

        self.assertTrue(Path(nested).exists())


class BuildStoreTests(unittest.TestCase):
    """Verify the build_store factory function."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_memory_sentinel_returns_in_memory_store(self) -> None:
        """':memory:' should return an InMemoryStore, not a SqliteStore."""
        from langgraph.store.memory import InMemoryStore

        store = build_store(":memory:")
        self.assertIsInstance(store, InMemoryStore)

    def test_file_path_returns_sqlite_store(self) -> None:
        """A file path should return a SqliteStore."""
        path = str(Path(self.tmp) / "store.sqlite")
        store = build_store(path)
        self.assertIsInstance(store, SqliteStore)


if __name__ == "__main__":
    unittest.main()
