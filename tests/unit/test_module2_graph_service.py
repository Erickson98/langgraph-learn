"""Tests for module 2 graph service helpers."""

from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from app.module2.services.graph_service import (
    build_config,
    build_graph,
    get_summary_with_sqlite,
    get_memory_db_path,
    maybe_render_graph,
)


class FakeDrawable:
    """Graph drawing test double."""

    def __init__(self) -> None:
        """Create an empty drawable recorder."""
        self.png_path: str | None = None

    def draw_mermaid(self) -> str:
        """Return deterministic Mermaid text."""
        return "graph TD"

    def draw_mermaid_png(self, output_file_path: str) -> None:
        """Record the requested PNG path.

        Args:
            output_file_path: Destination PNG path.
        """
        self.png_path = output_file_path


class FakeGraph:
    """Compiled graph drawing test double."""

    def __init__(self) -> None:
        """Create a graph wrapper with a drawable object."""
        self.drawable = FakeDrawable()

    def get_graph(self) -> FakeDrawable:
        """Return the drawable graph object."""
        return self.drawable


class Module2GraphServiceTests(unittest.TestCase):
    """Verify module 2 helpers that do not need a live LLM."""

    def test_build_config_uses_thread_id(self) -> None:
        """LangGraph config should place thread id under configurable."""
        self.assertEqual(
            build_config("thread-123"),
            {"configurable": {"thread_id": "thread-123"}},
        )

    def test_get_memory_db_path_creates_parent_directory(self) -> None:
        """SQLite path resolution should prepare the parent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "nested" / "module2.sqlite"

            resolved = get_memory_db_path(memory_db)
            self.assertEqual(resolved, str(memory_db))
            self.assertTrue(memory_db.parent.exists())

    def test_build_graph_rejects_invalid_summarization_threshold(self) -> None:
        """Summarization threshold should fail early when invalid."""
        with self.assertRaisesRegex(ValueError, "summarize_after"):
            build_graph(checkpointer=object(), summarize_after=0)

    def test_get_summary_with_sqlite_returns_empty_for_unknown_thread(self) -> None:
        """Summary reads should not require a live model or existing thread."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "module2.sqlite"

            summary = get_summary_with_sqlite(
                thread_id="unknown-thread",
                memory_db=memory_db,
            )

        self.assertEqual(summary, "")

    def test_maybe_render_graph_prints_and_saves_when_requested(self) -> None:
        """Graph rendering helper should delegate only when requested."""
        graph = FakeGraph()

        with patch("sys.stdout", new_callable=StringIO) as output:
            maybe_render_graph(
                graph,
                print_mermaid=True,
                save_mermaid_png="graph.png",
            )

        self.assertIn("graph TD", output.getvalue())
        self.assertEqual(graph.drawable.png_path, "graph.png")


if __name__ == "__main__":
    unittest.main()
