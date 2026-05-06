"""Tests for module 1 graph service helpers."""

from __future__ import annotations

import unittest
from io import StringIO
from unittest.mock import patch

from app.module1.main import run_interactive
from app.module1.services.graph_service import build_config


class Module1GraphServiceTests(unittest.TestCase):
    """Verify graph helper behavior that does not require an LLM call."""

    def test_build_config_uses_thread_id(self) -> None:
        """LangGraph config should place thread id under configurable."""
        self.assertEqual(
            build_config("thread-123"),
            {"configurable": {"thread_id": "thread-123"}},
        )

    def test_run_interactive_exits_on_quit_command(self) -> None:
        """Interactive mode should exit cleanly when the user quits."""
        with (
            patch("builtins.input", return_value="quit"),
            patch(
                "sys.stdout",
                new_callable=StringIO,
            ) as output,
        ):
            exit_code = run_interactive(graph=object(), thread_id="thread-123")

        self.assertEqual(exit_code, 0)
        self.assertIn("Interactive mode. thread_id=thread-123", output.getvalue())

    def test_run_interactive_sends_prompt_to_graph(self) -> None:
        """Interactive mode should pass non-command prompts to run_turn."""
        prompts = iter(["What is 2 + 2?", "exit"])

        with (
            patch("builtins.input", side_effect=lambda _: next(prompts)),
            patch(
                "app.module1.main.run_turn",
                return_value="4",
            ) as run_turn,
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            exit_code = run_interactive(graph="graph", thread_id="thread-123")

        self.assertEqual(exit_code, 0)
        run_turn.assert_called_once_with("graph", "What is 2 + 2?", "thread-123")
        self.assertIn("Assistant: 4", output.getvalue())

    def test_run_interactive_exits_on_eof(self) -> None:
        """Interactive mode should handle EOF as a clean exit."""
        with (
            patch("builtins.input", side_effect=EOFError),
            patch(
                "sys.stdout",
                new_callable=StringIO,
            ),
        ):
            self.assertEqual(run_interactive(graph=object(), thread_id="thread-123"), 0)


if __name__ == "__main__":
    unittest.main()
