"""Tests for the module 2 CLI surface."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import app.module2.main as module2_main
from app.config.settings import Settings
from app.module2.main import build_parser, run_interactive
from app.module2.schemas import (
    DEFAULT_MEMORY_DB,
    DEFAULT_PROMPT,
    DEFAULT_SUMMARIZE_AFTER,
    DEFAULT_THREAD_ID,
)


class Module2CliTests(unittest.TestCase):
    """Verify module 2 command-line behavior."""

    def test_parser_uses_module_defaults(self) -> None:
        """The CLI should preserve the module 2 defaults."""
        args = build_parser().parse_args([])

        self.assertEqual(args.prompt, DEFAULT_PROMPT)
        self.assertEqual(args.thread_id, DEFAULT_THREAD_ID)
        self.assertEqual(args.summarize_after, DEFAULT_SUMMARIZE_AFTER)
        self.assertIsNone(args.memory_db)
        self.assertFalse(args.interactive)
        self.assertIsNone(args.model)
        self.assertIsNone(args.model_provider)

    def test_parser_accepts_model_and_memory_configuration(self) -> None:
        """Users should be able to override model and SQLite settings."""
        args = build_parser().parse_args(
            [
                "--model",
                "claude-3-5-haiku-latest",
                "--model-provider",
                "anthropic",
                "--memory-db",
                ".tmp/module2.sqlite",
                "--summarize-after",
                "3",
            ]
        )

        self.assertEqual(args.model, "claude-3-5-haiku-latest")
        self.assertEqual(args.model_provider, "anthropic")
        self.assertEqual(args.memory_db, ".tmp/module2.sqlite")
        self.assertEqual(args.summarize_after, 3)

    def test_parser_rejects_invalid_summarize_after(self) -> None:
        """Summarization threshold should be a positive integer."""
        with (
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as error,
        ):
            build_parser().parse_args(["--summarize-after", "0"])

        self.assertEqual(error.exception.code, 2)

    def test_main_passes_configuration_to_graph_builder(self) -> None:
        """The CLI should pass provider and SQLite config into the graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "module2.sqlite"
            argv = [
                "main.py",
                "--model",
                "claude-3-5-haiku-latest",
                "--model-provider",
                "anthropic",
                "--prompt",
                "Remember this.",
                "--thread-id",
                "thread-id",
                "--memory-db",
                str(memory_db),
                "--summarize-after",
                "3",
                "--show-summary",
            ]
            memory_context = MagicMock()
            memory_context.__enter__.return_value = "memory"
            memory_context.__exit__.return_value = None

            with (
                patch.object(sys, "argv", argv),
                patch(
                    "app.module2.main.SqliteSaver.from_conn_string",
                    return_value=memory_context,
                ) as from_conn_string,
                patch(
                    "app.module2.main.build_graph",
                    return_value="graph",
                ) as build_graph,
                patch(
                    "app.module2.main.run_turn",
                    return_value="answer",
                ) as run_turn,
                patch(
                    "app.module2.main.get_summary",
                    return_value="summary",
                ) as get_summary,
                patch(
                    "app.module2.dependencies.get_settings",
                    return_value=Settings(
                        _env_file=None,
                        anthropic_api_key="test-key",
                        module2_memory_db=str(DEFAULT_MEMORY_DB),
                    ),
                ),
                patch("builtins.print") as print_output,
            ):
                exit_code = module2_main.main()

        self.assertEqual(exit_code, 0)
        from_conn_string.assert_called_once_with(str(memory_db))
        build_graph.assert_called_once_with(
            "memory",
            summarize_after=3,
            model="claude-3-5-haiku-latest",
            model_provider="anthropic",
            settings=ANY,
        )
        run_turn.assert_called_once_with("graph", "Remember this.", "thread-id")
        get_summary.assert_called_once_with("graph", "thread-id")
        print_output.assert_any_call("answer")
        print_output.assert_any_call("\nSummary:\nsummary")

    def test_main_errors_when_known_provider_key_is_missing(self) -> None:
        """Hosted providers should fail before graph construction without credentials."""
        argv = ["main.py", "--model-provider", "openai"]

        with (
            patch.object(sys, "argv", argv),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "app.module2.dependencies.get_settings",
                return_value=Settings(_env_file=None),
            ),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as error,
        ):
            module2_main.main()

        self.assertEqual(error.exception.code, 2)

    def test_main_uses_settings_memory_db_when_cli_value_is_absent(self) -> None:
        """The CLI should default SQLite storage from shared settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "from-settings.sqlite"
            argv = ["main.py", "--prompt", "Remember this."]
            memory_context = MagicMock()
            memory_context.__enter__.return_value = "memory"
            memory_context.__exit__.return_value = None

            with (
                patch.object(sys, "argv", argv),
                patch(
                    "app.module2.main.SqliteSaver.from_conn_string",
                    return_value=memory_context,
                ) as from_conn_string,
                patch("app.module2.main.build_graph", return_value="graph"),
                patch("app.module2.main.run_turn", return_value="answer"),
                patch(
                    "app.module2.dependencies.get_settings",
                    return_value=Settings(
                        _env_file=None,
                        openai_api_key="test-key",
                        module2_memory_db=str(memory_db),
                    ),
                ),
                patch("builtins.print"),
            ):
                exit_code = module2_main.main()

        self.assertEqual(exit_code, 0)
        from_conn_string.assert_called_once_with(str(memory_db))

    def test_run_interactive_prints_summary(self) -> None:
        """Interactive mode should expose the current graph summary."""
        prompts = iter(["/summary", "/quit"])

        with (
            patch("builtins.input", side_effect=lambda _: next(prompts)),
            patch(
                "app.module2.main.get_summary",
                return_value="stored summary",
            ) as get_summary,
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            exit_code = run_interactive(graph="graph", thread_id="thread-123")

        self.assertEqual(exit_code, 0)
        get_summary.assert_called_once_with("graph", "thread-123")
        self.assertIn("Summary: stored summary", output.getvalue())

    def test_run_interactive_sends_prompt_to_graph(self) -> None:
        """Interactive mode should pass non-command prompts to run_turn."""
        prompts = iter(["Hello", "/exit"])

        with (
            patch("builtins.input", side_effect=lambda _: next(prompts)),
            patch(
                "app.module2.main.run_turn",
                return_value="Hi",
            ) as run_turn,
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            exit_code = run_interactive(graph="graph", thread_id="thread-123")

        self.assertEqual(exit_code, 0)
        run_turn.assert_called_once_with("graph", "Hello", "thread-123")
        self.assertIn("Assistant: Hi", output.getvalue())


if __name__ == "__main__":
    unittest.main()
