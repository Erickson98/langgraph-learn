"""Tests for the module 3 CLI surface."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import app.module3.main as module3_main
from app.config.settings import Settings
from app.module3.main import build_parser
from app.module3.schemas import (
    DEFAULT_DEMO,
    DEFAULT_MEMORY_DB,
    DEFAULT_PROMPT,
    DEFAULT_THREAD_ID,
    Module3TurnResult,
)


class Module3CliTests(unittest.TestCase):
    """Verify module 3 command-line behavior."""

    def test_parser_uses_module_defaults(self) -> None:
        """The CLI should preserve module 3 defaults."""
        args = build_parser().parse_args([])

        self.assertEqual(args.demo, DEFAULT_DEMO)
        self.assertEqual(args.prompt, DEFAULT_PROMPT)
        self.assertEqual(args.thread_id, DEFAULT_THREAD_ID)
        self.assertIsNone(args.memory_db)
        self.assertFalse(args.auto_approve)
        self.assertIsNone(args.model)
        self.assertIsNone(args.model_provider)

    def test_parser_accepts_model_and_memory_configuration(self) -> None:
        """Users should be able to override model and SQLite settings."""
        args = build_parser().parse_args(
            [
                "breakpoints",
                "--model",
                "claude-3-5-haiku-latest",
                "--model-provider",
                "anthropic",
                "--memory-db",
                ".tmp/module3.sqlite",
                "--auto-approve",
            ]
        )

        self.assertEqual(args.model, "claude-3-5-haiku-latest")
        self.assertEqual(args.model_provider, "anthropic")
        self.assertEqual(args.memory_db, ".tmp/module3.sqlite")
        self.assertTrue(args.auto_approve)

    def test_main_runs_dynamic_demo_without_credentials(self) -> None:
        """Dynamic breakpoints should not require a live model provider."""
        argv = ["main.py", "dynamic-breakpoints"]

        with (
            patch.object(sys, "argv", argv),
            patch("app.module3.main.demo_dynamic_breakpoints") as demo_dynamic,
            patch(
                "app.module3.dependencies.get_settings",
                return_value=Settings(_env_file=None),
            ),
        ):
            exit_code = module3_main.main()

        self.assertEqual(exit_code, 0)
        demo_dynamic.assert_called_once_with()

    def test_main_passes_configuration_to_breakpoint_graph(self) -> None:
        """The CLI should pass provider and SQLite config into the graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "module3.sqlite"
            argv = [
                "main.py",
                "breakpoints",
                "--model",
                "claude-3-5-haiku-latest",
                "--model-provider",
                "anthropic",
                "--prompt",
                "Multiply 2 and 3.",
                "--thread-id",
                "thread-id",
                "--memory-db",
                str(memory_db),
                "--auto-approve",
            ]
            memory_context = MagicMock()
            memory_context.__enter__.return_value = "memory"
            memory_context.__exit__.return_value = None

            with (
                patch.object(sys, "argv", argv),
                patch(
                    "app.module3.main.SqliteSaver.from_conn_string",
                    return_value=memory_context,
                ) as from_conn_string,
                patch(
                    "app.module3.main.build_breakpoint_graph",
                    return_value="graph",
                ) as build_graph,
                patch(
                    "app.module3.main.run_breakpoint_turn",
                    return_value=Module3TurnResult(
                        status="completed",
                        response="6",
                        pending_next=(),
                        pending_tool_calls=[],
                        message_count=2,
                    ),
                ) as run_turn,
                patch("app.module3.main.render_graph_if_requested"),
                patch(
                    "app.module3.dependencies.get_settings",
                    return_value=Settings(
                        _env_file=None,
                        anthropic_api_key="test-key",
                        module3_memory_db=str(DEFAULT_MEMORY_DB),
                    ),
                ),
                patch("builtins.print"),
            ):
                exit_code = module3_main.main()

        self.assertEqual(exit_code, 0)
        from_conn_string.assert_called_once_with(str(memory_db))
        build_graph.assert_called_once_with(
            "memory",
            interrupt_before=["tools"],
            model="claude-3-5-haiku-latest",
            model_provider="anthropic",
            settings=ANY,
        )
        run_turn.assert_called_once_with("graph", "Multiply 2 and 3.", "thread-id")

    def test_main_errors_when_known_provider_key_is_missing(self) -> None:
        """Hosted providers should fail before graph construction without credentials."""
        argv = ["main.py", "breakpoints", "--model-provider", "openai"]

        with (
            patch.object(sys, "argv", argv),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "app.module3.dependencies.get_settings",
                return_value=Settings(_env_file=None),
            ),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as error,
        ):
            module3_main.main()

        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
