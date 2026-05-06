"""Tests for the module 1 CLI surface."""

from __future__ import annotations

import os
import sys
import unittest
from io import StringIO
from unittest.mock import ANY, patch

import app.module1.main as module1_main
from app.config.settings import Settings
from app.module1.main import build_parser
from app.module1.schemas import DEFAULT_PROMPT, DEFAULT_THREAD_ID


class Module1CliTests(unittest.TestCase):
    """Verify module 1 command-line defaults."""

    def test_parser_uses_module_defaults(self) -> None:
        """The CLI should preserve the original default prompt and thread id."""
        args = build_parser().parse_args([])

        self.assertEqual(args.prompt, DEFAULT_PROMPT)
        self.assertEqual(args.thread_id, DEFAULT_THREAD_ID)
        self.assertFalse(args.interactive)
        self.assertIsNone(args.model)
        self.assertIsNone(args.model_provider)

    def test_parser_accepts_custom_prompt_and_thread(self) -> None:
        """Users should still be able to override prompt and thread id."""
        args = build_parser().parse_args(
            ["--prompt", "What is 2 + 2?", "--thread-id", "test-thread"]
        )

        self.assertEqual(args.prompt, "What is 2 + 2?")
        self.assertEqual(args.thread_id, "test-thread")

    def test_parser_accepts_model_configuration(self) -> None:
        """Users should be able to choose a LangChain model provider."""
        args = build_parser().parse_args(
            [
                "--model",
                "claude-3-5-haiku-latest",
                "--model-provider",
                "anthropic",
            ]
        )

        self.assertEqual(args.model, "claude-3-5-haiku-latest")
        self.assertEqual(args.model_provider, "anthropic")

    def test_main_passes_model_configuration_to_graph_builder(self) -> None:
        """The CLI should pass provider-agnostic model config into the graph."""
        argv = [
            "main.py",
            "--model",
            "claude-3-5-haiku-latest",
            "--model-provider",
            "anthropic",
            "--prompt",
            "What is 2 + 2?",
            "--thread-id",
            "thread-id",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch(
                "app.module1.main.build_graph",
                return_value="graph",
            ) as build_graph,
            patch(
                "app.module1.main.run_turn",
                return_value="4",
            ) as run_turn,
            patch(
                "app.module1.dependencies.get_settings",
                return_value=Settings(_env_file=None, anthropic_api_key="test-key"),
            ),
            patch("builtins.print") as print_output,
        ):
            exit_code = module1_main.main()

        self.assertEqual(exit_code, 0)
        build_graph.assert_called_once_with(
            model="claude-3-5-haiku-latest",
            model_provider="anthropic",
            settings=ANY,
        )
        run_turn.assert_called_once_with("graph", "What is 2 + 2?", "thread-id")
        print_output.assert_called_once_with("4")

    def test_main_errors_when_known_provider_key_is_missing(self) -> None:
        """Hosted providers should fail before graph construction without credentials."""
        argv = ["main.py", "--model-provider", "openai"]

        with (
            patch.object(sys, "argv", argv),
            patch.dict(
                os.environ,
                {},
                clear=True,
            ),
            patch(
                "app.module1.dependencies.get_settings",
                return_value=Settings(_env_file=None),
            ),
            patch(
                "sys.stderr",
                new_callable=StringIO,
            ),
            self.assertRaises(SystemExit) as error,
        ):
            module1_main.main()

        self.assertEqual(error.exception.code, 2)

    def test_main_errors_when_provider_is_unknown(self) -> None:
        """Unknown providers should fail before graph construction."""
        argv = ["main.py", "--model-provider", "opneai"]

        with (
            patch.object(sys, "argv", argv),
            patch.dict(
                os.environ,
                {},
                clear=True,
            ),
            patch(
                "app.module1.dependencies.get_settings",
                return_value=Settings(_env_file=None),
            ),
            patch(
                "sys.stderr",
                new_callable=StringIO,
            ),
            self.assertRaises(SystemExit) as error,
        ):
            module1_main.main()

        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
