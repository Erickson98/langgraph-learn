"""Tests for the module 5 CLI surface."""

from __future__ import annotations

import sys
import unittest
from io import StringIO
from unittest.mock import ANY, patch

import app.module5.main as module5_main
from app.config.settings import Settings
from app.module5.main import build_parser
from app.module5.schemas import DEFAULT_USER_ID


class Module5CliTests(unittest.TestCase):
    """Verify module 5 command-line behavior."""

    def test_parser_uses_module_defaults(self) -> None:
        """The CLI should preserve module 5 defaults."""
        args = build_parser().parse_args([])

        self.assertEqual(args.user_id, DEFAULT_USER_ID)
        self.assertIsNone(args.model)
        self.assertIsNone(args.model_provider)

    def test_parser_accepts_model_and_user_configuration(self) -> None:
        """Users should be able to override model provider and user id."""
        args = build_parser().parse_args(
            [
                "--user-id",
                "user-1",
                "--model",
                "claude-3-5-haiku-latest",
                "--model-provider",
                "anthropic",
            ]
        )

        self.assertEqual(args.user_id, "user-1")
        self.assertEqual(args.model, "claude-3-5-haiku-latest")
        self.assertEqual(args.model_provider, "anthropic")

    def test_main_resolves_model_config_and_runs_chat(self) -> None:
        """The CLI should pass resolved provider config into the chat runner."""
        argv = [
            "main.py",
            "--user-id",
            "user-1",
            "--model",
            "claude-3-5-haiku-latest",
            "--model-provider",
            "anthropic",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch(
                "app.module5.main.prepare_environment",
                return_value=Settings(_env_file=None, anthropic_api_key="test-key"),
            ),
            patch("app.module5.main.run_chat") as run_chat,
        ):
            exit_code = module5_main.main()

        self.assertEqual(exit_code, 0)
        run_chat.assert_called_once_with(
            user_id="user-1",
            model="claude-3-5-haiku-latest",
            model_provider="anthropic",
            settings=ANY,
        )

    def test_main_errors_when_known_provider_key_is_missing(self) -> None:
        """Hosted providers should fail before graph construction without credentials."""
        argv = ["main.py", "--model-provider", "openai"]

        with (
            patch.object(sys, "argv", argv),
            patch(
                "app.module5.main.prepare_environment",
                return_value=Settings(_env_file=None, openai_api_key=""),
            ),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as error,
        ):
            module5_main.main()

        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
