"""Tests for the module 4 CLI surface."""

from __future__ import annotations

import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, patch

import app.module4.main as module4_main
from app.config.settings import Settings
from app.module4.main import build_parser
from app.module4.schemas import (
    DEFAULT_AUDIENCE,
    DEFAULT_MAX_SECTIONS,
    DEFAULT_TOPIC,
    ResearchBriefState,
)
from app.module4.services.graph_service import Module4GraphExecutionError


class Module4CliTests(unittest.TestCase):
    """Verify module 4 command-line behavior."""

    def test_parser_uses_module_defaults(self) -> None:
        """The CLI should preserve module 4 defaults."""
        args = build_parser().parse_args([])

        self.assertEqual(args.topic, DEFAULT_TOPIC)
        self.assertEqual(args.audience, DEFAULT_AUDIENCE)
        self.assertEqual(args.sections, DEFAULT_MAX_SECTIONS)
        self.assertFalse(args.no_wikipedia)
        self.assertFalse(args.no_web)
        self.assertIsNone(args.output)
        self.assertIsNone(args.model)
        self.assertIsNone(args.model_provider)

    def test_parser_accepts_model_retrieval_and_output_configuration(self) -> None:
        """Users should be able to override model, retrieval, and output settings."""
        args = build_parser().parse_args(
            [
                "AI sourcing",
                "--audience",
                "operators",
                "--sections",
                "2",
                "--no-web",
                "--model",
                "claude-3-5-haiku-latest",
                "--model-provider",
                "anthropic",
                "--output",
                "brief.md",
            ]
        )

        self.assertEqual(args.topic, "AI sourcing")
        self.assertEqual(args.audience, "operators")
        self.assertEqual(args.sections, 2)
        self.assertTrue(args.no_web)
        self.assertEqual(args.model, "claude-3-5-haiku-latest")
        self.assertEqual(args.model_provider, "anthropic")
        self.assertEqual(args.output, "brief.md")

    def test_parser_rejects_out_of_range_section_count(self) -> None:
        """Section counts should stay within the supported graph fan-out limit."""
        with (
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as error,
        ):
            build_parser().parse_args(["--sections", "99"])

        self.assertEqual(error.exception.code, 2)

    def test_main_runs_brief_and_writes_output(self) -> None:
        """The CLI should pass provider config into the graph service."""
        state: ResearchBriefState = {
            "topic": "LangGraph",
            "audience": "leaders",
            "max_sections": 1,
            "include_wikipedia": False,
            "include_web": False,
            "planned_sections": [],
            "completed_sections": [],
            "overview": "- Start small.",
            "final_report": "# Research Brief: LangGraph",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "brief.md"
            argv = [
                "main.py",
                "LangGraph",
                "--audience",
                "leaders",
                "--sections",
                "1",
                "--no-wikipedia",
                "--no-web",
                "--model",
                "claude-3-5-haiku-latest",
                "--model-provider",
                "anthropic",
                "--output",
                str(output_path),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch(
                    "app.module4.main.prepare_environment",
                    return_value=Settings(_env_file=None, anthropic_api_key="test-key"),
                ),
                patch(
                    "app.module4.main.run_brief",
                    return_value=state,
                ) as run_brief,
                patch("builtins.print"),
            ):
                exit_code = module4_main.main()

            written = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(written, "# Research Brief: LangGraph")
        run_brief.assert_called_once_with(
            topic="LangGraph",
            audience="leaders",
            max_sections=1,
            include_wikipedia=False,
            include_web=False,
            model="claude-3-5-haiku-latest",
            model_provider="anthropic",
            settings=ANY,
        )

    def test_main_errors_when_known_provider_key_is_missing(self) -> None:
        """Hosted providers should fail before graph construction without credentials."""
        argv = ["main.py", "LangGraph", "--model-provider", "openai"]

        with (
            patch.object(sys, "argv", argv),
            patch(
                "app.module4.main.prepare_environment",
                return_value=Settings(_env_file=None, openai_api_key=""),
            ),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as error,
        ):
            module4_main.main()

        self.assertEqual(error.exception.code, 2)

    def test_main_wraps_graph_runtime_errors(self) -> None:
        """CLI runtime failures should exit without a Python traceback."""
        argv = ["main.py", "LangGraph", "--model-provider", "openai"]
        stderr = StringIO()

        with (
            patch.object(sys, "argv", argv),
            patch(
                "app.module4.main.prepare_environment",
                return_value=Settings(_env_file=None, openai_api_key="test-key"),
            ),
            patch(
                "app.module4.main.run_brief",
                side_effect=Module4GraphExecutionError("provider failed"),
            ),
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit) as error,
        ):
            module4_main.main()

        self.assertEqual(error.exception.code, 1)
        self.assertIn("brief generation failed", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_does_not_wrap_unexpected_errors(self) -> None:
        """Unexpected CLI errors should keep their original exception type."""
        argv = ["main.py", "LangGraph", "--model-provider", "openai"]

        with (
            patch.object(sys, "argv", argv),
            patch(
                "app.module4.main.prepare_environment",
                return_value=Settings(_env_file=None, openai_api_key="test-key"),
            ),
            patch(
                "app.module4.main.run_brief",
                side_effect=RuntimeError("bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "bug"),
        ):
            module4_main.main()


if __name__ == "__main__":
    unittest.main()
