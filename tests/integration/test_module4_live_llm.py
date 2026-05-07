"""Opt-in live LLM integration test for module 4."""

from __future__ import annotations

import os
import unittest

from langsmith.run_helpers import tracing_context

from app.module4.dependencies import (
    get_chat_model_config,
    has_model_credentials,
    prepare_environment,
)
from app.module4.services.graph_service import run_brief


def live_llm_enabled() -> bool:
    """Return whether live provider tests should run."""
    settings = prepare_environment()
    config = get_chat_model_config(settings=settings)
    return os.environ.get("RUN_LIVE_LLM_TESTS") == "1" and has_model_credentials(
        config.model_provider, settings=settings
    )


@unittest.skipUnless(
    live_llm_enabled(),
    "Set RUN_LIVE_LLM_TESTS=1 and provider credentials to run live LLM tests.",
)
class Module4LiveLlmIntegrationTests(unittest.TestCase):
    """Verify the real configured provider can execute a module 4 graph."""

    def test_live_llm_generates_one_section_brief(self) -> None:
        """The configured live model should produce a minimal research brief."""
        config = get_chat_model_config()

        with tracing_context(enabled=False):
            result = run_brief(
                topic="LangGraph memory for support agents",
                audience="engineering leaders",
                max_sections=1,
                include_wikipedia=False,
                include_web=False,
                model=config.model,
                model_provider=config.model_provider,
            )

        self.assertIn("# Research Brief: LangGraph memory", result["final_report"])
        self.assertEqual(len(result["completed_sections"]), 1)
        self.assertTrue(result["overview"].strip())


if __name__ == "__main__":
    unittest.main()
