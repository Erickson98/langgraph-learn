"""Opt-in live LLM integration test for module 1."""

from __future__ import annotations

import os
import unittest

from langsmith.run_helpers import tracing_context

from app.module1.dependencies import (
    get_chat_model_config,
    has_model_credentials,
    prepare_environment,
)
from app.module1.services.graph_service import build_graph, run_turn


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
class Module1LiveLlmIntegrationTests(unittest.TestCase):
    """Verify the real configured provider can execute a module 1 graph turn."""

    def test_live_llm_answers_arithmetic_prompt(self) -> None:
        """The configured live model should answer through the real graph."""
        config = get_chat_model_config()
        graph = build_graph(
            model=config.model,
            model_provider=config.model_provider,
        )

        with tracing_context(enabled=False):
            response = run_turn(
                graph,
                "What is 2 + 2? Answer with only the number.",
                "live-llm-test",
            )

        self.assertIn("4", response)
