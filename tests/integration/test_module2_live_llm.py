"""Opt-in live LLM integration test for module 2."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from langsmith.run_helpers import tracing_context

from app.module2.dependencies import (
    get_chat_model_config,
    has_model_credentials,
    prepare_environment,
)
from app.module2.services.graph_service import run_turn_with_sqlite


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
class Module2LiveLlmIntegrationTests(unittest.TestCase):
    """Verify the real configured provider can execute a module 2 graph turn."""

    def test_live_llm_answers_chat_prompt(self) -> None:
        """The configured live model should answer through the SQLite graph."""
        config = get_chat_model_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "module2-live.sqlite"

            with tracing_context(enabled=False):
                result = run_turn_with_sqlite(
                    prompt="Reply with only the word pong.",
                    thread_id="live-llm-test",
                    summarize_after=20,
                    model=config.model,
                    model_provider=config.model_provider,
                    memory_db=memory_db,
                )

        self.assertIn("pong", result.response.lower())


if __name__ == "__main__":
    unittest.main()
