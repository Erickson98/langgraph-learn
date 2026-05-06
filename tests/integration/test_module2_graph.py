"""Integration tests for the module 2 LangGraph wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith.run_helpers import tracing_context

from app.module2.services.graph_service import (
    CREATE_SUMMARY_PROMPT,
    build_graph,
    get_summary,
    run_turn,
    run_turn_with_sqlite,
)


class FakeChatModel:
    """Small chat model test double for graph integration tests."""

    def __init__(self) -> None:
        """Create a fake model with an invocation recorder."""
        self.calls: list[list[str]] = []

    def invoke(self, messages):
        """Return deterministic assistant and summary messages."""
        contents = [getattr(message, "content", "") for message in messages]
        self.calls.append(contents)

        if contents and (
            contents[-1] == CREATE_SUMMARY_PROMPT
            or contents[-1].startswith("Current summary:")
        ):
            return AIMessage(content=f"summary calls={len(self.calls)}")

        return AIMessage(content=f"messages={len(messages)}")


class Module2GraphIntegrationTests(unittest.TestCase):
    """Verify module 2 graph wiring with a patched chat model."""

    def test_graph_runs_one_turn_without_external_llm_call(self) -> None:
        """The graph should return the model response without live credentials."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "module2.sqlite"
            fake_model = FakeChatModel()

            with (
                patch(
                    "app.module2.services.graph_service.get_chat_model",
                    return_value=fake_model,
                ),
                SqliteSaver.from_conn_string(str(memory_db)) as memory,
            ):
                graph = build_graph(memory)

                with tracing_context(enabled=False):
                    response = run_turn(graph, "Remember this.", "integration-thread")
                    summary = get_summary(graph, "integration-thread")

        self.assertEqual(response, "messages=1")
        self.assertEqual(summary, "")

    def test_graph_summarizes_when_message_threshold_is_exceeded(self) -> None:
        """The graph should compact older messages into a summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "module2.sqlite"
            fake_model = FakeChatModel()

            with (
                patch(
                    "app.module2.services.graph_service.get_chat_model",
                    return_value=fake_model,
                ),
                SqliteSaver.from_conn_string(str(memory_db)) as memory,
            ):
                graph = build_graph(memory, summarize_after=2)

                with tracing_context(enabled=False):
                    run_turn(graph, "First message.", "summary-thread")
                    response = run_turn(graph, "Second message.", "summary-thread")
                    summary = get_summary(graph, "summary-thread")

        self.assertEqual(response, "messages=3")
        self.assertTrue(summary.startswith("summary calls="))
        self.assertTrue(
            any(call[-1] == CREATE_SUMMARY_PROMPT for call in fake_model.calls)
        )

    def test_run_turn_with_sqlite_persists_thread_state(self) -> None:
        """SQLite-backed service should preserve state across graph instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "module2.sqlite"
            fake_model = FakeChatModel()

            with patch(
                "app.module2.services.graph_service.get_chat_model",
                return_value=fake_model,
            ):
                with tracing_context(enabled=False):
                    first = run_turn_with_sqlite(
                        prompt="Remember this.",
                        thread_id="persistent-thread",
                        summarize_after=20,
                        memory_db=memory_db,
                    )
                    second = run_turn_with_sqlite(
                        prompt="What changed?",
                        thread_id="persistent-thread",
                        summarize_after=20,
                        memory_db=memory_db,
                    )

        self.assertEqual(first.response, "messages=1")
        self.assertEqual(second.response, "messages=3")
        self.assertEqual(second.summary, "")


if __name__ == "__main__":
    unittest.main()
