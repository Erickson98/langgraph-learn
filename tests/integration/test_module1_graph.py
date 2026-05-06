"""Integration tests for the module 1 LangGraph wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langsmith.run_helpers import tracing_context

from app.module1.services.graph_service import build_graph, run_turn


class FakeChatModel:
    """Small chat model test double for graph integration tests."""

    def __init__(self) -> None:
        """Create a fake model with no bound tools."""
        self.bound_tool_count = 0

    def bind_tools(self, tools):
        """Record bound tools and return an invokable model object."""
        self.bound_tool_count = len(tools)
        return self

    def invoke(self, messages):
        """Return a deterministic assistant message for the graph."""
        return AIMessage(
            content=f"messages={len(messages)} tools={self.bound_tool_count}"
        )


class Module1GraphIntegrationTests(unittest.TestCase):
    """Verify module 1 graph wiring with a patched chat model."""

    def test_graph_runs_one_turn_without_external_llm_call(self) -> None:
        """The graph should bind tools and return the model response."""
        with patch(
            "app.module1.services.graph_service.get_chat_model",
            return_value=FakeChatModel(),
        ):
            graph = build_graph()

        with tracing_context(enabled=False):
            response = run_turn(graph, "What is 2 + 2?", "integration-thread")

        self.assertEqual(response, "messages=2 tools=8")

    def test_graph_uses_memory_for_same_thread(self) -> None:
        """The in-memory checkpointer should preserve prior thread messages."""
        with patch(
            "app.module1.services.graph_service.get_chat_model",
            return_value=FakeChatModel(),
        ):
            graph = build_graph()

        with tracing_context(enabled=False):
            first_response = run_turn(graph, "Remember 2.", "memory-thread")
            second_response = run_turn(graph, "Now add 2.", "memory-thread")

        self.assertEqual(first_response, "messages=2 tools=8")
        self.assertEqual(second_response, "messages=4 tools=8")


if __name__ == "__main__":
    unittest.main()
