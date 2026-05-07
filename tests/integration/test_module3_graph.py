"""Integration tests for the module 3 LangGraph wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langsmith.run_helpers import tracing_context

from app.module3.services.graph_service import (
    approve_pending_turn,
    approve_pending_turn_with_sqlite,
    build_breakpoint_graph,
    fork_checkpoint,
    get_state_with_sqlite,
    list_thread_history,
    replay_checkpoint,
    run_breakpoint_turn,
    run_turn_with_sqlite,
)


class FakeToolCallingChatModel:
    """Small chat model test double for module 3 graph integration tests."""

    def __init__(self) -> None:
        """Create a fake model with an invocation recorder."""
        self.bound_tool_count = 0
        self.calls: list[list[str]] = []

    def bind_tools(self, tools: list[object]) -> "FakeToolCallingChatModel":
        """Record bound tools and return the fake model.

        Args:
            tools: Tools bound to the model.

        Returns:
            This fake model.
        """
        self.bound_tool_count = len(tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return a deterministic tool call, then a final answer."""
        message_types = [message.type for message in messages]
        self.calls.append(message_types)

        if "tool" in message_types:
            return AIMessage(content=f"tool-count={self.bound_tool_count} answer=6")

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "multiply",
                    "args": {"a": 2, "b": 3},
                    "type": "tool_call",
                }
            ],
        )


class Module3GraphIntegrationTests(unittest.TestCase):
    """Verify module 3 graph wiring with a patched chat model."""

    def test_breakpoint_graph_pauses_and_approves_tool_call(self) -> None:
        """The graph should pause before tools and resume after approval."""
        fake_model = FakeToolCallingChatModel()

        with patch(
            "app.module3.services.graph_service.get_chat_model",
            return_value=fake_model,
        ):
            graph = build_breakpoint_graph(
                MemorySaver(),
                interrupt_before=["tools"],
            )

        with tracing_context(enabled=False):
            paused = run_breakpoint_turn(graph, "Multiply 2 and 3.", "thread-id")
            completed = approve_pending_turn(graph, "thread-id")

        self.assertEqual(paused.status, "paused")
        self.assertEqual(paused.pending_next, ("tools",))
        self.assertEqual(paused.pending_tool_calls[0].name, "multiply")
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.response, "tool-count=4 answer=6")

    def test_sqlite_helpers_persist_paused_thread_state(self) -> None:
        """SQLite-backed helpers should preserve state across graph instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "module3.sqlite"
            fake_model = FakeToolCallingChatModel()

            with patch(
                "app.module3.services.graph_service.get_chat_model",
                return_value=fake_model,
            ):
                with tracing_context(enabled=False):
                    paused = run_turn_with_sqlite(
                        prompt="Multiply 2 and 3.",
                        thread_id="sqlite-thread",
                        memory_db=memory_db,
                    )
                    state = get_state_with_sqlite(
                        thread_id="sqlite-thread",
                        memory_db=memory_db,
                    )
                    completed = approve_pending_turn_with_sqlite(
                        thread_id="sqlite-thread",
                        memory_db=memory_db,
                    )

        self.assertEqual(paused.status, "paused")
        self.assertEqual(state.status, "paused")
        self.assertEqual(completed.response, "tool-count=4 answer=6")

    def test_history_supports_replay_and_fork(self) -> None:
        """Checkpoint history should support replay and fork operations."""
        fake_model = FakeToolCallingChatModel()

        with patch(
            "app.module3.services.graph_service.get_chat_model",
            return_value=fake_model,
        ):
            graph = build_breakpoint_graph(
                MemorySaver(),
                interrupt_before=["tools"],
            )

        with tracing_context(enabled=False):
            run_breakpoint_turn(graph, "Multiply 2 and 3.", "history-thread")
            history = list_thread_history(graph, "history-thread")
            replayable = next(entry for entry in history if entry.can_replay)
            forkable = next(entry for entry in history if entry.can_fork)
            replayed = replay_checkpoint(
                graph,
                "history-thread",
                replayable.checkpoint_id,
            )
            forked = fork_checkpoint(
                graph,
                "history-thread",
                forkable.checkpoint_id,
                "Multiply 5 and 3.",
            )

        self.assertIn(replayed.status, {"paused", "completed"})
        self.assertEqual(forked.status, "paused")
        self.assertEqual(forked.pending_tool_calls[0].name, "multiply")


if __name__ == "__main__":
    unittest.main()
