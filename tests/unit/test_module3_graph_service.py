"""Tests for module 3 graph service helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import StateSnapshot

from app.module3.services.graph_service import (
    build_config,
    build_history_entry,
    build_state_result,
    build_turn_result,
    extract_pending_tool_calls,
    get_memory_db_path,
    message_content_to_text,
)


class Module3GraphServiceTests(unittest.TestCase):
    """Verify pure module 3 graph helper behavior."""

    def test_build_config_uses_thread_id(self) -> None:
        """LangGraph config should place thread id under configurable."""
        self.assertEqual(
            build_config("thread-123"),
            {"configurable": {"thread_id": "thread-123"}},
        )

    def test_get_memory_db_path_creates_parent_directory(self) -> None:
        """SQLite path resolution should prepare the parent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db = Path(temp_dir) / "nested" / "module3.sqlite"

            resolved = get_memory_db_path(memory_db)
            parent_exists = memory_db.parent.exists()

        self.assertEqual(resolved, str(memory_db))
        self.assertTrue(parent_exists)

    def test_message_content_to_text_handles_structured_content(self) -> None:
        """Structured message content should become readable plain text."""
        content = [{"type": "text", "text": "hello"}, "world"]

        self.assertEqual(message_content_to_text(content), "hello\nworld")

    def test_extract_pending_tool_calls_reads_latest_message(self) -> None:
        """Tool call extraction should inspect the latest assistant message."""
        message = AIMessage(
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
        snapshot = StateSnapshot(
            values={"messages": [message]},
            next=("tools",),
            config=build_config("thread"),
            metadata={},
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )

        tool_calls = extract_pending_tool_calls(snapshot)

        self.assertEqual(tool_calls[0].name, "multiply")
        self.assertEqual(tool_calls[0].args, {"a": 2, "b": 3})

    def test_build_turn_result_marks_paused_state(self) -> None:
        """Snapshots with next nodes should produce paused turn results."""
        snapshot = StateSnapshot(
            values={"messages": [AIMessage(content="")]},
            next=("tools",),
            config=build_config("thread"),
            metadata={},
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )

        result = build_turn_result(snapshot)

        self.assertEqual(result.status, "paused")
        self.assertEqual(result.pending_next, ("tools",))

    def test_build_state_result_serializes_messages(self) -> None:
        """Thread state should expose serializable message views."""
        snapshot = StateSnapshot(
            values={"messages": [HumanMessage(content="hello")]},
            next=(),
            config=build_config("thread"),
            metadata={},
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )

        result = build_state_result(snapshot)

        self.assertEqual(result.status, "idle")
        self.assertEqual(result.messages[0].content, "hello")

    def test_build_history_entry_marks_forkable_assistant_checkpoint(self) -> None:
        """History entries should expose replay and fork capabilities."""
        snapshot = StateSnapshot(
            values={"messages": [HumanMessage(content="Multiply 2 and 3.")]},
            next=("assistant",),
            config={
                "configurable": {
                    "thread_id": "thread",
                    "checkpoint_id": "checkpoint-1",
                }
            },
            metadata={"source": "loop", "step": 1},
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )

        entry = build_history_entry(snapshot)

        self.assertTrue(entry.can_replay)
        self.assertTrue(entry.can_fork)
        self.assertEqual(entry.checkpoint_id, "checkpoint-1")


if __name__ == "__main__":
    unittest.main()
