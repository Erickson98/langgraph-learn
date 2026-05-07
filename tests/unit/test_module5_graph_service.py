"""Tests for module 5 graph service helpers."""

from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.store.memory import InMemoryStore

from app.module5.services.graph_service import (
    build_config,
    extract_tool_info,
    get_memory_snapshot,
    get_user_id,
    latest_tool_call,
    message_content_to_text,
    new_thread_id,
    route_after_assistant,
)


class Module5GraphServiceTests(unittest.IsolatedAsyncioTestCase):
    """Verify pure module 5 graph helper behavior."""

    def test_build_config_and_get_user_id_use_configurable_values(self) -> None:
        """Runnable config should carry thread and user identifiers."""
        config = build_config(user_id="user-1", thread_id="thread-1")

        self.assertEqual(config["configurable"]["thread_id"], "thread-1")
        self.assertEqual(get_user_id(config), "user-1")

    def test_new_thread_id_uses_prefix(self) -> None:
        """Generated thread ids should be readable in logs and debugging."""
        self.assertRegex(new_thread_id("memory"), r"^memory-[0-9a-f]{8}$")

    def test_message_content_to_text_handles_structured_content(self) -> None:
        """Structured message content should become readable plain text."""
        content = [{"type": "text", "text": "hello"}, {"content": "world"}, "done"]

        self.assertEqual(message_content_to_text(content), "hello\nworld\ndone")

    async def test_memory_snapshot_reads_store_values(self) -> None:
        """Memory snapshots should render profile, todos, and instructions."""
        store = InMemoryStore()
        store.put(("profile", "user-1"), "profile", {"name": "Ada"})
        store.put(("todo", "user-1"), "todo-1", {"task": "Ship module 5"})
        store.put(
            ("instructions", "user-1"),
            "user_instructions",
            {"memory": "Prioritize urgent tasks."},
        )

        snapshot = await get_memory_snapshot(store, "user-1")

        self.assertIn("Ada", snapshot.profile)
        self.assertIn("Ship module 5", snapshot.todos)
        self.assertEqual(snapshot.instructions, "Prioritize urgent tasks.")

    def test_extract_tool_info_summarizes_new_and_updated_todos(self) -> None:
        """Trustcall tool calls should become a useful tool response summary."""
        summary = extract_tool_info(
            [
                [
                    {"name": "ToDo", "args": {"task": "Write tests"}},
                    {
                        "name": "PatchDoc",
                        "args": {
                            "json_doc_id": "todo-1",
                            "planned_edits": "Add solution",
                            "patches": [{"value": "Use pytest"}],
                        },
                    },
                ]
            ],
            "ToDo",
        )

        self.assertIn("New ToDo created", summary)
        self.assertIn("Document todo-1 updated", summary)

    def test_route_after_assistant_ends_without_tool_call(self) -> None:
        """Assistant responses without tool calls should end the graph."""
        state = {"messages": [AIMessage(content="Done")]}

        self.assertEqual(route_after_assistant(state), END)

    def test_route_after_assistant_routes_supported_tool_calls(self) -> None:
        """Memory update tool calls should route to the matching update node."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "UpdateMemory",
                            "args": {"update_type": "todo"},
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

        self.assertEqual(route_after_assistant(state), "update_todos")
        self.assertEqual(latest_tool_call(state)["id"], "call-1")

    def test_latest_tool_call_requires_tool_call(self) -> None:
        """Missing tool calls should fail loudly in update nodes."""
        with self.assertRaisesRegex(ValueError, "Expected a tool call"):
            latest_tool_call({"messages": [AIMessage(content="Done")]})

    def test_route_after_assistant_raises_on_unknown_update_type(self) -> None:
        """Unknown update types should fail loudly rather than silently skip."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "UpdateMemory",
                            "args": {"update_type": "unknown"},
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
        with self.assertRaises(ValueError):
            route_after_assistant(state)


if __name__ == "__main__":
    unittest.main()
