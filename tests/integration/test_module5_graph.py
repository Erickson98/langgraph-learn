"""Integration tests for the module 5 LangGraph wiring."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, BaseMessage
from langsmith.run_helpers import tracing_context

from app.config.settings import Settings
from app.module5.schemas import Profile, ToDo
from app.module5.services.graph_service import build_graph, run_turn


class FakeMemoryChatModel:
    """Small chat model test double for module 5 graph integration tests."""

    def bind_tools(
        self,
        tools: list[object],
        *,
        parallel_tool_calls: bool = False,
    ) -> "FakeMemoryChatModel":
        """Return this model as a tool-bound fake."""
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return a todo tool call first, then a final answer."""
        await asyncio.sleep(0)
        if any(message.type == "tool" for message in messages):
            return AIMessage(content="Todo updated.")

        return AIMessage(
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


class FakeProfileChatModel:
    """Test double that triggers the profile update route."""

    def bind_tools(
        self,
        tools: list[object],
        *,
        parallel_tool_calls: bool = False,
    ) -> "FakeProfileChatModel":
        """Return this model as a tool-bound fake."""
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return a user tool call first, then a final answer."""
        await asyncio.sleep(0)
        if any(message.type == "tool" for message in messages):
            return AIMessage(content="Profile updated.")

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "UpdateMemory",
                    "args": {"update_type": "user"},
                    "type": "tool_call",
                }
            ],
        )


class FakeInstructionsChatModel:
    """Test double that triggers the instructions update route."""

    def __init__(self) -> None:
        """Initialize call counter."""
        self._call_count = 0

    def bind_tools(
        self,
        tools: list[object],
        *,
        parallel_tool_calls: bool = False,
    ) -> "FakeInstructionsChatModel":
        """Return this model as a tool-bound fake."""
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Route by call order: instruct → update text → final answer."""
        await asyncio.sleep(0)
        self._call_count += 1
        if self._call_count == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "UpdateMemory",
                        "args": {"update_type": "instructions"},
                        "type": "tool_call",
                    }
                ],
            )
        if self._call_count == 2:
            # update_instructions node calls llm.ainvoke directly (no bind_tools)
            return AIMessage(content="Always prioritize urgent tasks.")
        return AIMessage(content="Instructions saved.")


class FakeExtractor:
    """Trustcall extractor test double for todo extraction."""

    def with_listeners(self, **_: object) -> "FakeExtractor":
        """Return this fake when listeners are attached."""
        return self

    async def ainvoke(self, _: dict[str, object]) -> dict[str, object]:
        """Return a deterministic todo extraction."""
        await asyncio.sleep(0)
        return {
            "responses": [ToDo(task="Ship module 5")],
            "response_metadata": [{"json_doc_id": "todo-1"}],
        }


class FakeProfileExtractor:
    """Trustcall extractor test double for profile extraction."""

    async def ainvoke(self, _: dict[str, object]) -> dict[str, object]:
        """Return a deterministic profile extraction."""
        await asyncio.sleep(0)
        return {
            "responses": [Profile(name="Ada", location="London")],
            "response_metadata": [{"json_doc_id": "profile-1"}],
        }


def _settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="test-key")


class Module5GraphIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Verify module 5 graph wiring with patched model and Trustcall."""

    async def test_graph_updates_todo_memory_without_external_calls(self) -> None:
        """The graph should update todo memory and return a final response."""
        with (
            patch(
                "app.module5.services.graph_service.get_chat_model",
                return_value=FakeMemoryChatModel(),
            ),
            patch(
                "app.module5.services.graph_service.create_extractor",
                return_value=FakeExtractor(),
            ),
            tracing_context(enabled=False),
        ):
            graph = build_graph(settings=_settings())
            result = await run_turn(
                graph,
                prompt="I need to ship module 5.",
                user_id="user-1",
                thread_id="thread-1",
            )

        self.assertEqual(result.response, "Todo updated.")
        self.assertIn("Ship module 5", result.memory.todos)
        self.assertEqual(result.thread_id, "thread-1")
        self.assertEqual(result.user_id, "user-1")

    async def test_graph_updates_profile_memory_via_user_route(self) -> None:
        """UpdateMemory with update_type='user' should store profile data."""
        with (
            patch(
                "app.module5.services.graph_service.get_chat_model",
                return_value=FakeProfileChatModel(),
            ),
            patch(
                "app.module5.services.graph_service.create_extractor",
                return_value=FakeProfileExtractor(),
            ),
            tracing_context(enabled=False),
        ):
            graph = build_graph(settings=_settings())
            result = await run_turn(
                graph,
                prompt="My name is Ada and I live in London.",
                user_id="user-profile",
                thread_id="thread-profile",
            )

        self.assertEqual(result.response, "Profile updated.")
        self.assertIn("Ada", result.memory.profile)

    async def test_graph_updates_instructions_via_instructions_route(self) -> None:
        """UpdateMemory with update_type='instructions' should store preferences."""
        model = FakeInstructionsChatModel()
        with (
            patch(
                "app.module5.services.graph_service.get_chat_model",
                return_value=model,
            ),
            tracing_context(enabled=False),
        ):
            graph = build_graph(settings=_settings())
            result = await run_turn(
                graph,
                prompt="Always show urgent tasks at the top.",
                user_id="user-instr",
                thread_id="thread-instr",
            )

        self.assertEqual(result.response, "Instructions saved.")
        self.assertIn("urgent", result.memory.instructions)


if __name__ == "__main__":
    unittest.main()
