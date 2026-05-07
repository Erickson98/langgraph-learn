"""Tests for the module 5 application service."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from langgraph.store.memory import InMemoryStore

from app.config.settings import Settings
from app.module5.schemas import (
    MemorySnapshot,
    Module5ChatRequest,
    Module5TurnResult,
)
from app.module5.services.module_service import Module5Service, Module5ServiceError


def _make_service(store: InMemoryStore | None = None) -> Module5Service:
    return Module5Service(
        settings=Settings(_env_file=None, openai_api_key="test-key"),
        store=store or InMemoryStore(),
    )


def _fake_turn_result(
    response: str = "Hello!",
    thread_id: str = "module5-abc123",
    user_id: str = "demo-user",
) -> Module5TurnResult:
    return Module5TurnResult(
        response=response,
        thread_id=thread_id,
        user_id=user_id,
        memory=MemorySnapshot(profile="None", todos="None", instructions="None"),
    )


class Module5ServiceTests(unittest.IsolatedAsyncioTestCase):
    """Verify module 5 service orchestration logic."""

    async def test_chat_creates_thread_id_when_not_provided(self) -> None:
        """A new thread id should be generated when the request omits one."""
        service = _make_service()

        with patch(
            "app.module5.services.module_service.run_turn_with_sqlite_async",
            new_callable=AsyncMock,
            return_value=_fake_turn_result(),
        ) as run_turn:
            result = await service.chat(Module5ChatRequest(prompt="Hello"))

        self.assertIsNotNone(result.thread_id)
        _, kwargs = run_turn.call_args
        self.assertIsNotNone(kwargs["thread_id"])

    async def test_chat_uses_provided_thread_id(self) -> None:
        """An explicit thread id from the request should be forwarded."""
        service = _make_service()

        with patch(
            "app.module5.services.module_service.run_turn_with_sqlite_async",
            new_callable=AsyncMock,
            return_value=_fake_turn_result(thread_id="my-thread", user_id="user-1"),
        ) as run_turn:
            result = await service.chat(
                Module5ChatRequest(
                    prompt="test", thread_id="my-thread", user_id="user-1"
                )
            )

        self.assertEqual(result.thread_id, "my-thread")
        _, kwargs = run_turn.call_args
        self.assertEqual(kwargs["thread_id"], "my-thread")

    async def test_chat_passes_memory_db_from_settings(self) -> None:
        """The service should forward module5_memory_db to the graph runner."""
        service = Module5Service(
            settings=Settings(
                _env_file=None,
                openai_api_key="test-key",
                module5_memory_db="/tmp/test.sqlite",
            ),
            store=InMemoryStore(),
        )

        with patch(
            "app.module5.services.module_service.run_turn_with_sqlite_async",
            new_callable=AsyncMock,
            return_value=_fake_turn_result(),
        ) as run_turn:
            await service.chat(Module5ChatRequest(prompt="test"))

        _, kwargs = run_turn.call_args
        self.assertEqual(kwargs["memory_db"], "/tmp/test.sqlite")

    async def test_chat_maps_turn_result_to_response_schema(self) -> None:
        """Service should convert Module5TurnResult to Module5ChatResponse fields."""
        service = _make_service()

        with patch(
            "app.module5.services.module_service.run_turn_with_sqlite_async",
            new_callable=AsyncMock,
            return_value=_fake_turn_result(response="Got it!", user_id="user-1"),
        ):
            result = await service.chat(
                Module5ChatRequest(prompt="test", user_id="user-1")
            )

        self.assertEqual(result.response, "Got it!")
        self.assertEqual(result.user_id, "user-1")
        self.assertEqual(result.memory.profile, "None")

    async def test_chat_raises_service_error_on_graph_failure(self) -> None:
        """Graph failures should be wrapped in Module5ServiceError."""
        service = _make_service()

        with (
            patch(
                "app.module5.services.module_service.run_turn_with_sqlite_async",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM timeout"),
            ),
            self.assertRaises(Module5ServiceError) as ctx,
        ):
            await service.chat(Module5ChatRequest(prompt="test"))

        self.assertEqual(ctx.exception.code, "chat_turn_failed")

    async def test_get_memory_returns_snapshot_from_store(self) -> None:
        """get_memory should read the live store attached to the service."""
        store = InMemoryStore()
        store.put(("profile", "user-1"), "profile", {"name": "Ada"})
        store.put(("todo", "user-1"), "todo-1", {"task": "Ship module 5"})
        store.put(
            ("instructions", "user-1"),
            "user_instructions",
            {"memory": "Prioritize urgent tasks."},
        )
        service = _make_service(store)

        result = await service.get_memory("user-1")

        self.assertIn("Ada", result.profile)
        self.assertIn("Ship module 5", result.todos)
        self.assertEqual(result.instructions, "Prioritize urgent tasks.")

    async def test_get_memory_returns_none_strings_for_unknown_user(self) -> None:
        """Empty store should return 'None' placeholders, not raise."""
        result = await _make_service().get_memory("unknown-user")

        self.assertEqual(result.profile, "None")
        self.assertEqual(result.todos, "None")
        self.assertEqual(result.instructions, "None")


if __name__ == "__main__":
    unittest.main()
