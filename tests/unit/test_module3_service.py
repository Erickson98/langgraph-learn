"""Tests for the module 3 application service."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.config.settings import Settings
from app.module3.schemas import (
    MessageView,
    Module3ApproveRequest,
    Module3ForkRequest,
    Module3HistoryEntry,
    Module3HistoryRequest,
    Module3ReplayRequest,
    Module3StateRequest,
    Module3StateResult,
    Module3TurnRequest,
    Module3TurnResult,
    PendingToolCall,
)
from app.module3.services.module_service import Module3Service, Module3ServiceError


class Module3ServiceTests(unittest.IsolatedAsyncioTestCase):
    """Verify module 3 service behavior without FastAPI or live LLM calls."""

    def build_service(self, **overrides: str) -> Module3Service:
        """Build a service with isolated settings.

        Args:
            overrides: Settings values to override.

        Returns:
            Module 3 service instance.
        """
        defaults = {
            "langchain_chat_model": "gpt-4o-mini",
            "langchain_model_provider": "openai",
            "openai_api_key": "test-key",
            "anthropic_api_key": "",
            "module3_memory_db": "test-module3.sqlite",
        }
        settings = Settings(_env_file=None, **(defaults | overrides))
        return Module3Service(settings=settings)

    async def test_run_turn_returns_response_schema(self) -> None:
        """Service should run the graph and return the API response contract."""
        service = self.build_service()
        result = Module3TurnResult(
            status="paused",
            response="",
            pending_next=("tools",),
            pending_tool_calls=[
                PendingToolCall(
                    id="call-1",
                    name="multiply",
                    args={"a": 2, "b": 3},
                )
            ],
            message_count=2,
        )

        with patch(
            "app.module3.services.module_service.run_turn_with_sqlite_async",
            new_callable=AsyncMock,
            return_value=result,
        ) as run_turn_with_sqlite:
            response = await service.run_turn(
                Module3TurnRequest(prompt="Multiply 2 and 3.", thread_id="thread-id")
            )

        self.assertEqual(response.status, "paused")
        self.assertEqual(response.pending_next, ["tools"])
        self.assertEqual(response.pending_tool_calls[0].name, "multiply")
        run_turn_with_sqlite.assert_awaited_once_with(
            prompt="Multiply 2 and 3.",
            thread_id="thread-id",
            model="gpt-4o-mini",
            model_provider="openai",
            memory_db="test-module3.sqlite",
            settings=service.settings,
        )

    async def test_run_turn_raises_service_error_for_unknown_provider(self) -> None:
        """Service should raise a domain error for unsupported providers."""
        service = self.build_service()

        with self.assertRaises(Module3ServiceError) as error:
            await service.run_turn(Module3TurnRequest(model_provider="opneai"))

        self.assertEqual(error.exception.code, "unsupported_model_provider")

    async def test_run_turn_raises_service_error_for_missing_credentials(self) -> None:
        """Service should raise a domain error when provider credentials are absent."""
        service = self.build_service(openai_api_key="")

        with self.assertRaises(Module3ServiceError) as error:
            await service.run_turn(Module3TurnRequest(model_provider="openai"))

        self.assertEqual(error.exception.code, "missing_model_credentials")

    async def test_approve_turn_returns_response_schema(self) -> None:
        """Service should approve paused turns through the async graph helper."""
        service = self.build_service()

        with patch(
            "app.module3.services.module_service.approve_pending_turn_with_sqlite_async",
            new_callable=AsyncMock,
            return_value=Module3TurnResult(
                status="completed",
                response="6",
                pending_next=(),
                pending_tool_calls=[],
                message_count=4,
            ),
        ) as approve_turn:
            response = await service.approve_turn(
                Module3ApproveRequest(thread_id="thread-id")
            )

        self.assertEqual(response.status, "completed")
        self.assertEqual(response.response, "6")
        approve_turn.assert_awaited_once()

    async def test_get_state_returns_response_schema(self) -> None:
        """Service should read SQLite state through the async graph helper."""
        service = self.build_service()

        with patch(
            "app.module3.services.module_service.get_state_with_sqlite_async",
            new_callable=AsyncMock,
            return_value=Module3StateResult(
                status="idle",
                pending_next=(),
                pending_tool_calls=[],
                message_count=1,
                messages=[MessageView(id="m1", type="human", content="hello")],
            ),
        ) as get_state:
            response = await service.get_state(
                Module3StateRequest(thread_id="thread-id")
            )

        self.assertEqual(response.messages[0].content, "hello")
        get_state.assert_awaited_once_with(
            thread_id="thread-id",
            memory_db="test-module3.sqlite",
        )

    async def test_get_history_returns_response_schema(self) -> None:
        """Service should serialize checkpoint history entries."""
        service = self.build_service()

        with patch(
            "app.module3.services.module_service.list_history_with_sqlite_async",
            new_callable=AsyncMock,
            return_value=[
                Module3HistoryEntry(
                    checkpoint_id="checkpoint-1",
                    next_nodes=("assistant",),
                    source="loop",
                    step=1,
                    message_count=1,
                    can_replay=True,
                    can_fork=True,
                )
            ],
        ):
            response = await service.get_history(
                Module3HistoryRequest(thread_id="thread-id")
            )

        self.assertEqual(response.checkpoints[0].checkpoint_id, "checkpoint-1")
        self.assertTrue(response.checkpoints[0].can_fork)

    async def test_replay_checkpoint_maps_invalid_checkpoint(self) -> None:
        """Invalid checkpoint errors should become service errors."""
        service = self.build_service()

        with patch(
            "app.module3.services.module_service.replay_checkpoint_with_sqlite_async",
            new_callable=AsyncMock,
            side_effect=ValueError("bad checkpoint"),
        ):
            with self.assertRaises(Module3ServiceError) as error:
                await service.replay_checkpoint(
                    Module3ReplayRequest(
                        thread_id="thread-id",
                        checkpoint_id="checkpoint-1",
                    )
                )

        self.assertEqual(error.exception.code, "invalid_checkpoint")

    async def test_fork_checkpoint_maps_invalid_checkpoint(self) -> None:
        """Invalid fork errors should become service errors."""
        service = self.build_service()

        with patch(
            "app.module3.services.module_service.fork_checkpoint_with_sqlite_async",
            new_callable=AsyncMock,
            side_effect=ValueError("bad checkpoint"),
        ):
            with self.assertRaises(Module3ServiceError) as error:
                await service.fork_checkpoint(
                    Module3ForkRequest(
                        thread_id="thread-id",
                        checkpoint_id="checkpoint-1",
                        replacement_prompt="Multiply 5 and 3.",
                    )
                )

        self.assertEqual(error.exception.code, "invalid_checkpoint")


if __name__ == "__main__":
    unittest.main()
