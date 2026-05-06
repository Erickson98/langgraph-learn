"""Tests for the module 2 application service."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.config.settings import Settings
from app.module2.schemas import (
    Module2SummaryRequest,
    Module2TurnRequest,
    Module2TurnResult,
)
from app.module2.services.module_service import Module2Service, Module2ServiceError


class Module2ServiceTests(unittest.IsolatedAsyncioTestCase):
    """Verify module 2 service behavior without FastAPI or live LLM calls."""

    def build_service(self, **overrides: str) -> Module2Service:
        """Build a service with isolated settings.

        Args:
            overrides: Settings values to override.

        Returns:
            Module 2 service instance.
        """
        defaults = {
            "langchain_chat_model": "gpt-4o-mini",
            "langchain_model_provider": "openai",
            "openai_api_key": "test-key",
            "anthropic_api_key": "",
            "module2_memory_db": "test-module2.sqlite",
        }
        settings = Settings(_env_file=None, **(defaults | overrides))
        return Module2Service(settings=settings)

    async def test_run_turn_returns_response_schema(self) -> None:
        """Service should run the graph and return the API response contract."""
        service = self.build_service()

        with patch(
            "app.module2.services.module_service.run_turn_with_sqlite_async",
            new_callable=AsyncMock,
            return_value=Module2TurnResult(response="answer", summary="summary"),
        ) as run_turn_with_sqlite:
            response = await service.run_turn(
                Module2TurnRequest(
                    prompt="Remember this.",
                    thread_id="thread-id",
                    summarize_after=3,
                )
            )

        self.assertEqual(response.response, "answer")
        self.assertEqual(response.summary, "summary")
        self.assertEqual(response.thread_id, "thread-id")
        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertEqual(response.model_provider, "openai")
        run_turn_with_sqlite.assert_awaited_once_with(
            prompt="Remember this.",
            thread_id="thread-id",
            summarize_after=3,
            model="gpt-4o-mini",
            model_provider="openai",
            memory_db="test-module2.sqlite",
            settings=service.settings,
        )

    async def test_run_turn_raises_service_error_for_unknown_provider(self) -> None:
        """Service should raise a domain error for unsupported providers."""
        service = self.build_service()

        with self.assertRaises(Module2ServiceError) as error:
            await service.run_turn(Module2TurnRequest(model_provider="opneai"))

        self.assertEqual(error.exception.code, "unsupported_model_provider")

    async def test_run_turn_raises_service_error_for_missing_credentials(self) -> None:
        """Service should raise a domain error when provider credentials are absent."""
        service = self.build_service(openai_api_key="")

        with self.assertRaises(Module2ServiceError) as error:
            await service.run_turn(Module2TurnRequest(model_provider="openai"))

        self.assertEqual(error.exception.code, "missing_model_credentials")

    async def test_get_summary_returns_response_schema(self) -> None:
        """Service should read SQLite summary through the async graph helper."""
        service = self.build_service()

        with patch(
            "app.module2.services.module_service.get_summary_with_sqlite_async",
            new_callable=AsyncMock,
            return_value="stored summary",
        ) as get_summary_with_sqlite:
            response = await service.get_summary(
                Module2SummaryRequest(thread_id="thread-id")
            )

        self.assertEqual(response.summary, "stored summary")
        self.assertEqual(response.thread_id, "thread-id")
        get_summary_with_sqlite.assert_awaited_once_with(
            thread_id="thread-id",
            memory_db="test-module2.sqlite",
        )


if __name__ == "__main__":
    unittest.main()
