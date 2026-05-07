"""Tests for the module 4 application service."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.config.settings import Settings
from app.module4.schemas import Module4BriefRequest, ResearchBriefState
from app.module4.services.graph_service import Module4GraphExecutionError
from app.module4.services.module_service import Module4Service, Module4ServiceError


class Module4ServiceTests(unittest.IsolatedAsyncioTestCase):
    """Verify module 4 service behavior without FastAPI or live LLM calls."""

    def build_service(self, **overrides: str) -> Module4Service:
        """Build a service with isolated settings.

        Args:
            overrides: Settings values to override.

        Returns:
            Module 4 service instance.
        """
        defaults = {
            "langchain_chat_model": "gpt-4o-mini",
            "langchain_model_provider": "openai",
            "openai_api_key": "test-key",
            "anthropic_api_key": "",
            "tavily_api_key": "",
        }
        settings = Settings(_env_file=None, **(defaults | overrides))
        return Module4Service(settings=settings)

    async def test_generate_brief_returns_response_schema(self) -> None:
        """Service should run the graph and return the API response contract."""
        service = self.build_service()
        state: ResearchBriefState = {
            "topic": "LangGraph",
            "audience": "leaders",
            "max_sections": 1,
            "include_wikipedia": True,
            "include_web": False,
            "planned_sections": [],
            "completed_sections": [
                {
                    "order": 1,
                    "key": "01-adoption",
                    "title": "Adoption",
                    "markdown": "### Adoption\nUse it.",
                    "sources": ["[W1] https://wiki"],
                }
            ],
            "overview": "- Start small.",
            "final_report": "# Research Brief: LangGraph",
        }

        with patch(
            "app.module4.services.module_service.run_brief_async",
            new_callable=AsyncMock,
            return_value=state,
        ) as run_brief:
            response = await service.generate_brief(
                Module4BriefRequest(
                    topic="LangGraph",
                    audience="leaders",
                    max_sections=1,
                    include_web=False,
                )
            )

        self.assertEqual(response.topic, "LangGraph")
        self.assertEqual(response.sections[0].title, "Adoption")
        self.assertEqual(response.sources, ["[W1] https://wiki"])
        run_brief.assert_awaited_once_with(
            topic="LangGraph",
            audience="leaders",
            max_sections=1,
            include_wikipedia=True,
            include_web=False,
            model="gpt-4o-mini",
            model_provider="openai",
            settings=service.settings,
        )

    async def test_generate_brief_raises_service_error_for_unknown_provider(
        self,
    ) -> None:
        """Service should raise a domain error for unsupported providers."""
        service = self.build_service()

        with self.assertRaises(Module4ServiceError) as error:
            await service.generate_brief(Module4BriefRequest(model_provider="opneai"))

        self.assertEqual(error.exception.code, "unsupported_model_provider")

    async def test_generate_brief_raises_service_error_for_missing_credentials(
        self,
    ) -> None:
        """Service should raise a domain error when provider credentials are absent."""
        service = self.build_service(openai_api_key="")

        with self.assertRaises(Module4ServiceError) as error:
            await service.generate_brief(Module4BriefRequest(model_provider="openai"))

        self.assertEqual(error.exception.code, "missing_model_credentials")

    async def test_generate_brief_wraps_graph_runtime_errors(self) -> None:
        """Provider or graph failures should become stable service errors."""
        service = self.build_service()

        with patch(
            "app.module4.services.module_service.run_brief_async",
            new_callable=AsyncMock,
            side_effect=Module4GraphExecutionError("provider failed"),
        ):
            with self.assertRaises(Module4ServiceError) as error:
                await service.generate_brief(Module4BriefRequest(topic="LangGraph"))

        self.assertEqual(error.exception.code, "brief_generation_failed")
        self.assertEqual(error.exception.message, "provider failed")

    async def test_generate_brief_does_not_wrap_unexpected_errors(self) -> None:
        """Programming errors should still surface as unexpected failures."""
        service = self.build_service()

        with patch(
            "app.module4.services.module_service.run_brief_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("bug"),
        ):
            with self.assertRaisesRegex(RuntimeError, "bug"):
                await service.generate_brief(Module4BriefRequest(topic="LangGraph"))


if __name__ == "__main__":
    unittest.main()
