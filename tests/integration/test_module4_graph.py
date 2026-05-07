"""Integration tests for the module 4 LangGraph wiring."""

from __future__ import annotations

import unittest
from unittest.mock import ANY, patch

from langchain_core.messages import AIMessage, BaseMessage
from langsmith.run_helpers import tracing_context

from app.config.settings import Settings
from app.module4.schemas import (
    PlannedSectionModel,
    PlannedSectionsModel,
    RetrievalContext,
    RetrievalQueries,
)
from app.module4.services.graph_service import (
    build_initial_state,
    compile_report,
    get_sources,
    run_brief,
)


class FakeStructuredModel:
    """Structured-output fake for module 4 graph integration tests."""

    def __init__(self, schema: type) -> None:
        """Store the requested schema.

        Args:
            schema: Structured output schema.
        """
        self.schema = schema

    def invoke(self, messages: list[BaseMessage]) -> object:
        """Return deterministic structured responses."""
        if self.schema is PlannedSectionsModel:
            return PlannedSectionsModel(
                sections=[
                    PlannedSectionModel(
                        title="Adoption Path",
                        focus="How teams can adopt the workflow.",
                        guiding_question="What should leaders do first?",
                    )
                ]
            )

        if self.schema is RetrievalQueries:
            return RetrievalQueries(
                web_query="LangGraph production adoption",
                wiki_query="LangGraph",
            )

        raise AssertionError(f"Unexpected schema: {self.schema}")


class FakeResearchChatModel:
    """Small chat model test double for module 4 graph integration tests."""

    def with_structured_output(self, schema: type) -> FakeStructuredModel:
        """Return a fake structured-output model for the requested schema."""
        return FakeStructuredModel(schema)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return deterministic section and overview text."""
        prompt = messages[0].content
        if "synthesizing a finished research brief" in prompt:
            return AIMessage(content="- Start with a bounded pilot.")

        return AIMessage(
            content=(
                "### Adoption Path\n"
                "Start with a bounded pilot and measure operational risk [S1-W1]."
            )
        )


class Module4GraphIntegrationTests(unittest.TestCase):
    """Verify module 4 graph wiring with patched LLM and retrieval."""

    def test_build_initial_state_uses_request_values(self) -> None:
        """Initial graph state should preserve request configuration."""
        state = build_initial_state(
            topic="LangGraph",
            audience="leaders",
            max_sections=1,
            include_wikipedia=False,
            include_web=False,
        )

        self.assertEqual(state["topic"], "LangGraph")
        self.assertFalse(state["include_wikipedia"])
        self.assertFalse(state["include_web"])

    def test_compile_report_orders_sections_and_sources(self) -> None:
        """Report compilation should order sections and de-duplicate sources."""
        state = build_initial_state(topic="Topic", audience="Audience")
        state["overview"] = "- Summary"
        state["completed_sections"] = [
            {
                "order": 2,
                "key": "02-b",
                "title": "B",
                "markdown": "### B",
                "sources": ["[W1] wiki"],
            },
            {
                "order": 1,
                "key": "01-a",
                "title": "A",
                "markdown": "### A",
                "sources": ["[W1] wiki", "[T1] web"],
            },
        ]

        report = compile_report(state)["final_report"]

        self.assertLess(report.index("### A"), report.index("### B"))
        self.assertEqual(
            get_sources(state["completed_sections"]), ["[W1] wiki", "[T1] web"]
        )

    def test_graph_generates_brief_without_external_calls(self) -> None:
        """The graph should generate a report with patched LLM and retrieval."""
        with (
            patch(
                "app.module4.services.graph_service.get_chat_model",
                return_value=FakeResearchChatModel(),
            ),
            patch(
                "app.module4.services.graph_service.search_wikipedia",
                return_value=RetrievalContext(
                    context_blocks=["[S1-W1] Source: https://wiki\nContext."],
                    source_items=["[S1-W1] https://wiki"],
                ),
            ) as search_wikipedia,
            patch(
                "app.module4.services.graph_service.search_web",
                return_value=RetrievalContext(context_blocks=[], source_items=[]),
            ) as search_web,
            tracing_context(enabled=False),
        ):
            result = run_brief(
                topic="LangGraph",
                audience="leaders",
                max_sections=1,
                include_web=False,
                settings=Settings(_env_file=None, openai_api_key="test-key"),
            )

        self.assertIn("# Research Brief: LangGraph", result["final_report"])
        self.assertEqual(len(result["completed_sections"]), 1)
        self.assertIn("[S1-W1] https://wiki", result["final_report"])
        search_wikipedia.assert_called_once_with(
            "LangGraph",
            enabled=True,
            label_prefix="S1-W",
        )
        search_web.assert_called_once_with(
            "LangGraph production adoption",
            settings=ANY,
            enabled=False,
            label_prefix="S1-T",
        )


if __name__ == "__main__":
    unittest.main()
