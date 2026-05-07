"""Tests for module 4 retrieval helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.config.settings import Settings
from app.module4.services.retrieval_service import search_web, search_wikipedia


class Module4RetrievalServiceTests(unittest.TestCase):
    """Verify retrieval helpers without external network calls."""

    def test_search_wikipedia_returns_disabled_context(self) -> None:
        """Disabled Wikipedia retrieval should return a clear context block."""
        result = search_wikipedia("LangGraph", enabled=False)

        self.assertEqual(
            result.context_blocks, ["Wikipedia search disabled for this request."]
        )
        self.assertEqual(result.source_items, [])

    def test_search_wikipedia_formats_documents(self) -> None:
        """Wikipedia documents should become labeled context and sources."""
        loader = MagicMock()
        loader.load.return_value = [
            Document(
                page_content="LangGraph is a graph framework.",
                metadata={"source": "https://example.com/wiki"},
            )
        ]

        with patch(
            "app.module4.services.retrieval_service.WikipediaLoader",
            return_value=loader,
        ):
            result = search_wikipedia("LangGraph")

        self.assertIn("[W1] Source: https://example.com/wiki", result.context_blocks[0])
        self.assertEqual(result.source_items, ["[W1] https://example.com/wiki"])

    def test_search_wikipedia_accepts_label_prefix(self) -> None:
        """Graph callers should be able to make source labels section-scoped."""
        loader = MagicMock()
        loader.load.return_value = [
            Document(
                page_content="LangGraph is a graph framework.",
                metadata={"source": "https://example.com/wiki"},
            )
        ]

        with patch(
            "app.module4.services.retrieval_service.WikipediaLoader",
            return_value=loader,
        ):
            result = search_wikipedia("LangGraph", label_prefix="S2-W")

        self.assertIn(
            "[S2-W1] Source: https://example.com/wiki", result.context_blocks[0]
        )
        self.assertEqual(result.source_items, ["[S2-W1] https://example.com/wiki"])

    def test_search_wikipedia_sanitizes_loader_errors(self) -> None:
        """Wikipedia failures should not expose raw exception details to the LLM."""
        loader = MagicMock()
        loader.load.side_effect = RuntimeError("secret-token")

        with (
            patch(
                "app.module4.services.retrieval_service.WikipediaLoader",
                return_value=loader,
            ),
            self.assertLogs(
                "app.module4.services.retrieval_service",
                level="WARNING",
            ) as logs,
        ):
            result = search_wikipedia("LangGraph")

        self.assertEqual(
            result.context_blocks,
            ["Wikipedia lookup failed; continuing without that source."],
        )
        self.assertNotIn("secret-token", result.context_blocks[0])
        self.assertIn("Wikipedia lookup failed", "\n".join(logs.output))

    def test_search_web_skips_without_tavily_key(self) -> None:
        """Missing Tavily credentials should skip web retrieval."""
        result = search_web(
            "LangGraph production",
            settings=Settings(_env_file=None, tavily_api_key=""),
        )

        self.assertEqual(result.source_items, [])
        self.assertEqual(
            result.context_blocks, ["TAVILY_API_KEY is not set; skipping web search."]
        )

    def test_search_web_formats_tavily_results(self) -> None:
        """Tavily results should become labeled context and sources."""
        tavily = MagicMock()
        tavily.invoke.return_value = {
            "results": [
                {
                    "url": "https://example.com",
                    "content": "Useful research context.",
                }
            ]
        }

        with (
            patch(
                "app.module4.services.retrieval_service.TavilySearch",
                return_value=tavily,
            ) as tavily_search,
        ):
            result = search_web(
                "LangGraph production",
                settings=Settings(_env_file=None, tavily_api_key="test-key"),
            )

        tavily_search.assert_called_once()
        self.assertIn("[T1] Source: https://example.com", result.context_blocks[0])
        self.assertEqual(result.source_items, ["[T1] https://example.com"])

    def test_search_web_accepts_label_prefix(self) -> None:
        """Web source labels should also support section-scoped prefixes."""
        tavily = MagicMock()
        tavily.invoke.return_value = {
            "results": [
                {
                    "url": "https://example.com",
                    "content": "Useful research context.",
                }
            ]
        }

        with (
            patch(
                "app.module4.services.retrieval_service.TavilySearch",
                return_value=tavily,
            ),
        ):
            result = search_web(
                "LangGraph production",
                settings=Settings(_env_file=None, tavily_api_key="test-key"),
                label_prefix="S2-T",
            )

        self.assertIn("[S2-T1] Source: https://example.com", result.context_blocks[0])
        self.assertEqual(result.source_items, ["[S2-T1] https://example.com"])

    def test_search_web_sanitizes_tavily_errors(self) -> None:
        """Tavily failures should not expose raw exception details to the LLM."""
        tavily = MagicMock()
        tavily.invoke.side_effect = RuntimeError("secret-token")

        with (
            patch(
                "app.module4.services.retrieval_service.TavilySearch",
                return_value=tavily,
            ),
            self.assertLogs(
                "app.module4.services.retrieval_service",
                level="WARNING",
            ) as logs,
        ):
            result = search_web(
                "LangGraph production",
                settings=Settings(_env_file=None, tavily_api_key="test-key"),
            )

        self.assertEqual(
            result.context_blocks,
            ["Web search failed; continuing without that source."],
        )
        self.assertNotIn("secret-token", result.context_blocks[0])
        self.assertIn("Web search failed", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
