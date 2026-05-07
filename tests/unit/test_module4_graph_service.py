"""Tests for module 4 graph service helpers."""

from __future__ import annotations

import unittest

from app.module4.services.graph_service import (
    build_initial_state,
    compile_report,
    get_sources,
    is_expected_execution_error,
    normalize_planned_sections,
)
from app.module4.schemas import MAX_SECTIONS_LIMIT, PlannedSectionModel


class Module4GraphServiceTests(unittest.TestCase):
    """Verify pure module 4 graph helper behavior."""

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
        self.assertEqual(state["audience"], "leaders")
        self.assertEqual(state["max_sections"], 1)
        self.assertFalse(state["include_wikipedia"])
        self.assertFalse(state["include_web"])
        self.assertEqual(state["planned_sections"], [])
        self.assertEqual(state["completed_sections"], [])

    def test_build_initial_state_clamps_section_count(self) -> None:
        """Direct graph callers should still stay within supported fan-out limits."""
        self.assertEqual(
            build_initial_state(topic="Topic", audience="Audience", max_sections=0)[
                "max_sections"
            ],
            1,
        )
        self.assertEqual(
            build_initial_state(topic="Topic", audience="Audience", max_sections=99)[
                "max_sections"
            ],
            MAX_SECTIONS_LIMIT,
        )

    def test_normalize_planned_sections_returns_exact_count(self) -> None:
        """Planner output should be deterministic even when model output is short."""
        sections = normalize_planned_sections(
            [
                PlannedSectionModel(
                    title="",
                    focus="",
                    guiding_question="",
                )
            ],
            max_sections=2,
        )

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["order"], 1)
        self.assertEqual(sections[0]["title"], "Current State")
        self.assertEqual(sections[1]["title"], "Decision Drivers")

    def test_fallback_templates_cover_section_limit(self) -> None:
        """Fallback planning should support the full public section limit."""
        sections = normalize_planned_sections([], max_sections=MAX_SECTIONS_LIMIT)

        self.assertEqual(len(sections), MAX_SECTIONS_LIMIT)
        self.assertEqual(sections[-1]["title"], "Recommendation")

    def test_expected_execution_error_detection_checks_exception_chain(self) -> None:
        """External provider errors should be detected without masking code bugs."""

        class ProviderError(Exception):
            pass

        ProviderError.__module__ = "openai._exceptions"

        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = ProviderError("provider")
        self.assertTrue(is_expected_execution_error(wrapper))
        self.assertFalse(is_expected_execution_error(RuntimeError("bug")))

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
        self.assertIn("- [W1] wiki", report)
        self.assertIn("- [T1] web", report)
        self.assertEqual(
            get_sources(state["completed_sections"]),
            ["[W1] wiki", "[T1] web"],
        )


if __name__ == "__main__":
    unittest.main()
