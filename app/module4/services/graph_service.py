"""LangGraph orchestration service for module 4 research briefs."""

from __future__ import annotations

from functools import partial
from typing import Any, Protocol

import anyio
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

try:
    from langgraph.types import Send
except ImportError:  # pragma: no cover
    from langgraph.constants import Send

from app.config.settings import Settings
from app.module4.dependencies import get_chat_model
from app.module4.schemas import (
    DEFAULT_MAX_SECTIONS,
    DEFAULT_MODEL,
    MAX_SECTIONS_LIMIT,
    PlannedSectionModel,
    CompletedSection,
    PlannedSectionsModel,
    ResearchBriefState,
    RetrievalQueries,
    SectionPlan,
    SectionSubgraphOutputState,
    SectionSubgraphState,
)
from app.module4.services.retrieval_service import search_web, search_wikipedia
from app.module4.services.text import dedupe, message_content_to_text, slugify

EXTERNAL_ERROR_MODULE_PREFIXES: tuple[str, ...] = (
    "anthropic.",
    "httpcore.",
    "httpx.",
    "langchain_anthropic.",
    "langchain_core.exceptions",
    "langchain_openai.",
    "openai.",
)

PLANNER_PROMPT = """You are planning a research brief.

Topic:
{topic}

Audience:
{audience}

Create exactly {max_sections} sections for a concise but strong research brief.

Rules:
1. Each section must cover a distinct angle.
2. Prefer sections that help the reader make decisions.
3. Avoid generic filler like "Introduction" or "Conclusion".
4. The section title should be short.
5. The focus should explain what the section is trying to deliver.
6. The guiding question should be answerable with research.
"""

QUERY_PROMPT = """You are preparing retrieval queries for one section of a research brief.

Overall topic: {topic}
Target audience: {audience}

Section title: {title}
Section focus: {focus}
Guiding question: {guiding_question}

Return:
- one web search query for current and practical sources
- one Wikipedia query for background context
"""

SECTION_PROMPT = """You are writing one section of a research brief.

Overall topic: {topic}
Target audience: {audience}

Section title: {title}
Section focus: {focus}
Guiding question: {guiding_question}

Use the context below.

Context:
{context}

Writing rules:
1. Write in markdown.
2. Start with exactly this heading: ### {title}
3. Answer the guiding question directly.
4. Prefer concrete claims and tradeoffs over broad summaries.
5. Use citations like [S1-W1] or [S2-T2] only when supported by the provided context.
6. If the evidence is thin or missing, say so clearly.
7. Keep this section to 2-5 short paragraphs plus optional bullets.
"""

OVERVIEW_PROMPT = """You are synthesizing a finished research brief into a short executive summary.

Topic: {topic}
Audience: {audience}

Section drafts:
{sections}

Write:
- 4 to 6 bullets
- each bullet should be decision-oriented
- no heading
- no preamble
"""

FALLBACK_SECTION_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    (
        "Current State",
        "Establish the most relevant current context for the topic.",
        "What context does the audience need before making a decision?",
    ),
    (
        "Decision Drivers",
        "Identify practical criteria and tradeoffs for the topic.",
        "Which factors should the audience weigh most heavily?",
    ),
    (
        "Risks And Constraints",
        "Surface limitations, risks, and operational constraints.",
        "What could make this approach fail or require extra care?",
    ),
    (
        "Adoption Path",
        "Outline concrete next steps and validation points.",
        "What should the audience do first and how should progress be measured?",
    ),
    (
        "Evidence Gaps",
        "Call out missing evidence and research questions.",
        "What should be validated before acting?",
    ),
    (
        "Recommendation",
        "Turn the research into a concise recommendation.",
        "What action is best supported by the available evidence?",
    ),
)


class StructuredModelLike(Protocol):
    """Minimal structured-output model protocol used by graph nodes."""

    def invoke(self, messages: list[BaseMessage]) -> Any:
        """Invoke the structured-output model.

        Args:
            messages: Prompt messages.

        Returns:
            Provider-specific structured model output.
        """
        ...


class ChatModelLike(Protocol):
    """Minimal chat model protocol used by the research brief graph."""

    def with_structured_output(self, schema: type[Any]) -> StructuredModelLike:
        """Return a structured-output model.

        Args:
            schema: Pydantic schema requested by the graph node.

        Returns:
            Structured-output model wrapper.
        """
        ...

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """Invoke the chat model.

        Args:
            messages: Prompt messages.

        Returns:
            Assistant message.
        """
        ...


class Module4GraphExecutionError(Exception):
    """Expected external failure while executing the research brief graph."""


def is_expected_execution_error(exc: BaseException) -> bool:
    """Return whether an exception looks like an external provider failure.

    Args:
        exc: Exception raised while executing the graph.

    Returns:
        True when the exception class comes from known network, LangChain, or
        provider packages.
    """
    current: BaseException | None = exc
    while current is not None:
        module = current.__class__.__module__
        if module.startswith(EXTERNAL_ERROR_MODULE_PREFIXES):
            return True
        current = current.__cause__ or current.__context__

    return False


def build_initial_state(
    *,
    topic: str,
    audience: str,
    max_sections: int = DEFAULT_MAX_SECTIONS,
    include_wikipedia: bool = True,
    include_web: bool = True,
) -> ResearchBriefState:
    """Build the initial LangGraph state for a research brief.

    Args:
        topic: Research topic.
        audience: Target audience.
        max_sections: Number of sections to plan.
        include_wikipedia: Whether to retrieve Wikipedia context.
        include_web: Whether to retrieve Tavily web context.

    Returns:
        Initial graph state.
    """
    return {
        "topic": topic,
        "audience": audience,
        "max_sections": normalize_section_count(max_sections),
        "include_wikipedia": include_wikipedia,
        "include_web": include_web,
        "planned_sections": [],
        "completed_sections": [],
        "overview": "",
        "final_report": "",
    }


def normalize_section_count(max_sections: int) -> int:
    """Clamp a requested section count to the graph's supported range.

    Args:
        max_sections: Requested section count.

    Returns:
        Bounded section count.
    """
    return min(max(max_sections, 1), MAX_SECTIONS_LIMIT)


def normalize_planned_sections(
    sections: list[PlannedSectionModel],
    *,
    max_sections: int,
) -> list[SectionPlan]:
    """Return exactly ``max_sections`` planned sections.

    The model is asked for an exact count, but provider behavior is not a
    contract. This keeps graph fan-out deterministic even when structured output
    is short, long, or contains blank fields.

    Args:
        sections: Planned sections returned by the structured model.
        max_sections: Requested section count.

    Returns:
        Normalized section plans.
    """
    target_count = normalize_section_count(max_sections)
    selected = list(sections[:target_count])
    while len(selected) < target_count:
        title, focus, guiding_question = FALLBACK_SECTION_TEMPLATES[len(selected)]
        selected.append(
            PlannedSectionModel(
                title=title,
                focus=focus,
                guiding_question=guiding_question,
            )
        )

    planned: list[SectionPlan] = []
    for index, section in enumerate(selected, start=1):
        fallback_title, fallback_focus, fallback_question = FALLBACK_SECTION_TEMPLATES[
            index - 1
        ]
        title = section.title.strip() or fallback_title
        focus = section.focus.strip() or fallback_focus
        guiding_question = section.guiding_question.strip() or fallback_question
        planned.append(
            {
                "order": index,
                "key": f"{index:02d}-{slugify(title)}",
                "title": title,
                "focus": focus,
                "guiding_question": guiding_question,
            }
        )

    return planned


def build_graph(
    *,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Build the research brief graph.

    Args:
        model: Chat model name used by graph nodes.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        Compiled LangGraph graph.
    """
    resolved_settings = settings or Settings()
    llm = get_chat_model(
        model=model,
        model_provider=model_provider,
        settings=resolved_settings,
    )
    section_subgraph = build_section_subgraph(
        llm=llm,
        settings=resolved_settings,
    )

    def plan_sections(state: ResearchBriefState) -> dict[str, list[SectionPlan]]:
        planner = llm.with_structured_output(PlannedSectionsModel)
        response = planner.invoke(
            [
                SystemMessage(
                    content=PLANNER_PROMPT.format(
                        topic=state["topic"],
                        audience=state["audience"],
                        max_sections=state["max_sections"],
                    )
                ),
                HumanMessage(content="Plan the sections."),
            ]
        )

        return {
            "planned_sections": normalize_planned_sections(
                response.sections,
                max_sections=state["max_sections"],
            )
        }

    def dispatch_sections(state: ResearchBriefState) -> list[Send]:
        return [
            Send(
                "build_section",
                {
                    "topic": state["topic"],
                    "audience": state["audience"],
                    "include_wikipedia": state["include_wikipedia"],
                    "include_web": state["include_web"],
                    "section": section,
                },
            )
            for section in state["planned_sections"]
        ]

    def write_overview(state: ResearchBriefState) -> dict[str, str]:
        ordered = sorted(state["completed_sections"], key=lambda item: item["order"])
        joined_sections = "\n\n".join(section["markdown"] for section in ordered)

        response = llm.invoke(
            [
                SystemMessage(
                    content=OVERVIEW_PROMPT.format(
                        topic=state["topic"],
                        audience=state["audience"],
                        sections=joined_sections,
                    )
                ),
                HumanMessage(content="Write the executive summary bullets."),
            ]
        )
        return {"overview": message_content_to_text(response.content).strip()}

    workflow = StateGraph(ResearchBriefState)
    workflow.add_node("plan_sections", plan_sections)
    workflow.add_node("build_section", section_subgraph)
    workflow.add_node("write_overview", write_overview)
    workflow.add_node("compile_report", compile_report)
    workflow.add_edge(START, "plan_sections")
    workflow.add_conditional_edges(
        "plan_sections", dispatch_sections, ["build_section"]
    )
    workflow.add_edge("build_section", "write_overview")
    workflow.add_edge("write_overview", "compile_report")
    workflow.add_edge("compile_report", END)
    return workflow.compile()


def build_section_subgraph(
    *,
    llm: ChatModelLike,
    settings: Settings,
) -> CompiledStateGraph:
    """Build the section-generation subgraph.

    Args:
        llm: Chat model used by the parent graph.
        settings: Runtime settings.

    Returns:
        Compiled section subgraph.
    """

    def plan_retrieval_queries(
        state: SectionSubgraphState,
    ) -> dict[str, str]:
        planner = llm.with_structured_output(RetrievalQueries)
        section = state["section"]
        response = planner.invoke(
            [
                SystemMessage(
                    content=QUERY_PROMPT.format(
                        topic=state["topic"],
                        audience=state["audience"],
                        title=section["title"],
                        focus=section["focus"],
                        guiding_question=section["guiding_question"],
                    )
                ),
                HumanMessage(content="Generate the retrieval queries."),
            ]
        )
        return {
            "web_query": response.web_query.strip(),
            "wiki_query": response.wiki_query.strip(),
        }

    def search_wikipedia_node(
        state: SectionSubgraphState,
    ) -> dict[str, list[str]]:
        section = state["section"]
        result = search_wikipedia(
            state["wiki_query"],
            enabled=state["include_wikipedia"],
            label_prefix=f"S{section['order']}-W",
        )
        return {
            "context_blocks": result.context_blocks,
            "source_items": result.source_items,
        }

    def search_web_node(state: SectionSubgraphState) -> dict[str, list[str]]:
        section = state["section"]
        result = search_web(
            state["web_query"],
            settings=settings,
            enabled=state["include_web"],
            label_prefix=f"S{section['order']}-T",
        )
        return {
            "context_blocks": result.context_blocks,
            "source_items": result.source_items,
        }

    def draft_section(
        state: SectionSubgraphState,
    ) -> dict[str, list[CompletedSection]]:
        section = state["section"]
        context = (
            "\n\n---\n\n".join(state.get("context_blocks", []))
            or "No context available."
        )
        sources = dedupe(state.get("source_items", []))
        response = llm.invoke(
            [
                SystemMessage(
                    content=SECTION_PROMPT.format(
                        topic=state["topic"],
                        audience=state["audience"],
                        title=section["title"],
                        focus=section["focus"],
                        guiding_question=section["guiding_question"],
                        context=context,
                    )
                ),
                HumanMessage(content="Draft this section."),
            ]
        )

        return {
            "completed_sections": [
                {
                    "order": section["order"],
                    "key": section["key"],
                    "title": section["title"],
                    "markdown": message_content_to_text(response.content).strip(),
                    "sources": sources,
                }
            ]
        }

    workflow = StateGraph(
        SectionSubgraphState, output_schema=SectionSubgraphOutputState
    )
    workflow.add_node("plan_retrieval_queries", plan_retrieval_queries)
    workflow.add_node("search_wikipedia", search_wikipedia_node)
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("draft_section", draft_section)
    workflow.add_edge(START, "plan_retrieval_queries")
    workflow.add_edge("plan_retrieval_queries", "search_wikipedia")
    workflow.add_edge("plan_retrieval_queries", "search_web")
    workflow.add_edge("search_wikipedia", "draft_section")
    workflow.add_edge("search_web", "draft_section")
    workflow.add_edge("draft_section", END)
    return workflow.compile()


def compile_report(state: ResearchBriefState) -> dict[str, str]:
    """Compile completed sections into a final markdown report.

    Args:
        state: Research brief state.

    Returns:
        Final report update.
    """
    ordered = sorted(state["completed_sections"], key=lambda item: item["order"])
    source_lines = get_sources(ordered)
    parts = [
        f"# Research Brief: {state['topic']}",
        "",
        f"_Audience: {state['audience']}_",
        "",
        "## Executive Summary",
        state["overview"].strip(),
        "",
    ]

    for section in ordered:
        parts.append(section["markdown"].strip())
        parts.append("")

    parts.append("## Sources")
    if source_lines:
        for item in source_lines:
            parts.append(f"- {item}")
    else:
        parts.append("- No external sources were captured.")

    return {"final_report": "\n".join(parts).strip()}


def get_sources(sections: list[CompletedSection]) -> list[str]:
    """Collect unique sources from completed sections.

    Args:
        sections: Completed report sections.

    Returns:
        De-duplicated source labels.
    """
    return dedupe(
        [source for section in sections for source in section.get("sources", [])]
    )


def run_brief(
    *,
    topic: str,
    audience: str,
    max_sections: int = DEFAULT_MAX_SECTIONS,
    include_wikipedia: bool = True,
    include_web: bool = True,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> ResearchBriefState:
    """Run the research brief graph to completion.

    Args:
        topic: Research topic.
        audience: Target audience.
        max_sections: Number of sections to plan.
        include_wikipedia: Whether to retrieve Wikipedia context.
        include_web: Whether to retrieve Tavily web context.
        model: Chat model name used by graph nodes.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        Final graph state.
    """
    try:
        graph = build_graph(
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
        return graph.invoke(
            build_initial_state(
                topic=topic,
                audience=audience,
                max_sections=max_sections,
                include_wikipedia=include_wikipedia,
                include_web=include_web,
            )
        )
    except Exception as exc:
        if is_expected_execution_error(exc):
            raise Module4GraphExecutionError(
                "Brief generation failed while calling the model or retrieval providers."
            ) from exc
        raise


async def run_brief_async(
    *,
    topic: str,
    audience: str,
    max_sections: int = DEFAULT_MAX_SECTIONS,
    include_wikipedia: bool = True,
    include_web: bool = True,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> ResearchBriefState:
    """Async wrapper for ``run_brief``.

    Args:
        topic: Research topic.
        audience: Target audience.
        max_sections: Number of sections to plan.
        include_wikipedia: Whether to retrieve Wikipedia context.
        include_web: Whether to retrieve Tavily web context.
        model: Chat model name used by graph nodes.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        Final graph state.
    """
    return await anyio.to_thread.run_sync(
        partial(
            run_brief,
            topic=topic,
            audience=audience,
            max_sections=max_sections,
            include_wikipedia=include_wikipedia,
            include_web=include_web,
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
    )
