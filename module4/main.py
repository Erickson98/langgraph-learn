from __future__ import annotations

import argparse
import operator
import os
from pathlib import Path
import re
from typing import Annotated

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

try:
    from langchain_tavily import TavilySearch
except ImportError:
    TavilySearch = None

try:
    from langgraph.types import Send
except ImportError:
    from langgraph.constants import Send

def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

def require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")


def get_model() -> ChatOpenAI:
    require_openai_key()
    return ChatOpenAI(model="gpt-4o", temperature=0)


def clip(text: str, limit: int = 1400) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


class PlannedSectionModel(BaseModel):
    title: str = Field(description="Short, concrete section title.")
    focus: str = Field(description="What this section should explain.")
    guiding_question: str = Field(description="The key question this section answers.")


class PlannedSectionsModel(BaseModel):
    sections: list[PlannedSectionModel]


class RetrievalQueries(BaseModel):
    web_query: str = Field(description="Search-engine style query for fresh web sources.")
    wiki_query: str = Field(description="Wikipedia-style query for background context.")


class SectionPlan(TypedDict):
    order: int
    key: str
    title: str
    focus: str
    guiding_question: str


class CompletedSection(TypedDict):
    order: int
    key: str
    title: str
    markdown: str
    sources: list[str]


class ResearchBriefState(TypedDict):
    topic: str
    audience: str
    max_sections: int
    planned_sections: list[SectionPlan]
    completed_sections: Annotated[list[CompletedSection], operator.add]
    overview: str
    final_report: str


class SectionSubgraphState(TypedDict):
    topic: str
    audience: str
    section: SectionPlan
    web_query: str
    wiki_query: str
    context_blocks: Annotated[list[str], operator.add]
    source_items: Annotated[list[str], operator.add]
    completed_sections: list[CompletedSection]


class SectionSubgraphOutputState(TypedDict):
    completed_sections: list[CompletedSection]


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
5. Use citations like [W1], [T2] only when supported by the provided context.
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


def plan_sections(state: ResearchBriefState) -> dict:
    llm = get_model().with_structured_output(PlannedSectionsModel)

    response = llm.invoke(
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

    sections: list[SectionPlan] = []
    for idx, section in enumerate(response.sections, start=1):
        sections.append(
            {
                "order": idx,
                "key": f"{idx:02d}-{slugify(section.title)}",
                "title": section.title.strip(),
                "focus": section.focus.strip(),
                "guiding_question": section.guiding_question.strip(),
            }
        )

    return {"planned_sections": sections}


def dispatch_sections(state: ResearchBriefState):
    return [
        Send(
            "build_section",
            {
                "topic": state["topic"],
                "audience": state["audience"],
                "section": section,
            },
        )
        for section in state["planned_sections"]
    ]


def plan_retrieval_queries(state: SectionSubgraphState) -> dict:
    llm = get_model().with_structured_output(RetrievalQueries)
    section = state["section"]

    response = llm.invoke(
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


def search_wikipedia(state: SectionSubgraphState) -> dict:
    query = state["wiki_query"]
    blocks: list[str] = []
    sources: list[str] = []

    try:
        docs = WikipediaLoader(query=query, load_max_docs=2).load()
    except Exception as exc:
        return {
            "context_blocks": [f"Wikipedia lookup failed for '{query}': {exc}"],
            "source_items": [],
        }

    for idx, doc in enumerate(docs, start=1):
        label = f"W{idx}"
        source = doc.metadata.get("source", query)
        blocks.append(f"[{label}] Source: {source}\n{clip(doc.page_content)}")
        sources.append(f"[{label}] {source}")

    if not blocks:
        blocks.append(f"Wikipedia returned no useful results for '{query}'.")

    return {"context_blocks": blocks, "source_items": sources}


def search_web(state: SectionSubgraphState) -> dict:
    query = state["web_query"]
    blocks: list[str] = []
    sources: list[str] = []

    if TavilySearch is None:
        return {
            "context_blocks": ["Tavily package is not installed; skipping web search."],
            "source_items": [],
        }

    if not os.getenv("TAVILY_API_KEY"):
        return {
            "context_blocks": ["TAVILY_API_KEY is not set; skipping web search."],
            "source_items": [],
        }

    try:
        tavily = TavilySearch(max_results=3)
        data = tavily.invoke({"query": query})
        results = data.get("results", data)
    except Exception as exc:
        return {
            "context_blocks": [f"Web search failed for '{query}': {exc}"],
            "source_items": [],
        }

    for idx, item in enumerate(results, start=1):
        label = f"T{idx}"
        url = item.get("url", "unknown-url")
        content = clip(item.get("content", ""))
        if content:
            blocks.append(f"[{label}] Source: {url}\n{content}")
            sources.append(f"[{label}] {url}")

    if not blocks:
        blocks.append(f"Web search returned no useful results for '{query}'.")

    return {"context_blocks": blocks, "source_items": sources}


def draft_section(state: SectionSubgraphState) -> dict:
    llm = get_model()
    section = state["section"]
    context = "\n\n---\n\n".join(state.get("context_blocks", [])) or "No context available."
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
                "markdown": response.content.strip(),
                "sources": sources,
            }
        ]
    }


def write_overview(state: ResearchBriefState) -> dict:
    llm = get_model()
    ordered = sorted(state["completed_sections"], key=lambda x: x["order"])
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

    return {"overview": response.content.strip()}


def compile_report(state: ResearchBriefState) -> dict:
    ordered = sorted(state["completed_sections"], key=lambda x: x["order"])
    source_lines = dedupe(
        [source for section in ordered for source in section.get("sources", [])]
    )

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


def build_section_subgraph():
    builder = StateGraph(SectionSubgraphState, output_schema=SectionSubgraphOutputState)
    builder.add_node("plan_retrieval_queries", plan_retrieval_queries)
    builder.add_node("search_wikipedia", search_wikipedia)
    builder.add_node("search_web", search_web)
    builder.add_node("draft_section", draft_section)

    builder.add_edge(START, "plan_retrieval_queries")
    builder.add_edge("plan_retrieval_queries", "search_wikipedia")
    builder.add_edge("plan_retrieval_queries", "search_web")
    builder.add_edge("search_wikipedia", "draft_section")
    builder.add_edge("search_web", "draft_section")
    builder.add_edge("draft_section", END)

    return builder.compile()


def build_research_brief_graph():
    section_subgraph = build_section_subgraph()

    builder = StateGraph(ResearchBriefState)
    builder.add_node("plan_sections", plan_sections)
    builder.add_node("build_section", section_subgraph)
    builder.add_node("write_overview", write_overview)
    builder.add_node("compile_report", compile_report)

    builder.add_edge(START, "plan_sections")
    builder.add_conditional_edges("plan_sections", dispatch_sections, ["build_section"])
    builder.add_edge("build_section", "write_overview")
    builder.add_edge("write_overview", "compile_report")
    builder.add_edge("compile_report", END)

    return builder.compile()


def run(topic: str, audience: str, max_sections: int) -> str:
    app = build_research_brief_graph()
    result = app.invoke(
        {
            "topic": topic,
            "audience": audience,
            "max_sections": max_sections,
            "planned_sections": [],
            "completed_sections": [],
            "overview": "",
            "final_report": "",
        }
    )
    return result["final_report"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file Module 4 project.")
    parser.add_argument("topic", help="Research topic to analyze.")
    parser.add_argument(
        "--audience",
        default="engineering and product leadership",
        help="Who the brief is for.",
    )
    parser.add_argument(
        "--sections",
        type=int,
        default=3,
        help="How many sections to plan.",
    )
    parser.add_argument(
        "--output",
        help="Optional markdown output path.",
    )
    args = parser.parse_args()

    report = run(args.topic, args.audience, args.sections)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
