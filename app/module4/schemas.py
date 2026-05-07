"""Shared constants and typed values for module 4."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Final

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

DEFAULT_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_MODEL_PROVIDER: Final[str] = "openai"
DEFAULT_TOPIC: Final[str] = "LangGraph for production AI agents"
DEFAULT_AUDIENCE: Final[str] = "engineering and product leadership"
DEFAULT_MAX_SECTIONS: Final[int] = 3
MAX_SECTIONS_LIMIT: Final[int] = 6


@dataclass(frozen=True)
class ChatModelConfig:
    """Resolved provider-specific model configuration."""

    model: str
    model_provider: str


@dataclass(frozen=True)
class RetrievalContext:
    """Retrieved context blocks and source labels for one section."""

    context_blocks: list[str]
    source_items: list[str]


class PlannedSectionModel(BaseModel):
    """Structured model output for one planned section."""

    title: str = Field(description="Short, concrete section title.")
    focus: str = Field(description="What this section should explain.")
    guiding_question: str = Field(description="The key question this section answers.")


class PlannedSectionsModel(BaseModel):
    """Structured model output for all planned sections."""

    sections: list[PlannedSectionModel]


class RetrievalQueries(BaseModel):
    """Structured model output for retrieval queries."""

    web_query: str = Field(
        description="Search-engine style query for fresh web sources."
    )
    wiki_query: str = Field(description="Wikipedia-style query for background context.")


class SectionPlan(TypedDict):
    """State representation for one planned section."""

    order: int
    key: str
    title: str
    focus: str
    guiding_question: str


class CompletedSection(TypedDict):
    """State representation for one completed report section."""

    order: int
    key: str
    title: str
    markdown: str
    sources: list[str]


class ResearchBriefState(TypedDict):
    """Top-level LangGraph state for the research brief workflow."""

    topic: str
    audience: str
    max_sections: int
    include_wikipedia: bool
    include_web: bool
    planned_sections: list[SectionPlan]
    completed_sections: Annotated[list[CompletedSection], operator.add]
    overview: str
    final_report: str


class SectionSubgraphState(TypedDict):
    """LangGraph state for one section-building subgraph."""

    topic: str
    audience: str
    include_wikipedia: bool
    include_web: bool
    section: SectionPlan
    web_query: str
    wiki_query: str
    context_blocks: Annotated[list[str], operator.add]
    source_items: Annotated[list[str], operator.add]
    completed_sections: list[CompletedSection]


class SectionSubgraphOutputState(TypedDict):
    """Output state for one section-building subgraph."""

    completed_sections: list[CompletedSection]


class Module4BriefRequest(BaseModel):
    """Request body for generating a module 4 research brief."""

    topic: str = DEFAULT_TOPIC
    audience: str = DEFAULT_AUDIENCE
    max_sections: int = Field(default=DEFAULT_MAX_SECTIONS, ge=1, le=MAX_SECTIONS_LIMIT)
    include_wikipedia: bool = True
    include_web: bool = True
    model: str | None = None
    model_provider: str | None = None


class Module4SectionResponse(BaseModel):
    """Response body for one completed research brief section."""

    order: int
    key: str
    title: str
    markdown: str
    sources: list[str]


class Module4BriefResponse(BaseModel):
    """Response body for a generated module 4 research brief."""

    topic: str
    audience: str
    max_sections: int
    overview: str
    final_report: str
    sections: list[Module4SectionResponse]
    sources: list[str]
    model: str
    model_provider: str
