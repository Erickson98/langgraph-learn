"""Shared constants and typed values for module 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field

DEFAULT_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_MODEL_PROVIDER: Final[str] = "openai"
DEFAULT_THREAD_ID: Final[str] = "module2-demo"
DEFAULT_PROMPT: Final[str] = "My favorite number is 7. Please remember it."
DEFAULT_SUMMARIZE_AFTER: Final[int] = 6
DEFAULT_MEMORY_DB: Final[Path] = Path("data/module2.sqlite")


@dataclass(frozen=True)
class ChatModelConfig:
    """Configuration for LangChain chat model initialization."""

    model: str
    model_provider: str


@dataclass(frozen=True)
class Module2TurnResult:
    """Result returned by the module 2 graph service."""

    response: str
    summary: str


class Module2TurnRequest(BaseModel):
    """Request body for a module 2 graph turn."""

    prompt: str = DEFAULT_PROMPT
    thread_id: str = DEFAULT_THREAD_ID
    summarize_after: int = Field(default=DEFAULT_SUMMARIZE_AFTER, ge=1)
    model: str | None = None
    model_provider: str | None = None


class Module2TurnResponse(BaseModel):
    """Response body for a module 2 graph turn."""

    response: str
    summary: str
    thread_id: str
    summarize_after: int
    model: str
    model_provider: str


class Module2SummaryRequest(BaseModel):
    """Request query params for reading a module 2 summary."""

    thread_id: str = DEFAULT_THREAD_ID


class Module2SummaryResponse(BaseModel):
    """Response body for reading a module 2 thread summary."""

    summary: str
    thread_id: str
