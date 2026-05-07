"""Shared constants and typed values for module 5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, Field

DEFAULT_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_MODEL_PROVIDER: Final[str] = "openai"
DEFAULT_USER_ID: Final[str] = "demo-user"


@dataclass(frozen=True)
class ChatModelConfig:
    """Resolved provider-specific model configuration."""

    model: str
    model_provider: str


@dataclass(frozen=True)
class MemorySnapshot:
    """Readable memory values for one user."""

    profile: str
    todos: str
    instructions: str


@dataclass(frozen=True)
class Module5TurnResult:
    """Result of one module 5 graph turn."""

    response: str
    thread_id: str
    user_id: str
    memory: MemorySnapshot


class Profile(BaseModel):
    """Structured long-term profile memory for the user."""

    name: str | None = Field(default=None, description="The user's name")
    location: str | None = Field(default=None, description="The user's location")
    job: str | None = Field(default=None, description="The user's job")
    connections: list[str] = Field(
        default_factory=list,
        description="People connected to the user",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="The user's interests",
    )


class ToDo(BaseModel):
    """Structured long-term task memory for the user."""

    task: str = Field(description="The task to be completed")
    time_to_complete: int | None = Field(
        default=None,
        description="Estimated time in minutes",
    )
    deadline: datetime | None = Field(
        default=None,
        description="When the task should be completed",
    )
    solutions: list[str] = Field(
        default_factory=list,
        description="Concrete solution ideas",
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        default="not started",
        description="Current status of the task",
    )


class UpdateMemory(BaseModel):
    """Tool call selected by the assistant when memory should be updated."""

    update_type: Literal["user", "todo", "instructions"]


class Module5MemoryResponse(BaseModel):
    """Serialized memory snapshot for one user."""

    profile: str
    todos: str
    instructions: str


class Module5ChatRequest(BaseModel):
    """Request body for one module 5 conversation turn."""

    prompt: str = Field(description="User message for this turn")
    user_id: str = Field(
        default=DEFAULT_USER_ID,
        description="Long-term memory user id",
    )
    thread_id: str | None = Field(
        default=None,
        description="Checkpoint thread id; a new thread is created when omitted",
    )


class Module5ChatResponse(BaseModel):
    """Response body for one module 5 conversation turn."""

    response: str
    thread_id: str
    user_id: str
    memory: Module5MemoryResponse
