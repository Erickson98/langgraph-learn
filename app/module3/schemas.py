"""Shared constants and typed values for module 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel

DEFAULT_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_MODEL_PROVIDER: Final[str] = "openai"
DEFAULT_PROMPT: Final[str] = "Multiply 2 and 3."
DEFAULT_THREAD_ID: Final[str] = "module3-demo"
DEFAULT_MEMORY_DB: Final[Path] = Path("data/module3.sqlite")
DEFAULT_DEMO: Final[str] = "breakpoints"


@dataclass(frozen=True)
class ChatModelConfig:
    """Resolved provider-specific model configuration."""

    model: str
    model_provider: str


@dataclass(frozen=True)
class PendingToolCall:
    """Structured view of a pending tool call."""

    id: str | None
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class MessageView:
    """Serializable message details for thread inspection."""

    id: str | None
    type: str
    content: str


@dataclass(frozen=True)
class Module3TurnResult:
    """Result returned after a breakpoint graph transition."""

    status: Literal["paused", "completed"]
    response: str
    pending_next: tuple[str, ...]
    pending_tool_calls: list[PendingToolCall]
    message_count: int


@dataclass(frozen=True)
class Module3StateResult:
    """Current thread state exposed by the module 3 API."""

    status: Literal["idle", "paused"]
    pending_next: tuple[str, ...]
    pending_tool_calls: list[PendingToolCall]
    message_count: int
    messages: list[MessageView]


@dataclass(frozen=True)
class Module3HistoryEntry:
    """History entry returned for a checkpointed thread."""

    checkpoint_id: str
    next_nodes: tuple[str, ...]
    source: str | None
    step: int | None
    message_count: int
    can_replay: bool
    can_fork: bool


class Module3TurnRequest(BaseModel):
    """Request body for a module 3 graph turn."""

    prompt: str = DEFAULT_PROMPT
    thread_id: str = DEFAULT_THREAD_ID
    model: str | None = None
    model_provider: str | None = None


class Module3ApproveRequest(BaseModel):
    """Request body for approving a paused tool call."""

    thread_id: str = DEFAULT_THREAD_ID
    model: str | None = None
    model_provider: str | None = None


class Module3StateRequest(BaseModel):
    """Request query params for reading a module 3 thread state."""

    thread_id: str = DEFAULT_THREAD_ID


class Module3HistoryRequest(BaseModel):
    """Request query params for listing module 3 checkpoint history."""

    thread_id: str = DEFAULT_THREAD_ID


class Module3ReplayRequest(BaseModel):
    """Request body for replaying a stored checkpoint."""

    thread_id: str = DEFAULT_THREAD_ID
    checkpoint_id: str
    model: str | None = None
    model_provider: str | None = None


class Module3ForkRequest(BaseModel):
    """Request body for forking a checkpoint with a replacement prompt."""

    thread_id: str = DEFAULT_THREAD_ID
    checkpoint_id: str
    replacement_prompt: str
    model: str | None = None
    model_provider: str | None = None


class ToolCallResponse(BaseModel):
    """Serialized pending tool call."""

    id: str | None = None
    name: str
    args: dict[str, Any]


class MessageResponse(BaseModel):
    """Serialized thread message."""

    id: str | None = None
    type: str
    content: str


class Module3TurnResponse(BaseModel):
    """Response body for module 3 graph transitions."""

    status: Literal["paused", "completed"]
    response: str
    pending_next: list[str]
    pending_tool_calls: list[ToolCallResponse]
    message_count: int
    thread_id: str
    model: str
    model_provider: str


class Module3StateResponse(BaseModel):
    """Response body for thread state inspection."""

    status: Literal["idle", "paused"]
    pending_next: list[str]
    pending_tool_calls: list[ToolCallResponse]
    message_count: int
    messages: list[MessageResponse]
    thread_id: str


class Module3HistoryEntryResponse(BaseModel):
    """Response entry for one checkpoint in thread history."""

    checkpoint_id: str
    next_nodes: list[str]
    source: str | None = None
    step: int | None = None
    message_count: int
    can_replay: bool
    can_fork: bool


class Module3HistoryResponse(BaseModel):
    """Response body for module 3 checkpoint history."""

    thread_id: str
    checkpoints: list[Module3HistoryEntryResponse]
