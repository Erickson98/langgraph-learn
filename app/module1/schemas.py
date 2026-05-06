"""Shared constants and typed values for module 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel

DEFAULT_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_MODEL_PROVIDER: Final[str] = "openai"
MODEL_ENV_VAR: Final[str] = "LANGCHAIN_CHAT_MODEL"
MODEL_PROVIDER_ENV_VAR: Final[str] = "LANGCHAIN_MODEL_PROVIDER"
DEFAULT_THREAD_ID: Final[str] = "math-demo"
DEFAULT_PROMPT: Final[str] = (
    "What is ((7 * 6) - 5) / 3? Also tell me the remainder of 43 divided by 5."
)
SYSTEM_PROMPT: Final[str] = (
    "You are a helpful assistant tasked with performing arithmetic and basic math "
    "operations on a set of inputs. Reuse relevant context from earlier messages "
    "in the same thread."
)


@dataclass(frozen=True)
class ChatModelConfig:
    """Configuration for LangChain chat model initialization."""

    model: str
    model_provider: str


class Module1TurnRequest(BaseModel):
    """Request body for a module 1 graph turn."""

    prompt: str = DEFAULT_PROMPT
    thread_id: str = DEFAULT_THREAD_ID
    model: str | None = None
    model_provider: str | None = None


class Module1TurnResponse(BaseModel):
    """Response body for a module 1 graph turn."""

    response: str
    thread_id: str
    model: str
    model_provider: str
