"""Shared API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response body."""

    status: str


class ErrorDetail(BaseModel):
    """Standard API error details."""

    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    """Standard API error response body."""

    error: ErrorDetail
