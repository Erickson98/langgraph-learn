"""Shared application routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return application health.

    Returns:
        Health check response.
    """
    return HealthResponse(status="ok")
