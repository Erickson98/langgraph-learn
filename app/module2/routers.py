"""HTTP router for module 2."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.errors import ApiError
from app.logging import get_logger
from app.module2.dependencies import get_module2_service
from app.module2.schemas import (
    Module2SummaryRequest,
    Module2SummaryResponse,
    Module2TurnRequest,
    Module2TurnResponse,
)
from app.module2.services.module_service import Module2Service, Module2ServiceError

router = APIRouter(prefix="/module2", tags=["module2"])
logger = get_logger(__name__)


@router.post("/turn", response_model=Module2TurnResponse)
async def run_module2_turn(
    request: Module2TurnRequest,
    service: Annotated[Module2Service, Depends(get_module2_service)],
) -> Module2TurnResponse:
    """Run one module 2 graph turn.

    Args:
        request: Prompt, thread, summarization, and model configuration.
        service: Module 2 application service.

    Returns:
        Assistant response and current conversation summary.

    Raises:
        ApiError: If model provider configuration is invalid or credentials are missing.
    """
    logger.info("Running module2 turn for thread_id=%s", request.thread_id)
    try:
        return await service.run_turn(request)
    except Module2ServiceError as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=exc.code,
            message=exc.message,
        ) from exc


@router.get("/summary", response_model=Module2SummaryResponse)
async def get_module2_summary(
    request: Annotated[Module2SummaryRequest, Depends()],
    service: Annotated[Module2Service, Depends(get_module2_service)],
) -> Module2SummaryResponse:
    """Read the current summary for a module 2 thread.

    Args:
        request: Summary query request.
        service: Module 2 application service.

    Returns:
        Current summary for the requested thread.
    """
    logger.info("Reading module2 summary for thread_id=%s", request.thread_id)
    return await service.get_summary(request)
