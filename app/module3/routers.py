"""HTTP router for module 3."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.errors import ApiError
from app.logging import get_logger
from app.module3.dependencies import get_module3_service
from app.module3.schemas import (
    Module3ApproveRequest,
    Module3ForkRequest,
    Module3HistoryRequest,
    Module3HistoryResponse,
    Module3ReplayRequest,
    Module3StateRequest,
    Module3StateResponse,
    Module3TurnRequest,
    Module3TurnResponse,
)
from app.module3.services.module_service import Module3Service, Module3ServiceError

router = APIRouter(prefix="/module3", tags=["module3"])
logger = get_logger(__name__)


@router.post("/turn", response_model=Module3TurnResponse)
async def run_module3_turn(
    request: Module3TurnRequest,
    service: Annotated[Module3Service, Depends(get_module3_service)],
) -> Module3TurnResponse:
    """Run one module 3 breakpoint graph turn.

    Args:
        request: Prompt and model configuration for the turn.
        service: Module 3 application service.

    Returns:
        Serialized graph transition result.

    Raises:
        ApiError: If model configuration or credentials are invalid.
    """
    logger.info("Running module3 turn for thread_id=%s", request.thread_id)
    try:
        return await service.run_turn(request)
    except Module3ServiceError as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=exc.code,
            message=exc.message,
        ) from exc


@router.post("/approve", response_model=Module3TurnResponse)
async def approve_module3_turn(
    request: Module3ApproveRequest,
    service: Annotated[Module3Service, Depends(get_module3_service)],
) -> Module3TurnResponse:
    """Approve a paused module 3 tool call and resume execution.

    Args:
        request: Thread and model configuration for the paused turn.
        service: Module 3 application service.

    Returns:
        Serialized graph transition result after approval.

    Raises:
        ApiError: If model configuration or credentials are invalid.
    """
    logger.info("Approving module3 turn for thread_id=%s", request.thread_id)
    try:
        return await service.approve_turn(request)
    except Module3ServiceError as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=exc.code,
            message=exc.message,
        ) from exc


@router.get("/state", response_model=Module3StateResponse)
async def get_module3_state(
    request: Annotated[Module3StateRequest, Depends()],
    service: Annotated[Module3Service, Depends(get_module3_service)],
) -> Module3StateResponse:
    """Read the current module 3 thread state.

    Args:
        request: Thread query parameters.
        service: Module 3 application service.

    Returns:
        Serialized thread state.
    """
    logger.info("Reading module3 state for thread_id=%s", request.thread_id)
    return await service.get_state(request)


@router.get("/history", response_model=Module3HistoryResponse)
async def get_module3_history(
    request: Annotated[Module3HistoryRequest, Depends()],
    service: Annotated[Module3Service, Depends(get_module3_service)],
) -> Module3HistoryResponse:
    """Read checkpoint history for a module 3 thread.

    Args:
        request: Thread query parameters.
        service: Module 3 application service.

    Returns:
        Serialized checkpoint history.
    """
    logger.info("Reading module3 history for thread_id=%s", request.thread_id)
    return await service.get_history(request)


@router.post("/replay", response_model=Module3TurnResponse)
async def replay_module3_checkpoint(
    request: Module3ReplayRequest,
    service: Annotated[Module3Service, Depends(get_module3_service)],
) -> Module3TurnResponse:
    """Replay a stored checkpoint for a module 3 thread.

    Args:
        request: Thread, checkpoint, and model configuration.
        service: Module 3 application service.

    Returns:
        Serialized graph transition result after replay.

    Raises:
        ApiError: If model configuration, credentials, or checkpoint are invalid.
    """
    logger.info(
        "Replaying module3 checkpoint for thread_id=%s checkpoint_id=%s",
        request.thread_id,
        request.checkpoint_id,
    )
    try:
        return await service.replay_checkpoint(request)
    except Module3ServiceError as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=exc.code,
            message=exc.message,
        ) from exc


@router.post("/fork", response_model=Module3TurnResponse)
async def fork_module3_checkpoint(
    request: Module3ForkRequest,
    service: Annotated[Module3Service, Depends(get_module3_service)],
) -> Module3TurnResponse:
    """Fork a checkpoint by replacing the pending human prompt.

    Args:
        request: Thread, checkpoint, replacement prompt, and model configuration.
        service: Module 3 application service.

    Returns:
        Serialized graph transition result after the fork.

    Raises:
        ApiError: If model configuration, credentials, or checkpoint are invalid.
    """
    logger.info(
        "Forking module3 checkpoint for thread_id=%s checkpoint_id=%s",
        request.thread_id,
        request.checkpoint_id,
    )
    try:
        return await service.fork_checkpoint(request)
    except Module3ServiceError as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=exc.code,
            message=exc.message,
        ) from exc
