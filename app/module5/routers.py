"""HTTP router for module 5."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.errors import ApiError
from app.logging import get_logger
from app.module5.dependencies import get_module5_service
from app.module5.schemas import (
    Module5ChatRequest,
    Module5ChatResponse,
    Module5MemoryResponse,
)
from app.module5.services.module_service import Module5Service, Module5ServiceError

router = APIRouter(prefix="/module5", tags=["module5"])
logger = get_logger(__name__)

ERROR_STATUS_CODES = {
    "chat_turn_failed": status.HTTP_502_BAD_GATEWAY,
}


@router.post("/chat", response_model=Module5ChatResponse)
async def chat(
    request: Module5ChatRequest,
    service: Annotated[Module5Service, Depends(get_module5_service)],
) -> Module5ChatResponse:
    """Run one module 5 conversation turn.

    Args:
        request: Conversation turn request.
        service: Module 5 application service.

    Returns:
        Assistant response with updated memory snapshot.

    Raises:
        ApiError: If the graph turn fails.
    """
    logger.info(
        "module5 chat turn for user_id=%s thread_id=%s",
        request.user_id,
        request.thread_id or "<new>",
    )
    try:
        return await service.chat(request)
    except Module5ServiceError as exc:
        raise ApiError(
            status_code=ERROR_STATUS_CODES.get(exc.code, status.HTTP_400_BAD_REQUEST),
            code=exc.code,
            message=exc.message,
        ) from exc


@router.get("/memory/{user_id}", response_model=Module5MemoryResponse)
async def get_memory(
    user_id: str,
    service: Annotated[Module5Service, Depends(get_module5_service)],
) -> Module5MemoryResponse:
    """Return the current memory snapshot for a user.

    Args:
        user_id: Long-term memory user id.
        service: Module 5 application service.

    Returns:
        Serialized memory snapshot.
    """
    logger.info("module5 memory read for user_id=%s", user_id)
    return await service.get_memory(user_id)
