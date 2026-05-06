"""HTTP router for module 2."""

from __future__ import annotations

from functools import partial
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, Query, status

from app.config.settings import Settings, get_settings
from app.errors import ApiError
from app.logging import get_logger
from app.module2.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
)
from app.module2.schemas import (
    DEFAULT_THREAD_ID,
    Module2SummaryResponse,
    Module2TurnRequest,
    Module2TurnResponse,
)
from app.module2.services.graph_service import (
    get_summary_with_sqlite,
    run_turn_with_sqlite,
)

router = APIRouter(prefix="/module2", tags=["module2"])
logger = get_logger(__name__)


@router.post("/turn", response_model=Module2TurnResponse)
async def run_module2_turn(
    request: Module2TurnRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Module2TurnResponse:
    """Run one module 2 graph turn.

    Args:
        request: Prompt, thread, summarization, and model configuration.
        settings: Runtime settings injected by FastAPI.

    Returns:
        Assistant response and current conversation summary.

    Raises:
        ApiError: If model provider configuration is invalid or credentials are missing.
    """
    logger.info("Running module2 turn for thread_id=%s", request.thread_id)
    try:
        model_config = get_chat_model_config(
            model=request.model,
            model_provider=request.model_provider,
            settings=settings,
        )
    except ValueError as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="unsupported_model_provider",
            message=str(exc),
        ) from exc

    if not has_model_credentials(model_config.model_provider, settings=settings):
        api_key_name = get_required_api_key_name(model_config.model_provider)
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="missing_model_credentials",
            message=f"{api_key_name} is not set for provider '{model_config.model_provider}'.",
        )

    result = await anyio.to_thread.run_sync(
        partial(
            run_turn_with_sqlite,
            prompt=request.prompt,
            thread_id=request.thread_id,
            summarize_after=request.summarize_after,
            model=model_config.model,
            model_provider=model_config.model_provider,
            memory_db=settings.module2_memory_db,
            settings=settings,
        )
    )
    return Module2TurnResponse(
        response=result.response,
        summary=result.summary,
        thread_id=request.thread_id,
        summarize_after=request.summarize_after,
        model=model_config.model,
        model_provider=model_config.model_provider,
    )


@router.get("/summary", response_model=Module2SummaryResponse)
async def get_module2_summary(
    settings: Annotated[Settings, Depends(get_settings)],
    thread_id: str = Query(default=DEFAULT_THREAD_ID),
) -> Module2SummaryResponse:
    """Read the current summary for a module 2 thread.

    Args:
        settings: Runtime settings injected by FastAPI.
        thread_id: Conversation thread identifier.

    Returns:
        Current summary for the requested thread.
    """
    logger.info("Reading module2 summary for thread_id=%s", thread_id)
    summary = await anyio.to_thread.run_sync(
        partial(
            get_summary_with_sqlite,
            thread_id=thread_id,
            memory_db=settings.module2_memory_db,
        )
    )
    return Module2SummaryResponse(summary=summary, thread_id=thread_id)
