"""HTTP router for module 1."""

from __future__ import annotations

from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, status

from app.config.settings import Settings, get_settings
from app.errors import ApiError
from app.logging import get_logger
from app.module1.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
)
from app.module1.schemas import Module1TurnRequest, Module1TurnResponse
from app.module1.services.graph_service import build_graph, run_turn

router = APIRouter(prefix="/module1", tags=["module1"])
logger = get_logger(__name__)


@router.post("/turn", response_model=Module1TurnResponse)
async def run_module1_turn(
    request: Module1TurnRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Module1TurnResponse:
    """Run one module 1 graph turn.

    Args:
        request: Prompt and model configuration for the turn.
        settings: Runtime settings injected by FastAPI.

    Returns:
        Assistant response for the requested graph turn.

    Raises:
        ApiError: If model provider configuration is invalid or credentials are missing.
    """
    logger.info("Running module1 turn for thread_id=%s", request.thread_id)
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

    graph = build_graph(
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    response = await anyio.to_thread.run_sync(
        run_turn,
        graph,
        request.prompt,
        request.thread_id,
    )
    return Module1TurnResponse(
        response=response,
        thread_id=request.thread_id,
        model=model_config.model,
        model_provider=model_config.model_provider,
    )
