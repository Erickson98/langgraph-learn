"""HTTP router for module 4."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.errors import ApiError
from app.logging import get_logger
from app.module4.dependencies import get_module4_service
from app.module4.schemas import Module4BriefRequest, Module4BriefResponse
from app.module4.services.module_service import Module4Service, Module4ServiceError

router = APIRouter(prefix="/module4", tags=["module4"])
logger = get_logger(__name__)
ERROR_STATUS_CODES = {
    "brief_generation_failed": status.HTTP_502_BAD_GATEWAY,
    "missing_model_credentials": status.HTTP_400_BAD_REQUEST,
    "unsupported_model_provider": status.HTTP_400_BAD_REQUEST,
}


@router.post("/brief", response_model=Module4BriefResponse)
async def generate_module4_brief(
    request: Module4BriefRequest,
    service: Annotated[Module4Service, Depends(get_module4_service)],
) -> Module4BriefResponse:
    """Generate a module 4 research brief.

    Args:
        request: Research brief request.
        service: Module 4 application service.

    Returns:
        Generated research brief.

    Raises:
        ApiError: If model configuration, credentials, or generation fail.
    """
    logger.info("Generating module4 brief for topic=%s", request.topic)
    try:
        return await service.generate_brief(request)
    except Module4ServiceError as exc:
        raise ApiError(
            status_code=ERROR_STATUS_CODES.get(
                exc.code,
                status.HTTP_400_BAD_REQUEST,
            ),
            code=exc.code,
            message=exc.message,
        ) from exc
