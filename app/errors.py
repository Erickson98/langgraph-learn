"""Shared API error handling."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.logging import get_logger
from app.schemas import ErrorDetail, ErrorResponse

logger = get_logger(__name__)


class ApiError(Exception):
    """Expected API error with a stable client-facing code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
    ) -> None:
        """Create an API error.

        Args:
            status_code: HTTP status code.
            code: Stable client-facing error code.
            message: Human-readable error message.
            details: Optional structured error details.
        """
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


def build_error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    """Build a standardized JSON error response.

    Args:
        status_code: HTTP status code.
        code: Stable client-facing error code.
        message: Human-readable error message.
        details: Optional structured error details.

    Returns:
        JSON response with the standard error body.
    """
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=details,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(body, exclude_none=True),
    )


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """Handle expected application errors.

    Args:
        request: Incoming request.
        exc: Application error.

    Returns:
        Standardized JSON error response.
    """
    logger.warning("%s %s failed: %s", request.method, request.url.path, exc.code)
    return build_error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle framework HTTP exceptions.

    Args:
        request: Incoming request.
        exc: HTTP exception.

    Returns:
        Standardized JSON error response.
    """
    logger.warning("%s %s failed: http_error", request.method, request.url.path)
    message = exc.detail if isinstance(exc.detail, str) else "HTTP error."
    details = None if isinstance(exc.detail, str) else exc.detail
    return build_error_response(
        status_code=exc.status_code,
        code="http_error",
        message=message,
        details=details,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle request validation errors.

    Args:
        request: Incoming request.
        exc: Validation exception.

    Returns:
        Standardized JSON error response.
    """
    logger.warning("%s %s failed: validation_error", request.method, request.url.path)
    return build_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="validation_error",
        message="Request validation failed.",
        details={"errors": exc.errors()},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions.

    Args:
        request: Incoming request.
        exc: Unexpected exception.

    Returns:
        Standardized JSON error response.
    """
    logger.exception("Unhandled error for %s %s", request.method, request.url.path)
    return build_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        message="Internal server error.",
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register global API error handlers.

    Args:
        app: FastAPI application.
    """
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
