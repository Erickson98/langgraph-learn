"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from app.config.settings import get_settings
from app.errors import register_error_handlers
from app.logging import configure_logging, get_logger
from app.module1.routers import router as module1_router
from app.module2.routers import router as module2_router
from app.module3.routers import router as module3_router
from app.routers import router as health_router

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="LangGraph Learn", version="0.1.0")
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(module1_router)
    app.include_router(module2_router)
    app.include_router(module3_router)
    logger.info("FastAPI application configured")
    return app


app = create_app()
