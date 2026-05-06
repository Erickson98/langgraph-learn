"""Application service for module 2 API workflows."""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import Settings
from app.module2.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
)
from app.module2.schemas import (
    Module2SummaryRequest,
    Module2SummaryResponse,
    Module2TurnRequest,
    Module2TurnResponse,
)
from app.module2.services.graph_service import (
    get_summary_with_sqlite_async,
    run_turn_with_sqlite_async,
)


class Module2ServiceError(Exception):
    """Expected module 2 service error with a stable client-facing code."""

    def __init__(self, code: str, message: str) -> None:
        """Create a module 2 service error.

        Args:
            code: Stable client-facing error code.
            message: Human-readable error message.
        """
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class Module2Service:
    """Application service for module 2 graph operations."""

    settings: Settings

    async def run_turn(self, request: Module2TurnRequest) -> Module2TurnResponse:
        """Run one module 2 graph turn.

        Args:
            request: Prompt, thread, summarization, and model configuration.

        Returns:
            Assistant response and current conversation summary.

        Raises:
            Module2ServiceError: If provider configuration or credentials are invalid.
        """
        try:
            model_config = get_chat_model_config(
                model=request.model,
                model_provider=request.model_provider,
                settings=self.settings,
            )
        except ValueError as exc:
            raise Module2ServiceError(
                code="unsupported_model_provider",
                message=str(exc),
            ) from exc

        if not has_model_credentials(
            model_config.model_provider,
            settings=self.settings,
        ):
            api_key_name = get_required_api_key_name(model_config.model_provider)
            raise Module2ServiceError(
                code="missing_model_credentials",
                message=(
                    f"{api_key_name} is not set for provider "
                    f"'{model_config.model_provider}'."
                ),
            )

        result = await run_turn_with_sqlite_async(
            prompt=request.prompt,
            thread_id=request.thread_id,
            summarize_after=request.summarize_after,
            model=model_config.model,
            model_provider=model_config.model_provider,
            memory_db=self.settings.module2_memory_db,
            settings=self.settings,
        )
        return Module2TurnResponse(
            response=result.response,
            summary=result.summary,
            thread_id=request.thread_id,
            summarize_after=request.summarize_after,
            model=model_config.model,
            model_provider=model_config.model_provider,
        )

    async def get_summary(
        self,
        request: Module2SummaryRequest,
    ) -> Module2SummaryResponse:
        """Read the current summary for a module 2 thread.

        Args:
            request: Summary query request.

        Returns:
            Current summary for the requested thread.
        """
        summary = await get_summary_with_sqlite_async(
            thread_id=request.thread_id,
            memory_db=self.settings.module2_memory_db,
        )
        return Module2SummaryResponse(summary=summary, thread_id=request.thread_id)
