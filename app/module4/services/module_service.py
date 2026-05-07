"""Application service for module 4 API workflows."""

from __future__ import annotations

from app.config.settings import Settings
from app.logging import get_logger
from app.module4.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
)
from app.module4.schemas import (
    CompletedSection,
    Module4BriefRequest,
    Module4BriefResponse,
    Module4SectionResponse,
)
from app.module4.services.graph_service import (
    Module4GraphExecutionError,
    get_sources,
    run_brief_async,
)

logger = get_logger(__name__)


class Module4ServiceError(Exception):
    """Expected module 4 service error with a stable client-facing code."""

    def __init__(self, code: str, message: str) -> None:
        """Create a module 4 service error.

        Args:
            code: Stable client-facing error code.
            message: Human-readable error message.
        """
        self.code = code
        self.message = message
        super().__init__(message)


class Module4Service:
    """Application service for module 4 research brief operations."""

    def __init__(self, settings: Settings) -> None:
        """Store runtime settings for the service.

        Args:
            settings: Runtime settings used by graph helpers.
        """
        self.settings = settings

    def _resolve_model_config(
        self,
        model: str | None,
        model_provider: str | None,
    ) -> tuple[str, str]:
        """Resolve and validate model configuration.

        Args:
            model: Optional explicit chat model name.
            model_provider: Optional explicit provider name.

        Returns:
            Resolved model name and provider.

        Raises:
            Module4ServiceError: If provider configuration or credentials are invalid.
        """
        try:
            config = get_chat_model_config(
                model=model,
                model_provider=model_provider,
                settings=self.settings,
            )
        except ValueError as exc:
            raise Module4ServiceError("unsupported_model_provider", str(exc)) from exc

        if not has_model_credentials(config.model_provider, settings=self.settings):
            api_key_name = get_required_api_key_name(config.model_provider)
            raise Module4ServiceError(
                "missing_model_credentials",
                f"{api_key_name} is not set for provider '{config.model_provider}'.",
            )

        return config.model, config.model_provider

    @staticmethod
    def _section_response(section: CompletedSection) -> Module4SectionResponse:
        """Convert a completed section into the API response schema."""
        return Module4SectionResponse(
            order=section["order"],
            key=section["key"],
            title=section["title"],
            markdown=section["markdown"],
            sources=section["sources"],
        )

    async def generate_brief(
        self,
        request: Module4BriefRequest,
    ) -> Module4BriefResponse:
        """Generate a research brief.

        Args:
            request: Research brief request.

        Returns:
            Generated research brief response.

        Raises:
            Module4ServiceError: If provider configuration, credentials, or external
                graph execution fail.
        """
        model, model_provider = self._resolve_model_config(
            request.model,
            request.model_provider,
        )
        try:
            result = await run_brief_async(
                topic=request.topic,
                audience=request.audience,
                max_sections=request.max_sections,
                include_wikipedia=request.include_wikipedia,
                include_web=request.include_web,
                model=model,
                model_provider=model_provider,
                settings=self.settings,
            )
        except Module4GraphExecutionError as exc:
            logger.exception("Module 4 brief generation failed")
            raise Module4ServiceError(
                "brief_generation_failed",
                str(exc),
            ) from exc

        ordered_sections = sorted(
            result["completed_sections"],
            key=lambda section: section["order"],
        )
        return Module4BriefResponse(
            topic=request.topic,
            audience=request.audience,
            max_sections=request.max_sections,
            overview=result["overview"],
            final_report=result["final_report"],
            sections=[self._section_response(section) for section in ordered_sections],
            sources=get_sources(ordered_sections),
            model=model,
            model_provider=model_provider,
        )
