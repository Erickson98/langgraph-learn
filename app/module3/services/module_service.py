"""Application service for module 3 API workflows."""

from __future__ import annotations

from app.config.settings import Settings
from app.module3.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
)
from app.module3.schemas import (
    Module3ApproveRequest,
    Module3ForkRequest,
    Module3HistoryEntryResponse,
    Module3HistoryRequest,
    Module3HistoryResponse,
    Module3ReplayRequest,
    Module3StateRequest,
    Module3StateResponse,
    Module3TurnRequest,
    Module3TurnResponse,
    MessageResponse,
    MessageView,
    PendingToolCall,
    ToolCallResponse,
)
from app.module3.services.graph_service import (
    approve_pending_turn_with_sqlite_async,
    fork_checkpoint_with_sqlite_async,
    get_state_with_sqlite_async,
    list_history_with_sqlite_async,
    replay_checkpoint_with_sqlite_async,
    run_turn_with_sqlite_async,
)


class Module3ServiceError(Exception):
    """Expected module 3 service error with a stable client-facing code."""

    def __init__(self, code: str, message: str) -> None:
        """Create a module 3 service error.

        Args:
            code: Stable client-facing error code.
            message: Human-readable error message.
        """
        self.code = code
        self.message = message
        super().__init__(message)


class Module3Service:
    """Application service for module 3 breakpoint operations."""

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
        """Resolve and validate model configuration for LLM-backed operations.

        Args:
            model: Optional explicit chat model name.
            model_provider: Optional explicit provider name.

        Returns:
            Resolved model name and provider.

        Raises:
            Module3ServiceError: If provider configuration or credentials are invalid.
        """
        try:
            config = get_chat_model_config(
                model=model,
                model_provider=model_provider,
                settings=self.settings,
            )
        except ValueError as exc:
            raise Module3ServiceError("unsupported_model_provider", str(exc)) from exc

        if not has_model_credentials(
            config.model_provider,
            settings=self.settings,
        ):
            api_key_name = get_required_api_key_name(config.model_provider)
            raise Module3ServiceError(
                "missing_model_credentials",
                f"{api_key_name} is not set for provider '{config.model_provider}'.",
            )

        return config.model, config.model_provider

    @staticmethod
    def _tool_call_response(tool_call: PendingToolCall) -> ToolCallResponse:
        """Convert an internal tool call into the API response schema."""
        return ToolCallResponse(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
        )

    @staticmethod
    def _message_response(message: MessageView) -> MessageResponse:
        """Convert an internal message view into the API response schema."""
        return MessageResponse(
            id=message.id,
            type=message.type,
            content=message.content,
        )

    async def run_turn(self, request: Module3TurnRequest) -> Module3TurnResponse:
        """Run one module 3 breakpoint graph turn.

        Args:
            request: Prompt and model configuration for the turn.

        Returns:
            Serialized breakpoint turn result.
        """
        model, model_provider = self._resolve_model_config(
            request.model,
            request.model_provider,
        )
        result = await run_turn_with_sqlite_async(
            prompt=request.prompt,
            thread_id=request.thread_id,
            model=model,
            model_provider=model_provider,
            memory_db=self.settings.module3_memory_db,
            settings=self.settings,
        )
        return Module3TurnResponse(
            status=result.status,
            response=result.response,
            pending_next=list(result.pending_next),
            pending_tool_calls=[
                self._tool_call_response(tool_call)
                for tool_call in result.pending_tool_calls
            ],
            message_count=result.message_count,
            thread_id=request.thread_id,
            model=model,
            model_provider=model_provider,
        )

    async def approve_turn(
        self,
        request: Module3ApproveRequest,
    ) -> Module3TurnResponse:
        """Approve a paused tool call and resume execution.

        Args:
            request: Thread and model configuration for the paused run.

        Returns:
            Serialized breakpoint turn result after approval.
        """
        model, model_provider = self._resolve_model_config(
            request.model,
            request.model_provider,
        )
        result = await approve_pending_turn_with_sqlite_async(
            thread_id=request.thread_id,
            model=model,
            model_provider=model_provider,
            memory_db=self.settings.module3_memory_db,
            settings=self.settings,
        )
        return Module3TurnResponse(
            status=result.status,
            response=result.response,
            pending_next=list(result.pending_next),
            pending_tool_calls=[
                self._tool_call_response(tool_call)
                for tool_call in result.pending_tool_calls
            ],
            message_count=result.message_count,
            thread_id=request.thread_id,
            model=model,
            model_provider=model_provider,
        )

    async def get_state(self, request: Module3StateRequest) -> Module3StateResponse:
        """Read the current state for a module 3 thread.

        Args:
            request: Thread identifier to inspect.

        Returns:
            Serialized thread state.
        """
        result = await get_state_with_sqlite_async(
            thread_id=request.thread_id,
            memory_db=self.settings.module3_memory_db,
        )
        return Module3StateResponse(
            status=result.status,
            pending_next=list(result.pending_next),
            pending_tool_calls=[
                self._tool_call_response(tool_call)
                for tool_call in result.pending_tool_calls
            ],
            message_count=result.message_count,
            messages=[self._message_response(message) for message in result.messages],
            thread_id=request.thread_id,
        )

    async def get_history(
        self,
        request: Module3HistoryRequest,
    ) -> Module3HistoryResponse:
        """Read checkpoint history for a module 3 thread.

        Args:
            request: Thread identifier to inspect.

        Returns:
            Serialized checkpoint history.
        """
        history = await list_history_with_sqlite_async(
            thread_id=request.thread_id,
            memory_db=self.settings.module3_memory_db,
        )
        return Module3HistoryResponse(
            thread_id=request.thread_id,
            checkpoints=[
                Module3HistoryEntryResponse(
                    checkpoint_id=entry.checkpoint_id,
                    next_nodes=list(entry.next_nodes),
                    source=entry.source,
                    step=entry.step,
                    message_count=entry.message_count,
                    can_replay=entry.can_replay,
                    can_fork=entry.can_fork,
                )
                for entry in history
            ],
        )

    async def replay_checkpoint(
        self,
        request: Module3ReplayRequest,
    ) -> Module3TurnResponse:
        """Replay execution from one checkpoint.

        Args:
            request: Thread, checkpoint, and model configuration.

        Returns:
            Serialized breakpoint turn result after replay.

        Raises:
            Module3ServiceError: If the checkpoint is invalid.
        """
        model, model_provider = self._resolve_model_config(
            request.model,
            request.model_provider,
        )
        try:
            result = await replay_checkpoint_with_sqlite_async(
                thread_id=request.thread_id,
                checkpoint_id=request.checkpoint_id,
                model=model,
                model_provider=model_provider,
                memory_db=self.settings.module3_memory_db,
                settings=self.settings,
            )
        except ValueError as exc:
            raise Module3ServiceError("invalid_checkpoint", str(exc)) from exc

        return Module3TurnResponse(
            status=result.status,
            response=result.response,
            pending_next=list(result.pending_next),
            pending_tool_calls=[
                self._tool_call_response(tool_call)
                for tool_call in result.pending_tool_calls
            ],
            message_count=result.message_count,
            thread_id=request.thread_id,
            model=model,
            model_provider=model_provider,
        )

    async def fork_checkpoint(
        self,
        request: Module3ForkRequest,
    ) -> Module3TurnResponse:
        """Fork execution from one checkpoint with a replacement prompt.

        Args:
            request: Thread, checkpoint, replacement prompt, and model config.

        Returns:
            Serialized breakpoint turn result after the fork.

        Raises:
            Module3ServiceError: If the checkpoint is invalid.
        """
        model, model_provider = self._resolve_model_config(
            request.model,
            request.model_provider,
        )
        try:
            result = await fork_checkpoint_with_sqlite_async(
                thread_id=request.thread_id,
                checkpoint_id=request.checkpoint_id,
                replacement_prompt=request.replacement_prompt,
                model=model,
                model_provider=model_provider,
                memory_db=self.settings.module3_memory_db,
                settings=self.settings,
            )
        except ValueError as exc:
            raise Module3ServiceError("invalid_checkpoint", str(exc)) from exc

        return Module3TurnResponse(
            status=result.status,
            response=result.response,
            pending_next=list(result.pending_next),
            pending_tool_calls=[
                self._tool_call_response(tool_call)
                for tool_call in result.pending_tool_calls
            ],
            message_count=result.message_count,
            thread_id=request.thread_id,
            model=model,
            model_provider=model_provider,
        )
