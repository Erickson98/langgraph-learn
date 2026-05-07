"""Application service for module 5 API workflows."""

from __future__ import annotations

from app.config.settings import Settings
from app.logging import get_logger
from app.module5.schemas import (
    MemorySnapshot,
    Module5ChatRequest,
    Module5ChatResponse,
    Module5MemoryResponse,
)
from app.module5.services.graph_service import (
    get_memory_snapshot,
    new_thread_id,
    run_turn_with_sqlite_async,
)

from langgraph.store.base import BaseStore

logger = get_logger(__name__)


class Module5ServiceError(Exception):
    """Expected module 5 service error with a stable client-facing code."""

    def __init__(self, code: str, message: str) -> None:
        """Create a module 5 service error.

        Args:
            code: Stable client-facing error code.
            message: Human-readable error message.
        """
        self.code = code
        self.message = message
        super().__init__(message)


class Module5Service:
    """Application service for module 5 memory assistant operations.

    The service is stateless with respect to conversation history — each turn
    opens the SQLite checkpointer, executes the graph, and closes the
    connection.  Long-term memory (profile, todos, instructions) lives in the
    ``store``, which the caller is responsible for keeping alive across
    requests.
    """

    def __init__(self, settings: Settings, store: BaseStore) -> None:
        """Store runtime settings and the shared long-term memory store.

        Args:
            settings: Runtime settings, including the SQLite database path.
            store: Long-term memory store shared across all requests.
        """
        self._settings = settings
        self._store = store

    @staticmethod
    def _memory_response(snapshot: MemorySnapshot) -> Module5MemoryResponse:
        """Convert a MemorySnapshot dataclass into the API response schema."""
        return Module5MemoryResponse(
            profile=snapshot.profile,
            todos=snapshot.todos,
            instructions=snapshot.instructions,
        )

    async def chat(self, request: Module5ChatRequest) -> Module5ChatResponse:
        """Run one conversation turn and return the assistant response with memory.

        Args:
            request: Conversation turn request.

        Returns:
            Assistant response with updated memory snapshot.

        Raises:
            Module5ServiceError: If the graph turn fails.
        """
        thread_id = request.thread_id or new_thread_id("module5")
        try:
            result = await run_turn_with_sqlite_async(
                prompt=request.prompt,
                user_id=request.user_id,
                thread_id=thread_id,
                memory_db=self._settings.module5_memory_db,
                settings=self._settings,
                store=self._store,
            )
        except Exception as exc:
            logger.exception(
                "Module 5 chat turn failed for user_id=%s", request.user_id
            )
            raise Module5ServiceError("chat_turn_failed", str(exc)) from exc

        return Module5ChatResponse(
            response=result.response,
            thread_id=result.thread_id,
            user_id=result.user_id,
            memory=self._memory_response(result.memory),
        )

    async def get_memory(self, user_id: str) -> Module5MemoryResponse:
        """Return the current memory snapshot for a user.

        Args:
            user_id: Long-term memory user id.

        Returns:
            Serialized memory snapshot.
        """
        return self._memory_response(await get_memory_snapshot(self._store, user_id))
