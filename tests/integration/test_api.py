"""FastAPI integration tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import ANY, AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, BaseMessage
from langsmith.run_helpers import tracing_context

from app.config.settings import Settings, get_settings
from app.main import create_app
from app.module2.schemas import Module2TurnResult
from app.module2.services.graph_service import CREATE_SUMMARY_PROMPT
from app.module3.schemas import Module3TurnResult, PendingToolCall


class Module2FakeChatModel:
    """Small chat model test double for module 2 API integration tests."""

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return deterministic assistant and summary messages."""
        contents = [getattr(message, "content", "") for message in messages]

        if contents and (
            contents[-1] == CREATE_SUMMARY_PROMPT
            or contents[-1].startswith("Current summary:")
        ):
            return AIMessage(content="stored summary")

        return AIMessage(content=f"messages={len(messages)}")


class Module3FakeToolCallingChatModel:
    """Small chat model test double for module 3 API integration tests."""

    def __init__(self) -> None:
        """Create a fake model with no bound tools."""
        self.bound_tool_count = 0

    def bind_tools(self, tools: list[object]) -> "Module3FakeToolCallingChatModel":
        """Record bound tools and return this fake model.

        Args:
            tools: Tools bound to the model.

        Returns:
            This fake model.
        """
        self.bound_tool_count = len(tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return a deterministic tool call, then a final answer."""
        if any(message.type == "tool" for message in messages):
            return AIMessage(content=f"tool-count={self.bound_tool_count} answer=6")

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "multiply",
                    "args": {"a": 2, "b": 3},
                    "type": "tool_call",
                }
            ],
        )


@pytest.mark.anyio
async def test_health_check_returns_ok() -> None:
    """Health endpoint should return a stable API response."""
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_module1_turn_runs_graph_with_mocked_llm() -> None:
    """Module 1 API should call the graph service without a live LLM in tests."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        langchain_chat_model="gpt-4o-mini",
        langchain_model_provider="openai",
        openai_api_key="test-key",
    )
    transport = ASGITransport(app=app)

    with (
        patch("app.module1.routers.build_graph", return_value="graph") as build_graph,
        patch(
            "app.module1.routers.run_turn",
            return_value="4",
        ) as run_turn,
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/module1/turn",
                json={"prompt": "What is 2 + 2?", "thread_id": "api-thread"},
            )

    assert response.status_code == 200
    assert response.json() == {
        "response": "4",
        "thread_id": "api-thread",
        "model": "gpt-4o-mini",
        "model_provider": "openai",
    }
    build_graph.assert_called_once_with(
        model="gpt-4o-mini",
        model_provider="openai",
        settings=ANY,
    )
    run_turn.assert_called_once_with("graph", "What is 2 + 2?", "api-thread")


@pytest.mark.anyio
async def test_module1_turn_returns_standard_error_for_unknown_provider() -> None:
    """Module 1 API should return a stable error body for bad providers."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        langchain_chat_model="gpt-4o-mini",
        langchain_model_provider="openai",
        openai_api_key="test-key",
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/module1/turn",
            json={
                "prompt": "What is 2 + 2?",
                "thread_id": "api-thread",
                "model_provider": "opneai",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_model_provider"
    assert "Unsupported model provider 'opneai'" in response.json()["error"]["message"]


@pytest.mark.anyio
async def test_module1_turn_returns_standard_error_for_missing_credentials() -> None:
    """Module 1 API should return a stable error body when credentials are absent."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        langchain_chat_model="gpt-4o-mini",
        langchain_model_provider="openai",
        openai_api_key="",
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/module1/turn",
            json={"prompt": "What is 2 + 2?", "thread_id": "api-thread"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "missing_model_credentials",
            "message": "OPENAI_API_KEY is not set for provider 'openai'.",
        }
    }


@pytest.mark.anyio
async def test_module1_turn_returns_standard_error_for_validation_error() -> None:
    """FastAPI validation errors should use the shared error response shape."""
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/module1/turn")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"] == "Request validation failed."
    assert "errors" in body["error"]["details"]


@pytest.mark.anyio
async def test_module2_turn_runs_graph_with_mocked_llm() -> None:
    """Module 2 API should call the graph service without a live LLM in tests."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        langchain_chat_model="gpt-4o-mini",
        langchain_model_provider="openai",
        openai_api_key="test-key",
        module2_memory_db="test-module2.sqlite",
    )
    transport = ASGITransport(app=app)

    with patch(
        "app.module2.services.module_service.run_turn_with_sqlite_async",
        new_callable=AsyncMock,
        return_value=Module2TurnResult(response="answer", summary="stored summary"),
    ) as run_turn_with_sqlite:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/module2/turn",
                json={
                    "prompt": "Remember this.",
                    "thread_id": "api-thread",
                    "summarize_after": 3,
                },
            )

    assert response.status_code == 200
    assert response.json() == {
        "response": "answer",
        "summary": "stored summary",
        "thread_id": "api-thread",
        "summarize_after": 3,
        "model": "gpt-4o-mini",
        "model_provider": "openai",
    }
    run_turn_with_sqlite.assert_awaited_once_with(
        prompt="Remember this.",
        thread_id="api-thread",
        summarize_after=3,
        model="gpt-4o-mini",
        model_provider="openai",
        memory_db="test-module2.sqlite",
        settings=ANY,
    )


@pytest.mark.anyio
async def test_module2_turn_returns_standard_error_for_unknown_provider() -> None:
    """Module 2 API should return a stable error body for bad providers."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        langchain_chat_model="gpt-4o-mini",
        langchain_model_provider="openai",
        openai_api_key="test-key",
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/module2/turn",
            json={
                "prompt": "Remember this.",
                "thread_id": "api-thread",
                "model_provider": "opneai",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_model_provider"
    assert "Unsupported model provider 'opneai'" in response.json()["error"]["message"]


@pytest.mark.anyio
async def test_module2_turn_returns_standard_error_for_missing_credentials() -> None:
    """Module 2 API should return a stable error body when credentials are absent."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        langchain_chat_model="gpt-4o-mini",
        langchain_model_provider="openai",
        openai_api_key="",
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/module2/turn",
            json={"prompt": "Remember this.", "thread_id": "api-thread"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "missing_model_credentials",
            "message": "OPENAI_API_KEY is not set for provider 'openai'.",
        }
    }


@pytest.mark.anyio
async def test_module2_api_persists_sqlite_state_with_mocked_model() -> None:
    """Module 2 API should use real SQLite while mocking only the LLM."""
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_db = str(Path(temp_dir) / "module2.sqlite")
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            _env_file=None,
            langchain_chat_model="gpt-4o-mini",
            langchain_model_provider="openai",
            openai_api_key="test-key",
            module2_memory_db=memory_db,
        )
        transport = ASGITransport(app=app)

        with (
            patch(
                "app.module2.services.graph_service.get_chat_model",
                return_value=Module2FakeChatModel(),
            ),
            tracing_context(enabled=False),
        ):
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                first_response = await client.post(
                    "/module2/turn",
                    json={
                        "prompt": "First message.",
                        "thread_id": "sqlite-api-thread",
                        "summarize_after": 2,
                    },
                )
                second_response = await client.post(
                    "/module2/turn",
                    json={
                        "prompt": "Second message.",
                        "thread_id": "sqlite-api-thread",
                        "summarize_after": 2,
                    },
                )
                summary_response = await client.get(
                    "/module2/summary",
                    params={"thread_id": "sqlite-api-thread"},
                )

    assert first_response.status_code == 200
    assert first_response.json()["response"] == "messages=1"
    assert first_response.json()["summary"] == ""

    assert second_response.status_code == 200
    assert second_response.json()["response"] == "messages=3"
    assert second_response.json()["summary"] == "stored summary"

    assert summary_response.status_code == 200
    assert summary_response.json() == {
        "summary": "stored summary",
        "thread_id": "sqlite-api-thread",
    }


@pytest.mark.anyio
async def test_module2_summary_returns_empty_for_unknown_thread() -> None:
    """Module 2 summary endpoint should return an empty summary for new threads."""
    with tempfile.TemporaryDirectory() as temp_dir:
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            _env_file=None,
            module2_memory_db=str(Path(temp_dir) / "module2.sqlite"),
        )
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/module2/summary",
                params={"thread_id": "unknown-thread"},
            )

    assert response.status_code == 200
    assert response.json() == {"summary": "", "thread_id": "unknown-thread"}


@pytest.mark.anyio
async def test_module3_turn_runs_graph_with_mocked_llm() -> None:
    """Module 3 API should call the graph service without a live LLM in tests."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        langchain_chat_model="gpt-4o-mini",
        langchain_model_provider="openai",
        openai_api_key="test-key",
        module3_memory_db="test-module3.sqlite",
    )
    transport = ASGITransport(app=app)

    with patch(
        "app.module3.services.module_service.run_turn_with_sqlite_async",
        new_callable=AsyncMock,
        return_value=Module3TurnResult(
            status="paused",
            response="",
            pending_next=("tools",),
            pending_tool_calls=[
                PendingToolCall(
                    id="call-1",
                    name="multiply",
                    args={"a": 2, "b": 3},
                )
            ],
            message_count=2,
        ),
    ) as run_turn_with_sqlite:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/module3/turn",
                json={
                    "prompt": "Multiply 2 and 3.",
                    "thread_id": "api-thread",
                },
            )

    assert response.status_code == 200
    assert response.json() == {
        "status": "paused",
        "response": "",
        "pending_next": ["tools"],
        "pending_tool_calls": [
            {
                "id": "call-1",
                "name": "multiply",
                "args": {"a": 2, "b": 3},
            }
        ],
        "message_count": 2,
        "thread_id": "api-thread",
        "model": "gpt-4o-mini",
        "model_provider": "openai",
    }
    run_turn_with_sqlite.assert_awaited_once_with(
        prompt="Multiply 2 and 3.",
        thread_id="api-thread",
        model="gpt-4o-mini",
        model_provider="openai",
        memory_db="test-module3.sqlite",
        settings=ANY,
    )


@pytest.mark.anyio
async def test_module3_turn_returns_standard_error_for_unknown_provider() -> None:
    """Module 3 API should return a stable error body for bad providers."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        langchain_chat_model="gpt-4o-mini",
        langchain_model_provider="openai",
        openai_api_key="test-key",
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/module3/turn",
            json={
                "prompt": "Multiply 2 and 3.",
                "thread_id": "api-thread",
                "model_provider": "opneai",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_model_provider"
    assert "Unsupported model provider 'opneai'" in response.json()["error"]["message"]


@pytest.mark.anyio
async def test_module3_api_persists_sqlite_state_with_mocked_model() -> None:
    """Module 3 API should pause, approve, and expose state/history via SQLite."""
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_db = str(Path(temp_dir) / "module3.sqlite")
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            _env_file=None,
            langchain_chat_model="gpt-4o-mini",
            langchain_model_provider="openai",
            openai_api_key="test-key",
            module3_memory_db=memory_db,
        )
        transport = ASGITransport(app=app)

        with (
            patch(
                "app.module3.services.graph_service.get_chat_model",
                return_value=Module3FakeToolCallingChatModel(),
            ),
            tracing_context(enabled=False),
        ):
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                turn_response = await client.post(
                    "/module3/turn",
                    json={
                        "prompt": "Multiply 2 and 3.",
                        "thread_id": "sqlite-api-thread",
                    },
                )
                state_response = await client.get(
                    "/module3/state",
                    params={"thread_id": "sqlite-api-thread"},
                )
                history_response = await client.get(
                    "/module3/history",
                    params={"thread_id": "sqlite-api-thread"},
                )
                approve_response = await client.post(
                    "/module3/approve",
                    json={"thread_id": "sqlite-api-thread"},
                )

    assert turn_response.status_code == 200
    assert turn_response.json()["status"] == "paused"
    assert turn_response.json()["pending_next"] == ["tools"]
    assert turn_response.json()["pending_tool_calls"][0]["name"] == "multiply"

    assert state_response.status_code == 200
    assert state_response.json()["status"] == "paused"

    assert history_response.status_code == 200
    assert history_response.json()["checkpoints"]

    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "completed"
    assert approve_response.json()["response"] == "tool-count=4 answer=6"
