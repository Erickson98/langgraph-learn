"""FastAPI integration tests."""

from __future__ import annotations

from unittest.mock import ANY, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config.settings import Settings, get_settings
from app.main import create_app


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
