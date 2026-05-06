# Wiki

This is the running table of contents for the project.

## Structure

| Path | Purpose |
| --- | --- |
| `app/main.py` | FastAPI application factory. |
| `app/config/settings.py` | Shared Pydantic settings. |
| `app/module1/` | Reference migrated LangGraph agent. |
| `app/module2/` | Moved module placeholder for a later internal refactor. |
| `app/module3/` | Moved module placeholder for a later internal refactor. |
| `app/module4/` | Moved module placeholder for a later internal refactor. |
| `app/module5/` | Moved module placeholder for a later internal refactor. |
| `tests/` | Unit and integration tests. |
| `docs/` | Project and use-case documentation. |

## Docs

| Document | Purpose |
| --- | --- |
| [Project Context](../docs/PROJECT.md) | Project direction, constraints, and decisions. |
| [Module 1 Use Case](../docs/USE-CASE-MODULE1.md) | Reference agent behavior and design decisions. |

## Quick Start

Install dependencies with `uv sync --all-groups`, run tests with `uv run pytest`, and run the API with `uv run uvicorn app.main:app --reload`.

