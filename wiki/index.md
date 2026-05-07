# Wiki

This is the running table of contents for the project.

## Structure

| Path | Purpose |
| --- | --- |
| `app/main.py` | FastAPI application factory. |
| `app/config/settings.py` | Shared Pydantic settings. |
| `app/module1/` | Reference migrated LangGraph agent. |
| `app/module2/` | SQLite-backed summarizing chatbot with API and CLI support. |
| `app/module3/` | Checkpointing, breakpoint, approval, replay, and fork demos with API and CLI support. |
| `app/module4/` | Research brief generator with section planning, retrieval, API, and CLI support. |
| `app/module5/` | Long-term memory productivity assistant with profile, todo, and preference memory. |
| `tests/` | Unit and integration tests. |
| `docs/` | Project and use-case documentation. |

## Docs

| Document | Purpose |
| --- | --- |
| [Project Context](../docs/PROJECT.md) | Project direction, constraints, and decisions. |
| [Module 1 Use Case](../docs/USE-CASE-MODULE1.md) | Reference agent behavior and design decisions. |
| [Module 3 Use Case](../docs/USE-CASE-MODULE3.md) | Checkpointing and human approval behavior. |
| [Module 4 Use Case](../docs/USE-CASE-MODULE4.md) | Research brief planning and retrieval behavior. |
| [Module 5 Use Case](../docs/USE-CASE-MODULE5.md) | Long-term profile, todo, and preference memory behavior. |

## Quick Start

Install dependencies with `uv sync --all-groups`, run tests with `uv run pytest`, and run the API with `uv run uvicorn app.main:app --reload`.
