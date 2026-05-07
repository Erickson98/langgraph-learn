# Project Context

LangGraph Learn is a collection of small agent modules used to explore LangGraph patterns, model-agnostic LangChain setup, and API exposure through a shared FastAPI application.

## Current Direction

| Area | Decision |
| --- | --- |
| Module ownership | Each migrated module lives under `app/moduleN` and acts as its own agent workstream. |
| Runtime configuration | Shared settings live in `app/config/settings.py` and load environment values through Pydantic. |
| Dependency management | `pyproject.toml` and `uv.lock` are the source of truth. |
| API shape | FastAPI exposes shared health endpoints and module-specific routers. |
| Tests | Unit tests isolate service behavior, and API tests avoid live LLM calls. |

## Constraints

Modules 1, 2, 3, 4, and 5 now follow the migrated folder pattern. Module 5 remains CLI-only until persistent long-term memory storage and an API workflow are explicitly scoped.
