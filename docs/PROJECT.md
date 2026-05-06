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

Modules 2 through 5 have only been moved under `app/` so their internal refactor can happen in later iterations. Module 1 is the reference implementation for the folder pattern.

