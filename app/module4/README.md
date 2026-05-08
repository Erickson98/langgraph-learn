# Module 4 Research Brief Agent

Module 4 plans and writes a structured research brief on any topic. It uses a main LangGraph graph with a `Send`-powered fan-out to per-section subgraphs, demonstrating parallel node dispatch, structured LLM output, multi-source retrieval, and report compilation.

Graph flow:

```
START → plan_sections → [Send × N] → build_section → write_overview → compile_report → END
```

Each `build_section` subgraph runs independently:

```
START → plan_retrieval_queries → [search_wikipedia ‖ search_web] → draft_section → END
```

## Runtime

The module runs through both FastAPI and the CLI. The LLM is configured through `LANGCHAIN_CHAT_MODEL` and `LANGCHAIN_MODEL_PROVIDER`, with provider credentials loaded from shared settings. Wikipedia retrieval works without provider keys. Web retrieval uses Tavily only when `TAVILY_API_KEY` is set.

## API

Start the app:

```bash
uv run uvicorn app.main:app --reload
```

Generate a research brief:

```bash
curl -X POST http://127.0.0.1:8000/module4/brief \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "LangGraph for production support agents",
    "audience": "engineering leadership",
    "max_sections": 2,
    "include_wikipedia": true,
    "include_web": false,
    "model": "gpt-4o-mini",
    "model_provider": "openai"
  }'
```

## CLI

Run a brief with local settings:

```bash
uv run python -m app.module4.main "LangGraph for production support agents" \
  --audience "engineering leadership" \
  --sections 2 \
  --no-web
```

Write the markdown report to disk:

```bash
uv run python -m app.module4.main "AI sourcing research" \
  --output data/module4-brief.md
```

## File map

| File | Purpose |
| --- | --- |
| `schemas.py` | State types (`ResearchBriefState`, `SectionSubgraphState`), planning and retrieval models, request/response DTOs |
| `dependencies.py` | Provider validation, credential checking, `get_chat_model`, FastAPI DI |
| `routers.py` | `POST /module4/brief` |
| `services/graph_service.py` | Main graph and section subgraph builders, `run_brief`, async wrapper |
| `services/retrieval_service.py` | Wikipedia and Tavily retrieval with graceful fallback |
| `services/text.py` | `clip`, `dedupe`, `message_content_to_text`, `slugify` helpers |
| `services/module_service.py` | Application service used by the router |
| `main.py` | CLI entry point |

## Tests

Module 4 has unit coverage for dependency resolution, retrieval formatting, graph helpers, application service behavior, and CLI parsing. Integration tests cover LangGraph wiring with patched chat models and the FastAPI endpoint without live provider calls.
