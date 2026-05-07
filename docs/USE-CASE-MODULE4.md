# Module 4 Research Brief Agent

Module 4 explores how an agent can plan a report, fan out section-level work, retrieve supporting context, and compile a decision-ready markdown brief. It is used to learn retrieval-assisted LangGraph orchestration before those patterns are applied to product research workflows.

## User Stories

| User | Need | Outcome |
| --- | --- | --- |
| Researcher | Turn a topic into a structured brief | The graph plans focused sections and compiles a markdown report. |
| Engineering lead | Understand tradeoffs quickly | The output emphasizes decision-oriented bullets and concise sections. |
| Tester | Validate graph wiring without external calls | Tests patch the LLM and retrieval services while exercising LangGraph flow. |
| API client | Generate briefs through a stable endpoint | FastAPI exposes one request and response schema for brief generation. |

## Design Decisions

| Decision | Reason |
| --- | --- |
| Keep retrieval behind service helpers | Wikipedia and Tavily behavior can be mocked, disabled, or replaced without changing graph flow. |
| Use a section subgraph | Section planning and drafting are isolated from top-level overview and report compilation. |
| Treat Tavily as optional | The module remains usable with only an LLM key, while web retrieval improves freshness when configured. |
| Use shared model dependencies | The module follows the model-agnostic pattern used by the earlier migrated modules. |
| Keep CLI and API surfaces | The CLI is useful for local exploration, while the API keeps the module accessible from the shared app. |

## Constraints

| Constraint | Impact |
| --- | --- |
| Live retrieval is not required for tests | Unit and integration tests avoid network calls by patching retrieval helpers. |
| Web retrieval depends on Tavily credentials | Requests can disable web search or run with a clear skipped-context message when no key is configured. |
| Section count is bounded | The graph limits fan-out to keep local runs predictable and avoid accidental provider cost. |
| Sources are only as strong as retrieved context | The section prompt asks the model to cite only evidence present in the context blocks. |
