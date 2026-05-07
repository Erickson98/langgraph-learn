# Module 5 Memory Productivity Agent

Module 5 explores how an assistant can maintain long-term user memory across a conversation. It is used to learn profile, todo, and preference memory patterns before adding a persistent backend or API surface.

## User Stories

| User | Need | Outcome |
| --- | --- | --- |
| Individual contributor | Track tasks mentioned in natural language | The assistant updates a structured todo memory. |
| Returning user | Preserve personal context during a session | The assistant reads profile memory before responding. |
| Power user | Set preferences for how tasks are tracked | The assistant stores task-management instructions. |
| Developer | Validate graph wiring without external calls | Tests patch the model and Trustcall extractor while exercising LangGraph flow. |

## Design Decisions

| Decision | Reason |
| --- | --- |
| Keep module 5 CLI-only for now | The previous module did not expose an API and long-term storage is still in-memory. |
| Use shared model dependencies | The module follows the provider-agnostic `init_chat_model` pattern used by migrated modules. |
| Keep graph logic in services | CLI parsing stays separate from LangGraph nodes and memory behavior. |
| Use Trustcall behind graph nodes | Structured memory extraction stays isolated and can be patched in tests. |
| Use in-memory store initially | The module remains lightweight until persistence requirements are explicit. |

## Constraints

| Constraint | Impact |
| --- | --- |
| Memory is process-local | Stored profile, todos, and preferences disappear when the CLI exits. |
| Live provider calls are not required for tests | Unit and integration tests patch LLM and extractor behavior. |
| One memory update route per assistant tool call | The assistant binds `UpdateMemory` with parallel tool calls disabled. |
