# Module 5 Memory Productivity Agent

Module 5 explores how an assistant can maintain long-term user memory across conversations. It tracks profile, todo, and preference memory using a SQLite-backed store so that data survives process restarts. The module exposes both a CLI and a FastAPI surface that mirrors the patterns used by earlier modules.

## User Stories

| User | Need | Outcome |
| --- | --- | --- |
| Individual contributor | Track tasks mentioned in natural language | The assistant updates a structured todo memory. |
| Returning user | Preserve personal context across sessions | The assistant reads SQLite-persisted profile memory before responding. |
| Power user | Set preferences for how tasks are tracked | The assistant stores task-management instructions that survive restarts. |
| API client | Send messages via HTTP | `POST /module5/chat` returns the assistant reply and the current memory snapshot. |
| API client | Inspect stored memory for a user | `GET /module5/memory/{user_id}` returns profile, todos, and instructions as text. |
| Developer | Validate graph wiring without external calls | Tests patch the model and Trustcall extractor while exercising LangGraph flow. |

## Design Decisions

| Decision | Reason |
| --- | --- |
| Add FastAPI surface alongside CLI | Consistent with modules 2–4; enables HTTP integration tests and downstream service use. |
| Use shared model dependencies | The module follows the provider-agnostic `init_chat_model` pattern used by migrated modules. |
| Keep graph logic in services | CLI parsing and HTTP routing stay separate from LangGraph nodes and memory behavior. |
| Use Trustcall behind graph nodes | Structured memory extraction stays isolated and can be patched in tests. |
| Implement `SqliteStore(BaseStore)` | `InMemoryStore` has no persistence; `AsyncSqliteSaver` only covers conversation history. A custom `BaseStore` wrapping SQLite is the minimal change that adds durability without an external service. |
| Share one SQLite file for checkpointer and store | The checkpointer uses the `checkpoints` table; `SqliteStore` uses `module5_store`. Single `module5_memory_db` setting covers both. |
| Lazy singleton for the shared store | The `_shared_store` in `dependencies.py` is built on first request, enabling `dependency_overrides` in tests to inject an `InMemoryStore` before the singleton is initialized. |
| Use ASCII unit separator (`\x1f`) as namespace delimiter | Safe within namespace segments and never appears in user data; avoids collisions with common delimiters. |
| Apply `SearchOp.filter` in-process | SQLite has no JSON index on the value column. Post-query Python filtering keeps the schema simple for the typical small dataset. |

## Constraints

| Constraint | Impact |
| --- | --- |
| Profile, todos, and instructions persist to SQLite | Data survives process restarts when `module5_memory_db` points to a file path. |
| `:memory:` sentinel returns `InMemoryStore` | Use `:memory:` in tests and quick CLI runs where persistence is not needed. |
| Live provider calls are not required for tests | Unit and integration tests patch LLM and extractor behavior. |
| One memory update route per assistant tool call | The assistant binds `UpdateMemory` with parallel tool calls disabled. |
| `SearchOp.filter` is applied in-process | Filtering is O(n) on the result set; suitable for per-user memory sizes (< 1 000 items). |
