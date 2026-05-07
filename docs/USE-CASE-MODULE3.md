# Module 3 Checkpointing Agent

Module 3 explores how an agent can pause, inspect, resume, replay, and fork graph execution. It is used to learn LangGraph checkpointing behavior before those patterns are reused in more product-oriented agents.

## User Stories

| User | Need | Outcome |
| --- | --- | --- |
| Developer | Pause before tool execution | Tool calls can be inspected before they run. |
| API client | Approve a paused action | The same thread resumes from SQLite-backed state. |
| Tester | Validate checkpoint history | Replay and fork behavior can be tested without live providers. |
| Operator | Inspect current state | Pending nodes, tool calls, messages, and history are visible through API endpoints. |

## Design Decisions

| Decision | Reason |
| --- | --- |
| Use SQLite checkpointing for API workflows | API requests are separate processes from the graph turn and need durable thread state. |
| Keep CLI demos available | Module 3 is educational and still benefits from direct terminal workflows. |
| Use `init_chat_model` through module dependencies | The module stays provider-agnostic and follows the migrated module pattern. |
| Expose approval as a separate endpoint | Human approval is an explicit workflow step, not hidden inside a single request. |
| Keep replay and fork behind checkpoint ids | Clients can choose a concrete historical state instead of relying on implicit latest behavior. |

## Constraints

| Constraint | Impact |
| --- | --- |
| Tool approval depends on checkpointed state | Clients must reuse the same `thread_id` when approving a paused turn. |
| Live LLM calls are not required for tests | Tests patch the chat model and verify LangGraph wiring locally. |
| Dynamic breakpoint demo is CLI-only | It demonstrates interruption mechanics without creating additional API surface. |
