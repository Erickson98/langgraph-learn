# Module 2 Summarizing Chatbot

Module 2 builds a multi-turn conversational agent that compacts its own message history as it grows. When the message count crosses a configurable threshold the graph summarizes the older turns, removes them from the active window, and continues the conversation with a condensed system prompt instead. This keeps the context window manageable for long sessions without losing the thread of the conversation.

The module introduces `SqliteSaver` for checkpoint persistence, conditional graph edges, and the `RemoveMessage` pattern for in-graph state mutation.

## LangGraph concepts covered

| Concept | Where |
| --- | --- |
| `MessagesState` with a custom `summary` field | `services/graph_service.py` |
| Conditional edge (`should_continue`) | `services/graph_service.py` |
| `RemoveMessage` to compact history | `services/graph_service.py` |
| `SqliteSaver` for thread persistence | `services/graph_service.py` |
| Async offload with `anyio.to_thread` | `services/graph_service.py` |

## Runtime

Set your model credentials in `.env`:

```
OPENAI_API_KEY=...          # or ANTHROPIC_API_KEY=
MODULE2_MEMORY_DB=data/module2.sqlite
```

## API

Start the app:

```bash
uv run uvicorn app.main:app --reload
```

Send a single turn:

```bash
curl -X POST http://127.0.0.1:8000/module2/turn \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Remember that my favorite number is 7.",
    "thread_id": "my-thread",
    "model": "gpt-4o-mini",
    "model_provider": "openai"
  }'
```

Read the current summary for a thread:

```bash
curl "http://127.0.0.1:8000/module2/summary?thread_id=my-thread"
```

## CLI

Single-turn mode:

```bash
uv run python -m app.module2.main --prompt "My favorite color is blue."
```

Interactive mode (multi-turn REPL):

```bash
uv run python -m app.module2.main --interactive
```

Inside the REPL:

```text
/summary  print the current conversation summary
/exit     quit
```

Custom summarization threshold and database path:

```bash
uv run python -m app.module2.main \
  --interactive \
  --summarize-after 4 \
  --memory-db data/my-chat.sqlite
```

## File map

| File | Purpose |
| --- | --- |
| `schemas.py` | Constants, `ChatModelConfig`, request/response DTOs |
| `dependencies.py` | Provider validation, credential checking, `get_chat_model` |
| `routers.py` | `POST /module2/turn`, `GET /module2/summary` |
| `services/graph_service.py` | Graph builder, `run_turn`, `get_summary`, async wrappers |
| `services/module_service.py` | Application service used by the router |
| `main.py` | CLI entry point |

## Tests

Unit tests cover dependency resolution, graph helpers, service behavior, and CLI parsing. Integration tests cover LangGraph wiring with a patched chat model and SQLite persistence across turns. Live-LLM tests (opt-in via `RUN_LIVE_LLM_TESTS=1`) verify the full provider flow.
