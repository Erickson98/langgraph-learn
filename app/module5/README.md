# Module 5 Memory Productivity Agent

Module 5 is a long-term memory assistant for profile facts, todos, and task-management preferences. It uses LangGraph with a SQLite-backed store for long-term memory (profile, todos, instructions) and an async SQLite checkpointer for conversation history. Both are persisted to the same file so that data survives process restarts.

The module exposes a FastAPI surface alongside the CLI, following the same patterns as modules 2–4.

## Runtime

Configure the memory database path and model credentials in `.env`:

```
MODULE5_MEMORY_DB=data/module5.sqlite
OPENAI_API_KEY=...
```

## API

The module registers two routes under `/module5`:

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/module5/chat` | Send a message; returns the assistant reply and memory snapshot. |
| `GET` | `/module5/memory/{user_id}` | Inspect stored profile, todos, and preferences for a user. |

Start the API server:

```bash
uv run uvicorn app.main:app --reload
```

## CLI

Run the interactive memory assistant:

```bash
uv run python -m app.module5.main --user-id demo-user
```

Use a different provider:

```bash
uv run python -m app.module5.main \
  --model claude-3-5-haiku-latest \
  --model-provider anthropic
```

Inside the CLI:

```text
/memory  inspect stored profile, todos, and preferences
/quit    exit the session
```

## Tests

Module 5 has unit coverage for dependency resolution, CLI parsing, graph helpers, memory formatting, the SQLite store, and the application service. Integration tests cover LangGraph wiring with patched chat models and patched Trustcall extraction without live provider calls. HTTP integration tests cover the `/chat` and `/memory` endpoints via the FastAPI test client.
