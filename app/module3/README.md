# Module 3 Checkpointing Agent

Module 3 demonstrates LangGraph checkpointing patterns using a simple arithmetic agent as the vehicle. The agent can add, multiply, subtract, and divide numbers; most patterns shown here apply equally to any tool-using graph.

Patterns covered:

| Demo | Pattern |
| --- | --- |
| `breakpoints` | Static interrupt-before a node; resume after human approval |
| `interactive-breakpoints` | Multi-turn breakpoint loop |
| `dynamic-breakpoints` | `NodeInterrupt` raised from inside a node at runtime |
| `edit-state` | Modify the last human message before the LLM continues |
| `human-feedback` | Dedicated feedback node that writes approval into state |
| `time-travel` | Replay execution from a past checkpoint |
| `time-travel` (fork) | Replace a message and re-run from a historical point |
| `streaming` | Token-level streaming with automatic summarization |
| `interactive-chat` | Multi-turn summarizing REPL |
| `streaming-events` | `astream_events` v2 with per-chunk printing |

## Runtime

The module runs through both FastAPI and the CLI. API state uses SQLite through `MODULE3_MEMORY_DB`, which defaults to `data/module3.sqlite`. The CLI uses the same shared settings and model provider configuration as the other migrated modules.

## API

Start the app:

```bash
uv run uvicorn app.main:app --reload
```

Start a breakpoint turn:

```bash
curl -X POST http://127.0.0.1:8000/module3/turn \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Multiply 2 and 3.",
    "thread_id": "manual-module3",
    "model": "gpt-4o-mini",
    "model_provider": "openai"
  }'
```

Approve the paused tool call:

```bash
curl -X POST http://127.0.0.1:8000/module3/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "manual-module3"}'
```

Inspect state and history:

```bash
curl "http://127.0.0.1:8000/module3/state?thread_id=manual-module3"
curl "http://127.0.0.1:8000/module3/history?thread_id=manual-module3"
```

## CLI

Run a static breakpoint demo:

```bash
uv run python -m app.module3.main breakpoints --auto-approve
```

Run the dynamic breakpoint demo without provider credentials:

```bash
uv run python -m app.module3.main dynamic-breakpoints
```

Run interactive breakpoint approval:

```bash
uv run python -m app.module3.main interactive-breakpoints
```

Run replay and fork behavior:

```bash
uv run python -m app.module3.main time-travel --auto-approve
```

## File map

| File | Purpose |
| --- | --- |
| `schemas.py` | Constants, `PendingToolCall`, `MessageView`, all request/response DTOs |
| `dependencies.py` | Provider validation, credential checking, `get_chat_model`, FastAPI DI |
| `routers.py` | `POST /turn`, `POST /approve`, `GET /state`, `GET /history`, `POST /replay`, `POST /fork` |
| `services/tools.py` | Arithmetic tools (`add`, `multiply`, `subtract`, `divide`) |
| `services/graph_service.py` | Graph builders, snapshot serializers, checkpoint operations, async wrappers |
| `services/module_service.py` | Application service used by the router |
| `main.py` | CLI entry point with nine demo modes |

## Tests

Module 3 has unit coverage for dependency resolution, graph helpers, application service behavior, and CLI parsing. Integration tests cover LangGraph checkpointing with patched chat models and FastAPI endpoints with SQLite-backed state.
