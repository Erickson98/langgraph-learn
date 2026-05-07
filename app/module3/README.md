# Module 3 Checkpointing Agent

Module 3 demonstrates LangGraph checkpointing patterns: static breakpoints, pending tool approval, state inspection, checkpoint history, replay, forked state edits, human feedback, and streaming events.

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

## Tests

Module 3 has unit coverage for dependency resolution, graph helpers, application service behavior, and CLI parsing. Integration tests cover LangGraph checkpointing with patched chat models and FastAPI endpoints with SQLite-backed state.
