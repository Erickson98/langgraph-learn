# Module 1 Arithmetic Assistant

Module 1 is the smallest LangGraph example in this repo. It builds a single
tool-calling assistant that answers arithmetic prompts, keeps short-term
conversation state in memory, and exposes the same behavior through the CLI and
FastAPI.

This module is intentionally in-memory only. It uses `MemorySaver`, so thread
state lasts for the lifetime of the running process and is not persisted across
restarts.

## LangGraph concepts covered

| Concept | Where |
| --- | --- |
| `StateGraph(MessagesState)` for message-based state | `services/graph_service.py` |
| Tool binding with `llm.bind_tools(...)` | `services/graph_service.py` |
| `ToolNode` to execute arithmetic tools | `services/graph_service.py` |
| `tools_condition` for model-call routing | `services/graph_service.py` |
| `MemorySaver` for process-local thread memory | `services/graph_service.py` |

## Runtime

Set one provider credential in `.env`:

```env
OPENAI_API_KEY=...
# or
ANTHROPIC_API_KEY=...
```

Optional model overrides:

```env
LANGCHAIN_CHAT_MODEL=gpt-4o-mini
LANGCHAIN_MODEL_PROVIDER=openai
```

## API

Start the app:

```bash
uv run uvicorn app.main:app --reload
```

Send one turn:

```bash
curl -X POST http://127.0.0.1:8000/module1/turn \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is ((7 * 6) - 5) / 3?",
    "thread_id": "math-demo",
    "model": "gpt-4o-mini",
    "model_provider": "openai"
  }'
```

## CLI

Single-turn mode:

```bash
uv run python -m app.module1.main \
  --prompt "What is ((7 * 6) - 5) / 3?"
```

Interactive mode:

```bash
uv run python -m app.module1.main --interactive
```

Override the provider:

```bash
uv run python -m app.module1.main \
  --model claude-3-5-haiku-latest \
  --model-provider anthropic \
  --prompt "What is 12 * 8?"
```

## File map

| File | Purpose |
| --- | --- |
| `schemas.py` | Constants, `ChatModelConfig`, request/response DTOs |
| `dependencies.py` | Provider validation, credential checking, `get_chat_model` |
| `routers.py` | `POST /module1/turn` |
| `services/graph_service.py` | Graph builder, runtime config, `run_turn` |
| `services/tools.py` | Arithmetic tools exposed to the model |
| `main.py` | CLI entry point |

## External docs

These links are kept here so module 1 remains self-contained:

- [LangGraph memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangChain tools and ToolNode](https://docs.langchain.com/oss/python/langchain/tools)
- [StateGraph reference](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)
- [tools_condition reference](https://reference.langchain.com/python/langgraph.prebuilt/tool_node/tools_condition)

## Tests

Unit tests cover provider configuration, arithmetic tools, graph wiring, router
validation, and CLI behavior. The live-LLM test is opt-in:

```bash
RUN_LIVE_LLM_TESTS=1 uv run pytest tests/integration/test_module1_live_llm.py
```
