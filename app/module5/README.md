# Module 5 Memory Productivity Agent

Module 5 is a long-term memory assistant for profile facts, todos, and task-management preferences. It uses LangGraph with checkpointed conversation state and an in-memory store for long-term memory during the CLI session.

## Runtime

The module currently runs through the CLI. The LLM is configured through `LANGCHAIN_CHAT_MODEL` and `LANGCHAIN_MODEL_PROVIDER`, with provider credentials loaded from shared settings.

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

Module 5 has unit coverage for dependency resolution, CLI parsing, graph helpers, and memory formatting. Integration tests cover LangGraph wiring with patched chat models and patched Trustcall extraction without live provider calls.
