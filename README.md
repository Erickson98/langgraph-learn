# LangGraph Learn

Small LangGraph learning modules, organized as package-style modules instead of single-file demos.

## Module Folder Pattern

Each migrated module is treated as its own agent workstream under `app/`. A module should keep `main.py` as the CLI entrypoint and move implementation into focused layers:

```text
app/
|-- config/
|   `-- settings.py      # Pydantic BaseSettings for shared runtime config
`-- moduleN/
    |-- __init__.py
    |-- main.py          # CLI parsing and process entrypoint only
    |-- dependencies.py  # model and dependency setup
    |-- schemas.py       # shared constants, typed config, and DTOs
    `-- services/        # graph orchestration and business logic
```

Rules:

- `main.py` should wire dependencies and call services, not define graph internals or tools.
- `services/` owns LangGraph construction, tool registries, and runtime behavior.
- `dependencies.py` should use LangChain abstractions, such as `init_chat_model`, rather than vendor-specific chat classes.
- Runtime settings should live in `app/config/settings.py` and be loaded through Pydantic `BaseSettings`.
- Tests should cover pure service logic, CLI behavior, graph wiring with patched models, and any opt-in live provider checks.
- Add folders such as `utils/` or `repositories/` only when the module actually needs them.

## Documentation

- [Project context](docs/PROJECT.md)
- [Module 1 use case](docs/USE-CASE-MODULE1.md)
- [Module 3 use case](docs/USE-CASE-MODULE3.md)
- [Module 4 use case](docs/USE-CASE-MODULE4.md)
- [Module 5 use case](docs/USE-CASE-MODULE5.md)
- [Wiki index](wiki/index.md)

## Modules

| Module | Purpose | Runtime |
| --- | --- | --- |
| [Module 1](app/module1/README.md) | Arithmetic LangGraph agent with deterministic tools and FastAPI support. | API and CLI |
| [Module 2](app/module2/README.md) | SQLite-backed chatbot that summarizes older conversation turns. | API and CLI |
| [Module 3](app/module3/README.md) | Human-in-the-loop, checkpoint, state editing, replay, and streaming demos. | API and CLI |
| [Module 4](app/module4/README.md) | Research brief generator with section planning and retrieval. | API and CLI |
| [Module 5](app/module5/README.md) | Long-term memory productivity assistant for profile, todos, and preferences. | CLI |

## Tests

Install dependencies from `pyproject.toml`:

```bash
uv sync --all-groups
```

Create local environment values when you need live provider calls:

```bash
cp .env.example .env
```

Edit `.env` and set the provider credentials you need:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=
TAVILY_API_KEY=
LANGCHAIN_CHAT_MODEL=gpt-4o-mini
LANGCHAIN_MODEL_PROVIDER=openai
RUN_LIVE_LLM_TESTS=
LOG_LEVEL=INFO
MODULE2_MEMORY_DB=data/module2.sqlite
MODULE3_MEMORY_DB=data/module3.sqlite
```

Run the test suite:

```bash
uv run pytest
```

Run the FastAPI app:

```bash
uv run uvicorn app.main:app --reload
```

OpenAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Run a module 1 turn through the API:

```bash
curl -X POST http://127.0.0.1:8000/module1/turn \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2 + 2?",
    "thread_id": "manual-test",
    "model": "gpt-4o-mini",
    "model_provider": "openai"
  }'
```

Successful response:

```json
{
  "response": "4",
  "thread_id": "manual-test",
  "model": "gpt-4o-mini",
  "model_provider": "openai"
}
```

Run a module 2 turn through the API:

```bash
curl -X POST http://127.0.0.1:8000/module2/turn \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "My favorite number is 7. Please remember it.",
    "thread_id": "manual-module2",
    "summarize_after": 6,
    "model": "gpt-4o-mini",
    "model_provider": "openai"
  }'
```

Successful response:

```json
{
  "response": "I will remember that your favorite number is 7.",
  "summary": "",
  "thread_id": "manual-module2",
  "summarize_after": 6,
  "model": "gpt-4o-mini",
  "model_provider": "openai"
}
```

Read the current module 2 summary for a thread:

```bash
curl "http://127.0.0.1:8000/module2/summary?thread_id=manual-module2"
```

Run a module 3 breakpoint turn through the API:

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

Approve a paused module 3 tool call:

```bash
curl -X POST http://127.0.0.1:8000/module3/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "manual-module3"}'
```

Inspect module 3 state and checkpoint history:

```bash
curl "http://127.0.0.1:8000/module3/state?thread_id=manual-module3"
curl "http://127.0.0.1:8000/module3/history?thread_id=manual-module3"
```

Generate a module 4 research brief through the API:

```bash
curl -X POST http://127.0.0.1:8000/module4/brief \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "LangGraph for production support agents",
    "audience": "engineering leadership",
    "max_sections": 2,
    "include_wikipedia": true,
    "include_web": false,
    "model": "gpt-4o-mini",
    "model_provider": "openai"
  }'
```

Error responses use this shape:

```json
{
  "error": {
    "code": "missing_model_credentials",
    "message": "OPENAI_API_KEY is not set for provider 'openai'."
  }
}
```

Live LLM integration tests are opt-in because they make provider API calls:

```bash
RUN_LIVE_LLM_TESTS=1 uv run pytest \
  tests/integration/test_module1_live_llm.py \
  tests/integration/test_module2_live_llm.py \
  tests/integration/test_module4_live_llm.py
```

## Docker

Docker Compose reads `.env` automatically for variable interpolation, and `docker-compose.yml` only forwards the provider variables the app needs. Normal Docker commands do not need `--env-file`.

Avoid sharing `docker compose config` output when provider keys are set in your shell or `.env`; Compose prints resolved allowlisted environment values.

Build the dev/test image used by Compose:

```bash
docker compose build
```

Run the FastAPI app in Docker:

```bash
docker compose up
```

Then test the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Module 2 and module 3 SQLite memory are persisted through the `module_data` Docker volume mounted at `/app/data`. The default database paths are `data/module2.sqlite` and `data/module3.sqlite`, and you can override them with `MODULE2_MEMORY_DB` and `MODULE3_MEMORY_DB`.

Build the smaller runtime image without dev tools:

```bash
docker build --target runtime -t langgraph-learn:runtime .
```

Run a module CLI by overriding the default API command:

```bash
docker compose run --rm app python -m app.module1.main --help
docker compose run --rm app python -m app.module1.main --prompt "What is 2 + 2?"
docker compose run --rm app python -m app.module2.main --prompt "Remember that my favorite number is 7."
docker compose run --rm app python -m app.module3.main breakpoints --auto-approve
docker compose run --rm app python -m app.module4.main "LangGraph for production support agents" --sections 2 --no-web
docker compose run --rm app python -m app.module5.main --user-id demo-user
```

Run tests in the container:

```bash
docker compose run --rm app python -m pytest
```

Run the opt-in live LLM test:

```bash
docker compose run --rm -e RUN_LIVE_LLM_TESTS=1 app python -m pytest \
  tests/integration/test_module1_live_llm.py \
  tests/integration/test_module2_live_llm.py \
  tests/integration/test_module4_live_llm.py
```

If you want to pass a different env file manually, put `--env-file` before `run`:

```bash
docker compose --env-file .env run --rm app python -m app.module1.main --prompt "What is 2 + 2?"
```

The Docker setup is intentionally root-level. All modules share `pyproject.toml`, so one Dockerfile can run every `app.moduleN.main` entrypoint without duplicating dependency setup. Compose targets the `dev` stage so pytest is available; the `runtime` stage omits dev-only packages such as notebooks and CLI tooling.
