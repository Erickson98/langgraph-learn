# Module 1 Arithmetic Agent

Module 1 is the reference migrated agent. It demonstrates a small LangGraph assistant with arithmetic tools, model-agnostic LangChain initialization, CLI execution, and a FastAPI endpoint.

## User Stories

| User | Need | Outcome |
| --- | --- | --- |
| Developer | Run the module from the CLI | The agent can answer one prompt or run interactively. |
| API client | Submit a prompt over HTTP | The FastAPI router returns a typed response. |
| Tester | Verify behavior without provider calls | Tests patch the chat model and graph turn execution. |

## Design Decisions

| Decision | Reason |
| --- | --- |
| Use `init_chat_model` | Keeps the module provider-agnostic. |
| Keep graph logic in services | The CLI and HTTP layers share the same orchestration code. |
| Use Pydantic settings | Environment configuration is centralized and testable. |

