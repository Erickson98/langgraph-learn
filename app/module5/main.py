from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.config.settings import Settings
from app.logging import get_logger
from app.module5.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
    prepare_environment,
)
from app.module5.schemas import DEFAULT_USER_ID, ChatModelConfig, MemorySnapshot

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the module 5 CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run the module 5 long-term memory productivity assistant."
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"Long-term memory user id. Default: {DEFAULT_USER_ID}",
    )
    parser.add_argument(
        "--model",
        help="LangChain chat model name. Defaults to LANGCHAIN_CHAT_MODEL or gpt-4o-mini.",
    )
    parser.add_argument(
        "--model-provider",
        help="LangChain model provider. Defaults to LANGCHAIN_MODEL_PROVIDER or openai.",
    )
    return parser


def resolve_model_config(
    *,
    parser: argparse.ArgumentParser,
    settings: Settings,
    model: str | None,
    model_provider: str | None,
) -> ChatModelConfig:
    """Resolve CLI model configuration or stop with a parser error.

    Args:
        parser: CLI parser used to render configuration errors.
        settings: Runtime settings.
        model: Optional explicit model name.
        model_provider: Optional explicit model provider.

    Returns:
        Resolved model configuration.
    """
    try:
        model_config = get_chat_model_config(
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not has_model_credentials(model_config.model_provider, settings=settings):
        api_key_name = get_required_api_key_name(model_config.model_provider)
        parser.error(
            f"{api_key_name} is not set for provider '{model_config.model_provider}'. "
            "Add it to `.env` or export it in your shell."
        )

    return model_config


def print_memory_snapshot(memory: MemorySnapshot) -> None:
    """Print a readable memory snapshot for the CLI.

    Args:
        memory: Snapshot to print.
    """
    print("\n--- MEMORY SNAPSHOT ---")
    print("Profile:")
    print(memory.profile)
    print("\nToDos:")
    print(memory.todos)
    print("\nPreferences:")
    print(memory.instructions)
    print("-----------------------\n")


def run_chat(
    *,
    user_id: str,
    model: str,
    model_provider: str,
    settings: Settings,
) -> None:
    """Run the interactive module 5 CLI chat.

    Args:
        user_id: Long-term memory user id.
        model: Chat model name.
        model_provider: LangChain model provider.
        settings: Runtime settings.
    """
    asyncio.run(
        _run_chat_async(
            user_id=user_id,
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
    )


async def _run_chat_async(
    *,
    user_id: str,
    model: str,
    model_provider: str,
    settings: Settings,
) -> None:
    """Async implementation of the module 5 CLI chat loop.

    Args:
        user_id: Long-term memory user id.
        model: Chat model name.
        model_provider: LangChain model provider.
        settings: Runtime settings.
    """
    from app.module5.services.graph_service import build_graph, new_thread_id, run_turn
    from app.module5.services.sqlite_store import build_store

    store = build_store(settings.module5_memory_db)
    graph = build_graph(
        model=model,
        model_provider=model_provider,
        settings=settings,
        store=store,
    )
    thread_id = new_thread_id("module5")

    logger.info("Memory Agent started for user_id=%s", user_id)
    print("Type /memory to inspect stored memory, /quit to exit.\n")

    while True:
        user_input = (await asyncio.to_thread(input, "You: ")).strip()
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/memory":
            print_memory_snapshot(memory=await run_turn_memory(graph, user_id))
            continue

        result = await run_turn(
            graph,
            prompt=user_input,
            user_id=user_id,
            thread_id=thread_id,
        )
        print(f"Assistant: {result.response}\n")


async def run_turn_memory(graph: CompiledStateGraph, user_id: str) -> MemorySnapshot:
    """Read graph memory for the CLI.

    Args:
        graph: Compiled graph with an attached store.
        user_id: Long-term memory user id.

    Returns:
        Current memory snapshot.
    """
    from app.module5.services.graph_service import get_memory_snapshot

    return await get_memory_snapshot(graph.store, user_id)


def main() -> int:
    """Run module 5 from the command line.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    settings = prepare_environment()
    model_config = resolve_model_config(
        parser=parser,
        settings=settings,
        model=args.model,
        model_provider=args.model_provider,
    )
    run_chat(
        user_id=args.user_id,
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
