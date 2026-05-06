from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.module1.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
    prepare_environment,
)
from app.module1.schemas import DEFAULT_PROMPT, DEFAULT_THREAD_ID
from app.module1.services.graph_service import (
    build_graph,
    run_turn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the module 1 LangGraph arithmetic assistant."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt to send to the graph.",
    )
    parser.add_argument(
        "--thread-id",
        default=DEFAULT_THREAD_ID,
        help=f"Conversation thread id used for MemorySaver state. Default: {DEFAULT_THREAD_ID}",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the process alive and chat across multiple turns with MemorySaver.",
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


def run_interactive(graph: Any, thread_id: str) -> int:
    """Run the graph in a terminal chat loop.

    Args:
        graph: Compiled LangGraph graph.
        thread_id: Conversation thread identifier.

    Returns:
        Process exit code.
    """
    print(f"Interactive mode. thread_id={thread_id}")
    print("Type `exit` or `quit` to stop.")

    while True:
        try:
            prompt = input("You: ").strip()
        except EOFError:
            print()
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return 0

        print(f"Assistant: {run_turn(graph, prompt, thread_id)}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    settings = prepare_environment()

    try:
        model_config = get_chat_model_config(
            model=args.model,
            model_provider=args.model_provider,
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

    graph = build_graph(
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )

    if args.interactive:
        return run_interactive(graph, args.thread_id)

    print(run_turn(graph, args.prompt, args.thread_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
