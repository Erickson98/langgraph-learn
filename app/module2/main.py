from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.module2.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
    prepare_environment,
)
from app.module2.schemas import (
    DEFAULT_MEMORY_DB,
    DEFAULT_PROMPT,
    DEFAULT_SUMMARIZE_AFTER,
    DEFAULT_THREAD_ID,
)
from app.module2.services.graph_service import (
    build_graph,
    get_memory_db_path,
    get_summary,
    maybe_render_graph,
    run_turn,
)


def positive_int(value: str) -> int:
    """Parse an argparse value that must be greater than zero.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the module 2 CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run the module 2 LangGraph summarizing chatbot."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt to send to the graph.",
    )
    parser.add_argument(
        "--thread-id",
        default=DEFAULT_THREAD_ID,
        help=f"Conversation thread id used for SQLite state. Default: {DEFAULT_THREAD_ID}",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the process alive and chat across multiple turns with SQLite-backed memory.",
    )
    parser.add_argument(
        "--summarize-after",
        type=positive_int,
        default=DEFAULT_SUMMARIZE_AFTER,
        help=f"Summarize once message count is above this threshold. Default: {DEFAULT_SUMMARIZE_AFTER}",
    )
    parser.add_argument(
        "--memory-db",
        help=(
            "SQLite file used for persistent graph memory. "
            f"Defaults to MODULE2_MEMORY_DB or {DEFAULT_MEMORY_DB}."
        ),
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print the current summary after the turn finishes.",
    )
    parser.add_argument(
        "--print-mermaid",
        action="store_true",
        help="Print the graph Mermaid diagram to stdout.",
    )
    parser.add_argument(
        "--save-mermaid-png",
        help="Render the graph to a PNG file using draw_mermaid_png().",
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
    print("Commands: `/summary`, `/exit`, `/quit`")

    while True:
        try:
            prompt = input("You: ").strip()
        except EOFError:
            print()
            return 0

        if not prompt:
            continue

        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            return 0

        if prompt.lower() == "/summary":
            summary = get_summary(graph, thread_id)
            print(f"Summary: {summary or '<empty>'}")
            continue

        print(f"Assistant: {run_turn(graph, prompt, thread_id)}")


def main() -> int:
    """Run module 2 from the command line.

    Returns:
        Process exit code.
    """
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

    memory_db = get_memory_db_path(args.memory_db or settings.module2_memory_db)

    with SqliteSaver.from_conn_string(memory_db) as memory:
        graph = build_graph(
            memory,
            summarize_after=args.summarize_after,
            model=model_config.model,
            model_provider=model_config.model_provider,
            settings=settings,
        )
        maybe_render_graph(
            graph,
            print_mermaid=args.print_mermaid,
            save_mermaid_png=args.save_mermaid_png,
        )

        if args.interactive:
            return run_interactive(graph, args.thread_id)

        print(run_turn(graph, args.prompt, args.thread_id))

        if args.show_summary:
            summary = get_summary(graph, args.thread_id)
            print(f"\nSummary:\n{summary or '<empty>'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
