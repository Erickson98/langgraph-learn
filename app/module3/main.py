"""Module 3 CLI entry point.

Runs nine LangGraph demos from the command line, each highlighting a different
checkpointing or human-in-the-loop pattern:

- ``breakpoints``             — static node-level interrupt, approval via CLI
- ``interactive-breakpoints`` — multi-turn breakpoint loop
- ``dynamic-breakpoints``     — ``NodeInterrupt`` raised from inside a node
- ``edit-state``              — modify the last human message before the LLM runs
- ``human-feedback``          — dedicated feedback node that updates state
- ``time-travel``             — checkpoint replay and forked state edits
- ``streaming``               — token-level streaming with summarization
- ``interactive-chat``        — long-running summarizing REPL
- ``streaming-events``        — async ``astream_events`` with chunk printing

Usage examples::

    uv run python -m app.module3.main breakpoints --auto-approve
    uv run python -m app.module3.main time-travel --auto-approve
    uv run python -m app.module3.main streaming-events
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.config.settings import Settings
from app.module3.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
    prepare_environment,
)
from app.module3.schemas import (
    DEFAULT_DEMO,
    DEFAULT_MEMORY_DB,
    DEFAULT_PROMPT,
    DEFAULT_THREAD_ID,
    ChatModelConfig,
    Module3TurnResult,
)
from app.module3.services.graph_service import (
    approve_pending_turn,
    build_breakpoint_graph,
    build_config,
    build_dynamic_breakpoint_graph,
    build_streaming_graph,
    fork_checkpoint,
    get_graph_mermaid,
    get_memory_db_path,
    get_thread_state,
    list_thread_history,
    new_thread_id,
    replay_checkpoint,
    run_breakpoint_turn,
    save_graph_png,
)

DEMO_CHOICES = (
    "breakpoints",
    "interactive-breakpoints",
    "dynamic-breakpoints",
    "edit-state",
    "human-feedback",
    "time-travel",
    "streaming",
    "interactive-chat",
    "streaming-events",
    "all",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the module 3 CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run module 3 LangGraph checkpointing and breakpoint demos."
    )
    parser.add_argument(
        "demo",
        nargs="?",
        default=DEFAULT_DEMO,
        choices=DEMO_CHOICES,
        help=f"Demo to run. Default: {DEFAULT_DEMO}",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt for demos that start a graph turn.",
    )
    parser.add_argument(
        "--replacement-prompt",
        default="Multiply 5 and 3.",
        help="Replacement prompt used by the time-travel fork demo.",
    )
    parser.add_argument(
        "--feedback",
        default="Actually, multiply 10 and 10.",
        help="Feedback text used by the non-interactive human feedback demo.",
    )
    parser.add_argument(
        "--thread-id",
        default=DEFAULT_THREAD_ID,
        help=f"Conversation thread id. Default: {DEFAULT_THREAD_ID}",
    )
    parser.add_argument(
        "--memory-db",
        help=(
            "SQLite file used for persistent graph memory. "
            f"Defaults to MODULE3_MEMORY_DB or {DEFAULT_MEMORY_DB}."
        ),
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve paused tool calls without prompting.",
    )
    parser.add_argument(
        "--print-mermaid",
        action="store_true",
        help="Print the breakpoint graph Mermaid diagram to stdout.",
    )
    parser.add_argument(
        "--save-mermaid-png",
        help="Render the breakpoint graph to a PNG file using draw_mermaid_png().",
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


def print_header(title: str) -> None:
    """Print a visible CLI section header.

    Args:
        title: Header text.
    """
    print(f"\n{'=' * 18} {title} {'=' * 18}")


def print_message(message: BaseMessage) -> None:
    """Print one LangChain message using its pretty printer when available.

    Args:
        message: LangChain message to print.
    """
    if hasattr(message, "pretty_print"):
        message.pretty_print()
    else:
        print(message)


def print_last_message(messages: list[BaseMessage]) -> None:
    """Print the last message from a LangGraph message list.

    Args:
        messages: Messages from graph state.
    """
    if messages:
        print_message(messages[-1])


def print_turn_result(result: Module3TurnResult) -> None:
    """Print a module 3 breakpoint transition result.

    Args:
        result: Graph transition result.
    """
    if result.response:
        print(f"Assistant: {result.response}")

    print(f"Status: {result.status}")
    if result.pending_next:
        print(f"Pending next node: {result.pending_next}")

    if result.pending_tool_calls:
        print("Pending tool calls:")
        for index, tool_call in enumerate(result.pending_tool_calls, start=1):
            print(f"{index}. {tool_call.name} args={tool_call.args}")


def render_graph_if_requested(
    graph: Any,
    *,
    print_mermaid: bool,
    save_mermaid_png: str | None,
) -> None:
    """Render graph diagrams for CLI requests.

    Args:
        graph: Compiled LangGraph graph.
        print_mermaid: Whether to print Mermaid syntax.
        save_mermaid_png: Optional PNG output path.
    """
    if print_mermaid:
        print(get_graph_mermaid(graph))

    if save_mermaid_png:
        save_graph_png(graph, save_mermaid_png)
        print(f"Saved graph PNG to {save_mermaid_png}")


def demo_breakpoints(
    *,
    memory: Any,
    prompt: str,
    thread_id: str,
    model_config: ChatModelConfig,
    settings: Settings,
    auto_approve: bool,
    print_mermaid: bool = False,
    save_mermaid_png: str | None = None,
) -> None:
    """Run the static breakpoint demo.

    Args:
        memory: LangGraph checkpointer.
        prompt: User prompt.
        thread_id: Conversation thread identifier.
        model_config: Resolved model configuration.
        settings: Runtime settings.
        auto_approve: Whether to resume without prompting.
        print_mermaid: Whether to print Mermaid syntax.
        save_mermaid_png: Optional PNG output path.
    """
    print_header("BREAKPOINTS")
    graph = build_breakpoint_graph(
        memory,
        interrupt_before=["tools"],
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    render_graph_if_requested(
        graph,
        print_mermaid=print_mermaid,
        save_mermaid_png=save_mermaid_png,
    )

    result = run_breakpoint_turn(graph, prompt, thread_id)
    print_turn_result(result)

    if result.status != "paused":
        return

    approved = "yes"
    if not auto_approve:
        approved = input("Do you want to call the tool? [yes/no]: ").strip().lower()

    if approved in {"y", "yes"}:
        print_turn_result(approve_pending_turn(graph, thread_id))
    else:
        print("Operation cancelled by user.")


def demo_interactive_breakpoints(
    *,
    memory: Any,
    thread_id: str,
    model_config: ChatModelConfig,
    settings: Settings,
) -> None:
    """Run the interactive breakpoint approval loop.

    Args:
        memory: LangGraph checkpointer.
        thread_id: Starting conversation thread identifier.
        model_config: Resolved model configuration.
        settings: Runtime settings.
    """
    print_header("INTERACTIVE BREAKPOINTS")
    graph = build_breakpoint_graph(
        memory,
        interrupt_before=["tools"],
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    base_thread_id = thread_id
    current_thread_id = thread_id

    print(f"Interactive mode. thread_id={current_thread_id}")
    print("Available tools: add, multiply, subtract, divide")
    print("Commands: `/approve`, `/deny`, `/state`, `/thread`, `/new`, `/quit`")

    while True:
        state = get_thread_state(graph, current_thread_id)
        pending = state.status == "paused"

        try:
            prompt = input("You: ").strip()
        except EOFError:
            print()
            return

        if not prompt:
            continue

        lowered = prompt.lower()

        if lowered in {"/quit", "/exit", "quit", "exit"}:
            return

        if lowered == "/thread":
            print(f"Current thread_id: {current_thread_id}")
            continue

        if lowered == "/new":
            current_thread_id = new_thread_id(base_thread_id)
            print(f"Started a new thread: {current_thread_id}")
            continue

        if lowered == "/state":
            print(f"Current thread_id: {current_thread_id}")
            print(f"Status: {state.status}")
            print(f"Pending next node: {state.pending_next or '<none>'}")
            for tool_call in state.pending_tool_calls:
                print(f"{tool_call.name} args={tool_call.args}")
            continue

        if lowered == "/approve":
            if not pending:
                print("No pending tool call in the current thread.")
                continue
            print_turn_result(approve_pending_turn(graph, current_thread_id))
            continue

        if lowered == "/deny":
            if not pending:
                print("No pending tool call in the current thread.")
                continue
            current_thread_id = new_thread_id(base_thread_id)
            print("Pending tool call discarded.")
            print(f"Started a new thread: {current_thread_id}")
            continue

        if pending:
            print("This thread is paused before a tool call.")
            print(
                "Use `/approve` to run the tool or `/deny` to discard this paused thread."
            )
            continue

        print_turn_result(run_breakpoint_turn(graph, prompt, current_thread_id))


def demo_dynamic_breakpoints() -> None:
    """Run the dynamic breakpoint demo."""
    print_header("DYNAMIC BREAKPOINTS")
    graph = build_dynamic_breakpoint_graph()
    thread_id = new_thread_id("dynamic")
    config = build_config(thread_id)

    print("Running with an input that should interrupt...")
    for event in graph.stream({"input": "hello world"}, config, stream_mode="values"):
        print(event)

    state = graph.get_state(config)
    print("Pending next node:", state.next)

    print("Updating state so the graph can continue...")
    graph.update_state(config, {"input": "hi"})

    for event in graph.stream(None, config, stream_mode="values"):
        print(event)


def demo_edit_state(
    *,
    memory: Any,
    prompt: str,
    replacement_prompt: str,
    thread_id: str,
    model_config: ChatModelConfig,
    settings: Settings,
) -> None:
    """Run the state editing demo.

    Args:
        memory: LangGraph checkpointer.
        prompt: Initial user prompt.
        replacement_prompt: Prompt used to update graph state before assistant runs.
        thread_id: Conversation thread identifier.
        model_config: Resolved model configuration.
        settings: Runtime settings.
    """
    print_header("EDIT STATE")
    graph = build_breakpoint_graph(
        memory,
        interrupt_before=["assistant"],
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    config = build_config(thread_id)
    graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    print("Updating the message before the assistant runs...")
    graph.update_state(
        config,
        {"messages": [HumanMessage(content=replacement_prompt)]},
    )

    for message in graph.get_state(config).values.get("messages", []):
        print_message(message)

    graph.invoke(None, config=config)
    print_last_message(graph.get_state(config).values.get("messages", []))


def demo_human_feedback(
    *,
    memory: Any,
    prompt: str,
    feedback: str,
    thread_id: str,
    model_config: ChatModelConfig,
    settings: Settings,
    auto_approve: bool,
) -> None:
    """Run the human feedback state update demo.

    Args:
        memory: LangGraph checkpointer.
        prompt: Initial user prompt.
        feedback: Feedback text used when auto-approved.
        thread_id: Conversation thread identifier.
        model_config: Resolved model configuration.
        settings: Runtime settings.
        auto_approve: Whether to use the provided feedback without prompting.
    """
    print_header("HUMAN FEEDBACK")
    graph = build_breakpoint_graph(
        memory,
        interrupt_before=["human_feedback"],
        with_human_feedback=True,
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    config = build_config(thread_id)
    graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    resolved_feedback = feedback
    if not auto_approve:
        resolved_feedback = input("Tell me how you want to update the state: ").strip()

    if resolved_feedback:
        graph.update_state(
            config,
            {"messages": [HumanMessage(content=resolved_feedback)]},
            as_node="human_feedback",
        )

    graph.invoke(None, config=config)
    print_last_message(graph.get_state(config).values.get("messages", []))


def demo_time_travel(
    *,
    memory: Any,
    prompt: str,
    replacement_prompt: str,
    thread_id: str,
    model_config: ChatModelConfig,
    settings: Settings,
) -> None:
    """Run replay and fork operations against checkpoint history.

    Args:
        memory: LangGraph checkpointer.
        prompt: Initial user prompt.
        replacement_prompt: Prompt used for the forked branch.
        thread_id: Conversation thread identifier.
        model_config: Resolved model configuration.
        settings: Runtime settings.
    """
    print_header("TIME TRAVEL")
    graph = build_breakpoint_graph(
        memory,
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    run_breakpoint_turn(graph, prompt, thread_id)

    history = list_thread_history(graph, thread_id)
    print("Checkpoint count:", len(history))
    replayable = next((entry for entry in history if entry.can_replay), None)
    forkable = next((entry for entry in history if entry.can_fork), None)

    if replayable:
        print("Replaying from checkpoint:", replayable.checkpoint_id)
        print_turn_result(replay_checkpoint(graph, thread_id, replayable.checkpoint_id))

    if forkable:
        print("Forking from checkpoint:", forkable.checkpoint_id)
        print_turn_result(
            fork_checkpoint(
                graph,
                thread_id,
                forkable.checkpoint_id,
                replacement_prompt,
            )
        )


def demo_streaming(
    *,
    model_config: ChatModelConfig,
    settings: Settings,
) -> None:
    """Run the streaming summarization demo.

    Args:
        model_config: Resolved model configuration.
        settings: Runtime settings.
    """
    print_header("STREAMING + SUMMARIZATION")
    graph = build_streaming_graph(
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    thread_id = new_thread_id("streaming")
    turns = [
        "Hi! I'm Lance.",
        "I live in San Francisco.",
        "I work on LangGraph.",
        "My favorite sport is basketball.",
    ]

    for turn in turns:
        print(f"\nUSER: {turn}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=turn)]},
            build_config(thread_id),
        )
        print_last_message(result["messages"])

        state = graph.get_state(build_config(thread_id))
        summary = state.values.get("summary", "")
        print("Summary:", summary or "<none yet>")


def demo_interactive_chat(
    *,
    thread_id: str,
    model_config: ChatModelConfig,
    settings: Settings,
) -> None:
    """Run an interactive streaming chat loop.

    Args:
        thread_id: Conversation thread identifier.
        model_config: Resolved model configuration.
        settings: Runtime settings.
    """
    print_header("INTERACTIVE CHAT")
    graph = build_streaming_graph(
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    config = build_config(thread_id)

    print(f"Interactive mode. thread_id={thread_id}")
    print("Commands: `/summary`, `/messages`, `/exit`, `/quit`")

    while True:
        try:
            prompt = input("You: ").strip()
        except EOFError:
            print()
            return

        if not prompt:
            continue

        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            return

        if prompt.lower() == "/summary":
            state = graph.get_state(config)
            summary = state.values.get("summary", "")
            print(f"Summary: {summary or '<none yet>'}")
            continue

        if prompt.lower() == "/messages":
            messages = graph.get_state(config).values.get("messages", [])
            if not messages:
                print("Messages: <empty>")
                continue
            for message in messages:
                print_message(message)
            continue

        result = graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        print_last_message(result["messages"])


async def demo_streaming_events(
    *,
    model_config: ChatModelConfig,
    settings: Settings,
) -> None:
    """Stream model chunks from LangGraph events.

    Args:
        model_config: Resolved model configuration.
        settings: Runtime settings.
    """
    print_header("STREAMING EVENTS")
    graph = build_streaming_graph(
        model=model_config.model,
        model_provider=model_config.model_provider,
        settings=settings,
    )
    config = build_config(new_thread_id("stream-events"))
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")

    async for event in graph.astream_events(
        {"messages": [input_message]},
        config,
        version="v2",
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node") == "conversation"
        ):
            chunk = event["data"]["chunk"].content
            if chunk:
                print(chunk, end="", flush=True)
    print()


def run_llm_demo(args: argparse.Namespace, settings: Settings) -> None:
    """Run the selected LLM-backed demo.

    Args:
        args: Parsed CLI arguments.
        settings: Runtime settings.
    """
    parser = build_parser()
    model_config = resolve_model_config(
        parser=parser,
        settings=settings,
        model=args.model,
        model_provider=args.model_provider,
    )
    memory_db = get_memory_db_path(args.memory_db or settings.module3_memory_db)

    if args.demo in {"streaming", "interactive-chat", "streaming-events"}:
        if args.demo == "streaming":
            demo_streaming(model_config=model_config, settings=settings)
        elif args.demo == "interactive-chat":
            demo_interactive_chat(
                thread_id=args.thread_id,
                model_config=model_config,
                settings=settings,
            )
        else:
            asyncio.run(
                demo_streaming_events(model_config=model_config, settings=settings)
            )
        return

    with SqliteSaver.from_conn_string(memory_db) as memory:
        if args.demo == "breakpoints":
            demo_breakpoints(
                memory=memory,
                prompt=args.prompt,
                thread_id=args.thread_id,
                model_config=model_config,
                settings=settings,
                auto_approve=args.auto_approve,
                print_mermaid=args.print_mermaid,
                save_mermaid_png=args.save_mermaid_png,
            )
        elif args.demo == "interactive-breakpoints":
            demo_interactive_breakpoints(
                memory=memory,
                thread_id=args.thread_id,
                model_config=model_config,
                settings=settings,
            )
        elif args.demo == "edit-state":
            demo_edit_state(
                memory=memory,
                prompt=args.prompt,
                replacement_prompt=args.replacement_prompt,
                thread_id=args.thread_id,
                model_config=model_config,
                settings=settings,
            )
        elif args.demo == "human-feedback":
            demo_human_feedback(
                memory=memory,
                prompt=args.prompt,
                feedback=args.feedback,
                thread_id=args.thread_id,
                model_config=model_config,
                settings=settings,
                auto_approve=args.auto_approve,
            )
        elif args.demo == "time-travel":
            demo_time_travel(
                memory=memory,
                prompt=args.prompt,
                replacement_prompt=args.replacement_prompt,
                thread_id=args.thread_id,
                model_config=model_config,
                settings=settings,
            )


def run_all(args: argparse.Namespace, settings: Settings) -> None:
    """Run the non-interactive module 3 demo set.

    Args:
        args: Parsed CLI arguments.
        settings: Runtime settings.
    """
    demo_dynamic_breakpoints()

    for demo in [
        "breakpoints",
        "edit-state",
        "human-feedback",
        "time-travel",
        "streaming",
        "streaming-events",
    ]:
        args.demo = demo
        args.auto_approve = True
        try:
            run_llm_demo(args, settings)
        except RuntimeError as exc:
            print(f"Skipped {demo}: {exc}")


def main() -> int:
    """Run module 3 from the command line.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    settings = prepare_environment()

    if args.demo == "dynamic-breakpoints":
        demo_dynamic_breakpoints()
    elif args.demo == "all":
        run_all(args, settings)
    else:
        run_llm_demo(args, settings)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
