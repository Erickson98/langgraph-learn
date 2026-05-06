"""LangGraph orchestration service for the module 2 chatbot."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal

import anyio
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config.settings import Settings
from app.module2.dependencies import get_chat_model
from app.module2.schemas import (
    DEFAULT_MEMORY_DB,
    DEFAULT_MODEL,
    DEFAULT_SUMMARIZE_AFTER,
    Module2TurnResult,
)

SUMMARY_CONTEXT_PROMPT = (
    "Summary of the conversation so far:\n"
    "{summary}\n\n"
    "Use this summary as context for the latest user request."
)
CREATE_SUMMARY_PROMPT = "Create a concise summary of the conversation above."
EXTEND_SUMMARY_PROMPT = (
    "Current summary:\n"
    "{summary}\n\n"
    "Extend the summary using the latest messages above."
)


class ConversationState(MessagesState):
    """State carried by the summarizing chatbot graph."""

    summary: str


def build_graph(
    checkpointer: BaseCheckpointSaver,
    summarize_after: int = DEFAULT_SUMMARIZE_AFTER,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Build the summarizing chatbot graph.

    Args:
        checkpointer: LangGraph checkpointer used for thread persistence.
        summarize_after: Message-count threshold for summary compaction.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        Compiled LangGraph graph with the provided checkpointer.

    Raises:
        ValueError: If summarize_after is lower than one.
    """
    if summarize_after < 1:
        raise ValueError("summarize_after must be greater than zero.")

    llm = get_chat_model(
        model=model,
        model_provider=model_provider,
        settings=settings,
    )

    def call_model(state: ConversationState) -> dict[str, list[BaseMessage]]:
        summary = state.get("summary", "")

        if summary:
            messages = [
                SystemMessage(content=SUMMARY_CONTEXT_PROMPT.format(summary=summary))
            ] + state["messages"]
        else:
            messages = state["messages"]

        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(
        state: ConversationState,
    ) -> Literal["summarize_conversation", "__end__"]:
        if len(state["messages"]) > summarize_after:
            return "summarize_conversation"
        return END

    def summarize_conversation(
        state: ConversationState,
    ) -> dict[str, str | list[RemoveMessage]]:
        summary = state.get("summary", "")

        if summary:
            summary_message = EXTEND_SUMMARY_PROMPT.format(summary=summary)
        else:
            summary_message = CREATE_SUMMARY_PROMPT

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = llm.invoke(messages)

        delete_messages = [
            RemoveMessage(id=message.id) for message in state["messages"][:-2]
        ]
        return {"summary": response.content, "messages": delete_messages}

    workflow = StateGraph(ConversationState)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)
    return workflow.compile(checkpointer=checkpointer)


def build_state_reader_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Build a graph shell that can read checkpointed conversation state.

    Args:
        checkpointer: LangGraph checkpointer used for thread persistence.

    Returns:
        Compiled graph that exposes thread state without initializing an LLM.
    """

    def noop(_: ConversationState) -> dict[str, list[BaseMessage]]:
        return {}

    workflow = StateGraph(ConversationState)
    workflow.add_node("noop", noop)
    workflow.add_edge(START, "noop")
    workflow.add_edge("noop", END)
    return workflow.compile(checkpointer=checkpointer)


def build_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Build LangGraph runtime config for a conversation thread.

    Args:
        thread_id: Conversation thread identifier.

    Returns:
        LangGraph config dictionary.
    """
    return {"configurable": {"thread_id": thread_id}}


def get_memory_db_path(memory_db: str | Path) -> str:
    """Return a writable SQLite path for graph checkpointing.

    Args:
        memory_db: SQLite path from CLI, API, or service defaults.

    Returns:
        Expanded SQLite path as a string.
    """
    memory_path = Path(memory_db).expanduser()
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    return str(memory_path)


def run_turn(graph: CompiledStateGraph, prompt: str, thread_id: str) -> str:
    """Run one prompt through the graph.

    Args:
        graph: Compiled LangGraph graph.
        prompt: User prompt.
        thread_id: Conversation thread identifier.

    Returns:
        Assistant response text.
    """
    result = graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=build_config(thread_id),
    )
    return result["messages"][-1].content


def get_summary(graph: CompiledStateGraph, thread_id: str) -> str:
    """Read the current summary for a thread.

    Args:
        graph: Compiled LangGraph graph.
        thread_id: Conversation thread identifier.

    Returns:
        Current summary text, or an empty string when no summary exists.
    """
    snapshot = graph.get_state(build_config(thread_id))
    return snapshot.values.get("summary", "")


def maybe_render_graph(
    graph: CompiledStateGraph,
    *,
    print_mermaid: bool,
    save_mermaid_png: str | None,
) -> None:
    """Render the graph diagram when requested by the CLI.

    Args:
        graph: Compiled LangGraph graph.
        print_mermaid: Whether to print Mermaid syntax to stdout.
        save_mermaid_png: Optional output path for a rendered PNG.
    """
    if print_mermaid:
        print(graph.get_graph().draw_mermaid())

    if save_mermaid_png:
        graph.get_graph().draw_mermaid_png(output_file_path=save_mermaid_png)
        print(f"Saved graph PNG to {save_mermaid_png}")


def run_turn_with_sqlite(
    *,
    prompt: str,
    thread_id: str,
    summarize_after: int = DEFAULT_SUMMARIZE_AFTER,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module2TurnResult:
    """Run one graph turn with a SQLite checkpointer.

    Args:
        prompt: User prompt.
        thread_id: Conversation thread identifier.
        summarize_after: Message-count threshold for summary compaction.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Assistant response and the current summary for the thread.
    """
    memory_db_path = get_memory_db_path(memory_db)

    with SqliteSaver.from_conn_string(memory_db_path) as memory:
        graph = build_graph(
            memory,
            summarize_after=summarize_after,
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
        response = run_turn(graph, prompt, thread_id)
        summary = get_summary(graph, thread_id)

    return Module2TurnResult(response=response, summary=summary)


async def run_turn_with_sqlite_async(
    *,
    prompt: str,
    thread_id: str,
    summarize_after: int = DEFAULT_SUMMARIZE_AFTER,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module2TurnResult:
    """Run one graph turn with a SQLite checkpointer without blocking the event loop.

    Args:
        prompt: User prompt.
        thread_id: Conversation thread identifier.
        summarize_after: Message-count threshold for summary compaction.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Assistant response and the current summary for the thread.
    """
    return await anyio.to_thread.run_sync(
        partial(
            run_turn_with_sqlite,
            prompt=prompt,
            thread_id=thread_id,
            summarize_after=summarize_after,
            model=model,
            model_provider=model_provider,
            memory_db=memory_db,
            settings=settings,
        )
    )


def get_summary_with_sqlite(
    *,
    thread_id: str,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
) -> str:
    """Read a thread summary from a SQLite checkpointer.

    Args:
        thread_id: Conversation thread identifier.
        memory_db: SQLite checkpoint file path.

    Returns:
        Current summary text, or an empty string when no summary exists.
    """
    memory_db_path = get_memory_db_path(memory_db)

    with SqliteSaver.from_conn_string(memory_db_path) as memory:
        graph = build_state_reader_graph(memory)
        return get_summary(graph, thread_id)


async def get_summary_with_sqlite_async(
    *,
    thread_id: str,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
) -> str:
    """Read a thread summary from SQLite without blocking the event loop.

    Args:
        thread_id: Conversation thread identifier.
        memory_db: SQLite checkpoint file path.

    Returns:
        Current summary text, or an empty string when no summary exists.
    """
    return await anyio.to_thread.run_sync(
        partial(
            get_summary_with_sqlite,
            thread_id=thread_id,
            memory_db=memory_db,
        )
    )
