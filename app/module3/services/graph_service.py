"""LangGraph orchestration helpers for module 3."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import anyio
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import StateSnapshot
from typing_extensions import TypedDict

from app.config.settings import Settings
from app.module3.dependencies import get_chat_model
from app.module3.schemas import (
    DEFAULT_MEMORY_DB,
    DEFAULT_MODEL,
    MessageView,
    Module3HistoryEntry,
    Module3StateResult,
    Module3TurnResult,
    PendingToolCall,
)
from app.module3.services.tools import ARITHMETIC_TOOLS

ARITHMETIC_SYSTEM = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


class DynamicState(TypedDict):
    """State for the dynamic breakpoint demo."""

    input: str


class ConversationState(MessagesState):
    """State carried by the streaming conversation demo."""

    summary: str


def new_thread_id(prefix: str) -> str:
    """Return a short generated thread identifier.

    Args:
        prefix: Prefix used to group related demo threads.

    Returns:
        Generated thread id.
    """
    return f"{prefix}-{uuid4().hex[:8]}"


def build_breakpoint_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    interrupt_before: list[str] | None = None,
    with_human_feedback: bool = False,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Build the arithmetic graph used for static breakpoints and replay.

    Args:
        checkpointer: LangGraph checkpointer used for thread persistence.
        interrupt_before: Optional node names where execution should pause.
        with_human_feedback: Whether to insert a feedback node before the assistant.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        Compiled graph with breakpoint support.
    """
    llm = get_chat_model(
        model=model,
        model_provider=model_provider,
        settings=settings,
    )
    llm_with_tools = llm.bind_tools(ARITHMETIC_TOOLS)

    def assistant(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = llm_with_tools.invoke([ARITHMETIC_SYSTEM] + state["messages"])
        return {"messages": [response]}

    def human_feedback(_: MessagesState) -> dict[str, list[BaseMessage]]:
        return {}

    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", assistant)
    workflow.add_node("tools", ToolNode(ARITHMETIC_TOOLS))

    if with_human_feedback:
        workflow.add_node("human_feedback", human_feedback)
        workflow.add_edge(START, "human_feedback")
        workflow.add_edge("human_feedback", "assistant")
        workflow.add_edge("tools", "human_feedback")
    else:
        workflow.add_edge(START, "assistant")
        workflow.add_edge("tools", "assistant")

    workflow.add_conditional_edges("assistant", tools_condition)
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )


def build_dynamic_breakpoint_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the graph used for the dynamic breakpoint demo.

    Args:
        checkpointer: Optional checkpointer override.

    Returns:
        Compiled graph that may raise a ``NodeInterrupt``.
    """
    saver = checkpointer or MemorySaver()

    def step_1(state: DynamicState) -> DynamicState:
        return state

    def step_2(state: DynamicState) -> DynamicState:
        if len(state["input"]) > 5:
            raise NodeInterrupt(
                f"Received input that is longer than 5 characters: {state['input']}"
            )
        return state

    def step_3(state: DynamicState) -> DynamicState:
        return state

    workflow = StateGraph(DynamicState)
    workflow.add_node("step_1", step_1)
    workflow.add_node("step_2", step_2)
    workflow.add_node("step_3", step_3)
    workflow.add_edge(START, "step_1")
    workflow.add_edge("step_1", "step_2")
    workflow.add_edge("step_2", "step_3")
    workflow.add_edge("step_3", END)
    return workflow.compile(checkpointer=saver)


def build_streaming_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Build the streaming conversation demo graph.

    Args:
        checkpointer: Optional checkpointer override.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        Compiled graph for the streaming demo.
    """
    saver = checkpointer or MemorySaver()
    llm = get_chat_model(
        model=model,
        model_provider=model_provider,
        settings=settings,
    )

    def call_model(state: ConversationState) -> dict[str, list[BaseMessage]]:
        summary = state.get("summary", "")
        messages = state["messages"]

        if summary:
            messages = [
                SystemMessage(content=f"Summary of conversation earlier: {summary}")
            ] + messages

        response = llm.invoke(messages)
        return {"messages": [response]}

    def summarize_conversation(
        state: ConversationState,
    ) -> dict[str, str | list[RemoveMessage]]:
        summary = state.get("summary", "")

        if summary:
            prompt = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            prompt = "Create a summary of the conversation above:"

        response = llm.invoke(state["messages"] + [HumanMessage(content=prompt)])
        delete_messages = [
            RemoveMessage(id=message.id) for message in state["messages"][:-2]
        ]
        return {"summary": response.content, "messages": delete_messages}

    def should_continue(
        state: ConversationState,
    ) -> Literal["summarize_conversation", "__end__"]:
        if len(state["messages"]) > 6:
            return "summarize_conversation"
        return END

    workflow = StateGraph(ConversationState)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)
    return workflow.compile(checkpointer=saver)


def build_state_reader_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Build a graph shell that can read checkpointed message state.

    Args:
        checkpointer: LangGraph checkpointer used for thread persistence.

    Returns:
        Compiled graph that exposes stored thread state without an LLM.
    """

    def noop(_: MessagesState) -> dict[str, list[BaseMessage]]:
        return {}

    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", noop)
    workflow.add_node("tools", noop)
    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges("assistant", tools_condition)
    workflow.add_edge("tools", "assistant")
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )


def build_config(thread_id: str) -> RunnableConfig:
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


def message_content_to_text(content: Any) -> str:
    """Convert a LangChain message content payload into plain text.

    Args:
        content: Raw message content.

    Returns:
        Human-readable text representation.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    return str(content)


def message_to_view(message: BaseMessage) -> MessageView:
    """Convert a LangChain message into a serializable view.

    Args:
        message: LangChain message instance.

    Returns:
        Simplified message details.
    """
    return MessageView(
        id=getattr(message, "id", None),
        type=message.type,
        content=message_content_to_text(getattr(message, "content", "")),
    )


def extract_pending_tool_calls(snapshot: StateSnapshot) -> list[PendingToolCall]:
    """Return pending tool calls from the latest assistant message.

    Args:
        snapshot: LangGraph state snapshot.

    Returns:
        Structured pending tool calls.
    """
    messages = snapshot.values.get("messages", [])
    if not messages:
        return []

    tool_calls = getattr(messages[-1], "tool_calls", None) or []
    return [
        PendingToolCall(
            id=call.get("id"),
            name=call.get("name", ""),
            args=call.get("args", {}),
        )
        for call in tool_calls
    ]


def build_turn_result(snapshot: StateSnapshot) -> Module3TurnResult:
    """Convert a LangGraph snapshot into a turn result.

    Args:
        snapshot: LangGraph state snapshot.

    Returns:
        Simplified turn result for CLI and API use.
    """
    messages = snapshot.values.get("messages", [])
    last_message = messages[-1] if messages else AIMessage(content="")
    status = "paused" if snapshot.next else "completed"
    return Module3TurnResult(
        status=status,
        response=message_content_to_text(last_message.content),
        pending_next=tuple(snapshot.next),
        pending_tool_calls=extract_pending_tool_calls(snapshot),
        message_count=len(messages),
    )


def build_state_result(snapshot: StateSnapshot) -> Module3StateResult:
    """Convert a LangGraph snapshot into a thread state view.

    Args:
        snapshot: LangGraph state snapshot.

    Returns:
        Simplified thread state for inspection endpoints.
    """
    messages = snapshot.values.get("messages", [])
    status = "paused" if snapshot.next else "idle"
    return Module3StateResult(
        status=status,
        pending_next=tuple(snapshot.next),
        pending_tool_calls=extract_pending_tool_calls(snapshot),
        message_count=len(messages),
        messages=[message_to_view(message) for message in messages],
    )


def build_history_entry(snapshot: StateSnapshot) -> Module3HistoryEntry:
    """Convert one checkpoint snapshot into API-friendly history metadata.

    Args:
        snapshot: LangGraph state snapshot.

    Returns:
        Simplified checkpoint history entry.
    """
    configurable = snapshot.config.get("configurable", {})
    next_nodes = tuple(snapshot.next)
    messages = snapshot.values.get("messages", [])
    metadata = snapshot.metadata or {}
    return Module3HistoryEntry(
        checkpoint_id=str(configurable.get("checkpoint_id", "")),
        next_nodes=next_nodes,
        source=metadata.get("source"),
        step=metadata.get("step"),
        message_count=len(messages),
        can_replay=bool(next_nodes),
        can_fork="assistant" in next_nodes
        and any(isinstance(message, HumanMessage) for message in messages),
    )


def get_thread_state(
    graph: CompiledStateGraph,
    thread_id: str,
) -> Module3StateResult:
    """Read the current state for a thread.

    Args:
        graph: Compiled LangGraph graph.
        thread_id: Conversation thread identifier.

    Returns:
        Current thread state.
    """
    snapshot = graph.get_state(build_config(thread_id))
    return build_state_result(snapshot)


def list_thread_history(
    graph: CompiledStateGraph,
    thread_id: str,
) -> list[Module3HistoryEntry]:
    """List stored checkpoints for a thread.

    Args:
        graph: Compiled LangGraph graph.
        thread_id: Conversation thread identifier.

    Returns:
        Checkpoint history ordered by LangGraph.
    """
    return [
        build_history_entry(snapshot)
        for snapshot in graph.get_state_history(build_config(thread_id))
    ]


def get_history_snapshot(
    graph: CompiledStateGraph,
    thread_id: str,
    checkpoint_id: str,
) -> StateSnapshot:
    """Return one checkpoint snapshot from a thread history.

    Args:
        graph: Compiled LangGraph graph.
        thread_id: Conversation thread identifier.
        checkpoint_id: Checkpoint identifier from history.

    Returns:
        Matching history snapshot.

    Raises:
        ValueError: If the checkpoint cannot be found.
    """
    for snapshot in graph.get_state_history(build_config(thread_id)):
        entry = build_history_entry(snapshot)
        if entry.checkpoint_id == checkpoint_id:
            return snapshot

    raise ValueError(
        f"Unknown checkpoint_id '{checkpoint_id}' for thread '{thread_id}'."
    )


def run_breakpoint_turn(
    graph: CompiledStateGraph,
    prompt: str,
    thread_id: str,
) -> Module3TurnResult:
    """Run one prompt through the breakpoint graph.

    Args:
        graph: Compiled LangGraph graph.
        prompt: User prompt.
        thread_id: Conversation thread identifier.

    Returns:
        Result after the graph pauses or completes.
    """
    graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=build_config(thread_id),
    )
    snapshot = graph.get_state(build_config(thread_id))
    return build_turn_result(snapshot)


def approve_pending_turn(
    graph: CompiledStateGraph,
    thread_id: str,
) -> Module3TurnResult:
    """Resume a paused breakpoint graph and run the pending tool call.

    Args:
        graph: Compiled LangGraph graph.
        thread_id: Conversation thread identifier.

    Returns:
        Result after the resumed execution finishes or pauses again.
    """
    graph.invoke(None, config=build_config(thread_id))
    snapshot = graph.get_state(build_config(thread_id))
    return build_turn_result(snapshot)


def replay_checkpoint(
    graph: CompiledStateGraph,
    thread_id: str,
    checkpoint_id: str,
) -> Module3TurnResult:
    """Replay execution from an existing checkpoint.

    Args:
        graph: Compiled LangGraph graph.
        thread_id: Conversation thread identifier.
        checkpoint_id: Checkpoint identifier from history.

    Returns:
        Result after replay finishes or pauses again.
    """
    snapshot = get_history_snapshot(graph, thread_id, checkpoint_id)
    graph.invoke(None, config=snapshot.config)
    latest = graph.get_state(build_config(thread_id))
    return build_turn_result(latest)


def fork_checkpoint(
    graph: CompiledStateGraph,
    thread_id: str,
    checkpoint_id: str,
    replacement_prompt: str,
) -> Module3TurnResult:
    """Fork a historical checkpoint by replacing the latest human prompt.

    Args:
        graph: Compiled LangGraph graph.
        thread_id: Conversation thread identifier.
        checkpoint_id: Checkpoint identifier from history.
        replacement_prompt: New prompt that should replace the selected human message.

    Returns:
        Result after the forked branch runs until the next pause or completion.

    Raises:
        ValueError: If the checkpoint cannot be forked safely.
    """
    snapshot = get_history_snapshot(graph, thread_id, checkpoint_id)

    if "assistant" not in snapshot.next:
        raise ValueError(
            f"Checkpoint '{checkpoint_id}' cannot be forked because it does not resume at assistant."
        )

    messages = snapshot.values.get("messages", [])
    target_message = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    if target_message is None:
        raise ValueError(
            f"Checkpoint '{checkpoint_id}' has no human message to replace."
        )

    replacement = HumanMessage(
        content=replacement_prompt,
        id=target_message.id,
    )
    fork_config = graph.update_state(snapshot.config, {"messages": [replacement]})
    graph.invoke(None, config=fork_config)
    latest = graph.get_state(build_config(thread_id))
    return build_turn_result(latest)


def get_graph_mermaid(graph: CompiledStateGraph) -> str:
    """Return the Mermaid representation for a compiled graph.

    Args:
        graph: Compiled LangGraph graph.

    Returns:
        Mermaid graph definition.
    """
    return graph.get_graph().draw_mermaid()


def save_graph_png(graph: CompiledStateGraph, output_file_path: str) -> None:
    """Render a compiled graph to a PNG file.

    Args:
        graph: Compiled LangGraph graph.
        output_file_path: Destination PNG path.
    """
    graph.get_graph().draw_mermaid_png(output_file_path=output_file_path)


def run_turn_with_sqlite(
    *,
    prompt: str,
    thread_id: str,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module3TurnResult:
    """Run one breakpoint graph turn with a SQLite checkpointer.

    Args:
        prompt: User prompt.
        thread_id: Conversation thread identifier.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Result after the graph pauses or completes.
    """
    memory_db_path = get_memory_db_path(memory_db)

    with SqliteSaver.from_conn_string(memory_db_path) as memory:
        graph = build_breakpoint_graph(
            memory,
            interrupt_before=["tools"],
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
        return run_breakpoint_turn(graph, prompt, thread_id)


def approve_pending_turn_with_sqlite(
    *,
    thread_id: str,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module3TurnResult:
    """Resume a paused SQLite-backed breakpoint thread.

    Args:
        thread_id: Conversation thread identifier.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Result after the resumed execution finishes or pauses again.
    """
    memory_db_path = get_memory_db_path(memory_db)

    with SqliteSaver.from_conn_string(memory_db_path) as memory:
        graph = build_breakpoint_graph(
            memory,
            interrupt_before=["tools"],
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
        return approve_pending_turn(graph, thread_id)


def get_state_with_sqlite(
    *,
    thread_id: str,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
) -> Module3StateResult:
    """Read the current state for a SQLite-backed breakpoint thread.

    Args:
        thread_id: Conversation thread identifier.
        memory_db: SQLite checkpoint file path.

    Returns:
        Current thread state.
    """
    memory_db_path = get_memory_db_path(memory_db)

    with SqliteSaver.from_conn_string(memory_db_path) as memory:
        graph = build_state_reader_graph(memory)
        return get_thread_state(graph, thread_id)


def list_history_with_sqlite(
    *,
    thread_id: str,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
) -> list[Module3HistoryEntry]:
    """List history for a SQLite-backed breakpoint thread.

    Args:
        thread_id: Conversation thread identifier.
        memory_db: SQLite checkpoint file path.

    Returns:
        Checkpoint history ordered by LangGraph.
    """
    memory_db_path = get_memory_db_path(memory_db)

    with SqliteSaver.from_conn_string(memory_db_path) as memory:
        graph = build_state_reader_graph(memory)
        return list_thread_history(graph, thread_id)


def replay_checkpoint_with_sqlite(
    *,
    thread_id: str,
    checkpoint_id: str,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module3TurnResult:
    """Replay a stored checkpoint using a SQLite checkpointer.

    Args:
        thread_id: Conversation thread identifier.
        checkpoint_id: Checkpoint identifier from history.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Result after replay finishes or pauses again.
    """
    memory_db_path = get_memory_db_path(memory_db)

    with SqliteSaver.from_conn_string(memory_db_path) as memory:
        graph = build_breakpoint_graph(
            memory,
            interrupt_before=["tools"],
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
        return replay_checkpoint(graph, thread_id, checkpoint_id)


def fork_checkpoint_with_sqlite(
    *,
    thread_id: str,
    checkpoint_id: str,
    replacement_prompt: str,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module3TurnResult:
    """Fork a stored checkpoint using a SQLite checkpointer.

    Args:
        thread_id: Conversation thread identifier.
        checkpoint_id: Checkpoint identifier from history.
        replacement_prompt: New prompt that should replace the selected human message.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Result after the forked branch runs until the next pause or completion.
    """
    memory_db_path = get_memory_db_path(memory_db)

    with SqliteSaver.from_conn_string(memory_db_path) as memory:
        graph = build_breakpoint_graph(
            memory,
            interrupt_before=["tools"],
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
        return fork_checkpoint(
            graph,
            thread_id,
            checkpoint_id,
            replacement_prompt,
        )


async def run_turn_with_sqlite_async(
    *,
    prompt: str,
    thread_id: str,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module3TurnResult:
    """Async wrapper for ``run_turn_with_sqlite``.

    Args:
        prompt: User prompt.
        thread_id: Conversation thread identifier.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Result after the graph pauses or completes.
    """
    return await anyio.to_thread.run_sync(
        partial(
            run_turn_with_sqlite,
            prompt=prompt,
            thread_id=thread_id,
            model=model,
            model_provider=model_provider,
            memory_db=memory_db,
            settings=settings,
        )
    )


async def approve_pending_turn_with_sqlite_async(
    *,
    thread_id: str,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module3TurnResult:
    """Async wrapper for ``approve_pending_turn_with_sqlite``.

    Args:
        thread_id: Conversation thread identifier.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Result after the resumed execution finishes or pauses again.
    """
    return await anyio.to_thread.run_sync(
        partial(
            approve_pending_turn_with_sqlite,
            thread_id=thread_id,
            model=model,
            model_provider=model_provider,
            memory_db=memory_db,
            settings=settings,
        )
    )


async def get_state_with_sqlite_async(
    *,
    thread_id: str,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
) -> Module3StateResult:
    """Async wrapper for ``get_state_with_sqlite``.

    Args:
        thread_id: Conversation thread identifier.
        memory_db: SQLite checkpoint file path.

    Returns:
        Current thread state.
    """
    return await anyio.to_thread.run_sync(
        partial(
            get_state_with_sqlite,
            thread_id=thread_id,
            memory_db=memory_db,
        )
    )


async def list_history_with_sqlite_async(
    *,
    thread_id: str,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
) -> list[Module3HistoryEntry]:
    """Async wrapper for ``list_history_with_sqlite``.

    Args:
        thread_id: Conversation thread identifier.
        memory_db: SQLite checkpoint file path.

    Returns:
        Checkpoint history ordered by LangGraph.
    """
    return await anyio.to_thread.run_sync(
        partial(
            list_history_with_sqlite,
            thread_id=thread_id,
            memory_db=memory_db,
        )
    )


async def replay_checkpoint_with_sqlite_async(
    *,
    thread_id: str,
    checkpoint_id: str,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module3TurnResult:
    """Async wrapper for ``replay_checkpoint_with_sqlite``.

    Args:
        thread_id: Conversation thread identifier.
        checkpoint_id: Checkpoint identifier from history.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Result after replay finishes or pauses again.
    """
    return await anyio.to_thread.run_sync(
        partial(
            replay_checkpoint_with_sqlite,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            model=model,
            model_provider=model_provider,
            memory_db=memory_db,
            settings=settings,
        )
    )


async def fork_checkpoint_with_sqlite_async(
    *,
    thread_id: str,
    checkpoint_id: str,
    replacement_prompt: str,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    settings: Settings | None = None,
) -> Module3TurnResult:
    """Async wrapper for ``fork_checkpoint_with_sqlite``.

    Args:
        thread_id: Conversation thread identifier.
        checkpoint_id: Checkpoint identifier from history.
        replacement_prompt: New prompt that should replace the selected human message.
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        memory_db: SQLite checkpoint file path.
        settings: Optional settings override for tests.

    Returns:
        Result after the forked branch runs until the next pause or completion.
    """
    return await anyio.to_thread.run_sync(
        partial(
            fork_checkpoint_with_sqlite,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            replacement_prompt=replacement_prompt,
            model=model,
            model_provider=model_provider,
            memory_db=memory_db,
            settings=settings,
        )
    )
