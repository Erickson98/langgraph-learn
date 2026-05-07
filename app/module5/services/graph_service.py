"""LangGraph orchestration service for module 5 memory workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    merge_message_runs,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from trustcall import create_extractor

from app.config.settings import Settings
from app.logging import get_logger
from app.module5.dependencies import get_chat_model
from app.module5.schemas import (
    DEFAULT_MODEL,
    DEFAULT_USER_ID,
    MemorySnapshot,
    Module5TurnResult,
    Profile,
    ToDo,
    UpdateMemory,
)

logger = get_logger(__name__)

MODEL_SYSTEM_MESSAGE = """You are a helpful memory-based productivity assistant.

You maintain three types of long-term memory:
1. User profile
2. ToDo list
3. User preferences for how tasks should be stored or updated

Current user profile:
<user_profile>
{user_profile}
</user_profile>

Current todo list:
<todo>
{todo}
</todo>

Current task-management preferences:
<instructions>
{instructions}
</instructions>

Rules:
1. If the user shares personal facts, call UpdateMemory with `user`.
2. If the user mentions a task, project, reminder, plan, deadline, or next step, call UpdateMemory with `todo`.
3. If the user says how they want tasks tracked, prioritized, or updated, call UpdateMemory with `instructions`.
4. Prefer updating the todo list rather than ignoring actionable details.
5. After memory updates, respond naturally to the user.
6. Do not announce profile updates explicitly.
7. You may mention that the todo list was updated when relevant.
"""

TRUSTCALL_INSTRUCTION = """Reflect on the following interaction.

Use the provided tools to retain necessary memory.
Use parallel tool calling when helpful.

System Time: {time}
"""

CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Update the user's task-management preferences based on what they asked for.

Current preferences:
<current_instructions>
{current_instructions}
</current_instructions>

Return only the updated preference text.
"""


class BoundToolModelLike(Protocol):
    """Minimal tool-bound chat model protocol used by module 5."""

    async def ainvoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """Invoke the bound chat model.

        Args:
            messages: Prompt messages.

        Returns:
            Assistant message.
        """
        ...


class ChatModelLike(Protocol):
    """Minimal chat model protocol used by the module 5 graph."""

    def bind_tools(
        self,
        tools: list[type[Any]],
        *,
        parallel_tool_calls: bool = False,
    ) -> BoundToolModelLike:
        """Return a tool-bound model.

        Args:
            tools: Tool schemas to expose to the model.
            parallel_tool_calls: Whether the model may call tools in parallel.

        Returns:
            Bound chat model.
        """
        ...

    async def ainvoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """Invoke the chat model.

        Args:
            messages: Prompt messages.

        Returns:
            Assistant message.
        """
        ...


class Spy:
    """Collect tool calls emitted by Trustcall child runs."""

    def __init__(self) -> None:
        """Initialize an empty tool-call collection."""
        self.called_tools: list[list[dict[str, Any]]] = []

    def __call__(self, run: Any) -> None:
        """Capture tool calls from a LangSmith run tree.

        Args:
            run: Root run object passed by LangChain listeners.
        """
        queue = [run]
        while queue:
            item = queue.pop()
            if item.child_runs:
                queue.extend(item.child_runs)
            if item.run_type == "chat_model":
                try:
                    tool_calls = item.outputs["generations"][0][0]["message"][
                        "kwargs"
                    ].get("tool_calls", [])
                except (KeyError, IndexError, TypeError):
                    logger.warning(
                        "Spy: unexpected run tree shape; skipping tool call capture"
                    )
                    tool_calls = []
                if tool_calls:
                    self.called_tools.append(tool_calls)


def message_content_to_text(content: object) -> str:
    """Normalize LangChain message content into plain text.

    Args:
        content: Message content returned by a chat model.

    Returns:
        Text representation of the message content.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(str(text) if text is not None else "")
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def extract_tool_info(tool_calls: list[list[dict[str, Any]]], schema_name: str) -> str:
    """Summarize Trustcall tool calls for the assistant follow-up.

    Args:
        tool_calls: Tool calls captured from Trustcall runs.
        schema_name: Insert tool name expected for new memory records.

    Returns:
        Human-readable summary used as the tool response.
    """
    changes: list[dict[str, Any]] = []
    for call_group in tool_calls:
        for call in call_group:
            if call["name"] == "PatchDoc":
                changes.append(
                    {
                        "type": "update",
                        "doc_id": call["args"]["json_doc_id"],
                        "planned_edits": call["args"]["planned_edits"],
                        "value": call["args"]["patches"][0]["value"],
                    }
                )
            elif call["name"] == schema_name:
                changes.append({"type": "new", "value": call["args"]})

    if not changes:
        return "updated todos"

    parts: list[str] = []
    for change in changes:
        if change["type"] == "update":
            parts.append(
                f"Document {change['doc_id']} updated. "
                f"Plan: {change['planned_edits']}. "
                f"Added content: {change['value']}"
            )
        else:
            parts.append(f"New {schema_name} created: {change['value']}")
    return "\n".join(parts)


def new_thread_id(prefix: str = "module5") -> str:
    """Return a generated thread id.

    Args:
        prefix: Prefix used for readability.

    Returns:
        Generated thread id.
    """
    return f"{prefix}-{uuid4().hex[:8]}"


def build_config(user_id: str, thread_id: str) -> RunnableConfig:
    """Build a LangGraph runnable config for one user thread.

    Args:
        user_id: Long-term memory user id.
        thread_id: Checkpoint thread id.

    Returns:
        Runnable config with module 5 identifiers.
    """
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}}


def get_user_id(config: RunnableConfig | None) -> str:
    """Read the user id from a LangGraph config.

    Args:
        config: Optional runnable config.

    Returns:
        Configured user id or the module default.
    """
    if config and "configurable" in config and config["configurable"].get("user_id"):
        return config["configurable"]["user_id"]
    return DEFAULT_USER_ID


async def profile_to_text(store: BaseStore, user_id: str) -> str:
    """Return the user's profile memory as text.

    Args:
        store: LangGraph store.
        user_id: Long-term memory user id.

    Returns:
        Readable profile memory.
    """
    memories = await store.asearch(("profile", user_id))
    if not memories:
        return "None"
    return str(memories[0].value)


async def todos_to_text(store: BaseStore, user_id: str) -> str:
    """Return the user's todo memory as text.

    Args:
        store: LangGraph store.
        user_id: Long-term memory user id.

    Returns:
        Readable todo memory.
    """
    memories = await store.asearch(("todo", user_id))
    if not memories:
        return "None"
    return "\n".join(str(item.value) for item in memories)


async def instructions_to_text(store: BaseStore, user_id: str) -> str:
    """Return the user's task-management preferences as text.

    Args:
        store: LangGraph store.
        user_id: Long-term memory user id.

    Returns:
        Readable preference memory.
    """
    memory = await store.aget(("instructions", user_id), "user_instructions")
    if not memory:
        return "None"
    return str(memory.value.get("memory", "None"))


async def get_memory_snapshot(store: BaseStore, user_id: str) -> MemorySnapshot:
    """Read all module 5 memory for one user.

    Args:
        store: LangGraph store.
        user_id: Long-term memory user id.

    Returns:
        Snapshot of profile, todos, and instructions.
    """
    return MemorySnapshot(
        profile=await profile_to_text(store, user_id),
        todos=await todos_to_text(store, user_id),
        instructions=await instructions_to_text(store, user_id),
    )


def latest_tool_call(state: MessagesState) -> dict[str, Any]:
    """Return the first tool call from the latest assistant message.

    Args:
        state: LangGraph message state.

    Returns:
        Tool call dictionary.

    Raises:
        ValueError: If the latest message does not contain a tool call.
    """
    message = state["messages"][-1]
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        raise ValueError("Expected a tool call in the latest assistant message.")
    return tool_calls[0]


def route_after_assistant(state: MessagesState) -> str:
    """Route the graph after an assistant message.

    Args:
        state: LangGraph message state.

    Returns:
        Next node name or LangGraph END.

    Raises:
        ValueError: If the assistant emits an unsupported update type.
    """
    message = state["messages"][-1]
    if not getattr(message, "tool_calls", None):
        return END

    update_type = message.tool_calls[0]["args"]["update_type"]
    if update_type == "user":
        return "update_profile"
    if update_type == "todo":
        return "update_todos"
    if update_type == "instructions":
        return "update_instructions"
    raise ValueError(f"Unknown update type: {update_type}")


def build_graph(
    *,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build the module 5 long-term memory graph.

    Args:
        model: Chat model name used by graph nodes.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.
        checkpointer: Optional checkpointer override.
        store: Optional long-term memory store override.

    Returns:
        Compiled LangGraph graph.
    """
    llm = get_chat_model(
        model=model,
        model_provider=model_provider,
        settings=settings,
    )
    saver = checkpointer or MemorySaver()
    memory_store = store or InMemoryStore()

    async def assistant_node(
        state: MessagesState,
        config: RunnableConfig,
        store: BaseStore,
    ) -> dict[str, list[BaseMessage]]:
        user_id = get_user_id(config)
        system_message = MODEL_SYSTEM_MESSAGE.format(
            user_profile=await profile_to_text(store, user_id),
            todo=await todos_to_text(store, user_id),
            instructions=await instructions_to_text(store, user_id),
        )
        response = await llm.bind_tools(
            [UpdateMemory],
            parallel_tool_calls=False,
        ).ainvoke([SystemMessage(content=system_message)] + state["messages"])
        return {"messages": [response]}

    async def update_profile(
        state: MessagesState,
        config: RunnableConfig,
        store: BaseStore,
    ) -> dict[str, list[ToolMessage]]:
        user_id = get_user_id(config)
        namespace = ("profile", user_id)
        existing_items = await store.asearch(namespace)
        existing_memories = (
            [(item.key, "Profile", item.value) for item in existing_items]
            if existing_items
            else None
        )
        instruction = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        merged = list(
            merge_message_runs(
                messages=[SystemMessage(content=instruction)] + state["messages"][:-1]
            )
        )
        extractor = create_extractor(
            llm,
            tools=[Profile],
            tool_choice="Profile",
            enable_inserts=True,
        )
        result = await extractor.ainvoke(
            {"messages": merged, "existing": existing_memories}
        )

        for response, meta in zip(result["responses"], result["response_metadata"]):
            await store.aput(
                namespace,
                meta.get("json_doc_id", "profile"),
                response.model_dump(mode="json"),
            )

        tool_call = latest_tool_call(state)
        return {
            "messages": [
                ToolMessage(content="updated profile", tool_call_id=tool_call["id"])
            ]
        }

    async def update_todos(
        state: MessagesState,
        config: RunnableConfig,
        store: BaseStore,
    ) -> dict[str, list[ToolMessage]]:
        user_id = get_user_id(config)
        namespace = ("todo", user_id)
        existing_items = await store.asearch(namespace)
        existing_memories = (
            [(item.key, "ToDo", item.value) for item in existing_items]
            if existing_items
            else None
        )
        instruction = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        merged = list(
            merge_message_runs(
                messages=[SystemMessage(content=instruction)] + state["messages"][:-1]
            )
        )
        spy = Spy()
        extractor = create_extractor(
            llm,
            tools=[ToDo],
            tool_choice="ToDo",
            enable_inserts=True,
        ).with_listeners(on_end=spy)
        result = await extractor.ainvoke(
            {"messages": merged, "existing": existing_memories}
        )

        for response, meta in zip(result["responses"], result["response_metadata"]):
            await store.aput(
                namespace,
                meta.get("json_doc_id", str(uuid4())),
                response.model_dump(mode="json"),
            )

        tool_call = latest_tool_call(state)
        return {
            "messages": [
                ToolMessage(
                    content=extract_tool_info(spy.called_tools, "ToDo"),
                    tool_call_id=tool_call["id"],
                )
            ]
        }

    async def update_instructions(
        state: MessagesState,
        config: RunnableConfig,
        store: BaseStore,
    ) -> dict[str, list[ToolMessage]]:
        user_id = get_user_id(config)
        namespace = ("instructions", user_id)
        existing = await store.aget(namespace, "user_instructions")
        system_message = CREATE_INSTRUCTIONS.format(
            current_instructions=existing.value["memory"] if existing else "None"
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_message)]
            + state["messages"][:-1]
            + [HumanMessage(content="Update the task-management preferences.")]
        )
        await store.aput(
            namespace,
            "user_instructions",
            {"memory": message_content_to_text(response.content)},
        )

        tool_call = latest_tool_call(state)
        return {
            "messages": [
                ToolMessage(
                    content="updated instructions",
                    tool_call_id=tool_call["id"],
                )
            ]
        }

    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("update_profile", update_profile)
    workflow.add_node("update_todos", update_todos)
    workflow.add_node("update_instructions", update_instructions)
    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges("assistant", route_after_assistant)
    workflow.add_edge("update_profile", "assistant")
    workflow.add_edge("update_todos", "assistant")
    workflow.add_edge("update_instructions", "assistant")
    return workflow.compile(checkpointer=saver, store=memory_store)


async def run_turn_with_sqlite_async(
    *,
    prompt: str,
    user_id: str,
    thread_id: str,
    memory_db: str,
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    settings: Settings | None = None,
    store: BaseStore | None = None,
) -> Module5TurnResult:
    """Run one module 5 turn using a SQLite-backed checkpointer.

    Conversation history is persisted to ``memory_db`` so it survives process
    restarts.  The long-term memory ``store`` is kept alive by the caller; it
    must be the same instance across requests so that profile, todos, and
    instructions accumulate over time.

    Args:
        prompt: User prompt for this turn.
        user_id: Long-term memory user id.
        thread_id: Checkpoint thread id.
        memory_db: SQLite file path, or ``':memory:'`` for ephemeral mode.
        model: Chat model name.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.
        store: Long-term memory store shared across requests.

    Returns:
        Assistant response and current memory snapshot.
    """
    from pathlib import Path

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    memory_store = store or InMemoryStore()

    if memory_db != ":memory:":
        Path(memory_db).parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(memory_db) as checkpointer:
        graph = build_graph(
            model=model,
            model_provider=model_provider,
            settings=settings,
            checkpointer=checkpointer,
            store=memory_store,
        )
        return await run_turn(
            graph,
            prompt=prompt,
            user_id=user_id,
            thread_id=thread_id,
        )


async def run_turn(
    graph: CompiledStateGraph,
    *,
    prompt: str,
    user_id: str,
    thread_id: str,
) -> Module5TurnResult:
    """Run one module 5 conversation turn.

    Args:
        graph: Compiled module 5 graph.
        prompt: User prompt for this turn.
        user_id: Long-term memory user id.
        thread_id: Checkpoint thread id.

    Returns:
        Assistant response and current memory snapshot.
    """
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=build_config(user_id=user_id, thread_id=thread_id),
    )
    final_message = result["messages"][-1]
    return Module5TurnResult(
        response=message_content_to_text(final_message.content),
        thread_id=thread_id,
        user_id=user_id,
        memory=await get_memory_snapshot(graph.store, user_id),
    )
