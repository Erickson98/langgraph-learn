"""LangGraph orchestration service for the module 1 assistant."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.config.settings import Settings
from app.module1.dependencies import get_chat_model
from app.module1.schemas import DEFAULT_MODEL, SYSTEM_PROMPT
from app.module1.services.tools import ARITHMETIC_TOOLS

SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)


def build_graph(
    model: str = DEFAULT_MODEL,
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Build the arithmetic assistant graph.

    Args:
        model: Chat model name used by the assistant node.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        Compiled LangGraph graph with in-memory checkpointing.
    """
    llm = get_chat_model(
        model=model,
        model_provider=model_provider,
        settings=settings,
    )
    llm_with_tools = llm.bind_tools(ARITHMETIC_TOOLS)

    def assistant(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = llm_with_tools.invoke([SYSTEM_MESSAGE] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(ARITHMETIC_TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=MemorySaver())


def build_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Build LangGraph runtime config for a conversation thread.

    Args:
        thread_id: Conversation thread identifier.

    Returns:
        LangGraph config dictionary.
    """
    return {"configurable": {"thread_id": thread_id}}


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
