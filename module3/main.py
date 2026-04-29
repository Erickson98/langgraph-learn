from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from pathlib import Path
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


ARITHMETIC_SYSTEM = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)
DEFAULT_THREAD_ID = "module3-demo"


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


def divide(a: int, b: int) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


def require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for this demo.")


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()


def get_model(*, temperature: float = 0.0, model: str = "gpt-4o") -> ChatOpenAI:
    require_openai_key()
    return ChatOpenAI(model=model, temperature=temperature)


def new_thread_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def print_header(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}")


def print_last_message(messages: list[BaseMessage]) -> None:
    if not messages:
        return
    last = messages[-1]
    if hasattr(last, "pretty_print"):
        last.pretty_print()
    else:
        print(last)


ARITHMETIC_TOOLS = [add, multiply, subtract, divide]


def build_arithmetic_graph(*, interrupt_before: list[str] | None = None, with_human_feedback: bool = False):
    llm = get_model()
    llm_with_tools = llm.bind_tools(ARITHMETIC_TOOLS)

    def assistant(state: MessagesState):
        response = llm_with_tools.invoke([ARITHMETIC_SYSTEM] + state["messages"])
        return {"messages": [response]}

    def human_feedback(_: MessagesState):
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(ARITHMETIC_TOOLS))

    if with_human_feedback:
        builder.add_node("human_feedback", human_feedback)
        builder.add_edge(START, "human_feedback")
        builder.add_edge("human_feedback", "assistant")
        builder.add_edge("tools", "human_feedback")
    else:
        builder.add_edge(START, "assistant")
        builder.add_edge("tools", "assistant")

    builder.add_conditional_edges("assistant", tools_condition)

    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=interrupt_before,
    )


def print_pending_tool_calls(state) -> None:
    messages = state.values.get("messages", [])
    if not messages:
        return

    tool_calls = getattr(messages[-1], "tool_calls", None) or []
    if not tool_calls:
        return

    print("Pending tool calls:")
    for index, call in enumerate(tool_calls, start=1):
        print(f"{index}. {call.get('name')} args={call.get('args', {})}")


class DynamicState(TypedDict):
    input: str


def build_dynamic_breakpoint_graph():
    def step_1(state: DynamicState) -> DynamicState:
        print("--- Step 1 ---")
        return state

    def step_2(state: DynamicState) -> DynamicState:
        if len(state["input"]) > 5:
            raise NodeInterrupt(
                f"Received input that is longer than 5 characters: {state['input']}"
            )
        print("--- Step 2 ---")
        return state

    def step_3(state: DynamicState) -> DynamicState:
        print("--- Step 3 ---")
        return state

    builder = StateGraph(DynamicState)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_node("step_3", step_3)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)
    return builder.compile(checkpointer=MemorySaver())


class ConversationState(MessagesState):
    summary: str


def build_streaming_graph():
    model = get_model(temperature=0)

    def call_model(state: ConversationState, config: RunnableConfig):
        summary = state.get("summary", "")
        messages = state["messages"]
        if summary:
            messages = [SystemMessage(content=f"Summary of conversation earlier: {summary}")] + messages
        response = model.invoke(messages, config)
        return {"messages": [response]}

    def summarize_conversation(state: ConversationState):
        summary = state.get("summary", "")
        if summary:
            prompt = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            prompt = "Create a summary of the conversation above:"

        response = model.invoke(state["messages"] + [HumanMessage(content=prompt)])
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    def should_continue(state: ConversationState):
        return "summarize_conversation" if len(state["messages"]) > 6 else END

    builder = StateGraph(ConversationState)
    builder.add_node("conversation", call_model)
    builder.add_node("summarize_conversation", summarize_conversation)
    builder.add_edge(START, "conversation")
    builder.add_conditional_edges("conversation", should_continue)
    builder.add_edge("summarize_conversation", END)
    return builder.compile(checkpointer=MemorySaver())


def demo_breakpoints(*, interactive: bool = True) -> None:
    print_header("BREAKPOINTS")
    graph = build_arithmetic_graph(interrupt_before=["tools"])
    thread = {"configurable": {"thread_id": new_thread_id("breakpoints")}}
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print_last_message(event["messages"])

    state = graph.get_state(thread)
    print("Pending next node:", state.next)
    print_pending_tool_calls(state)

    if interactive:
        approved = input("Do you want to call the tool? [yes/no]: ").strip().lower()
    else:
        approved = "yes"

    if approved in {"y", "yes"}:
        for event in graph.stream(None, thread, stream_mode="values"):
            print_last_message(event["messages"])
    else:
        print("Operation cancelled by user.")


def demo_interactive_breakpoints(*, thread_id: str) -> None:
    print_header("INTERACTIVE BREAKPOINTS")
    graph = build_arithmetic_graph(interrupt_before=["tools"])
    base_thread_id = thread_id
    current_thread_id = thread_id

    print(f"Interactive mode. thread_id={current_thread_id}")
    print("Available tools: add, multiply, subtract, divide")
    print("Commands: `/approve`, `/deny`, `/state`, `/thread`, `/new`, `/quit`")

    while True:
        thread = thread_config(current_thread_id)
        state = graph.get_state(thread)
        pending = bool(state.next)

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
            print(f"Pending next node: {state.next or '<none>'}")
            print_pending_tool_calls(state)
            continue

        if lowered == "/approve":
            if not pending:
                print("No pending tool call in the current thread.")
                continue

            for event in graph.stream(None, thread, stream_mode="values"):
                print_last_message(event["messages"])
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
            print("Use `/approve` to run the tool or `/deny` to discard this paused thread.")
            print_pending_tool_calls(state)
            continue

        for event in graph.stream(
            {"messages": [HumanMessage(content=prompt)]},
            thread,
            stream_mode="values",
        ):
            print_last_message(event["messages"])

        updated_state = graph.get_state(thread)
        if updated_state.next:
            print(f"Pending next node: {updated_state.next}")
            print_pending_tool_calls(updated_state)
            print("Use `/approve` to execute the tool or `/deny` to skip it.")


def demo_dynamic_breakpoints() -> None:
    print_header("DYNAMIC BREAKPOINTS")
    graph = build_dynamic_breakpoint_graph()
    thread = {"configurable": {"thread_id": new_thread_id("dynamic")}}

    print("Running with an input that should interrupt...")
    for event in graph.stream({"input": "hello world"}, thread, stream_mode="values"):
        print(event)

    state = graph.get_state(thread)
    print("Pending next node:", state.next)

    print("Updating state so the graph can continue...")
    graph.update_state(thread, {"input": "hi"})

    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)


def demo_edit_state() -> None:
    print_header("EDIT STATE")
    graph = build_arithmetic_graph(interrupt_before=["assistant"])
    thread = {"configurable": {"thread_id": new_thread_id("edit-state")}}
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print_last_message(event["messages"])

    print("Updating the message before the assistant runs...")
    graph.update_state(
        thread,
        {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
    )

    state = graph.get_state(thread)
    for message in state.values["messages"]:
        if hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(message)

    for event in graph.stream(None, thread, stream_mode="values"):
        print_last_message(event["messages"])


def demo_human_feedback(*, interactive: bool = True) -> None:
    print_header("HUMAN FEEDBACK")
    graph = build_arithmetic_graph(
        interrupt_before=["human_feedback"],
        with_human_feedback=True,
    )
    thread = {"configurable": {"thread_id": new_thread_id("human-feedback")}}
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print_last_message(event["messages"])

    if interactive:
        feedback = input("Tell me how you want to update the state: ").strip()
    else:
        feedback = "Actually, multiply 10 and 10."

    if feedback:
        graph.update_state(
            thread,
            {"messages": [HumanMessage(content=feedback)]},
            as_node="human_feedback",
        )

    for event in graph.stream(None, thread, stream_mode="values"):
        print_last_message(event["messages"])


def demo_time_travel() -> None:
    print_header("TIME TRAVEL")
    graph = build_arithmetic_graph()
    thread = {"configurable": {"thread_id": new_thread_id("time-travel")}}
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print_last_message(event["messages"])

    history = list(graph.get_state_history(thread))
    print("Checkpoint count:", len(history))
    if not history:
        return

    replay_state = history[-2] if len(history) >= 2 else history[0]
    print("Replaying from checkpoint with next:", replay_state.next)
    for event in graph.stream(None, replay_state.config, stream_mode="values"):
        print_last_message(event["messages"])

    to_fork = replay_state
    original_messages = to_fork.values["messages"]
    target_message = original_messages[0]
    replacement = HumanMessage(content="Multiply 5 and 3", id=target_message.id)

    fork_config = graph.update_state(
        to_fork.config,
        {"messages": [replacement]},
    )
    print("Forked from earlier checkpoint. Running fork...")
    for event in graph.stream(None, fork_config, stream_mode="values"):
        print_last_message(event["messages"])


def demo_streaming() -> None:
    print_header("STREAMING + SUMMARIZATION")
    graph = build_streaming_graph()
    thread = {"configurable": {"thread_id": new_thread_id("streaming")}}
    turns = [
        "Hi! I'm Lance.",
        "I live in San Francisco.",
        "I work on LangGraph.",
        "My favorite sport is basketball.",
    ]

    for turn in turns:
        print(f"\nUSER: {turn}")
        for event in graph.stream(
            {"messages": [HumanMessage(content=turn)]},
            thread,
            stream_mode="values",
        ):
            print_last_message(event["messages"])

        state = graph.get_state(thread)
        summary = state.values.get("summary", "")
        print("Summary:", summary or "<none yet>")


def demo_interactive_chat(*, thread_id: str) -> None:
    print_header("INTERACTIVE CHAT")
    graph = build_streaming_graph()
    thread = thread_config(thread_id)

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
            state = graph.get_state(thread)
            summary = state.values.get("summary", "")
            print(f"Summary: {summary or '<none yet>'}")
            continue

        if prompt.lower() == "/messages":
            state = graph.get_state(thread)
            messages = state.values.get("messages", [])
            if not messages:
                print("Messages: <empty>")
                continue
            for message in messages:
                if hasattr(message, "pretty_print"):
                    message.pretty_print()
                else:
                    print(message)
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            thread,
        )
        print_last_message(result["messages"])


async def demo_streaming_events() -> None:
    print_header("STREAMING EVENTS")
    graph = build_streaming_graph()
    thread = {"configurable": {"thread_id": new_thread_id("stream-events")}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")

    async for event in graph.astream_events(
        {"messages": [input_message]},
        thread,
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


def run_all() -> None:
    demo_dynamic_breakpoints()

    for fn in [demo_breakpoints, demo_edit_state, demo_human_feedback, demo_time_travel, demo_streaming]:
        try:
            fn(interactive=False) if fn in {demo_breakpoints, demo_human_feedback} else fn()
        except RuntimeError as exc:
            print(f"Skipped {fn.__name__}: {exc}")

    try:
        asyncio.run(demo_streaming_events())
    except RuntimeError as exc:
        print(f"Skipped demo_streaming_events: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-in-one local implementation for LangChain Academy module 3."
    )
    parser.add_argument(
        "demo",
        choices=[
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
        ],
    )
    parser.add_argument(
        "--thread-id",
        default=DEFAULT_THREAD_ID,
        help=f"Conversation thread id for interactive chat. Default: {DEFAULT_THREAD_ID}",
    )
    args = parser.parse_args()

    if args.demo == "breakpoints":
        demo_breakpoints()
    elif args.demo == "interactive-breakpoints":
        demo_interactive_breakpoints(thread_id=args.thread_id)
    elif args.demo == "dynamic-breakpoints":
        demo_dynamic_breakpoints()
    elif args.demo == "edit-state":
        demo_edit_state()
    elif args.demo == "human-feedback":
        demo_human_feedback()
    elif args.demo == "time-travel":
        demo_time_travel()
    elif args.demo == "streaming":
        demo_streaming()
    elif args.demo == "interactive-chat":
        demo_interactive_chat(thread_id=args.thread_id)
    elif args.demo == "streaming-events":
        asyncio.run(demo_streaming_events())
    else:
        run_all()


if __name__ == "__main__":
    main()
