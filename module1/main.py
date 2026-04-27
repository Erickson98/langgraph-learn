import argparse
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


DEFAULT_THREAD_ID = "math-demo"


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


def subtract(a: int, b: int) -> int:
    """Subtracts b from a.

    Args:
        a: first int
        b: second int
    """
    return a - b


def power(a: int, b: int) -> int:
    """Raises a to the power of b.

    Args:
        a: base int
        b: exponent int
    """
    return a**b


def modulo(a: int, b: int) -> int:
    """Returns the remainder of a divided by b.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot take modulo by zero.")
    return a % b


def floor_divide(a: int, b: int) -> int:
    """Divides a by b and returns the integer quotient.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot floor divide by zero.")
    return a // b


def absolute_value(a: int) -> int:
    """Returns the absolute value of a.

    Args:
        a: input int
    """
    return abs(a)


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


tools = [add, multiply, divide, subtract, power, modulo, floor_divide, absolute_value]

load_local_env()

sys_msg = SystemMessage(
    content=(
        "You are a helpful assistant tasked with performing arithmetic and basic math "
        "operations on a set of inputs. Reuse relevant context from earlier messages "
        "in the same thread."
    )
)


def build_graph():
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def build_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def run_turn(graph, prompt: str, thread_id: str) -> str:
    result = graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=build_config(thread_id),
    )
    return result["messages"][-1].content


def run_interactive(graph, thread_id: str) -> int:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the module 1 LangGraph arithmetic assistant.")
    parser.add_argument(
        "--prompt",
        default="What is ((7 * 6) - 5) / 3? Also tell me the remainder of 43 divided by 5.",
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set. Add it to `.env` or export it in your shell.")

    graph = build_graph()

    if args.interactive:
        return run_interactive(graph, args.thread_id)

    print(run_turn(graph, args.prompt, args.thread_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
