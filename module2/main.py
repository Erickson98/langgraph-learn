import argparse
import os
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph


DEFAULT_THREAD_ID = "module2-demo"
DEFAULT_SUMMARIZE_AFTER = 6
DEFAULT_MEMORY_DB = Path(__file__).resolve().parent / "graph_memory.sqlite"


class State(MessagesState):
    summary: str


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


def build_graph(checkpointer: SqliteSaver, summarize_after: int = DEFAULT_SUMMARIZE_AFTER):
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    def call_model(state: State):
        summary = state.get("summary", "")

        if summary:
            system_message = (
                "Summary of the conversation so far:\n"
                f"{summary}\n\n"
                "Use this summary as context for the latest user request."
            )
            messages = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = state["messages"]

        response = model.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: State) -> Literal["summarize_conversation", "__end__"]:
        if len(state["messages"]) > summarize_after:
            return "summarize_conversation"
        return END

    def summarize_conversation(state: State):
        summary = state.get("summary", "")

        if summary:
            summary_message = (
                f"Current summary:\n{summary}\n\n"
                "Extend the summary using the latest messages above."
            )
        else:
            summary_message = "Create a concise summary of the conversation above."

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = model.invoke(messages)

        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)
    return workflow.compile(checkpointer=checkpointer)


def build_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def get_memory_db_path(memory_db: str) -> str:
    memory_path = Path(memory_db).expanduser()
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    return str(memory_path)


def run_turn(graph, prompt: str, thread_id: str) -> str:
    result = graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=build_config(thread_id),
    )
    return result["messages"][-1].content


def get_summary(graph, thread_id: str) -> str:
    snapshot = graph.get_state(build_config(thread_id))
    return snapshot.values.get("summary", "")


def maybe_render_graph(graph, *, print_mermaid: bool, save_mermaid_png: str | None) -> None:
    if print_mermaid:
        print(graph.get_graph().draw_mermaid())

    if save_mermaid_png:
        graph.get_graph().draw_mermaid_png(output_file_path=save_mermaid_png)
        print(f"Saved graph PNG to {save_mermaid_png}")


def run_interactive(graph, thread_id: str) -> int:
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
            if summary:
                print(f"Summary: {summary}")
            else:
                print("Summary: <empty>")
            continue

        print(f"Assistant: {run_turn(graph, prompt, thread_id)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the module 2 LangGraph summarizing chatbot."
    )
    parser.add_argument(
        "--prompt",
        default="My favorite number is 7. Please remember it.",
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
        type=int,
        default=DEFAULT_SUMMARIZE_AFTER,
        help=f"Summarize once message count is above this threshold. Default: {DEFAULT_SUMMARIZE_AFTER}",
    )
    parser.add_argument(
        "--memory-db",
        default=str(DEFAULT_MEMORY_DB),
        help=f"SQLite file used for persistent graph memory. Default: {DEFAULT_MEMORY_DB}",
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set. Add it to `.env` or export it in your shell.")

    memory_db = get_memory_db_path(args.memory_db)

    with SqliteSaver.from_conn_string(memory_db) as memory:
        graph = build_graph(memory, args.summarize_after)
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
            if summary:
                print(f"\nSummary:\n{summary}")
            else:
                print("\nSummary:\n<empty>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
