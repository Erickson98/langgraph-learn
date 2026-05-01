from __future__ import annotations

import argparse
import os
from pathlib import Path
import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field
from trustcall import create_extractor

from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


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
def require_openai_key() -> None:
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")


class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        queue = [run]
        while queue:
            item = queue.pop()
            if item.child_runs:
                queue.extend(item.child_runs)
            if item.run_type == "chat_model":
                tool_calls = item.outputs["generations"][0][0]["message"]["kwargs"].get("tool_calls", [])
                if tool_calls:
                    self.called_tools.append(tool_calls)


def extract_tool_info(tool_calls, schema_name: str) -> str:
    changes = []
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

    parts = []
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


class Profile(BaseModel):
    name: Optional[str] = Field(default=None, description="The user's name")
    location: Optional[str] = Field(default=None, description="The user's location")
    job: Optional[str] = Field(default=None, description="The user's job")
    connections: list[str] = Field(default_factory=list, description="People connected to the user")
    interests: list[str] = Field(default_factory=list, description="The user's interests")


class ToDo(BaseModel):
    task: str = Field(description="The task to be completed")
    time_to_complete: Optional[int] = Field(default=None, description="Estimated time in minutes")
    deadline: Optional[datetime] = Field(default=None, description="When the task should be completed")
    solutions: list[str] = Field(default_factory=list, description="Concrete solution ideas")
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        default="not started", description="Current status of the task"
    )


class UpdateMemory(BaseModel):
    update_type: Literal["user", "todo", "instructions"]


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


def get_model() -> ChatOpenAI:
    require_openai_key()
    return ChatOpenAI(model="gpt-4o", temperature=0)


def get_user_id(config: RunnableConfig | None) -> str:
    if config and "configurable" in config and config["configurable"].get("user_id"):
        return config["configurable"]["user_id"]
    return "default-user"


def profile_to_text(store: BaseStore, user_id: str) -> str:
    memories = store.search(("profile", user_id))
    if not memories:
        return "None"
    return str(memories[0].value)


def todos_to_text(store: BaseStore, user_id: str) -> str:
    memories = store.search(("todo", user_id))
    if not memories:
        return "None"
    return "\n".join(str(item.value) for item in memories)


def instructions_to_text(store: BaseStore, user_id: str) -> str:
    memory = store.get(("instructions", user_id), "user_instructions")
    if not memory:
        return "None"
    return str(memory.value.get("memory", "None"))


def assistant_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = get_user_id(config)
    system_message = MODEL_SYSTEM_MESSAGE.format(
        user_profile=profile_to_text(store, user_id),
        todo=todos_to_text(store, user_id),
        instructions=instructions_to_text(store, user_id),
    )
    model = get_model()
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_message)] + state["messages"]
    )
    return {"messages": [response]}


def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = get_user_id(config)
    namespace = ("profile", user_id)
    existing_items = store.search(namespace)
    existing_memories = (
        [(item.key, "Profile", item.value) for item in existing_items] if existing_items else None
    )

    instruction = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    merged = list(merge_message_runs(messages=[SystemMessage(content=instruction)] + state["messages"][:-1]))

    extractor = create_extractor(
        get_model(),
        tools=[Profile],
        tool_choice="Profile",
        enable_inserts=True,
    )
    result = extractor.invoke({"messages": merged, "existing": existing_memories})

    for response, meta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, meta.get("json_doc_id", "profile"), response.model_dump(mode="json"))

    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id": tool_call_id}]}


def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = get_user_id(config)
    namespace = ("todo", user_id)
    existing_items = store.search(namespace)
    existing_memories = (
        [(item.key, "ToDo", item.value) for item in existing_items] if existing_items else None
    )

    instruction = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    merged = list(merge_message_runs(messages=[SystemMessage(content=instruction)] + state["messages"][:-1]))

    spy = Spy()
    extractor = create_extractor(
        get_model(),
        tools=[ToDo],
        tool_choice="ToDo",
        enable_inserts=True,
    ).with_listeners(on_end=spy)

    result = extractor.invoke({"messages": merged, "existing": existing_memories})

    for response, meta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, meta.get("json_doc_id", str(uuid.uuid4())), response.model_dump(mode="json"))

    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    summary = extract_tool_info(spy.called_tools, "ToDo")
    return {"messages": [{"role": "tool", "content": summary, "tool_call_id": tool_call_id}]}


def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = get_user_id(config)
    namespace = ("instructions", user_id)
    existing = store.get(namespace, "user_instructions")

    model = get_model()
    system_message = CREATE_INSTRUCTIONS.format(
        current_instructions=existing.value["memory"] if existing else "None"
    )
    response = model.invoke(
        [SystemMessage(content=system_message)]
        + state["messages"][:-1]
        + [HumanMessage(content="Update the task-management preferences.")]
    )

    store.put(namespace, "user_instructions", {"memory": response.content})

    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id": tool_call_id}]}


def route_after_assistant(state: MessagesState):
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


def build_app():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("update_profile", update_profile)
    builder.add_node("update_todos", update_todos)
    builder.add_node("update_instructions", update_instructions)

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", route_after_assistant)
    builder.add_edge("update_profile", "assistant")
    builder.add_edge("update_todos", "assistant")
    builder.add_edge("update_instructions", "assistant")

    return builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())


def print_memory_snapshot(app, user_id: str) -> None:
    store = app.store
    print("\n--- MEMORY SNAPSHOT ---")
    print("Profile:")
    print(profile_to_text(store, user_id))
    print("\nToDos:")
    print(todos_to_text(store, user_id))
    print("\nPreferences:")
    print(instructions_to_text(store, user_id))
    print("-----------------------\n")


def run_chat(user_id: str) -> None:
    app = build_app()
    thread_id = f"thread-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    print(f"Memory Agent started for user_id={user_id}")
    print("Type /memory to inspect stored memory, /quit to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/memory":
            print_memory_snapshot(app, user_id)
            continue

        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        final_message = result["messages"][-1]
        print(f"Assistant: {final_message.content}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file memory agent.")
    parser.add_argument("--user-id", default="demo-user")
    args = parser.parse_args()
    run_chat(args.user_id)


if __name__ == "__main__":
    main()
