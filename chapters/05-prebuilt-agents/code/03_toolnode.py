"""ToolNode, against the fourteen lines you wrote in chapter 4.

Chapter 4's hand-written tool node was a lookup table and a
loop, and it got four details right: read the LAST message, run
EVERY call, tag every result with its tool_call_id, and turn
exceptions into ToolMessages.

ToolNode gets three of those four right out of the box. The
fourth is the surprise in this file, and it is the reason
"the prebuilt handles errors for you" is not true.

Run:  python 03_toolnode.py
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from scripted_model import tool_call


class State(TypedDict):
    messages: Annotated[list, add_messages]


def as_graph(node: ToolNode):
    """Wrap a ToolNode in the smallest graph that can run it.

    On LangGraph 1.x a ToolNode cannot be `.invoke()`d on its own --
    it asks the runtime for config and raises `Missing required config
    key` when nobody is there to answer. That is a fair reminder of
    what it is: a NODE, not a helper function.
    """
    g = StateGraph(State)
    g.add_node("tools", node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


@tool
def search(query: str) -> str:
    """Look up a fact about the series."""
    return f"found: {query}"


@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b


def run(node, calls: list[dict], label: str) -> None:
    """Feed one AI message to the node and print what comes back."""
    print(f"  {label}")
    ask = AIMessage(content="", tool_calls=calls)
    try:
        # [1:] drops the AI message we fed in; we want the replies.
        for m in node.invoke({"messages": [ask]})["messages"][1:]:
            text = str(m.content).replace("\n", " ")[:30]
            print(f"    {m.tool_call_id:4} {text}")
    except Exception as exc:
        print(f"    RAISED  {type(exc).__name__}")


plain = as_graph(ToolNode([search, divide]))

print("=" * 50)
print("1. every call in the list, not just the first")
print("=" * 50)
run(
    plain,
    [
        tool_call("search", {"query": "persistence"}, "c1"),
        tool_call("search", {"query": "streaming"}, "c2"),
    ],
    "one message, two calls",
)

print()
print("2. a tool that does not exist")
run(plain, [tool_call("nope", {}, "c9")], "wrong name")

print()
print("3. arguments that do not fit the schema")
run(plain, [tool_call("divide", {"a": 1}, "c8")], "missing b")

print()
print("   both came back as MESSAGES. that is the")
print("   part everyone remembers about ToolNode.")

print()
print("4. now an exception from inside the tool")
run(plain, [tool_call("divide", {"a": 1, "b": 0}, "c7")], "1 / 0")

print()
print("   it did NOT become a message. the default")
print("   handler returns text for a bad CALL and")
print("   re-raises anything the tool itself threw,")
print("   so one bad division ends the whole run.")
print("   chapter 4's fourteen lines caught broader.")

print()
print("5. the fix is one argument")
forgiving = as_graph(
    ToolNode(
        [search, divide],
        handle_tool_errors=lambda e: f"error: {type(e).__name__}",
    )
)
run(forgiving, [tool_call("divide", {"a": 1, "b": 0}, "c7")], "same call")

print()
print("   handle_tool_errors takes a callable, and")
print("   that is where your policy lives.")
