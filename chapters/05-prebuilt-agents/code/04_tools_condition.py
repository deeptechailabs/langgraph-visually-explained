"""tools_condition: the router you already wrote, with a contract.

Chapter 4's router was one line:

    if state["messages"][-1].tool_calls: return "tools"
    return "done"

`tools_condition` is that, plus three things worth knowing
before you trust it: the exact strings it returns, the state
shapes it accepts, and what happens when your node is not
called "tools".

Run:  python 04_tools_condition.py
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from scripted_model import ScriptedModel, tool_call, show

WANTS_TOOL = AIMessage(
    content="", tool_calls=[tool_call("search", {"query": "p"}, "c1")]
)
PLAIN = AIMessage(content="all done")

print("=" * 50)
print("1. it returns two strings, and only two")
print("=" * 50)
for msg, label in [(WANTS_TOOL, "last msg has tool_calls"), (PLAIN, "last msg is plain text")]:
    print(f"  {label:24} -> {tools_condition({'messages': [msg]})!r}")

print()
print("   note the second one. it is not 'end',")
print("   and it is not 'done' -- it is the real")
print("   END sentinel, so it needs no path map.")
print(f"   END == {END!r}")

print()
print("2. it reads a bare list too")
print(f"   list form  -> {tools_condition([WANTS_TOOL])!r}")

print()
print("3. and a different messages key")
state = {"chat": [WANTS_TOOL]}
print(f"   messages_key='chat' -> {tools_condition(state, 'chat')!r}")

print()
print("4. so the node MUST be named 'tools'")
print("   unless you translate with a path map")


@tool
def search(query: str) -> str:
    """Look up a fact about the series."""
    return "chapter 6 covers persistence"


class State(TypedDict):
    messages: Annotated[list, add_messages]


SCRIPT = [
    AIMessage(content="", tool_calls=[tool_call("search", {"query": "p"}, "c1")]),
    AIMessage(content="persistence is chapter 6"),
]

model = ScriptedModel(script=SCRIPT).bind_tools([search])


def agent(state: State) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


# The node is called run_tools, not tools. Without the third
# argument compile() fails validation because tools_condition's
# inferred "tools" target is not a registered node.
g = StateGraph(State)
g.add_node("agent", agent)
g.add_node("run_tools", ToolNode([search]))
g.add_edge(START, "agent")
g.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "run_tools", END: END},  # <- the translation
)
g.add_edge("run_tools", "agent")
app = g.compile()

out = app.invoke({"messages": [HumanMessage("which chapter?")]})
show(out["messages"])

print()
print("   the path map is what lets the prebuilt")
print("   router drive a graph it did not name.")
