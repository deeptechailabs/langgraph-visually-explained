"""Read the graph the prebuilt wrote.

A prebuilt is not magic and it is not a black box -- it is a
StateGraph like any other, and it will tell you exactly what
it built if you ask. `get_graph()` returns the compiled shape:
nodes, edges, and which of those edges are conditional.

Set that next to the graph chapter 4 wrote by hand and the
point of this chapter lands: the same core loop is visible,
along with the prebuilt's additional conditional return edge.

Run:  python 02_what_it_generates.py
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from scripted_model import ScriptedModel, tool_call


@tool
def search(query: str) -> str:
    """Look up a fact about the series."""
    return "chapter 6 covers persistence"


SCRIPT = [
    AIMessage(
        content="",
        tool_calls=[tool_call("search", {"query": "persistence"}, "c1")],
    ),
    AIMessage(content="persistence is chapter 6"),
]


def draw(graph, title: str) -> None:
    """Print a compiled graph's shape, not its picture."""
    g = graph.get_graph()
    nodes = [n for n in g.nodes if n not in ("__start__", "__end__")]
    print(f"  {title}")
    print(f"    nodes  {nodes}")
    for e in g.edges:
        arrow = "..>" if e.conditional else "-->"
        print(f"    {e.source:9} {arrow} {e.target}")


# ---------------------------------------------- the prebuilt
from langchain.agents import create_agent  # noqa: E402

prebuilt = create_agent(ScriptedModel(script=list(SCRIPT)), [search])


# ------------------------------------------- chapter 4, again
class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ScriptedModel(script=list(SCRIPT)).bind_tools([search])


def agent(state: State) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


hand = StateGraph(State)
hand.add_node("agent", agent)
hand.add_node("tools", ToolNode([search]))
hand.add_edge(START, "agent")
hand.add_conditional_edges("agent", tools_condition)
hand.add_edge("tools", "agent")
hand = hand.compile()

print("=" * 50)
print("1. the same two graphs")
print("=" * 50)
draw(prebuilt, "created for you")
print()
draw(hand, "written by hand")

print()
print("2. what the router is")
print("   the conditional edge out of the agent node")
print("   is not a mystery -- it is a function you can")
print("   import and call on a plain dict:")
print()
for last, label in [
    (AIMessage(content="", tool_calls=[tool_call("search", {}, "c1")]), "with a tool call"),
    (AIMessage(content="all done"), "with plain text"),
]:
    where = tools_condition({"messages": [last]})
    print(f"   {label:17} -> {where!r}")

print()
print("3. so the prebuilt saved you the typing,")
print("   not the understanding.")
