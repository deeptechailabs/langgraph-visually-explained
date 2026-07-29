"""The running example, rebuilt on prebuilts.

Chapter 3 gave the research assistant a planner and a Send
fan-out. Chapter 4 gave it a model and forty lines. This is the
same assistant a third time -- and the graph is now three lines
of wiring, because ToolNode and tools_condition are doing the
parts you have already written by hand twice.

The point is not that it is shorter. It is that you can read
what it generated and say why every piece is there.

Run:  python 06_research_agent.py
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from scripted_model import ScriptedModel, tool_call, show

NOTES = {
    "persistence": "checkpointers save thread state",
    "streaming": "stream modes: values, updates",
    "checkpointers": "SqliteSaver writes one file",
}


@tool
def search(query: str) -> str:
    """Look up a note about the series."""
    for key, value in NOTES.items():
        if key in query.lower():
            return value
    return "no match"


# Turn 1 asks for two things at once. The width now lives in one
# model message rather than in a list of Send objects.
SCRIPT = [
    AIMessage(
        content="",
        tool_calls=[
            tool_call("search", {"query": "persistence"}, "t1"),
            tool_call("search", {"query": "streaming"}, "t2"),
        ],
    ),
    # Turn 2 is scripted to emulate following a word from a result.
    AIMessage(
        content="",
        tool_calls=[tool_call("search", {"query": "checkpointers"}, "t3")],
    ),
    AIMessage(content="report: checkpointers over threads"),
]


class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ScriptedModel(script=SCRIPT).bind_tools([search])


def agent(state: State) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


# Three lines of wiring. That is the whole graph.
g = StateGraph(State)
g.add_node("agent", agent)
g.add_node("tools", ToolNode([search]))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)
g.add_edge("tools", "agent")
app = g.compile()

print("=" * 50)
print("1. three turns, one conversation")
print("=" * 50)
out = app.invoke({"messages": [HumanMessage("research persistence and streaming")]})
show(out["messages"])

print()
print("2. the tally")
ai = [m for m in out["messages"] if isinstance(m, AIMessage)]
calls = sum(len(m.tool_calls) for m in ai)
print(f"   agent turns  {len(ai)}")
print(f"   tool calls   {calls}")
print(f"   messages     {len(out['messages'])}")

print()
print("3. and the multi-call width is still there")
print("   turn one asked for two searches inside")
print("   ONE message, and the tool node ran both")
print("   inside one graph node. That resembles the")
print("   width chapter 3 built with Send, but it is")
print("   not the same graph topology.")

print()
print("4. using a live model")
print("   install a provider integration, configure")
print("   credentials, choose a current tool-capable")
print("   model, and bind the same tools.")
