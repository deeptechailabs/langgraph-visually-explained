"""Choose middleware or a custom graph for an execution budget.

`create_agent` has more than a recursion limit: middleware can
impose graceful model-call and tool-call budgets. This file
compares the raw runtime guard, a built-in model-call budget,
and a hand-wired graph with an explicit custom terminal route.

Run:  python 05_when_to_drop_back.py
"""

from typing import Annotated, TypedDict

from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from scripted_model import ScriptedModel, tool_call


@tool
def search(query: str) -> str:
    """Look up a fact about the series."""
    return f"found: {query}"


# A model that never stops asking -- the failure this is about.
GREEDY = [
    AIMessage(content="", tool_calls=[tool_call("search", {"query": "a"}, "c1")]),
    AIMessage(content="", tool_calls=[tool_call("search", {"query": "b"}, "c2")]),
    AIMessage(content="", tool_calls=[tool_call("search", {"query": "c"}, "c3")]),
    AIMessage(content="", tool_calls=[tool_call("search", {"query": "d"}, "c4")]),
]

ASK = {"messages": [HumanMessage("research everything")]}

print("=" * 50)
print("1. the prebuilt, with a model that never stops")
print("=" * 50)

from langchain.agents import create_agent  # noqa: E402

agent = create_agent(ScriptedModel(script=list(GREEDY)), [search])
try:
    agent.invoke(ASK, {"recursion_limit": 8})
except GraphRecursionError:
    print("   GraphRecursionError")
    print("   recursion_limit is the low-level")
    print("   safety backstop, so it raises.")

print()
print("2. the prebuilt, with a graceful call budget")
print("=" * 50)
limited = create_agent(
    ScriptedModel(script=list(GREEDY)),
    [search],
    middleware=[
        ModelCallLimitMiddleware(run_limit=2, exit_behavior="end")
    ],
)
limited_out = limited.invoke(ASK)
print(f"   last message {limited_out['messages'][-1].content}")

print()
print("3. the same budget, with a custom route")
print("=" * 50)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int


model = ScriptedModel(script=list(GREEDY)).bind_tools([search])
BUDGET = 2


def agent_node(state: State) -> dict:
    reply = model.invoke(state["messages"])
    return {"messages": [reply], "turns": state.get("turns", 0) + 1}


def route(state: State) -> str:
    """Branch into a domain-specific terminal node at the budget."""
    if state.get("turns", 0) >= BUDGET:
        return "spent"
    return tools_condition(state)


def spent(state: State) -> dict:
    completed = sum(
        isinstance(message, ToolMessage) for message in state["messages"]
    )
    return {
        "messages": [
            AIMessage(
                content=(
                    f"out of turns after {state['turns']} model turns; "
                    f"{completed} search completed"
                )
            )
        ]
    }


g = StateGraph(State)
g.add_node("agent", agent_node)
g.add_node("tools", ToolNode([search]))
g.add_node("spent", spent)
g.add_edge(START, "agent")
g.add_conditional_edges(
    "agent", route, {"tools": "tools", "spent": "spent", END: END}
)
g.add_edge("tools", "agent")
g.add_edge("spent", END)
app = g.compile()

out = app.invoke(ASK)
print(f"   turns used   {out['turns']}")
print(f"   last message {out['messages'][-1].content}")

print()
print("4. the practical rule")
print("   use create_agent plus middleware for")
print("   standard loop policies. write a StateGraph")
print("   when custom topology or an explicit routing")
print("   decision is clearer.")
