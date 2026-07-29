"""Scene 3 -- Command: update the state AND choose the next node, together.

A conditional edge needs two pieces: the node returns a patch, then a separate
router reads the state back and decides. Command collapses that into one
return value, so the decision lives with the code that had the information.

The Literal annotation is not decoration -- it is how the graph knows where
this node can go, which is what lets compile() draw and validate the edges.

Expected output:

      [triage] 2 findings -> escalate
      [escalate] handing to a human
    route taken: escalate
"""

from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


class State(TypedDict):
    findings: int
    route: str


def triage(
    state: State,
) -> Command[Literal["escalate", "autofix"]]:
    n = state["findings"]
    goto = "escalate" if n >= 2 else "autofix"
    print(f"  [triage] {n} findings -> {goto}")
    patch = {"route": goto}
    return Command(update=patch, goto=goto)


def escalate(state: State) -> dict:
    print("  [escalate] handing to a human")
    return {}


def autofix(state: State) -> dict:
    print("  [autofix] patching it automatically")
    return {}


builder = StateGraph(State)
builder.add_node("triage", triage)
builder.add_node("escalate", escalate)
builder.add_node("autofix", autofix)

builder.add_edge(START, "triage")
builder.add_edge("escalate", END)
builder.add_edge("autofix", END)

graph = builder.compile()

final = graph.invoke({"findings": 2, "route": ""})
print(f"route taken: {final['route']}")
