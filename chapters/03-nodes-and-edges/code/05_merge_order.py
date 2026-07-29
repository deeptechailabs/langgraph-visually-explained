"""Scene 5 -- how parallel branches actually merge.

Chapter 2 left this open. Three facts, all demonstrated below:

  1. Branches in one superstep all read the SAME input state. `slow` cannot
     see what `fast` wrote, because nothing has been written yet.
  2. Their patches are applied together, at the end of the step.
  3. The order they merge in is deterministic but it is LangGraph's order,
     NOT the order the edges were declared. Declaring `c`, `a`, `b` below
     still merges a, b, c.

If order matters to you, sort inside the reducer -- do not rely on the edges.

Expected output:

      [a] sees notes=[]
      [b] sees notes=[]
      [c] sees notes=[]
    merged: ['from a', 'from b', 'from c']
    declared: c, a, b

Note the print order as well as the merge order: declaring c, a, b got you
a, b, c in BOTH. The declaration order is not a scheduling hint.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    notes: Annotated[list[str], operator.add]


def make(name: str):
    def node(state: State) -> dict:
        seen = state["notes"]
        print(f"  [{name}] sees notes={seen}")
        return {"notes": [f"from {name}"]}

    return node


builder = StateGraph(State)
for n in ("a", "b", "c"):
    builder.add_node(n, make(n))

# Declared c, a, b -- not alphabetical.
for n in ("c", "a", "b"):
    builder.add_edge(START, n)
    builder.add_edge(n, END)

graph = builder.compile()

final = graph.invoke({"notes": []})
print(f"merged: {final['notes']}")
print("declared: c, a, b")
