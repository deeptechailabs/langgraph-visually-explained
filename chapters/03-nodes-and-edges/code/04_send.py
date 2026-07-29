"""Scene 4 -- Send: a fan-out whose WIDTH is decided at run time.

Every edge so far was fixed when the graph compiled. Send is not an edge in
that sense: the routing function returns a LIST of Sends, one per item it
found in the state, so the graph runs three searchers today and seven
tomorrow without recompiling.

Each Send carries its own payload, and that payload is the state the target
node sees -- so `search` reads `query`, a key the parent state does not even
have. The results come back through a reducer, exactly as in chapter 2.

Expected output:

      [plan] 3 queries
      [search] langgraph channels
      [search] langgraph reducers
      [search] langgraph send api
    notes gathered: 3
      - notes on langgraph channels
      - notes on langgraph reducers
      - notes on langgraph send api
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class State(TypedDict):
    topic: str
    queries: list[str]
    notes: Annotated[list[str], operator.add]


class SearchTask(TypedDict):
    q: str


def plan(state: State) -> dict:
    qs = [
        f"{state['topic']} channels",
        f"{state['topic']} reducers",
        f"{state['topic']} send api",
    ]
    print(f"  [plan] {len(qs)} queries")
    return {"queries": qs}


def search(task: SearchTask) -> dict:
    print(f"  [search] {task['q']}")
    return {"notes": [f"notes on {task['q']}"]}


def fan_out(state: State):
    """Returns one Send per query -- the width is data."""
    return [
        Send("search", {"q": q})
        for q in state["queries"]
    ]


builder = StateGraph(State)
builder.add_node("plan", plan)
builder.add_node("search", search)

builder.add_edge(START, "plan")
builder.add_conditional_edges(
    "plan", fan_out, ["search"]
)
builder.add_edge("search", END)

graph = builder.compile()

start = {
    "topic": "langgraph",
    "queries": [],
    "notes": [],
}
final = graph.invoke(start)

print(f"notes gathered: {len(final['notes'])}")
for n in final["notes"]:
    print(f"  - {n}")
