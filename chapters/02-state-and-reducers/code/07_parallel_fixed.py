"""Scene 5 -- the fan-out that crashed, fixed by one annotation.

Only the `notes` line differs from 06_parallel_crash.py. With a reducer the
channel knows how to combine two writes from the same superstep, so the
fan-out becomes legal and both branches land.

Order note: parallel branches merge in a deterministic order, but it is the
order LangGraph picks, not the order the edges were declared -- `blogs`
lands first below even though its edge was written second. If order matters,
sort inside the reducer.

Expected output:

    notes kept: 2
      - blogs: graphs can loop.
      - docs: nodes share one state.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]


def search_docs(state: State) -> dict:
    note = "docs: nodes share one state."
    return {"notes": [note]}


def search_blogs(state: State) -> dict:
    note = "blogs: graphs can loop."
    return {"notes": [note]}


builder = StateGraph(State)
builder.add_node("search_docs", search_docs)
builder.add_node("search_blogs", search_blogs)

builder.add_edge(START, "search_docs")
builder.add_edge(START, "search_blogs")
builder.add_edge("search_docs", END)
builder.add_edge("search_blogs", END)

graph = builder.compile()

final = graph.invoke({"topic": "langgraph", "notes": []})

print(f"notes kept: {len(final['notes'])}")
for note in final["notes"]:
    print(f"  - {note}")
