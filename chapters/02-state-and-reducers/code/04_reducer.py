"""Scene 4 -- one annotation changes what a write means.

`Annotated[list[str], operator.add]` swaps the channel's LastValue rule for
a reducer. The node bodies and the edges are byte-for-byte what they were in
01_the_overwrite.py; only the schema line changed, and now both notes live.

Expected output:

    notes kept: 2
      - docs: nodes share one state.
      - blogs: graphs can loop.
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
builder.add_edge("search_docs", "search_blogs")
builder.add_edge("search_blogs", END)

graph = builder.compile()

start = {"topic": "langgraph", "notes": []}
final = graph.invoke(start)

print(f"notes kept: {len(final['notes'])}")
for note in final["notes"]:
    print(f"  - {note}")
