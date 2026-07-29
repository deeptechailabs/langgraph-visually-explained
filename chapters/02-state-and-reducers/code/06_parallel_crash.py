"""Scene 5 -- the same graph, fanned out, with no reducer. It raises.

Both searchers now hang off START, so they run in the same superstep and
both write `notes` before anything merges. A LastValue channel has no answer
for two writes at once, so LangGraph refuses rather than pick a winner.

This file is MEANT to fail -- the traceback is the teaching material.
07_parallel_fixed.py is the same graph with the fix.

Expected tail:

    langgraph.errors.InvalidUpdateError: At key 'notes': Can receive
    only one value per step. Use an Annotated key to handle multiple
    values.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    notes: list[str]


def search_docs(state: State) -> dict:
    note = "docs: nodes share one state."
    return {"notes": [note]}


def search_blogs(state: State) -> dict:
    note = "blogs: graphs can loop."
    return {"notes": [note]}


builder = StateGraph(State)
builder.add_node("search_docs", search_docs)
builder.add_node("search_blogs", search_blogs)

# Fan out: both edges leave START, so both run in one step.
builder.add_edge(START, "search_docs")
builder.add_edge(START, "search_blogs")
builder.add_edge("search_docs", END)
builder.add_edge("search_blogs", END)

graph = builder.compile()

print(graph.invoke({"topic": "langgraph", "notes": []}))
