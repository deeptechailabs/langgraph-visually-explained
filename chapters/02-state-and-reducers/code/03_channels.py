"""Scene 3 -- every key in the schema is a channel, and channels have rules.

The default rule is LastValue: the channel holds one value, and a write
replaces it. Printing what each node reads on entry makes the replacement
visible -- `search_blogs` reads exactly what `search_docs` wrote, and then
overwrites it.

Expected output:

      [docs]   in: []
      [blogs]  in: ['docs: nodes share one state.']
      [sum]    in: ['blogs: graphs can loop.']
    channels: ['topic', 'notes', 'summary']
    notes kept: 1
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    notes: list[str]
    summary: str


def search_docs(state: State) -> dict:
    print(f"  [docs]   in: {state['notes']}")
    note = "docs: nodes share one state."
    return {"notes": [note]}


def search_blogs(state: State) -> dict:
    print(f"  [blogs]  in: {state['notes']}")
    note = "blogs: graphs can loop."
    return {"notes": [note]}


def summarise(state: State) -> dict:
    notes = state["notes"]
    print(f"  [sum]    in: {notes}")
    return {"summary": f"{len(notes)} note(s)"}


builder = StateGraph(State)
builder.add_node("search_docs", search_docs)
builder.add_node("search_blogs", search_blogs)
builder.add_node("summarise", summarise)

builder.add_edge(START, "search_docs")
builder.add_edge("search_docs", "search_blogs")
builder.add_edge("search_blogs", "summarise")
builder.add_edge("summarise", END)

graph = builder.compile()

start = {"topic": "langgraph", "notes": [], "summary": ""}
final = graph.invoke(start)

# The schema is not documentation -- it is the channel list.
print(f"channels: {list(State.__annotations__)}")
print(f"notes kept: {len(final['notes'])}")
