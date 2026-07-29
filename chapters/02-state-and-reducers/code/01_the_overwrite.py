"""Scene 1 -- two nodes, one key, and a note that quietly disappears.

Chapter 1's state held three scalar fields, so nothing ever collided. The
moment a second node writes the same key, the default channel rule shows up:
the last write wins and the first one is gone. No error, no warning.

Every line here is <= 50 columns so it renders at 38px beside the dock in
the video. That constraint is why the notes are short and why locals are
pulled out of the return statements.

Expected output:

    notes kept: 1
      - blogs: graphs can loop.
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

builder.add_edge(START, "search_docs")
builder.add_edge("search_docs", "search_blogs")
builder.add_edge("search_blogs", END)

graph = builder.compile()

start = {"topic": "langgraph", "notes": []}
final = graph.invoke(start)

print(f"notes kept: {len(final['notes'])}")
for note in final["notes"]:
    print(f"  - {note}")
