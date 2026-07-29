"""Scene 4 -- a reducer is just a function of (current, update).

`operator.add` is not special. Any two-argument function works, which means
the merge policy for a channel is ordinary Python you can read and test.
This one drops duplicates and keeps the newest three, so a research loop
cannot grow the state without bound.

Expected output:

    merge([], ['a'])              -> ['a']
    merge(['a'], ['a', 'b'])      -> ['a', 'b']
    merge(['a', 'b', 'c'], ['d']) -> ['b', 'c', 'd']
    after two nodes: ['blogs', 'papers', 'forums']

'docs' is missing from that last line, and that is the cap working: the two
nodes offer five distinct notes between them and only the newest three stay.
"""

from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

KEEP = 3


def merge_notes(current: list, update: list):
    out = list(current)
    for item in update:
        if item not in out:
            out.append(item)
    return out[-KEEP:]


class State(TypedDict):
    topic: str
    notes: Annotated[list[str], merge_notes]


def search(state: State) -> dict:
    # Returns a duplicate on purpose.
    return {"notes": ["docs", "docs", "blogs"]}


def search_more(state: State) -> dict:
    return {"notes": ["blogs", "papers", "forums"]}


builder = StateGraph(State)
builder.add_node("search", search)
builder.add_node("search_more", search_more)
builder.add_edge(START, "search")
builder.add_edge("search", "search_more")
builder.add_edge("search_more", END)

graph = builder.compile()

# The reducer alone, called the way the runtime calls it.
print(f"merge([], ['a'])              -> {merge_notes([], ['a'])}")
print(f"merge(['a'], ['a', 'b'])      -> {merge_notes(['a'], ['a', 'b'])}")
print(f"merge(['a', 'b', 'c'], ['d']) -> {merge_notes(['a', 'b', 'c'], ['d'])}")

final = graph.invoke({"topic": "langgraph", "notes": []})
print(f"after two nodes: {final['notes']}")
