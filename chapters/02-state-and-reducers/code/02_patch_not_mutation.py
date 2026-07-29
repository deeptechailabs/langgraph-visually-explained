"""Scene 2 -- the update protocol: return a patch, never mutate.

Two things this file proves at run time:

  1. A node's return value is the ONLY thing that reaches the state.
     `sneaky` assigns into the dict it was handed and returns nothing;
     the assignment does not survive.
  2. Keys you do not return are left alone. `score_it` returns only
     `score`, so `topic` and `notes` come out of the run untouched.

Expected output:

      [sneaky] state['topic'] = 'HACKED'
      [score_it] 1 note(s)
    topic -> langgraph
    notes -> ['docs: nodes share one state.']
    score -> 28
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    notes: list[str]
    score: int


def collect(state: State) -> dict:
    note = "docs: nodes share one state."
    return {"notes": [note]}


def sneaky(state: State) -> dict:
    """Assigning into state is not how you update it."""
    state["topic"] = "HACKED"
    print("  [sneaky] state['topic'] = 'HACKED'")
    return {}


def score_it(state: State) -> dict:
    notes = state["notes"]
    print(f"  [score_it] {len(notes)} note(s)")
    return {"score": sum(len(n) for n in notes)}


builder = StateGraph(State)
builder.add_node("collect", collect)
builder.add_node("sneaky", sneaky)
builder.add_node("score_it", score_it)

builder.add_edge(START, "collect")
builder.add_edge("collect", "sneaky")
builder.add_edge("sneaky", "score_it")
builder.add_edge("score_it", END)

graph = builder.compile()

start = {"topic": "langgraph", "notes": [], "score": 0}
final = graph.invoke(start)

for key in ("topic", "notes", "score"):
    print(f"{key:<5} -> {final[key]}")
