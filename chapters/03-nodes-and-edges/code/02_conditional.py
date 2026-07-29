"""Scene 2 -- conditional edges: a router picks the next hop.

Three things worth being precise about:

  1. The router is NOT a node. It does no work and writes nothing. It reads
     the state and returns a string.
  2. That string is a KEY, not a node name. The path map translates keys to
     destinations, which keeps the router readable when the node names are
     long or change.
  3. The router runs after `review`, on the state `review` produced.

Expected output:

      [review] 42 -> retry
      [review] 84 -> finish
    ended after 2 review(s), score 84
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    score: int
    reviews: int


def review(state: State) -> dict:
    score = state["score"] + 42
    reviews = state["reviews"] + 1
    verdict = "finish" if score >= 80 else "retry"
    print(f"  [review] {score} -> {verdict}")
    return {"score": score, "reviews": reviews}


def revise(state: State) -> dict:
    return {}


def good_enough(state: State) -> str:
    """Not a node. Reads state, returns a KEY."""
    if state["score"] >= 80:
        return "finish"
    return "retry"


builder = StateGraph(State)
builder.add_node("review", review)
builder.add_node("revise", revise)

builder.add_edge(START, "review")
builder.add_conditional_edges(
    "review",
    good_enough,
    {"retry": "revise", "finish": END},
)
builder.add_edge("revise", "review")

graph = builder.compile()

final = graph.invoke({"score": 0, "reviews": 0})
n, s = final["reviews"], final["score"]
print(f"ended after {n} review(s), score {s}")
