"""Scene 9 — a conditional edge turns the graph into a loop.

Expected output:

      [review] attempt 0 -> score 28
      [review] attempt 1 -> score 47
      [review] attempt 2 -> score 66
      [review] attempt 3 -> score 85
    {'topic': 'graph based agents', 'draft': 'Notes on graph based agents. More
     detail added. More detail added. More detail added.', 'score': 85, 'attempts': 3}
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    draft: str
    score: int
    attempts: int


def research(state: State) -> dict:
    return {"draft": f"Notes on {state['topic']}.", "attempts": 0}


def review(state: State) -> dict:
    score = len(state["draft"])
    print(f"  [review] attempt {state['attempts']} -> score {score}")
    return {"score": score}


def revise(state: State) -> dict:
    return {
        "draft": state["draft"] + " More detail added.",
        "attempts": state["attempts"] + 1,
    }


def good_enough(state: State) -> str:
    """Not a node. Does no work. Just picks the next hop."""
    if state["score"] >= 80 or state["attempts"] >= 3:
        return "done"
    return "revise"


builder = StateGraph(State)
builder.add_node("research", research)
builder.add_node("review", review)
builder.add_node("revise", revise)

builder.add_edge(START, "research")
builder.add_edge("research", "review")
builder.add_conditional_edges(
    "review", good_enough, {"revise": "revise", "done": END}
)
builder.add_edge("revise", "review")

graph = builder.compile()

print(graph.invoke({"topic": "graph based agents"}))
