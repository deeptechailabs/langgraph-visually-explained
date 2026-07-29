"""Scenes 5-8 — the first compiled graph.

Expected output:

      [research] looking up: graph based agents
      [review] scoring draft (28 chars)
    {'topic': 'graph based agents', 'draft': 'Notes on graph based agents.', 'score': 28}
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    draft: str
    score: int


def research(state: State) -> dict:
    print(f"  [research] looking up: {state['topic']}")
    return {"draft": f"Notes on {state['topic']}."}


def review(state: State) -> dict:
    print(f"  [review] scoring draft ({len(state['draft'])} chars)")
    return {"score": len(state["draft"])}


builder = StateGraph(State)
builder.add_node("research", research)
builder.add_node("review", review)

builder.add_edge(START, "research")
builder.add_edge("research", "review")
builder.add_edge("review", END)

graph = builder.compile()

result = graph.invoke({"topic": "graph based agents"})
print(result)
