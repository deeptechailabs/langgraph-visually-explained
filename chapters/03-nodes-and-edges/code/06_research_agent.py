"""Scene 6 -- every piece of this chapter in one graph.

  START -> plan            a normal edge
  plan  -> Send x N        dynamic fan-out, width from the state
  search-> gather          the branches converge
  gather-> Command         update and route in one return
  critic-> plan | END      the loop, closed by a conditional route

The running example from chapter 1 finally has a real shape: it decides how
many searches to run, runs them together, judges the result, and either
tightens the plan or stops.

Still no model call -- the "scoring" is len(). Chapter 4 swaps that for an LLM
and nothing about this wiring changes.

Expected output:

      [plan]   round 1: 2 q
      [search] langgraph edges
      [search] langgraph nodes
      [gather] 2 notes, score 20 -> retry
      [plan]   round 2: 3 q
      [search] langgraph edges
      [search] langgraph nodes
      [search] langgraph command
      [gather] 5 notes, score 50 -> done
    finished in 2 rounds with 5 notes
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

TERMS = ["edges", "nodes", "command", "send"]


class State(TypedDict):
    topic: str
    rounds: int
    notes: Annotated[list[str], operator.add]
    score: int


class Task(TypedDict):
    q: str


def plan(state: State) -> dict:
    rounds = state["rounds"] + 1
    n = rounds + 1
    print(f"  [plan]   round {rounds}: {n} q")
    return {"rounds": rounds}


def fan_out(state: State):
    n = state["rounds"] + 1
    return [
        Send("search", {"q": f"{state['topic']} {t}"})
        for t in TERMS[:n]
    ]


def search(task: Task) -> dict:
    print(f"  [search] {task['q']}")
    return {"notes": [f"note on {task['q']}"]}


Route = Command[Literal["plan", "__end__"]]


def gather(state: State) -> Route:
    score = len(state["notes"]) * 10
    enough = score >= 40 or state["rounds"] >= 3
    goto = END if enough else "plan"
    label = "done" if enough else "retry"
    print(
        f"  [gather] {len(state['notes'])} notes, "
        f"score {score} -> {label}"
    )
    patch = {"score": score}
    return Command(update=patch, goto=goto)


builder = StateGraph(State)
builder.add_node("plan", plan)
builder.add_node("search", search)
builder.add_node("gather", gather)

builder.add_edge(START, "plan")
builder.add_conditional_edges(
    "plan", fan_out, ["search"]
)
builder.add_edge("search", "gather")

graph = builder.compile()

start = {
    "topic": "langgraph",
    "rounds": 0,
    "notes": [],
    "score": 0,
}
final = graph.invoke(start)

print(
    f"finished in {final['rounds']} rounds "
    f"with {len(final['notes'])} notes"
)
