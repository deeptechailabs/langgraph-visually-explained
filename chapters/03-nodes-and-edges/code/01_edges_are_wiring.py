"""Scene 1 -- edges are wiring, not the order you typed them.

The add_edge calls below are deliberately SHUFFLED: end first, middle second,
start last. The run order is unchanged, because add_edge declares a connection
rather than performing a step. Python's line order builds the graph; the graph
decides what runs.

Every line is <= 50 columns so it renders at 38px beside a dock.

Expected output:

      [plan]    building a query list
      [search]  running the search
      [report]  writing it up
    done: planned -> search -> report
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    trail: str


def plan(state: State) -> dict:
    print("  [plan]    building a query list")
    return {"trail": "planned"}


def search(state: State) -> dict:
    print("  [search]  running the search")
    return {"trail": state["trail"] + " -> search"}


def report(state: State) -> dict:
    print("  [report]  writing it up")
    return {"trail": state["trail"] + " -> report"}


builder = StateGraph(State)
builder.add_node("plan", plan)
builder.add_node("search", search)
builder.add_node("report", report)

# Declared bottom-up. The run order is unchanged.
builder.add_edge("report", END)
builder.add_edge("search", "report")
builder.add_edge("plan", "search")
builder.add_edge(START, "plan")

graph = builder.compile()

start = {"topic": "langgraph", "trail": ""}
final = graph.invoke(start)
print(f"done: {final['trail']}")
