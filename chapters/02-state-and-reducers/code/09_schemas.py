"""Scene 7 -- three schemas: what goes in, what comes out, what stays inside.

The internal state is the union of every channel the nodes need. A caller
should not have to know about all of it. `input_schema` narrows what invoke
accepts; `output_schema` narrows what it returns. Everything else -- `raw`,
`attempts` -- is a private channel: real, durable for the length of the run,
and invisible at the edges.

The LangGraph 1.x keyword names are `input_schema=` / `output_schema=`.
The pre-1.0 spelling was `input=` / `output=`; do not put that on screen.

Expected output:

      [fetch]  wrote raw + attempts
      [polish] read 2 raw, attempt 1
    returned: {'summary': '2 sources on langgraph'}
    private keys the caller can see: []
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class Input(TypedDict):
    topic: str


class Output(TypedDict):
    summary: str


class State(TypedDict):
    topic: str
    raw: list[str]
    attempts: int
    summary: str


def fetch(state: State) -> dict:
    print("  [fetch]  wrote raw + attempts")
    return {"raw": ["docs", "blogs"], "attempts": 1}


def polish(state: State) -> dict:
    n = len(state["raw"])
    print(f"  [polish] read {n} raw, attempt {state['attempts']}")
    return {"summary": f"{n} sources on {state['topic']}"}


builder = StateGraph(
    State,
    input_schema=Input,
    output_schema=Output,
)
builder.add_node("fetch", fetch)
builder.add_node("polish", polish)

builder.add_edge(START, "fetch")
builder.add_edge("fetch", "polish")
builder.add_edge("polish", END)

graph = builder.compile()

# Only `topic` goes in -- no empty `raw`, no zero `attempts`.
final = graph.invoke({"topic": "langgraph"})

print(f"returned: {final}")
private = [k for k in ("raw", "attempts") if k in final]
print(f"private keys the caller can see: {private}")
