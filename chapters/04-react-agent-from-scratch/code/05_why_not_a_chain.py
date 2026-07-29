"""Scene 5 -- the loop a fixed one-hop pipeline does not express.

Chapter 1 opened with the claim that chains are one-way. Four
chapters later we can finally make it concrete, because the
difference needs a tool result to exist before it shows up.

The question is deliberately two-hop:

    "how long is the persistence chapter?"

The chapter lookup has to happen before the length lookup. The
number of required lookups depends on the result of the first
lookup rather than a fixed hop-count constant.

A fixed straight-line pipeline has a predetermined number of
steps. Here it contains one search step, so it returns chapter
six and stops one hop short.

The graph is the same two nodes as scene 2, and no fixed hop-count
constant controls it. Its scripted trace searches twice; a real
model could request a different number of hops at runtime.

The edge back from tools to agent is the structural capability
the fixed one-hop pipeline lacks.

Expected output:

    Q: how long is the persistence chapter?

    fixed pipeline   (one search, then stop)
      search('persistence')
      -> chapter 6 covers persistence
      1 tool call, one hop short

    agent   (loop, hops decided at run time)
      search('persistence')
      search('chapter 6 length')
      -> chapter 6, and it runs 42 minutes
      2 tool calls, no fixed hop count controlled the loop
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from scripted_model import ScriptedModel, tool_call

Q = "how long is the persistence chapter?"

FACTS = {
    "persistence": "chapter 6 covers persistence",
    "chapter 6 length": "chapter 6 runs 42 minutes",
}


@tool
def search(q: str) -> str:
    """Look up a fact about the series."""
    print(f"  search({q!r})")
    return FACTS.get(q, "no result")


# The model asks for the second lookup only because the first one
# came back naming chapter 6. Scripted here, but the ORDER is the
# thing to notice: hop two is a consequence of hop one's answer.
SCRIPT = [
    AIMessage(content="", tool_calls=[tool_call("search", {"q": "persistence"}, "a")]),
    AIMessage(
        content="",
        tool_calls=[tool_call("search", {"q": "chapter 6 length"}, "b")],
    ),
    AIMessage(content="chapter 6, and it runs 42 minutes"),
]


class State(TypedDict):
    messages: Annotated[list, add_messages]


def count_calls(messages: list) -> int:
    return sum(len(getattr(m, "tool_calls", []) or []) for m in messages)


# --------------------------------------------------------------- the chain
# One search, then return its result. A straight line, so the
# number of hops is whatever the author typed -- here, one.


def run_chain() -> None:
    model = ScriptedModel(script=SCRIPT)
    messages = [HumanMessage(Q)]

    reply = model.invoke(messages)                     # step 1: think
    messages.append(reply)
    for call in reply.tool_calls:                      # step 2: act
        messages.append(
            ToolMessage(
                content=search.invoke(call["args"]),
                tool_call_id=call["id"],
            )
        )
    # step 3: return the single lookup result; no model answer follows.
    print(f"  -> {messages[-1].content}")
    print(f"  {count_calls(messages)} tool call, one hop short")


# --------------------------------------------------------------- the graph
# The same two nodes as scene 2. Note what is NOT here: any number.


def run_agent() -> None:
    model = ScriptedModel(script=SCRIPT)

    def agent(state: State) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    def tools(state: State) -> dict:
        return {
            "messages": [
                ToolMessage(
                    content=search.invoke(c["args"]), tool_call_id=c["id"]
                )
                for c in state["messages"][-1].tool_calls
            ]
        }

    b = StateGraph(State)
    b.add_node("agent", agent)
    b.add_node("tools", tools)
    b.add_edge(START, "agent")
    b.add_conditional_edges(
        "agent",
        lambda s: "tools" if s["messages"][-1].tool_calls else "done",
        {"tools": "tools", "done": END},
    )
    b.add_edge("tools", "agent")  # <-- the edge a chain does not have

    final = b.compile().invoke({"messages": [HumanMessage(Q)]})
    print(f"  -> {final['messages'][-1].content}")
    print(f"  {count_calls(final['messages'])} tool calls, "
          "no fixed hop count controlled the loop")


print(f"Q: {Q}\n")
print("fixed pipeline   (one search, then stop)")
run_chain()
print("\nagent   (loop, hops decided at run time)")
run_agent()
