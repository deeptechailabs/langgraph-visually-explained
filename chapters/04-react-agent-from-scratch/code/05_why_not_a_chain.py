"""Scene 5 -- the loop a chain cannot express.

Chapter 1 opened with the claim that chains are one-way. Four
chapters later we can finally make it concrete, because the
difference needs a tool result to exist before it shows up.

The question is deliberately two-hop:

    "how long is the persistence chapter?"

Neither half can be answered first. You cannot look up the length
until you know which chapter it is, and you do not know which
chapter it is until you have searched. The number of lookups is
not a property of your code -- it is a property of the ANSWER to
the first lookup.

A chain is a fixed pipeline. Its author writes the steps, so its
author has to know the count in advance. Here it is written with
one search step, which is the honest way to write it: you cannot
write "however many it takes" as a straight line. It returns
chapter six and stops, one hop short.

The graph is the same two nodes as scene 2, and nothing in it
mentions two. It searched twice because the first result made the
model ask again -- and if the question had needed four hops, the
same code would have run four.

The edge back from tools to agent is the only difference. That
edge is what a chain does not have.

Expected output:

    Q: how long is the persistence chapter?

    chain   (search -> answer, fixed at 1 hop)
      search('persistence')
      -> chapter 6 covers persistence
      1 tool call, one hop short

    agent   (loop, hops decided at run time)
      search('persistence')
      search('chapter 6 length')
      -> chapter 6, and it runs 42 minutes
      2 tool calls, nothing in the code said 2
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
# One search, then answer. A straight line, so the number of hops
# is whatever the author typed -- here, one.


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
    # step 3: answer. There is no step 4, and no way to ask for one.
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
          f"nothing in the code said 2")


print(f"Q: {Q}\n")
print("chain   (search -> answer, fixed at 1 hop)")
run_chain()
print("\nagent   (loop, hops decided at run time)")
run_agent()
