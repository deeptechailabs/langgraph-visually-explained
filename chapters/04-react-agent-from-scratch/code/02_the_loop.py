"""Scene 2 -- the loop, as small as it goes.

ReAct is three words: think, act, observe. As a graph that is
two nodes and one conditional edge:

    START -> agent -> (tool_calls?) -> tools -> agent -> ...
                            |
                            no -> END

The edge back from `tools` to `agent` is the whole point. A
chain has no such edge, so a chain can never look at a tool
result and decide to use another tool. Everything else in
this chapter is detail hung on this shape.

Note what decides the loop: not a counter, not an `if` in
your code. The presence of `tool_calls` on the last message.
The model decides how many times round we go.

Expected output:

    [agent] thinking
    [tools] running search
    [agent] thinking
    stopped after 2 agent turns
    final: found it: 13 chapters
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from scripted_model import ScriptedModel, tool_call


class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ScriptedModel(
    script=[
        AIMessage(
            content="",
            tool_calls=[tool_call("search", {"q": "chapters"}, "c1")],
        ),
        AIMessage(content="found it: 13 chapters"),
    ]
)


def agent(state: State) -> dict:
    print("[agent] thinking")
    return {"messages": [model.invoke(state["messages"])]}


def tools(state: State) -> dict:
    last = state["messages"][-1]
    out = []
    for call in last.tool_calls:
        print(f"[tools] running {call['name']}")
        out.append(
            ToolMessage(
                content="13 chapters",
                tool_call_id=call["id"],
            )
        )
    return {"messages": out}


def should_continue(state: State) -> str:
    """The stopping condition, in one line of truth."""
    return "tools" if state["messages"][-1].tool_calls else "done"


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "done": END},
)
builder.add_edge("tools", "agent")

graph = builder.compile()

final = graph.invoke(
    {"messages": [HumanMessage("how many chapters?")]}
)

turns = sum(
    1 for m in final["messages"] if isinstance(m, AIMessage)
)
print(f"stopped after {turns} agent turns")
print(f"final: {final['messages'][-1].content}")
