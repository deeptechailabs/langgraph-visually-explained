"""Scene 6 -- add_messages, the reducer every agent ends up using.

Still no model call and still no API key: these are plain message objects
built by hand, so channel behaviour is the only thing on screen.

add_messages does two jobs `operator.add` cannot:
  * appends a message with an unseen id, and
  * REPLACES an existing message when the update carries the same id.

That second rule is the mechanism behind editing an agent's history --
correcting a tool call before it runs (ch. 7), trimming an old conversation
to fit the window (ch. 6).

Expected output:

    append  -> ['human: what is a reducer?',
                'ai: it merges writes.']
    replace -> ['human: what is a reducer, exactly?']
    --- graph run ---
    human: what is a reducer?
    ai: it merges the write into the channel.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


def show(messages) -> list:
    return [f"{m.type}: {m.content}" for m in messages]


# 1. The reducer alone, called the way the runtime calls it.
current = [HumanMessage("what is a reducer?", id="m1")]
fresh = [AIMessage("it merges writes.", id="m2")]
same = [HumanMessage("what is a reducer, exactly?", id="m1")]

print(f"append  -> {show(add_messages(current, fresh))}")
print(f"replace -> {show(add_messages(current, same))}")


# 2. The same reducer inside a graph. The node returns one
#    message; the channel appends it to the history.
def answer(state: State) -> dict:
    reply = "it merges the write into the channel."
    return {"messages": [AIMessage(reply)]}


builder = StateGraph(State)
builder.add_node("answer", answer)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile()

opening = HumanMessage("what is a reducer?")
final = graph.invoke({"messages": [opening]})

print("--- graph run ---")
for line in show(final["messages"]):
    print(line)
