"""Scene 3 -- ToolNode, written out.

`ToolNode` is not magic and it is not long. It is a lookup
table and a loop, and there are exactly four things it has to
get right:

  1. Read tool_calls off the LAST message, not the first.
  2. Run EVERY call in the list. One AIMessage can ask for
     several tools at once -- that is a fan-out the model
     chose, and dropping the extras silently loses work.
  3. Put `tool_call_id` on every ToolMessage. That id is how
     a result is matched to its request; without it the model
     sees answers with no questions.
  4. Turn an exception into a ToolMessage, not a crash. The
     model can recover from being told "that failed". It
     cannot recover from a traceback.

Expected output:

    model asked for 2 tools at once
      search -> found: 13 chapters
      divide -> error: ValueError: b must not be zero
    both came back tagged: ['c1', 'c2']
      nope   -> error: no tool named 'nope'
"""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from scripted_model import tool_call


@tool
def search(q: str) -> str:
    """Look up a fact."""
    return "found: 13 chapters"


@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    # A tool that checks its own arguments and raises a sentence
    # worth reading. The model only ever sees this string, so it
    # is the entire error message as far as the agent is concerned.
    if b == 0:
        raise ValueError("b must not be zero")
    return a / b


REGISTRY = {t.name: t for t in (search, divide)}


def tool_node(state: dict) -> dict:
    """Everything langgraph.prebuilt.ToolNode does, in 14 lines."""
    last = state["messages"][-1]
    out = []
    for call in last.tool_calls:
        fn = REGISTRY.get(call["name"])
        if fn is None:
            text = f"error: no tool named {call['name']!r}"
        else:
            try:
                text = str(fn.invoke(call["args"]))
            except Exception as exc:
                text = f"error: {type(exc).__name__}: {exc}"
        out.append(
            ToolMessage(content=text, tool_call_id=call["id"])
        )
    return {"messages": out}


asked = AIMessage(
    content="",
    tool_calls=[
        tool_call("search", {"q": "chapters"}, "c1"),
        tool_call("divide", {"a": 1, "b": 0}, "c2"),
    ],
)

print(f"model asked for {len(asked.tool_calls)} tools at once")
result = tool_node({"messages": [asked]})
for call, msg in zip(asked.tool_calls, result["messages"]):
    print(f"  {call['name']:6} -> {msg.content}")

ids = [m.tool_call_id for m in result["messages"]]
print(f"both came back tagged: {ids}")

missing = AIMessage(
    content="",
    tool_calls=[tool_call("nope", {}, "c3")],
)
only = tool_node({"messages": [missing]})["messages"][0]
print(f"  {'nope':6} -> {only.content}")
