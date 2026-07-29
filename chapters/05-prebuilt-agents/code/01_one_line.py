"""The whole of chapter 4, in one line -- and the rename nobody mentions.

Chapter 4 built a ReAct agent by hand: a state, an agent node,
a tool node, a router, and an edge pointing backwards. About
forty lines.

`create_react_agent` produces that same graph from one call.
It also, on LangGraph 1.x, prints a deprecation warning that
almost every tutorial online predates. We show the warning
rather than describe it, the same way chapter 4 pointed an
empty graph at the recursion limit instead of trusting a blog
post about the number.

Run:  python 01_one_line.py
"""

import textwrap
import warnings

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from scripted_model import ScriptedModel, tool_call, show

FACTS = {
    "persistence": "chapter 6 covers persistence",
    "streaming": "chapter 8 covers streaming",
}


@tool
def search(query: str) -> str:
    """Look up a fact about the series."""
    for key, value in FACTS.items():
        if key in query.lower():
            return value
    return "no match"


def script() -> list[AIMessage]:
    """A fresh script per agent: ScriptedModel counts its calls."""
    return [
        AIMessage(
            content="",
            tool_calls=[tool_call("search", {"query": "persistence"}, "c1")],
        ),
        AIMessage(content="persistence is chapter 6"),
    ]


ASK = [HumanMessage("which chapter?")]

print("=" * 50)
print("1. forty lines of chapter 4, in one call")
print("=" * 50)

# Record the warning instead of letting it scroll past in red:
# it is the headline of this chapter, not noise.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from langgraph.prebuilt import create_react_agent

    old = create_react_agent(ScriptedModel(script=script()), [search])

show(old.invoke({"messages": ASK})["messages"])

print()
print("2. it works -- and it is on the way out")
for w in caught:
    print(f"   {w.category.__name__}")
    for line in textwrap.wrap(str(w.message), width=44):
        print(f"   {line}")

print()
print("3. the current path is langchain.agents")
from langchain.agents import create_agent  # noqa: E402

new = create_agent(ScriptedModel(script=script()), [search])
show(new.invoke({"messages": ASK})["messages"])

print()
print("4. same question, same answer, two doorways")
print("   the graph underneath is what matters.")
print("   the rest of this chapter reads it.")
