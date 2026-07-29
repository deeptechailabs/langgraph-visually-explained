"""Scene 4 -- what actually stops the loop.

The loop in 02 has no counter and no maximum. It runs until the
model stops asking for tools. That is the correct default, and it
is also the thing that will bite you, so it is worth being exact
about who decides what.

Three stops, in the order you meet them:

  1. THE MODEL STOPS ASKING. An AIMessage with no tool_calls.
     The normal exit, and the only one that means the agent
     finished. Nothing in your code decided it.
  2. THE RECURSION LIMIT. LangGraph counts SUPERSTEPS and raises
     GraphRecursionError. It is a backstop against a runaway
     graph, not a policy: you get an exception, not an answer.
  3. YOUR OWN BUDGET. Count turns in the state and let the router
     stop on its own terms, so the agent returns a real message
     instead of raising.

The third is the one to reach for. A limit you enforce yourself
can say "I ran out of turns, here is what I have"; a limit that
raises can only say nothing at all.

About that default: nearly every tutorial says 25. It was 25 in
LangGraph 0.x. This file does not take anyone's word for it -- it
runs an empty graph into the wall and prints the number the
library actually used, which on 1.2.9 is 10007. At one model call
per superstep that is not a safety net you will ever feel; it is
a stop for infinite loops, and the budget in step 3 is what
actually protects you.

Expected output:

    1. model stops asking
       2 turns -> found it: chapter 6

    2. the recursion limit, measured not assumed
       default recursion_limit = 10007
       ~1s of doing nothing useful

    3. same graph, recursion_limit=6
       GraphRecursionError, not an answer

    4. your own budget: stop at 3 turns
       [turn 1] tool call
       [turn 2] tool call
       [turn 3] budget spent -> stopping
       3 turns -> out of turns; best: chapter 6
"""

import re
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from scripted_model import ScriptedModel, tool_call

ASK = AIMessage(content="", tool_calls=[tool_call("search", {"q": "x"}, "c")])
DONE = AIMessage(content="found it: chapter 6")

MAX_TURNS = 3


class State(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int


def build(script: list[AIMessage], budget: bool = False):
    """The same two-node graph every time; only the router changes."""
    model = ScriptedModel(script=script)

    def agent(state: State) -> dict:
        return {
            "messages": [model.invoke(state["messages"])],
            "turns": state["turns"] + 1,
        }

    def tools(state: State) -> dict:
        return {
            "messages": [
                ToolMessage(content="chapter 6", tool_call_id=c["id"])
                for c in state["messages"][-1].tool_calls
            ]
        }

    def out_of_turns(state: State) -> dict:
        """A real answer, not an exception. This is the point."""
        return {
            "messages": [
                AIMessage(content="out of turns; best: chapter 6")
            ]
        }

    def should_continue(state: State) -> str:
        if not state["messages"][-1].tool_calls:
            return "done"
        if budget and state["turns"] >= MAX_TURNS:
            print(f"   [turn {state['turns']}] budget spent -> stopping")
            return "give_up"
        if budget:
            print(f"   [turn {state['turns']}] tool call")
        return "tools"

    b = StateGraph(State)
    b.add_node("agent", agent)
    b.add_node("tools", tools)
    b.add_edge(START, "agent")
    b.add_edge("tools", "agent")

    routes = {"tools": "tools", "done": END}
    if budget:
        b.add_node("give_up", out_of_turns)
        b.add_edge("give_up", END)
        routes["give_up"] = "give_up"
    b.add_conditional_edges("agent", should_continue, routes)
    return b.compile()


def run(graph, **kw) -> dict:
    return graph.invoke(
        {"messages": [HumanMessage("which chapter?")], "turns": 0}, **kw
    )


# 1 ------------------------------------------------ the model stops asking
print("1. model stops asking")
final = run(build([ASK, DONE]))
print(f"   {final['turns']} turns -> {final['messages'][-1].content}")

# 2 ------------------------------------- what the limit really is, measured
print("\n2. the recursion limit, measured not assumed")
spin = StateGraph(TypedDict("Tick", {"n": int}))
spin.add_node("tick", lambda s: {"n": s["n"] + 1})
spin.add_edge(START, "tick")
spin.add_edge("tick", "tick")

t0 = time.time()
try:
    spin.compile().invoke({"n": 0})
except GraphRecursionError as exc:
    # Read off the library's own error rather than hard-coding it.
    limit = int(re.search(r"limit of (\d+)", str(exc)).group(1))
    print(f"   default recursion_limit = {limit}")
print(f"   ~{time.time() - t0:.0f}s of doing nothing useful")

# 3 -------------------------------------- what hitting it gets you: nothing
print("\n3. same graph, recursion_limit=6")
try:
    run(build([ASK]), config={"recursion_limit": 6})
except GraphRecursionError:
    print("   GraphRecursionError, not an answer")

# 4 --------------------------------------------- a budget you enforce yourself
print("\n4. your own budget: stop at 3 turns")
final = run(build([ASK], budget=True))
print(f"   {final['turns']} turns -> {final['messages'][-1].content}")
