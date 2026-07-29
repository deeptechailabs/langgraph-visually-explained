"""Scene 6 -- the research assistant, with a model in the loop.

Chapter 3 left this agent with a planner that decided how many
searches to run by checking a length. Everything else about the
wiring survives; the planner is now a language model, and the
graph got SMALLER rather than bigger:

    chapter 3      plan -> Send fan-out -> gather -> Command loop
    chapter 4      agent -> tools -> agent

A model message can contain a runtime-sized list of tool_calls.
This teaching tool node processes every call sequentially. It
demonstrates variable-width requests, not parallel execution.
Production ToolNode adds concurrency and other capabilities.

Three things worth watching in the trace:

  * Turn 1 contains TWO search requests. One tools-node visit
    returns two results, executed sequentially by this runner.
  * The scripted turn-2 response follows "checkpointers" from a
    turn-1 result. ScriptedModel replays this trace; it does not
    infer the follow-up from the messages.
  * Turn 3 has no tool_calls, so should_continue routes to END.
    Nothing counted to three.

A MAX_TURNS guard is present and does not fire in this scripted
run. The third response has no tool_calls, so the normal route
reaches END.

Using a live model additionally requires a provider integration,
credentials and a current tool-capable model identifier. Replace
ScriptedModel with that chat model and bind TOOLS; the graph
wiring can remain the same.

Expected output:

    [turn 1] 2 tool calls
       search('persistence')
       search('streaming')
    [turn 2] 1 tool call
       search('checkpointers')
    [turn 3] no tool calls -> END

    conversation:
      Human     research persistence and streaming
      AI        -> search({'q': 'persistence'})
      AI        -> search({'q': 'streaming'})
      Tool      checkpointers save thread state
      Tool      stream modes: values, updates
      AI        -> search({'q': 'checkpointers'})
      Tool      SqliteSaver writes one file
      AI        report: checkpointers over threads

    3 agent turns, 3 tool calls, 5 nodes visited
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from scripted_model import ScriptedModel, show, tool_call

NOTES = {
    "persistence": "checkpointers save thread state",
    "streaming": "stream modes: values, updates",
    "checkpointers": "SqliteSaver writes one file",
}

MAX_TURNS = 6


@tool
def search(q: str) -> str:
    """Search the LangGraph docs for a topic."""
    return NOTES.get(q, "no result")


TOOLS = [search]
REGISTRY = {t.name: t for t in TOOLS}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int


# A deterministic trace representing a follow-up on a term from
# turn 1. ScriptedModel replays it; it does not reason over input.
model = ScriptedModel(
    script=[
        AIMessage(
            content="",
            tool_calls=[
                tool_call("search", {"q": "persistence"}, "t1"),
                tool_call("search", {"q": "streaming"}, "t2"),
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[tool_call("search", {"q": "checkpointers"}, "t3")],
        ),
        AIMessage(content="report: checkpointers over threads"),
    ]
).bind_tools(TOOLS)


def agent(state: State) -> dict:
    """Think. One model call, whatever it decides to ask for."""
    reply = model.invoke(state["messages"])
    n = len(reply.tool_calls)
    turn = state["turns"] + 1
    print(f"[turn {turn}] {n} tool call{'s' * (n != 1)}" if n
          else f"[turn {turn}] no tool calls -> END")
    return {"messages": [reply], "turns": turn}


def tools(state: State) -> dict:
    """Act. Scene 3's tool node, unchanged."""
    out = []
    for call in state["messages"][-1].tool_calls:
        print(f"   {call['name']}({call['args']['q']!r})")
        fn = REGISTRY.get(call["name"])
        try:
            text = str(fn.invoke(call["args"]))
        except Exception as exc:
            text = f"error: {type(exc).__name__}: {exc}"
        out.append(ToolMessage(content=text, tool_call_id=call["id"]))
    return {"messages": out}


def should_continue(state: State) -> str:
    """Observe. The only thing that decides another lap."""
    if not state["messages"][-1].tool_calls:
        return "done"
    return "done" if state["turns"] >= MAX_TURNS else "tools"


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "done": END}
)
builder.add_edge("tools", "agent")

graph = builder.compile()

final = graph.invoke(
    {
        "messages": [HumanMessage("research persistence and streaming")],
        "turns": 0,
    }
)

print("\nconversation:")
show(final["messages"])

calls = sum(len(getattr(m, "tool_calls", []) or []) for m in final["messages"])
# agent ran once per turn; tools ran once per turn that asked for anything.
rounds = sum(1 for m in final["messages"] if getattr(m, "tool_calls", None))
print(
    f"\n{final['turns']} agent turns, {calls} tool calls, "
    f"{final['turns'] + rounds} nodes visited"
)
