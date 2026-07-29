"""Scene 1 -- a tool is a schema, and binding calls nothing.

Two facts that surprise people:

  1. `@tool` does not give the model your function
     implementation. It exposes a model-facing schema derived
     from the tool name, description and typed arguments.
  2. `bind_tools` does not call anything and does not change
     the model. It returns a NEW model that knows the
     schemas. Nothing executes.

When the model decides to use one, it does not run it either.
It returns an ordinary AIMessage whose `tool_calls` list says
"please run this". Running it is YOUR job -- which is exactly
the job the graph in this chapter does.

Expected output:

    what the model is given:
      search(['query']) -- Look up a fact.
      add(['a', 'b']) -- Add two integers.

    bind_tools returned a new model: True
    it knows about: ['search', 'add']
    calls made so far: 0

    the model replied with a REQUEST:
      content  ''
      call.name  'search'
      call.args  {'query': 'graphs'}
      call.id    'c1'
      call.type  'tool_call'

    nothing ran until we ran it ourselves:
      results for graphs
"""

from langchain_core.tools import tool

from scripted_model import ScriptedModel, tool_call
from langchain_core.messages import AIMessage, HumanMessage


@tool
def search(query: str) -> str:
    """Look up a fact."""
    return f"results for {query}"


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


print("what the model is given:")
for t in (search, add):
    print(f"  {t.name}({list(t.args.keys())}) -- {t.description}")

model = ScriptedModel(
    script=[
        AIMessage(
            content="",
            tool_calls=[tool_call("search", {"query": "graphs"}, "c1")],
        )
    ]
)

bound = model.bind_tools([search, add])
print(f"\nbind_tools returned a new model: {bound is not model}")
print(f"it knows about: {bound.bound}")
print(f"calls made so far: {bound.calls}")

reply = bound.invoke([HumanMessage("what is a graph?")])
print("\nthe model replied with a REQUEST:")
print(f"  content  {reply.content!r}")
# A field at a time, because the whole dict on one line is the
# fastest way to make a tool call look more mysterious than it is.
# Four keys, all of them ordinary data.
for key, value in reply.tool_calls[0].items():
    print(f"  call.{key:<5} {value!r}")

print("\nnothing ran until we ran it ourselves:")
print(f"  {search.invoke({'query': 'graphs'})}")
