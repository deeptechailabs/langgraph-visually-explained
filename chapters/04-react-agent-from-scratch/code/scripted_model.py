"""A deterministic stand-in for a chat model.

Every other file in this chapter imports this. It is a real
BaseChatModel -- LangGraph cannot tell the difference -- but
instead of calling an API it replays a fixed script.

Why: this chapter is about the LOOP around the model, not
about the model. A scripted one means the terminal output in
the video is exactly what you get when you run the file, and
it costs nothing to run. Swap it for a provider and not one
line of the graph changes:

    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="claude-sonnet-5")

That is the only edit. The wiring is the lesson.
"""

from typing import Any, Iterator, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedModel(BaseChatModel):
    """Replays `script` one AIMessage per call."""

    script: list[AIMessage]
    calls: int = 0
    # Set by bind_tools. The script does not consult it; it
    # exists so you can see that binding is just attaching a
    # list of schemas to the model.
    bound: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = min(self.calls, len(self.script) - 1)
        self.calls += 1
        msg = self.script[i]
        # A fresh copy each turn: LangGraph appends what we
        # return to the state, and handing back the same
        # object twice would alias it into the history.
        out = AIMessage(
            content=msg.content,
            tool_calls=list(msg.tool_calls),
        )
        return ChatResult(generations=[ChatGeneration(message=out)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        """Attach tool schemas. This does NOT call anything."""
        names = [getattr(t, "name", str(t)) for t in tools]
        return self.__class__(script=self.script, bound=names)


def tool_call(name: str, args: dict, id: str) -> dict:
    """The shape a model emits when it wants a tool run."""
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def show(messages: Iterator[BaseMessage] | list[BaseMessage]) -> None:
    """Print a message list the way the video shows it.

    Content is clipped to 36 characters so no line can exceed the
    50-column terminal panel the scenes render into.
    """
    for m in messages:
        kind = m.__class__.__name__.replace("Message", "")
        if getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                print(f"  {kind:9} -> {c['name']}({c['args']})")
        else:
            text = str(m.content).replace("\n", " ")
            print(f"  {kind:9} {text[:36]}")
