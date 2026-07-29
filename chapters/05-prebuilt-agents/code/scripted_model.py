"""A deterministic stand-in for a chat model.

Every other file in this chapter imports this. It implements
BaseChatModel, but instead of calling an API it replays a fixed
script and ignores the semantic content of incoming messages.

Why: this chapter is about the LOOP around the model, not
about the model. A scripted one means the terminal output in
the video is exactly what you get when you run the file, and
it costs nothing to run. To use a provider, install its
integration, configure credentials, and choose a current
tool-capable model:

    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="<YOUR_MODEL_ID>")

Then bind the lesson's tools. The wiring is the lesson.
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
        # Which turn is this? Count the AI messages already in the
        # conversation instead of keeping a counter on the object.
        #
        # This matters here in a way it did not in chapter 4.
        # `create_agent` re-binds the model on every step, and
        # bind_tools returns a NEW instance -- so a counter living on
        # the object resets each turn, the script rewinds to its first
        # entry, and the agent asks for the same tool forever. Reading
        # the position out of the messages is immune to that, and is
        # closer to what a real model does anyway: it is handed the
        # whole conversation and answers the end of it.
        turn = sum(1 for m in messages if isinstance(m, AIMessage))
        i = min(turn, len(self.script) - 1)
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
