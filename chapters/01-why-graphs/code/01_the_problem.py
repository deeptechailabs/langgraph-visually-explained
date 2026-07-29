"""Scene 2 — the chain, and the wall it hits.

Plain Python. No LangGraph yet. This is the "before" picture.
"""


def research(topic):
    return f"Notes on {topic}."


def review(draft):
    return len(draft)


def revise(draft):
    return draft + " More detail added."


draft = research("graph based agents")
score = review(draft)
result = revise(draft)

# score was low. now go back to research.
#   ...we cannot. the next line is already written.
#
#   - revise always runs, even when the draft was fine
#   - review can never run twice
#   - nothing here can pause and wait for a human
#
# the control flow is the order of these lines,
# and the order of these lines never changes.

print(result)
