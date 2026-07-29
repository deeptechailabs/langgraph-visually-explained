# LangGraph, Visually Explained

Runnable companion code for the **DEEPTECH AI LABS** LangGraph course.
Each chapter starts with a visual mental model, then proves it with small
Python programs you can run locally.

The published examples use no model, API key, or paid service.

## Published chapters

| Chapter | Topic | Video | Code |
|---|---|---|---|
| 1 | Why agents need graphs: state, nodes, edges, loops | [Watch Chapter 1](https://youtu.be/GWBM2Ueqp6o) | [Open the examples](chapters/01-why-graphs) |
| 2 | State, channels, reducers, `add_messages` | [Watch Chapter 2](https://youtu.be/IfGZrTuZDNA) | [Open the examples](chapters/02-state-and-reducers) |

New chapter code will be added when its video is published, so the repository
and the playlist stay in sync.

## Quick start

Python 3.11 or newer is recommended. The published examples are verified with
Python 3.14.3.

```bash
git clone https://github.com/deeptechailabs/langgraph-visually-explained.git
cd langgraph-visually-explained

python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the versions used to verify the examples:

```bash
python -m pip install -r requirements.txt
```

Run one example:

```bash
python chapters/01-why-graphs/code/02_first_graph.py
```

Or check every published example:

```bash
python run_examples.py
```

`chapters/02-state-and-reducers/code/06_parallel_crash.py` fails intentionally.
It demonstrates the `InvalidUpdateError` raised when parallel nodes write to a
channel that has no reducer. The verification runner treats that expected error
as a pass.

## Course roadmap

1. Why graphs?
2. State, channels, and reducers
3. Nodes and edges in depth
4. Build a ReAct agent from scratch
5. Prebuilt agents and tool routing
6. Persistence and memory
7. Human in the loop
8. Streaming
9. Subgraphs and multi-agent systems
10. Time travel
11. Reliability
12. Observability and deployment
13. Capstone research agent

## Tested environment

- Python 3.14.3
- `langgraph==1.2.9`
- `langchain-core==1.5.1`

If an example behaves differently on a newer release, open an issue and include
your Python and LangGraph versions.

## Repository policy

This repository contains only code for lessons that have already been
published. Video-production files, voice tracks, renders, private working
notes, and unreleased chapters are intentionally excluded.
