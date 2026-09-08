# LangGraph Agent

Multi-step reasoning over complex questions about a listing, orchestrating
`rag-service` and `image-analyser` as tools. Served as `POST /agent/run`.

```bash
cd services/langgraph-agent
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

set GEMINI_API_KEY=your-key-here   # Windows
uvicorn main:app --reload --port 8004
```

## LLM Model

`gemini-3.5-flash-lite` (native tool-calling required for the planner node).

## Graph

Three nodes, a single-pass pipeline (no re-planning loop):

```
planner --[tool calls needed?]--> tool_executor --> synthesiser --> END
   \_________________ no tools needed __________________/
```

- `planner`: decides which tool(s) - zero, one, or several - the query
  needs, via function-calling. Sees only the query and the tool
  descriptions, not any tool results.
- `tool_executor`: runs every tool call the planner requested (each a
  direct HTTP call to `rag-service` or `image-analyser`). Skipped via a
  conditional edge if the planner requested no tools.
- `synthesiser`: a second Gemini call that writes the final answer from
  the query and any tool results, grounding claims in the tool output.

## Tool descriptions

`tools.py`'s docstrings are what the planner's function-calling reads to
select a tool. Each states what the tool does, its input, when to use
it, and what it does not do. See `docs/prompt-engineering-log.md`
("LangGraph Agent Tool Selection") for the v1 test results.

## `POST /agent/run`

```json
{"query": "Is $850,000 reasonable for a newly built 5-bedroom house with solar panels?"}
```

```json
{
  "answer": "...",
  "tools_used": ["search_similar_listings"],
  "reasoning_steps": [
    "Planner decided to call: search_similar_listings",
    "Called search_similar_listings({...}) -> ...",
    "Synthesiser combined results into a final answer."
  ]
}
```

`400` for an empty query. `503` if `GEMINI_API_KEY` isn't set - the
service still starts rather than crashing.

## Docker

```bash
docker build -t langgraph-agent .
docker run -d -p 8004:8004 -e GEMINI_API_KEY=your-key-here langgraph-agent
```

Via `docker compose up` from the project root, it reads `GEMINI_API_KEY`
from a `.env` file in the project root (gitignored) or your shell
environment - see the root README.
