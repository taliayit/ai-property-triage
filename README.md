# Property Triage

Local dev stack for the AI-powered real estate property triage system.
See `CLAUDE.md` for the full project context and architecture.

## Quick start

```bash
docker compose up -d --build
```

This builds and starts all five services on a shared `property-triage`
network:

| Service               | Container name       | Host port | Compose network address                |
|------------------------|----------------------|-----------|-------------------------------------------|
| n8n                    | `n8n`                | 5678      | `http://n8n:5678`                         |
| RAG service            | `rag-service`         | 8001      | `http://rag-service:8001`                 |
| Guardrails service     | `guardrails-service`  | 8002      | `http://guardrails-service:8002`          |
| Image Analyser         | `image-analyser`      | 8003      | `http://image-analyser:8003`              |
| LangGraph Agent        | `langgraph-agent`     | 8004      | `http://langgraph-agent:8004`             |

n8n's workflow HTTP Request nodes should call the other services by
their **compose network address** above (e.g.
`http://rag-service:8001/query`), not `host.docker.internal` — all
containers share one Docker network and can reach each other by service
name.

## Before first run

`rag-service` needs a local GGUF model that isn't part of the repo (see
[services/rag-service/README.md](services/rag-service/README.md)):

```
services/rag-service/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

`docker compose up` mounts that directory into the container read-only;
without the file present, `rag-service` still starts and serves
retrieval, but `/query`'s insight generation returns `503` until the
model is added.

`langgraph-agent` and `guardrails-service` both need a Gemini API key.
Create a `.env` file in the project root (gitignored - never commit it)
containing:

```
GEMINI_API_KEY=your-key-here
```

`docker-compose.yml` reads it automatically for both services. Without
it, both still start, but `langgraph-agent`'s `/agent/run` and
`guardrails-service`'s `/check/input`/`/check/output` all return `503`
until the key is set.

## Data persistence

- `n8n` uses the `n8n_data` Docker volume - the same one your existing
  manual n8n container already uses (declared `external` in
  `docker-compose.yml`), so your workflows and credentials carry over.
- `rag-service`'s ChromaDB data lives in the `rag_chroma_data` named
  volume. It's auto-populated from the image's build-time-seeded data the
  first time the volume is created; see
  [services/rag-service/README.md](services/rag-service/README.md) for
  how to force a re-seed after changing `seed_data.py`.
- `guardrails-service`, `image-analyser`, and `langgraph-agent` are all
  stateless - no volume needed for any of them. `image-analyser`'s
  trained checkpoint is baked into its image at build time (see its
  README), not mounted.

## Per-service docs

- [services/rag-service/README.md](services/rag-service/README.md)
- [services/guardrails-service/README.md](services/guardrails-service/README.md)
- [services/image-analyser/README.md](services/image-analyser/README.md)
- [services/langgraph-agent/README.md](services/langgraph-agent/README.md)
- [docs/prompt-engineering-log.md](docs/prompt-engineering-log.md) - RAG
  insight-generation prompt iteration history and evaluation results.

## Stopping / cleaning up

```bash
docker compose down          # stop and remove containers, keep volumes
docker compose down -v       # also remove rag_chroma_data (forces re-seed on next up)
```

`n8n_data` is `external` and is never removed by `docker compose down -v`.
