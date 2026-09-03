# RAG Service

Finds similar property listings for a given free-text description
(sentence-transformer embeddings + a local ChromaDB vector store), then
uses a local LLM (via LangChain + llama-cpp-python) to write a short,
citation-grounded insight comparing the submission to what it retrieved.

## Setup (without Docker)

```bash
cd services/rag-service
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### GGUF model

Insight generation needs a local GGUF chat model. Place it at:

```
services/rag-service/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Using a different file or filename? Point `RAG_LLM_MODEL_PATH` at it instead:

```bash
set RAG_LLM_MODEL_PATH=C:\path\to\your-model.gguf   # Windows
```

If the file isn't found at startup, `/query` returns `503` rather than
crashing the service — the knowledge-base retrieval side doesn't need the
LLM, only insight generation does.

## Seed the knowledge base

Populates ChromaDB with 20 synthetic listings (apartments, houses,
commercial). Only needs to be run once — data persists in `chroma_db/`.

```bash
python seed_data.py
```

## Run the service

```bash
uvicorn main:app --reload --port 8001
```

## Docker

The image needs a C/C++ toolchain to compile `llama-cpp-python`, which a
slim runtime image doesn't carry — the [Dockerfile](Dockerfile) handles
this with a multi-stage build: a `builder` stage with `build-essential` +
`cmake` produces wheels, and the final `python:3.11-slim` stage only
installs those prebuilt wheels, so the compiler toolchain never ends up in
the shipped image. The synthetic listings are seeded **at build time**, so
the container can serve `/query` immediately on start.

The GGUF model is deliberately **not** baked into the image (it's hundreds
of MB to several GB) — mount it at runtime instead:

```bash
docker build -t rag-service .

docker run -d -p 8001:8001 \
  -v "$(pwd)/models:/app/models:ro" \
  rag-service
```

`chroma_db/` lives inside the image from the build-time seed step; add a
`-v $(pwd)/chroma_db:/app/chroma_db` mount if you want re-seeding to
persist across container restarts.

## Test it

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d "{\"description\": \"Newly built 5-bedroom family home with an open-concept living area, modern kitchen island, and a two-car garage. Energy efficient with solar panels installed.\"}"
```

Response shape:

```json
{
  "similar_listings": [
    {
      "description": "Newly built 5-bedroom family home with an open-concept living area, modern kitchen island, and a two-car garage. Energy efficient with solar panels installed.",
      "property_type": "house",
      "price": 615000,
      "rooms": 6,
      "distance": 0.0
    }
  ],
  "insight": "This new submission is comparable to [Listing 1], a newly built 5-bedroom family home with an open-concept living area, modern kitchen island, and a two-car garage."
}
```

If the closest retrieved listing is too dissimilar to be worth comparing
against (distance above `MAX_RELEVANT_DISTANCE` in `config.py`), the LLM
is skipped entirely and `insight` is a canned
`"No sufficiently similar listings were found for this submission."` —
this is a deterministic guard, not something the model decides for
itself.

Every generated insight is also checked server-side for `[Listing N]`
citations that don't correspond to an actually-retrieved listing, and a
warning is logged when found — this is a mechanical guard, not a
fact-checker, and isn't part of the response. It **cannot** catch every
way the model can violate the "don't fabricate" rule — see
[docs/prompt-engineering-log.md](../../docs/prompt-engineering-log.md)
for the full iteration history and a measured **7/12 pass rate** on a
test suite covering exact matches, off-domain queries, and generic
multi-candidate queries (the last category is where it still fails most
often — inventing or implying a submission feature that only exists in a
retrieved listing). Treat `insight` as a draft for a human to review, not
a verified fact sheet.
