# RAG Service

Finds similar property listings for a given free-text description, using
sentence-transformer embeddings and a local ChromaDB vector store.

## Setup

```bash
cd services/rag-service
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Seed the knowledge base

Populates ChromaDB with 12 synthetic listings (apartments, houses,
commercial). Only needs to be run once — data persists in `chroma_db/`.

```bash
python seed_data.py
```

## Run the service

```bash
uvicorn main:app --reload --port 8001
```

## Test it

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d "{\"description\": \"Modern 2-bedroom apartment near downtown with a renovated kitchen and balcony\"}"
```

Response shape:

```json
{
  "similar_listings": [
    {
      "description": "...",
      "property_type": "apartment",
      "price": 285000,
      "rooms": 3,
      "distance": 0.42
    }
  ],
  "insight": "Found 3 similar listings, 2 of them apartment properties. ..."
}
```
