import logging
from collections import Counter
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY

logger = logging.getLogger(__name__)

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        logger.warning(
            "Collection '%s' not found. Run seed_data.py before querying.",
            COLLECTION_NAME,
        )
    yield


app = FastAPI(title="RAG Service", lifespan=lifespan)


class QueryRequest(BaseModel):
    description: str


def build_insight(metadatas: list[dict], distances: list[float]) -> str:
    if not metadatas:
        return "No similar listings were found in the knowledge base."

    type_counts = Counter(m["property_type"] for m in metadatas)
    dominant_type, dominant_count = type_counts.most_common(1)[0]
    prices = [m["price"] for m in metadatas]
    rooms = [m["rooms"] for m in metadatas]

    return (
        f"Found {len(metadatas)} similar listings, {dominant_count} of them "
        f"{dominant_type} properties. Prices range from ${min(prices):,} to "
        f"${max(prices):,}, with room counts between {min(rooms)} and "
        f"{max(rooms)}. The closest match has a similarity distance of "
        f"{distances[0]:.3f}."
    )


@app.post("/query")
def query(request: QueryRequest):
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="description must not be empty")

    if collection is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not seeded yet. Run seed_data.py first.",
        )

    embedding = model.encode([request.description]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=3)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    similar_listings = [
        {
            "description": doc,
            "property_type": meta["property_type"],
            "price": meta["price"],
            "rooms": meta["rooms"],
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]

    return {
        "similar_listings": similar_listings,
        "insight": build_insight(metadatas, distances),
    }
