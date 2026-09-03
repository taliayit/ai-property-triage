import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_THREADS,
    LLM_TEMPERATURE,
    MAX_RELEVANT_DISTANCE,
    PERSIST_DIRECTORY,
)

logger = logging.getLogger(__name__)

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
collection = None
insight_chain = None

CITATION_PATTERN = re.compile(r"\[Listing (\d+)\]")

# --- Insight prompt ---
# A single fixed few-shot example is baked in as prior chat turns (not a
# template variable) to anchor the citation format for a small model.
INSIGHT_SYSTEM_PROMPT = (
    "You are a real estate assistant that compares a new property submission "
    "to similar listings retrieved from a database and writes a short insight "
    "for a real estate agent.\n\n"
    "Rules you must follow exactly:\n"
    "1. Use ONLY facts stated in the listings you're given. Never invent a "
    "price, feature, room count, or location that isn't explicitly written "
    "in one of the listings.\n"
    "2. Every specific claim about a listing must be immediately followed by "
    "its citation tag, e.g. [Listing 1].\n"
    "3. Only cite a listing if it's genuinely relevant to the new submission.\n"
    "4. If none of the listings are a good match, say so plainly instead of "
    "forcing a comparison.\n"
    "5. Write 2-4 sentences. No preamble, no headers - just the insight.\n"
    "6. The new submission's own price, room count, and features are ONLY "
    "what's written in its \"New submission\" text. Never state a price, "
    "room count, or feature for the submission itself unless it appears "
    "there - even if a similar number or feature appears in one of the "
    "listings. If you want to mention that a listing has a certain price "
    "or feature, attribute it explicitly to that listing, not to the "
    "submission.\n"
    "7. Never say the submission and a listing \"share\", \"both have\", or "
    "\"both feature\" something unless that exact thing is written in the "
    "submission text. If a feature is only in the listing, say \"[Listing "
    "N] has...\" - don't imply the submission has it too."
)

TASK_TEMPLATE = (
    'New submission: "{query_description}"\nListings:\n{listings_block}\n'
    "(Remember: cite every listing you reference using its exact tag, e.g. "
    "[Listing 1].)\nInsight:"
)

EXAMPLE_LISTINGS = [
    {
        "property_type": "apartment",
        "price": 210000,
        "rooms": 2,
        "description": "Updated 1-bedroom condo with a renovated bathroom, close to the harbor.",
    },
    {
        "property_type": "house",
        "price": 610000,
        "rooms": 6,
        "description": "Large family home with a pool and a three-car garage.",
    },
]
EXAMPLE_INSIGHT = (
    "This submission is closely comparable to [Listing 1], which shares the "
    "renovated bathroom and a similar 1-bedroom layout near the water, priced "
    "at $210,000. [Listing 2] is a much larger family home and isn't a "
    "meaningful comparison for this submission."
)


def format_listings_block(listings: list[dict]) -> str:
    lines = []
    for i, listing in enumerate(listings, start=1):
        lines.append(
            f"[Listing {i}] ({listing['property_type']}, ${listing['price']:,}, "
            f"{listing['rooms']} rooms): \"{listing['description']}\""
        )
    return "\n".join(lines)


insight_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", INSIGHT_SYSTEM_PROMPT),
        (
            "human",
            TASK_TEMPLATE.format(
                query_description="Cozy 1-bedroom condo with an updated bathroom near the waterfront.",
                listings_block=format_listings_block(EXAMPLE_LISTINGS),
            ),
        ),
        ("ai", EXAMPLE_INSIGHT),
        ("human", TASK_TEMPLATE),
    ]
)


def log_citation_issues(insight_text: str, num_listings: int) -> None:
    """Mechanical guard, logged only - not part of the /query response schema.

    Only catches out-of-range [Listing N] tags; see
    docs/prompt-engineering-log.md for the fabrication modes it misses
    (e.g. misattributing a retrieved listing's price to the submission).
    """
    cited = {int(n) for n in CITATION_PATTERN.findall(insight_text)}
    invalid = sorted(n for n in cited if n < 1 or n > num_listings)
    if invalid:
        logger.warning(
            "Generated insight cites listing number(s) not in the retrieved set: %s",
            invalid,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection, insight_chain

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        logger.warning(
            "Collection '%s' not found. Run seed_data.py before querying.",
            COLLECTION_NAME,
        )

    if Path(LLM_MODEL_PATH).exists():
        llm = ChatLlamaCpp(
            model_path=LLM_MODEL_PATH,
            chat_format="llama-3",
            n_ctx=LLM_N_CTX,
            n_threads=LLM_N_THREADS,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stop=["\nNew submission:", "\n\nNew submission:"],
            verbose=False,
        )
        insight_chain = insight_prompt | llm | StrOutputParser()
    else:
        logger.warning(
            "LLM model not found at '%s'. Set RAG_LLM_MODEL_PATH or place a "
            "GGUF file there before querying.",
            LLM_MODEL_PATH,
        )

    yield


app = FastAPI(title="RAG Service", lifespan=lifespan)


class QueryRequest(BaseModel):
    description: str


def generate_insight(query_description: str, similar_listings: list[dict]) -> str:
    if not similar_listings:
        return "No similar listings were found in the knowledge base."

    if similar_listings[0]["distance"] > MAX_RELEVANT_DISTANCE:
        return "No sufficiently similar listings were found for this submission."

    raw = insight_chain.invoke(
        {
            "query_description": query_description,
            "listings_block": format_listings_block(similar_listings),
        }
    )
    text = raw.content if isinstance(raw, AIMessage) else raw
    text = text.strip()
    log_citation_issues(text, len(similar_listings))
    return text


@app.post("/query")
def query(request: QueryRequest):
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="description must not be empty")

    if collection is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not seeded yet. Run seed_data.py first.",
        )

    if insight_chain is None:
        raise HTTPException(
            status_code=503,
            detail=f"LLM model not found at '{LLM_MODEL_PATH}'. Set RAG_LLM_MODEL_PATH or place a GGUF file there.",
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
        "insight": generate_insight(request.description, similar_listings),
    }
