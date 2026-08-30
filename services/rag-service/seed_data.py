"""Creates and populates the ChromaDB collection with synthetic listings.

Run once before starting the API: `python seed_data.py`
Safe to re-run — it recreates the collection from scratch each time.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY

LISTINGS = [
    {
        "property_type": "apartment",
        "description": (
            "Bright 2-bedroom apartment on the 4th floor with a renovated "
            "kitchen, hardwood floors, and a small balcony overlooking the "
            "courtyard. Walking distance to the metro station."
        ),
        "price": 285000,
        "rooms": 3,
    },
    {
        "property_type": "apartment",
        "description": (
            "Cozy studio apartment in a converted warehouse building, exposed "
            "brick walls, high ceilings, and large industrial windows. Ideal "
            "for a young professional."
        ),
        "price": 165000,
        "rooms": 1,
    },
    {
        "property_type": "apartment",
        "description": (
            "Spacious 3-bedroom penthouse with a private rooftop terrace, "
            "panoramic city views, central air conditioning, and a dedicated "
            "parking spot in the building garage."
        ),
        "price": 520000,
        "rooms": 4,
    },
    {
        "property_type": "house",
        "description": (
            "Charming 4-bedroom single-family house with a fenced backyard, "
            "detached garage, and a recently updated roof. Quiet suburban "
            "neighborhood close to good schools."
        ),
        "price": 410000,
        "rooms": 5,
    },
    {
        "property_type": "house",
        "description": (
            "Newly built 5-bedroom family home with an open-concept living "
            "area, modern kitchen island, and a two-car garage. Energy "
            "efficient with solar panels installed."
        ),
        "price": 615000,
        "rooms": 6,
    },
    {
        "property_type": "house",
        "description": (
            "Rustic 3-bedroom cottage on a large wooded lot, wood-burning "
            "fireplace, wraparound porch, and a detached workshop. Great for "
            "buyers wanting privacy and outdoor space."
        ),
        "price": 320000,
        "rooms": 4,
    },
    {
        "property_type": "house",
        "description": (
            "Split-level 4-bedroom house with a finished basement, in-ground "
            "pool, and a landscaped garden. Needs some cosmetic updates but "
            "structurally sound."
        ),
        "price": 355000,
        "rooms": 5,
    },
    {
        "property_type": "commercial",
        "description": (
            "Ground-floor retail unit on a busy high street, large storefront "
            "windows, storage room in the back, and existing HVAC system. "
            "Previously operated as a boutique clothing store."
        ),
        "price": 480000,
        "rooms": 2,
    },
    {
        "property_type": "commercial",
        "description": (
            "Open-plan office space on the 6th floor of a modern high-rise, "
            "suitable for a team of 20-30, with meeting rooms, a kitchenette, "
            "and 24/7 building access."
        ),
        "price": 890000,
        "rooms": 8,
    },
    {
        "property_type": "commercial",
        "description": (
            "Standalone warehouse with loading dock access, 6-meter ceiling "
            "height, and an attached office area. Zoned for light industrial "
            "use, close to the highway interchange."
        ),
        "price": 750000,
        "rooms": 3,
    },
    {
        "property_type": "apartment",
        "description": (
            "Compact 1-bedroom apartment with a modern bathroom, in-unit "
            "laundry, and access to a shared rooftop garden. Low HOA fees "
            "and pet-friendly building."
        ),
        "price": 195000,
        "rooms": 2,
    },
    {
        "property_type": "commercial",
        "description": (
            "Restaurant space with a full commercial kitchen, walk-in cooler, "
            "and seating for 60 guests. Includes existing liquor license and "
            "outdoor patio area."
        ),
        "price": 560000,
        "rooms": 4,
    },
]


def seed() -> None:
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    descriptions = [listing["description"] for listing in LISTINGS]
    embeddings = model.encode(descriptions).tolist()

    collection.add(
        ids=[f"listing-{i}" for i in range(len(LISTINGS))],
        embeddings=embeddings,
        documents=descriptions,
        metadatas=[
            {
                "property_type": listing["property_type"],
                "price": listing["price"],
                "rooms": listing["rooms"],
            }
            for listing in LISTINGS
        ],
    )

    print(f"Seeded {len(LISTINGS)} listings into '{COLLECTION_NAME}' at {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    seed()
