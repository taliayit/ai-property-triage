"""Tool wrappers around rag-service and image-analyser. Docstrings are
read by the planner's function-calling - see README."""

import requests
from langchain_core.tools import tool

from config import IMAGE_ANALYSER_URL, RAG_SERVICE_URL, TOOL_HTTP_TIMEOUT_SECONDS


@tool
def search_similar_listings(description: str) -> str:
    """Find similar property listings and get an AI-written comparison
    insight, given a TEXT description of a property (features, price,
    room count, location, etc - not a photo).

    Use this for questions about how a property compares to similar
    listings on the market, whether an asking price seems reasonable, or
    what comparable properties look like.

    Does NOT look at photos or judge a room's physical condition - for
    that, use analyse_room_photo instead.
    """
    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/query",
            json={"description": description},
            timeout=TOOL_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"Error calling rag-service: {exc}"

    data = response.json()
    listings = data.get("similar_listings", [])
    lines = [f"Insight: {data.get('insight', '')}", f"Found {len(listings)} similar listing(s):"]
    for listing in listings:
        lines.append(
            f"- {listing['property_type']}, ${listing['price']:,}, {listing['rooms']} rooms "
            f"(similarity distance {listing['distance']:.2f}): \"{listing['description']}\""
        )
    return "\n".join(lines)


@tool
def analyse_room_photo(image_url: str) -> str:
    """Classify a single room PHOTO by room type (bathroom, bedroom,
    kitchen, living_room, exterior, or other) and assess its physical
    condition on a 1-5 scale (1 = very poor/needs major repairs, 5 =
    excellent/pristine), given a direct URL to one image.

    Use this for questions about what a specific photo shows, or about a
    room's condition, renovation state, or how much work it needs. If a
    question involves multiple photos (e.g. "which of these rooms needs
    the most attention"), call this tool once per photo/URL.

    Does NOT compare properties, read text descriptions, or accept more
    than one image per call - for property-level comparisons, use
    search_similar_listings instead.
    """
    try:
        response = requests.post(
            f"{IMAGE_ANALYSER_URL}/analyse",
            json={"image_url": image_url},
            timeout=TOOL_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"Error calling image-analyser: {exc}"

    data = response.json()
    if data["room_type"] == "uncertain":
        return (
            f"Could not confidently classify this image "
            f"(confidence {data['confidence']:.2f}, below threshold)."
        )
    return (
        f"Room type: {data['room_type']}. Condition score: {data['condition_score']}/5 "
        f"(confidence {data['confidence']:.2f})."
    )


TOOLS = [search_similar_listings, analyse_room_photo]
