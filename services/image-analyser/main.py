"""Image Analyser service - room_type + condition_score from a photo URL.

Confidence design: `confidence` is the room_type head's own softmax
probability, not a combination of both heads. Checked empirically against
the test set (see README): above ~0.4, every room_type prediction was
correct (13/13); below it, only 6/20 were. condition_score is only
meaningful once the room has actually been identified correctly, so when
room_type is uncertain, condition_score is hidden too rather than being
reported as if it were trustworthy on its own.
"""

import logging
from contextlib import asynccontextmanager
from io import BytesIO
from urllib.parse import urlparse

import requests
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from config import (
    ALLOWED_URL_SCHEMES,
    CHECKPOINT_PATH,
    CONDITION_SCORES,
    CONFIDENCE_THRESHOLD,
    IMAGE_DOWNLOAD_MAX_BYTES,
    IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
    ROOM_TYPES,
)
from dataset import EVAL_TRANSFORM
from model import load_model

logger = logging.getLogger(__name__)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if CHECKPOINT_PATH.exists():
        model = load_model(CHECKPOINT_PATH)
    else:
        logger.warning(
            "Model checkpoint not found at '%s'. Run train.py before calling /analyse.",
            CHECKPOINT_PATH,
        )
    yield


app = FastAPI(title="Image Analyser", lifespan=lifespan)


class AnalyseRequest(BaseModel):
    image_url: str


def download_image(url: str) -> Image.Image:
    scheme = urlparse(url).scheme.lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise HTTPException(status_code=400, detail=f"Unsupported URL scheme '{scheme}'")

    try:
        response = requests.get(url, timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Could not download image: {exc}")

    content = BytesIO()
    for chunk in response.iter_content(chunk_size=65536):
        content.write(chunk)
        if content.tell() > IMAGE_DOWNLOAD_MAX_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Image exceeds the {IMAGE_DOWNLOAD_MAX_BYTES // (1024 * 1024)} MB limit",
            )

    try:
        return Image.open(content).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="URL did not return a valid image")


@app.post("/analyse")
def analyse(request: AnalyseRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found at '{CHECKPOINT_PATH}'. Run train.py first.",
        )

    image = download_image(request.image_url)
    tensor = EVAL_TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        room_logits, condition_logits = model(tensor)
        room_probs = room_logits.softmax(dim=1)[0]
        condition_probs = condition_logits.softmax(dim=1)[0]

    room_confidence, room_idx = room_probs.max(dim=0)
    condition_idx = int(condition_probs.argmax(dim=0))
    confidence = round(float(room_confidence), 4)

    if confidence < CONFIDENCE_THRESHOLD:
        return {"room_type": "uncertain", "condition_score": None, "confidence": confidence}

    return {
        "room_type": ROOM_TYPES[int(room_idx)],
        "condition_score": CONDITION_SCORES[condition_idx],
        "confidence": confidence,
    }
