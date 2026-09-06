"""Assigns a condition_score (1-5) to every image in the manifest, via
CLIP zero-shot similarity to category-specific condition prompts, and
writes a random spot-check sample for manual review.

Run after preprocess.py: `python label_clip.py`
"""

import csv
import random

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from config import (
    CATEGORY_PROMPT_NOUN,
    CLIP_MODEL_NAME,
    CONDITION_PROMPTS,
    LABELS_CSV,
    MANIFEST_CSV,
    PROCESSED_DIR,
    RANDOM_SEED,
    SPOTCHECK_CSV,
    SPOTCHECK_SAMPLE_SIZE,
)

CONDITION_SCORES = sorted(CONDITION_PROMPTS)  # [1, 2, 3, 4, 5]


def build_category_prompts(category: str) -> list[str]:
    noun = CATEGORY_PROMPT_NOUN[category]
    return [f"a photo of a {noun} in {CONDITION_PROMPTS[score]}" for score in CONDITION_SCORES]


def load_manifest() -> list[dict]:
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def text_embeds(model, inputs) -> torch.Tensor:
    """model.get_text_features() is broken in the installed transformers
    version (returns the raw encoder output instead of the projected,
    normalized embedding) - this replicates it manually. Verified to
    match the model's own text_embeds output from a full forward pass."""
    pooled = model.text_model(**inputs).pooler_output
    projected = model.text_projection(pooled)
    return projected / projected.norm(dim=-1, keepdim=True)


def image_embeds(model, inputs) -> torch.Tensor:
    pooled = model.vision_model(**inputs).pooler_output
    projected = model.visual_projection(pooled)
    return projected / projected.norm(dim=-1, keepdim=True)


def score_images(model, processor, rows: list[dict]) -> list[dict]:
    logit_scale = model.logit_scale.exp().item()

    category_text_embeds: dict[str, torch.Tensor] = {}
    for category in sorted({row["room_type"] for row in rows}):
        prompts = build_category_prompts(category)
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            category_text_embeds[category] = text_embeds(model, inputs)

    results = []
    for row in tqdm(rows, desc="Scoring condition"):
        image_path = PROCESSED_DIR / row["split"] / row["room_type"] / row["filename"]
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            img_embed = image_embeds(model, inputs)

        similarities = (img_embed @ category_text_embeds[row["room_type"]].T).squeeze(0)
        probs = (similarities * logit_scale).softmax(dim=-1)
        best_idx = int(similarities.argmax())

        results.append(
            {
                "filename": row["filename"],
                "room_type": row["room_type"],
                "condition_score": CONDITION_SCORES[best_idx],
                "confidence": round(float(probs[best_idx]), 4),
                "split": row["split"],
            }
        )
    return results


def write_spotcheck_sample(results: list[dict]) -> list[dict]:
    rng = random.Random(RANDOM_SEED + 1)  # distinct from preprocess.py's selection/split seed
    by_category: dict[str, list[dict]] = {}
    for row in results:
        by_category.setdefault(row["room_type"], []).append(row)

    categories = sorted(by_category)
    base = SPOTCHECK_SAMPLE_SIZE // len(categories)
    remainder = SPOTCHECK_SAMPLE_SIZE - base * len(categories)

    sample: list[dict] = []
    for i, category in enumerate(categories):
        quota = min(base + (1 if i < remainder else 0), len(by_category[category]))
        sample.extend(rng.sample(by_category[category], quota))
    rng.shuffle(sample)

    with open(SPOTCHECK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "room_type", "condition_score"])
        writer.writeheader()
        writer.writerows(
            {k: row[k] for k in ("filename", "room_type", "condition_score")} for row in sample
        )
    return sample


def main() -> None:
    print(f"Loading CLIP model '{CLIP_MODEL_NAME}'...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    rows = load_manifest()
    results = score_images(model, processor, rows)

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "room_type", "condition_score", "confidence", "split"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} labels to {LABELS_CSV}")

    sample = write_spotcheck_sample(results)
    print(f"Wrote {len(sample)}-image spot-check sample to {SPOTCHECK_CSV}")


if __name__ == "__main__":
    main()
