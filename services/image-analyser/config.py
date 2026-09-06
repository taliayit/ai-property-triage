from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
RAW_DIR = SERVICE_DIR / "data" / "raw"
PROCESSED_DIR = SERVICE_DIR / "data" / "processed"

HOUSE_ROOM_DATASET_DIR = RAW_DIR / "House_Room_Dataset"

# Source folder (under House_Room_Dataset/) -> target room_type category
CATEGORY_FOLDER_MAP = {
    "Bathroom": "bathroom",
    "Bedroom": "bedroom",
    "Exterior": "exterior",
    "Kitchen": "kitchen",
    "Livingroom": "living_room",
    "Dinning": "other",
}

IMAGES_PER_ROOM_CATEGORY = 40
RANDOM_SEED = 42

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
SPOTCHECK_CSV = PROCESSED_DIR / "spotcheck_sample.csv"

SPOTCHECK_SAMPLE_SIZE = 25

# --- CLIP condition-scoring config ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Descriptive noun phrase used in condition prompts per category - grounds
# CLIP in what the image actually shows.
CATEGORY_PROMPT_NOUN = {
    "bathroom": "bathroom",
    "bedroom": "bedroom",
    "kitchen": "kitchen",
    "living_room": "living room",
    "other": "dining room",
    "exterior": "house exterior",
}

# condition_score -> descriptive phrase. Each image is compared against
# its own category's 5 prompts (built as "a photo of a {noun} in
# {phrase}") and assigned the score of the best-matching prompt.
CONDITION_PROMPTS = {
    1: "very poor condition, damaged and deteriorated, in need of major repairs",
    2: "poor condition, worn and outdated, needs significant renovation",
    3: "fair condition, average wear, functional but dated",
    4: "good condition, clean and well maintained",
    5: "excellent condition, pristine, modern, and beautifully renovated",
}
