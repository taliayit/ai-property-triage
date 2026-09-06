from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
RAW_DIR = SERVICE_DIR / "data" / "raw"
PROCESSED_DIR = SERVICE_DIR / "data" / "processed"

HOUSE_ROOM_DATASET_DIR = RAW_DIR / "House_Room_Dataset"

# Source folder -> target room_type category
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

# CLIP condition-scoring config
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Noun used in CLIP prompts per category
CATEGORY_PROMPT_NOUN = {
    "bathroom": "bathroom",
    "bedroom": "bedroom",
    "kitchen": "kitchen",
    "living_room": "living room",
    "other": "dining room",
    "exterior": "house exterior",
}

# condition_score -> descriptive phrase
CONDITION_PROMPTS = {
    1: "very poor condition, damaged and deteriorated, in need of major repairs",
    2: "poor condition, worn and outdated, needs significant renovation",
    3: "fair condition, average wear, functional but dated",
    4: "good condition, clean and well maintained",
    5: "excellent condition, pristine, modern, and beautifully renovated",
}

# Room/condition classifier (train.py + main.py)
MODELS_DIR = SERVICE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "room_condition_model.pt"

# Sorted for a deterministic class <-> index mapping, saved into the
# checkpoint so main.py never has to guess it independently of train.py.
ROOM_TYPES = sorted(set(CATEGORY_FOLDER_MAP.values()))
CONDITION_SCORES = sorted(CONDITION_PROMPTS)  # [1, 2, 3, 4, 5]

IMAGE_SIZE = 224
# Standard ImageNet normalization - required since the backbone is
# pretrained on ImageNet and expects inputs on this distribution.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_EPOCHS = 40
TRAIN_BATCH_SIZE = 16
TRAIN_LEARNING_RATE = 1e-3

# Below this, /analyse returns "uncertain" instead of the predicted
# room_type/condition_score - see README for how this was chosen from
# actual test-set confidence values.
CONFIDENCE_THRESHOLD = 0.4

# /analyse endpoint 
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 10
IMAGE_DOWNLOAD_MAX_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_URL_SCHEMES = {"http", "https"}
