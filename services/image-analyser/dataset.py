"""Train-split augmentation, kept mild on purpose - with only 154
training images, aggressive crops/rotations risk teaching the model to
recognize texture/crop artifacts instead of actual room features.
Val/test use the plain eval transform, no augmentation.
"""

import csv

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LABELS_CSV, PROCESSED_DIR, CONDITION_SCORES, ROOM_TYPES
from model import build_transform

ROOM_TO_IDX = {name: i for i, name in enumerate(ROOM_TYPES)}
CONDITION_TO_IDX = {score: i for i, score in enumerate(CONDITION_SCORES)}

EVAL_TRANSFORM = build_transform()

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class RoomConditionDataset(Dataset):
    def __init__(self, split: str, augment: bool):
        self.transform = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM
        self.samples: list[tuple] = []
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                path = PROCESSED_DIR / row["split"] / row["room_type"] / row["filename"]
                self.samples.append((path, row["room_type"], int(row["condition_score"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, room_type, condition_score = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, ROOM_TO_IDX[room_type], CONDITION_TO_IDX[condition_score]

    def label_indices(self, kind: str) -> list[int]:
        if kind == "room":
            return [ROOM_TO_IDX[s[1]] for s in self.samples]
        return [CONDITION_TO_IDX[s[2]] for s in self.samples]
