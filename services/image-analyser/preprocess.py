"""Selects, splits, and copies raw House_Room_Dataset images into
data/processed/{train,val,test}/{category}/, and writes a manifest CSV
that label_clip.py reads to assign condition scores.

Run once: `python preprocess.py`
Safe to re-run - it clears and rebuilds data/processed/ from scratch.
"""

import csv
import random
import shutil
from pathlib import Path

from config import (
    CATEGORY_FOLDER_MAP,
    HOUSE_ROOM_DATASET_DIR,
    IMAGES_PER_ROOM_CATEGORY,
    MANIFEST_CSV,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_SEED,
    SPLIT_RATIOS,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def select_images(images: list[Path], target_count: int, rng: random.Random) -> list[Path]:
    """Caps a category at target_count images, sampling deterministically
    (seeded) if more are available than needed. Neither branch fires for
    a category that already has exactly target_count images."""
    if len(images) < target_count:
        print(
            f"  Note: only {len(images)} available, fewer than the "
            f"requested {target_count} - using all of them."
        )
        return list(images)
    if len(images) > target_count:
        print(
            f"  Note: {len(images)} available, more than the requested "
            f"{target_count} - randomly sampling {target_count}."
        )
        return rng.sample(images, target_count)
    return list(images)


def split_sizes(n: int) -> dict[str, int]:
    train_n = round(n * SPLIT_RATIOS["train"])
    val_n = round(n * SPLIT_RATIOS["val"])
    test_n = n - train_n - val_n  # remainder, guarantees the three sum to n
    return {"train": train_n, "val": val_n, "test": test_n}


def split_images(images: list[Path], rng: random.Random) -> dict[str, list[Path]]:
    shuffled = list(images)
    rng.shuffle(shuffled)
    sizes = split_sizes(len(shuffled))
    result = {}
    i = 0
    for split_name in ("train", "val", "test"):
        n = sizes[split_name]
        result[split_name] = shuffled[i : i + n]
        i += n
    return result


def prepare_category(
    category: str,
    source_dir: Path,
    target_count: int,
    rng: random.Random,
    manifest_rows: list[dict],
) -> None:
    print(f"\n{category} <- {source_dir}")
    images = list_images(source_dir)
    selected = select_images(images, target_count, rng)
    splits = split_images(selected, rng)

    for split_name, files in splits.items():
        dest_dir = PROCESSED_DIR / split_name / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            shutil.copy2(src, dest_dir / src.name)
            manifest_rows.append(
                {
                    "filename": src.name,
                    "room_type": category,
                    "split": split_name,
                    "source_path": str(src.relative_to(RAW_DIR)),
                }
            )
        print(f"  {split_name}: {len(files)}")


def main() -> None:
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True)

    rng = random.Random(RANDOM_SEED)
    manifest_rows: list[dict] = []

    for source_folder, category in CATEGORY_FOLDER_MAP.items():
        source_dir = HOUSE_ROOM_DATASET_DIR / source_folder
        prepare_category(category, source_dir, IMAGES_PER_ROOM_CATEGORY, rng, manifest_rows)

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "room_type", "split", "source_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} images to {PROCESSED_DIR}")
    print(f"Manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
