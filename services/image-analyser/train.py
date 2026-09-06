"""Trains the two-head room_type/condition_score classifier and reports
held-out test-set accuracy against a majority-class baseline.

Run after preprocess.py and label_clip.py: `python train.py`
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CHECKPOINT_PATH,
    CONDITION_SCORES,
    MODELS_DIR,
    ROOM_TYPES,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LEARNING_RATE,
)
from dataset import RoomConditionDataset
from model import RoomConditionNet


def class_weights(dataset: RoomConditionDataset, kind: str, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for idx in dataset.label_indices(kind):
        counts[idx] += 1
    weights = 1.0 / counts.clamp(min=1)
    return weights / weights.sum() * num_classes


def majority_baseline_accuracy(dataset: RoomConditionDataset, kind: str, num_classes: int) -> float:
    counts = [0] * num_classes
    indices = dataset.label_indices(kind)
    for idx in indices:
        counts[idx] += 1
    return max(counts) / len(indices)


def run_epoch(model, loader, room_loss_fn, condition_loss_fn, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total = 0
    room_correct = 0
    condition_correct = 0
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for images, room_labels, condition_labels in loader:
            room_logits, condition_logits = model(images)
            room_loss = room_loss_fn(room_logits, room_labels)
            condition_loss = condition_loss_fn(condition_logits, condition_labels)
            loss = room_loss + condition_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            room_correct += (room_logits.argmax(dim=1) == room_labels).sum().item()
            condition_correct += (condition_logits.argmax(dim=1) == condition_labels).sum().item()

    return {
        "loss": total_loss / total,
        "room_acc": room_correct / total,
        "condition_acc": condition_correct / total,
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_set = RoomConditionDataset("train", augment=True)
    val_set = RoomConditionDataset("val", augment=False)
    test_set = RoomConditionDataset("test", augment=False)

    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    # Inverse-frequency weighting matters most for condition_score: the
    # labeled training set has zero score-1 examples and is ~64% score-5
    # (see README) - unweighted CrossEntropyLoss would just learn to
    # always predict the majority class.
    room_loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_set, "room", len(ROOM_TYPES)))
    condition_loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(train_set, "condition", len(CONDITION_SCORES))
    )

    model = RoomConditionNet(freeze_backbone=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=TRAIN_LEARNING_RATE)

    best_val_score = -1.0
    best_state = None

    for epoch in tqdm(range(1, TRAIN_EPOCHS + 1), desc="Training"):
        train_metrics = run_epoch(model, train_loader, room_loss_fn, condition_loss_fn, optimizer)
        val_metrics = run_epoch(model, val_loader, room_loss_fn, condition_loss_fn)
        val_score = val_metrics["room_acc"] + val_metrics["condition_acc"]
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        tqdm.write(
            f"epoch {epoch:2d}  train loss {train_metrics['loss']:.3f} "
            f"room_acc {train_metrics['room_acc']:.3f} cond_acc {train_metrics['condition_acc']:.3f}  |  "
            f"val loss {val_metrics['loss']:.3f} room_acc {val_metrics['room_acc']:.3f} "
            f"cond_acc {val_metrics['condition_acc']:.3f}"
        )

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nSaved best checkpoint (highest val room_acc + condition_acc) to {CHECKPOINT_PATH}")

    test_metrics = run_epoch(model, test_loader, room_loss_fn, condition_loss_fn)
    room_baseline = majority_baseline_accuracy(test_set, "room", len(ROOM_TYPES))
    condition_baseline = majority_baseline_accuracy(test_set, "condition", len(CONDITION_SCORES))

    print("\n=== Test set results ===")
    print(
        f"room_type accuracy:       {test_metrics['room_acc']:.3f}  "
        f"(majority-class baseline: {room_baseline:.3f})"
    )
    print(
        f"condition_score accuracy: {test_metrics['condition_acc']:.3f}  "
        f"(majority-class baseline: {condition_baseline:.3f})"
    )


if __name__ == "__main__":
    main()
