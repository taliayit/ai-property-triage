# Image Analyser

Classifies a property photo by `room_type` (6 categories) and
`condition_score` (1-5), served as `POST /analyse`.

```bash
cd services/image-analyser
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

python preprocess.py     # select, split, copy raw images -> data/processed/
python label_clip.py     # CLIP zero-shot condition scoring -> labels.csv
python train.py          # trains models/room_condition_model.pt, prints test accuracy

uvicorn main:app --reload --port 8003
```

## Model

Two linear heads on top of a **frozen**, ImageNet-pretrained
**EfficientNet-B0** backbone (`torchvision`) - one 6-way head for
`room_type`, one 5-way head for `condition_score` (scores 1-5 map to
class indices 0-4). See `model.py`.

EfficientNet-B0 (5.3M params) was picked over ResNet-50 (25M params)
mainly for CPU training/inference speed: the backbone is entirely frozen
(`requires_grad=False`), so a bigger backbone wouldn't extract more
useful features from only 154 training images anyway - it would just cost
more compute per forward pass. Only the two linear heads (~14K params
combined) are ever trained. This is the standard "feature extraction"
transfer-learning approach, chosen over fine-tuning the backbone because
fine-tuning tens of millions of parameters on 154 images would overfit
almost immediately.

The saved checkpoint (`models/room_condition_model.pt`, ~16MB) contains
the full model state, including the unchanged backbone weights - kept
simple/self-contained rather than splitting head-only weights out for a
small disk saving.

## Training data and augmentation

Trained on the dataset prepared by `preprocess.py` + `label_clip.py`
(154 train / 33 val / 33 test images, 6 room types). See "Dataset
preparation" below for exactly how that data was built and labeled.

Train-time augmentation (`dataset.py`, `TRAIN_TRANSFORM`) - deliberately
mild, since aggressive augmentation on only 154 images risks teaching the
model to recognize crops/textures instead of plausible room photos:

- `RandomResizedCrop(224, scale=(0.8, 1.0))` - mild crop/zoom jitter
- `RandomHorizontalFlip(p=0.5)` - room photos are meaningfully mirrored
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` - real
  listing photos vary a lot in lighting/white balance
- `RandomRotation(10°)` - handles slightly uneven handheld shots

Validation and test images use the backbone's own unmodified eval
transform (`EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()` - resize,
center-crop, ImageNet normalization) - no augmentation, so evaluation
reflects the images as-is.

`train.py` also applies inverse-frequency class weighting to both losses.
This barely matters for `room_type` (fairly balanced, 28 each per
category except 14 for exterior) but matters a great deal for
`condition_score`: the labeled training set has **zero** score-1 examples
and is ~64% score-5 (see "Dataset preparation" below) - unweighted
`CrossEntropyLoss` would just learn to always predict 5.

## Test-set results

40 epochs, Adam (lr `1e-3`, only the two heads' params), checkpoint saved
at the epoch with the best combined validation accuracy (epoch 11 of 40 -
validation accuracy peaked early and mostly declined afterward, i.e. the
heads start overfitting the 154 training images well before epoch 40):

| Task | Test accuracy | Majority-class baseline |
|---|---|---|
| `room_type` (6-way) | **54.5%** (18/33) | 18.2% (6/33) |
| `condition_score` (5-way) | **78.8%** (26/33) | 66.7% (22/33) |

Read these with the sample size in mind - 33 test images means each
image is ~3 percentage points, so treat these as rough estimates, not
precise metrics.

`room_type` clearly learned something real - 3x the naive baseline.
`condition_score` beats its baseline by a much smaller margin (+12
points), which is the honest, expected consequence of the extreme class
imbalance: with 22/33 test images actually scoring 5, "always guess 5"
is already a strong baseline, and there's very little signal in the
training data for the model to learn what a 2 or 3 actually looks like
(the training set has no 1s and only 7 2s).

## Confidence and the "uncertain" fallback

`confidence` in the response is the **room_type head's own softmax
probability** for its predicted class - not a combination of both heads'
confidences. This was checked empirically against the test set rather
than assumed:

| room_type confidence | accuracy in that range |
|---|---|
| ≥ 0.407 | 13/13 (100%) |
| < 0.407 | 6/20 (30%) |

That's a clean separation, so `CONFIDENCE_THRESHOLD = 0.4` (`config.py`)
sits right at it. Below the threshold, the response replaces `room_type`
with `"uncertain"` and `condition_score` with `null` - `condition_score`
is hidden too, not just `room_type`, because a condition assessment is
only meaningful once the room has actually been identified correctly; a
kitchen-condition judgment on an image that's actually a bathroom isn't
useful just because the condition head itself felt confident.

At this threshold, roughly 60% of this test set falls below it and would
be reported `"uncertain"` rather than guessed - which reflects a real
constraint of training two classifier heads on 154 images, not a bug.

## `POST /analyse`

```json
{"image_url": "https://example.com/listing-photo.jpg"}
```

```json
{"room_type": "kitchen", "condition_score": 5, "confidence": 0.5051}
```

```json
{"room_type": "uncertain", "condition_score": null, "confidence": 0.2252}
```

The URL is fetched server-side (`requests`, `ALLOWED_URL_SCHEMES` in
`config.py` restricts it to `http`/`https`, a 10s timeout, and a 15MB
download cap) - `400` for an invalid scheme, an unreachable URL, or a
response that isn't a decodable image. `503` if `models/room_condition_model.pt`
doesn't exist yet (run `train.py` first) - the service still starts
rather than crashing, matching the same pattern used in the other
services here for a missing model file.

Note this only does basic SSRF hardening (scheme allowlist, timeout, size
cap) - it doesn't block requests to private/internal IP ranges. Fine for
a local demo; a production deployment fetching arbitrary user-supplied
URLs should add that.

## Docker

Single-stage - unlike rag-service, nothing here needs compiling
(`torch`/`torchvision` ship prebuilt wheels). The trained checkpoint is
copied in, and the ImageNet-pretrained backbone weights are downloaded
once at build time (mirroring rag-service's build-time seeding) so the
container needs no internet load the model at runtime:

```bash
docker build -t image-analyser .
docker run -d -p 8003:8003 image-analyser
```

---

## Dataset preparation

### 1. Source data

```
data/raw/House_Room_Dataset/{Bathroom,Bedroom,Dinning,Kitchen,Livingroom,Exterior}/
```

All six categories, exterior included, live under this one parent folder.
Bathroom/Bedroom/Dinning/Kitchen/Livingroom each have exactly 40 images;
Exterior has 20. **No subsampling was actually needed** - `preprocess.py`
caps every category at `IMAGES_PER_ROOM_CATEGORY` (40) via the same
per-category logic regardless of category, sampling down with a fixed
seed (`RANDOM_SEED = 42` in `config.py`) only if a folder ever has more
than that; none did here, and Exterior (20 available) simply uses all of
what exists since that's fewer than 40.

#### Folder -> category mapping

| Source folder | `room_type` |
|---|---|
| Bathroom | `bathroom` |
| Bedroom | `bedroom` |
| Kitchen | `kitchen` |
| Livingroom | `living_room` |
| Dinning | `other` |
| Exterior | `exterior` |

### 2. Split

70/15/15 per category (stratified - each category is shuffled and split
independently, not the pooled set), seeded for reproducibility:

| Category | train | val | test | total |
|---|---|---|---|---|
| bathroom / bedroom / kitchen / living_room / other | 28 | 6 | 6 | 40 |
| exterior | 14 | 3 | 3 | 20 |
| **all** | **154** | **33** | **33** | **220** |

Images are copied (not moved) into
`data/processed/{train,val,test}/{room_type}/`, so `data/raw/` is
untouched and the result is directly usable with
`torchvision.datasets.ImageFolder`-style loaders. `data/processed/manifest.csv`
records `filename, room_type, split, source_path` for every image.

### 3. CLIP condition scoring

`label_clip.py` uses `openai/clip-vit-base-patch32` (via `transformers`)
zero-shot: each image's embedding is compared against 5 condition-level
text prompts *for its own category only* (not all 30 prompts), and gets
the `condition_score` (1-5) of the best match.

Prompt template: `"a photo of a {noun} in {condition phrase}"`, where
`{noun}` is a category-specific, descriptive noun phrase (not necessarily
the output `room_type` label - e.g. `other`'s images are dining rooms, so
its noun is "dining room" for a more grounded CLIP match) and
`{condition phrase}` is one of:

| score | condition phrase |
|---|---|
| 1 | very poor condition, damaged and deteriorated, in need of major repairs |
| 2 | poor condition, worn and outdated, needs significant renovation |
| 3 | fair condition, average wear, functional but dated |
| 4 | good condition, clean and well maintained |
| 5 | excellent condition, pristine, modern, and beautifully renovated |

Category nouns: bathroom, bedroom, kitchen, "living room", "dining room"
(for `other`), "house exterior" (see `CATEGORY_PROMPT_NOUN` in
`config.py`).

Output: `data/processed/labels.csv` -
`filename, room_type, condition_score, confidence, split`. `confidence`
is the softmax probability (over that category's 5 prompts, using CLIP's
own learned logit scale) assigned to the winning score - not requested
by the spec, but cheap to compute and useful for prioritizing which
labels to spot-check first.

#### A finding worth flagging

The label distribution is heavily skewed toward "excellent" and has
**zero** images scored 1:

```
score 2:   8 images
score 3:  24 images
score 4:  51 images
score 5: 137 images   <- 62% of the dataset
```

This could mean the dataset genuinely skews toward well-kept properties
(plausible - these look like real-estate-style listing photos, which
tend to be professionally shot and well-lit even for average
properties), or it could mean CLIP's zero-shot scoring is systematically
biased toward "pristine/modern" phrasing regardless of actual condition.
This directly explains the `condition_score` model results above - it
never saw a 1 during training and barely saw any 2s or 3s.

### 4. Spot-check sample for manual review

`data/processed/spotcheck_sample.csv` - 25 images, randomly sampled
(seeded, distinct seed from the split/selection step) with an even
~4-5 per category spread, columns `filename, room_type, condition_score`
only (no confidence - this file is for a human eyeballing the images
against their assigned score). Cross-reference `split` and the file's
full path via `manifest.csv` or `labels.csv` if needed; the image itself
is at `data/processed/{split}/{room_type}/{filename}`.
