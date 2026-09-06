# Image Analyser - dataset preparation

This stage only prepares a labeled, split dataset. No FastAPI service and
no model training yet - that comes after this data is reviewed.

## Pipeline

```bash
cd services/image-analyser
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

python preprocess.py     # select, split, copy raw images -> data/processed/
python label_clip.py     # CLIP zero-shot condition scoring -> labels.csv
```

## 1. Source data

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

### Folder -> category mapping

| Source folder | `room_type` |
|---|---|
| Bathroom | `bathroom` |
| Bedroom | `bedroom` |
| Kitchen | `kitchen` |
| Livingroom | `living_room` |
| Dinning | `other` |
| Exterior | `exterior` |

## 2. Split

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

## 3. CLIP condition scoring

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

### A finding worth flagging before you review

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
Worth specifically checking during your review - the spot-check sample
below includes several score-2 and score-3 images so you have some
lower-scored examples to sanity-check, not just 4s and 5s.

## 4. Spot-check sample for manual review

`data/processed/spotcheck_sample.csv` - 25 images, randomly sampled
(seeded, distinct seed from the split/selection step) with an even
~4-5 per category spread, columns `filename, room_type, condition_score`
only (no confidence - this file is for a human eyeballing the images
against their assigned score). Cross-reference `split` and the file's
full path via `manifest.csv` or `labels.csv` if needed; the image itself
is at `data/processed/{split}/{room_type}/{filename}`.
